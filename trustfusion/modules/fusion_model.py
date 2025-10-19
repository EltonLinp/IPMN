"""
Simple linear fusion over video, audio, and synchrony detectors.

This module stitches together the modality-specific detectors and produces
a consolidated trust score. It serves as a baseline before introducing more
expressive fusion operators (e.g., attention or gating).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .audio_detector import AudioDeepfakeDetector, AudioDetectorOutput
from .sync_transformer import SyncTransformerOutput, VisionAudioSyncTransformer
from .video_detector import VideoDeepfakeDetector, VideoDetectorOutput
from ..utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class FusionOutput:
    logits: torch.Tensor
    scores: torch.Tensor
    modality_logits: Dict[str, torch.Tensor]
    modality_scores: Dict[str, torch.Tensor]
    joint_embedding: torch.Tensor
    metadata: Dict[str, Any]


class LinearFusionModel(nn.Module):
    """
    Baseline multimodal fusion using a single linear layer over modality logits.

    Parameters
    ----------
    video_detector : Optional[VideoDeepfakeDetector]
        Video detector instance. If ``None`` a default placeholder is created.
    audio_detector : Optional[AudioDeepfakeDetector]
        Audio detector instance. If ``None`` uses the default recurrent model.
    sync_module : Optional[VisionAudioSyncTransformer]
        Synchrony module. Must be provided because it depends on ViT checkpoints.
    """

    def __init__(
        self,
        video_detector: Optional[VideoDeepfakeDetector] = None,
        audio_detector: Optional[AudioDeepfakeDetector] = None,
        sync_module: Optional[VisionAudioSyncTransformer] = None,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if sync_module is None:
            raise ValueError("VisionAudioSyncTransformer must be supplied to LinearFusionModel.")

        self.video_detector = video_detector or VideoDeepfakeDetector()
        self.audio_detector = audio_detector or AudioDeepfakeDetector()
        self.sync_module = sync_module

        auxiliary_dim = self._infer_auxiliary_dim()
        self.modality_proj = nn.Linear(auxiliary_dim, fusion_hidden_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
        )
        self.fusion_head = nn.Linear((fusion_hidden_dim // 2) + 3, 1)

    def _infer_auxiliary_dim(self) -> int:
        video_dim = self.video_detector.rppg_head[-1].out_features  # type: ignore[index]
        audio_dim = self.audio_detector.spectral_head[-1].out_features  # type: ignore[index]
        sync_dim = 0
        if hasattr(self.sync_module, "classifier"):
            sync_dim = self.sync_module.classifier[0].in_features  # type: ignore[index]
        return video_dim + audio_dim + sync_dim

    def forward(
        self,
        video_clip: torch.Tensor,
        audio_segment: torch.Tensor,
        sync_video_frames: torch.Tensor,
        sync_audio_seq: torch.Tensor,
    ) -> FusionOutput:
        """
        Args
        ----
        video_clip : torch.Tensor
            Tensor shaped (B, C, T, H, W) consumed by the video detector.
        audio_segment : torch.Tensor
            Tensor shaped (B, 1, F, T) consumed by the audio detector.
        sync_video_frames : torch.Tensor
            Tensor shaped (B, T_sync, C, H, W) consumed by the sync transformer.
        sync_audio_seq : torch.Tensor
            Tensor shaped (B, T_sync, F_sync) consumed by the sync transformer.
        """

        video_out: VideoDetectorOutput = self.video_detector(video_clip)
        audio_out: AudioDetectorOutput = self.audio_detector(audio_segment)
        sync_out: SyncTransformerOutput = self.sync_module(sync_video_frames, sync_audio_seq)

        logits_stack = torch.stack(
            [video_out.logits, audio_out.logits, sync_out.logits], dim=-1
        )
        auxiliary = torch.cat(
            [
                video_out.rppg_trace,
                audio_out.spectral_flow,
                sync_out.joint_state.unsqueeze(1).expand_as(video_out.rppg_trace),
            ],
            dim=-1,
        )
        pooled_aux = auxiliary.mean(dim=1)
        fused_embedding = self.fusion_mlp(self.modality_proj(pooled_aux))
        fused_logits = self.fusion_head(
            torch.cat([fused_embedding, logits_stack], dim=-1)
        ).squeeze(-1)
        fused_scores = torch.sigmoid(fused_logits)

        return FusionOutput(
            logits=fused_logits,
            scores=fused_scores,
            modality_logits={
                "video": video_out.logits,
                "audio": audio_out.logits,
                "synchrony": sync_out.logits,
            },
            modality_scores={
                "video": video_out.scores,
                "audio": audio_out.scores,
                "synchrony": sync_out.scores,
            },
            joint_embedding=fused_embedding,
            metadata={
                "fusion_weights": self.fusion_head.weight.detach().cpu().clone(),
                "fusion_bias": self.fusion_head.bias.detach().cpu().clone(),
            },
        )
