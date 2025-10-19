"""
Multimodal fusion logic for TrustFusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from ..modules import (
    AudioDeepfakeDetector,
    AVConsistencyModule,
    BioSyncAnalyzer,
    CrossModalAttention,
    DiffusionResidualExplainer,
    VideoDeepfakeDetector,
)
from ..utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class FusionOutput:
    trust_score: torch.Tensor
    modality_scores: Dict[str, torch.Tensor]
    biosync_score: torch.Tensor
    sync_score: torch.Tensor
    diffusion_confidence: torch.Tensor


class FusionEngine:
    """
    Collects modality outputs and derives a unified trust score.
    """

    def __init__(
        self,
        video_detector: VideoDeepfakeDetector,
        audio_detector: AudioDeepfakeDetector,
        av_module: AVConsistencyModule,
        attention: CrossModalAttention,
        biosync: BioSyncAnalyzer,
        diffusion: DiffusionResidualExplainer,
        trust_thresholds: Tuple[float, float],
    ) -> None:
        self.video_detector = video_detector
        self.audio_detector = audio_detector
        self.av_module = av_module
        self.attention = attention
        self.biosync = biosync
        self.diffusion = diffusion
        self.low_threshold, self.high_threshold = trust_thresholds

    def forward(
        self,
        video_tensor: torch.Tensor,
        audio_tensor: torch.Tensor,
        lip_feats: torch.Tensor,
        audio_feats: torch.Tensor,
        rppg_input: torch.Tensor,
        audio_energy: torch.Tensor,
        sample_rate: int,
    ) -> FusionOutput:
        video_out = self.video_detector(video_tensor)
        audio_out = self.audio_detector(audio_tensor)
        sync_out = self.av_module(lip_feats, audio_feats)
        fused_feats = self.attention(video_out.rppg_trace.unsqueeze(1), audio_out.spectral_flow.unsqueeze(1))
        biosync_out = self.biosync.compute(rppg_input, audio_energy, sample_rate)
        diffusion_out = self.diffusion.explain(video_tensor.permute(0, 2, 1, 3, 4))

        video_score = video_out.scores
        audio_score = audio_out.scores
        sync_score = sync_out.sync_score.mean(dim=-1)
        biosync_score = biosync_out.biosync_score
        diffusion_score = diffusion_out.confidence_score.mean(dim=-1)

        trust_components = torch.stack(
            [
                video_score,
                audio_score,
                sync_score,
                biosync_score,
                diffusion_score,
            ],
            dim=-1,
        )
        trust_score = trust_components.mean(dim=-1)

        LOGGER.debug("Computed trust components: %s", trust_components)
        return FusionOutput(
            trust_score=trust_score,
            modality_scores={
                "video": video_score,
                "audio": audio_score,
                "sync": sync_score,
                "biosync": biosync_score,
                "diffusion": diffusion_score,
            },
            biosync_score=biosync_score,
            sync_score=sync_score,
            diffusion_confidence=diffusion_score,
        )
