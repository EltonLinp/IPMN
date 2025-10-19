"""
Audio-visual synchrony module backed by a Vision Transformer visual encoder.

This module ingests per-frame video tensors alongside aligned audio embeddings
and produces a joint representation suitable for downstream fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from ..utils.logger import get_logger

LOGGER = get_logger(__name__)

try:
    from transformers import ViTConfig, ViTModel
except ImportError:  # pragma: no cover - optional dependency
    ViTConfig = None  # type: ignore[assignment]
    ViTModel = None  # type: ignore[assignment]


@dataclass
class SyncTransformerOutput:
    logits: torch.Tensor
    scores: torch.Tensor
    joint_state: torch.Tensor
    metadata: Dict[str, Any]


class VisionAudioSyncTransformer(nn.Module):
    """
    Vision Transformer-based synchrony scorer.

    Parameters
    ----------
    vit_checkpoint : Path
        Path to a directory containing ViT weights (config.json, model binaries, etc.).
    hidden_dim : int
        Projection dimension for the joint audio-visual embedding space.
    num_heads : int
        Number of attention heads used during fusion.
    dropout : float
        Dropout applied to the fusion transformer block.
    """

    def __init__(
        self,
        vit_checkpoint: Path,
        audio_feature_dim: int = 128,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if ViTModel is None or ViTConfig is None:
            raise ImportError(
                "transformers is required for VisionAudioSyncTransformer. "
                "Install it via `pip install transformers`."
            )

        if not vit_checkpoint.exists():
            raise FileNotFoundError(f"ViT checkpoint directory not found at {vit_checkpoint}")

        LOGGER.info("Loading ViT weights from %s", vit_checkpoint)
        self.vit = ViTModel.from_pretrained(
            pretrained_model_name_or_path=str(vit_checkpoint),
            local_files_only=True,
        )
        self.vit.eval()
        for param in self.vit.parameters():
            param.requires_grad = False

        vit_hidden = self.vit.config.hidden_size
        self.visual_proj = nn.Linear(vit_hidden, hidden_dim)
        self.audio_proj = nn.Linear(audio_feature_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.device = device
        if device is not None:
            self.to(device)

    @staticmethod
    def _reshape_video(video_frames: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if video_frames.ndim != 5:
            raise ValueError("Video tensor must have shape (B, T, C, H, W)")
        batch, time, channels, height, width = video_frames.shape
        flattened = video_frames.reshape(batch * time, channels, height, width)
        return flattened, (batch, time)

    def _encode_visual(self, video_frames: torch.Tensor) -> torch.Tensor:
        flattened, (batch, time) = self._reshape_video(video_frames)
        if flattened.size(-1) != self.vit.config.image_size:
            flattened = F.interpolate(
                flattened,
                size=(self.vit.config.image_size, self.vit.config.image_size),
                mode="bilinear",
                align_corners=False,
            )
        outputs = self.vit(pixel_values=flattened, output_hidden_states=False)
        cls_tokens = outputs.last_hidden_state[:, 0]  # (B*T, hidden)
        projected = self.visual_proj(cls_tokens)
        return projected.view(batch, time, -1)

    def _encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        if audio_features.ndim != 3:
            raise ValueError("Audio features must have shape (B, T, F)")
        return self.audio_proj(audio_features)

    def forward(
        self,
        video_frames: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> SyncTransformerOutput:
        if video_frames.shape[0] != audio_features.shape[0] or video_frames.shape[1] != audio_features.shape[1]:
            raise ValueError("Video and audio inputs must share batch and time dimensions.")

        device = self.device or video_frames.device
        video_frames = video_frames.to(device)
        audio_features = audio_features.to(device)

        visual_repr = self._encode_visual(video_frames)
        audio_repr = self._encode_audio(audio_features)

        joint_seq = visual_repr + audio_repr  # simple residual fusion
        fused = self.fusion_encoder(joint_seq)
        pooled = fused.mean(dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        scores = torch.sigmoid(logits)

        return SyncTransformerOutput(
            logits=logits,
            scores=scores,
            joint_state=pooled,
            metadata={
                "video_hidden_dim": visual_repr.shape[-1],
                "audio_hidden_dim": audio_repr.shape[-1],
                "fusion_hidden_dim": pooled.shape[-1],
            },
        )

    def load_pretrained(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Sync transformer checkpoint not found at {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.load_state_dict(state_dict, strict=False)
        LOGGER.info("Loaded sync transformer checkpoint from %s", checkpoint_path)

