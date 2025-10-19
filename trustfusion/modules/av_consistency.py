"""
Audio-visual consistency module inspired by SyncNet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from ..utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class AVConsistencyOutput:
    sync_score: torch.Tensor
    attention_map: torch.Tensor
    metadata: Dict[str, Any]


class AVConsistencyModule(nn.Module):
    """
    Lightweight proxy for lip-audio synchrony analysis.
    """

    def __init__(self, feature_dim: int = 128, backbone: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.backbone = backbone if backbone else nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.scorer = nn.Linear(feature_dim // 2, 1)

    def forward(self, lip_feats: torch.Tensor, audio_feats: torch.Tensor) -> AVConsistencyOutput:
        if lip_feats.shape != audio_feats.shape:
            raise ValueError("Lip and audio features must share shape for temporal comparison")
        concat = torch.cat([lip_feats, audio_feats], dim=-1)
        joint = self.backbone(concat)
        sync_score = torch.sigmoid(self.scorer(joint)).squeeze(-1)
        attention_map = joint  # placeholder for actual attention weights
        return AVConsistencyOutput(
            sync_score=sync_score,
            attention_map=attention_map,
            metadata={"latent_dim": joint.shape[-1]},
        )

    def load_pretrained(self, checkpoint_path: str) -> None:
        LOGGER.info("Loading AV consistency checkpoint from %s", checkpoint_path)
