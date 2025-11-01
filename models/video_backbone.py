from __future__ import annotations

import torch
import torch.nn as nn


class VideoClassifier(nn.Module):
    """
    Lightweight 3D CNN backbone with dual heads (classification + rPPG regression).
    """

    def __init__(
        self,
        num_classes: int = 2,
        rppg_dim: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        feature_dim = 128
        self.feature_dim = feature_dim
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.rppg_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, rppg_dim),
        )

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video: Tensor with shape [B, 3, T, H, W] in [-1, 1].
        Returns:
            logits: [B, num_classes]
            rppg: [B, rppg_dim]
        """
        feats = self.backbone(video)
        pooled = self.pool(feats).flatten(1)
        logits = self.classifier_head(pooled)
        rppg = self.rppg_head(pooled)
        return logits, rppg, pooled

    def forward_features(self, video: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(video)
        return self.pool(feats).flatten(1)
