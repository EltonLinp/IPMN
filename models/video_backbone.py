from __future__ import annotations

import torch
import torch.nn as nn


class VideoClassifier(nn.Module):
    """
    Configurable 3D video backbone with dual heads (classification + rPPG regression).
    """

    def __init__(
        self,
        num_classes: int = 2,
        rppg_dim: int = 1,
        dropout: float = 0.3,
        backbone: str = "light",
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_type = backbone
        if backbone == "light":
            self.feature_extractor = nn.Sequential(
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
        elif backbone == "r3d18":
            try:
                from torchvision.models.video import R3D_18_Weights, r3d_18
            except ImportError as exc:  # pragma: no cover - handled during runtime
                raise ImportError(
                    "torchvision>=0.13.0 is required to use the r3d18 backbone."
                ) from exc
            weights = R3D_18_Weights.DEFAULT if pretrained else None
            model = r3d_18(weights=weights)
            feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            self.feature_extractor = model
            self.pool = None
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'.")

        self.feature_dim = feature_dim
        self.classifier_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.rppg_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, rppg_dim),
        )

    def forward(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            video: Tensor with shape [B, 3, T, H, W] in [-1, 1].
        Returns:
            logits: [B, num_classes]
            rppg: [B, rppg_dim]
            pooled: [B, feature_dim]
        """
        if self.backbone_type == "light":
            feats = self.feature_extractor(video)
            pooled = self.pool(feats).flatten(1)
        else:
            pooled = self.feature_extractor(video)
        logits = self.classifier_head(pooled)
        rppg = self.rppg_head(pooled)
        return logits, rppg, pooled

    def forward_features(self, video: torch.Tensor) -> torch.Tensor:
        if self.backbone_type == "light":
            feats = self.feature_extractor(video)
            return self.pool(feats).flatten(1)
        return self.feature_extractor(video)

    def freeze_backbone(self) -> None:
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
