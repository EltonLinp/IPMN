from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .video_backbone import VideoClassifier
from .audio_backbone import AudioClassifier
from .sync_module import SyncModule


class MultimodalFusionModel(nn.Module):
    def __init__(
        self,
        vit_path: str | Path,
        num_classes: int = 2,
        fusion_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.video_branch = VideoClassifier(num_classes=num_classes)
        self.audio_branch = AudioClassifier(num_classes=num_classes)
        self.sync_branch = SyncModule(vit_path=vit_path)

        self.video_proj = nn.Sequential(
            nn.Linear(self.video_branch.feature_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_branch.feature_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.sync_proj = nn.Sequential(
            nn.Linear(self.sync_branch.hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=4,
            dim_feedforward=fusion_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        v_logits, v_rppg, v_feat = self.video_branch(video)
        a_logits, a_spec, a_feat = self.audio_branch(audio)
        joint_state, sync_logits = self.sync_branch(
            video.permute(0, 2, 1, 3, 4),
            audio,
        )

        v_proj = self.video_proj(v_feat)
        a_proj = self.audio_proj(a_feat)
        s_proj = self.sync_proj(joint_state)

        tokens = torch.stack([v_proj, a_proj, s_proj], dim=1)
        fused = self.fusion_encoder(tokens)
        pooled = fused.mean(dim=1)
        logits = self.head(pooled)

        return {
            "logits": logits,
            "video_logits": v_logits,
            "audio_logits": a_logits,
            "sync_logits": sync_logits,
            "rppg": v_rppg,
            "spectral": a_spec,
            "joint_state": pooled,
        }
