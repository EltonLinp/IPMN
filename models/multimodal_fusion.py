from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .video_backbone import VideoClassifier
from .aasist_lite import AASISTLite
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
        self.audio_branch = AASISTLite(num_classes=num_classes)
        self.sync_branch = SyncModule(vit_path=vit_path, audio_dim=self.audio_branch.mel_bins)

        self.video_proj = nn.Sequential(
            nn.Linear(self.video_branch.feature_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_branch.feature_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.sync_proj = nn.Sequential(
            nn.Linear(self.sync_branch.joint_dim, fusion_dim),
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
        audio_seq = audio
        if audio_seq.dim() == 4:
            if audio_seq.size(1) == 1:
                audio_seq = audio_seq.squeeze(1).transpose(1, 2)  # [B, T, mel]
            else:
                audio_seq = audio_seq.transpose(1, 3).squeeze(3)
        elif audio_seq.dim() == 3 and audio_seq.size(1) == self.audio_branch.mel_bins:
            audio_seq = audio_seq.transpose(1, 2)

        if audio_seq.size(-1) != self.audio_branch.mel_bins:
            audio_seq = F.interpolate(
                audio_seq.transpose(1, 2).unsqueeze(1),
                size=(self.audio_branch.mel_bins, audio_seq.size(1)),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).transpose(1, 2)

        audio_input = audio_seq.transpose(1, 2).unsqueeze(1)
        a_logits, a_segments, a_feat = self.audio_branch(audio_input)
        joint_state, sync_logits = self.sync_branch(
            video.permute(0, 2, 1, 3, 4),
            audio_seq,
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
            "spectral": a_segments,
            "segment_logits": a_segments,
            "joint_state": pooled,
        }
