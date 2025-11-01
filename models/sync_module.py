from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import ViTModel


class SyncModule(nn.Module):
    """
    Synchronisation branch combining frozen ViT frame features with projected audio embeddings.
    """

    def __init__(
        self,
        vit_path: str | Path,
        audio_dim: int = 64,
        transformer_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        vit_path = Path(vit_path)
        if not vit_path.exists():
            raise FileNotFoundError(f"ViT model path not found: {vit_path}")
        self.vit = ViTModel.from_pretrained(vit_path, local_files_only=True)
        for param in self.vit.parameters():
            param.requires_grad = False

        self.hidden_dim = self.vit.config.hidden_size
        mean = getattr(self.vit.config, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(self.vit.config, "image_std", [0.5, 0.5, 0.5])
        self.register_buffer("image_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        self.audio_proj = nn.Linear(audio_dim, self.hidden_dim)
        self.video_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm_audio = nn.LayerNorm(self.hidden_dim)
        self.norm_video = nn.LayerNorm(self.hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=transformer_heads,
            dropout=dropout,
            batch_first=True,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.sync_head = nn.Linear(self.hidden_dim, 2)

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, T, 3, H, W] in [-1, 1] -> convert to ViT expected range
        frames = (frames + 1.0) * 0.5  # [0, 1]
        mean = self.image_mean.to(frames.device, frames.dtype)
        std = self.image_std.to(frames.device, frames.dtype)
        frames = (frames - mean) / std
        return frames

    def forward(self, video: torch.Tensor, audio_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video: [B, T, 3, 224, 224]
            audio_seq: [B, T, 64]
        Returns:
            joint_state: [B, hidden_dim]
            logits: [B, 2]
        """
        bsz, frames, _, _, _ = video.shape
        video_norm = self._normalize_frames(video)
        vit_input = video_norm.reshape(bsz * frames, 3, video.shape[-2], video.shape[-1])
        vit_outputs = self.vit(pixel_values=vit_input, output_hidden_states=False)
        frame_emb = vit_outputs.pooler_output.view(bsz, frames, self.hidden_dim)
        frame_emb = self.video_proj(frame_emb)

        audio_emb = self.audio_proj(audio_seq)  # [B, T, hidden_dim]
        audio_norm = self.norm_audio(audio_emb)
        video_norm = self.norm_video(frame_emb)
        attn_out, _ = self.cross_attn(audio_norm, video_norm, video_norm)
        fused = audio_emb + attn_out
        encoded = self.temporal_encoder(fused)
        joint_state = encoded.mean(dim=1)
        logits = self.sync_head(joint_state)
        return joint_state, logits
