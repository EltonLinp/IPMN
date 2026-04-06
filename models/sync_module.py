from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        vit_unfreeze_layers: int = 0,
        temporal_layers: int = 1,
    ) -> None:
        super().__init__()
        vit_path = Path(vit_path)
        if not vit_path.exists():
            raise FileNotFoundError(f"ViT model path not found: {vit_path}")
        self.vit = ViTModel.from_pretrained(vit_path, local_files_only=True)
        for param in self.vit.parameters():
            param.requires_grad = False
        vit_unfreeze_layers = max(int(vit_unfreeze_layers), 0)
        if vit_unfreeze_layers > 0 and hasattr(self.vit, "encoder"):
            encoder_layers = getattr(self.vit.encoder, "layer", [])
            layers_to_unfreeze = encoder_layers[-vit_unfreeze_layers:]
            for block in layers_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True
            if hasattr(self.vit, "layernorm"):
                for param in self.vit.layernorm.parameters():
                    param.requires_grad = True

        self.hidden_dim = self.vit.config.hidden_size
        self.joint_dim = self.hidden_dim * 3
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
        temporal_layers = max(int(temporal_layers), 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=temporal_layers)

        self.sync_head = nn.Sequential(
            nn.LayerNorm(self.joint_dim),
            nn.Linear(self.joint_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 2),
        )

    def _normalize_frames(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B, T, 3, H, W] in [-1, 1] -> convert to ViT expected range
        frames = (frames + 1.0) * 0.5  # [0, 1]
        mean = self.image_mean.to(frames.device, frames.dtype)
        std = self.image_std.to(frames.device, frames.dtype)
        frames = (frames - mean) / std
        return frames

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        bsz, frames, _, _, _ = video.shape
        video_norm = self._normalize_frames(video)
        vit_input = video_norm.reshape(bsz * frames, 3, video.shape[-2], video.shape[-1])
        vit_outputs = self.vit(pixel_values=vit_input, output_hidden_states=False)
        frame_emb = vit_outputs.pooler_output.view(bsz, frames, self.hidden_dim)
        return frame_emb

    def forward(
        self,
        video: torch.Tensor | None,
        audio_seq: torch.Tensor,
        *,
        video_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video: [B, T, 3, 224, 224] when video_emb is None
            audio_seq: [B, T, 64]
            video_emb: Optional precomputed ViT embeddings [B, T, hidden_dim]
        Returns:
            joint_state: [B, hidden_dim]
            logits: [B, 2]
        """
        if video_emb is None:
            if video is None:
                raise ValueError("Either video tensor or video_emb must be provided.")
            frame_emb = self.encode_video(video)
        else:
            frame_emb = video_emb
        frame_emb = self.video_proj(frame_emb)
        if frame_emb.size(1) != audio_seq.size(1):
            frame_emb = frame_emb.transpose(1, 2)
            frame_emb = F.interpolate(
                frame_emb,
                size=audio_seq.size(1),
                mode="linear",
                align_corners=False,
            )
            frame_emb = frame_emb.transpose(1, 2)

        audio_emb = self.audio_proj(audio_seq)  # [B, T, hidden_dim]
        audio_norm = self.norm_audio(audio_emb)
        video_norm = self.norm_video(frame_emb)
        pad_mask = torch.zeros(
            video_norm.size(0),
            video_norm.size(1),
            dtype=torch.bool,
            device=video_norm.device,
        )
        attn_out, _ = self.cross_attn(
            audio_norm,
            video_norm,
            video_norm,
            key_padding_mask=pad_mask,
            attn_mask=None,
        )
        fused = audio_emb + attn_out
        encoded = self.temporal_encoder(fused)
        encoded_mean = encoded.mean(dim=1)
        diff_mean = (audio_emb - frame_emb).mean(dim=1)
        prod_mean = (audio_emb * frame_emb).mean(dim=1)
        joint_state = torch.cat([encoded_mean, diff_mean, prod_mean], dim=1)
        logits = self.sync_head(joint_state)
        return joint_state, logits
