"""
Cross-modal attention block coordinating audio and video representations.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


class CrossModalAttention(nn.Module):
    """
    Simplified cross-modal attention layer.
    """

    def __init__(self, d_video: int, d_audio: int, d_out: int) -> None:
        super().__init__()
        self.query_v = nn.Linear(d_video, d_out)
        self.key_a = nn.Linear(d_audio, d_out)
        self.value_a = nn.Linear(d_audio, d_out)
        self.proj = nn.Linear(d_out, d_out)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, video_feats: torch.Tensor, audio_feats: torch.Tensor) -> torch.Tensor:
        if video_feats.shape[:-1] != audio_feats.shape[:-1]:
            raise ValueError("Audio and video tensors must align on batch and temporal dims")
        q = self.query_v(video_feats)
        k = self.key_a(audio_feats)
        v = self.value_a(audio_feats)
        scale = math.sqrt(q.shape[-1])
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = self.softmax(attn)
        out = torch.matmul(attn, v)
        return self.proj(out)
