from __future__ import annotations

import torch
import torch.nn as nn


class AudioClassifier(nn.Module):
    """
    Bidirectional GRU over Mel-spectrogram sequences.

    Input shape: [B, 1, F, T] or [B, T, F]
    Outputs:
        - logits: [B, num_classes]
        - spectral_head: [B, F] auxiliary representation
    """

    def __init__(
        self,
        num_classes: int = 2,
        mel_bins: int = 64,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.mel_bins = mel_bins
        self.gru = nn.GRU(
            input_size=mel_bins,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        feature_dim = hidden_size * 2
        self.feature_dim = feature_dim
        self.norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        self.spectral_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, mel_bins),
        )

    def _prepare_input(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() == 4:
            mel = mel.squeeze(1).permute(0, 2, 1)
        elif mel.dim() == 3 and mel.shape[1] == self.mel_bins:
            mel = mel.permute(0, 2, 1)
        return mel

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self._prepare_input(mel)
        output, _ = self.gru(seq)
        pooled = self.norm(output.mean(dim=1))
        logits = self.classifier(pooled)
        spectral = self.spectral_head(pooled)
        return logits, spectral, pooled

    def forward_features(self, mel: torch.Tensor) -> torch.Tensor:
        seq = self._prepare_input(mel)
        output, _ = self.gru(seq)
        return self.norm(output.mean(dim=1))
