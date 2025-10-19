"""
Audio deepfake detector leveraging recurrent modelling over spectrogram sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from ..utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class AudioDetectorOutput:
    logits: torch.Tensor
    scores: torch.Tensor
    spectral_flow: torch.Tensor
    metadata: Dict[str, Any]


class AudioDeepfakeDetector(nn.Module):
    """
    Placeholder wrapper capturing recurrent modelling of temporal audio features.
    """

    def __init__(
        self,
        input_bins: int = 80,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        rnn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.input_bins = input_bins
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        if rnn is not None:
            self.rnn = rnn
        else:
            self.rnn = nn.GRU(
                input_size=input_bins,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )

        rnn_feature_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_feature_dim, rnn_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_feature_dim // 2, 1),
        )
        self.spectral_head = nn.Sequential(
            nn.Linear(rnn_feature_dim, rnn_feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(rnn_feature_dim // 2, rnn_feature_dim // 4),
        )

    def forward(self, audio_batch: torch.Tensor) -> AudioDetectorOutput:
        """
        Args:
            audio_batch: tensor shaped (B, 1, F, T) containing log-mel spectrograms.
        """
        if audio_batch.ndim != 4:
            raise ValueError("Expected audio batch of shape (B, 1, F, T)")
        if audio_batch.size(2) != self.input_bins:
            raise ValueError(
                f"Expected {self.input_bins} mel bins, received {audio_batch.size(2)}."
            )

        sequence = audio_batch.squeeze(1).transpose(1, 2)  # (B, T, F)
        rnn_output, _ = self.rnn(sequence)  # (B, T, hidden*dir)
        summary = rnn_output.mean(dim=1)
        logits = self.classifier(summary).squeeze(-1)
        scores = torch.sigmoid(logits)
        spectral_flow = self.spectral_head(summary)
        return AudioDetectorOutput(
            logits=logits,
            scores=scores,
            spectral_flow=spectral_flow,
            metadata={
                "feature_dim": summary.shape[-1],
                "bidirectional": self.bidirectional,
                "num_layers": self.num_layers,
            },
        )

    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        Stub for loading AASIST-like weights.
        """
        LOGGER.info("Loading audio detector checkpoint from %s", checkpoint_path)
        # To be implemented: torch.load and strict loading logic.
