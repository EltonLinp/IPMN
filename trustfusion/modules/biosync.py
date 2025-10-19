"""
Bio-Sync module comparing physiological signals between modalities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class BioSyncOutput:
    biosync_score: torch.Tensor
    rppg_bpm: torch.Tensor
    audio_bpm: torch.Tensor
    metadata: Dict[str, Any]


class BioSyncAnalyzer:
    """
    Compute coherence between rPPG-derived heart rate and audio breathing tempo.
    """

    def __init__(self, min_bpm: float = 40.0, max_bpm: float = 180.0) -> None:
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm

    def _estimate_bpm(self, signal: torch.Tensor, sample_rate: int) -> torch.Tensor:
        fft = torch.fft.rfft(signal, dim=-1)
        freqs = torch.fft.rfftfreq(signal.shape[-1], d=1.0 / sample_rate)
        power = torch.abs(fft) ** 2
        mask = (freqs >= self.min_bpm / 60.0) & (freqs <= self.max_bpm / 60.0)
        masked_power = power[..., mask]
        masked_freqs = freqs[mask]
        if masked_power.numel() == 0:
            return torch.zeros(signal.shape[:-1], device=signal.device)
        peak_indices = masked_power.argmax(dim=-1)
        peak_freqs = masked_freqs[peak_indices]
        return peak_freqs * 60.0

    def compute(self, rppg_trace: torch.Tensor, audio_energy: torch.Tensor, sample_rate: int) -> BioSyncOutput:
        """
        Args:
            rppg_trace: tensor (B, T)
            audio_energy: tensor (B, T)
        """
        if rppg_trace.shape != audio_energy.shape:
            raise ValueError("rPPG and audio energy traces must align")
        rppg_bpm = self._estimate_bpm(rppg_trace, sample_rate)
        audio_bpm = self._estimate_bpm(audio_energy, sample_rate)
        biosync_score = 1 - torch.abs(rppg_bpm - audio_bpm) / torch.maximum(rppg_bpm, audio_bpm + 1e-6)
        biosync_score = torch.clamp(biosync_score, min=0.0, max=1.0)
        return BioSyncOutput(
            biosync_score=biosync_score,
            rppg_bpm=rppg_bpm,
            audio_bpm=audio_bpm,
            metadata={"sample_rate": sample_rate},
        )
