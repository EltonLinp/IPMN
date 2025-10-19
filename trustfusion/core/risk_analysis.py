"""
Risk analysis utilities translating model scores into human-readable insights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass
class RiskAssessment:
    trust_index: float
    risk_level: str
    modality_contributions: Dict[str, float]
    alerts: List[str]


class RiskAnalyser:
    """
    Simple heuristic converter for trust scores.
    """

    def __init__(self, thresholds: Tuple[float, float]) -> None:
        self.low_threshold, self.high_threshold = thresholds

    def analyse(self, trust_score: torch.Tensor, modality_scores: Dict[str, torch.Tensor]) -> RiskAssessment:
        mean_trust = trust_score.mean().item()
        if mean_trust >= self.high_threshold:
            risk_level = "low"
        elif mean_trust <= self.low_threshold:
            risk_level = "high"
        else:
            risk_level = "medium"

        contributions = {name: float(score.mean().item()) for name, score in modality_scores.items()}
        alerts: List[str] = []
        if contributions["biosync"] < 0.5:
            alerts.append("Physiological mismatch detected.")
        if contributions["sync"] < 0.5:
            alerts.append("Audio-visual synchrony anomaly.")
        if contributions["diffusion"] < 0.4:
            alerts.append("Diffusion residuals suggest synthetic artefacts.")

        return RiskAssessment(
            trust_index=mean_trust,
            risk_level=risk_level,
            modality_contributions=contributions,
            alerts=alerts,
        )
