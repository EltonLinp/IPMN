from __future__ import annotations

from typing import Mapping, Optional


DEFAULT_BRANCH_WEIGHTS: dict[str, float] = {
    "video": 0.7,
    "audio": 0.15,
    "sync": 0.15,
}


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_fake_prob(branch: object) -> Optional[float]:
    if not isinstance(branch, Mapping):
        return None
    if "fake" not in branch:
        return None
    return _safe_float(branch.get("fake"))


def compute_deepfake_score(
    branch_probs: Mapping[str, object],
    weights: Mapping[str, float] | None = None,
    *,
    threshold: float = 0.5,
) -> dict[str, object]:
    resolved_weights = dict(DEFAULT_BRANCH_WEIGHTS)
    if weights:
        for name, value in weights.items():
            try:
                resolved_weights[str(name)] = float(value)
            except (TypeError, ValueError):
                continue

    weighted_terms: dict[str, float] = {}
    weight_terms: dict[str, float] = {}
    for name in ("video", "audio", "sync"):
        fake_prob = _extract_fake_prob(branch_probs.get(name))
        if fake_prob is None:
            continue
        weight = float(resolved_weights.get(name, 0.0))
        weighted_terms[name] = fake_prob * weight
        weight_terms[name] = weight

    total_weight = float(sum(weight_terms.values()))
    if total_weight <= 0.0:
        return {
            "weighted_score": None,
            "label": "Unknown",
            "weights": resolved_weights,
            "used_weights": weight_terms,
        }

    weighted_score = float(sum(weighted_terms.values()) / total_weight)
    label = "Fake" if weighted_score >= float(threshold) else "Real"
    return {
        "weighted_score": weighted_score,
        "label": label,
        "weights": resolved_weights,
        "used_weights": weight_terms,
    }
