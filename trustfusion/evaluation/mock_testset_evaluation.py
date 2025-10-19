"""
Lightweight evaluation stub used during UI prototyping.

This module returns the mock evaluation payload so the FastAPI server can
exercise the reporting UI without running the full pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

ProgressCallback = Optional[Callable[[int, int], None]]


def _load_mock_payload() -> dict:
    mock_path = (
        Path(__file__).resolve().parent.parent
        / "api"
        / "static"
        / "mock"
        / "mock.json"
    )
    if not mock_path.exists():
        raise FileNotFoundError(f"Mock evaluation payload not found at {mock_path}")
    with mock_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate(
    model_path: Optional[Path] = None,
    progress_callback: ProgressCallback = None,
    group_sample_limit: Optional[int] = None,
) -> dict:
    """
    Return a canned evaluation summary for UI testing.

    Parameters are accepted to preserve compatiblity with the production
    evaluation entry point, but they are currently ignored.
    """

    payload = _load_mock_payload()
    summary = payload.get("evaluation")
    if summary is None:
        raise ValueError("Mock payload does not contain an 'evaluation' section.")

    details = summary.get("details", [])
    total = summary.get("total", len(details))
    if callable(progress_callback):
        progress_callback(0, total)
        progress_callback(total, total)
    return summary
