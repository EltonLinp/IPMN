"""
Pydantic models for TrustFusion API.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class VerificationRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Client-provided session identifier.")
    video_frames: List[List[List[float]]] = Field(..., description="Placeholder for video tensor data.")
    audio_spectrogram: List[List[List[float]]] = Field(..., description="Placeholder for audio tensor data.")


class VerificationResponse(BaseModel):
    session_id: str
    trust_index: float
    risk_level: str
    modality_contributions: Dict[str, float]
    alerts: List[str]
    report_paths: Dict[str, str]
