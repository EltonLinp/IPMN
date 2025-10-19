"""
FastAPI server exposing the TrustFusion pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import build_default_system
from ..config import FusionConfig
from ..evaluation.mock_testset_evaluation import evaluate as evaluate_testset
from ..utils.logger import get_logger
from .schema import VerificationRequest, VerificationResponse

LOGGER = get_logger(__name__)

app = FastAPI(title="TrustFusion Deepfake Defense API")
system = build_default_system()
LATEST_EVAL: Optional[dict] = None
EVAL_PER_GROUP_LIMIT: Optional[int] = 100
EVAL_PROGRESS = {
    "status": "idle",
    "processed": 0,
    "total": 0,
    "message": "Idle",
}
_PROGRESS_LOCK = Lock()

app_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(app_dir / "templates"))
static_dir = app_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    summary = system.dataset_summary()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "summary": summary, "eval_summary": LATEST_EVAL},
    )


@app.get("/detection", response_class=HTMLResponse)
def detection(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("detection.html", {"request": request})


@app.get("/dataset/summary")
def dataset_summary() -> dict:
    return system.dataset_summary()


@app.post("/evaluation/run")
def run_evaluation() -> dict:
    """Run the test-set evaluation and persist the summary."""

    global LATEST_EVAL
    with _PROGRESS_LOCK:
        if EVAL_PROGRESS.get("status") == "running":
            raise HTTPException(status_code=409, detail="Evaluation already in progress.")
        if EVAL_PER_GROUP_LIMIT:
            message = f"Preparing evaluation dataset (≤{EVAL_PER_GROUP_LIMIT} per folder)..."
        else:
            message = "Preparing evaluation dataset..."
        EVAL_PROGRESS.update(
            {
                "status": "running",
                "processed": 0,
                "total": 0,
                "message": message,
            }
        )

    def _progress_callback(processed: int, total: int) -> None:
        message = (
            f"Processing batches: {processed}/{total}" if total else "Processing batches..."
        )
        with _PROGRESS_LOCK:
            current_total = max(total, EVAL_PROGRESS.get("total", 0))
            current_processed = min(processed, current_total) if current_total else processed
            EVAL_PROGRESS.update(
                {
                    "status": "running",
                    "processed": current_processed,
                    "total": current_total,
                    "message": message,
                }
            )

    model_path = Path("outputs/trained_classifier.pt")
    try:
        LATEST_EVAL = evaluate_testset(
            model_path=model_path if model_path.exists() else None,
            progress_callback=_progress_callback,
            group_sample_limit=EVAL_PER_GROUP_LIMIT,
        )
    except Exception as exc:  # pragma: no cover - surfaced to client
        with _PROGRESS_LOCK:
            EVAL_PROGRESS.update(
                {
                    "status": "error",
                    "message": f"Evaluation failed: {exc}",
                }
            )
        raise

    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = outputs_dir / "last_test_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(LATEST_EVAL, handle, ensure_ascii=False, indent=2)

    with _PROGRESS_LOCK:
        total = LATEST_EVAL.get("total", EVAL_PROGRESS.get("total", 0))
        EVAL_PROGRESS.update(
            {
                "status": "completed",
                "processed": total,
                "total": total,
                "message": "Evaluation complete.",
            }
        )

    return LATEST_EVAL


@app.get("/evaluation/summary")
def get_latest_evaluation() -> dict:
    """Return the most recent evaluation summary if available."""

    if LATEST_EVAL is not None:
        return LATEST_EVAL
    summary_path = Path("outputs/last_test_summary.json")
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {"message": "Evaluation not run yet"}


@app.get("/evaluation/progress")
def get_evaluation_progress() -> dict:
    """Return the most recent evaluation progress snapshot."""

    with _PROGRESS_LOCK:
        return dict(EVAL_PROGRESS)


@app.post("/verify", response_model=VerificationResponse)
def verify(request: VerificationRequest) -> VerificationResponse:
    """Handle verification requests with placeholder tensor conversion."""

    LOGGER.info("Received verification request for session %s", request.session_id)
    if not request.video_frames or not request.audio_spectrogram:
        raise HTTPException(status_code=400, detail="Video and audio payloads cannot be empty.")

    video_tensor = torch.tensor(request.video_frames, dtype=torch.float32)
    if video_tensor.ndim != 5:
        raise HTTPException(status_code=400, detail="Video tensor must have shape (B, C, T, H, W).")
    audio_tensor = torch.tensor(request.audio_spectrogram, dtype=torch.float32)
    if audio_tensor.ndim != 4:
        raise HTTPException(status_code=400, detail="Audio tensor must have shape (B, 1, F, T).")

    batch, _, time_steps, _, _ = video_tensor.shape
    lip_feats = torch.zeros(batch, time_steps, 64)
    audio_feats = torch.zeros(batch, time_steps, 64)
    rppg_trace = torch.zeros(batch, time_steps)
    audio_energy = torch.zeros(batch, time_steps)

    result = system.run(
        video_tensor=video_tensor,
        audio_tensor=audio_tensor,
        lip_feats=lip_feats,
        audio_feats=audio_feats,
        rppg_trace=rppg_trace,
        audio_energy=audio_energy,
        sample_rate=FusionConfig().detector.sample_rate,
        session_id=request.session_id,
        generate_report=False,
    )
    return VerificationResponse(
        session_id=result.session_id,
        trust_index=result.risk_assessment.trust_index,
        risk_level=result.risk_assessment.risk_level,
        modality_contributions=result.risk_assessment.modality_contributions,
        alerts=result.risk_assessment.alerts,
        report_paths=result.report_paths,
    )


@app.post("/demo/verify")
def demo_verify() -> dict:
    """Run the system with synthetic tensors for front-end demonstration."""

    video_tensor = torch.rand(1, 3, 8, 32, 32)
    audio_tensor = torch.rand(1, 1, 64, 8)
    lip_feats = torch.rand(1, 8, 64)
    audio_feats = torch.rand(1, 8, 64)
    rppg_trace = torch.rand(1, 8)
    audio_energy = torch.rand(1, 8)

    result = system.run(
        video_tensor=video_tensor,
        audio_tensor=audio_tensor,
        lip_feats=lip_feats,
        audio_feats=audio_feats,
        rppg_trace=rppg_trace,
        audio_energy=audio_energy,
        sample_rate=FusionConfig().detector.sample_rate,
        generate_report=False,
    )

    return {
        "session_id": result.session_id,
        "trust_index": result.risk_assessment.trust_index,
        "risk_level": result.risk_assessment.risk_level,
        "modality_contributions": result.risk_assessment.modality_contributions,
        "alerts": result.risk_assessment.alerts,
    }
