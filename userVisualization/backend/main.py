from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import Mapping, Optional

from .db import UploadRecord, cleanup_expired_uploads, get_session, init_db
from .deepfake_scoring import compute_deepfake_score
from .model import get_service
from .id_matcher import match_id_to_video, match_id_to_selfie
from .runtime_config import get_setting

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("tri-modal-web")

SYNC_LOW_CONFIDENCE = "SYNC_LOW_CONFIDENCE"
SYNC_LOW_CONFIDENCE_POLICY_NO_REJECT = "no_reject"
SYNC_LOW_CONFIDENCE_POLICY_REVIEW_ALL = "review_all"
VALID_SYNC_LOW_CONFIDENCE_POLICIES = {
    SYNC_LOW_CONFIDENCE_POLICY_NO_REJECT,
    SYNC_LOW_CONFIDENCE_POLICY_REVIEW_ALL,
}
DECISION_REASON_TO_MESSAGE: dict[str, str] = {
    SYNC_LOW_CONFIDENCE: "Sync alignment unreliable (mismatch/interpolation/length).",
}

try:
    from term2.code.face_extractor import extract_id_face
except Exception as exc:  # pragma: no cover - optional dependency
    extract_id_face = None  # type: ignore[assignment]
    LOGGER.warning("ID face extractor unavailable: %s", exc)

app = FastAPI(title="Tri-Modal Deepfake Demo", version="0.1.0")

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
STORAGE_DIR = Path(__file__).resolve().parents[1] / "storage"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
if STORAGE_DIR.exists():
    app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", STORAGE_DIR / "uploads"))
ID_UPLOAD_DIR = Path(os.environ.get("ID_UPLOAD_DIR", STORAGE_DIR / "id_photos"))
ID_FACE_DIR = Path(os.environ.get("ID_FACE_DIR", STORAGE_DIR / "id_faces"))
EKYC_ID_UPLOAD_DIR = Path(os.environ.get("EKYC_ID_UPLOAD_DIR", STORAGE_DIR / "id_uploads"))
EKYC_SELFIE_UPLOAD_DIR = Path(os.environ.get("EKYC_SELFIE_UPLOAD_DIR", STORAGE_DIR / "selfie_uploads"))
SELFIE_FACE_DIR = Path(os.environ.get("SELFIE_FACE_DIR", STORAGE_DIR / "selfie_faces"))
VIDEO_FRAME_DIR = Path(os.environ.get("VIDEO_FRAME_DIR", STORAGE_DIR / "video_frames"))
RETENTION_DAYS = max(int(os.environ.get("UPLOAD_RETENTION_DAYS", "30")), 1)
_cleanup_lock = threading.Lock()
_last_cleanup: Optional[datetime] = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    try:
        init_db()
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("Database init failed; continuing without DB: %s", exc)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ID_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ID_FACE_DIR.mkdir(parents=True, exist_ok=True)
    EKYC_ID_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    EKYC_SELFIE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    SELFIE_FACE_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_FRAME_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/")
async def index() -> FileResponse:
    if not FRONTEND_DIR.exists():
        raise HTTPException(status_code=404, detail="Front-end assets not found.")
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(...),
    id_photo: UploadFile = File(...),
    user_name: Optional[str] = Form(default=None),
    user_phone: Optional[str] = Form(default=None),
    db: Session = Depends(get_session),
) -> dict[str, object]:
    _maybe_cleanup(db)
    filename = video.filename or "upload.webm"
    suffix = Path(filename).suffix or ".webm"
    target_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    content = await video.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload.")
    try:
        target_path.write_bytes(content)
    except OSError as exc:
        LOGGER.exception("Failed to save upload: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to store upload.") from exc

    id_content = await id_photo.read()
    if not id_content:
        raise HTTPException(status_code=400, detail="Empty ID photo upload.")
    id_name = id_photo.filename or "id_photo.jpg"
    id_suffix = Path(id_name).suffix or ".jpg"
    id_path = ID_UPLOAD_DIR / f"{uuid4().hex}{id_suffix}"
    try:
        id_path.write_bytes(id_content)
    except OSError as exc:
        LOGGER.exception("Failed to save ID photo: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to store ID photo.") from exc

    record = UploadRecord(
        video_path=str(target_path),
        user_name=_normalize_text(user_name),
        user_phone=_normalize_text(user_phone),
        id_photo_path=str(id_path),
    )
    try:
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as exc:  # pragma: no cover - runtime safety
        db.rollback()
        LOGGER.warning("Failed to persist upload metadata: %s", exc)

    id_face_result: dict[str, object] = {"ok": False, "error": "read_error"}
    if extract_id_face is not None:
        try:
            id_face_result = extract_id_face(str(id_path), str(ID_FACE_DIR))
        except Exception as exc:  # pragma: no cover - runtime safety
            LOGGER.exception("ID face extraction failed: %s", exc)
            id_face_result = {"ok": False, "error": "read_error"}
    if id_face_result.get("ok") and isinstance(id_face_result.get("crop_path"), str):
        record.id_face_path = str(id_face_result["crop_path"])
        try:
            db.add(record)
            db.commit()
            db.refresh(record)
        except Exception as exc:  # pragma: no cover - runtime safety
            db.rollback()
            LOGGER.exception("Failed to update ID face path: %s", exc)

    try:
        service = get_service()
        result = service.analyze(target_path)
    except Exception as exc:  # pragma: no cover - runtime safety
        LOGGER.exception("Video analysis failed: %s", exc)
        raise HTTPException(status_code=500, detail="Video analysis failed. Please retry.") from exc
    id_face_payload = dict(id_face_result)
    crop_path = id_face_payload.get("crop_path")
    if isinstance(crop_path, str):
        id_face_payload["crop_url"] = f"/storage/id_faces/{Path(crop_path).name}"
    id_face_payload["photo_url"] = f"/storage/id_photos/{Path(id_path).name}"
    if id_face_payload.get("ok") and isinstance(crop_path, str):
        id_match = match_id_to_video(id_face_path=crop_path, video_path=str(target_path))
    else:
        id_match = {"ok": False, "error": "id_face_unavailable"}
    return {
        "result": result,
        "id_face": id_face_payload,
        "id_match": id_match,
    }


@app.post("/api/ekyc/evaluate")
async def evaluate_ekyc(
    id_image: Optional[UploadFile] = File(default=None),
    selfie_image: Optional[UploadFile] = File(default=None),
    video: Optional[UploadFile] = File(default=None),
) -> JSONResponse:
    if id_image is None:
        return _error_response("missing_file", "id_image is required.")
    if selfie_image is None:
        return _error_response("missing_file", "selfie_image is required.")
    if video is None:
        return _error_response("missing_file", "video is required.")

    save_start = perf_counter()
    try:
        id_path = await _save_upload(id_image, EKYC_ID_UPLOAD_DIR, "id_image.jpg")
        selfie_path = await _save_upload(selfie_image, EKYC_SELFIE_UPLOAD_DIR, "selfie_image.jpg")
        video_path = await _save_upload(video, UPLOAD_DIR, "upload.webm")
    except ValueError as exc:
        return _error_response("empty_upload", str(exc))
    except OSError as exc:
        LOGGER.exception("Failed to save uploads: %s", exc)
        return _error_response("save_failed", "Failed to store uploaded files.", status_code=500)
    LOGGER.info("Saved uploads in %.3fs", perf_counter() - save_start)

    id_face_result: dict[str, object] = {"ok": False, "error": "read_error"}
    id_face_path: Optional[str] = None
    if extract_id_face is not None:
        try:
            id_face_result = extract_id_face(str(id_path), str(ID_FACE_DIR))
        except Exception as exc:  # pragma: no cover - runtime safety
            LOGGER.exception("ID face extraction failed: %s", exc)
            id_face_result = {"ok": False, "error": "read_error"}
    if id_face_result.get("ok") and isinstance(id_face_result.get("crop_path"), str):
        id_face_path = str(id_face_result["crop_path"])

    selfie_face_path: Optional[str] = None
    selfie_face_result: dict[str, object] = {"ok": False, "error": "read_error"}
    if extract_id_face is not None:
        try:
            selfie_face_result = extract_id_face(str(selfie_path), str(SELFIE_FACE_DIR))
        except Exception as exc:  # pragma: no cover - runtime safety
            LOGGER.exception("Selfie face extraction failed: %s", exc)
            selfie_face_result = {"ok": False, "error": "read_error"}
    if selfie_face_result.get("ok") and isinstance(selfie_face_result.get("crop_path"), str):
        selfie_face_path = str(selfie_face_result["crop_path"])

    deepfake_start = perf_counter()
    try:
        service = get_service()
        deepfake_result = service.analyze(video_path)
    except Exception as exc:  # pragma: no cover - runtime safety
        LOGGER.exception("Video analysis failed: %s", exc)
        return _error_response("deepfake_failed", "Video analysis failed.", status_code=500)
    LOGGER.info("Deepfake analysis in %.3fs", perf_counter() - deepfake_start)

    id_video_match: dict[str, object] = {"ok": False, "error": "id_face_unavailable"}
    id_selfie_match: dict[str, object] = {"ok": False, "error": "id_face_unavailable"}
    if id_face_path:
        match_start = perf_counter()
        id_video_match = match_id_to_video(id_face_path=id_face_path, video_path=str(video_path))
        LOGGER.info("ID-video match in %.3fs", perf_counter() - match_start)
        selfie_source = selfie_face_path or str(selfie_path)
        selfie_out = None if selfie_face_path else str(_selfie_face_target(selfie_path))
        match_start = perf_counter()
        id_selfie_match = match_id_to_selfie(
            id_face_path=id_face_path,
            selfie_path=selfie_source,
            out_aligned_face_path=selfie_out,
        )
        LOGGER.info("ID-selfie match in %.3fs", perf_counter() - match_start)
        if not selfie_face_path and isinstance(id_selfie_match.get("aligned_face_path"), str):
            selfie_face_path = id_selfie_match.get("aligned_face_path")  # type: ignore[assignment]

    deepfake_score, deepfake_label = _compute_deepfake_score(deepfake_result)
    id_selfie_score = _safe_float(id_selfie_match.get("prob"))
    id_video_score = _safe_float(id_video_match.get("prob"))
    sync_quality = _extract_sync_quality(deepfake_result)
    sync_low_confidence = _is_sync_low_confidence(sync_quality)
    sync_quality_payload = _sync_quality_payload(sync_quality)
    sync_low_confidence_policy = _sync_low_confidence_policy_cfg()
    deepfake_quality_flags = [SYNC_LOW_CONFIDENCE] if sync_low_confidence else []

    risk, decision, reasons, decision_reason = _fuse_decision(
        deepfake_score=deepfake_score,
        id_selfie_score=id_selfie_score,
        id_video_score=id_video_score,
        id_selfie_ok=bool(id_selfie_match.get("ok")),
        id_video_ok=bool(id_video_match.get("ok")),
        sync_low_confidence=sync_low_confidence,
        sync_low_confidence_policy=sync_low_confidence_policy,
    )

    best_frame_index = _safe_int(id_video_match.get("best_frame"))
    video_best_frame_path: Optional[str] = None
    if best_frame_index is not None and best_frame_index >= 0:
        extracted = _extract_video_frame(video_path, best_frame_index, VIDEO_FRAME_DIR)
        if extracted is not None:
            video_best_frame_path = str(extracted)

    payload = {
        "deepfake": {
            "score": deepfake_score,
            "label": deepfake_label,
            "details": {
                "video": deepfake_result.get("video"),
                "audio": deepfake_result.get("audio"),
                "sync": deepfake_result.get("sync"),
            },
            "branch_raw": {
                "final_raw": deepfake_result.get("final_raw"),
            },
            "quality_flags": deepfake_quality_flags,
            "quality": {
                "sync": sync_quality_payload,
            },
        },
        "match": {
            "id_selfie": id_selfie_match,
            "id_video": id_video_match,
            "best_video_frame_index": best_frame_index,
            "thresholds": {
                "id_selfie_pass": 0.6,
                "id_video_pass": 0.6,
            },
        },
        "fusion": {
            "risk": risk,
            "decision": decision,
            "reason": reasons,
        },
        "decision_reason": decision_reason,
        "artifacts": {
            "id_face_path": _storage_url(id_face_path),
            "selfie_face_path": _storage_url(selfie_face_path),
            "video_best_frame_path": _storage_url(video_best_frame_path),
        },
    }
    return JSONResponse(content=payload)


def _normalize_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned if cleaned else None


def _error_response(error_code: str, message: str, *, status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error_code": error_code, "message": message},
    )


async def _save_upload(upload: UploadFile, target_dir: Path, default_name: str) -> Path:
    content = await upload.read()
    if not content:
        raise ValueError(f"Empty upload for {default_name}.")
    filename = upload.filename or default_name
    suffix = Path(filename).suffix or Path(default_name).suffix or ".bin"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{uuid4().hex}{suffix}"
    target_path.write_bytes(content)
    return target_path


def _selfie_face_target(selfie_path: Path) -> Path:
    stem = selfie_path.stem or uuid4().hex
    return SELFIE_FACE_DIR / f"{stem}_selfie.png"


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_cfg(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        raw = get_setting(name, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _compute_deepfake_score(result: dict[str, object]) -> tuple[float, str]:
    final = result.get("final", {}) if isinstance(result, dict) else {}
    if isinstance(final, Mapping):
        final_score = _safe_float(final.get("fake"))
        if final_score is not None:
            return final_score, str(final.get("label", "Unknown"))

    weights = {
        "video": _float_cfg("DEEPFAKE_WEIGHT_VIDEO", 0.7),
        "audio": _float_cfg("DEEPFAKE_WEIGHT_AUDIO", 0.15),
        "sync": _float_cfg("DEEPFAKE_WEIGHT_SYNC", 0.15),
    }
    scored = compute_deepfake_score(
        {
            "audio": result.get("audio"),
            "video": result.get("video"),
            "sync": result.get("sync"),
        },
        weights=weights,
    )
    weighted_score = scored.get("weighted_score")
    if isinstance(weighted_score, (int, float)):
        return float(weighted_score), str(scored.get("label", "Unknown"))
    fallback = result.get("final_raw", {}) if isinstance(result, dict) else {}
    score = _safe_float(fallback.get("fake"))
    label = str(fallback.get("label", "Unknown"))
    return score, label


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _fuse_decision(
    *,
    deepfake_score: float,
    id_selfie_score: float,
    id_video_score: float,
    id_selfie_ok: bool,
    id_video_ok: bool,
    sync_low_confidence: bool = False,
    sync_low_confidence_policy: str = SYNC_LOW_CONFIDENCE_POLICY_NO_REJECT,
) -> tuple[float, str, list[str], list[str]]:
    w1 = _float_cfg("RISK_WEIGHT_DEEPFAKE", 1.0 / 3.0)
    w2 = _float_cfg("RISK_WEIGHT_ID_SELFIE", 1.0 / 3.0)
    w3 = _float_cfg("RISK_WEIGHT_ID_VIDEO", 1.0 / 3.0)
    total_w = w1 + w2 + w3
    if total_w <= 0.0:
        w1 = w2 = w3 = 1.0 / 3.0
    else:
        w1 /= total_w
        w2 /= total_w
        w3 /= total_w

    selfie_val = id_selfie_score if id_selfie_ok else 0.0
    video_val = id_video_score if id_video_ok else 0.0
    risk = w1 * deepfake_score + w2 * (1.0 - selfie_val) + w3 * (1.0 - video_val)
    pass_threshold = _float_cfg("RISK_THRESHOLD_PASS", 0.4)
    reject_threshold = _float_cfg("RISK_THRESHOLD_REJECT", 0.7)
    hard_reject_df_threshold = _float_cfg("RISK_HARD_REJECT_DF_THRESHOLD", 0.8)
    if reject_threshold < pass_threshold:
        reject_threshold = pass_threshold

    hard_reject_df = deepfake_score >= hard_reject_df_threshold
    if hard_reject_df:
        decision = "REJECT"
    elif risk < pass_threshold:
        decision = "PASS"
    elif risk < reject_threshold:
        decision = "REVIEW"
    else:
        decision = "REJECT"
    reasons: list[str] = []
    if hard_reject_df:
        reasons.append("Deepfake score critical")
    if deepfake_score > 0.6:
        reasons.append("Deepfake score high")
    if not id_selfie_ok or id_selfie_score < 0.6:
        reasons.append("ID vs selfie low")
    if not id_video_ok or id_video_score < 0.6:
        reasons.append("ID vs video low")
    decision_reason: list[str] = []
    if sync_low_confidence and not hard_reject_df:
        decision_reason.append(SYNC_LOW_CONFIDENCE)
        policy = _normalize_sync_low_confidence_policy(sync_low_confidence_policy)
        if policy == SYNC_LOW_CONFIDENCE_POLICY_REVIEW_ALL:
            decision = "REVIEW"
        elif policy == SYNC_LOW_CONFIDENCE_POLICY_NO_REJECT and decision == "REJECT":
            decision = "REVIEW"
    if not reasons and not decision_reason:
        reasons.append("All checks within threshold")
    reasons = _merge_reason_messages(reasons, decision_reason)
    return float(risk), decision, reasons, _unique_codes(decision_reason)


def _extract_sync_quality(result: dict[str, object]) -> dict[str, object]:
    if not isinstance(result, Mapping):
        return {}
    direct = result.get("sync_quality")
    if isinstance(direct, Mapping):
        return dict(direct)
    debug_payload = result.get("debug")
    if not isinstance(debug_payload, Mapping):
        return {}
    sync_quality = debug_payload.get("sync_quality")
    if isinstance(sync_quality, Mapping):
        return dict(sync_quality)
    preprocess = debug_payload.get("preprocess")
    if not isinstance(preprocess, Mapping):
        return {}
    sync = preprocess.get("sync")
    if not isinstance(sync, Mapping):
        return {}
    return {
        "mismatch": _coerce_bool(sync.get("mismatch")) or _coerce_bool(sync.get("t_mismatch")),
        "interpolated": _coerce_bool(sync.get("interpolated")),
        "length_bad": _coerce_bool(sync.get("length_bad")),
    }


def _is_sync_low_confidence(sync_quality: Mapping[str, object] | None) -> bool:
    if not isinstance(sync_quality, Mapping):
        return False
    return bool(
        _coerce_bool(sync_quality.get("mismatch"))
        or _coerce_bool(sync_quality.get("interpolated"))
        or _coerce_bool(sync_quality.get("length_bad"))
    )


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _sync_quality_payload(sync_quality: Mapping[str, object] | None) -> dict[str, bool]:
    if not isinstance(sync_quality, Mapping):
        return {
            "mismatch": False,
            "interpolated": False,
            "length_bad": False,
        }
    return {
        "mismatch": _coerce_bool(sync_quality.get("mismatch")),
        "interpolated": _coerce_bool(sync_quality.get("interpolated")),
        "length_bad": _coerce_bool(sync_quality.get("length_bad")),
    }


def _sync_low_confidence_policy_cfg() -> str:
    raw = os.environ.get("SYNC_LOW_CONFIDENCE_POLICY")
    if raw is None:
        raw = get_setting("sync_low_confidence_policy", SYNC_LOW_CONFIDENCE_POLICY_NO_REJECT)
    if raw is None:
        raw = get_setting("SYNC_LOW_CONFIDENCE_POLICY", SYNC_LOW_CONFIDENCE_POLICY_NO_REJECT)
    return _normalize_sync_low_confidence_policy(raw)


def _normalize_sync_low_confidence_policy(value: object) -> str:
    text = str(value).strip().lower()
    if text not in VALID_SYNC_LOW_CONFIDENCE_POLICIES:
        return SYNC_LOW_CONFIDENCE_POLICY_NO_REJECT
    return text


def _merge_reason_messages(reasons: list[str], decision_reason: list[str]) -> list[str]:
    merged = list(reasons)
    for code in _unique_codes(decision_reason):
        message = DECISION_REASON_TO_MESSAGE.get(code)
        if message:
            merged.append(message)
    return _unique_messages(merged)


def _unique_codes(codes: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for code in codes:
        normalized = str(code).strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _unique_messages(messages: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for message in messages:
        text = str(message).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _storage_url(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    try:
        rel = Path(path_value).resolve().relative_to(STORAGE_DIR.resolve())
    except Exception:
        return None
    return f"/storage/{rel.as_posix()}"


def _extract_video_frame(video_path: Path, frame_index: int, out_dir: Path) -> Optional[Path]:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("OpenCV unavailable for frame extraction: %s", exc)
        return None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total > 0 and frame_index >= total:
            frame_index = max(total // 2, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if not success or frame is None:
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_path.stem}_frame_{frame_index:06d}.jpg"
        if not cv2.imwrite(str(out_path), frame):
            return None
        return out_path
    finally:
        cap.release()


def _maybe_cleanup(session: Session) -> None:
    global _last_cleanup
    now = datetime.now(timezone.utc)
    if _last_cleanup and now - _last_cleanup < timedelta(hours=24):
        return
    with _cleanup_lock:
        if _last_cleanup and now - _last_cleanup < timedelta(hours=24):
            return
        try:
            deleted = cleanup_expired_uploads(session, retention_days=RETENTION_DAYS)
            if deleted:
                LOGGER.info("Removed %d expired uploads.", deleted)
        except Exception as exc:  # pragma: no cover - runtime safety
            LOGGER.warning("Failed to cleanup uploads: %s", exc)
        _last_cleanup = now
