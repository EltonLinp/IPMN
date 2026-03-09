from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple
from uuid import uuid4

import cv2  # type: ignore
import numpy as np
import onnxruntime as ort  # noqa: F401
from insightface.app import FaceAnalysis  # type: ignore
try:
    from insightface.utils.face_align import norm_crop  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    norm_crop = None  # type: ignore

LOGGER = logging.getLogger(__name__)
_FACE_APP: Optional[FaceAnalysis] = None
_FACE_MODEL_NAME: Optional[str] = None

try:
    import decord  # type: ignore

    _DECORD_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    decord = None  # type: ignore
    _DECORD_AVAILABLE = False


def _get_face_app() -> FaceAnalysis:
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP
    global _FACE_MODEL_NAME
    model_candidates = _get_model_candidates()
    last_exc: Exception | None = None
    for name in model_candidates:
        try:
            app = FaceAnalysis(name=name)
            try:
                app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.1)
            except Exception:
                app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.1)
            _FACE_APP = app
            _FACE_MODEL_NAME = name
            LOGGER.info("Face matcher using model: %s", name)
            return app
        except Exception as exc:
            last_exc = exc
            LOGGER.warning("Failed to init face model '%s': %s", name, exc)
            continue
    raise RuntimeError("No face recognition model could be initialized.") from last_exc


def _get_model_candidates() -> list[str]:
    preferred = os.environ.get("FACE_REC_MODEL", "").strip()
    candidates = [preferred] if preferred else []
    candidates.extend(["magface", "antelopev2", "buffalo_l"])
    seen: set[str] = set()
    ordered: list[str] = []
    for name in candidates:
        if not name:
            continue
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def match_id_to_video(
    *,
    id_face_path: str,
    video_path: str,
    min_face_size: int = 50,
    target_frames: int = 24,
    top_k: int = 3,
) -> dict:
    try:
        id_face = _read_image_bgr(id_face_path)
        if id_face is None:
            return {"ok": False, "error": "read_error", "debug": {"stage": "id_face_read"}}
        id_embedding = _extract_embedding_from_aligned(id_face)
        if id_embedding is None:
            id_embedding, id_face_size, has_face = _extract_embedding(
                id_face,
                min_face_size=min_face_size,
            )
            if id_embedding is None:
                if has_face:
                    return {
                        "ok": False,
                        "error": "face_too_small",
                        "debug": {"stage": "id_face_detect", "max_face_size": id_face_size},
                    }
                return {
                    "ok": False,
                    "error": "no_face_detected",
                    "debug": {"stage": "id_face_detect"},
                }
        debug_dir = _get_debug_dir(video_path)
        similarities, best_frame, stats = _match_video_frames(
            video_path,
            id_embedding,
            min_face_size=min_face_size,
            target_frames=target_frames,
            debug_dir=debug_dir,
        )
        if not similarities:
            fallback_min = max(20, min_face_size // 2)
            if fallback_min != min_face_size:
                similarities, best_frame, stats = _match_video_frames(
                    video_path,
                    id_embedding,
                    min_face_size=fallback_min,
                    target_frames=target_frames,
                    debug_dir=debug_dir,
                )
            if not similarities:
                if not stats.get("decode_ok", True):
                    return {"ok": False, "error": "read_error", "debug": stats}
                if stats.get("frames_with_faces", 0) > 0 and stats.get("max_face_size", 0) < min_face_size:
                    return {"ok": False, "error": "face_too_small", "debug": stats}
                return {"ok": False, "error": "no_face_detected", "debug": stats}
        score = _topk_mean(similarities, top_k)
        prob = _sigmoid(score, scale=12.0, bias=-4.2)
        return {
            "ok": True,
            "score": float(score),
            "prob": float(prob),
            "best_frame": best_frame,
            "samples": len(similarities),
            "debug": stats,
        }
    except Exception as exc:
        LOGGER.exception("ID-video match failed: %s", exc)
    return {"ok": False, "error": "read_error"}


def match_id_to_selfie(
    *,
    id_face_path: str,
    selfie_path: str,
    out_aligned_face_path: Optional[str] = None,
    min_face_size: int = 50,
) -> dict:
    try:
        id_face = _read_image_bgr(id_face_path)
        if id_face is None:
            return {"ok": False, "error": "read_error", "debug": {"stage": "id_face_read"}}
        id_embedding = _extract_embedding_from_aligned(id_face)
        if id_embedding is None:
            id_embedding, id_face_size, has_face = _extract_embedding(
                id_face,
                min_face_size=min_face_size,
            )
            if id_embedding is None:
                if has_face:
                    return {
                        "ok": False,
                        "error": "face_too_small",
                        "debug": {"stage": "id_face_detect", "max_face_size": id_face_size},
                    }
                return {
                    "ok": False,
                    "error": "no_face_detected",
                    "debug": {"stage": "id_face_detect"},
                }

        selfie_img = _read_image_bgr(selfie_path)
        if selfie_img is None:
            return {"ok": False, "error": "read_error", "debug": {"stage": "selfie_read"}}

        selfie_embedding = None
        if selfie_img.shape[0] == 112 and selfie_img.shape[1] == 112:
            selfie_embedding = _extract_embedding_from_aligned(selfie_img)
        if selfie_embedding is None:
            selfie_embedding, selfie_face_size, has_face = _extract_embedding(
                selfie_img,
                min_face_size=min_face_size,
            )
            if selfie_embedding is None:
                if has_face:
                    return {
                        "ok": False,
                        "error": "face_too_small",
                        "debug": {"stage": "selfie_detect", "max_face_size": selfie_face_size},
                    }
                return {
                    "ok": False,
                    "error": "no_face_detected",
                    "debug": {"stage": "selfie_detect"},
                }

        aligned_path = None
        if out_aligned_face_path:
            aligned_path = _save_aligned_face(selfie_img, out_aligned_face_path, min_face_size=min_face_size)

        score = float(np.dot(id_embedding, selfie_embedding))
        prob = float((score + 1.0) / 2.0)
        payload = {
            "ok": True,
            "score": score,
            "prob": prob,
        }
        if aligned_path:
            payload["aligned_face_path"] = aligned_path
        return payload
    except Exception as exc:
        LOGGER.exception("ID-selfie match failed: %s", exc)
        return {"ok": False, "error": "read_error"}


def _read_image_bgr(path: str) -> Optional[np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image


def _extract_embedding(
    image_bgr: np.ndarray,
    *,
    min_face_size: int,
) -> tuple[Optional[np.ndarray], float, bool]:
    app = _get_face_app()
    max_face_size = 0.0
    has_face = False
    for image, scale in _iter_detection_images(image_bgr):
        faces = app.get(image)
        if not faces:
            continue
        has_face = True
        best = max(
            faces,
            key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
        )
        x1, y1, x2, y2 = best.bbox.astype(float)
        w = max(x2 - x1, 0.0)
        h = max(y2 - y1, 0.0)
        face_size = float(min(w, h))
        face_size_orig = face_size / max(scale, 1e-6)
        max_face_size = max(max_face_size, face_size_orig)
        if face_size_orig < float(min_face_size):
            continue
        embedding = getattr(best, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(best, "embedding", None)
        if embedding is None:
            continue
        emb = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb)
        if norm <= 0:
            continue
        return emb / norm, face_size_orig, True
    return None, max_face_size, has_face


def _iter_detection_images(image_bgr: np.ndarray) -> list[tuple[np.ndarray, float]]:
    h, w = image_bgr.shape[:2]
    long_side = max(h, w)
    if long_side <= 0:
        return [(image_bgr, 1.0)]
    target_long_sizes = [960, 768, 640, 512, 384]
    candidates: list[tuple[np.ndarray, float]] = []
    seen: set[float] = set()
    for target in target_long_sizes:
        scale = float(target) / float(long_side)
        if scale <= 0.0 or scale > 4.0:
            continue
        key = round(scale, 3)
        if key in seen:
            continue
        seen.add(key)
        if abs(scale - 1.0) < 1e-3:
            candidates.append((image_bgr, 1.0))
            continue
        new_w = max(int(round(w * scale)), 1)
        new_h = max(int(round(h * scale)), 1)
        if new_w < 64 or new_h < 64:
            continue
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        candidates.append((resized, scale))
    if 1.0 not in seen:
        candidates.append((image_bgr, 1.0))
    return candidates


def _extract_embedding_from_aligned(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    app = _get_face_app()
    model = app.models.get("recognition")
    if model is None:
        return None
    if image_bgr.size == 0:
        return None
    face = cv2.resize(image_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
    embedding = model.get_feat(face)
    emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(emb)
    if norm <= 0:
        return None
    return emb / norm


def _save_aligned_face(
    image_bgr: np.ndarray,
    out_path: str,
    *,
    min_face_size: int,
) -> Optional[str]:
    try:
        if image_bgr is None or image_bgr.size == 0:
            return None
        if image_bgr.shape[0] == 112 and image_bgr.shape[1] == 112:
            aligned = image_bgr
        else:
            aligned = _align_face(image_bgr, min_face_size=min_face_size)
            if aligned is None:
                return None
        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), aligned):
            return None
        return str(target)
    except Exception as exc:  # pragma: no cover - runtime safety
        LOGGER.warning("Failed to save aligned face: %s", exc)
        return None


def _align_face(
    image_bgr: np.ndarray,
    *,
    min_face_size: int,
) -> Optional[np.ndarray]:
    app = _get_face_app()
    best_face = None
    best_image = None
    best_scale = 1.0
    best_area = 0.0
    for image, scale in _iter_detection_images(image_bgr):
        faces = app.get(image)
        if not faces:
            continue
        candidate = max(
            faces,
            key=lambda face: float((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])),
        )
        x1, y1, x2, y2 = candidate.bbox.astype(float)
        w = max(x2 - x1, 0.0)
        h = max(y2 - y1, 0.0)
        face_size_orig = float(min(w, h)) / max(scale, 1e-6)
        if face_size_orig < float(min_face_size):
            continue
        area = float(w * h)
        if area > best_area:
            best_area = area
            best_face = candidate
            best_image = image
            best_scale = scale
    if best_face is None or best_image is None:
        return None
    kps = getattr(best_face, "kps", None)
    if kps is not None and norm_crop is not None:
        try:
            return norm_crop(best_image, kps, image_size=112)
        except Exception:
            pass
    x1, y1, x2, y2 = best_face.bbox.astype(float)
    w = max(x2 - x1, 0.0)
    h = max(y2 - y1, 0.0)
    margin = 0.25
    mx1 = int(max(x1 - w * margin, 0))
    my1 = int(max(y1 - h * margin, 0))
    mx2 = int(min(x2 + w * margin, best_image.shape[1]))
    my2 = int(min(y2 + h * margin, best_image.shape[0]))
    if mx2 <= mx1 or my2 <= my1:
        return None
    crop = best_image[my1:my2, mx1:mx2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)


def _match_video_frames(
    video_path: str,
    id_embedding: np.ndarray,
    *,
    min_face_size: int,
    target_frames: int,
    debug_dir: Optional[Path] = None,
) -> Tuple[list[float], int, dict]:
    stats = {
        "decode_ok": True,
        "frames_seen": 0,
        "frames_with_faces": 0,
        "max_face_size": 0.0,
        "backend": None,
    }
    if debug_dir is not None:
        stats["debug_dir"] = str(debug_dir)
    if _DECORD_AVAILABLE:
        try:
            frames, indices = _read_frames_decord(video_path, target_frames)
            if frames:
                stats["backend"] = "decord"
                _maybe_dump_frames(frames, indices, debug_dir)
                similarities, best_frame, scored = _score_frames(
                    frames,
                    indices,
                    id_embedding,
                    min_face_size,
                )
                stats.update(scored)
                if stats.get("frames_with_faces", 0) > 0:
                    return similarities, best_frame, stats
                LOGGER.warning("No faces found via decord; falling back to OpenCV.")
        except Exception as exc:
            LOGGER.warning("Decord failed, falling back to OpenCV: %s", exc)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        stats["decode_ok"] = False
        stats["backend"] = "opencv"
        return [], -1, stats
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        dense_target = min(target_frames * 2, total)
        indices = np.linspace(0, total - 1, num=dense_target, dtype=np.int64)
        target_set = set(indices.tolist())
    else:
        target_set = set()
    similarities: list[float] = []
    best_frame = -1
    best_score = -1.0
    frame_idx = 0
    collected = 0
    dumped = 0
    dump_limit = _get_debug_max_frames()
    stride = 1
    while True:
        success, frame = capture.read()
        if not success:
            break
        should_take = False
        if target_set:
            should_take = frame_idx in target_set
        else:
            should_take = frame_idx % stride == 0
        if should_take:
            if debug_dir is not None and dumped < dump_limit:
                _dump_frame(debug_dir, frame, frame_idx)
                dumped += 1
            embedding, face_size, has_face = _extract_embedding(frame, min_face_size=min_face_size)
            stats["frames_seen"] += 1
            if has_face:
                stats["frames_with_faces"] += 1
                stats["max_face_size"] = max(stats["max_face_size"], face_size)
            if embedding is not None:
                score = float(np.dot(id_embedding, embedding))
                similarities.append(score)
                if score > best_score:
                    best_score = score
                    best_frame = frame_idx
            collected += 1
            if target_set and collected >= len(target_set):
                break
            if not target_set and collected >= target_frames:
                break
        frame_idx += 1
    capture.release()
    stats["backend"] = "opencv"
    return similarities, best_frame, stats


def _read_frames_decord(video_path: str, target_frames: int) -> Tuple[list[np.ndarray], list[int]]:
    if not _DECORD_AVAILABLE:
        return [], []
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    total = len(reader)
    if total <= 0:
        return [], []
    count = min(target_frames, total)
    indices = np.linspace(0, total - 1, num=count, dtype=np.int64).tolist()
    batch = reader.get_batch(indices).asnumpy()
    frames = []
    for frame in batch:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frames.append(bgr)
    return frames, indices


def _score_frames(
    frames: list[np.ndarray],
    indices: list[int],
    id_embedding: np.ndarray,
    min_face_size: int,
) -> Tuple[list[float], int, dict]:
    similarities: list[float] = []
    best_frame = -1
    best_score = -1.0
    stats = {
        "frames_seen": 0,
        "frames_with_faces": 0,
        "max_face_size": 0.0,
    }
    for idx, frame in zip(indices, frames):
        embedding, face_size, has_face = _extract_embedding(frame, min_face_size=min_face_size)
        stats["frames_seen"] += 1
        if has_face:
            stats["frames_with_faces"] += 1
            stats["max_face_size"] = max(stats["max_face_size"], face_size)
        if embedding is None:
            continue
        score = float(np.dot(id_embedding, embedding))
        similarities.append(score)
        if score > best_score:
            best_score = score
            best_frame = idx
    return similarities, best_frame, stats


def _get_debug_dir(video_path: str) -> Optional[Path]:
    if os.environ.get("ID_MATCH_DEBUG", "").lower() not in {"1", "true", "yes", "on"}:
        return None
    repo_root = Path(__file__).resolve().parents[2]
    default_root = repo_root / "userVisualization" / "storage" / "id_match_debug"
    base = Path(os.environ.get("ID_MATCH_DEBUG_DIR", str(default_root)))
    video_name = Path(video_path).stem or "video"
    run_id = uuid4().hex[:8]
    target = base / f"{video_name}_{run_id}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _get_debug_max_frames() -> int:
    raw = os.environ.get("ID_MATCH_DEBUG_MAX_FRAMES", "30")
    try:
        value = int(raw)
    except ValueError:
        return 30
    return max(1, min(value, 200))


def _maybe_dump_frames(
    frames: list[np.ndarray],
    indices: list[int],
    debug_dir: Optional[Path],
) -> None:
    if debug_dir is None:
        return
    dumped = 0
    limit = _get_debug_max_frames()
    for idx, frame in zip(indices, frames):
        if dumped >= limit:
            break
        _dump_frame(debug_dir, frame, idx)
        dumped += 1


def _dump_frame(debug_dir: Path, frame: np.ndarray, frame_idx: int) -> None:
    try:
        filename = debug_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(filename), frame)
    except Exception:
        return


def _topk_mean(values: list[float], k: int) -> float:
    if not values:
        return 0.0
    k = max(1, min(k, len(values)))
    sorted_vals = sorted(values, reverse=True)
    return float(np.mean(sorted_vals[:k]))


def _sigmoid(score: float, *, scale: float, bias: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(score * scale + bias))))
