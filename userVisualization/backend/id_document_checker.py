from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Optional

import cv2  # type: ignore
import numpy as np

LOGGER = logging.getLogger(__name__)


def analyze_id_document(
    image_path: str | Path,
    *,
    face_result: Mapping[str, object] | None = None,
) -> dict[str, object]:
    path = Path(image_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return _build_result(
            status="REUPLOAD",
            risk_level="High",
            risk_score=1.0,
            summary="The uploaded ID image could not be read.",
            user_message="Please re-upload the ID photo. The current file cannot be processed.",
            issues=[
                _issue(
                    code="read_error",
                    severity="high",
                    title="Image read failed",
                    message="The system could not decode the uploaded ID image.",
                )
            ],
            metrics={},
        )

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    metrics = {
        "width": int(width),
        "height": int(height),
        "aspect_ratio": float(width / max(height, 1)),
        "blur_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "contrast_std": float(gray.std()),
        "overexposed_ratio": float(np.mean(gray >= 245)),
        "underexposed_ratio": float(np.mean(gray <= 30)),
        "edge_density": float(np.mean(cv2.Canny(gray, 80, 160) > 0)),
    }
    metrics.update(_document_layout_metrics(gray))

    issues: list[dict[str, object]] = []
    _append_resolution_issue(issues, metrics)
    _append_blur_issue(issues, metrics)
    _append_exposure_issue(issues, metrics)
    _append_layout_issue(issues, metrics)
    _append_face_issue(issues, face_result)

    risk_score = _risk_score(issues)
    status = _status_from_issues(issues, risk_score)
    risk_level = {
        "PASS": "Low",
        "REVIEW": "Medium",
        "REUPLOAD": "High",
    }.get(status, "Medium")

    summary, user_message = _messages_from_status(status, issues)
    return _build_result(
        status=status,
        risk_level=risk_level,
        risk_score=risk_score,
        summary=summary,
        user_message=user_message,
        issues=issues,
        metrics=metrics,
    )


def _document_layout_metrics(gray: np.ndarray) -> dict[str, float | bool]:
    height, width = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = float(max(height * width, 1))
    best_area = 0.0
    best_found = False
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.1:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) not in {4, 5}:
            continue
        x, y, w, h = cv2.boundingRect(approx)
        coverage = float((w * h) / image_area)
        if coverage > best_area:
            best_area = coverage
            best_found = True

    return {
        "document_contour_found": bool(best_found),
        "document_coverage": float(best_area),
    }


def _append_resolution_issue(issues: list[dict[str, object]], metrics: Mapping[str, object]) -> None:
    width = _safe_float(metrics.get("width"))
    height = _safe_float(metrics.get("height"))
    if width < 720 or height < 420:
        issues.append(
            _issue(
                code="low_resolution",
                severity="high",
                title="Resolution too low",
                message="The ID photo is too small for reliable inspection. Re-upload a clearer, higher-resolution image.",
            )
        )
    elif width < 1000 or height < 580:
        issues.append(
            _issue(
                code="resolution_borderline",
                severity="medium",
                title="Resolution is borderline",
                message="The ID photo is usable, but a clearer image would improve document and portrait checks.",
            )
        )


def _append_blur_issue(issues: list[dict[str, object]], metrics: Mapping[str, object]) -> None:
    blur_variance = _safe_float(metrics.get("blur_variance"))
    if blur_variance < 70.0:
        issues.append(
            _issue(
                code="blur_high",
                severity="high",
                title="Image is blurry",
                message="The ID image is too blurry to inspect fine details. Re-upload a sharper photo.",
            )
        )
    elif blur_variance < 130.0:
        issues.append(
            _issue(
                code="blur_medium",
                severity="medium",
                title="Image sharpness is weak",
                message="The ID image is slightly blurry. Staff should be careful when trusting the result.",
            )
        )


def _append_exposure_issue(issues: list[dict[str, object]], metrics: Mapping[str, object]) -> None:
    overexposed_ratio = _safe_float(metrics.get("overexposed_ratio"))
    underexposed_ratio = _safe_float(metrics.get("underexposed_ratio"))
    contrast_std = _safe_float(metrics.get("contrast_std"))

    if overexposed_ratio > 0.18:
        issues.append(
            _issue(
                code="glare_high",
                severity="high",
                title="Strong glare or overexposure",
                message="Bright reflection hides part of the ID. Re-upload the image without glare.",
            )
        )
    elif overexposed_ratio > 0.10:
        issues.append(
            _issue(
                code="glare_medium",
                severity="medium",
                title="Partial glare detected",
                message="Some bright reflection is present on the ID image. Staff should verify covered regions manually.",
            )
        )

    if underexposed_ratio > 0.35:
        issues.append(
            _issue(
                code="dark_high",
                severity="high",
                title="Image is too dark",
                message="The ID photo is underexposed. Re-upload the image in better lighting.",
            )
        )
    elif underexposed_ratio > 0.18:
        issues.append(
            _issue(
                code="dark_medium",
                severity="medium",
                title="Lighting is weak",
                message="The ID image is darker than expected, which reduces inspection confidence.",
            )
        )

    if contrast_std < 32.0:
        issues.append(
            _issue(
                code="contrast_low",
                severity="medium",
                title="Document contrast is weak",
                message="The ID image has weak contrast, so text and document boundaries may be unclear.",
            )
        )


def _append_layout_issue(issues: list[dict[str, object]], metrics: Mapping[str, object]) -> None:
    aspect_ratio = _safe_float(metrics.get("aspect_ratio"))
    contour_found = bool(metrics.get("document_contour_found"))
    coverage = _safe_float(metrics.get("document_coverage"))
    edge_density = _safe_float(metrics.get("edge_density"))

    if aspect_ratio < 1.2 or aspect_ratio > 1.95:
        issues.append(
            _issue(
                code="aspect_unusual",
                severity="medium",
                title="Document framing looks unusual",
                message="The uploaded image does not look like a full ID card capture. Check whether the whole card is visible.",
            )
        )

    if not contour_found and edge_density < 0.02:
        issues.append(
            _issue(
                code="document_layout_unclear",
                severity="high",
                title="Document boundary is unclear",
                message="The system cannot confidently locate the ID card boundary. Re-upload a flat, full-card image.",
            )
        )
    elif contour_found and coverage < 0.45:
        issues.append(
            _issue(
                code="document_too_small",
                severity="medium",
                title="ID card occupies a small region",
                message="The ID card appears too small in the photo. Move closer and keep the full card in frame.",
            )
        )


def _append_face_issue(
    issues: list[dict[str, object]],
    face_result: Mapping[str, object] | None,
) -> None:
    if not isinstance(face_result, Mapping):
        return
    if bool(face_result.get("ok")):
        return
    error_code = str(face_result.get("error") or "").strip().lower()
    if error_code == "face_too_small":
        issues.append(
            _issue(
                code="portrait_small",
                severity="medium",
                title="Portrait region is too small",
                message="The portrait area on the ID is too small for stable face matching.",
            )
        )
    elif error_code == "no_face_detected":
        issues.append(
            _issue(
                code="portrait_unclear",
                severity="high",
                title="Portrait region is unclear",
                message="The system cannot detect a clear portrait on the ID. Re-upload a straighter and clearer ID photo.",
            )
        )


def _risk_score(issues: list[dict[str, object]]) -> float:
    score = 0.0
    for issue in issues:
        severity = str(issue.get("severity") or "").strip().lower()
        if severity == "high":
            score += 0.34
        elif severity == "medium":
            score += 0.17
    return float(max(0.0, min(score, 1.0)))


def _status_from_issues(issues: list[dict[str, object]], risk_score: float) -> str:
    has_high = any(str(issue.get("severity") or "").strip().lower() == "high" for issue in issues)
    if has_high or risk_score >= 0.55:
        return "REUPLOAD"
    if issues:
        return "REVIEW"
    return "PASS"


def _messages_from_status(status: str, issues: list[dict[str, object]]) -> tuple[str, str]:
    issue_titles = [str(item.get("title") or "").strip() for item in issues]
    top_issues = ", ".join([title for title in issue_titles if title][:2])

    if status == "PASS":
        return (
            "ID document quality looks acceptable. No obvious capture or integrity issue was detected.",
            "ID photo looks acceptable. You can continue with the current upload.",
        )
    if status == "REVIEW":
        suffix = f" Main concerns: {top_issues}." if top_issues else ""
        return (
            "ID document has some quality or integrity warnings. Staff should review the image carefully." + suffix,
            "This ID photo is usable, but a clearer re-upload is recommended to reduce review risk." + suffix,
        )
    suffix = f" Main concerns: {top_issues}." if top_issues else ""
    return (
        "ID document quality is not reliable enough for automated checking. A new upload is recommended." + suffix,
        "Please re-upload the ID photo with all four corners visible, better lighting, and sharper focus." + suffix,
    )


def _build_result(
    *,
    status: str,
    risk_level: str,
    risk_score: float,
    summary: str,
    user_message: str,
    issues: list[dict[str, object]],
    metrics: Mapping[str, object],
) -> dict[str, object]:
    return {
        "status": status,
        "risk_level": risk_level,
        "risk_score": float(max(0.0, min(risk_score, 1.0))),
        "needs_reupload": status == "REUPLOAD",
        "summary": summary,
        "user_message": user_message,
        "issues": issues,
        "metrics": dict(metrics),
    }


def _issue(
    *,
    code: str,
    severity: str,
    title: str,
    message: str,
) -> dict[str, object]:
    return {
        "code": code,
        "severity": severity,
        "title": title,
        "message": message,
    }


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0
