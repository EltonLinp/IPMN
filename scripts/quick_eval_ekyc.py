#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick eKYC contrast evaluation runner.")
    parser.add_argument("--data_dir", default="data/quick_eval", help="Input data root.")
    parser.add_argument("--out_dir", default="outputs", help="Output directory.")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000", help="Backend base URL.")
    parser.add_argument("--timeout", type=float, default=180.0, help="HTTP timeout seconds.")
    return parser.parse_args()


def find_people(data_dir: Path) -> Tuple[List[Dict[str, Path]], List[Tuple[str, str]]]:
    people: List[Dict[str, Path]] = []
    skipped: List[Tuple[str, str]] = []
    for person_dir in sorted(data_dir.glob("person_*")):
        if not person_dir.is_dir():
            continue
        id_path = person_dir / "id.jpg"
        selfie_path = person_dir / "selfie.jpg"
        video_path = person_dir / "video.mp4"
        missing = [name for name, path in [("id.jpg", id_path), ("selfie.jpg", selfie_path), ("video.mp4", video_path)] if not path.exists()]
        if missing:
            skipped.append((person_dir.name, f"missing {', '.join(missing)}"))
            continue
        people.append(
            {
                "person": person_dir.name,
                "id": id_path,
                "selfie": selfie_path,
                "video": video_path,
            }
        )
    return people, skipped


def build_samples(people: List[Dict[str, Path]]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    total = len(people)
    for idx, person in enumerate(people):
        samples.append(
            {
                "case_type": "pos",
                "id_person": person["person"],
                "selfie_person": person["person"],
                "video_person": person["person"],
                "id_path": person["id"],
                "selfie_path": person["selfie"],
                "video_path": person["video"],
            }
        )
        if total < 2:
            continue
        other = people[(idx + 1) % total]
        samples.append(
            {
                "case_type": "neg_id",
                "id_person": person["person"],
                "selfie_person": other["person"],
                "video_person": other["person"],
                "id_path": person["id"],
                "selfie_path": other["selfie"],
                "video_path": other["video"],
            }
        )
        samples.append(
            {
                "case_type": "neg_selfie",
                "id_person": other["person"],
                "selfie_person": person["person"],
                "video_person": other["person"],
                "id_path": other["id"],
                "selfie_path": person["selfie"],
                "video_path": other["video"],
            }
        )
        samples.append(
            {
                "case_type": "neg_video",
                "id_person": other["person"],
                "selfie_person": other["person"],
                "video_person": person["person"],
                "id_path": other["id"],
                "selfie_path": other["selfie"],
                "video_path": person["video"],
            }
        )
    return samples


def _guess_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def build_multipart(files: Dict[str, Path]) -> Tuple[str, bytes]:
    boundary = uuid.uuid4().hex
    parts: List[bytes] = []
    for field, path in files.items():
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n'
            f"Content-Type: {_guess_type(path)}\r\n\r\n"
        )
        parts.append(header.encode("utf-8"))
        parts.append(path.read_bytes())
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return boundary, b"".join(parts)


def post_eval(
    base_url: str,
    *,
    id_path: Path,
    selfie_path: Path,
    video_path: Path,
    timeout: float,
) -> Tuple[Optional[int], Optional[Dict[str, Any]], Optional[str]]:
    url = base_url.rstrip("/") + "/api/ekyc/evaluate"
    boundary, body = build_multipart(
        {"id_image": id_path, "selfie_image": selfie_path, "video": video_path}
    )
    req = request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Accept", "application/json")
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            raw = resp.read()
            payload = json.loads(raw.decode("utf-8")) if raw else None
            return status, payload, None
    except error.HTTPError as exc:
        raw = exc.read()
        payload = None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            payload = None
        return exc.code, payload, f"HTTPError: {exc}"
    except Exception as exc:
        return None, None, f"Request failed: {exc}"


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_row(
    sample: Dict[str, Any],
    status: Optional[int],
    payload: Optional[Dict[str, Any]],
    err: Optional[str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "case_type": sample["case_type"],
        "id_person": sample["id_person"],
        "selfie_person": sample["selfie_person"],
        "video_person": sample["video_person"],
        "deepfake_score": None,
        "deepfake_label": None,
        "id_selfie_score": None,
        "id_selfie_ok": None,
        "id_video_score": None,
        "id_video_ok": None,
        "risk": None,
        "decision": None,
        "reasons": None,
        "id_face_path": None,
        "selfie_face_path": None,
        "video_best_frame_path": None,
        "error_code": None,
        "error_message": None,
        "http_status": status,
    }

    if payload is None:
        row["error_code"] = "request_failed"
        row["error_message"] = err or "no_response"
        return row

    if "error_code" in payload:
        row["error_code"] = payload.get("error_code")
        row["error_message"] = payload.get("message")
        return row

    deepfake = payload.get("deepfake", {}) if isinstance(payload, dict) else {}
    row["deepfake_score"] = safe_float(deepfake.get("score"))
    row["deepfake_label"] = deepfake.get("label")

    match = payload.get("match", {}) if isinstance(payload, dict) else {}
    id_selfie = match.get("id_selfie", {}) if isinstance(match, dict) else {}
    id_video = match.get("id_video", {}) if isinstance(match, dict) else {}
    row["id_selfie_score"] = safe_float(id_selfie.get("score"))
    row["id_selfie_ok"] = bool(id_selfie.get("ok")) if isinstance(id_selfie, dict) else None
    row["id_video_score"] = safe_float(id_video.get("score"))
    row["id_video_ok"] = bool(id_video.get("ok")) if isinstance(id_video, dict) else None

    fusion = payload.get("fusion", {}) if isinstance(payload, dict) else {}
    row["risk"] = safe_float(fusion.get("risk"))
    row["decision"] = fusion.get("decision")
    reasons = fusion.get("reason")
    if isinstance(reasons, list):
        row["reasons"] = ";".join(str(item) for item in reasons)

    artifacts = payload.get("artifacts", {}) if isinstance(payload, dict) else {}
    row["id_face_path"] = artifacts.get("id_face_path")
    row["selfie_face_path"] = artifacts.get("selfie_face_path")
    row["video_best_frame_path"] = artifacts.get("video_best_frame_path")

    return row


def print_summary(rows: List[Dict[str, Any]]) -> None:
    case_counts = Counter(row.get("case_type") or "unknown" for row in rows)
    decision_counts = Counter(row.get("decision") or "N/A" for row in rows)
    groups: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        bucket = "pos" if row.get("case_type") == "pos" else "neg"
        for key in ("id_selfie_score", "id_video_score", "risk"):
            value = row.get(key)
            if isinstance(value, (int, float)):
                groups[bucket][key].append(float(value))

    def mean(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    print("\nSummary:")
    print("Case counts:", dict(case_counts))
    print("Decision counts:", dict(decision_counts))
    for bucket in ("pos", "neg"):
        print(f"{bucket} means:")
        for key in ("id_selfie_score", "id_video_score", "risk"):
            avg = mean(groups[bucket][key])
            if avg is None:
                print(f"  {key}: N/A")
            else:
                print(f"  {key}: {avg:.4f}")


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = out_dir / "quick_eval_results.csv"

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return 1

    people, skipped = find_people(data_dir)
    if skipped:
        print("Skipped entries:")
        for name, reason in skipped:
            print(f"  {name}: {reason}")

    if not people:
        print("No valid person_* directories found.")
        return 1

    samples = build_samples(people)
    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        start = time.time()
        status, payload, err = post_eval(
            args.base_url,
            id_path=sample["id_path"],
            selfie_path=sample["selfie_path"],
            video_path=sample["video_path"],
            timeout=args.timeout,
        )
        row = build_row(sample, status, payload, err)
        rows.append(row)
        elapsed = time.time() - start
        status_label = status if status is not None else "ERR"
        print(f"[{idx}/{len(samples)}] {sample['case_type']} -> {status_label} ({elapsed:.2f}s)")

    fieldnames = [
        "case_type",
        "id_person",
        "selfie_person",
        "video_person",
        "deepfake_score",
        "deepfake_label",
        "id_selfie_score",
        "id_selfie_ok",
        "id_video_score",
        "id_video_ok",
        "risk",
        "decision",
        "reasons",
        "id_face_path",
        "selfie_face_path",
        "video_best_frame_path",
        "error_code",
        "error_message",
        "http_status",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote CSV: {output_csv}")
    print("\nCSV preview (first 5 lines):")
    with output_csv.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= 5:
                break
            print(line.rstrip("\n"))

    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
