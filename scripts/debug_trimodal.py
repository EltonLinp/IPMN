#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from userVisualization.backend.model import get_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tri-modal inference with debug payload.")
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.4f}"


def build_summary(payload: dict[str, Any]) -> str:
    final = payload.get("final", {}) if isinstance(payload, dict) else {}
    audio = payload.get("audio", {}) if isinstance(payload, dict) else {}
    video = payload.get("video", {}) if isinstance(payload, dict) else {}
    sync = payload.get("sync", {}) if isinstance(payload, dict) else {}
    debug = payload.get("debug", {}) if isinstance(payload, dict) else {}
    gated = debug.get("gated_fusion", {}) if isinstance(debug, dict) else {}
    preprocess = debug.get("preprocess", {}) if isinstance(debug, dict) else {}
    sync_info = preprocess.get("sync", {}) if isinstance(preprocess, dict) else {}
    weights = gated.get("branch_weights", {}) if isinstance(gated, dict) else {}
    return (
        f"final_fake={_fmt(_safe_float(final.get('fake')))} "
        f"audio_fake={_fmt(_safe_float(audio.get('fake')))} "
        f"video_fake={_fmt(_safe_float(video.get('fake')))} "
        f"sync_fake={_fmt(_safe_float(sync.get('fake')))} "
        f"weights={weights if isinstance(weights, dict) else None} "
        f"sync_mismatch={sync_info.get('t_mismatch') if isinstance(sync_info, dict) else None}"
    )


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1

    service = get_service()
    result = service.analyze(video_path, debug=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(build_summary(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
