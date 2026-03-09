#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from userVisualization.backend.model import get_service


DEFAULT_ALPHAS: list[float] = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep sync_uncertainty_alpha on one real + one fake sample.")
    parser.add_argument("--real", required=True, help="Path to real video.")
    parser.add_argument("--fake", required=True, help="Path to fake video.")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
        help="Alpha values to sweep (space-separated).",
    )
    parser.add_argument("--out", default=None, help="Optional output JSON path.")
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
        return "N/A"
    return f"{value:.4f}"


def _fmt_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def main() -> int:
    args = parse_args()
    real_path = Path(args.real).expanduser().resolve()
    fake_path = Path(args.fake).expanduser().resolve()
    if not real_path.exists():
        print(f"Real video not found: {real_path}")
        return 1
    if not fake_path.exists():
        print(f"Fake video not found: {fake_path}")
        return 1

    alphas = [max(0.0, min(1.0, float(a))) for a in args.alphas]
    service = get_service()
    total = len(alphas)
    started_at = perf_counter()

    rows: list[dict[str, Any]] = []
    tqdm.write("alpha\treal_final\tfake_final\treal_sync\tfake_sync")
    for idx, alpha in enumerate(tqdm(alphas, desc="alpha sweep", unit="alpha"), start=1):
        real_res = service.analyze(
            real_path,
            debug=True,
            sync_uncertainty_alpha=alpha,
            sync_mismatch_penalty=1.0,
        )
        fake_res = service.analyze(
            fake_path,
            debug=True,
            sync_uncertainty_alpha=alpha,
            sync_mismatch_penalty=1.0,
        )
        real_final = _safe_float(((real_res.get("final", {}) or {}).get("fake")))
        fake_final = _safe_float(((fake_res.get("final", {}) or {}).get("fake")))
        real_sync = _safe_float(((real_res.get("sync", {}) or {}).get("fake")))
        fake_sync = _safe_float(((fake_res.get("sync", {}) or {}).get("fake")))
        row = {
            "alpha": alpha,
            "real_final": real_final,
            "fake_final": fake_final,
            "real_sync": real_sync,
            "fake_sync": fake_sync,
            "real": real_res,
            "fake": fake_res,
        }
        rows.append(row)
        tqdm.write(f"{alpha:.2f}\t{_fmt(real_final)}\t{_fmt(fake_final)}\t{_fmt(real_sync)}\t{_fmt(fake_sync)}")
        elapsed = perf_counter() - started_at
        avg = elapsed / float(idx)
        eta = max(avg * float(total - idx), 0.0)
        print(f"[{idx}/{total}] elapsed={_fmt_seconds(elapsed)} ETA={_fmt_seconds(eta)}", flush=True)

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_payload = {
            "real_video": str(real_path),
            "fake_video": str(fake_path),
            "rows": rows,
        }
        out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved sweep JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
