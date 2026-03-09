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
    parser = argparse.ArgumentParser(description="Run tri-modal branch ablation diagnostics.")
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
        return "N/A"
    return f"{value:.4f}"


def _branch_fake_text(run: dict[str, Any], branch: str) -> str:
    debug = run.get("debug", {}) if isinstance(run, dict) else {}
    ablation = debug.get("ablation", {}) if isinstance(debug, dict) else {}
    requested = ablation.get("requested", {}) if isinstance(ablation, dict) else {}
    ignored = set(ablation.get("ignored_branches", [])) if isinstance(ablation, dict) else set()
    missing = set(ablation.get("missing_branches", [])) if isinstance(ablation, dict) else set()
    if bool(requested.get(branch)) or branch in ignored:
        return "N/A(ablate)"
    if branch in missing:
        return "N/A(missing)"
    branch_payload = run.get(branch, {}) if isinstance(run, dict) else {}
    if not isinstance(branch_payload, dict):
        return "N/A(missing)"
    return _fmt(_safe_float(branch_payload.get("fake")))


def _weight_triplet(run: dict[str, Any], key: str) -> str:
    debug = run.get("debug", {}) if isinstance(run, dict) else {}
    gated = debug.get("gated_fusion", {}) if isinstance(debug, dict) else {}
    weights = gated.get(key, {}) if isinstance(gated, dict) else {}
    if not isinstance(weights, dict):
        return "a=N/A,v=N/A,s=N/A"
    audio = _safe_float(weights.get("audio"))
    video = _safe_float(weights.get("video"))
    sync = _safe_float(weights.get("sync"))
    return f"a={_fmt(audio)},v={_fmt(video)},s={_fmt(sync)}"


def _gate_triplet(run: dict[str, Any]) -> str:
    debug = run.get("debug", {}) if isinstance(run, dict) else {}
    gated = debug.get("gated_fusion", {}) if isinstance(debug, dict) else {}
    gate = gated.get("model_gate", {}) if isinstance(gated, dict) else {}
    if not isinstance(gate, dict):
        return "a=N/A,v=N/A,s=N/A"
    audio = _safe_float(gate.get("audio"))
    video = _safe_float(gate.get("video"))
    sync = _safe_float(gate.get("sync"))
    return f"a={_fmt(audio)},v={_fmt(video)},s={_fmt(sync)}"


def _sync_mismatch(run: dict[str, Any]) -> Any:
    debug = run.get("debug", {}) if isinstance(run, dict) else {}
    preprocess = debug.get("preprocess", {}) if isinstance(debug, dict) else {}
    sync = preprocess.get("sync", {}) if isinstance(preprocess, dict) else {}
    if not isinstance(sync, dict):
        return None
    return sync.get("t_mismatch")


def print_summary(runs: dict[str, dict[str, Any]]) -> None:
    print("name      final_fake    audio_fake      video_fake      sync_fake       eff_w(a,v,s)                    gate(a,v,s)                     sync_mismatch")
    for name in ("baseline", "with_sync_clamp", "no_sync", "no_video", "no_audio"):
        run = runs.get(name, {})
        final_fake = _safe_float((run.get("final", {}) or {}).get("fake") if isinstance(run, dict) else None)
        audio_fake = _branch_fake_text(run, "audio")
        video_fake = _branch_fake_text(run, "video")
        sync_fake = _branch_fake_text(run, "sync")
        eff = _weight_triplet(run, "effective_weights_norm")
        gate = _gate_triplet(run)
        mismatch = _sync_mismatch(run)
        print(
            f"{name:<9} {_fmt(final_fake):<12} {audio_fake:<14} {video_fake:<14} {sync_fake:<14} {eff:<32} {gate:<32} {str(mismatch)}"
        )
    base = _safe_float(((runs.get("baseline", {}) or {}).get("final", {}) or {}).get("fake"))
    clamp = _safe_float(((runs.get("with_sync_clamp", {}) or {}).get("final", {}) or {}).get("fake"))
    if base is not None and clamp is not None:
        print(f"\nbaseline_vs_with_sync_clamp: {base:.4f} -> {clamp:.4f} (delta={clamp - base:+.4f})")


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return 1

    service = get_service()
    configs: list[tuple[str, dict[str, object]]] = [
        ("baseline", {"sync_uncertainty_alpha": 1.0, "sync_mismatch_penalty": 1.0}),
        ("with_sync_clamp", {"sync_mismatch_penalty": 1.0}),
        ("no_sync", {"ablate_sync": True, "sync_uncertainty_alpha": 1.0, "sync_mismatch_penalty": 1.0}),
        ("no_video", {"ablate_video": True, "sync_uncertainty_alpha": 1.0, "sync_mismatch_penalty": 1.0}),
        ("no_audio", {"ablate_audio": True, "sync_uncertainty_alpha": 1.0, "sync_mismatch_penalty": 1.0}),
    ]
    runs: dict[str, dict[str, Any]] = {}
    for name, cfg in configs:
        runs[name] = service.analyze(video_path, debug=True, **cfg)

    payload = {
        "video": str(video_path),
        "runs": runs,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
