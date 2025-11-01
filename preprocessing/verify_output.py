from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import torch


def describe_sample(sample_path: Path) -> dict:
    bundle = torch.load(sample_path, map_location="cpu")
    video = bundle.get("video")
    audio = bundle.get("audio", {})
    sync = bundle.get("sync", {})

    info = {
        "file": str(sample_path),
        "speaker_id": bundle.get("speaker_id"),
        "label": bundle.get("label"),
        "type": bundle.get("type"),
        "video_shape": list(video.shape) if torch.is_tensor(video) else None,
        "mel_shape": list(audio.get("mel").shape) if isinstance(audio.get("mel"), torch.Tensor) else None,
        "waveform_shape": (
            list(audio.get("waveform").shape) if isinstance(audio.get("waveform"), torch.Tensor) else None
        ),
        "sync_keys": list(sync.keys()) if isinstance(sync, dict) else [],
    }
    if isinstance(sync, dict):
        info["frame_count"] = int(sync.get("frame_indices", torch.tensor([])).numel()) if "frame_indices" in sync else None
        info["mel_steps"] = int(audio.get("mel").shape[-1]) if isinstance(audio.get("mel"), torch.Tensor) else None
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect preprocessed FakeAVCeleb samples.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing *.pt files.")
    parser.add_argument("--limit", type=int, default=3, help="Number of samples to summarise.")
    parser.add_argument("--seed", type=int, default=2024, help="Seed for random sampling.")
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    files: List[Path] = sorted(args.input_dir.glob("*.pt"))
    if not files:
        raise RuntimeError(f"No .pt files found in {args.input_dir}")

    random.seed(args.seed)
    chosen = files if args.limit >= len(files) else random.sample(files, args.limit)

    summaries = [describe_sample(path) for path in chosen]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

