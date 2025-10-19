"""
Legacy helper: split FaceForensics++ dataset into train/validation manifests.

TrustFusion now standardises on Celeb-DF v2 (see tools/generate_celebdf_split.py).
This script is kept for archival compatibility only.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable, List, Tuple

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _iter_video_dirs(base: Path) -> Iterable[Tuple[Path, str]]:
    """Yield (directory, method_name) pairs for directories containing videos."""
    if not base.exists():
        return []
    candidates: List[Tuple[Path, str]] = []
    # Look for .../cXX/videos structure
    for depth1 in base.iterdir():
        if not depth1.is_dir():
            continue
        # e.g. actors/, youtube/, DeepFakeDetection/, etc.
        for comp_dir in depth1.glob("c*/videos"):
            if comp_dir.is_dir():
                candidates.append((comp_dir, depth1.name))
    # Fallback: directly check immediate videos folder
    if not candidates:
        for depth1 in base.rglob("videos"):
            if depth1.is_dir():
                candidates.append((depth1, depth1.parent.name))
    return candidates


def collect_original_samples(root: Path) -> List[Tuple[Path, int, str]]:
    collected: List[Tuple[Path, int, str]] = []
    base = root / "original_sequences"
    for video_dir, subgroup in _iter_video_dirs(base):
        for file_path in video_dir.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTS:
                collected.append((file_path.resolve(), 0, f"original-{subgroup}"))
    return collected


def collect_fake_samples(root: Path) -> List[Tuple[Path, int, str]]:
    collected: List[Tuple[Path, int, str]] = []
    manip_root = root / "manipulated_sequences"
    if not manip_root.exists():
        return collected
    for method_dir in manip_root.iterdir():
        if not method_dir.is_dir():
            continue
        for video_dir, subgroup in _iter_video_dirs(method_dir):
            for file_path in video_dir.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTS:
                    collected.append((file_path.resolve(), 1, f"{method_dir.name}-{subgroup}"))
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description="Split FaceForensics++ dataset into train/val manifests.")
    parser.add_argument("--root", type=Path, required=True, help="Root directory of FaceForensics++ dataset.")
    parser.add_argument("--out", type=Path, default=Path("manifests"), help="Output directory for manifests.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    root = args.root.resolve()
    samples: List[Tuple[Path, int, str]] = []
    samples.extend(collect_original_samples(root))
    samples.extend(collect_fake_samples(root))

    if not samples:
        raise RuntimeError("未在指定目录中找到任何视频文件，请检查路径。")

    random.seed(args.seed)
    random.shuffle(samples)

    split_idx = int(len(samples) * args.train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    args.out.mkdir(parents=True, exist_ok=True)

    def write_manifest(manifest_path: Path, data: List[Tuple[Path, int, str]]) -> None:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["video_path", "label", "method"])
            writer.writeheader()
            for path, label, method in data:
                # store relative path for portability
                try:
                    rel_path = path.relative_to(root)
                except ValueError:
                    rel_path = path
                writer.writerow(
                    {
                        "video_path": str(rel_path),
                        "label": label,
                        "method": method,
                    }
                )

    write_manifest(args.out / "train_manifest.csv", train_samples)
    write_manifest(args.out / "val_manifest.csv", val_samples)

    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")
    print(f"Manifests written to {args.out.resolve()}")


if __name__ == "__main__":
    main()
