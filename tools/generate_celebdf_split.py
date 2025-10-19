"""
Generate stratified 80/20 train/validation manifests for Celeb-DF v2.

Usage:
    python tools/generate_celebdf_split.py
"""

from __future__ import annotations

import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_ROOT = Path(r"E:\CUHK\Industrial_Project\Celeb-DF-v2")
MANIFEST_DIR = Path("manifests")

LABEL_MAP = {
    "Celeb-synthesis": "fake",
    "Celeb-real": "real",
    "YouTube-real": "real",
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
ID_PATTERN = re.compile(r"(?i)id(\d+)")
TRAIN_RATIO = 0.8
SEED = 42


def extract_subject(file_name: str) -> str:
    match = ID_PATTERN.search(file_name)
    if match:
        return f"id{match.group(1)}"
    digest = hashlib.sha1(file_name.encode("utf-8")).hexdigest()[:10]
    return f"hash_{digest}"


def collect_entries() -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for subdir_name, label in LABEL_MAP.items():
        subdir_path = DATA_ROOT / subdir_name
        if not subdir_path.exists():
            continue
        for path in subdir_path.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in VIDEO_EXTS:
                continue
            subject = extract_subject(path.name)
            rel_path = path.relative_to(DATA_ROOT)
            entries.append(
                {
                    "video_path": rel_path.as_posix(),
                    "label": label,
                    "subject": subject,
                }
            )
    if not entries:
        raise RuntimeError(f"No video files found under {DATA_ROOT}.")
    return entries


def stratified_split(entries: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["label"]].append(entry)

    rng = random.Random(SEED)
    train_entries: list[dict[str, str]] = []
    val_entries: list[dict[str, str]] = []

    for label, items in grouped.items():
        if not items:
            continue
        rng.shuffle(items)
        split_idx = int(round(len(items) * TRAIN_RATIO))
        split_idx = max(1, min(split_idx, len(items) - 1))
        train_items = items[:split_idx]
        val_items = items[split_idx:]
        for source, split_name, bucket in ((train_items, "train", train_entries), (val_items, "val", val_entries)):
            for entry in source:
                sample = dict(entry)
                sample["split"] = split_name
                bucket.append(sample)

    rng.shuffle(train_entries)
    rng.shuffle(val_entries)
    return train_entries, val_entries


def write_manifests(train_entries: list[dict[str, str]], val_entries: list[dict[str, str]]) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    fields_all = ["video_path", "label", "subject", "split"]
    combined = train_entries + val_entries
    with (MANIFEST_DIR / "celebdf_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields_all)
        writer.writeheader()
        writer.writerows(combined)

    fields_subset = ["video_path", "label", "subject"]
    with (MANIFEST_DIR / "train.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields_subset)
        writer.writeheader()
        writer.writerows({k: e[k] for k in fields_subset} for e in train_entries)

    with (MANIFEST_DIR / "val.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields_subset)
        writer.writeheader()
        writer.writerows({k: e[k] for k in fields_subset} for e in val_entries)


def main() -> None:
    entries = collect_entries()
    train_entries, val_entries = stratified_split(entries)

    counts = Counter((entry["split"], entry["label"]) for entry in train_entries + val_entries)
    print("Split label counts:")
    for (split, label), count in sorted(counts.items()):
        print(f"  {split:5s} {label:4s}: {count}")

    write_manifests(train_entries, val_entries)

    print(f"Wrote manifests under {MANIFEST_DIR.resolve()}")
    print(f"  train samples: {len(train_entries)}")
    print(f"  val samples  : {len(val_entries)}")


if __name__ == "__main__":
    main()
