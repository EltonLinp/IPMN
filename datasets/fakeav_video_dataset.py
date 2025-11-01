from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch.nn.functional as F

import torch
from torch.utils.data import Dataset


def _read_index(index_path: Path) -> List[dict]:
    records: List[dict] = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(record)
    return records


@dataclass
class DatasetSplit:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    def as_cumulative(self) -> Tuple[float, float]:
        train = self.train_ratio
        val = self.val_ratio
        assert 0.0 < train < 1.0 and 0.0 <= val < 1.0
        total = train + val + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("DatasetSplit ratios must sum to 1.0")
        return train, train + val


class FakeAVVideoDataset(Dataset[Tuple[torch.Tensor, int]]):
    """
    Dataset wrapper that loads the video modality from preprocessed FakeAVCeleb samples.

    Each sample is expected to be a ``torch.save`` bundle containing the keys:
    ``video`` (frames in [T, C, H, W]) and ``label`` (0 for real, 1 for fake).
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        index_file: str | Path | None = None,
        split: str | None = None,
        split_scheme: DatasetSplit | None = None,
        seed: int = 1337,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_metadata: bool = False,
        target_frames: Optional[int] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found: {self.data_dir}")
        self.transform = transform
        self.return_metadata = return_metadata
        self.seed = seed
        self._rng = random.Random(seed)
        self.target_frames = target_frames

        index_path = Path(index_file) if index_file else self.data_dir / "preprocess_index.jsonl"
        if index_path.exists():
            records = _read_index(index_path)
            files = [
                self.data_dir / Path(record["output_path"]).name
                for record in records
                if record.get("status") == "ok"
            ]
        else:
            files = sorted(self.data_dir.glob("*.pt"))

        if not files:
            raise RuntimeError(f"No processed .pt samples found in {self.data_dir}")

        self.files = self._apply_split(files, split, split_scheme or DatasetSplit())

    def _apply_split(
        self,
        files: Sequence[Path],
        split: Optional[str],
        scheme: DatasetSplit,
    ) -> List[Path]:
        if split not in {None, "train", "val", "test"}:
            raise ValueError("split must be one of train/val/test or None.")
        grouped = {}
        for path in files:
            speaker = path.stem.split("_")[0]
            grouped.setdefault(speaker, []).append(path)
        speakers = list(grouped.keys())
        rng = random.Random(self.seed)
        rng.shuffle(speakers)
        train_cutoff, val_cutoff = scheme.as_cumulative()
        n = len(speakers)
        if n == 0:
            return []
        train_end = max(int(train_cutoff * n), 1)
        val_end = max(int(val_cutoff * n), train_end + 1) if n > 1 else n
        train_end = min(train_end, n)
        val_end = min(max(val_end, train_end), n)
        if val_end == n and val_end == train_end and n > 1:
            val_end = n - 1
        split_map = {
            "train": set(speakers[:train_end]),
            "val": set(speakers[train_end:val_end]),
            "test": set(speakers[val_end:]),
        }
        if split is None:
            selected = []
            for speaker in speakers:
                selected.extend(grouped[speaker])
            return selected
        selected: List[Path] = []
        for speaker in split_map[split]:
            selected.extend(grouped[speaker])
        return selected

    def __len__(self) -> int:
        return len(self.files)

    def _pad_crop(self, video: torch.Tensor) -> torch.Tensor:
        if self.target_frames is None or video.shape[0] == self.target_frames:
            return video
        frames = video.shape[0]
        target = self.target_frames
        if frames > target:
            start = (frames - target) // 2
            return video[start : start + target]
        pad = target - frames
        pad_before = pad // 2
        pad_after = pad - pad_before
        pieces = []
        if pad_before > 0:
            pieces.append(video[:1].repeat(pad_before, 1, 1, 1))
        pieces.append(video)
        if pad_after > 0:
            pieces.append(video[-1:].repeat(pad_after, 1, 1, 1))
        return torch.cat(pieces, dim=0)

    def __getitem__(self, index: int):
        path = self.files[index]
        bundle = torch.load(path, map_location="cpu")
        video: torch.Tensor = bundle["video"]
        if video.dim() != 4:
            raise RuntimeError(f"Unexpected video shape in {path}: {video.shape}")
        video = self._pad_crop(video)
        # Convert from [T, C, H, W] -> [C, T, H, W] for 3D CNN compatibility.
        video = video.permute(1, 0, 2, 3).contiguous().float().clone()
        if self.transform is not None:
            video = self.transform(video)
        label = int(bundle.get("label", 0))
        if self.return_metadata:
            metadata = {
                "path": str(path),
                "speaker_id": bundle.get("speaker_id"),
                "type": bundle.get("type"),
            }
            return video, label, metadata
        return video, label
