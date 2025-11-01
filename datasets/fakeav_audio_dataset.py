from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .fakeav_video_dataset import DatasetSplit, _read_index


@dataclass
class AudioDatasetConfig:
    target_steps: int = 400
    random_crop: bool = False


class FakeAVAudioDataset(Dataset[Tuple[torch.Tensor, int]]):
    """
    Loads Mel-spectrogram tensors from preprocessed FakeAVCeleb samples.

    Each sample returns a tensor of shape [1, 64, target_steps].
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        index_file: str | Path | None = None,
        split: str | None = None,
        split_scheme: DatasetSplit | None = None,
        seed: int = 1337,
        config: AudioDatasetConfig | None = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        return_metadata: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found: {self.data_dir}")
        self.transform = transform
        self.return_metadata = return_metadata
        self.config = config or AudioDatasetConfig()
        self.seed = seed
        self._rng = random.Random(seed)

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

    def _pad_crop(self, mel: torch.Tensor) -> torch.Tensor:
        target = self.config.target_steps
        length = mel.shape[-1]
        if length == target:
            return mel
        if length > target:
            if self.config.random_crop:
                start = self._rng.randint(0, length - target)
            else:
                start = max((length - target) // 2, 0)
            return mel[..., start : start + target]
        pad_total = target - length
        left = pad_total // 2
        right = pad_total - left
        return F.pad(mel, (left, right))

    def __getitem__(self, index: int):
        path = self.files[index]
        bundle = torch.load(path, map_location="cpu")
        audio = bundle.get("audio", {})
        mel: Optional[torch.Tensor] = audio.get("mel") if isinstance(audio, dict) else None
        if mel is None:
            raise RuntimeError(f"No mel spectrogram found in {path}")
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        mel = mel.float().clone()
        mel = self._pad_crop(mel)
        if self.transform is not None:
            mel = self.transform(mel)
        label = int(bundle.get("label", 0))
        if self.return_metadata:
            metadata = {
                "path": str(path),
                "speaker_id": bundle.get("speaker_id"),
                "type": bundle.get("type"),
            }
            return mel, label, metadata
        return mel, label

