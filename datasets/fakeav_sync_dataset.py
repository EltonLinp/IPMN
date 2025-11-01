from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .fakeav_video_dataset import DatasetSplit, _read_index


@dataclass
class SyncDatasetConfig:
    negative_prob: float = 0.5
    target_frames: Optional[int] = None  # If set, clip/pad frames to this length


class FakeAVSyncDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, int, int]]):
    """
    Dataset returning (video_frames, audio_sequence, sync_label, deepfake_label).

    - video_frames: [T, 3, 224, 224] in range [-1, 1]
    - audio_sequence: [T, 64] after temporal resampling of Mel features
    - sync_label: 1 for aligned audio/video, 0 for misaligned audio (roll)
    - deepfake_label: original sample label (0 real, 1 fake)
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        index_file: str | Path | None = None,
        split: str | None = None,
        split_scheme: DatasetSplit | None = None,
        seed: int = 1337,
        config: SyncDatasetConfig | None = None,
        return_metadata: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found: {self.data_dir}")

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

        self.split_scheme = split_scheme or DatasetSplit()
        self.seed = seed
        self.config = config or SyncDatasetConfig()
        self.return_metadata = return_metadata
        self.files = self._apply_split(files, split)

    def _apply_split(self, files: Sequence[Path], split: Optional[str]) -> List[Path]:
        if split not in {None, "train", "val", "test"}:
            raise ValueError("split must be one of train/val/test or None.")
        grouped: Dict[str, List[Path]] = {}
        for path in files:
            speaker = path.stem.split("_")[0]
            grouped.setdefault(speaker, []).append(path)
        speakers = list(grouped.keys())
        rng = random.Random(self.seed)
        rng.shuffle(speakers)
        train_cutoff, val_cutoff = self.split_scheme.as_cumulative()
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
            selected: List[Path] = []
            for speaker in speakers:
                selected.extend(grouped[speaker])
            return selected
        selected: List[Path] = []
        for speaker in split_map[split]:
            selected.extend(grouped[speaker])
        return selected

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _load_bundle(path: Path) -> Dict[str, object]:
        return torch.load(path, map_location="cpu")

    def _maybe_trim_frames(self, video: torch.Tensor) -> torch.Tensor:
        target = self.config.target_frames
        if target is None or video.shape[0] == target:
            return video
        if video.shape[0] > target:
            start = max((video.shape[0] - target) // 2, 0)
            return video[start : start + target]
        pad = target - video.shape[0]
        pad_before = pad // 2
        pad_after = pad - pad_before
        pad_tensor = F.pad(
            video.permute(1, 2, 3, 0),
            (pad_before, pad_after),
            mode="replicate",
        ).permute(3, 0, 1, 2)
        return pad_tensor

    def __getitem__(self, index: int):
        path = self.files[index]
        bundle = self._load_bundle(path)
        video: torch.Tensor = bundle["video"]  # [T, 3, 224, 224]
        video = video.float()
        video = self._maybe_trim_frames(video)

        audio = bundle.get("audio", {})
        mel: Optional[torch.Tensor] = None
        if isinstance(audio, dict):
            mel = audio.get("mel")
        if mel is None:
            raise RuntimeError(f"No mel spectrogram found in {path}")
        if mel.dim() == 3:
            mel = mel.squeeze(0)
        mel = mel.float()  # [64, S]

        sync = bundle.get("sync", {})
        frame_count = video.shape[0]
        mel_steps = mel.shape[-1]
        if mel_steps <= 1:
            mel_seq = mel.permute(1, 0).repeat(frame_count, 1)
        else:
            mel_seq = F.interpolate(
                mel.unsqueeze(0),
                size=frame_count,
                mode="linear",
                align_corners=False,
            ).squeeze(0).permute(1, 0)  # [T, 64]

        rng = random.Random(self.seed + index)
        mismatch = False
        if frame_count > 1 and self.config.negative_prob > 0:
            mismatch = rng.random() < self.config.negative_prob
        if mismatch:
            shift = rng.randint(1, frame_count - 1)
            mel_seq = mel_seq.roll(shifts=shift, dims=0)
            sync_label = 0
        else:
            sync_label = 1

        deepfake_label = int(bundle.get("label", 0))
        if self.return_metadata:
            metadata = {
                "path": str(path),
                "sync": sync,
            }
            return video, mel_seq, sync_label, deepfake_label, metadata
        return video, mel_seq, sync_label, deepfake_label
