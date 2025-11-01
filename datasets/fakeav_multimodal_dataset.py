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
class MultimodalDatasetConfig:
    target_frames: Optional[int] = 32
    target_steps: Optional[int] = 400
    random_crop: bool = True


class FakeAVMultimodalDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    Joint loader returning video frames, audio Mel features, sync meta, and labels.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        index_file: str | Path | None = None,
        split: str | None = None,
        split_scheme: DatasetSplit | None = None,
        seed: int = 1337,
        config: MultimodalDatasetConfig | None = None,
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
        self.config = config or MultimodalDatasetConfig()
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

    def _pad_crop_video(self, video: torch.Tensor) -> torch.Tensor:
        target = self.config.target_frames
        if target is None or video.shape[0] == target:
            return video
        frames = video.shape[0]
        if frames > target:
            if self.config.random_crop:
                start = random.randint(0, frames - target)
            else:
                start = max((frames - target) // 2, 0)
            return video[start : start + target]
        pad = target - frames
        pad_before = pad // 2
        pad_after = pad - pad_before
        padded = F.pad(
            video.permute(1, 2, 3, 0),
            (pad_before, pad_after),
            mode="replicate",
        ).permute(3, 0, 1, 2)
        return padded

    def _pad_crop_audio(self, mel: torch.Tensor) -> torch.Tensor:
        target = self.config.target_steps
        if target is None or mel.shape[-1] == target:
            return mel
        steps = mel.shape[-1]
        if steps > target:
            if self.config.random_crop:
                start = random.randint(0, steps - target)
            else:
                start = max((steps - target) // 2, 0)
            return mel[..., start : start + target]
        pad = target - steps
        left = pad // 2
        right = pad - left
        return F.pad(mel, (left, right))

    def __getitem__(self, index: int):
        path = self.files[index]
        bundle = torch.load(path, map_location="cpu")

        video = bundle["video"].float()  # [T, C, H, W]
        video = self._pad_crop_video(video)
        video = video.permute(1, 0, 2, 3).contiguous()

        audio = bundle.get("audio", {})
        mel = None
        if isinstance(audio, dict):
            mel = audio.get("mel")
        if mel is None:
            raise RuntimeError(f"No mel spectrogram found in {path}")
        if mel.dim() == 3:
            mel = mel.squeeze(0)
        mel = mel.float().clone()  # [64, S]
        mel = self._pad_crop_audio(mel)
        mel_seq = mel.permute(1, 0).contiguous()  # [T, 64]

        sync = bundle.get("sync", {})
        frame_indices = torch.as_tensor(sync.get("frame_indices", []), dtype=torch.long)
        frame_timestamps = torch.as_tensor(sync.get("frame_timestamps", []), dtype=torch.float32)
        mel_timestamps = torch.as_tensor(sync.get("mel_timestamps", []), dtype=torch.float32)

        label = int(bundle.get("label", 0))

        sample = {
            "video": video,
            "audio": mel_seq,
            "frame_indices": frame_indices,
            "frame_timestamps": frame_timestamps,
            "mel_timestamps": mel_timestamps,
            "label": torch.tensor(label, dtype=torch.long),
        }

        if self.return_metadata:
            sample["metadata"] = {
                "path": str(path),
                "speaker_id": bundle.get("speaker_id"),
                "type": bundle.get("type"),
            }
        return sample
