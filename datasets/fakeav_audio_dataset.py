from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .fakeav_video_dataset import DatasetSplit, _read_index


@dataclass
class AudioDatasetConfig:
    target_steps: int = 400
    random_crop: bool = False
    video_target_frames: int | None = None


class FakeAVAudioDataset(Dataset[Dict[str, object]]):
    """Dataset returning audio tensors (mel, waveform) with optional speaker labels."""

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
        return_waveform: bool = False,
        return_speaker: bool = False,
        speaker_map: Optional[Dict[str, int]] = None,
        return_video: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Processed data directory not found: {self.data_dir}")
        self.transform = transform
        self.return_metadata = return_metadata
        self.return_waveform = return_waveform
        self.return_speaker = return_speaker
        self.return_video = return_video
        self.config = config or AudioDatasetConfig()
        self.seed = seed
        self._rng = random.Random(seed)
        self._record_index: Dict[str, dict] = {}

        index_path = Path(index_file) if index_file else self.data_dir / "preprocess_index.jsonl"
        if index_path.exists():
            records = _read_index(index_path)
            files = [
                self.data_dir / Path(record["output_path"]).name
                for record in records
                if record.get("status") == "ok"
            ]
            for record in records:
                if record.get("status") == "ok":
                    name = Path(record["output_path"]).name
                    self._record_index[name] = record
        else:
            files = sorted(self.data_dir.glob("*.pt"))

        if not files:
            raise RuntimeError(f"No processed .pt samples found in {self.data_dir}")

        split_scheme = split_scheme or DatasetSplit()
        self.files = self._apply_split(files, split, split_scheme)

        if self.return_speaker:
            if speaker_map is not None:
                self.speaker_to_idx = dict(speaker_map)
            else:
                speakers = sorted({self._infer_speaker_id(path) for path in self.files})
                self.speaker_to_idx = {sid: idx for idx, sid in enumerate(speakers)}
            self.num_speakers = len(self.speaker_to_idx)
        else:
            self.speaker_to_idx = {}
            self.num_speakers = 0

    def _apply_split(
        self,
        files: Sequence[Path],
        split: Optional[str],
        scheme: DatasetSplit,
    ) -> List[Path]:
        if split not in {None, "train", "val", "test"}:
            raise ValueError("split must be one of train/val/test or None.")
        grouped: Dict[str, List[Path]] = {}
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
            ordered: List[Path] = []
            for speaker in speakers:
                ordered.extend(grouped[speaker])
            return ordered
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

    def _prepare_video_frames(self, video: torch.Tensor) -> torch.Tensor:
        target = self.config.video_target_frames
        if target is None or video.shape[0] == target:
            return video.float()
        frames = video.shape[0]
        if frames > target:
            start = max((frames - target) // 2, 0)
            return video[start : start + target].float()
        pad = target - frames
        pad_before = pad // 2
        pad_after = pad - pad_before
        pieces = []
        if pad_before > 0:
            pieces.append(video[:1].repeat(pad_before, 1, 1, 1))
        pieces.append(video)
        if pad_after > 0:
            pieces.append(video[-1:].repeat(pad_after, 1, 1, 1))
        return torch.cat(pieces, dim=0).float()

    def _infer_speaker_id(self, path: Path, bundle: Optional[dict] = None) -> str:
        if bundle is not None and bundle.get("speaker_id") is not None:
            return str(bundle["speaker_id"])
        record = self._record_index.get(path.name)
        if record is not None and record.get("speaker_id") is not None:
            return str(record["speaker_id"])
        return path.stem.split("_")[0]

    def __getitem__(self, index: int) -> Dict[str, object]:
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
        sample: Dict[str, object] = {
            "mel": mel,
            "label": torch.tensor(label, dtype=torch.long),
        }

        if self.return_waveform:
            waveform = audio.get("waveform") if isinstance(audio, dict) else None
            if waveform is None:
                raise RuntimeError(f"Waveform not stored for {path}. Re-run preprocessing with --save-waveform.")
            sample["waveform"] = waveform.float().clone()

        if self.return_video:
            video = bundle.get("video")
            if not isinstance(video, torch.Tensor):
                raise RuntimeError("Video frames requested but missing in sample.")
            video = video.float()
            video = self._prepare_video_frames(video)
            sample["video"] = video

        if self.return_speaker:
            speaker_id = self._infer_speaker_id(path, bundle)
            speaker_idx = self.speaker_to_idx.get(speaker_id)
            if speaker_idx is None:
                speaker_idx = len(self.speaker_to_idx)
                self.speaker_to_idx[speaker_id] = speaker_idx
                self.num_speakers = max(self.num_speakers, speaker_idx + 1)
            sample["speaker"] = torch.tensor(speaker_idx, dtype=torch.long)

        if self.return_metadata:
            sample["metadata"] = {
                "path": str(path),
                "speaker_id": bundle.get("speaker_id"),
                "type": bundle.get("type"),
            }

        return sample
