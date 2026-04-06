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
    target_frames: Optional[int] = None
    min_roll_ratio: float = 0.3
    max_roll_ratio: float = 0.85
    swap_prob: float = 0.8
    force_cross_speaker: bool = True
    paired_negatives: bool = False


class FakeAVSyncDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, int, int]]):
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
        metadata_fields: Optional[Sequence[str]] = None,
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
        self.metadata_fields = metadata_fields

        self.base_files = self._apply_split(files, split)
        if not self.base_files:
            raise RuntimeError("No samples selected for this split.")

        self.file_speakers: Dict[Path, str] = {}
        self.speaker_to_files: Dict[str, List[Path]] = {}
        for path in self.base_files:
            speaker = path.stem.split("_")[0]
            self.file_speakers[path] = speaker
            self.speaker_to_files.setdefault(speaker, []).append(path)

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
        return len(self.base_files) * 2 if self.config.paired_negatives else len(self.base_files)

    @staticmethod
    def _load_bundle(path: Path) -> Dict[str, object]:
        return torch.load(path, map_location="cpu")

    @staticmethod
    def _prepare_mel_sequence(mel: torch.Tensor, target_len: int) -> torch.Tensor:
        if mel.dim() == 3:
            mel = mel.squeeze(0)
        mel = mel.float()
        if target_len <= 1:
            return mel.permute(1, 0)
        mel_steps = mel.shape[-1]
        if mel_steps <= 1:
            return mel.permute(1, 0).repeat(target_len, 1)
        resized = (
            F.interpolate(
                mel.unsqueeze(0),
                size=target_len,
                mode="linear",
                align_corners=False,
            )
            .squeeze(0)
            .permute(1, 0)
        )
        return resized

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
        if self.config.paired_negatives:
            base_idx = index // 2
            force_negative = (index % 2) == 1
        else:
            base_idx = index
            force_negative = False

        path = self.base_files[base_idx]
        bundle = self._load_bundle(path)
        video: torch.Tensor = bundle["video"]
        video = self._maybe_trim_frames(video.float())

        audio = bundle.get("audio", {})
        mel = audio.get("mel") if isinstance(audio, dict) else None
        if not isinstance(mel, torch.Tensor):
            raise RuntimeError(f"No mel spectrogram found in {path}")

        frame_count = video.shape[0]
        mel_seq = self._prepare_mel_sequence(mel, frame_count)

        speaker_id = self.file_speakers[path]
        rng = random.Random(self.seed * 9973 + base_idx * 131 + (1 if force_negative else 0))
        # When paired_negatives is enabled, each base sample should emit one aligned
        # positive view and one deliberately misaligned negative view. Applying
        # negative_prob on top of the positive half skews the dataset toward negatives.
        make_negative = force_negative
        if not self.config.paired_negatives and not make_negative and self.config.negative_prob > 0.0:
            make_negative = rng.random() < self.config.negative_prob

        if make_negative:
            mel_seq = self._make_negative(
                mel_seq,
                frame_count=frame_count,
                rng=rng,
                source_path=path,
                speaker_id=speaker_id,
            )
            sync_label = 0
        else:
            sync_label = 1

        deepfake_label = int(bundle.get("label", 0))
        if self.return_metadata:
            metadata: Dict[str, object] = {}
            fields = set(self.metadata_fields or ())
            if not fields or "path" in fields:
                metadata["path"] = str(path)
            if not fields or "speaker_id" in fields:
                metadata["speaker_id"] = speaker_id
            return video, mel_seq, sync_label, deepfake_label, metadata
        return video, mel_seq, sync_label, deepfake_label

    def _sample_roll_shift(self, frame_count: int, rng: random.Random) -> int:
        if frame_count <= 1:
            return 0
        min_ratio = max(float(self.config.min_roll_ratio), 0.0)
        max_ratio = max(float(self.config.max_roll_ratio), min_ratio)
        min_shift = max(int(round(min_ratio * frame_count)), 1)
        max_shift = max(int(round(max_ratio * frame_count)), min_shift)
        max_shift = min(max_shift, frame_count - 1)
        if max_shift < 1:
            return 1
        return rng.randint(min_shift, max_shift)

    def _sample_swap_sequence(
        self,
        target_len: int,
        rng: random.Random,
        exclude_path: Path,
        speaker_id: str,
    ) -> Optional[torch.Tensor]:
        if len(self.base_files) <= 1:
            return None
        require_diff = self.config.force_cross_speaker and len(self.speaker_to_files) > 1
        for _ in range(10):
            if require_diff:
                candidates = [sp for sp in self.speaker_to_files if sp != speaker_id]
                if not candidates:
                    require_diff = False
                    continue
                speaker_choice = rng.choice(candidates)
                pool = self.speaker_to_files.get(speaker_choice, [])
                if not pool:
                    continue
                candidate_path = rng.choice(pool)
            else:
                candidate_path = rng.choice(self.base_files)
            if candidate_path == exclude_path:
                continue
            bundle = self._load_bundle(candidate_path)
            audio = bundle.get("audio", {})
            mel = audio.get("mel") if isinstance(audio, dict) else None
            if isinstance(mel, torch.Tensor):
                return self._prepare_mel_sequence(mel, target_len)
        return None

    def _make_negative(
        self,
        mel_seq: torch.Tensor,
        *,
        frame_count: int,
        rng: random.Random,
        source_path: Path,
        speaker_id: str,
    ) -> torch.Tensor:
        if self.config.swap_prob > 0.0 and rng.random() < self.config.swap_prob:
            swapped = self._sample_swap_sequence(frame_count, rng, source_path, speaker_id)
            if swapped is not None:
                return swapped
        shift = self._sample_roll_shift(frame_count, rng)
        return mel_seq.roll(shifts=shift, dims=0)
