from __future__ import annotations

import random
from dataclasses import dataclass
from os import PathLike
from typing import Dict, Optional, Sequence, TypedDict

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from datasets import AudioDatasetConfig, DatasetSplit, FakeAVAudioDataset

from .augmentations import MelAugmentation, VideoAugmentation


@dataclass
class TriModalDatasetConfig:
    target_steps: int = 400
    video_frames: int = 48
    sync_audio_steps: int = 64
    sync_video_frames: int = 16
    waveform_samples: int = 160000
    mel_random_crop: bool = True
    sync_negative_prob: float = 0.35
    sync_max_shift: int = 4
    video_size: int | None = None
    mel_bins: int | None = None


class TriModalBatch(TypedDict):
    mel: torch.Tensor
    mel_sync: torch.Tensor
    waveform: torch.Tensor
    waveform_lengths: torch.Tensor
    video: torch.Tensor
    video_sync: torch.Tensor
    label: torch.Tensor
    sync_label: torch.Tensor
    metadata: Optional[Sequence[Optional[Dict[str, object]]]]


class FakeAVTriModalDataset(FakeAVAudioDataset):
    """
    Extends FakeAVAudioDataset to produce synchronised waveform/mel/video tensors per sample.
    """

    def __init__(
        self,
        data_dir: str | PathLike[str],
        *,
        index_file: str | PathLike[str] | None = None,
        split: str | None = None,
        split_scheme: DatasetSplit | None = None,
        seed: int = 1337,
        config: TriModalDatasetConfig | None = None,
        audio_augment: MelAugmentation | None = None,
        video_augment: VideoAugmentation | None = None,
        train: bool = True,
        return_metadata: bool = False,
    ) -> None:
        cfg = config or TriModalDatasetConfig()
        audio_cfg = AudioDatasetConfig(
            target_steps=cfg.target_steps,
            random_crop=cfg.mel_random_crop if train else False,
            video_target_frames=max(cfg.video_frames, cfg.sync_video_frames),
        )
        super().__init__(
            data_dir,
            index_file=index_file,
            split=split,
            split_scheme=split_scheme,
            seed=seed,
            config=audio_cfg,
            transform=None,
            return_metadata=return_metadata,
            return_waveform=True,
            return_video=True,
        )
        self.train = train
        self.cfg = cfg
        self.audio_augment = audio_augment
        self.video_augment = video_augment
        self.return_metadata = return_metadata

    @staticmethod
    def _ensure_waveform_shape(waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            return waveform.unsqueeze(0)
        return waveform

    @staticmethod
    def _temporal_slice(seq: torch.Tensor, length: int, train: bool) -> torch.Tensor:
        if length <= 0:
            return seq
        total = seq.size(0)
        if total == length:
            return seq
        if total > length:
            if train:
                start = random.randint(0, total - length)
            else:
                start = max((total - length) // 2, 0)
            return seq[start : start + length]
        pad = length - total
        pad_before = pad // 2
        pad_after = pad - pad_before
        first = seq[:1].expand(pad_before, *seq.shape[1:]) if pad_before > 0 else None
        last = seq[-1:].expand(pad_after, *seq.shape[1:]) if pad_after > 0 else None
        pieces = [p for p in (first, seq, last) if p is not None]
        return torch.cat(pieces, dim=0)

    def _resize_spatial(self, video: torch.Tensor) -> torch.Tensor:
        if self.cfg.video_size is None:
            return video
        target = self.cfg.video_size
        if video.size(-1) == target and video.size(-2) == target:
            return video
        return F.interpolate(
            video,
            size=(target, target),
            mode="bilinear",
            align_corners=False,
        )

    def _prepare_video_views(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        video = video.float()
        video = self._resize_spatial(video)
        video_branch = self._temporal_slice(video, self.cfg.video_frames, self.train)
        sync_branch = self._temporal_slice(video, self.cfg.sync_video_frames, self.train)
        video_branch = video_branch.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
        sync_branch = sync_branch.contiguous()  # [T, C, H, W]
        if self.video_augment is not None and self.train:
            video_branch = self.video_augment(video_branch)
        return video_branch, sync_branch

    def _prepare_mel_views(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if mel.dim() == 3:
            mel = mel.squeeze(0)
        mel = mel.float()
        mel = self._maybe_resample_mel(mel)
        augmented = mel.unsqueeze(0)
        if self.audio_augment is not None and self.train:
            augmented = self.audio_augment(augmented)
        sync = augmented.squeeze(0).permute(1, 0).contiguous()
        sync_len = max(int(self.cfg.sync_audio_steps), 1)
        sync = self._temporal_slice(sync, sync_len, self.train)
        return augmented, sync

    def _maybe_resample_mel(self, mel: torch.Tensor) -> torch.Tensor:
        target = self.cfg.mel_bins
        if target is None or mel.size(-2) == target:
            return mel
        resized = F.interpolate(
            mel.unsqueeze(0).unsqueeze(0),
            size=(target, mel.size(-1)),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0).squeeze(0)

    def _prepare_waveform(self, waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
        waveform = self._ensure_waveform_shape(waveform.float())
        total = waveform.size(-1)
        target = max(int(self.cfg.waveform_samples), 1)
        if total > target:
            if self.train:
                start = random.randint(0, total - target)
            else:
                start = max((total - target) // 2, 0)
            chunk = waveform[..., start : start + target]
        elif total < target:
            pad = target - total
            chunk = F.pad(waveform, (0, pad))
        else:
            chunk = waveform
        chunk = chunk.squeeze(0)
        length = min(target, chunk.size(-1))
        return chunk, length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int | Dict[str, object]]:
        base_sample = super().__getitem__(index)
        mel_full, mel_sync = self._prepare_mel_views(base_sample["mel"])
        video_branch, sync_video = self._prepare_video_views(base_sample["video"])
        waveform, waveform_length = self._prepare_waveform(base_sample["waveform"])
        sync_label = torch.tensor(0, dtype=torch.long)
        if self.train and self.cfg.sync_negative_prob > 0.0 and random.random() < self.cfg.sync_negative_prob:
            shift = random.randint(1, max(1, self.cfg.sync_max_shift))
            sync_video = torch.roll(sync_video, shifts=shift, dims=0)
            sync_label = torch.tensor(1, dtype=torch.long)
        label = base_sample["label"]
        sample: Dict[str, torch.Tensor | int | Dict[str, object]] = {
            "mel": mel_full,
            "mel_sync": mel_sync,
            "waveform": waveform,
            "waveform_length": torch.tensor(waveform_length, dtype=torch.long),
            "video": video_branch,
            "video_sync": sync_video,
            "label": label,
            "sync_label": sync_label,
        }
        if self.return_metadata and "metadata" in base_sample:
            sample["metadata"] = base_sample["metadata"]
        return sample


@dataclass
class TriModalCollator:
    waveform_pad_value: float = 0.0

    def __call__(self, batch: Sequence[Dict[str, object]]) -> TriModalBatch:
        mel = torch.stack([item["mel"] for item in batch], dim=0)
        mel_sync = torch.stack([item["mel_sync"] for item in batch], dim=0)
        waveform_lengths = torch.stack([item["waveform_length"] for item in batch], dim=0)
        waveforms = pad_sequence([item["waveform"] for item in batch], batch_first=True, padding_value=self.waveform_pad_value)
        video = torch.stack([item["video"] for item in batch], dim=0)
        video_sync = torch.stack([item["video_sync"] for item in batch], dim=0)
        labels = torch.stack([item["label"] for item in batch], dim=0)
        sync_labels = torch.stack([item["sync_label"] for item in batch], dim=0)
        metadata = [item.get("metadata") for item in batch]
        collated: TriModalBatch = {
            "mel": mel,
            "mel_sync": mel_sync,
            "waveform": waveforms,
            "waveform_lengths": waveform_lengths,
            "video": video,
            "video_sync": video_sync,
            "label": labels,
            "sync_label": sync_labels,
            "metadata": metadata,
        }
        return collated
