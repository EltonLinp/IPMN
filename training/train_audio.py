from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import sys
from collections import Counter
from pathlib import Path, PureWindowsPath
from typing import Dict, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Beta
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from datasets import AudioDatasetConfig, DatasetSplit, FakeAVAudioDataset
from datasets.fakeav_video_dataset import _read_index
from models import AASISTLite, SyncModule, WavLMClassifier, WavLMConfig


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, help="Path to JSON config file.")
    config_args, remaining = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train an audio-only FakeAVCeleb classifier.",
        parents=[config_parser],
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing processed .pt files.")
    parser.add_argument("--index-file", type=Path, default=None, help="Optional preprocess_index.jsonl path.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for backbone/head.")
    parser.add_argument("--front-lr", type=float, default=5e-5, help="Learning rate for frozen front stages once unfrozen.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--target-steps", type=int, default=400, help="Temporal length of Mel spectrograms after pad/crop.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use.")
    parser.add_argument("--save-path", type=Path, default=Path("aasist_lite_audio.pt"), help="Model checkpoint path.")
    parser.add_argument("--balanced-sampler", action="store_true", help="Use class-balanced sampling for the training loader.")
    parser.add_argument("--class-weights", action="store_true", help="Apply class-weighted loss.")
    parser.add_argument("--verify-dataset", action="store_true", help="Validate processed samples before training.")
    parser.add_argument("--early-stop-patience", type=int, default=5, help="Early stopping patience on val MCC (0 disables).")
    parser.add_argument(
        "--target-val-eer",
        type=float,
        default=0.0,
        help="Optional validation EER target; training stops once best val EER <= target (>0 enables).",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Clip gradient norm to this value (<=0 disables).")
    parser.add_argument("--mel-bins", type=int, default=80, help="Target Mel bins (resampled if differs from stored features).")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k pooling size for segment logits aggregation.")
    parser.add_argument("--freeze-stages", type=int, default=2, help="Number of early backbone stages to freeze during warmup.")
    parser.add_argument("--freeze-epochs", type=int, default=4, help="Epochs to keep early stages frozen before fine-tuning.")
    parser.add_argument("--real-augment-prob", type=float, default=0.0, help="Probability of applying augmentation to real samples.")
    parser.add_argument("--real-augment-start", type=int, default=0, help="Epoch number (1-indexed) after which real sample augmentation is enabled.")
    parser.add_argument("--real-freq-mask", type=int, default=0, help="Max frequency bins to mask when augmenting real samples.")
    parser.add_argument("--real-time-mask", type=int, default=0, help="Max time steps to mask when augmenting real samples.")
    parser.add_argument("--real-noise-std", type=float, default=0.0, help="Std of Gaussian noise added to real samples during augmentation.")
    parser.add_argument("--real-gain-std", type=float, default=0.0, help="Std of multiplicative gain applied to real samples.")
    parser.add_argument("--real-shift-pct", type=float, default=0.0, help="Max percentage of temporal shift applied to real samples.")
    parser.add_argument("--real-mixup-prob", type=float, default=0.0, help="Probability of mixing two real samples together.")
    parser.add_argument("--real-mixup-alpha", type=float, default=0.2, help="Beta distribution alpha used for real sample mixup.")
    parser.add_argument("--fake-augment-prob", type=float, default=0.0, help="Probability of applying augmentation to fake samples.")
    parser.add_argument("--fake-freq-mask", type=int, default=0, help="Max frequency bins to mask when augmenting fake samples.")
    parser.add_argument("--fake-time-mask", type=int, default=0, help="Max time steps to mask when augmenting fake samples.")
    parser.add_argument("--fake-noise-std", type=float, default=0.0, help="Std of Gaussian noise added to fake samples during augmentation.")
    parser.add_argument("--global-augment-prob", type=float, default=0.0, help="Probability of applying augmentation to any sample.")
    parser.add_argument("--global-freq-mask", type=int, default=0, help="Max frequency bins to mask during global augmentation.")
    parser.add_argument("--global-time-mask", type=int, default=0, help="Max time steps to mask during global augmentation.")
    parser.add_argument("--global-noise-std", type=float, default=0.0, help="Std of Gaussian noise added during global augmentation.")
    parser.add_argument("--pseudo-fake-prob", type=float, default=0.0, help="Probability of creating pseudo fake samples from real inputs.")
    parser.add_argument("--pseudo-fake-max", type=int, default=0, help="Maximum number of pseudo fake samples per batch (0 disables cap).")
    parser.add_argument("--pseudo-fake-freq-mask", type=int, default=8, help="Frequency masking size for pseudo fake generation.")
    parser.add_argument("--pseudo-fake-time-mask", type=int, default=32, help="Time masking size for pseudo fake generation.")
    parser.add_argument("--pseudo-fake-noise", type=float, default=0.01, help="Noise std applied when crafting pseudo fake samples.")
    parser.add_argument("--pseudo-fake-start-epoch", type=int, default=1, help="Epoch to start enabling pseudo fake generation.")
    parser.add_argument("--pseudo-fake-ramp-epochs", type=int, default=0, help="Number of epochs to linearly ramp pseudo fake probability.")
    parser.add_argument("--val-real-copies", type=int, default=0, help="Number of duplicated real samples added to validation/test datasets.")
    parser.add_argument(
        "--val-augment-freq-mask",
        type=int,
        default=0,
        help="Frequency mask width applied to duplicated validation real samples.",
    )
    parser.add_argument(
        "--val-augment-time-mask",
        type=int,
        default=0,
        help="Time mask width applied to duplicated validation real samples.",
    )
    parser.add_argument(
        "--val-augment-noise-std",
        type=float,
        default=0.0,
        help="Noise std applied to duplicated validation real samples.",
    )
    parser.add_argument(
        "--val-augment-shift-pct",
        type=float,
        default=0.0,
        help="Temporal shift percentage applied to duplicated validation real samples.",
    )
    parser.add_argument(
        "--val-augment-gain-std",
        type=float,
        default=0.0,
        help="Gain std applied to duplicated validation real samples.",
    )
    parser.add_argument(
        "--val-augment-mixup-prob",
        type=float,
        default=0.0,
        help="Probability of mixing two validation real samples when duplicating.",
    )
    parser.add_argument(
        "--val-augment-mixup-alpha",
        type=float,
        default=0.2,
        help="Beta alpha for validation real mixup.",
    )
    parser.add_argument(
        "--use-sync-fusion",
        action="store_true",
        help="Fuse pretrained audio-video synchronization features with the audio branch.",
    )
    parser.add_argument(
        "--sync-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path for the pretrained sync module.",
    )
    parser.add_argument(
        "--sync-vit-path",
        type=Path,
        default=None,
        help="ViT weights directory used by the sync module.",
    )
    parser.add_argument(
        "--sync-target-frames",
        type=int,
        default=16,
        help="Frame count fed into the sync module.",
    )
    parser.add_argument(
        "--sync-audio-dim",
        type=int,
        default=64,
        help="Audio embedding dimension expected by the sync module.",
    )
    parser.add_argument(
        "--sync-fusion-dropout",
        type=float,
        default=0.2,
        help="Dropout probability inside the audio-sync fusion head.",
    )
    parser.add_argument(
        "--sync-transformer-heads",
        type=int,
        default=8,
        help="Number of attention heads in the sync module.",
    )
    parser.add_argument(
        "--sync-temporal-layers",
        type=int,
        default=1,
        help="Transformer encoder layers used by the sync module.",
    )
    parser.add_argument(
        "--sync-trainable",
        action="store_true",
        help="Fine-tune selected components of the sync module.",
    )
    parser.add_argument(
        "--sync-lr",
        type=float,
        default=1e-5,
        help="Learning rate for trainable sync module components.",
    )
    parser.add_argument(
        "--sync-distill-weight",
        type=float,
        default=0.05,
        help="Weight for distillation loss between sync logits and fused logits.",
    )
    parser.add_argument(
        "--sync-distill-temp",
        type=float,
        default=2.0,
        help="Temperature used for sync distillation soft targets.",
    )
    parser.add_argument(
        "--sync-gate-alpha",
        type=float,
        default=6.0,
        help="Scaling factor for sync logits gating.",
    )
    parser.add_argument(
        "--sync-gate-beta",
        type=float,
        default=0.2,
        help="Bias term for sync logits gating.",
    )
    parser.add_argument(
        "--real-margin-weight",
        type=float,
        default=0.0,
        help="Additional hinge penalty weight enforcing larger logit margins on real samples.",
    )
    parser.add_argument(
        "--real-margin-value",
        type=float,
        default=0.3,
        help="Target logit margin (real logit_0 - logit_1) for the hinge penalty.",
    )
    parser.add_argument(
        "--real-margin-start-epoch",
        type=int,
        default=1,
        help="Epoch at which real margin penalty becomes active.",
    )
    parser.add_argument(
        "--real-margin-warmup-epochs",
        type=int,
        default=4,
        help="Epochs to linearly ramp the real margin penalty.",
    )
    parser.add_argument(
        "--wave-branch-mode",
        type=str,
        default="none",
        choices=["none", "all", "real_only", "fake_only"],
        help="Waveform branch usage: none disables, real_only/fake_only enable per class, all enables every sample.",
    )
    parser.add_argument(
        "--wave-segment-samples",
        type=int,
        default=64000,
        help="Number of waveform samples kept per clip after segmentation (0 keeps original length).",
    )
    parser.add_argument(
        "--train-wave-segments",
        type=int,
        default=1,
        help="Number of random waveform segments averaged per training batch (>=1).",
    )
    parser.add_argument(
        "--audio-backbone",
        type=str,
        default="aasist",
        choices=["aasist", "wavlm"],
        help="Audio backbone to train (AASIST Lite or WavLM).",
    )
    parser.add_argument(
        "--wavlm-model-name",
        type=str,
        default="microsoft/wavlm-base-plus-sv",
        help="Hugging Face checkpoint to load for WavLM backbone.",
    )
    parser.add_argument(
        "--hf-local-files-only",
        action="store_true",
        help="Force Hugging Face models to load from local cache only (offline mode).",
    )
    parser.add_argument(
        "--wavlm-trainable",
        action="store_true",
        help="Fine-tune a subset of WavLM backbone layers.",
    )
    parser.add_argument(
        "--wavlm-dropout",
        type=float,
        default=0.2,
        help="Dropout probability applied to the WavLM classification head.",
    )
    parser.add_argument(
        "--use-arcface-head",
        action="store_true",
        help="Replace linear classifier with a CosFace/ArcFace style margin head (WavLM only).",
    )
    parser.add_argument(
        "--arcface-scale",
        type=float,
        default=30.0,
        help="Feature scaling factor for the ArcFace/CosFace head.",
    )
    parser.add_argument(
        "--arcface-margin",
        type=float,
        default=0.2,
        help="Margin applied to the target logit when using ArcFace/CosFace head.",
    )
    parser.add_argument(
        "--arcface-warmup-epochs",
        type=int,
        default=5,
        help="Number of epochs to train only the ArcFace head before unfreezing the backbone.",
    )
    parser.add_argument(
        "--wavlm-unfreeze-layers",
        type=int,
        default=6,
        help="Number of final transformer layers in WavLM backbone to unfreeze when trainable.",
    )
    parser.add_argument(
        "--wavlm-backbone-lr",
        type=float,
        default=1e-4,
        help="Learning rate for trainable WavLM backbone layers.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Number of warmup epochs applied before cosine decay.",
    )
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=None,
        help="Optional manual weight for the fake (positive) class to penalise missed detections.",
    )
    parser.add_argument(
        "--focal-loss",
        action="store_true",
        help="Use Focal Loss instead of standard cross entropy.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for Focal Loss (ignored if not enabled).",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps before optimizer update.",
    )
    parser.add_argument(
        "--eval-wave-segments",
        type=int,
        default=1,
        help="Number of random waveform segments averaged during evaluation (>=1).",
    )
    parser.add_argument(
        "--eval-augment-passes",
        type=int,
        default=1,
        help="Number of evaluation-time augmentation passes averaged per batch.",
    )
    parser.add_argument(
        "--eval-augment-prob",
        type=float,
        default=0.0,
        help="Probability of applying SpecAug-style masks per evaluation pass.",
    )
    parser.add_argument(
        "--eval-freq-mask",
        type=int,
        default=0,
        help="Max frequency bins masked during evaluation augmentation.",
    )
    parser.add_argument(
        "--eval-time-mask",
        type=int,
        default=0,
        help="Max time steps masked during evaluation augmentation.",
    )
    parser.add_argument(
        "--eval-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std applied during evaluation augmentation.",
    )
    parser.add_argument(
        "--use-center-loss",
        action="store_true",
        help="Enable real-class center loss regularization on WavLM embeddings.",
    )
    parser.add_argument(
        "--center-loss-weight",
        type=float,
        default=0.01,
        help="Multiplier for the center loss term added to the main loss.",
    )
    parser.add_argument(
        "--center-loss-start-epoch",
        type=int,
        default=1,
        help="Epoch at which center loss becomes active.",
    )
    parser.add_argument(
        "--center-loss-warmup-epochs",
        type=int,
        default=5,
        help="Epochs to linearly ramp center loss weight after it becomes active.",
    )
    parser.add_argument(
        "--center-loss-lr",
        type=float,
        default=0.001,
        help="Learning rate for the center loss parameters.",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Enable Exponential Moving Average (EMA) shadow model for evaluation/checkpointing.",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.998,
        help="EMA decay factor (closer to 1.0 means slower updates).",
    )
    parser.add_argument(
        "--ema-start-epoch",
        type=int,
        default=3,
        help="Epoch after which EMA updates begin.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume fine-tuning from.",
    )
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="When resuming, also load optimizer state (default loads weights only).",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "plateau"],
        help="Learning rate scheduler strategy.",
    )
    parser.add_argument(
        "--plateau-factor",
        type=float,
        default=0.5,
        help="ReduceLROnPlateau factor when scheduler='plateau'.",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=2,
        help="Epoch patience for ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--plateau-min-lr",
        type=float,
        default=1e-6,
        help="Minimum LR for ReduceLROnPlateau.",
    )

    if config_args.config:
        with config_args.config.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        cfg_cli: list[str] = []
        for key, value in config.items():
            if value is None:
                continue
            option = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cfg_cli.append(option)
            elif isinstance(value, list):
                cfg_cli.append(option)
                cfg_cli.extend(str(v) for v in value)
            else:
                cfg_cli.extend([option, str(value)])
        remaining = cfg_cli + remaining
    args = parser.parse_args(remaining)
    return args


def _load_labels_for_files(
    files: Sequence[Path],
    index_file: Path | None,
) -> list[int]:
    label_map: Dict[str, int] = {}
    if index_file and index_file.exists():
        records = _read_index(index_file)
        for record in records:
            if record.get("status") != "ok":
                continue
            name = Path(str(record.get("output_path", ""))).name
            label_map[name] = int(record.get("label", 0))
    labels: list[int] = []
    for path in files:
        key = path.name
        if key in label_map:
            labels.append(label_map[key])
            continue
        bundle = torch.load(path, map_location="cpu")
        labels.append(int(bundle.get("label", 0)))
    return labels


def _build_sampler(
    dataset: FakeAVAudioDataset,
    index_file: Path | None,
) -> WeightedRandomSampler:
    labels = _load_labels_for_files(dataset.files, index_file)
    counts = Counter(labels)
    weights = [1.0 / max(counts[label], 1) for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class AudioCollator:
    """Collate Mel tensors with optional waveform/video padding."""

    def __init__(
        self,
        *,
        include_waveform: bool,
        include_video: bool = False,
        include_metadata: bool = False,
    ) -> None:
        self.include_waveform = include_waveform
        self.include_video = include_video
        self.include_metadata = include_metadata

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        if not batch:
            return {}
        mels: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        metadata: list[dict] = []
        waveforms: list[torch.Tensor] = []
        lengths: list[int] = []
        videos: list[torch.Tensor] = []
        for sample in batch:
            mel = sample.get("mel")
            label = sample.get("label")
            if not isinstance(mel, torch.Tensor) or not isinstance(label, torch.Tensor):
                raise TypeError("Batch entries must include 'mel' and 'label' tensors.")
            mels.append(mel.float())
            labels.append(label.long())
            if self.include_waveform:
                waveform = sample.get("waveform")
                if not isinstance(waveform, torch.Tensor):
                    raise RuntimeError("Waveform requested but missing or not a Tensor in batch sample.")
                waveform = waveform.float()
                if waveform.dim() == 2:
                    waveform = waveform.mean(dim=0)
                elif waveform.dim() != 1:
                    raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")
                lengths.append(int(waveform.size(0)))
                waveforms.append(waveform)
            if self.include_video:
                video = sample.get("video")
                if not isinstance(video, torch.Tensor):
                    raise RuntimeError("Video requested but missing or not a Tensor in batch sample.")
                videos.append(video.float())
            if self.include_metadata:
                metadata.append(sample.get("metadata", {}))
        payload: Dict[str, object] = {
            "mel": torch.stack(mels, dim=0),
            "label": torch.stack(labels, dim=0),
        }
        if self.include_waveform:
            padded = pad_sequence([wf for wf in waveforms], batch_first=True)
            payload["waveform"] = padded
            payload["waveform_length"] = torch.tensor(lengths, dtype=torch.long)
        if self.include_video and videos:
            payload["video"] = torch.stack(videos, dim=0)
        if self.include_metadata:
            payload["metadata"] = metadata
        return payload


class ValidationAugmentedDataset(Dataset):
    """Dataset wrapper that replicates real (label=0) samples with fixed augmentations."""

    def __init__(
        self,
        base_dataset: FakeAVAudioDataset,
        real_indices: Sequence[int],
        *,
        copies: int,
        freq_mask: int,
        time_mask: int,
        noise_std: float,
        shift_pct: float,
        gain_std: float,
        mixup_prob: float,
        mixup_alpha: float,
    ) -> None:
        if copies <= 0 or not real_indices:
            raise ValueError("ValidationAugmentedDataset requires positive copies and at least one real sample.")
        self.base_dataset = base_dataset
        self.real_indices = list(real_indices)
        self.copies = copies
        self.freq_mask = int(freq_mask)
        self.time_mask = int(time_mask)
        self.noise_std = float(noise_std)
        self.shift_pct = float(max(shift_pct, 0.0))
        self.gain_std = float(max(gain_std, 0.0))
        self.mixup_prob = float(max(mixup_prob, 0.0))
        self.mixup_alpha = float(max(mixup_alpha, 1e-3))
        self.extra_count = len(self.real_indices) * self.copies

    def __len__(self) -> int:
        return len(self.base_dataset) + self.extra_count

    def __getitem__(self, index: int) -> Dict[str, object]:
        if index < len(self.base_dataset):
            return self.base_dataset[index]
        if not self.real_indices:
            return self.base_dataset[index % len(self.base_dataset)]
        offset = index - len(self.base_dataset)
        base_idx = self.real_indices[offset % len(self.real_indices)]
        sample = copy.deepcopy(self.base_dataset[base_idx])
        mel = sample.get("mel")
        if isinstance(mel, torch.Tensor):
            mel = mel.clone()
            mel = _augment_single_sample(
                mel,
                freq_mask=self.freq_mask,
                time_mask=self.time_mask,
                noise_std=self.noise_std,
            )
            mel = _apply_shift_gain(mel, shift_pct=self.shift_pct, gain_std=self.gain_std)
            sample["mel"] = mel
            if self.mixup_prob > 0.0 and len(self.real_indices) > 1 and torch.rand(1).item() <= self.mixup_prob:
                other_idx = self.real_indices[torch.randint(0, len(self.real_indices), (1,)).item()]
                if other_idx != base_idx:
                    other = self.base_dataset[other_idx]
                    other_mel = other.get("mel")
                    if isinstance(other_mel, torch.Tensor):
                        other_mel = other_mel.to(mel.device, mel.dtype)
                        alpha_tensor = torch.tensor(self.mixup_alpha, device=mel.device, dtype=mel.dtype)
                        lam = Beta(alpha_tensor, alpha_tensor).sample().clamp(0.0, 1.0)
                        sample["mel"] = lam * mel + (1.0 - lam) * other_mel
        waveform = sample.get("waveform")
        if isinstance(waveform, torch.Tensor) and self.noise_std > 0.0:
            sample["waveform"] = waveform.clone() + torch.randn_like(waveform) * (self.noise_std * 0.5)
        return sample


def build_dataloaders(
    data_dir: Path,
    index_file: Path | None,
    batch_size: int,
    num_workers: int,
    target_steps: int,
    balanced_sampler: bool,
    include_waveform: bool,
    include_video: bool,
    *,
    video_target_frames: int | None,
    val_real_copies: int,
    val_augment_freq_mask: int,
    val_augment_time_mask: int,
    val_augment_noise_std: float,
    val_augment_shift_pct: float,
    val_augment_gain_std: float,
    val_augment_mixup_prob: float,
    val_augment_mixup_alpha: float,
) -> Tuple[FakeAVAudioDataset, DataLoader, DataLoader, DataLoader]:
    split = DatasetSplit()
    train_dataset = FakeAVAudioDataset(
        data_dir=data_dir,
        index_file=index_file,
        split="train",
        split_scheme=split,
        seed=1337,
        config=AudioDatasetConfig(
            target_steps=target_steps,
            random_crop=True,
            video_target_frames=video_target_frames if include_video else None,
        ),
        return_waveform=include_waveform,
        return_speaker=False,
        return_video=include_video,
    )
    val_dataset = FakeAVAudioDataset(
        data_dir=data_dir,
        index_file=index_file,
        split="val",
        split_scheme=split,
        seed=1337,
        config=AudioDatasetConfig(
            target_steps=target_steps,
            random_crop=False,
            video_target_frames=video_target_frames if include_video else None,
        ),
        return_waveform=include_waveform,
        return_speaker=False,
        return_video=include_video,
    )
    test_dataset = FakeAVAudioDataset(
        data_dir=data_dir,
        index_file=index_file,
        split="test",
        split_scheme=split,
        seed=1337,
        config=AudioDatasetConfig(
            target_steps=target_steps,
            random_crop=False,
            video_target_frames=video_target_frames if include_video else None,
        ),
        return_waveform=include_waveform,
        return_speaker=False,
        return_video=include_video,
    )
    val_real_copies = max(int(val_real_copies), 0)
    val_real_indices = []
    test_real_indices = []
    if val_real_copies > 0:
        val_labels = _load_labels_for_files(val_dataset.files, index_file)
        test_labels = _load_labels_for_files(test_dataset.files, index_file)
        val_real_indices = [idx for idx, label in enumerate(val_labels) if label == 0]
        test_real_indices = [idx for idx, label in enumerate(test_labels) if label == 0]
        if val_real_indices:
            val_dataset = ValidationAugmentedDataset(
                val_dataset,
                val_real_indices,
                copies=val_real_copies,
                freq_mask=val_augment_freq_mask,
                time_mask=val_augment_time_mask,
                noise_std=val_augment_noise_std,
                shift_pct=val_augment_shift_pct,
                gain_std=val_augment_gain_std,
                mixup_prob=val_augment_mixup_prob,
                mixup_alpha=val_augment_mixup_alpha,
            )
        if test_real_indices:
            test_dataset = ValidationAugmentedDataset(
                test_dataset,
                test_real_indices,
                copies=val_real_copies,
                freq_mask=val_augment_freq_mask,
                time_mask=val_augment_time_mask,
                noise_std=val_augment_noise_std,
                shift_pct=val_augment_shift_pct,
                gain_std=val_augment_gain_std,
                mixup_prob=val_augment_mixup_prob,
                mixup_alpha=val_augment_mixup_alpha,
            )
    collator = AudioCollator(
        include_waveform=include_waveform,
        include_video=include_video,
        include_metadata=False,
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collator,
    }
    if balanced_sampler:
        sampler = _build_sampler(train_dataset, index_file)
        train_loader = DataLoader(train_dataset, shuffle=False, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_dataset, train_loader, val_loader, test_loader


def summarize_waveform_lengths(
    dataset: FakeAVAudioDataset,
    *,
    max_samples: int = 512,
) -> Dict[str, float]:
    lengths: list[int] = []
    sample_rates: Counter = Counter()
    for idx, path in enumerate(dataset.files):
        if max_samples > 0 and idx >= max_samples:
            break
        bundle = torch.load(path, map_location="cpu")
        audio = bundle.get("audio", {})
        waveform = audio.get("waveform")
        if isinstance(waveform, torch.Tensor):
            lengths.append(int(waveform.shape[-1]))
        sr = audio.get("sample_rate")
        if sr is not None:
            sample_rates[int(sr)] += 1
    if not lengths:
        return {"count": 0}
    lengths_sorted = sorted(lengths)
    count = len(lengths_sorted)
    median = lengths_sorted[count // 2]
    p10 = lengths_sorted[int(count * 0.1)]
    p90 = lengths_sorted[int(count * 0.9)]
    maximum = lengths_sorted[-1]
    minimum = lengths_sorted[0]
    sr_mode = max(sample_rates.items(), key=lambda kv: kv[1])[0] if sample_rates else 0
    return {
        "count": float(count),
        "min": float(minimum),
        "median": float(median),
        "p90": float(p90),
        "max": float(maximum),
        "p10": float(p10),
        "sample_rate": float(sr_mode),
    }


def verify_dataset_integrity(
    datasets: Sequence[FakeAVAudioDataset],
) -> Dict[str, object]:
    seen: set[Path] = set()
    missing_mel: list[str] = []
    sample_rates: Counter = Counter()
    mel_shapes: Counter = Counter()
    errors: list[tuple[str, str]] = []
    for dataset in datasets:
        for path in dataset.files:
            if path in seen:
                continue
            seen.add(path)
            try:
                bundle = torch.load(path, map_location="cpu")
            except Exception as exc:  # pragma: no cover - defensive diagnostic
                errors.append((path.name, str(exc)))
                continue
            audio = bundle.get("audio", {})
            mel_tensor = audio.get("mel")
            if not isinstance(mel_tensor, torch.Tensor):
                missing_mel.append(path.name)
            else:
                mel_shapes[str(tuple(mel_tensor.shape))] += 1
            sr = audio.get("sample_rate")
            if sr is not None:
                sample_rates[int(sr)] += 1
    summary = {
        "total_samples": len(seen),
        "missing_mel": missing_mel,
        "mel_shapes": dict(mel_shapes),
        "sample_rates": dict(sample_rates),
        "errors": errors,
    }
    if errors:
        raise RuntimeError(f"Encountered {len(errors)} errors while reading samples: {errors[:3]}...")
    if missing_mel:
        raise RuntimeError(f"{len(missing_mel)} samples are missing mel tensors; run preprocessing again.")
    return summary


def unpack_audio_batch(
    batch: Dict[str, object] | Sequence[object],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if isinstance(batch, dict):
        mels = batch["mel"]
        labels = batch["label"]
        waveform = batch.get("waveform")
        waveform_lengths = batch.get("waveform_length")
        video = batch.get("video")
    elif isinstance(batch, (list, tuple)):
        mels = batch[0]
        labels = batch[1]
        waveform = batch[2] if len(batch) > 2 else None
        waveform_lengths = batch[3] if len(batch) > 3 else None
        video = batch[4] if len(batch) > 4 else None
    else:
        raise TypeError("Unexpected batch structure.")
    if not isinstance(mels, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("Batch must provide 'mel' and 'label' tensors.")
    mels = mels.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.to(device, non_blocking=True).float()
    else:
        waveform = None
    if isinstance(waveform_lengths, torch.Tensor):
        waveform_lengths = waveform_lengths.to(device, non_blocking=True, dtype=torch.long)
    else:
        waveform_lengths = None
    if isinstance(video, torch.Tensor):
        video = video.to(device, non_blocking=True).float()
    else:
        video = None
    return mels, labels, waveform, waveform_lengths, video


def augment_class_mels(
    mels: torch.Tensor,
    labels: torch.Tensor,
    *,
    target_label: int,
    prob: float,
    freq_mask: int,
    time_mask: int,
    noise_std: float,
) -> torch.Tensor:
    if prob <= 0.0 and noise_std <= 0.0:
        return mels
    class_mask = (labels == target_label).nonzero(as_tuple=False).squeeze(1)
    if class_mask.numel() == 0:
        return mels

    augmented = mels.clone()
    max_freq = int(freq_mask)
    max_time = int(time_mask)
    prob = float(max(prob, 0.0))
    noise_std = float(max(noise_std, 0.0))

    for idx in class_mask:
        sample = augmented[idx]
        apply_aug = prob > 0.0 and torch.rand(1, device=sample.device).item() <= prob
        if apply_aug:
            if max_freq > 0:
                bins = sample.size(-2)
                width = min(max_freq, bins)
                if width > 0:
                    span = int(torch.randint(1, width + 1, (1,), device=sample.device).item())
                    start_max = bins - span
                    start = int(torch.randint(0, max(start_max + 1, 1), (1,), device=sample.device).item())
                    sample = sample.clone()
                    sample[:, start : start + span, :] = 0.0
            if max_time > 0:
                steps = sample.size(-1)
                width = min(max_time, steps)
                if width > 0:
                    span = int(torch.randint(1, width + 1, (1,), device=sample.device).item())
                    start_max = steps - span
                    start = int(torch.randint(0, max(start_max + 1, 1), (1,), device=sample.device).item())
                    sample = sample.clone()
                    sample[:, :, start : start + span] = 0.0
        if noise_std > 0.0:
            noise = torch.randn_like(sample) * noise_std
            sample = sample + noise
        augmented[idx] = sample
    return augmented


def augment_all_mels(
    mels: torch.Tensor,
    *,
    prob: float,
    freq_mask: int,
    time_mask: int,
    noise_std: float,
) -> torch.Tensor:
    if prob <= 0.0 and noise_std <= 0.0:
        return mels
    augmented = mels.clone()
    max_freq = int(freq_mask)
    max_time = int(time_mask)
    prob = float(max(prob, 0.0))
    noise_std = float(max(noise_std, 0.0))
    batch = augmented.size(0)
    for idx in range(batch):
        sample = augmented[idx]
        apply_aug = prob > 0.0 and torch.rand(1, device=sample.device).item() <= prob
        if apply_aug:
            if max_freq > 0:
                bins = sample.size(-2)
                width = min(max_freq, bins)
                if width > 0:
                    span = int(torch.randint(1, width + 1, (1,), device=sample.device).item())
                    start_max = bins - span
                    start = int(torch.randint(0, max(start_max + 1, 1), (1,), device=sample.device).item())
                    sample = sample.clone()
                    sample[:, start : start + span, :] = 0.0
            if max_time > 0:
                steps = sample.size(-1)
                width = min(max_time, steps)
                if width > 0:
                    span = int(torch.randint(1, width + 1, (1,), device=sample.device).item())
                    start_max = steps - span
                    start = int(torch.randint(0, max(start_max + 1, 1), (1,), device=sample.device).item())
                    sample = sample.clone()
                    sample[:, :, start : start + span] = 0.0
        if noise_std > 0.0:
            noise = torch.randn_like(sample) * noise_std
            sample = sample + noise
        augmented[idx] = sample
    return augmented


def _augment_single_sample(
    sample: torch.Tensor,
    *,
    freq_mask: int,
    time_mask: int,
    noise_std: float,
) -> torch.Tensor:
    if freq_mask > 0:
        bins = sample.size(-2)
        width = min(freq_mask, bins)
        if width > 0:
            span = int(torch.randint(1, width + 1, (1,), device=sample.device).item())
            start_max = bins - span
            start = int(torch.randint(0, max(start_max + 1, 1), (1,), device=sample.device).item())
            sample = sample.clone()
            sample[:, start : start + span, :] = 0.0
    if time_mask > 0:
        steps = sample.size(-1)
        width = min(time_mask, steps)
        if width > 0:
            span = int(torch.randint(1, width + 1, (1,), device=sample.device).item())
            start_max = steps - span
            start = int(torch.randint(0, max(start_max + 1, 1), (1,), device=sample.device).item())
            sample = sample.clone()
            sample[:, :, start : start + span] = 0.0
    if noise_std > 0.0:
        noise = torch.randn_like(sample) * noise_std
        sample = sample + noise
    return sample


def _apply_shift_gain(sample: torch.Tensor, *, shift_pct: float, gain_std: float) -> torch.Tensor:
    if gain_std > 0.0:
        gain = 1.0 + torch.randn(1, device=sample.device, dtype=sample.dtype).item() * gain_std
        sample = sample * gain
    if shift_pct > 0.0:
        max_shift = int(sample.size(-1) * abs(shift_pct))
        if max_shift > 0:
            shift = int(torch.randint(0, 2 * max_shift + 1, (1,), device=sample.device).item()) - max_shift
            if shift != 0:
                sample = torch.roll(sample, shifts=shift, dims=-1)
    return sample


def enrich_real_samples(
    mels: torch.Tensor,
    labels: torch.Tensor,
    *,
    gain_std: float,
    shift_pct: float,
    mixup_prob: float,
    mixup_alpha: float,
) -> torch.Tensor:
    gain_std = float(max(gain_std, 0.0))
    shift_pct = float(max(shift_pct, 0.0))
    mixup_prob = float(max(mixup_prob, 0.0))
    mixup_alpha = float(max(mixup_alpha, 1e-3))
    if gain_std <= 0.0 and shift_pct <= 0.0 and mixup_prob <= 0.0:
        return mels
    real_idx = (labels == 0).nonzero(as_tuple=False).squeeze(1)
    if real_idx.numel() == 0:
        return mels
    enriched = mels.clone()
    if gain_std > 0.0 or shift_pct > 0.0:
        for idx in real_idx:
            sample = enriched[idx]
            sample = _apply_shift_gain(sample, shift_pct=shift_pct, gain_std=gain_std)
            enriched[idx] = sample
    if mixup_prob > 0.0 and real_idx.numel() > 1:
        alpha_tensor = torch.tensor(mixup_alpha, device=mels.device, dtype=mels.dtype)
        beta_dist = Beta(alpha_tensor, alpha_tensor)
        for idx in real_idx:
            if torch.rand(1, device=mels.device).item() > mixup_prob:
                continue
            other_idx = real_idx[torch.randint(0, real_idx.numel(), (1,), device=mels.device).item()]
            if other_idx == idx:
                continue
            lam = beta_dist.sample().squeeze().clamp(0.0, 1.0)
            sample = enriched[idx]
            other_sample = mels[other_idx]
            sample = lam * sample + (1.0 - lam) * other_sample
            enriched[idx] = sample
    return enriched


def prepare_sync_inputs(
    mels: torch.Tensor,
    video: torch.Tensor | None,
    *,
    target_frames: int,
    audio_dim: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if video is None or video.size(1) <= 0:
        return None, None
    vid = video
    if vid.size(1) != target_frames:
        if vid.size(1) > target_frames:
            start = max((vid.size(1) - target_frames) // 2, 0)
            vid = vid[:, start : start + target_frames]
        else:
            pad = target_frames - vid.size(1)
            pad_before = pad // 2
            pad_after = pad - pad_before
            first = vid[:, :1].repeat(1, pad_before, 1, 1, 1) if pad_before > 0 else None
            last = vid[:, -1:].repeat(1, pad_after, 1, 1, 1) if pad_after > 0 else None
            pieces = [p for p in (first, vid, last) if p is not None]
            vid = torch.cat(pieces, dim=1)
    if mels.dim() == 4:
        mel_2d = mels.squeeze(1)
    elif mels.dim() == 3:
        mel_2d = mels
    else:
        mel_2d = mels.unsqueeze(1).squeeze(1)
    mel_2d = mel_2d.float()
    mel_resized = F.interpolate(
        mel_2d.unsqueeze(1),
        size=(max(int(audio_dim), 1), max(int(target_frames), 1)),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    audio_seq = mel_resized.permute(0, 2, 1).contiguous()
    return vid.contiguous(), audio_seq


def create_pseudo_fake_batch(
    mels: torch.Tensor,
    labels: torch.Tensor,
    waveform: torch.Tensor | None,
    waveform_lengths: torch.Tensor | None,
    *,
    prob: float,
    freq_mask: int,
    time_mask: int,
    noise_std: float,
    max_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if prob <= 0.0:
        return mels, labels, waveform, waveform_lengths
    real_indices = (labels == 0).nonzero(as_tuple=False).squeeze(1)
    if real_indices.numel() == 0:
        return mels, labels, waveform, waveform_lengths
    pseudo_mels: list[torch.Tensor] = []
    pseudo_waveforms: list[torch.Tensor] = []
    pseudo_lengths: list[int] = []
    count_limit = max_count if max_count > 0 else real_indices.numel()
    for idx in real_indices:
        if torch.rand(1, device=mels.device).item() > prob:
            continue
        sample = _augment_single_sample(
            mels[idx].clone(),
            freq_mask=freq_mask,
            time_mask=time_mask,
            noise_std=noise_std,
        )
        pseudo_mels.append(sample)
        if isinstance(waveform, torch.Tensor) and waveform.numel() > 0:
            wf = waveform[idx].clone()
            if noise_std > 0.0:
                wf = wf + torch.randn_like(wf) * noise_std * 0.5
            pseudo_waveforms.append(wf)
            pseudo_lengths.append(int(waveform_lengths[idx].item()) if isinstance(waveform_lengths, torch.Tensor) else wf.size(-1))
        if len(pseudo_mels) >= count_limit:
            break
    if not pseudo_mels:
        return mels, labels, waveform, waveform_lengths
    pseudo_mel_tensor = torch.stack(pseudo_mels, dim=0)
    new_mels = torch.cat([mels, pseudo_mel_tensor], dim=0)
    pseudo_labels = torch.ones(pseudo_mel_tensor.size(0), dtype=labels.dtype, device=labels.device)
    new_labels = torch.cat([labels, pseudo_labels], dim=0)
    if pseudo_waveforms and isinstance(waveform, torch.Tensor):
        pseudo_wave_tensor = torch.stack(pseudo_waveforms, dim=0)
        new_waveform = torch.cat([waveform, pseudo_wave_tensor], dim=0)
        if isinstance(waveform_lengths, torch.Tensor):
            pseudo_len_tensor = torch.tensor(pseudo_lengths, dtype=waveform_lengths.dtype, device=waveform_lengths.device)
            new_lengths = torch.cat([waveform_lengths, pseudo_len_tensor], dim=0)
        else:
            new_lengths = None
    else:
        new_waveform = waveform
        new_lengths = waveform_lengths
    return new_mels, new_labels, new_waveform, new_lengths


def prepare_waveform_segments(
    waveform: torch.Tensor | None,
    lengths: torch.Tensor | None,
    *,
    segment_samples: int,
    train: bool,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    if waveform is None or waveform.numel() == 0:
        return None, None
    if segment_samples <= 0:
        return waveform, lengths
    seg = max(int(segment_samples), 1)
    if lengths is None:
        lengths = torch.full(
            (waveform.size(0),),
            waveform.size(-1),
            dtype=torch.long,
            device=waveform.device,
        )
    processed = waveform.new_zeros((waveform.size(0), seg))
    out_lengths = torch.full((waveform.size(0),), seg, dtype=torch.long, device=waveform.device)
    for idx in range(waveform.size(0)):
        length = int(lengths[idx].item()) if idx < lengths.size(0) else waveform.size(-1)
        if length <= 0:
            continue
        length = min(length, waveform.size(-1))
        sample = waveform[idx, :length]
        if length >= seg:
            if train:
                max_start = length - seg
                start = int(torch.randint(0, max_start + 1, (1,), device=waveform.device).item()) if max_start > 0 else 0
            else:
                start = max((length - seg) // 2, 0)
            chunk = sample[start : start + seg]
        else:
            repeat = seg // max(length, 1) + 1
            chunk = sample.repeat(repeat)[:seg]
        processed[idx] = chunk
    return processed, out_lengths


def process_waveform_branch(
    waveform: torch.Tensor | None,
    lengths: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    mode: str,
    segment_samples: int,
    train: bool,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    if waveform is None or waveform.numel() == 0 or mode == "none":
        return None, None
    wave = waveform
    if lengths is None:
        lengths = torch.full(
            (wave.size(0),),
            wave.size(-1),
            dtype=torch.long,
            device=wave.device,
        )
    else:
        lengths = lengths.clone()
    if mode == "real_only":
        mask = (labels == 0)
    elif mode == "fake_only":
        mask = (labels == 1)
    else:
        mask = None
    if mask is not None:
        if not mask.any():
            return None, None
        disable = ~mask
        if disable.any():
            wave = wave.clone()
        wave[disable] = 0.0
        lengths[disable] = 0
    return prepare_waveform_segments(wave, lengths, segment_samples=segment_samples, train=train)


def prepare_model_waveform_inputs(
    waveform: torch.Tensor | None,
    lengths: torch.Tensor | None,
    labels: torch.Tensor,
    *,
    audio_backbone: str,
    wave_branch_mode: str,
    segment_samples: int,
    train: bool,
    random_segment: bool = False,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    if waveform is None or waveform.numel() == 0:
        return None, None
    if audio_backbone == "wavlm":
        return prepare_waveform_segments(
            waveform,
            lengths,
            segment_samples=segment_samples,
            train=train or random_segment,
        )
    return process_waveform_branch(
        waveform,
        lengths,
        labels,
        mode=wave_branch_mode,
        segment_samples=segment_samples,
        train=train or random_segment,
    )


def load_label_counts(dataset: FakeAVAudioDataset, index_file: Path | None) -> Counter:
    counts: Counter = Counter()
    name_set = {path.name for path in dataset.files}
    if index_file and index_file.exists():
        records = _read_index(index_file)
        for record in records:
            if record.get("status") != "ok":
                continue
            name = Path(str(record.get("output_path", ""))).name
            if name in name_set:
                counts[int(record.get("label", 0))] += 1
    if not counts:
        for path in dataset.files:
            bundle = torch.load(path, map_location="cpu")
            counts[int(bundle.get("label", 0))] += 1
    return counts


def compute_eer(scores: torch.Tensor, labels: torch.Tensor) -> float:
    if scores.numel() == 0 or labels.numel() == 0:
        return 0.5
    labels = labels.to(dtype=torch.float32)
    positives = labels.sum().item()
    negatives = labels.numel() - positives
    if positives == 0 or negatives == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]
    cum_pos = torch.cumsum(sorted_labels, dim=0)
    cum_neg = torch.cumsum(1.0 - sorted_labels, dim=0)
    tpr = cum_pos / positives
    fpr = cum_neg / negatives
    fnr = 1.0 - tpr
    diff = torch.abs(fpr - fnr)
    idx = torch.argmin(diff)
    eer = (fpr[idx] + fnr[idx]) * 0.5
    return float(eer.item())


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if isinstance(weight, torch.Tensor):
            weight = weight.to(device=logits.device, dtype=logits.dtype)
        ce = nn.functional.cross_entropy(logits, targets, weight=weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class ModelEma:
    def __init__(self, model: nn.Module, *, decay: float = 0.999, device: torch.device | None = None) -> None:
        self.decay = min(max(decay, 0.0), 0.9999)
        self.device = device or next(model.parameters()).device
        self.ema_model = copy.deepcopy(model).to(self.device)
        self.num_updates = 0
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    @property
    def ready(self) -> bool:
        return self.num_updates > 0

    def update(self, model: nn.Module) -> None:
        decay = self.decay
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters(), strict=False):
                if not ema_param.data.dtype.is_floating_point:
                    ema_param.data.copy_(model_param.data)
                    continue
                ema_param.data.mul_(decay).add_(model_param.data, alpha=1.0 - decay)
            for ema_buffer, model_buffer in zip(self.ema_model.buffers(), model.buffers(), strict=False):
                ema_buffer.copy_(model_buffer)
        self.num_updates += 1

    def state_dict(self) -> Dict[str, object]:
        return {"ema_state": self.ema_model.state_dict(), "num_updates": self.num_updates}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        ema_state = state.get("ema_state")
        if ema_state is not None:
            self.ema_model.load_state_dict(ema_state)
        self.num_updates = int(state.get("num_updates", 0))

    def load_from_model(self, model: nn.Module) -> None:
        self.ema_model.load_state_dict(model.state_dict())
        self.num_updates = 1


class AudioSyncFusionHead(nn.Module):
    def __init__(
        self,
        audio_dim: int,
        sync_hidden_dim: int,
        *,
        dropout: float = 0.2,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.audio_dim = audio_dim
        self.sync_hidden = sync_hidden_dim
        self.sync_proj = nn.Linear(sync_hidden_dim, audio_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=audio_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, audio_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(audio_dim, 2),
        )

    def forward(self, audio_embed: torch.Tensor, sync_joint: torch.Tensor | None) -> torch.Tensor:
        if sync_joint is None:
            fused = audio_embed
        else:
            if sync_joint.dim() == 2:
                segs = sync_joint.size(1) // self.sync_hidden
                if segs <= 0:
                    raise ValueError("Invalid sync feature dimension.")
                sync_seq = sync_joint.view(sync_joint.size(0), segs, self.sync_hidden)
            else:
                sync_seq = sync_joint
            sync_proj = self.sync_proj(sync_seq)
            query = audio_embed.unsqueeze(1)
            pad_mask = torch.zeros(sync_proj.size(0), sync_proj.size(1), dtype=torch.bool, device=sync_proj.device)
            attn_out, _ = self.attn(
                query,
                sync_proj,
                sync_proj,
                key_padding_mask=pad_mask,
                attn_mask=None,
            )
            fused = (query + attn_out).squeeze(1)
        return self.ffn(fused)


class RealCenterLoss(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.center = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features is None or features.numel() == 0:
            return features.new_tensor(0.0)
        mask = (labels == 0)
        if not mask.any():
            return features.new_tensor(0.0)
        real_feats = features[mask]
        if real_feats.numel() == 0:
            return features.new_tensor(0.0)
        diff = real_feats - self.center
        loss = 0.5 * (diff.pow(2).sum(dim=1)).mean()
        return loss


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, scale: float = 30.0, margin: float = 0.2) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb_norm = nn.functional.normalize(embeddings)
        weight_norm = nn.functional.normalize(self.weight)
        cosine = nn.functional.linear(emb_norm, weight_norm)
        if labels.dim() == 1:
            labels = labels.view(-1, 1)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.long(), 1.0)
        logits = cosine - one_hot * self.margin
        logits = logits * self.scale
        return logits


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    audio_backbone: str,
    wave_branch_mode: str,
    wave_segment_samples: int,
    eval_wave_segments: int,
    eval_augment_passes: int,
    eval_augment_prob: float,
    eval_freq_mask: int,
    eval_time_mask: int,
    eval_noise_std: float,
    arcface_head: ArcMarginProduct | None = None,
    sync_model: SyncModule | None = None,
    sync_head: nn.Module | None = None,
    sync_feature_dim: int = 0,
    sync_target_frames: int = 16,
    sync_audio_dim: int = 64,
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    tp = fp = tn = fn = 0.0
    score_buffer: list[torch.Tensor] = []
    label_buffer: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            mels, labels, waveform, waveform_lengths, video = unpack_audio_batch(batch, device=device)
            eval_segments = max(int(eval_wave_segments), 1)
            augment_passes = max(int(eval_augment_passes), 1)
            augment_active = augment_passes > 1 or eval_augment_prob > 0.0 or eval_noise_std > 0.0

            logits_accum: torch.Tensor | None = None
            passes = 0
            for _ in range(augment_passes):
                if augment_active:
                    mels_pass = augment_all_mels(
                        mels,
                        prob=float(eval_augment_prob),
                        freq_mask=int(eval_freq_mask),
                        time_mask=int(eval_time_mask),
                        noise_std=float(eval_noise_std),
                    )
                else:
                    mels_pass = mels
                sync_features_pass: torch.Tensor | None = None
                if sync_model is not None and sync_head is not None and video is not None:
                    sync_video_input, sync_audio_seq = prepare_sync_inputs(
                        mels_pass,
                        video,
                        target_frames=sync_target_frames,
                        audio_dim=sync_audio_dim,
                    )
                    if sync_video_input is not None and sync_audio_seq is not None:
                        sync_features_pass, _ = sync_model(video=sync_video_input, audio_seq=sync_audio_seq)
                if eval_segments > 1 and waveform is not None and waveform.numel() > 0:
                    logits_sum: torch.Tensor | None = None
                    embed_sum: torch.Tensor | None = None
                    for _ in range(eval_segments):
                        waveform_proc, waveform_len_proc = prepare_model_waveform_inputs(
                            waveform,
                            waveform_lengths,
                            labels,
                            audio_backbone=audio_backbone,
                            wave_branch_mode=wave_branch_mode,
                            segment_samples=wave_segment_samples,
                            train=False,
                            random_segment=True,
                        )
                        logits_iter, _, embed_iter = model(mels_pass, waveform=waveform_proc, waveform_lengths=waveform_len_proc)
                        if arcface_head is not None:
                            logits_iter = arcface_head(embed_iter, labels)
                        logits_sum = logits_iter if logits_sum is None else logits_sum + logits_iter
                        embed_sum = embed_iter if embed_sum is None else embed_sum + embed_iter
                    pass_logits = logits_sum / float(eval_segments)
                    embed = embed_sum / float(eval_segments) if embed_sum is not None else None
                else:
                    waveform_proc, waveform_len_proc = prepare_model_waveform_inputs(
                        waveform,
                        waveform_lengths,
                        labels,
                        audio_backbone=audio_backbone,
                        wave_branch_mode=wave_branch_mode,
                        segment_samples=wave_segment_samples,
                        train=False,
                    )
                    pass_logits, _, embed = model(mels_pass, waveform=waveform_proc, waveform_lengths=waveform_len_proc)
                    if arcface_head is not None:
                        pass_logits = arcface_head(embed, labels)
                if sync_model is not None and sync_head is not None:
                    sync_features = None
                    sync_video_input, sync_audio_seq = prepare_sync_inputs(
                        mels_pass,
                        video,
                        target_frames=sync_target_frames,
                        audio_dim=sync_audio_dim,
                    )
                    if sync_video_input is not None and sync_audio_seq is not None:
                        sync_features, _ = sync_model(video=sync_video_input, audio_seq=sync_audio_seq)
                    if sync_features is None and sync_feature_dim > 0:
                        sync_features = pass_logits.new_zeros((pass_logits.size(0), sync_feature_dim))
                    if sync_features is not None:
                        if embed is None:
                            raise RuntimeError("Sync fusion requires backbone embeddings.")
                        if sync_features.size(0) != embed.size(0):
                            if sync_features.size(0) < embed.size(0):
                                pad = embed.size(0) - sync_features.size(0)
                                sync_features = torch.cat(
                                    [sync_features, sync_features.new_zeros((pad, sync_features.size(1)))],
                                    dim=0,
                                )
                            else:
                                sync_features = sync_features[: embed.size(0)]
                        fused_in = torch.cat([embed, sync_features], dim=1)
                        pass_logits = sync_head(
                            embed if embed is not None else fused_in.squeeze(1),
                            sync_features_pass if sync_features_pass is not None else None,
                        )
                if sync_head is not None:
                    if embed is None:
                        raise RuntimeError("Sync fusion requires backbone embeddings.")
                    sync_feat = sync_features_pass
                    if sync_feat is None and sync_feature_dim > 0:
                        sync_feat = embed.new_zeros((embed.size(0), sync_feature_dim * 3))
                    pass_logits = sync_head(embed, sync_feat if sync_feat is not None else None)
                logits_accum = pass_logits if logits_accum is None else logits_accum + pass_logits
                passes += 1
            logits = logits_accum / float(max(passes, 1))
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            total_samples += labels.size(0)
            scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
            score_buffer.append(scores)
            label_buffer.append(labels.detach().cpu())
    avg_loss = total_loss / max(total_samples, 1)
    if score_buffer:
        all_scores = torch.cat(score_buffer, dim=0)
        all_labels = torch.cat(label_buffer, dim=0)
        eer = compute_eer(all_scores, all_labels)
    else:
        eer = 0.5
    stats = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    return avg_loss, eer, stats


def build_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
) -> optim.lr_scheduler.LambdaLR:
    warmup_epochs = max(int(warmup_epochs), 0)
    total_epochs = max(int(total_epochs), 1)

    def lr_lambda(epoch: int) -> float:
        current = epoch + 1
        if warmup_epochs > 0 and current <= warmup_epochs:
            return current / warmup_epochs
        progress = (current - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train() -> None:
    args = parse_args()
    if args.device.lower().startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; switching device to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    audio_backbone = args.audio_backbone.lower()
    wave_branch_mode = args.wave_branch_mode
    if audio_backbone == "wavlm":
        wave_branch_mode = "all"
    wave_branch_enabled = wave_branch_mode != "none"
    include_waveform = wave_branch_enabled or audio_backbone == "wavlm"
    include_video_for_sync = bool(args.use_sync_fusion)
    wave_segment_samples = max(int(args.wave_segment_samples), 0)
    real_aug_start = max(int(args.real_augment_start), 1)
    global_aug_active = (
        (args.global_augment_prob > 0.0 and (args.global_freq_mask > 0 or args.global_time_mask > 0))
        or args.global_noise_std > 0.0
    )
    train_wave_segments = max(int(args.train_wave_segments), 1)
    grad_accum_steps = max(int(args.grad_accum_steps), 1)
    pseudo_fake_start_epoch = max(int(args.pseudo_fake_start_epoch), 1)
    pseudo_fake_ramp_epochs = max(int(args.pseudo_fake_ramp_epochs), 0)
    real_margin_weight = max(float(args.real_margin_weight), 0.0)
    real_margin_value = float(args.real_margin_value)
    real_margin_start_epoch = max(int(args.real_margin_start_epoch), 1)
    real_margin_warmup_epochs = max(int(args.real_margin_warmup_epochs), 0)
    center_loss_start_epoch = max(int(args.center_loss_start_epoch), 1)
    center_loss_warmup_epochs = max(int(args.center_loss_warmup_epochs), 0)
    arcface_warmup_epochs = (
        max(int(args.arcface_warmup_epochs), 0) if args.use_arcface_head and audio_backbone == "wavlm" else 0
    )
    train_dataset, train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        index_file=args.index_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_steps=args.target_steps,
        balanced_sampler=args.balanced_sampler,
        include_waveform=include_waveform,
        include_video=include_video_for_sync,
        video_target_frames=int(args.sync_target_frames) if include_video_for_sync else None,
        val_real_copies=int(args.val_real_copies),
        val_augment_freq_mask=int(args.val_augment_freq_mask),
        val_augment_time_mask=int(args.val_augment_time_mask),
        val_augment_noise_std=float(args.val_augment_noise_std),
        val_augment_shift_pct=float(args.val_augment_shift_pct),
        val_augment_gain_std=float(args.val_augment_gain_std),
        val_augment_mixup_prob=float(args.val_augment_mixup_prob),
        val_augment_mixup_alpha=float(args.val_augment_mixup_alpha),
    )
    wave_stats = summarize_waveform_lengths(train_dataset, max_samples=512) if include_waveform else {}
    if wave_stats.get("count", 0) > 0:
        sr = wave_stats.get("sample_rate", 0.0) or 16000.0
        scale = 1.0 / sr
        print(
            f"Waveform length stats (samples -> seconds @ {int(sr)}Hz): "
            f"min={wave_stats['min']:.0f}({wave_stats['min'] * scale:.2f}s), "
            f"median={wave_stats['median']:.0f}({wave_stats['median'] * scale:.2f}s), "
            f"p90={wave_stats['p90']:.0f}({wave_stats['p90'] * scale:.2f}s), "
            f"max={wave_stats['max']:.0f}({wave_stats['max'] * scale:.2f}s)"
        )

    if args.verify_dataset:
        summary = verify_dataset_integrity(
            [train_dataset]
            + [loader.dataset for loader in (val_loader, test_loader) if isinstance(loader.dataset, FakeAVAudioDataset)]
        )
        tqdm.write(
            f"Verified {summary['total_samples']} processed samples. "
            f"Mel shapes: {summary['mel_shapes']} Sample rates: {summary['sample_rates']}"
        )

    manual_pos_weight = args.positive_class_weight
    if manual_pos_weight is not None and manual_pos_weight <= 0:
        raise ValueError("positive_class_weight must be > 0 if specified.")
    if args.class_weights or manual_pos_weight is not None:
        if manual_pos_weight is None:
            counts = load_label_counts(train_dataset, args.index_file)
            total = sum(counts.values())
            weight0 = total / counts.get(0, 1)
            weight1 = total / counts.get(1, 1)
            print(f"Class counts: {counts}, weights (real/fake): [{weight0:.2f}, {weight1:.2f}]")
        else:
            weight0 = 1.0
            weight1 = float(manual_pos_weight)
            print(f"Using manual class weights (real/fake): [{weight0:.2f}, {weight1:.2f}]")
        class_weights = torch.tensor([weight0, weight1], dtype=torch.float32, device=device)
    else:
        class_weights = None

    if args.focal_loss:
        criterion = FocalLoss(
            gamma=float(args.focal_gamma),
            weight=class_weights,
        )
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    resume_base_epoch = 0
    if audio_backbone == "aasist":
        model = AASISTLite(
            num_classes=2,
            mel_bins=int(args.mel_bins),
            top_k=int(args.top_k),
        ).to(device)
        param_groups = model.parameter_groups()
        optim_groups = []
        if param_groups["backbone"]:
            optim_groups.append(
                {"params": param_groups["backbone"], "lr": float(args.lr), "weight_decay": float(args.weight_decay), "name": "backbone"}
            )
        if param_groups["head"]:
            optim_groups.append(
                {"params": param_groups["head"], "lr": float(args.lr), "weight_decay": float(args.weight_decay), "name": "head"}
            )
        if param_groups["front"]:
            optim_groups.append(
                {"params": param_groups["front"], "lr": float(args.front_lr), "weight_decay": float(args.weight_decay), "name": "front"}
            )
        optimizer = optim.AdamW(optim_groups)
    else:
        wavlm_cfg = WavLMConfig(
            model_name=args.wavlm_model_name,
            num_classes=2,
            dropout=float(args.wavlm_dropout),
            train_backbone=bool(args.wavlm_trainable),
            unfreeze_layers=int(args.wavlm_unfreeze_layers),
            local_files_only=bool(args.hf_local_files_only),
        )
        model = WavLMClassifier(wavlm_cfg).to(device)
        param_groups = model.parameter_groups()
        optim_groups = []
        backbone_lr = float(args.wavlm_backbone_lr)
        if param_groups["backbone"]:
            optim_groups.append(
                {"params": param_groups["backbone"], "lr": backbone_lr, "weight_decay": float(args.weight_decay), "name": "backbone"}
            )
        if param_groups["head"]:
            optim_groups.append(
                {"params": param_groups["head"], "lr": float(args.lr), "weight_decay": float(args.weight_decay), "name": "head"}
            )
        if not optim_groups:
            raise RuntimeError("No trainable parameters found for WavLMClassifier.")
        optimizer = optim.AdamW(optim_groups)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    sync_model: SyncModule | None = None
    sync_fusion_head: AudioSyncFusionHead | None = None
    sync_feature_dim = 0
    sync_target_frames = max(int(args.sync_target_frames), 1)
    sync_audio_dim = max(int(args.sync_audio_dim), 1)
    sync_trainable = bool(args.sync_trainable)
    sync_distill_weight = float(args.sync_distill_weight)
    sync_distill_temp = float(max(args.sync_distill_temp, 1e-6))
    if args.use_sync_fusion:
        if args.sync_checkpoint is None or not args.sync_checkpoint.exists():
            raise FileNotFoundError("Sync fusion enabled but --sync-checkpoint is missing.")
        if args.sync_vit_path is None or not args.sync_vit_path.exists():
            raise FileNotFoundError("Sync fusion enabled but --sync-vit-path is missing.")
        sync_model = SyncModule(
            vit_path=args.sync_vit_path,
            audio_dim=sync_audio_dim,
            transformer_heads=int(args.sync_transformer_heads),
            dropout=float(args.sync_fusion_dropout),
            vit_unfreeze_layers=0,
            temporal_layers=int(args.sync_temporal_layers),
        ).to(device)
        torch.serialization.add_safe_globals([Path, PureWindowsPath])
        sync_ckpt = torch.load(args.sync_checkpoint, map_location="cpu", weights_only=False)
        sync_state = sync_ckpt.get("model_state") or sync_ckpt
        sync_model.load_state_dict(sync_state, strict=False)
        if sync_trainable:
            tunable_params: list[nn.Parameter] = []
            for module in (sync_model.temporal_encoder, sync_model.sync_head, sync_model.audio_proj, sync_model.video_proj):
                for param in module.parameters():
                    param.requires_grad_(True)
                    tunable_params.append(param)
            for param in sync_model.vit.parameters():
                param.requires_grad_(False)
            if tunable_params:
                optimizer.add_param_group(
                    {
                        "params": tunable_params,
                        "lr": float(args.sync_lr),
                        "weight_decay": float(args.weight_decay),
                        "name": "sync_module",
                    }
                )
            sync_model.train()
        else:
            sync_model.eval()
            for param in sync_model.parameters():
                param.requires_grad_(False)
        sync_feature_dim = sync_model.hidden_dim
        sync_fusion_head = AudioSyncFusionHead(
            audio_dim=model.hidden,
            sync_hidden_dim=sync_model.hidden_dim,
            dropout=float(args.sync_fusion_dropout),
            num_heads=int(args.sync_transformer_heads),
        ).to(device)
        optimizer.add_param_group(
            {
                "params": sync_fusion_head.parameters(),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "name": "sync_fusion",
            }
        )
    ema = ModelEma(model, decay=float(args.ema_decay), device=device) if args.use_ema else None
    ema_start_epoch = max(int(args.ema_start_epoch), 1)
    center_loss_weight = max(float(args.center_loss_weight), 0.0)
    center_loss_module: RealCenterLoss | None = None
    center_optimizer: optim.Optimizer | None = None
    if args.use_center_loss:
        feature_dim = getattr(model, "hidden", None)
        if feature_dim is None:
            raise ValueError("Center loss requires model.hidden dimension to be defined.")
        center_loss_module = RealCenterLoss(int(feature_dim)).to(device)
        center_optimizer = optim.SGD(center_loss_module.parameters(), lr=float(args.center_loss_lr))
    arcface_head: ArcMarginProduct | None = None
    if args.use_arcface_head:
        if audio_backbone != "wavlm":
            raise ValueError("ArcFace head is only supported with the WavLM backbone.")
        feature_dim = getattr(model, "hidden", None)
        if feature_dim is None:
            raise ValueError("WavLMClassifier does not expose hidden size for ArcFace.")
        arcface_head = ArcMarginProduct(
            feature_dim,
            2,
            scale=float(args.arcface_scale),
            margin=float(args.arcface_margin),
        ).to(device)
        optimizer.add_param_group(
            {
                "params": arcface_head.parameters(),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "name": "arcface",
            }
        )
    backbone_train_enabled = bool(args.wavlm_trainable)
    if audio_backbone == "wavlm":
        model.set_backbone_trainable(bool(args.wavlm_trainable), int(args.wavlm_unfreeze_layers))
        if args.use_arcface_head and arcface_warmup_epochs > 0:
            model.set_backbone_trainable(False, 0)
            backbone_train_enabled = False
            for group in optimizer.param_groups:
                if group.get("name") == "backbone":
                    group["lr"] = 0.0

    if args.resume_from and args.resume_from.exists():
        torch.serialization.add_safe_globals([Path, PureWindowsPath])
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state") or checkpoint
        model.load_state_dict(state)
        resume_base_epoch = int(checkpoint.get("epoch", 0))
        if args.resume_optimizer and "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if arcface_head is not None and "arcface_state" in checkpoint and checkpoint["arcface_state"] is not None:
            arcface_head.load_state_dict(checkpoint["arcface_state"])
        if ema is not None and "ema_state" in checkpoint and checkpoint["ema_state"] is not None:
            ema.load_state_dict(checkpoint["ema_state"])
        if center_loss_module is not None and "center_state" in checkpoint and checkpoint["center_state"] is not None:
            center_loss_module.load_state_dict(checkpoint["center_state"])
        if sync_fusion_head is not None and "sync_head_state" in checkpoint and checkpoint["sync_head_state"] is not None:
            try:
                sync_fusion_head.load_state_dict(checkpoint["sync_head_state"], strict=True)
            except RuntimeError as exc:
                print(
                    f"Warning: could not load legacy sync head weights ({exc}). "
                    "Continuing with newly initialised fusion head."
                )
        if sync_trainable and sync_model is not None and "sync_module_state" in checkpoint and checkpoint["sync_module_state"] is not None:
            sync_model.load_state_dict(checkpoint["sync_module_state"], strict=False)
        if audio_backbone == "wavlm" and args.use_arcface_head and not backbone_train_enabled:
            if arcface_warmup_epochs <= 0 or resume_base_epoch >= arcface_warmup_epochs:
                model.set_backbone_trainable(True, int(args.wavlm_unfreeze_layers))
                backbone_train_enabled = True
                for group in optimizer.param_groups:
                    if group.get("name") == "backbone":
                        group["lr"] = float(args.wavlm_backbone_lr)
    elif args.resume_from:
        print(f"Warning: resume checkpoint {args.resume_from} not found.")
    if args.scheduler == "cosine":
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_epochs=int(args.epochs),
            warmup_epochs=int(args.warmup_epochs),
        )
        scheduler_plateau = False
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(args.plateau_factor),
            patience=int(args.plateau_patience),
            min_lr=float(args.plateau_min_lr),
        )
        scheduler_plateau = True

    freeze_stages = max(int(args.freeze_stages), 0) if audio_backbone == "aasist" else 0
    freeze_epochs = max(int(args.freeze_epochs), 0) if audio_backbone == "aasist" else 0
    if freeze_stages > 0 and hasattr(model, "freeze_stages"):
        model.freeze_stages(freeze_stages)

    best_val_eer = float("inf")
    if args.resume_from and args.resume_from.exists():
        best_val_eer = checkpoint.get("val_eer", best_val_eer)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_sync_head_state: Optional[Dict[str, torch.Tensor]] = None
    best_sync_model_state: Optional[Dict[str, torch.Tensor]] = None
    early_stop_patience = max(int(args.early_stop_patience), 0)
    target_val_eer = max(float(args.target_val_eer), 0.0)
    epochs_no_improve = 0
    max_grad_norm = float(args.max_grad_norm)

    total_epochs = int(args.epochs)
    val_eer: float | None = None
    val_loss_value: float | None = None
    for epoch_idx in range(total_epochs):
        if audio_backbone == "wavlm" and args.use_arcface_head and not backbone_train_enabled:
            if arcface_warmup_epochs <= 0 or epoch_idx >= arcface_warmup_epochs:
                model.set_backbone_trainable(True, int(args.wavlm_unfreeze_layers))
                backbone_train_enabled = True
                for group in optimizer.param_groups:
                    if group.get("name") == "backbone":
                        group["lr"] = float(args.wavlm_backbone_lr)
        epoch = resume_base_epoch + epoch_idx + 1
        if audio_backbone == "aasist" and freeze_stages > 0 and epoch == freeze_epochs + 1:
            model.freeze_stages(0)
            for group in optimizer.param_groups:
                if group.get("name") == "front":
                    group["lr"] = float(args.front_lr)

        if sync_model is not None:
            if sync_trainable:
                sync_model.train()
            else:
                sync_model.eval()

        model.train()
        running_loss = 0.0
        total_samples = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        optimizer.zero_grad(set_to_none=True)
        if center_optimizer is not None:
            center_optimizer.zero_grad(set_to_none=True)
        accum_counter = 0
        if center_loss_module is not None and center_loss_weight > 0.0 and epoch >= center_loss_start_epoch:
            if center_loss_warmup_epochs > 0:
                center_progress = min(
                    1.0,
                    max(0.0, (epoch - center_loss_start_epoch + 1) / float(center_loss_warmup_epochs)),
                )
            else:
                center_progress = 1.0
        else:
            center_progress = 0.0
        center_weight_epoch = center_loss_weight * center_progress
        if real_margin_weight > 0.0 and epoch >= real_margin_start_epoch:
            if real_margin_warmup_epochs > 0:
                margin_progress = min(
                    1.0,
                    max(0.0, (epoch - real_margin_start_epoch + 1) / float(real_margin_warmup_epochs)),
                )
            else:
                margin_progress = 1.0
        else:
            margin_progress = 0.0
        margin_weight_epoch = real_margin_weight * margin_progress
        if args.pseudo_fake_prob > 0.0 and epoch >= pseudo_fake_start_epoch:
            if pseudo_fake_ramp_epochs > 0:
                ramp_progress = min(
                    1.0,
                    max(0.0, (epoch - pseudo_fake_start_epoch + 1) / float(pseudo_fake_ramp_epochs)),
                )
            else:
                ramp_progress = 1.0
            current_pseudo_prob = float(args.pseudo_fake_prob) * ramp_progress
            if args.pseudo_fake_max > 0:
                current_pseudo_max = max(1, int(round(args.pseudo_fake_max * ramp_progress)))
            else:
                current_pseudo_max = 0
        else:
            current_pseudo_prob = 0.0
            current_pseudo_max = 0
        for batch in progress:
            mels, labels, waveform, waveform_lengths, video = unpack_audio_batch(batch, device=device)
            if global_aug_active:
                mels = augment_all_mels(
                    mels,
                    prob=float(args.global_augment_prob),
                    freq_mask=int(args.global_freq_mask),
                    time_mask=int(args.global_time_mask),
                    noise_std=float(args.global_noise_std),
                )
            use_real_aug = (args.real_augment_prob > 0.0 or args.real_noise_std > 0.0) and epoch >= real_aug_start
            if use_real_aug:
                mels = augment_class_mels(
                    mels,
                    labels,
                    target_label=0,
                    prob=float(args.real_augment_prob),
                    freq_mask=int(args.real_freq_mask),
                    time_mask=int(args.real_time_mask),
                    noise_std=float(args.real_noise_std),
                )
            use_real_enrich = (
                epoch >= real_aug_start
                and (
                    args.real_gain_std > 0.0
                    or args.real_shift_pct > 0.0
                    or args.real_mixup_prob > 0.0
                )
            )
            if use_real_enrich:
                mels = enrich_real_samples(
                    mels,
                    labels,
                    gain_std=float(args.real_gain_std),
                    shift_pct=float(args.real_shift_pct),
                    mixup_prob=float(args.real_mixup_prob),
                    mixup_alpha=float(args.real_mixup_alpha),
                )
            sync_joint = None
            sync_teacher_logits = None
            if sync_model is not None and sync_fusion_head is not None and video is not None:
                sync_video_input, sync_audio_seq = prepare_sync_inputs(
                    mels,
                    video,
                    target_frames=sync_target_frames,
                    audio_dim=sync_audio_dim,
                )
                if sync_video_input is not None and sync_audio_seq is not None:
                    if sync_trainable:
                        sync_joint, sync_logits_raw = sync_model(video=sync_video_input, audio_seq=sync_audio_seq)
                    else:
                        with torch.no_grad():
                            sync_joint, sync_logits_raw = sync_model(video=sync_video_input, audio_seq=sync_audio_seq)
                    sync_teacher_logits = sync_logits_raw.detach()
            use_fake_aug = args.fake_augment_prob > 0.0 or args.fake_noise_std > 0.0
            if use_fake_aug:
                mels = augment_class_mels(
                    mels,
                    labels,
                    target_label=1,
                    prob=float(args.fake_augment_prob),
                    freq_mask=int(args.fake_freq_mask),
                    time_mask=int(args.fake_time_mask),
                    noise_std=float(args.fake_noise_std),
                )
            if current_pseudo_prob > 0.0:
                mels, labels, waveform, waveform_lengths = create_pseudo_fake_batch(
                    mels,
                    labels,
                    waveform,
                    waveform_lengths,
                    prob=float(current_pseudo_prob),
                    freq_mask=int(args.pseudo_fake_freq_mask),
                    time_mask=int(args.pseudo_fake_time_mask),
                    noise_std=float(args.pseudo_fake_noise),
                    max_count=int(current_pseudo_max),
                )
                if sync_joint is not None and sync_joint.size(0) < mels.size(0):
                    pad = mels.size(0) - sync_joint.size(0)
                    sync_joint = torch.cat(
                        [sync_joint, sync_joint.new_zeros((pad, sync_joint.size(1)))],
                        dim=0,
                    )
                if sync_teacher_logits is not None and sync_teacher_logits.size(0) < mels.size(0):
                    pad = mels.size(0) - sync_teacher_logits.size(0)
                    sync_teacher_logits = torch.cat(
                        [sync_teacher_logits, sync_teacher_logits.new_zeros((pad, sync_teacher_logits.size(1)))],
                        dim=0,
                    )
            if train_wave_segments > 1 and waveform is not None and waveform.numel() > 0:
                logits_sum: torch.Tensor | None = None
                embed_sum: torch.Tensor | None = None
                for _ in range(train_wave_segments):
                    waveform_proc, waveform_len_proc = prepare_model_waveform_inputs(
                        waveform,
                        waveform_lengths,
                        labels,
                        audio_backbone=audio_backbone,
                        wave_branch_mode=wave_branch_mode,
                        segment_samples=wave_segment_samples,
                        train=True,
                        random_segment=True,
                    )
                    with torch.amp.autocast("cuda") if scaler is not None else contextlib.nullcontext():
                        logits_iter, _, embed_iter = model(mels, waveform=waveform_proc, waveform_lengths=waveform_len_proc)
                        if arcface_head is not None:
                            logits_iter = arcface_head(embed_iter, labels)
                    logits_sum = logits_iter if logits_sum is None else logits_sum + logits_iter
                    embed_sum = embed_iter if embed_sum is None else embed_sum + embed_iter
                logits = logits_sum / float(train_wave_segments)
                embed = embed_sum / float(train_wave_segments) if embed_sum is not None else None
            else:
                waveform_proc, waveform_len_proc = prepare_model_waveform_inputs(
                    waveform,
                    waveform_lengths,
                    labels,
                    audio_backbone=audio_backbone,
                    wave_branch_mode=wave_branch_mode,
                    segment_samples=wave_segment_samples,
                    train=True,
                )
                with torch.amp.autocast("cuda") if scaler is not None else contextlib.nullcontext():
                    logits, _, embed = model(mels, waveform=waveform_proc, waveform_lengths=waveform_len_proc)
                    if arcface_head is not None:
                        logits = arcface_head(embed, labels)
            if sync_fusion_head is not None:
                if embed is None:
                    raise RuntimeError("Sync fusion requires backbone embeddings.")
                sync_features = sync_joint
                if sync_features is None and sync_feature_dim > 0:
                    sync_features = embed.new_zeros((embed.size(0), sync_feature_dim * 3))
                logits = sync_fusion_head(embed, sync_features)
            loss_raw = criterion(logits, labels)
            if sync_teacher_logits is not None and sync_distill_weight > 0.0:
                teacher_prob = torch.softmax(sync_teacher_logits / sync_distill_temp, dim=1)
                student_log_prob = torch.log_softmax(logits / sync_distill_temp, dim=1)
                distill = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (sync_distill_temp**2)
                loss_raw = loss_raw + sync_distill_weight * distill
            if center_loss_module is not None and embed is not None and center_weight_epoch > 0.0:
                center_penalty = center_loss_module(embed, labels) * center_weight_epoch
                loss_raw = loss_raw + center_penalty
            if margin_weight_epoch > 0.0:
                real_mask = (labels == 0)
                if real_mask.any():
                    real_logits = logits[real_mask]
                    if real_logits.size(0) > 0:
                        margin = real_logits[:, 0] - real_logits[:, 1]
                        hinge = torch.clamp(real_margin_value - margin, min=0.0)
                        loss_raw = loss_raw + margin_weight_epoch * hinge.mean()
            loss = loss_raw / grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_counter += 1
            running_loss += loss_raw.item() * labels.size(0)
            if accum_counter % grad_accum_steps == 0:
                if max_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        if center_optimizer is not None:
                            scaler.unscale_(center_optimizer)
                    clip_grad_norm_(model.parameters(), max_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                    if center_optimizer is not None:
                        center_optimizer.step()
                else:
                    optimizer.step()
                    if center_optimizer is not None:
                        center_optimizer.step()
                if ema is not None and epoch >= ema_start_epoch:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)
                if center_optimizer is not None:
                    center_optimizer.zero_grad(set_to_none=True)
            total_samples += labels.size(0)
            progress.set_postfix({"loss": running_loss / max(total_samples, 1)})
        if accum_counter % grad_accum_steps != 0:
            if max_grad_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    if center_optimizer is not None:
                        scaler.unscale_(center_optimizer)
                clip_grad_norm_(model.parameters(), max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
                if center_optimizer is not None:
                    center_optimizer.step()
            else:
                optimizer.step()
                if center_optimizer is not None:
                    center_optimizer.step()
            if ema is not None and epoch >= ema_start_epoch:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
            if center_optimizer is not None:
                center_optimizer.zero_grad(set_to_none=True)
        if scheduler_plateau:
            metric = val_eer if val_eer is not None else (val_loss_value if val_loss_value is not None else 0.0)
            scheduler.step(metric)
        else:
            scheduler.step()

        eval_model = ema.ema_model if ema is not None and ema.ready else model
        sync_prev_mode = None
        if sync_model is not None:
            sync_prev_mode = sync_model.training
            sync_model.eval()
        val_loss, val_eer, val_stats = evaluate(
            eval_model,
            val_loader,
            criterion,
            device,
            audio_backbone=audio_backbone,
            wave_branch_mode=wave_branch_mode,
            wave_segment_samples=wave_segment_samples,
            eval_wave_segments=int(args.eval_wave_segments),
            eval_augment_passes=int(args.eval_augment_passes),
            eval_augment_prob=float(args.eval_augment_prob),
            eval_freq_mask=int(args.eval_freq_mask),
            eval_time_mask=int(args.eval_time_mask),
            eval_noise_std=float(args.eval_noise_std),
            arcface_head=arcface_head,
            sync_model=sync_model,
            sync_head=sync_fusion_head,
            sync_feature_dim=sync_feature_dim,
            sync_target_frames=sync_target_frames,
            sync_audio_dim=sync_audio_dim,
        )
        val_loss_value = val_loss
        tqdm.write(
            f"Epoch {epoch}: val_loss={val_loss:.4f}, val_eer={val_eer:.4f}, "
            f"TP={val_stats['tp']:.0f}, FP={val_stats['fp']:.0f}, TN={val_stats['tn']:.0f}, FN={val_stats['fn']:.0f}"
        )
        if sync_model is not None and sync_prev_mode is not None:
            sync_model.train(sync_prev_mode)

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            source_model = ema.ema_model if ema is not None and ema.ready else model
            best_state = copy.deepcopy(source_model.state_dict())
            if sync_fusion_head is not None:
                best_sync_head_state = copy.deepcopy(sync_fusion_head.state_dict())
            if sync_trainable and sync_model is not None:
                best_sync_model_state = copy.deepcopy(sync_model.state_dict())
            save_payload = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_eer": val_eer,
                "args": vars(args),
            }
            if arcface_head is not None:
                save_payload["arcface_state"] = arcface_head.state_dict()
            if ema is not None:
                save_payload["ema_state"] = ema.state_dict()
            if center_loss_module is not None:
                save_payload["center_state"] = center_loss_module.state_dict()
            if sync_fusion_head is not None:
                save_payload["sync_head_state"] = sync_fusion_head.state_dict()
            if sync_trainable and sync_model is not None:
                save_payload["sync_module_state"] = sync_model.state_dict()
            torch.save(save_payload, args.save_path)
            tqdm.write(f"Saved new best model to {args.save_path}")
            epochs_no_improve = 0
            if target_val_eer > 0.0 and best_val_eer <= target_val_eer:
                tqdm.write(
                    f"Target val EER {target_val_eer:.4f} reached (current {best_val_eer:.4f}). Stopping training."
                )
                break
        else:
            epochs_no_improve += 1
            if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                tqdm.write(
                    f"Early stopping triggered after {epoch} epochs "
                f"(no val EER improvement for {early_stop_patience} epochs)."
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if sync_fusion_head is not None and best_sync_head_state is not None:
            sync_fusion_head.load_state_dict(best_sync_head_state)
        if sync_trainable and sync_model is not None and best_sync_model_state is not None:
            sync_model.load_state_dict(best_sync_model_state)
        if ema is not None:
            ema.load_from_model(model)
    final_eval_model = ema.ema_model if ema is not None and ema.ready else model
    if sync_model is not None:
        sync_model.eval()
    train_loss, train_eer, train_stats = evaluate(
        final_eval_model,
        train_loader,
        criterion,
        device,
        audio_backbone=audio_backbone,
        wave_branch_mode=wave_branch_mode,
        wave_segment_samples=wave_segment_samples,
        eval_wave_segments=1,
        eval_augment_passes=1,
        eval_augment_prob=0.0,
        eval_freq_mask=0,
        eval_time_mask=0,
        eval_noise_std=0.0,
        arcface_head=arcface_head,
        sync_model=sync_model,
        sync_head=sync_fusion_head,
        sync_feature_dim=sync_feature_dim,
        sync_target_frames=sync_target_frames,
        sync_audio_dim=sync_audio_dim,
    )
    val_loss, val_eer, val_stats = evaluate(
        final_eval_model,
        val_loader,
        criterion,
        device,
        audio_backbone=audio_backbone,
        wave_branch_mode=wave_branch_mode,
        wave_segment_samples=wave_segment_samples,
        eval_wave_segments=int(args.eval_wave_segments),
        eval_augment_passes=int(args.eval_augment_passes),
        eval_augment_prob=float(args.eval_augment_prob),
        eval_freq_mask=int(args.eval_freq_mask),
        eval_time_mask=int(args.eval_time_mask),
        eval_noise_std=float(args.eval_noise_std),
        arcface_head=arcface_head,
        sync_model=sync_model,
        sync_head=sync_fusion_head,
        sync_feature_dim=sync_feature_dim,
        sync_target_frames=sync_target_frames,
        sync_audio_dim=sync_audio_dim,
    )
    test_loss, test_eer, test_stats = evaluate(
        final_eval_model,
        test_loader,
        criterion,
        device,
        audio_backbone=audio_backbone,
        wave_branch_mode=wave_branch_mode,
        wave_segment_samples=wave_segment_samples,
        eval_wave_segments=int(args.eval_wave_segments),
        eval_augment_passes=int(args.eval_augment_passes),
        eval_augment_prob=float(args.eval_augment_prob),
        eval_freq_mask=int(args.eval_freq_mask),
        eval_time_mask=int(args.eval_time_mask),
        eval_noise_std=float(args.eval_noise_std),
        arcface_head=arcface_head,
        sync_model=sync_model,
        sync_head=sync_fusion_head,
        sync_feature_dim=sync_feature_dim,
        sync_target_frames=sync_target_frames,
        sync_audio_dim=sync_audio_dim,
    )
    print(
        f"Train loss={train_loss:.4f}, eer={train_eer:.4f}, "
        f"TP={train_stats['tp']:.0f}, FP={train_stats['fp']:.0f}, TN={train_stats['tn']:.0f}, FN={train_stats['fn']:.0f}"
    )
    print(
        f"Val loss={val_loss:.4f}, eer={val_eer:.4f}, "
        f"TP={val_stats['tp']:.0f}, FP={val_stats['fp']:.0f}, TN={val_stats['tn']:.0f}, FN={val_stats['fn']:.0f}"
    )
    print(
        f"Test loss={test_loss:.4f}, eer={test_eer:.4f}, "
        f"TP={test_stats['tp']:.0f}, FP={test_stats['fp']:.0f}, TN={test_stats['tn']:.0f}, FN={test_stats['fn']:.0f}"
    )


if __name__ == "__main__":
    train()
