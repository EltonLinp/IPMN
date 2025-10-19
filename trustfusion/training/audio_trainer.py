"""
Supervised training pipeline for the audio modality using preprocessed log-mel clips.

The pipeline mirrors the structure of ``video_trainer.py`` so that datasets can be
plugged in later without refactoring. At present it expects the preprocessing step
to have exported ``audio/mel.npy`` files alongside ``metadata.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..config import DatasetConfig
from ..modules.audio_detector import AudioDeepfakeDetector
from ..utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class AudioSample:
    mel_path: Path
    label: int
    subject: str
    video_id: str
    start_index: int
    segment_length: int
    total_frames: int


def _load_manifest(manifest_path: Path) -> Iterable[dict]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"video_path", "label", "subject"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{manifest_path} missing required headers {required}")
        for row in reader:
            yield row


class AudioPreprocessedDataset(Dataset):
    """
    Dataset turning preprocessed mel spectrograms into fixed-length sequences.
    """

    def __init__(
        self,
        manifest_path: Path,
        preprocess_root: Path,
        segment_length: int = 128,
        segment_hop: int = 64,
        augment: bool = False,
        max_segments_per_video: Optional[int] = None,
    ) -> None:
        self.segment_length = segment_length
        self.segment_hop = max(1, segment_hop)
        self.augment = augment
        self.preprocess_root = preprocess_root

        self.samples: List[AudioSample] = []
        manifest_records = list(_load_manifest(manifest_path))
        if not manifest_records:
            LOGGER.warning("Manifest %s is empty.", manifest_path)

        for record in manifest_records:
            subject = record["subject"]
            label_text = record["label"].strip().lower()
            label = 1 if label_text == "fake" else 0

            video_path = Path(record["video_path"])
            video_id = video_path.stem
            sample_root = preprocess_root / subject / video_id
            mel_path = sample_root / "audio" / "mel.npy"
            metadata_path = sample_root / "metadata.json"

            if not mel_path.exists():
                LOGGER.debug("Skipping missing mel spectrogram %s", mel_path)
                continue

            total_frames = self._resolve_total_frames(metadata_path, mel_path)
            if total_frames <= 0:
                LOGGER.debug("Skipping %s as mel frames are unavailable.", mel_path)
                continue

            segment_starts = self._compute_segment_starts(total_frames)
            if max_segments_per_video and len(segment_starts) > max_segments_per_video:
                segment_starts = segment_starts[:max_segments_per_video]

            for start in segment_starts:
                self.samples.append(
                    AudioSample(
                        mel_path=mel_path,
                        label=label,
                        subject=subject,
                        video_id=video_id,
                        start_index=start,
                        segment_length=self.segment_length,
                        total_frames=total_frames,
                    )
                )

        if not self.samples:
            LOGGER.warning("No audio samples assembled from %s.", manifest_path)
        else:
            LOGGER.info(
                "Audio dataset loaded: segments=%d videos=%d",
                len(self.samples),
                len(manifest_records),
            )

    def _resolve_total_frames(self, metadata_path: Path, mel_path: Path) -> int:
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
                mel_shape = metadata.get("audio", {}).get("mel_shape")
                if mel_shape and len(mel_shape) == 2:
                    return int(mel_shape[1])
            except Exception as exc:  # pragma: no cover - metadata errors are logged
                LOGGER.debug("Failed to read metadata %s: %s", metadata_path, exc)

        try:
            mel = np.load(mel_path)
            return int(mel.shape[1]) if mel.ndim == 2 else 0
        except Exception as exc:  # pragma: no cover - IO errors surfaced in debug
            LOGGER.debug("Failed to probe mel file %s: %s", mel_path, exc)
        return 0

    def _compute_segment_starts(self, total_frames: int) -> Sequence[int]:
        if total_frames <= self.segment_length:
            return (0,)
        starts = list(range(0, total_frames - self.segment_length + 1, self.segment_hop))
        if not starts:
            starts = [0]
        return starts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        mel = self._load_segment(sample)
        label = torch.tensor(float(sample.label), dtype=torch.float32)
        return {
            "audio": mel,
            "label": label,
            "meta": {
                "subject": sample.subject,
                "video_id": sample.video_id,
                "start": sample.start_index,
                "total_frames": sample.total_frames,
            },
        }

    def _load_segment(self, sample: AudioSample) -> Tensor:
        mel = np.load(sample.mel_path)
        if mel.ndim != 2:
            raise ValueError(f"Expected 2D mel spectrogram, received shape {mel.shape}")

        start = min(sample.start_index, max(sample.total_frames - 1, 0))
        end = start + sample.segment_length
        segment = mel[:, start:end]

        if segment.shape[1] < sample.segment_length:
            pad_width = sample.segment_length - segment.shape[1]
            pad_value = segment[:, -1:] if segment.shape[1] else np.zeros((mel.shape[0], 1))
            segment = np.concatenate([segment, np.repeat(pad_value, pad_width, axis=1)], axis=1)

        tensor = torch.from_numpy(segment).to(torch.float32).unsqueeze(0)  # (1, F, T)
        if self.augment:
            noise = torch.randn_like(tensor) * 0.01
            tensor = tensor + noise
        return tensor


def _compute_accuracy(predictions: Tensor, targets: Tensor) -> float:
    if predictions.numel() == 0:
        return 0.0
    preds = (predictions >= 0.5).float()
    return float((preds == targets).float().mean().item())


def train_epoch(
    model: AudioDeepfakeDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    desc: str = "Train",
) -> Tuple[float, float]:
    return _train_or_eval_epoch(
        model=model,
        loader=loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        train=True,
        desc=desc,
    )


@torch.no_grad()
def evaluate_epoch(
    model: AudioDeepfakeDetector,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    desc: str = "Validate",
) -> Tuple[float, float]:
    return _train_or_eval_epoch(
        model=model,
        loader=loader,
        device=device,
        criterion=criterion,
        optimizer=None,
        train=False,
        desc=desc,
    )


def _train_or_eval_epoch(
    model: AudioDeepfakeDetector,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    train: bool,
    desc: str,
) -> Tuple[float, float]:
    if train and optimizer is None:
        raise ValueError("Optimizer must be provided when training.")
    model.train(train)
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in progress:
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        output = model(audio)
        logits = output.logits
        loss = criterion(logits, labels)

        if train:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        batch_acc = _compute_accuracy(output.scores.detach(), labels.detach())
        running_acc += batch_acc * batch_size
        total += batch_size
        if total:
            progress.set_postfix(
                loss=f"{running_loss / total:.4f}",
                acc=f"{running_acc / total:.3f}",
            )

    if total == 0:
        return 0.0, 0.0
    return running_loss / total, running_acc / total


def save_checkpoint(model: AudioDeepfakeDetector, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    LOGGER.info("Saved audio checkpoint to %s", path)


def build_dataloaders(
    dataset_config: DatasetConfig,
    preprocessed_root: Path,
    train_manifest: Optional[Path],
    val_manifest: Optional[Path],
    batch_size: int,
    num_workers: int,
    segment_length: int,
    segment_hop: int,
    augment: bool,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    if train_manifest is None:
        train_manifest = dataset_config.train_manifest
    if val_manifest is None:
        val_manifest = dataset_config.val_manifest

    if train_manifest is None:
        raise ValueError("Train manifest path must be provided.")

    train_ds = AudioPreprocessedDataset(
        manifest_path=Path(train_manifest),
        preprocess_root=preprocessed_root,
        segment_length=segment_length,
        segment_hop=segment_hop,
        augment=augment,
    )
    if len(train_ds) == 0:
        raise ValueError(
            f"No audio segments were found for manifest {train_manifest}. "
            f"Ensure mel features exist under {preprocessed_root}."
        )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader: Optional[DataLoader] = None
    if val_manifest and Path(val_manifest).exists():
        val_ds = AudioPreprocessedDataset(
            manifest_path=Path(val_manifest),
            preprocess_root=preprocessed_root,
            segment_length=segment_length,
            segment_hop=segment_length,
            augment=False,
        )
        if len(val_ds) > 0:
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the audio deepfake detector on mel spectrogram segments.")
    parser.add_argument("--preprocessed-root", type=Path, default=Path("data/preprocessed/train"))
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--segment-length", type=int, default=128)
    parser.add_argument("--segment-hop", type=int, default=64)
    parser.add_argument("--no-augment", action="store_true", help="Disable simple additive noise augmentation.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string (e.g., cuda or cpu).",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/audio_detector.pt"))
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a machine with GPU support or set --device cpu.")
    return device


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    dataset_cfg = DatasetConfig()
    device = _resolve_device(args.device)
    LOGGER.info("Using device: %s", device)

    train_loader, val_loader = build_dataloaders(
        dataset_config=dataset_cfg,
        preprocessed_root=args.preprocessed_root,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        segment_length=args.segment_length,
        segment_hop=args.segment_hop,
        augment=not args.no_augment,
    )

    model = AudioDeepfakeDetector()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            desc=f"Train {epoch}/{args.epochs}",
        )
        LOGGER.info("Epoch %d/%d | train loss %.4f | train acc %.3f", epoch, args.epochs, train_loss, train_acc)

        if val_loader is not None:
            val_loss, val_acc = evaluate_epoch(
                model,
                val_loader,
                device,
                criterion,
                desc=f"Validate {epoch}/{args.epochs}",
            )
            LOGGER.info("Epoch %d/%d | val loss %.4f | val acc %.3f", epoch, args.epochs, val_loss, val_acc)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, Path(args.output))
        else:
            save_checkpoint(model, Path(args.output))


if __name__ == "__main__":
    main()

