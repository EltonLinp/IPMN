"""
Supervised training pipeline for the video modality using preprocessed Celeb-DF v2 clips.

The pipeline consumes the frame crops produced by ``tools/preprocess_celebdf.py`` and
optimises the placeholder :class:`~trustfusion.modules.video_detector.VideoDeepfakeDetector`.
It is intentionally lightweight so we can iterate quickly before swapping in a stronger
backbone. GPU acceleration (CUDA) is mandatory for all training runs.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from tqdm import tqdm

from ..config import DatasetConfig

from ..modules.video_detector import VideoDeepfakeDetector
from ..utils.logger import get_logger

LOGGER = get_logger(__name__)

ImageSize = Tuple[int, int]

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)


@dataclass
class ClipSample:
    frame_paths: Sequence[Path]
    label: int
    subject: str
    video_id: str
    start_index: int


def _load_manifest(manifest_path: Path) -> Iterable[dict]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"video_path", "label", "subject"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{manifest_path} missing required headers {required}")
        for row in reader:
            yield row


class VideoPreprocessedDataset(Dataset):
    """
    Dataset turning preprocessed frame folders into T×H×W clips.
    """

    def __init__(
        self,
        manifest_path: Path,
        preprocess_root: Path,
        clip_length: int = 16,
        clip_stride: int = 8,
        image_size: ImageSize = (112, 112),
        augment: bool = False,
        max_clips_per_video: Optional[int] = None,
    ) -> None:
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.image_size = image_size
        self.augment = augment
        self.preprocess_root = preprocess_root

        self.samples: List[ClipSample] = []
        manifest_records = list(_load_manifest(manifest_path))
        if not manifest_records:
            LOGGER.warning("Manifest %s is empty.", manifest_path)
        for record in manifest_records:
            subject = record["subject"]
            label_text = record["label"].strip().lower()
            label = 1 if label_text == "fake" else 0

            video_path = Path(record["video_path"])
            video_id = video_path.stem
            frame_dir = preprocess_root / subject / video_id / "full_face"
            if not frame_dir.exists():
                LOGGER.debug("Skipping missing frame directory %s", frame_dir)
                continue
            frame_paths = sorted(frame_dir.glob("*.png"))
            if not frame_paths:
                LOGGER.debug("Skipping %s as no frames found.", frame_dir)
                continue

            clip_indices = self._compute_clip_starts(len(frame_paths), max_clips_per_video)
            for start in clip_indices:
                self.samples.append(
                    ClipSample(
                        frame_paths=frame_paths,
                        label=label,
                        subject=subject,
                        video_id=video_id,
                        start_index=start,
                    )
                )

        if not self.samples:
            LOGGER.warning("No clip samples assembled from %s.", manifest_path)
        else:
            LOGGER.info("Video dataset loaded: clips=%d videos=%d", len(self.samples), len(manifest_records))

    def _compute_clip_starts(self, num_frames: int, max_clips_per_video: Optional[int]) -> Sequence[int]:
        if num_frames <= 0:
            return (0,)
        if num_frames <= self.clip_length:
            return (0,)
        stride = max(1, self.clip_stride)
        starts = list(range(0, num_frames - self.clip_length + 1, stride))
        if max_clips_per_video and len(starts) > max_clips_per_video:
            rng = random.Random(1337)
            rng.shuffle(starts)
            starts = sorted(starts[:max_clips_per_video])
        if not starts:
            starts = [0]
        return starts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        clip = self._load_clip(sample.frame_paths, sample.start_index)
        label = torch.tensor(float(sample.label), dtype=torch.float32)
        return {
            "video": clip,
            "label": label,
            "meta": {
                "subject": sample.subject,
                "video_id": sample.video_id,
                "start": sample.start_index,
                "num_frames": len(sample.frame_paths),
            },
        }

    def _load_clip(self, frame_paths: Sequence[Path], start: int) -> Tensor:
        frames: List[Tensor] = []
        max_index = len(frame_paths) - 1
        height, width = self.image_size
        for offset in range(self.clip_length):
            frame_idx = min(start + offset, max_index)
            frame = read_image(str(frame_paths[frame_idx])).float() / 255.0
            if frame.shape[1] != height or frame.shape[2] != width:
                frame = resize(frame, size=[height, width])
            frames.append(frame)
        clip = torch.stack(frames, dim=1)  # (C, T, H, W)
        if self.augment and random.random() < 0.5:
            clip = torch.flip(clip, dims=(3,))
        clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
        return clip


def _compute_accuracy(predictions: Tensor, targets: Tensor) -> float:
    if predictions.numel() == 0:
        return 0.0
    preds = (predictions >= 0.5).float()
    return float((preds == targets).float().mean().item())


def train_epoch(
    model: VideoDeepfakeDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    desc: str = "Train",
) -> Tuple[float, float]:
    return _train_or_eval_epoch(
        model=model,
        loader=loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        train=True,
        desc=desc,
    )


def _train_or_eval_epoch(
    model: VideoDeepfakeDetector,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    train: bool = True,
    desc: str = "Run",
) -> Tuple[float, float]:
    if train and optimizer is None:
        raise ValueError("Optimizer must be provided when training.")
    model.train(train)
    running_loss = 0.0
    running_acc = 0.0
    total = 0

    progress = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    for batch in progress:
        videos = batch["video"].to(device)
        labels = batch["label"].to(device)
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=scaler is not None)
            else:
                autocast_ctx = torch.cuda.amp.autocast(enabled=scaler is not None)
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            output = model(videos)
            scores = output.scores
            logits = output.logits
            loss = criterion(logits, labels)
        if train:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        batch_acc = _compute_accuracy(scores.detach(), labels.detach())
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


@torch.no_grad()
def evaluate_epoch(
    model: VideoDeepfakeDetector,
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
        scaler=None,
        train=False,
        desc=desc,
    )


def save_checkpoint(model: VideoDeepfakeDetector, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    LOGGER.info("Saved video checkpoint to %s", path)


def build_dataloaders(
    dataset_config: DatasetConfig,
    preprocessed_root: Path,
    train_manifest: Optional[Path],
    val_manifest: Optional[Path],
    batch_size: int,
    num_workers: int,
    clip_length: int,
    clip_stride: int,
    image_size: int,
    augment: bool,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    if train_manifest is None:
        train_manifest = dataset_config.train_manifest
    if val_manifest is None:
        val_manifest = dataset_config.val_manifest

    if train_manifest is None:
        raise ValueError("Train manifest path must be provided.")

    train_ds = VideoPreprocessedDataset(
        manifest_path=Path(train_manifest),
        preprocess_root=preprocessed_root,
        clip_length=clip_length,
        clip_stride=clip_stride,
        image_size=(image_size, image_size),
        augment=augment,
    )
    if len(train_ds) == 0:
        raise ValueError(
            f"No training clips were found for manifest {train_manifest}. "
            f"Ensure preprocessed frames exist under {preprocessed_root}."
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
        val_ds = VideoPreprocessedDataset(
            manifest_path=Path(val_manifest),
            preprocess_root=preprocessed_root,
            clip_length=clip_length,
            clip_stride=clip_length,  # deterministic clips
            image_size=(image_size, image_size),
            augment=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the video deepfake detector on preprocessed clips.")
    parser.add_argument("--preprocessed-root", type=Path, default=Path("data/preprocessed/train"))
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--clip-stride", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--no-augment", action="store_true", help="Disable simple spatial augmentations.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="CUDA device string (e.g., cuda or cuda:1). CPU training is not supported.",
    )
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--output", type=Path, default=Path("outputs/video_detector.pt"))
    return parser.parse_args()


def _resolve_cuda_device(device_arg: str) -> torch.device:
    if not device_arg.lower().startswith("cuda"):
        raise ValueError(f"CUDA is required for training; received device '{device_arg}'.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a machine with GPU support.")
    device = torch.device(device_arg)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"Requested CUDA device index {device.index} exceeds available GPUs.")
    return device


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = parse_args()

    dataset_cfg = DatasetConfig()
    device = _resolve_cuda_device(args.device)
    LOGGER.info("Using device: %s", device)

    train_loader, val_loader = build_dataloaders(
        dataset_config=dataset_cfg,
        preprocessed_root=args.preprocessed_root,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        image_size=args.image_size,
        augment=not args.no_augment,
    )

    model = VideoDeepfakeDetector()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler = None
    if args.amp and device.type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                scaler = torch.amp.GradScaler(device_type="cuda")
            except TypeError:
                scaler = torch.amp.GradScaler()
        elif hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
            scaler = torch.cuda.amp.GradScaler()

    best_val_loss = math.inf
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            criterion,
            scaler,
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
