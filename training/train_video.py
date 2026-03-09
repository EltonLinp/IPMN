from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from datasets import FakeAVVideoDataset
from datasets.fakeav_video_dataset import _read_index
from models import VideoClassifier


class VideoAugmentation:
    """
    Lightweight augmentation pipeline operating on [C, T, H, W] tensors.
    """

    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        temporal_jitter: int = 4,
        brightness: float = 0.1,
        contrast: float = 0.1,
        noise_std: float = 0.02,
    ) -> None:
        self.horizontal_flip_prob = float(max(0.0, min(horizontal_flip_prob, 1.0)))
        self.temporal_jitter = max(int(temporal_jitter), 0)
        self.brightness = max(float(brightness), 0.0)
        self.contrast = max(float(contrast), 0.0)
        self.noise_std = max(float(noise_std), 0.0)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if self.horizontal_flip_prob > 0.0 and random.random() < self.horizontal_flip_prob:
            video = torch.flip(video, dims=[-1])

        if self.temporal_jitter > 0 and video.shape[1] > 1:
            shift = random.randint(-self.temporal_jitter, self.temporal_jitter)
            if shift != 0:
                video = torch.roll(video, shifts=shift, dims=1)

        if self.brightness > 0.0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            video = (video * factor).clamp(-1.0, 1.0)

        if self.contrast > 0.0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = video.mean(dim=(-1, -2), keepdim=True)
            video = ((video - mean) * factor + mean).clamp(-1.0, 1.0)

        if self.noise_std > 0.0:
            noise = torch.randn_like(video) * self.noise_std
            video = (video + noise).clamp(-1.0, 1.0)

        return video.contiguous()


class ModelEMA:
    """
    Exponential moving average of model parameters stored on CPU.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self._update(model, initialize=True)

    def _update(self, model: nn.Module, initialize: bool = False) -> None:
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if not param.dtype.is_floating_point:
                    self.shadow[name] = param.detach().cpu()
                    continue
                data = param.detach().cpu()
                if initialize or name not in self.shadow:
                    self.shadow[name] = data.clone()
                else:
                    self.shadow[name].mul_(self.decay).add_(data, alpha=1.0 - self.decay)

    def update(self, model: nn.Module) -> None:
        self._update(model, initialize=False)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


def _move_batch_to_device(batch, device: torch.device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_batch_to_device(sample, device) for sample in batch)
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    return batch


class CudaPrefetchLoader:
    """
    Wraps a DataLoader to asynchronously move the next batch to GPU.
    """

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        self.iter = iter(self.loader)
        self.next_batch = None
        self._preload()
        return self

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_batch is None:
            raise StopIteration
        batch = self.next_batch
        self._preload()
        return batch

    def _preload(self) -> None:
        try:
            batch = next(self.iter)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = _move_batch_to_device(batch, self.device)


def maybe_resize(video: torch.Tensor, spatial_size: int | None) -> torch.Tensor:
    if spatial_size is None:
        return video
    if video.dim() != 5:
        return video
    if video.shape[-1] == spatial_size and video.shape[-2] == spatial_size:
        return video
    return F.interpolate(
        video,
        size=(video.shape[2], spatial_size, spatial_size),
        mode="trilinear",
        align_corners=False,
    )


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, help="Path to JSON config file.")
    config_args, remaining = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train a video-only FakeAVCeleb classifier.", parents=[config_parser]
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to directory containing preprocessed .pt files.")
    parser.add_argument("--index-file", type=Path, default=None, help="Optional preprocess_index.jsonl path.")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--target-frames", type=int, default=32, help="Number of frames per clip after pad/crop.")
    parser.add_argument(
        "--spatial-size",
        type=int,
        default=None,
        help="Optionally resize spatial dimensions (height/width) to this value before feeding the model.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Worker prefetch multiplier (set <=0 to disable; only valid when num_workers > 0).",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive across epochs for faster iterations.",
    )
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--lr-patience", type=int, default=2, help="ReduceLROnPlateau patience.")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR reduction factor on plateau.")
    parser.add_argument("--class-weights", action="store_true", help="Use class-weighted loss for the training set.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use.")
    parser.add_argument("--save-path", type=Path, default=Path("video_classifier.pt"), help="Path to save best model.")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation to the video inputs.")
    parser.add_argument("--balanced-sampler", action="store_true", help="Use class-balanced sampling for the training loader.")
    parser.add_argument("--horizontal-flip-prob", type=float, default=0.5, help="Probability of random horizontal flip.")
    parser.add_argument("--temporal-jitter", type=int, default=4, help="Maximum temporal roll (frames) applied to sequences.")
    parser.add_argument("--brightness-jitter", type=float, default=0.15, help="Maximum brightness jitter factor.")
    parser.add_argument("--contrast-jitter", type=float, default=0.15, help="Maximum contrast jitter factor.")
    parser.add_argument("--noise-std", type=float, default=0.02, help="Standard deviation of Gaussian noise added to frames.")
    parser.add_argument("--grad-clip-norm", type=float, default=2.0, help="Clip gradient norm to this value (0 to disable).")
    parser.add_argument("--ema", action="store_true", help="Enable exponential moving average of model weights.")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="Decay factor for EMA updates.")
    parser.add_argument("--ema-start-epoch", type=int, default=5, help="Epoch to start applying EMA updates.")
    parser.add_argument(
        "--backbone",
        choices=("light", "r3d18"),
        default="light",
        help="Video backbone architecture to use.",
    )
    parser.add_argument(
        "--backbone-pretrained",
        action="store_true",
        help="Initialise backbone with pretrained weights (if supported).",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze backbone parameters and only train classifier heads.",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout used before classification heads.")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Accumulate gradients across this many micro-batches before stepping the optimizer.",
    )
    parser.add_argument(
        "--cuda-prefetch",
        action="store_true",
        help="Enable CUDA stream prefetcher to overlap data transfer with compute.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable torch.backends.cudnn.benchmark for faster conv kernels on fixed-size inputs.",
    )

    if config_args.config:
        with config_args.config.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        cfg_cli: list[str] = []
        for key, value in config.items():
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
    missing: list[Path] = []
    for path in files:
        name = path.name
        if name not in label_map:
            missing.append(path)
    for path in missing:
        bundle = torch.load(path, map_location="cpu")
        label = int(bundle.get("label", 0))
        label_map[path.name] = label
    return [label_map[path.name] for path in files]


def _build_sampler(
    dataset: FakeAVVideoDataset,
    index_file: Path | None,
) -> WeightedRandomSampler:
    labels = _load_labels_for_files(dataset.files, index_file)
    counts = Counter(labels)
    weights = [1.0 / max(counts[label], 1) for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def build_dataloaders(
    data_dir: Path,
    index_file: Path | None,
    batch_size: int,
    num_workers: int,
    *,
    target_frames: int,
    augment: bool,
    augment_params: Dict[str, float],
    balanced_sampler: bool,
    pin_memory: bool,
    prefetch_factor: int | None,
    persistent_workers: bool,
) -> Tuple[FakeAVVideoDataset, DataLoader, DataLoader, DataLoader]:
    split_seed = 1337
    train_transform = VideoAugmentation(**augment_params) if augment else None
    train_dataset = FakeAVVideoDataset(
        data_dir=data_dir,
        index_file=index_file,
        split="train",
        return_metadata=False,
        seed=split_seed,
        target_frames=target_frames,
        transform=train_transform,
    )
    val_dataset = FakeAVVideoDataset(
        data_dir=data_dir,
        index_file=index_file,
        split="val",
        return_metadata=False,
        seed=split_seed,
        target_frames=target_frames,
    )
    test_dataset = FakeAVVideoDataset(
        data_dir=data_dir,
        index_file=index_file,
        split="test",
        return_metadata=False,
        seed=split_seed,
        target_frames=target_frames,
    )
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0 and prefetch_factor is not None and prefetch_factor > 0:
        common_kwargs["prefetch_factor"] = prefetch_factor
    if num_workers > 0 and persistent_workers:
        common_kwargs["persistent_workers"] = True
    if balanced_sampler:
        sampler = _build_sampler(train_dataset, index_file)
        train_loader = DataLoader(train_dataset, shuffle=False, sampler=sampler, **common_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_kwargs)
    return train_dataset, train_loader, val_loader, test_loader


def compute_mcc(tp: float, fp: float, tn: float, fn: float) -> float:
    numerator = tp * tn - fp * fn
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator ** 0.5)


def compute_eer(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    if scores.numel() == 0 or labels.numel() == 0:
        return 0.5, 0.5
    labels = labels.to(dtype=torch.float32)
    positives = labels.sum().item()
    negatives = labels.numel() - positives
    if positives == 0 or negatives == 0:
        return 0.0, 0.5
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
    threshold = scores[order[idx]]
    return float(eer.item()), float(threshold.item())


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    eval_state: dict[str, torch.Tensor] | None = None,
    spatial_size: int | None = None,
) -> Tuple[float, float, Dict[str, float]]:
    original_state: dict[str, torch.Tensor] | None = None
    if eval_state is not None:
        original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(eval_state, strict=False)
    model.eval()
    total_loss = 0.0
    total_samples = 0
    score_buffer: list[torch.Tensor] = []
    label_buffer: list[torch.Tensor] = []
    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device, non_blocking=True)
            videos = maybe_resize(videos, spatial_size)
            labels = labels.to(device, non_blocking=True)
            logits, _, _ = model(videos)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
            score_buffer.append(probs)
            label_buffer.append(labels.detach().cpu())
    avg_loss = total_loss / max(total_samples, 1)
    if score_buffer:
        all_scores = torch.cat(score_buffer, dim=0)
        all_labels = torch.cat(label_buffer, dim=0)
        eer, threshold = compute_eer(all_scores, all_labels)
        preds = (all_scores >= threshold).to(all_labels.dtype)
        tp = ((preds == 1) & (all_labels == 1)).sum().item()
        tn = ((preds == 0) & (all_labels == 0)).sum().item()
        fp = ((preds == 1) & (all_labels == 0)).sum().item()
        fn = ((preds == 0) & (all_labels == 1)).sum().item()
    else:
        eer, threshold = 0.5, 0.5
        tp = tn = fp = fn = 0.0
    mcc = compute_mcc(tp, fp, tn, fn)
    stats = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "eer": eer, "threshold": threshold}
    if original_state is not None:
        model.load_state_dict(original_state, strict=False)
    return avg_loss, mcc, stats


def load_label_counts(train_dataset: FakeAVVideoDataset, index_file: Path | None) -> Counter:
    counts: Counter = Counter()
    name_set = {Path(path).name for path in train_dataset.files}
    if index_file and index_file.exists():
        with index_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("status") != "ok":
                    continue
                name = Path(str(record.get("output_path", ""))).name
                if name in name_set:
                    counts[int(record.get("label", 0))] += 1
    if not counts:
        for path in train_dataset.files:
            bundle = torch.load(path, map_location="cpu")
            counts[int(bundle["label"])] += 1
    return counts


def train() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    if args.class_weights and args.balanced_sampler:
        print("Info: balanced sampler active; disabling class weights to avoid over-penalising minority class.")
        use_class_weights = False
    else:
        use_class_weights = args.class_weights

    augment_params = {
        "horizontal_flip_prob": args.horizontal_flip_prob,
        "temporal_jitter": args.temporal_jitter,
        "brightness": args.brightness_jitter,
        "contrast": args.contrast_jitter,
        "noise_std": args.noise_std,
    }

    prefetch_factor = args.prefetch_factor
    if prefetch_factor is not None and prefetch_factor <= 0:
        prefetch_factor = None

    train_dataset, train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        index_file=args.index_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_frames=args.target_frames,
        augment=args.augment,
        augment_params=augment_params,
        balanced_sampler=args.balanced_sampler,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        persistent_workers=args.persistent_workers,
    )

    if use_class_weights:
        counts = load_label_counts(train_dataset, args.index_file)
        total = sum(counts.values())
        weight0 = total / counts.get(0, 1)
        weight1 = total / counts.get(1, 1)
        class_weights = torch.tensor([weight0, weight1], dtype=torch.float32, device=device)
        print(f"Class counts: {counts}, weights: [{weight0:.2f}, {weight1:.2f}]")
    else:
        class_weights = None

    model = VideoClassifier(
        num_classes=2,
        dropout=args.dropout,
        backbone=args.backbone,
        pretrained=args.backbone_pretrained,
    ).to(device)
    if args.freeze_backbone:
        model.freeze_backbone()
        print("Backbone parameters frozen.")
    print(
        f"Using video backbone '{args.backbone}' (pretrained={args.backbone_pretrained}, "
        f"frozen={args.freeze_backbone})"
    )
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )

    best_val_metric = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improve = 0
    global_step = 0
    ema_ready = False

    use_cuda_prefetch = bool(args.cuda_prefetch and device.type == "cuda")

    grad_accum = max(1, args.grad_accum_steps)
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_iterable = CudaPrefetchLoader(train_loader, device) if use_cuda_prefetch else train_loader
        progress = tqdm(train_iterable, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        running_loss = 0.0
        total = 0
        accum_counter = 0
        optimizer.zero_grad(set_to_none=True)
        for videos, labels in progress:
            if not use_cuda_prefetch:
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            videos = maybe_resize(videos, args.spatial_size)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits, _, _ = model(videos)
                    loss = criterion(logits, labels)
            else:
                logits, _, _ = model(videos)
                loss = criterion(logits, labels)
            loss_value = loss.item()

            if scaler is not None:
                scaler.scale(loss / grad_accum).backward()
            else:
                (loss / grad_accum).backward()

            accum_counter += 1
            step_complete = accum_counter == grad_accum

            if step_complete:
                if scaler is not None:
                    if args.grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.grad_clip_norm > 0:
                        clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                if ema is not None and epoch >= args.ema_start_epoch:
                    ema.update(model)

            running_loss += loss_value * labels.size(0)
            total += labels.size(0)
            progress.set_postfix({"loss": running_loss / max(total, 1)})
            global_step += 1

        if accum_counter > 0:
            if scaler is not None:
                if args.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if args.grad_clip_norm > 0:
                    clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None and epoch >= args.ema_start_epoch:
                ema.update(model)

        ema_state = ema.state_dict() if ema is not None and epoch >= args.ema_start_epoch else None
        if ema_state is not None:
            ema_ready = True
        val_loss, val_mcc, val_stats = evaluate(
            model,
            val_loader,
            criterion,
            device,
            eval_state=ema_state,
            spatial_size=args.spatial_size,
        )
        val_eer = val_stats.get("eer", 0.5)
        tqdm.write(
            f"Epoch {epoch}: val_loss={val_loss:.4f}, val_mcc={val_mcc:.4f}, val_eer={val_eer:.4f}, "
            f"TP={val_stats['tp']:.0f}, FP={val_stats['fp']:.0f}, TN={val_stats['tn']:.0f}, FN={val_stats['fn']:.0f}"
        )
        scheduler.step(val_eer)

        if val_eer < best_val_metric:
            best_val_metric = val_eer
            if ema_state is not None:
                best_state = {k: v.clone() for k, v in ema_state.items()}
            else:
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state": best_state,
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_mcc": val_mcc,
                    "val_eer": val_eer,
                    "args": vars(args),
                    "ema_state": best_state if ema_state is not None else None,
                },
                args.save_path,
            )
            tqdm.write(f"Saved new best model to {args.save_path}")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                tqdm.write(f"Early stopping triggered after {epoch} epochs (no improvement for {args.patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    ema_test_state = ema.state_dict() if ema is not None and ema_ready else None
    test_loss, test_mcc, test_stats = evaluate(
        model,
        test_loader,
        criterion,
        device,
        eval_state=ema_test_state if ema_test_state is not None else None,
        spatial_size=args.spatial_size,
    )
    test_eer = test_stats.get("eer", 0.5)
    print(
        f"Test loss={test_loss:.4f}, test_mcc={test_mcc:.4f}, test_eer={test_eer:.4f}, "
        f"TP={test_stats['tp']:.0f}, FP={test_stats['fp']:.0f}, TN={test_stats['tn']:.0f}, FN={test_stats['fn']:.0f}"
    )


if __name__ == "__main__":
    train()
