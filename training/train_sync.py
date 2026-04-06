from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import FakeAVSyncDataset, SyncDatasetConfig, DatasetSplit
from models import SyncModule


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0.0:
            n_classes = logits.size(1)
            smooth = self.label_smoothing / max(n_classes - 1, 1)
            one_hot = torch.full_like(logits, smooth)
            one_hot.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            log_prob = torch.log_softmax(logits, dim=1)
            prob = log_prob.exp()
            focal_weight = (1.0 - prob) ** self.gamma
            loss = -(focal_weight * one_hot * log_prob).sum(dim=1)
            return loss.mean()
        log_prob = torch.log_softmax(logits, dim=1)
        prob = torch.exp(log_prob)
        gather_log_prob = log_prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        gather_prob = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - gather_prob) ** self.gamma
        return (-(focal_weight * gather_log_prob)).mean()


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, help="Path to JSON config file.")
    config_args, remaining = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train synchronisation branch.",
        parents=[config_parser],
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing processed .pt files.")
    parser.add_argument("--vit-path", type=Path, required=True, help="Path to frozen ViT checkpoint directory.")
    parser.add_argument("--index-file", type=Path, default=None, help="Optional preprocess_index.jsonl path.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--negative-prob", type=float, default=0.5, help="Probability of generating misaligned audio.")
    parser.add_argument("--target-frames", type=int, default=16, help="Fixed number of frames after trim/pad.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-path", type=Path, default=Path("sync_module.pt"), help="Checkpoint output path.")
    parser.add_argument(
        "--metrics-log-path",
        type=Path,
        default=None,
        help="Optional text file used to record per-epoch validation metrics.",
    )
    parser.add_argument("--args-json", type=Path, default=None, help="Optional path to save effective arguments as JSON.")
    parser.add_argument("--amp", action="store_true", help="Enable torch.cuda.amp mixed precision.")
    parser.add_argument("--vit-cache-dir", type=Path, default=None, help="Directory to cache/load ViT frame embeddings.")
    parser.add_argument(
        "--cache-dtype",
        choices=("fp16", "fp32"),
        default="fp16",
        help="Precision used when persisting cached ViT embeddings.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched by each worker (only applies when num_workers > 0).",
    )
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        help="Keep DataLoader workers alive between epochs (requires num_workers > 0).",
    )
    parser.add_argument("--swap-prob", type=float, default=0.6, help="Probability of swapping audio with another clip when generating negatives.")
    parser.add_argument("--force-cross-speaker", action="store_true", help="Require swapped audio to come from a different speaker.")
    parser.add_argument("--roll-min-ratio", type=float, default=0.3, help="Minimum proportion of frames to roll for temporal negatives.")
    parser.add_argument("--roll-max-ratio", type=float, default=0.8, help="Maximum proportion of frames to roll for temporal negatives.")
    parser.add_argument("--val-negative-prob", type=float, default=0.5, help="Negative probability used for validation.")
    parser.add_argument("--train-paired-negatives", action="store_true", help="Emit both aligned/misaligned versions for every training sample.")
    parser.add_argument("--val-paired-negatives", action="store_true", help="Emit both aligned/misaligned versions for every validation sample.")
    parser.add_argument("--temporal-layers", type=int, default=2, help="Number of transformer layers in the temporal encoder.")
    parser.add_argument("--vit-unfreeze-layers", type=int, default=0, help="Number of final ViT encoder blocks to fine-tune.")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing factor for cross-entropy (0 disables).")
    parser.add_argument("--focal-gamma", type=float, default=0.0, help="Gamma value for focal loss (0 disables).")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop if val EER fails to improve for this many epochs (0 disables).")
    parser.add_argument("--target-val-eer", type=float, default=0.0, help="Optional val EER target to stop training once reached (>0 enables).")
    if config_args.config:
        with config_args.config.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        cfg_cli: List[str] = []
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

    return parser.parse_args(remaining)


def maybe_dump_args(args: argparse.Namespace) -> None:
    if args.args_json is None:
        return
    path = Path(args.args_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolve_metrics_log_path(save_path: Path, metrics_log_path: Path | None) -> Path:
    if metrics_log_path is not None:
        return metrics_log_path
    return save_path.with_name(f"{save_path.stem}_epoch_metrics.log")


def initialize_metrics_log(path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Sync training epoch metrics\n")
        handle.write(f"# save_path={args.save_path}\n")


def append_metrics_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def format_epoch_metrics_line(epoch: int, loss: float, metrics: Dict[str, float]) -> str:
    return (
        f"Epoch {epoch}: val_loss={loss:.4f}, val_mcc={metrics.get('mcc', 0.0):.4f}, "
        f"val_eer={metrics.get('eer', 0.5):.4f}, TP={metrics.get('tp', 0.0):.0f}, "
        f"FP={metrics.get('fp', 0.0):.0f}, TN={metrics.get('tn', 0.0):.0f}, FN={metrics.get('fn', 0.0):.0f}"
    )


def create_grad_scaler(device: torch.device, use_amp: bool):
    if not use_amp or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler(device_type=device.type, enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def autocast_context(device: torch.device, use_amp: bool):
    if not use_amp or device.type != "cuda":
        return contextlib.nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    except TypeError:
        return torch.cuda.amp.autocast(enabled=True)


def _extract_metadata_paths(metadata: object | None) -> List[str] | None:
    if metadata is None:
        return None
    if isinstance(metadata, dict) and "path" in metadata:
        paths = metadata["path"]
    else:
        paths = metadata
    if isinstance(paths, (list, tuple)):
        return [str(item) for item in paths]
    if isinstance(paths, str):
        return [paths]
    return None


def prepare_video_inputs(
    model: SyncModule,
    videos: torch.Tensor,
    metadata: object | None,
    cache_dir: Path | None,
    device: torch.device,
    *,
    use_amp: bool,
    cache_dtype: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if cache_dir is None or metadata is None:
        return videos.to(device, non_blocking=True), None
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = _extract_metadata_paths(metadata)
    if paths is None:
        return videos.to(device, non_blocking=True), None
    features: List[torch.Tensor | None] = [None] * len(paths)
    missing_indices: List[int] = []
    cache_paths: List[Path] = []
    for idx, path_str in enumerate(paths):
        cache_path = cache_dir / f"{Path(path_str).stem}.pt"
        cache_paths.append(cache_path)
        if cache_path.exists():
            cache_payload = torch.load(cache_path, map_location="cpu")
            frame_emb = cache_payload.get("frame_emb")
            if not isinstance(frame_emb, torch.Tensor):
                raise RuntimeError(f"Invalid cache entry at {cache_path}")
            features[idx] = frame_emb.float()
        else:
            missing_indices.append(idx)
    if missing_indices:
        subset = torch.stack([videos[idx] for idx in missing_indices], dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            with autocast_context(device, use_amp):
                new_emb = model.encode_video(subset)
        for position, emb in zip(missing_indices, new_emb):
            tensor_to_save = emb.detach().cpu()
            if cache_dtype == "fp16":
                tensor_to_save = tensor_to_save.half()
            torch.save({"frame_emb": tensor_to_save}, cache_paths[position])
            features[position] = tensor_to_save.float()
    target_dtype = model.video_proj.weight.dtype
    if any(feat is None for feat in features):
        raise RuntimeError("Failed to prepare cached embeddings for batch.")
    stacked = torch.stack(
        [feat.to(device=device, dtype=target_dtype) for feat in features],
        dim=0,
    )
    return None, stacked


def build_criterion(
    *,
    label_smoothing: float,
    focal_gamma: float,
) -> nn.Module:
    if focal_gamma > 0.0:
        return FocalLoss(gamma=focal_gamma, label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss(label_smoothing=max(label_smoothing, 0.0))


def build_dataloaders(
    data_dir: Path,
    index_file: Path | None,
    batch_size: int,
    num_workers: int,
    *,
    train_config: SyncDatasetConfig,
    val_config: SyncDatasetConfig,
    return_metadata: bool,
    metadata_fields: List[str] | None,
    persistent_workers: bool,
    prefetch_factor: int,
) -> Tuple[DataLoader, DataLoader]:
    split = DatasetSplit()
    train_dataset = FakeAVSyncDataset(
        data_dir=data_dir,
        index_file=index_file,
        split="train",
        split_scheme=split,
        seed=1337,
        config=train_config,
        return_metadata=return_metadata,
        metadata_fields=metadata_fields,
    )
    val_dataset = FakeAVSyncDataset(
        data_dir=data_dir,
        index_file=index_file,
        split="val",
        split_scheme=split,
        seed=1337,
        config=val_config,
        return_metadata=return_metadata,
        metadata_fields=metadata_fields,
    )
    loader_kwargs: Dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = max(int(prefetch_factor), 2)
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


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


def compute_mcc(tp: float, fp: float, tn: float, fn: float) -> float:
    numerator = (tp * tn) - (fp * fn)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return float(numerator / denom)


def evaluate(
    model: SyncModule,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    cache_dir: Path | None,
    use_amp: bool,
    cache_dtype: str,
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    tp = fp = tn = fn = 0.0
    score_buffer: list[torch.Tensor] = []
    label_buffer: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                videos, audio_seq, sync_labels, _, metadata = batch
            else:
                videos, audio_seq, sync_labels, _ = batch
                metadata = None
            audio_seq = audio_seq.to(device, non_blocking=True)
            sync_labels = sync_labels.to(device, non_blocking=True)
            video_input, video_emb = prepare_video_inputs(
                model,
                videos,
                metadata,
                cache_dir,
                device,
                use_amp=use_amp,
                cache_dtype=cache_dtype,
            )
            with autocast_context(device, use_amp):
                _, logits = model(video_input, audio_seq, video_emb=video_emb)
            loss = criterion(logits, sync_labels)
            total_loss += loss.item() * sync_labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == sync_labels).sum().item()
            total_samples += sync_labels.size(0)
            tp += ((preds == 1) & (sync_labels == 1)).sum().item()
            fp += ((preds == 1) & (sync_labels == 0)).sum().item()
            tn += ((preds == 0) & (sync_labels == 0)).sum().item()
            fn += ((preds == 0) & (sync_labels == 1)).sum().item()
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
            score_buffer.append(probs)
            label_buffer.append(sync_labels.detach().cpu())
    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    if score_buffer:
        all_scores = torch.cat(score_buffer, dim=0)
        all_labels = torch.cat(label_buffer, dim=0)
        eer = compute_eer(all_scores, all_labels)
    else:
        eer = 0.5
    mcc = compute_mcc(tp, fp, tn, fn)
    metrics = {"eer": eer, "mcc": mcc, "tp": tp, "fp": fp, "tn": tn, "fn": fn}
    return avg_loss, accuracy, metrics


def train() -> None:
    args = parse_args()
    maybe_dump_args(args)
    device = torch.device(args.device)
    metrics_log_path = resolve_metrics_log_path(args.save_path, args.metrics_log_path)
    initialize_metrics_log(metrics_log_path, args)
    print(f"Per-epoch validation metrics will be written to {metrics_log_path}")
    cache_dir = Path(args.vit_cache_dir) if args.vit_cache_dir else None
    cache_enabled = cache_dir is not None
    vit_unfreeze_layers = max(int(args.vit_unfreeze_layers), 0)
    if cache_enabled and vit_unfreeze_layers > 0:
        print("ViT unfreezing enabled; disabling cached embeddings to keep features consistent.")
        cache_dir = None
        cache_enabled = False
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_fields = ["path", "speaker_id"] if cache_enabled else None
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = create_grad_scaler(device, use_amp)

    target_frames = max(int(args.target_frames), 1)
    train_config = SyncDatasetConfig(
        negative_prob=float(args.negative_prob),
        target_frames=target_frames,
        min_roll_ratio=float(args.roll_min_ratio),
        max_roll_ratio=float(args.roll_max_ratio),
        swap_prob=float(args.swap_prob),
        force_cross_speaker=bool(args.force_cross_speaker),
        paired_negatives=bool(args.train_paired_negatives),
    )
    val_config = SyncDatasetConfig(
        negative_prob=float(args.val_negative_prob),
        target_frames=target_frames,
        min_roll_ratio=float(args.roll_min_ratio),
        max_roll_ratio=float(args.roll_max_ratio),
        swap_prob=float(args.swap_prob),
        force_cross_speaker=bool(args.force_cross_speaker),
        paired_negatives=bool(args.val_paired_negatives),
    )

    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        index_file=args.index_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_config=train_config,
        val_config=val_config,
        return_metadata=cache_enabled,
        metadata_fields=metadata_fields,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    model = SyncModule(
        vit_path=args.vit_path,
        vit_unfreeze_layers=vit_unfreeze_layers,
        temporal_layers=max(int(args.temporal_layers), 1),
    ).to(device)
    criterion = build_criterion(
        label_smoothing=float(args.label_smoothing),
        focal_gamma=float(args.focal_gamma),
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_eer = float("inf")
    early_stop_patience = max(int(args.early_stop_patience), 0)
    target_val_eer = max(float(args.target_val_eer), 0.0)
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        running_loss = 0.0
        total_samples = 0
        for batch in progress:
            if len(batch) == 5:
                videos, audio_seq, sync_labels, _, metadata = batch
            else:
                videos, audio_seq, sync_labels, _ = batch
                metadata = None
            audio_seq = audio_seq.to(device, non_blocking=True)
            sync_labels = sync_labels.to(device, non_blocking=True)
            video_input, video_emb = prepare_video_inputs(
                model,
                videos,
                metadata,
                cache_dir,
                device,
                use_amp=use_amp,
                cache_dtype=args.cache_dtype,
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, use_amp):
                _, logits = model(video_input, audio_seq, video_emb=video_emb)
                loss = criterion(logits, sync_labels)
            loss_value = loss.item()
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss_value * sync_labels.size(0)
            total_samples += sync_labels.size(0)
            progress.set_postfix({"loss": running_loss / max(total_samples, 1)})

        val_loss, val_acc, val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            cache_dir=cache_dir,
            use_amp=use_amp,
            cache_dtype=args.cache_dtype,
        )
        val_eer = val_metrics.get("eer", float("inf"))
        val_line = format_epoch_metrics_line(epoch, val_loss, val_metrics)
        tqdm.write(val_line)
        append_metrics_log(metrics_log_path, val_line)

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_eer": val_eer,
                    "val_acc": val_acc,
                    "val_mcc": val_metrics.get("mcc", 0.0),
                    "val_stats": {
                        "tp": val_metrics.get("tp", 0.0),
                        "fp": val_metrics.get("fp", 0.0),
                        "tn": val_metrics.get("tn", 0.0),
                        "fn": val_metrics.get("fn", 0.0),
                    },
                    "args": vars(args),
                },
                args.save_path,
            )
            tqdm.write(f"Saved new best model to {args.save_path}")
            if target_val_eer > 0.0 and best_val_eer <= target_val_eer:
                tqdm.write(
                    f"Target val EER {target_val_eer:.4f} reached (current {best_val_eer:.4f}). Stopping early."
                )
                break
        else:
            epochs_no_improve += 1
            if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
                tqdm.write(
                    f"Early stopping triggered after {epoch} epochs (best val EER {best_val_eer:.4f})."
                )
                break


if __name__ == "__main__":
    train()
