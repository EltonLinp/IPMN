from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import (
    FakeAVMultimodalDataset,
    MultimodalDatasetConfig,
    DatasetSplit,
)
from models import MultimodalFusionModel


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    video = torch.stack([item["video"] for item in batch], dim=0)
    audio = torch.stack([item["audio"] for item in batch], dim=0)
    labels = torch.stack([item["label"] for item in batch], dim=0)
    frame_indices = [item["frame_indices"] for item in batch]
    frame_timestamps = [item["frame_timestamps"] for item in batch]
    mel_timestamps = [item["mel_timestamps"] for item in batch]
    return {
        "video": video,
        "audio": audio,
        "label": labels,
        "frame_indices": frame_indices,
        "frame_timestamps": frame_timestamps,
        "mel_timestamps": mel_timestamps,
    }


def build_dataloaders(
    data_dir: Path,
    index_file: Path | None,
    batch_size: int,
    num_workers: int,
    target_frames: int,
    target_steps: int,
) -> tuple[DataLoader, DataLoader]:
    split = DatasetSplit()
    train_dataset = FakeAVMultimodalDataset(
        data_dir=data_dir,
        index_file=index_file,
        split="train",
        split_scheme=split,
        seed=1337,
        config=MultimodalDatasetConfig(
            target_frames=target_frames,
            target_steps=target_steps,
            random_crop=True,
        ),
    )
    val_dataset = FakeAVMultimodalDataset(
        data_dir=data_dir,
        index_file=index_file,
        split="val",
        split_scheme=split,
        seed=1337,
        config=MultimodalDatasetConfig(
            target_frames=target_frames,
            target_steps=target_steps,
            random_crop=False,
        ),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    return train_loader, val_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multimodal FakeAVCeleb classifier.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Processed dataset root.")
    parser.add_argument("--vit-path", type=Path, required=True, help="Path to frozen ViT weights for sync module.")
    parser.add_argument("--index-file", type=Path, default=None, help="Optional preprocess_index.jsonl path.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-frames", type=int, default=32)
    parser.add_argument("--target-steps", type=int, default=400)
    parser.add_argument("--video-weight", type=float, default=0.0, help="Aux loss weight for video branch logits.")
    parser.add_argument("--audio-weight", type=float, default=0.0, help="Aux loss weight for audio branch logits.")
    parser.add_argument("--sync-weight", type=float, default=0.0, help="Aux loss weight for sync logits (aligned target).")
    parser.add_argument("--spectral-weight", type=float, default=0.0, help="Aux loss weight for spectral reconstruction.")
    parser.add_argument("--rppg-weight", type=float, default=0.0, help="Aux loss weight for rPPG regularisation.")
    parser.add_argument("--save-path", type=Path, default=Path("multimodal_model.pt"))
    return parser.parse_args()


def compute_losses(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    criterion: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Dict[str, float]]:
    labels = batch["label"]
    losses: Dict[str, torch.Tensor] = {}
    logits = outputs["logits"]
    losses["main"] = criterion(logits, labels)

    if args.video_weight > 0.0:
        losses["video"] = criterion(outputs["video_logits"], labels) * args.video_weight
    if args.audio_weight > 0.0:
        losses["audio"] = criterion(outputs["audio_logits"], labels) * args.audio_weight
    if args.sync_weight > 0.0:
        sync_target = torch.ones_like(labels)
        losses["sync"] = nn.functional.cross_entropy(outputs["sync_logits"], sync_target) * args.sync_weight
    if args.spectral_weight > 0.0:
        spectral_target = batch["audio"].mean(dim=1)
        losses["spectral"] = nn.functional.l1_loss(outputs["spectral"], spectral_target) * args.spectral_weight
    if args.rppg_weight > 0.0:
        losses["rppg"] = torch.mean(outputs["rppg"] ** 2) * args.rppg_weight

    total = sum(losses.values())
    log_dict = {name: loss.item() for name, loss in losses.items()}
    log_dict["total"] = total.item()
    return total, log_dict


def evaluate(
    model: MultimodalFusionModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in loader:
            videos = batch["video"].to(device, non_blocking=True)
            audio = batch["audio"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = model(videos, audio)
            loss, _ = compute_losses(outputs, {"label": labels, "audio": audio}, criterion, args)
            total_loss += loss.item() * labels.size(0)
            preds = outputs["logits"].argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def train() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        index_file=args.index_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_frames=args.target_frames,
        target_steps=args.target_steps,
    )

    model = MultimodalFusionModel(vit_path=args.vit_path).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        running_loss = 0.0
        total_samples = 0
        for batch in progress:
            videos = batch["video"].to(device, non_blocking=True)
            audio = batch["audio"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    outputs = model(videos, audio)
                    loss, log_dict = compute_losses(outputs, {"label": labels, "audio": audio}, criterion, args)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(videos, audio)
                loss, log_dict = compute_losses(outputs, {"label": labels, "audio": audio}, criterion, args)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            progress.set_postfix({k: f"{v:.3f}" for k, v in log_dict.items()})

        val_loss, val_acc = evaluate(model, val_loader, criterion, device, args)
        tqdm.write(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "args": vars(args),
                },
                args.save_path,
            )
            tqdm.write(f"Saved new best model to {args.save_path}")


if __name__ == "__main__":
    train()
