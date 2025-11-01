from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import FakeAVSyncDataset, SyncDatasetConfig, DatasetSplit
from models import SyncModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train synchronisation branch.")
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
    return parser.parse_args()


def build_dataloaders(
    data_dir: Path,
    index_file: Path | None,
    batch_size: int,
    num_workers: int,
    negative_prob: float,
    target_frames: int | None,
) -> Tuple[DataLoader, DataLoader]:
    split = DatasetSplit()
    train_dataset = FakeAVSyncDataset(
        data_dir=data_dir,
        index_file=index_file,
        split="train",
        split_scheme=split,
        seed=1337,
        config=SyncDatasetConfig(negative_prob=negative_prob, target_frames=target_frames),
        return_metadata=False,
    )
    val_dataset = FakeAVSyncDataset(
        data_dir=data_dir,
        index_file=index_file,
        split="val",
        split_scheme=split,
        seed=1337,
        config=SyncDatasetConfig(negative_prob=negative_prob, target_frames=target_frames),
        return_metadata=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def evaluate(model: SyncModule, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for videos, audio_seq, sync_labels, _ in loader:
            videos = videos.to(device, non_blocking=True)
            audio_seq = audio_seq.to(device, non_blocking=True)
            sync_labels = sync_labels.to(device, non_blocking=True)
            _, logits = model(videos, audio_seq)
            loss = criterion(logits, sync_labels)
            total_loss += loss.item() * sync_labels.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == sync_labels).sum().item()
            total_samples += sync_labels.size(0)
    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def train() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        index_file=args.index_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        negative_prob=args.negative_prob,
        target_frames=args.target_frames,
    )

    model = SyncModule(vit_path=args.vit_path).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        running_loss = 0.0
        total_samples = 0
        for videos, audio_seq, sync_labels, _ in progress:
            videos = videos.to(device, non_blocking=True)
            audio_seq = audio_seq.to(device, non_blocking=True)
            sync_labels = sync_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            _, logits = model(videos, audio_seq)
            loss = criterion(logits, sync_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * sync_labels.size(0)
            total_samples += sync_labels.size(0)
            progress.set_postfix({"loss": running_loss / max(total_samples, 1)})

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
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
