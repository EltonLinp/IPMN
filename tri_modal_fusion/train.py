from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from datasets import DatasetSplit
from models import SyncModule, VideoClassifier, WavLMClassifier, WavLMConfig
from tqdm import tqdm

from tri_modal_fusion import (
    FakeAVTriModalDataset,
    FusionConfig,
    MelAugmentation,
    ModelEMA,
    SpecAugParams,
    TriModalCollator,
    TriModalDatasetConfig,
    TriModalFusionModel,
    VideoAugmentation,
)
from tri_modal_fusion.model import load_state_partial


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, help="Optional JSON config overriding CLI args.")
    cfg_args, remaining = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="End-to-end tri-modal audio-sync-video training.",
        parents=[config_parser],
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing processed .pt bundles.")
    parser.add_argument("--index-file", type=Path, default=None, help="Optional preprocess_index.jsonl path.")
    parser.add_argument("--save-path", type=Path, default=Path("tri_modal_fusion.pt"), help="Checkpoint output path.")
    parser.add_argument("--epochs", type=int, default=24, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size per GPU.")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Prefetch multiplier when num_workers > 0.")
    parser.add_argument("--persistent-workers", action="store_true", help="Reuse workers between epochs.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-fusion", type=float, default=2e-4)
    parser.add_argument("--lr-audio-backbone", type=float, default=5e-6)
    parser.add_argument("--lr-audio-head", type=float, default=2e-4)
    parser.add_argument("--lr-video-backbone", type=float, default=5e-5)
    parser.add_argument("--lr-video-head", type=float, default=2e-4)
    parser.add_argument("--lr-sync", type=float, default=3e-5)
    parser.add_argument("--scheduler", choices=("cosine", "plateau", "none"), default="cosine")
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--video-unfreeze-epoch", type=int, default=3)
    parser.add_argument("--sync-unfreeze-epoch", type=int, default=1)
    parser.add_argument("--contrastive-margin", type=float, default=0.3)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temp", type=float, default=0.1)
    parser.add_argument("--audio-loss-weight", type=float, default=0.4)
    parser.add_argument("--video-loss-weight", type=float, default=0.8)
    parser.add_argument("--sync-loss-weight", type=float, default=0.6)
    parser.add_argument("--alignment-weight", type=float, default=0.3)
    parser.add_argument("--distill-temp", type=float, default=2.0)
    parser.add_argument("--distill-weight", type=float, default=0.25)
    parser.add_argument("--distill-epochs", type=int, default=5)
    parser.add_argument("--audio-teacher", type=Path, default=None)
    parser.add_argument("--video-teacher", type=Path, default=None)
    parser.add_argument("--sync-teacher", type=Path, default=None)
    parser.add_argument("--audio-ckpt", type=Path, default=None)
    parser.add_argument("--video-ckpt", type=Path, default=None)
    parser.add_argument("--sync-ckpt", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--target-steps", type=int, default=400)
    parser.add_argument("--video-frames", type=int, default=48)
    parser.add_argument("--sync-frames", type=int, default=16)
    parser.add_argument("--sync-audio-steps", type=int, default=64)
    parser.add_argument("--waveform-samples", type=int, default=160000)
    parser.add_argument("--video-size", type=int, default=224)
    parser.add_argument("--mel-bins", type=int, default=80)
    parser.add_argument("--sync-temporal-layers", type=int, default=2)
    parser.add_argument("--cross-attn-layers", type=int, default=1)
    parser.add_argument("--video-backbone", choices=("light", "r3d18"), default="light")
    parser.add_argument("--video-pretrained", action="store_true")
    parser.add_argument("--video-dropout", type=float, default=0.3)
    parser.add_argument("--sync-negative-prob", type=float, default=0.4)
    parser.add_argument("--sync-max-shift", type=int, default=4)
    parser.add_argument("--audio-augment-prob", type=float, default=0.45)
    parser.add_argument("--audio-augment-freq", type=int, default=8)
    parser.add_argument("--audio-augment-time", type=int, default=48)
    parser.add_argument("--audio-noise-std", type=float, default=0.002)
    parser.add_argument("--audio-gain-std", type=float, default=0.05)
    parser.add_argument("--audio-shift-pct", type=float, default=0.08)
    parser.add_argument("--pseudo-fake-prob", type=float, default=0.15)
    parser.add_argument("--pseudo-fake-freq", type=int, default=20)
    parser.add_argument("--pseudo-fake-time", type=int, default=64)
    parser.add_argument("--video-flip-prob", type=float, default=0.5)
    parser.add_argument("--video-temporal-jitter", type=int, default=4)
    parser.add_argument("--video-brightness", type=float, default=0.12)
    parser.add_argument("--video-contrast", type=float, default=0.18)
    parser.add_argument("--video-noise-std", type=float, default=0.02)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--wavlm-checkpoint", type=Path, default=None, help="Optional local WavLM weights.")
    parser.add_argument("--wavlm-train-backbone", action="store_true")
    parser.add_argument("--wavlm-unfreeze-layers", type=int, default=8)
    parser.add_argument("--hf-local-only", action="store_true")
    parser.add_argument("--sync-vit-path", type=Path, default=Path("vit_model"))

    if cfg_args.config:
        with cfg_args.config.open("r", encoding="utf-8") as handle:
            config_data = json.load(handle)
        parser.set_defaults(**config_data)

    args = parser.parse_args(remaining)
    if args.data_dir is None:
        parser.error("--data-dir is required (pass via CLI or config JSON)")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_eer(scores: torch.Tensor, labels: torch.Tensor) -> float:
    labels = labels.float()
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
    eer = 0.5 * (fpr[idx] + fnr[idx])
    return float(eer.item())


def build_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    common_cfg = dict(
        target_steps=args.target_steps,
        video_frames=args.video_frames,
        sync_audio_steps=args.sync_audio_steps,
        sync_video_frames=args.sync_frames,
        waveform_samples=args.waveform_samples,
        sync_negative_prob=args.sync_negative_prob,
        sync_max_shift=args.sync_max_shift,
        video_size=args.video_size,
        mel_bins=args.mel_bins,
    )
    train_cfg = TriModalDatasetConfig(mel_random_crop=True, **common_cfg)
    val_cfg = TriModalDatasetConfig(mel_random_crop=False, **common_cfg)
    val_cfg.sync_negative_prob = 0.0
    audio_params = SpecAugParams(
        freq_mask=args.audio_augment_freq,
        time_mask=args.audio_augment_time,
        prob=args.audio_augment_prob,
        noise_std=args.audio_noise_std,
        gain_std=args.audio_gain_std,
        shift_pct=args.audio_shift_pct,
        pseudo_fake_prob=args.pseudo_fake_prob,
        pseudo_fake_freq=args.pseudo_fake_freq,
        pseudo_fake_time=args.pseudo_fake_time,
    )
    audio_aug = MelAugmentation(audio_params)
    video_aug = VideoAugmentation(
        horizontal_flip_prob=args.video_flip_prob,
        temporal_jitter=args.video_temporal_jitter,
        brightness=args.video_brightness,
        contrast=args.video_contrast,
        noise_std=args.video_noise_std,
    )
    split = DatasetSplit()
    train_dataset = FakeAVTriModalDataset(
        data_dir=args.data_dir,
        index_file=args.index_file,
        split="train",
        split_scheme=split,
        seed=args.seed,
        config=train_cfg,
        audio_augment=audio_aug,
        video_augment=video_aug,
        train=True,
        return_metadata=True,
    )
    val_dataset = FakeAVTriModalDataset(
        data_dir=args.data_dir,
        index_file=args.index_file,
        split="val",
        split_scheme=split,
        seed=args.seed,
        config=val_cfg,
        audio_augment=None,
        video_augment=None,
        train=False,
        return_metadata=True,
    )
    collator = TriModalCollator()
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
        **{k: v for k, v in loader_kwargs.items() if v is not None},
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        **{k: v for k, v in loader_kwargs.items() if v is not None},
    )
    return train_loader, val_loader


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def build_model(args: argparse.Namespace) -> TriModalFusionModel:
    wavlm_cfg = WavLMConfig(
        model_name=str(args.wavlm_checkpoint) if args.wavlm_checkpoint else "microsoft/wavlm-base-plus-sv",
        dropout=0.2,
        train_backbone=args.wavlm_train_backbone,
        unfreeze_layers=args.wavlm_unfreeze_layers,
        num_classes=args.num_classes,
        local_files_only=args.hf_local_only,
    )
    fusion_cfg = FusionConfig(
        num_classes=args.num_classes,
        fusion_dim=512,
        cross_heads=4,
        cross_layers=2,
        dropout=0.2,
        sync_vit_path=args.sync_vit_path,
        sync_audio_dim=args.mel_bins,
        video_backbone=args.video_backbone,
        video_pretrained=args.video_pretrained,
        video_dropout=args.video_dropout,
        sync_temporal_layers=args.sync_temporal_layers,
        cross_attn_layers=args.cross_attn_layers,
        wavlm=wavlm_cfg,
    )
    model = TriModalFusionModel(fusion_cfg)
    if args.audio_ckpt:
        model.load_branch_checkpoint("audio", args.audio_ckpt)
    if args.video_ckpt:
        model.load_branch_checkpoint("video", args.video_ckpt)
    if args.sync_ckpt:
        model.load_branch_checkpoint("sync", args.sync_ckpt)
    return model


def build_optimizer(model: TriModalFusionModel, args: argparse.Namespace) -> optim.Optimizer:
    groups = model.parameter_groups()
    params = [
        {"params": groups["fusion_head"], "lr": args.lr_fusion},
        {"params": groups["audio_backbone"], "lr": args.lr_audio_backbone},
        {"params": groups["audio_head"], "lr": args.lr_audio_head},
        {"params": groups["video_backbone"], "lr": args.lr_video_backbone},
        {"params": groups["video_heads"], "lr": args.lr_video_head},
        {"params": groups["sync_branch"], "lr": args.lr_sync},
    ]
    optimizer = optim.AdamW(params, weight_decay=args.weight_decay)
    return optimizer


def build_scheduler(optimizer: optim.Optimizer, args: argparse.Namespace):
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            verbose=True,
        )
    return None


def _match_embedding_dims(
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = min(source.size(1), target.size(1))
    return source[:, :dim], target[:, :dim]


def alignment_loss(
    sync_emb: torch.Tensor,
    audio_emb: torch.Tensor,
    video_emb: torch.Tensor,
    sync_labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    sync_audio, audio_emb = _match_embedding_dims(sync_emb, audio_emb)
    sync_video, video_emb = _match_embedding_dims(sync_emb, video_emb)

    aligned = sync_labels == 0
    misaligned = sync_labels == 1
    loss = torch.tensor(0.0, device=sync_emb.device)
    if aligned.any():
        loss = loss + F.mse_loss(sync_audio[aligned], audio_emb[aligned])
        loss = loss + F.mse_loss(sync_video[aligned], video_emb[aligned])
    if misaligned.any():
        dist_audio = torch.norm(sync_audio[misaligned] - audio_emb[misaligned], dim=1)
        dist_video = torch.norm(sync_video[misaligned] - video_emb[misaligned], dim=1)
        loss = loss + F.relu(margin - dist_audio).mean() + F.relu(margin - dist_video).mean()
    return loss


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, temperature: float) -> torch.Tensor:
    anchor, positive = _match_embedding_dims(anchor, positive)
    if anchor.size(0) == 0:
        return torch.tensor(0.0, device=anchor.device)
    anchor = F.normalize(anchor, dim=1)
    positive = F.normalize(positive, dim=1)
    logits = torch.matmul(anchor, positive.transpose(0, 1)) / max(temperature, 1e-6)
    labels = torch.arange(anchor.size(0), device=anchor.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def distillation_loss(student: torch.Tensor, teacher: torch.Tensor, temp: float) -> torch.Tensor:
    if teacher is None:
        return torch.tensor(0.0, device=student.device)
    log_q = F.log_softmax(student / temp, dim=1)
    p = F.softmax(teacher / temp, dim=1)
    return F.kl_div(log_q, p, reduction="batchmean") * (temp**2)


def run_eval(
    model: TriModalFusionModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    all_scores = {"final": [], "audio": [], "video": [], "sync": []}
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            video_sync = batch["video_sync"]
            with autocast("cuda", enabled=use_amp):
                outputs = model(
                    waveform=batch["waveform"],
                    waveform_lengths=batch["waveform_lengths"],
                    mel_sync=batch["mel_sync"],
                    video=batch["video"],
                    video_sync=video_sync,
                )
            for key, scores in [
                ("final", outputs["logits"]),
                ("audio", outputs["audio_logits"]),
                ("video", outputs["video_logits"]),
                ("sync", outputs["sync_logits"]),
            ]:
                probs = torch.softmax(scores, dim=1)[:, 1]
                all_scores[key].append(probs.detach().cpu())
            all_labels.append(batch["label"].detach().cpu())
    results = {}
    labels = torch.cat(all_labels, dim=0)
    for key, tensors in all_scores.items():
        if tensors:
            scores = torch.cat(tensors, dim=0)
            results[f"{key}_eer"] = compute_eer(scores, labels)
    return results


def maybe_build_teacher(
    branch: str,
    args: argparse.Namespace,
    student: TriModalFusionModel,
    device: torch.device,
) -> Optional[torch.nn.Module]:
    ckpt_path = getattr(args, f"{branch}_teacher", None)
    if not ckpt_path:
        return None
    branch = branch.lower()
    if branch == "audio":
        teacher = WavLMClassifier(student.cfg.wavlm)
    elif branch == "video":
        teacher = VideoClassifier(
            num_classes=student.cfg.num_classes,
            backbone=student.cfg.video_backbone,
            pretrained=False,
            dropout=student.cfg.video_dropout,
        )
    else:
        teacher = SyncModule(
            vit_path=student.cfg.sync_vit_path,
            audio_dim=student.cfg.sync_audio_dim,
            transformer_heads=student.cfg.sync_transformer_heads,
            dropout=student.cfg.dropout,
            temporal_layers=student.cfg.sync_temporal_layers,
        )
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    load_state_partial(teacher, state, branch=f"{branch}_teacher", strict=False)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def save_checkpoint(
    path: Path,
    model: TriModalFusionModel,
    optimizer: optim.Optimizer,
    scaler: Optional[GradScaler],
    epoch: int,
    best_eer: float,
    args: argparse.Namespace,
    ema: Optional[ModelEMA],
) -> None:
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "best_eer": best_eer,
        "args": vars(args),
        "ema": ema.state_dict() if ema is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(
    path: Path,
    model: TriModalFusionModel,
    optimizer: optim.Optimizer,
    scaler: Optional[GradScaler],
    ema: Optional[ModelEMA],
) -> Tuple[int, float]:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    if scaler is not None and state.get("scaler_state"):
        scaler.load_state_dict(state["scaler_state"])
    if ema is not None and state.get("ema"):
        ema.shadow = state["ema"]
    return state.get("epoch", 0), state.get("best_eer", math.inf)


def train() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    train_loader, val_loader = build_dataloaders(args)
    model = build_model(args).to(device)
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)
    scaler = GradScaler("cuda", enabled=args.amp)
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    start_epoch = 1
    best_eer = math.inf
    if args.resume and Path(args.resume).exists():
        start_epoch, best_eer = load_checkpoint(Path(args.resume), model, optimizer, scaler, ema)
        start_epoch += 1
    teacher_audio = maybe_build_teacher("audio", args, model, device)
    teacher_video = maybe_build_teacher("video", args, model, device)
    teacher_sync = maybe_build_teacher("sync", args, model, device)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        freeze_video = epoch <= args.video_unfreeze_epoch
        model.freeze_video_backbone(freeze_video)
        model.freeze_sync_backbone(epoch <= args.sync_unfreeze_epoch)
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)
            video_sync = batch["video_sync"]
            with autocast("cuda", enabled=args.amp):
                outputs = model(
                    waveform=batch["waveform"],
                    waveform_lengths=batch["waveform_lengths"],
                    mel_sync=batch["mel_sync"],
                    video=batch["video"],
                    video_sync=video_sync,
                )
                loss_main = F.cross_entropy(outputs["logits"], batch["label"])
                aux_audio = F.cross_entropy(outputs["audio_logits"], batch["label"])
                aux_video = F.cross_entropy(outputs["video_logits"], batch["label"])
                aux_sync = F.cross_entropy(outputs["sync_logits"], batch["label"])
                align = alignment_loss(
                    outputs["sync_embedding"],
                    outputs["audio_embedding"],
                    outputs["video_embedding"],
                    batch["sync_label"],
                    args.contrastive_margin,
                )
                loss = loss_main
                loss = loss + args.audio_loss_weight * aux_audio
                loss = loss + args.video_loss_weight * aux_video
                loss = loss + args.sync_loss_weight * aux_sync
                loss = loss + args.alignment_weight * align
                if args.contrastive_weight > 0.0:
                    contrast_audio = info_nce_loss(
                        outputs["sync_embedding"],
                        outputs["audio_embedding"],
                        args.contrastive_temp,
                    )
                    contrast_video = info_nce_loss(
                        outputs["sync_embedding"],
                        outputs["video_embedding"],
                        args.contrastive_temp,
                    )
                    contrastive = 0.5 * (contrast_audio + contrast_video)
                    loss = loss + args.contrastive_weight * contrastive
                if epoch <= args.distill_epochs and args.distill_weight > 0.0:
                    if teacher_audio is not None:
                        t_logits, _, _ = teacher_audio(None, waveform=batch["waveform"], waveform_lengths=batch["waveform_lengths"])
                        loss = loss + args.distill_weight * distillation_loss(outputs["audio_logits"], t_logits, args.distill_temp)
                    if teacher_video is not None:
                        t_logits, _, _ = teacher_video(batch["video"])
                        loss = loss + args.distill_weight * distillation_loss(outputs["video_logits"], t_logits, args.distill_temp)
                    if teacher_sync is not None:
                        t_embed, t_logits = teacher_sync(video_sync, batch["mel_sync"])
                        loss = loss + args.distill_weight * distillation_loss(outputs["sync_logits"], t_logits, args.distill_temp)
                loss = loss / args.accumulation_steps
            scaler.scale(loss).backward()
            if step % args.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
            running_loss += loss.item() * args.accumulation_steps
            progress.set_postfix(loss=f"{running_loss / step:.4f}")
        progress.close()

        remainder = len(train_loader) % args.accumulation_steps
        if remainder != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        if scheduler is not None and args.scheduler != "plateau":
            scheduler.step()

        if epoch % args.eval_interval == 0:
            eval_model = model
            if ema is not None:
                ema_model = TriModalFusionModel(model.cfg).to(device)
                ema_model.load_state_dict(model.state_dict())
                ema.copy_to(ema_model)
                eval_model = ema_model
            metrics = run_eval(eval_model, val_loader, device, args.amp)
            final_eer = metrics.get("final_eer", math.inf)
            print(
                f"[Val] Epoch {epoch} EER final={final_eer:.4f} "
                f"audio={metrics.get('audio_eer', 0):.4f} "
                f"video={metrics.get('video_eer', 0):.4f} "
                f"sync={metrics.get('sync_eer', 0):.4f}"
            )
            if scheduler is not None and args.scheduler == "plateau":
                scheduler.step(final_eer)
            if final_eer < best_eer:
                best_eer = final_eer
                save_checkpoint(args.save_path, model, optimizer, scaler, epoch, best_eer, args, ema)


if __name__ == "__main__":
    train()
