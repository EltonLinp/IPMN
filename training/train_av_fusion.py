from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import AudioDatasetConfig, FakeAVAudioDataset
from models import SyncModule, VideoClassifier, WavLMClassifier, WavLMConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse audio+sync and video classifiers.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with processed .pt files.")
    parser.add_argument("--index-file", type=Path, default=None, help="Optional preprocess_index.jsonl path.")
    parser.add_argument("--audio-ckpt", type=Path, required=True, help="Checkpoint produced by train_audio_sync.")
    parser.add_argument("--video-ckpt", type=Path, required=True, help="Checkpoint produced by train_video.")
    parser.add_argument("--save-path", type=Path, default=Path("av_fusion.pt"), help="Output checkpoint path.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--target-steps", type=int, default=400)
    parser.add_argument("--video-frames", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--audio-only", action="store_true", help="Use only audio logits when true (debug).")
    return parser.parse_args()


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


def prepare_waveform_segments(
    waveform: torch.Tensor | None,
    lengths: torch.Tensor | None,
    *,
    segment_samples: int,
    train: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if waveform is None or waveform.numel() == 0 or segment_samples <= 0:
        return waveform, lengths
    seg = max(int(segment_samples), 1)
    if lengths is None:
        lengths = torch.full((waveform.size(0),), waveform.size(-1), dtype=torch.long, device=waveform.device)
    processed = waveform.new_zeros((waveform.size(0), seg))
    out_lengths = torch.full((waveform.size(0),), seg, dtype=torch.long, device=waveform.device)
    for idx in range(waveform.size(0)):
        length = int(lengths[idx].item())
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
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if waveform is None or waveform.numel() == 0 or mode == "none":
        return None, None
    wave = waveform
    if lengths is None:
        lengths = torch.full((wave.size(0),), wave.size(-1), dtype=torch.long, device=wave.device)
    else:
        lengths = lengths.clone()
    if mode == "real_only":
        mask = labels == 0
    elif mode == "fake_only":
        mask = labels == 1
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
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if waveform is None or waveform.numel() == 0:
        return None, None
    if audio_backbone == "wavlm":
        return prepare_waveform_segments(
            waveform,
            lengths,
            segment_samples=segment_samples,
            train=train,
        )
    return process_waveform_branch(
        waveform,
        lengths,
        labels,
        mode=wave_branch_mode,
        segment_samples=segment_samples,
        train=train,
    )


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
    if vid.size(2) != target_frames:
        if vid.size(2) > target_frames:
            start = max((vid.size(2) - target_frames) // 2, 0)
            vid = vid[:, :, start : start + target_frames]
        else:
            pad = target_frames - vid.size(2)
            before = pad // 2
            after = pad - before
            first = vid[:, :, :1].repeat(1, 1, before, 1, 1) if before > 0 else None
            last = vid[:, :, -1:].repeat(1, 1, after, 1, 1) if after > 0 else None
            pieces = [p for p in (first, vid, last) if p is not None]
            vid = torch.cat(pieces, dim=2)
    if mels.dim() == 4:
        mel_2d = mels.squeeze(1)
    elif mels.dim() == 3:
        mel_2d = mels
    else:
        mel_2d = mels.unsqueeze(1).squeeze(1)
    mel_resized = F.interpolate(
        mel_2d.unsqueeze(1),
        size=(max(int(audio_dim), 1), target_frames),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    audio_seq = mel_resized.permute(0, 2, 1).contiguous()
    return vid.contiguous(), audio_seq


def apply_sync_gating(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    *,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    if teacher_logits is None or alpha <= 0.0:
        return logits
    margin = (teacher_logits[:, 1] - teacher_logits[:, 0]).unsqueeze(1)
    gate = torch.sigmoid(alpha * (margin - beta))
    return logits * (1.0 - gate) + teacher_logits * gate


class AudioSyncPredictor(nn.Module):
    def __init__(self, ckpt_path: Path, device: torch.device) -> None:
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        args = SimpleNamespace(**ckpt.get("args", {}))
        if args.audio_backbone != "wavlm":
            raise ValueError("AV fusion currently expects audio checkpoints trained with WavLM backbone.")
        wavlm_cfg = WavLMConfig(
            model_name=args.wavlm_model_name,
            num_classes=2,
            dropout=float(args.wavlm_dropout),
            train_backbone=False,
            unfreeze_layers=0,
            local_files_only=bool(args.hf_local_files_only),
        )
        self.audio_backbone = args.audio_backbone
        self.wave_branch_mode = args.wave_branch_mode
        self.wave_segment_samples = int(args.wave_segment_samples)
        self.sync_target_frames = int(getattr(args, "sync_target_frames", 16))
        self.sync_audio_dim = int(getattr(args, "sync_audio_dim", 64))
        self.sync_gate_alpha = float(getattr(args, "sync_gate_alpha", 0.0))
        self.sync_gate_beta = float(getattr(args, "sync_gate_beta", 0.0))
        self.model = WavLMClassifier(wavlm_cfg).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        ckpt_sync = ckpt.get("sync_module_state")
        if ckpt_sync is None:
            ckpt_sync = torch.load(args.sync_checkpoint, map_location="cpu", weights_only=False).get("model_state")
        self.sync_module = SyncModule(
            vit_path=args.sync_vit_path,
            audio_dim=self.sync_audio_dim,
            transformer_heads=int(args.sync_transformer_heads),
            dropout=float(args.sync_fusion_dropout),
            vit_unfreeze_layers=0,
            temporal_layers=int(args.sync_temporal_layers),
        ).to(device)
        self.sync_module.load_state_dict(ckpt_sync, strict=False)
        self.sync_module.eval()
        self.sync_head = AudioSyncFusionHead(
            audio_dim=self.model.hidden,
            sync_hidden_dim=self.sync_module.hidden_dim,
            dropout=float(args.sync_fusion_dropout),
            num_heads=int(args.sync_transformer_heads),
        ).to(device)
        self.sync_head.load_state_dict(ckpt["sync_head_state"])
        self.sync_head.eval()

    @torch.no_grad()
    def forward(
        self,
        mels: torch.Tensor,
        labels: torch.Tensor,
        waveform: torch.Tensor | None,
        waveform_lengths: torch.Tensor | None,
        video: torch.Tensor | None,
    ) -> torch.Tensor:
        waveform_proc, waveform_len_proc = prepare_model_waveform_inputs(
            waveform,
            waveform_lengths,
            labels,
            audio_backbone=self.audio_backbone,
            wave_branch_mode=self.wave_branch_mode,
            segment_samples=self.wave_segment_samples,
            train=False,
        )
        logits, _, embed = self.model(mels, waveform=waveform_proc, waveform_lengths=waveform_len_proc)
        sync_feat = None
        sync_logits = None
        if video is not None:
            sync_video_input, sync_audio_seq = prepare_sync_inputs(
                mels,
                video,
                target_frames=self.sync_target_frames,
                audio_dim=self.sync_audio_dim,
            )
            if sync_video_input is not None and sync_audio_seq is not None:
                sync_feat, sync_logits = self.sync_module(video=sync_video_input, audio_seq=sync_audio_seq)
        if sync_feat is not None:
            logits = self.sync_head(embed, sync_feat)
            logits = apply_sync_gating(
                logits,
                sync_logits,
                alpha=self.sync_gate_alpha,
                beta=self.sync_gate_beta,
            )
        return logits


class AudioVideoCollator:
    def __init__(self, include_waveform: bool, include_video: bool) -> None:
        self.include_waveform = include_waveform
        self.include_video = include_video

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        if not batch:
            return {}
        mels: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        waveforms: list[torch.Tensor] = []
        lengths: list[int] = []
        videos: list[torch.Tensor] = []
        for sample in batch:
            mel = sample["mel"].float()
            label = sample["label"].long()
            mels.append(mel)
            labels.append(label)
            if self.include_waveform:
                waveform = sample["waveform"].float()
                if waveform.dim() == 2:
                    waveform = waveform.mean(dim=0)
                lengths.append(int(waveform.size(0)))
                waveforms.append(waveform)
            if self.include_video:
                video = sample["video"].float()
                if video.dim() == 4:  # [T, C, H, W]
                    video = video.permute(1, 0, 2, 3)
                videos.append(video)
        payload: Dict[str, object] = {
            "mel": torch.stack(mels, dim=0),
            "label": torch.stack(labels, dim=0),
        }
        if self.include_waveform:
            payload["waveform"] = pad_sequence(waveforms, batch_first=True)
            payload["waveform_length"] = torch.tensor(lengths, dtype=torch.long)
        if self.include_video:
            video_tensor = torch.stack(videos, dim=0)
            payload["video"] = video_tensor
        return payload


class AVFusionHead(nn.Module):
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.net(logits)


def run_eval(
    fusion_head: AVFusionHead,
    loader: DataLoader,
    *,
    audio_model: AudioSyncPredictor,
    video_model: VideoClassifier,
    device: torch.device,
    audio_only: bool,
) -> tuple[float, float, Dict[str, float]]:
    fusion_head.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    tp = fp = tn = fn = 0.0
    scores: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            mels = batch["mel"].to(device)
            waveform = batch["waveform"].to(device)
            waveform_lengths = batch["waveform_length"].to(device)
            video = batch["video"].to(device)
            audio_logits = audio_model(mels, labels, waveform, waveform_lengths, video)
            video_logits, _, _ = video_model(video)
            fusion_input = audio_logits if audio_only else torch.cat([audio_logits, video_logits], dim=1)
            logits = fusion_head(fusion_input)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            preds = logits.argmax(dim=1)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
            scores.append(torch.softmax(logits, dim=1)[:, 1].cpu())
            labels_all.append(labels.cpu())
    avg_loss = total_loss / max(total, 1)
    score_cat = torch.cat(scores) if scores else torch.zeros(1)
    label_cat = torch.cat(labels_all) if labels_all else torch.zeros(1)
    eer = compute_eer(score_cat, label_cat)
    stats = {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    return avg_loss, eer, stats


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    audio_predictor = AudioSyncPredictor(args.audio_ckpt, device)
    audio_predictor.eval()
    for param in audio_predictor.parameters():
        param.requires_grad_(False)
    video_ckpt = torch.load(args.video_ckpt, map_location="cpu", weights_only=False)
    video_args = SimpleNamespace(**video_ckpt.get("args", {}))
    video_model = VideoClassifier(
        num_classes=2,
        dropout=float(getattr(video_args, "dropout", 0.3)),
        backbone=getattr(video_args, "backbone", "light"),
    ).to(device)
    video_model.load_state_dict(video_ckpt["model_state"])
    video_model.eval()
    for param in video_model.parameters():
        param.requires_grad_(False)
    dataset_cfg = AudioDatasetConfig(
        target_steps=int(args.target_steps),
        random_crop=True,
        video_target_frames=int(args.video_frames),
    )
    train_dataset = FakeAVAudioDataset(
        data_dir=args.data_dir,
        index_file=args.index_file,
        split="train",
        config=dataset_cfg,
        return_waveform=True,
        return_video=True,
    )
    val_dataset = FakeAVAudioDataset(
        data_dir=args.data_dir,
        index_file=args.index_file,
        split="val",
        config=AudioDatasetConfig(target_steps=args.target_steps, random_crop=False, video_target_frames=args.video_frames),
        return_waveform=True,
        return_video=True,
    )
    test_dataset = FakeAVAudioDataset(
        data_dir=args.data_dir,
        index_file=args.index_file,
        split="test",
        config=AudioDatasetConfig(target_steps=args.target_steps, random_crop=False, video_target_frames=args.video_frames),
        return_waveform=True,
        return_video=True,
    )
    collator = AudioVideoCollator(include_waveform=True, include_video=True)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "collate_fn": collator,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    fusion_input_dim = 2 if args.audio_only else 4
    fusion_head = AVFusionHead(fusion_input_dim).to(device)
    optimizer = optim.AdamW(fusion_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    def run_epoch(loader: DataLoader, train: bool) -> tuple[float, float]:
        fusion_head.train(train)
        epoch_loss = 0.0
        total = 0
        for batch in loader:
            labels = batch["label"].to(device)
            mels = batch["mel"].to(device)
            waveform = batch["waveform"].to(device)
            waveform_lengths = batch["waveform_length"].to(device)
            video = batch["video"].to(device)  # [B, C, T, H, W]
            audio_logits = audio_predictor(
                mels,
                labels,
                waveform,
                waveform_lengths,
                video,
            )
            with torch.no_grad():
                video_logits, _, _ = video_model(video)
            if args.audio_only:
                fusion_input = audio_logits
            else:
                fusion_input = torch.cat([audio_logits, video_logits], dim=1)
            if train:
                optimizer.zero_grad(set_to_none=True)
                preds = fusion_head(fusion_input)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    preds = fusion_head(fusion_input)
                    loss = criterion(preds, labels)
            epoch_loss += loss.item() * labels.size(0)
            total += labels.size(0)
        return epoch_loss / max(total, 1), total

    best_eer = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(train_loader, train=True)
        if epoch % args.eval_every == 0:
            val_loss, val_eer, val_stats = run_eval(
                fusion_head,
                val_loader,
                audio_model=audio_predictor,
                video_model=video_model,
                device=device,
                audio_only=args.audio_only,
            )
            tqdm.write(
                f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_eer={val_eer:.4f}, TP={val_stats['tp']:.0f}, FP={val_stats['fp']:.0f}, "
                f"TN={val_stats['tn']:.0f}, FN={val_stats['fn']:.0f}"
            )
            if val_eer < best_eer:
                best_eer = val_eer
                best_state = fusion_head.state_dict()
                torch.save(
                    {
                        "fusion_state": best_state,
                        "audio_ckpt": str(args.audio_ckpt),
                        "video_ckpt": str(args.video_ckpt),
                        "args": vars(args),
                        "val_eer": val_eer,
                    },
                    args.save_path,
                )
                tqdm.write(f"Saved new best fusion head to {args.save_path}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    tqdm.write("Early stopping triggered (fusion head).")
                    break

    if best_state is not None:
        fusion_head.load_state_dict(best_state)

    test_loss, test_eer, test_stats = run_eval(
        fusion_head,
        test_loader,
        audio_model=audio_predictor,
        video_model=video_model,
        device=device,
        audio_only=args.audio_only,
    )
    print(
        f"Test loss={test_loss:.4f}, test_eer={test_eer:.4f}, "
        f"TP={test_stats['tp']:.0f}, FP={test_stats['fp']:.0f}, "
        f"TN={test_stats['tn']:.0f}, FN={test_stats['fn']:.0f}"
    )


class AudioSyncFusionHead(nn.Module):
    def __init__(self, audio_dim: int, sync_hidden_dim: int, *, dropout: float, num_heads: int) -> None:
        super().__init__()
        self.sync_hidden = sync_hidden_dim
        self.sync_proj = nn.Linear(sync_hidden_dim, audio_dim)
        self.attn = nn.MultiheadAttention(embed_dim=audio_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
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
                segs = max(segs, 1)
                sync_seq = sync_joint.view(sync_joint.size(0), segs, self.sync_hidden)
            else:
                sync_seq = sync_joint
            sync_proj = self.sync_proj(sync_seq)
            query = audio_embed.unsqueeze(1)
            pad_mask = torch.zeros(sync_proj.size(0), sync_proj.size(1), dtype=torch.bool, device=sync_proj.device)
            attn_out, _ = self.attn(query, sync_proj, sync_proj, key_padding_mask=pad_mask, attn_mask=None)
            fused = (query + attn_out).squeeze(1)
        return self.ffn(fused)


if __name__ == "__main__":
    main()
