from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from pathlib import Path
from typing import Dict, List
import warnings

import torch
import torch.nn as nn

from models import SyncModule, VideoClassifier, WavLMClassifier, WavLMConfig

try:
    from torch.serialization import add_safe_globals as _torch_add_safe_globals
except ImportError:  # pragma: no cover
    _torch_add_safe_globals = None

if _torch_add_safe_globals is not None:
    safe_classes = []
    for name in ("PosixPath", "WindowsPath"):
        cls = getattr(pathlib, name, None)
        if cls is not None:
            safe_classes.append(cls)
    if safe_classes:
        _torch_add_safe_globals(safe_classes)


def load_state_partial(
    module: nn.Module,
    state: Dict[str, torch.Tensor],
    *,
    branch: str,
    strict: bool = False,
) -> None:
    module_state = module.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for name, tensor in state.items():
        target = module_state.get(name)
        if target is None:
            skipped.append(f"{name} (missing in model)")
            continue
        if target.shape != tensor.shape:
            skipped.append(f"{name} (checkpoint {tuple(tensor.shape)} != model {tuple(target.shape)})")
            continue
        filtered[name] = tensor
    if skipped:
        warnings.warn(
            f"Skipped loading {len(skipped)} parameters for '{branch}' branch:\n  "
            + "\n  ".join(skipped[:10])
            + ("..." if len(skipped) > 10 else ""),
            RuntimeWarning,
        )
    missing_keys = [k for k in module_state.keys() if k not in filtered]
    module.load_state_dict(filtered, strict=False)
    if strict and (missing_keys or skipped):
        raise RuntimeError(
            f"Strict loading requested but {len(missing_keys)} keys were missing "
            f"and {len(skipped)} keys were skipped for branch '{branch}'."
        )


@dataclass
class FusionConfig:
    num_classes: int = 2 
    fusion_dim: int = 512
    cross_heads: int = 4
    cross_layers: int = 2
    cross_attn_layers: int = 1
    dropout: float = 0.2
    sync_vit_path: str | Path = Path("res/vit_model")
    sync_audio_dim: int = 80
    sync_transformer_heads: int = 8
    sync_temporal_layers: int = 2
    video_backbone: str = "light"
    video_pretrained: bool = False
    video_dropout: float = 0.3
    wavlm: WavLMConfig = field(default_factory=WavLMConfig)


class TriModalFusionModel(nn.Module):
    """
    End-to-end tri-modal network combining waveform (WavLM), sync branch, and video backbone.
    """

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        self.cfg = config
        self.audio_branch = WavLMClassifier(config.wavlm)
        self.video_branch = VideoClassifier(
            num_classes=config.num_classes,
            backbone=config.video_backbone,
            pretrained=config.video_pretrained,
            dropout=config.video_dropout,
        )
        self.sync_branch = SyncModule(
            vit_path=config.sync_vit_path,
            audio_dim=config.sync_audio_dim,
            transformer_heads=config.sync_transformer_heads,
            dropout=config.dropout,
            temporal_layers=config.sync_temporal_layers,
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(self.audio_branch.hidden, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
        )
        self.sync_proj = nn.Sequential(
            nn.Linear(self.sync_branch.hidden_dim * 3, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
        )
        self.video_proj = nn.Sequential(
            nn.Linear(self.video_branch.feature_dim, config.fusion_dim),
            nn.LayerNorm(config.fusion_dim),
        )
        self.cross_attn_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(
                            embed_dim=config.fusion_dim,
                            num_heads=config.cross_heads,
                            dropout=config.dropout,
                            batch_first=True,
                        ),
                        "norm": nn.LayerNorm(config.fusion_dim),
                    }
                )
                for _ in range(max(int(config.cross_attn_layers), 0))
            ]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.fusion_dim,
            nhead=config.cross_heads,
            dim_feedforward=config.fusion_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.cross_layers)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(config.fusion_dim),
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.GELU(),
            nn.Linear(config.fusion_dim // 2, 1),
        )
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(config.fusion_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim, config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim, config.num_classes),
        )

    def forward(
        self,
        *,
        waveform: torch.Tensor,
        waveform_lengths: torch.Tensor | None,
        mel_sync: torch.Tensor,
        video: torch.Tensor,
        video_sync: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        waveform = waveform.float()
        if waveform_lengths is not None:
            waveform_lengths = waveform_lengths.to(waveform.device)
        mel_sync = mel_sync.float()
        video = video.float()
        video_sync = video_sync.float()
        audio_logits, _, audio_feat = self.audio_branch(None, waveform=waveform, waveform_lengths=waveform_lengths)
        video_logits, video_rppg, video_feat = self.video_branch(video)
        sync_feat, sync_logits = self.sync_branch(video_sync, mel_sync)
        tokens = torch.stack(
            [
                self.audio_proj(audio_feat),
                self.sync_proj(sync_feat),
                self.video_proj(video_feat),
            ],
            dim=1,
        )
        for block in self.cross_attn_layers:
            residual = tokens
            attn_out, _ = block["attn"](tokens, tokens, tokens)
            tokens = block["norm"](residual + attn_out)
        encoded = self.fusion_encoder(tokens)
        gate_logits = self.gate_mlp(encoded).squeeze(-1)
        gates = torch.softmax(gate_logits, dim=-1)
        fused = (encoded * gates.unsqueeze(-1)).sum(dim=1)
        final_logits = self.fusion_head(fused)
        return {
            "logits": final_logits,
            "audio_logits": audio_logits,
            "sync_logits": sync_logits,
            "video_logits": video_logits,
            "audio_embedding": audio_feat,
            "sync_embedding": sync_feat,
            "video_embedding": video_feat,
            "rppg": video_rppg,
            "gates": gates,
        }

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        groups: Dict[str, List[nn.Parameter]] = {}
        audio_groups = self.audio_branch.parameter_groups()
        groups["audio_backbone"] = audio_groups.get("backbone", [])
        groups["audio_head"] = audio_groups.get("head", [])
        groups["video_backbone"] = list(self.video_branch.feature_extractor.parameters())
        groups["video_heads"] = list(self.video_branch.classifier_head.parameters()) + list(self.video_branch.rppg_head.parameters())
        groups["sync_branch"] = [p for p in self.sync_branch.parameters()]
        fusion_params: List[nn.Parameter] = []
        fusion_params += list(self.audio_proj.parameters())
        fusion_params += list(self.sync_proj.parameters())
        fusion_params += list(self.video_proj.parameters())
        fusion_params += list(self.fusion_encoder.parameters())
        fusion_params += list(self.gate_mlp.parameters())
        fusion_params += list(self.fusion_head.parameters())
        groups["fusion_head"] = fusion_params
        for name, params in list(groups.items()):
            groups[name] = [p for p in params if p.requires_grad]
        return groups

    def freeze_video_backbone(self, freeze: bool = True) -> None:
        for param in self.video_branch.feature_extractor.parameters():
            param.requires_grad = not freeze

    def freeze_sync_backbone(self, freeze: bool = True) -> None:
        for param in self.sync_branch.parameters():
            param.requires_grad = not freeze

    def load_branch_checkpoint(self, branch: str, checkpoint: str | Path, strict: bool = False) -> None:
        branch = branch.lower()
        path = Path(checkpoint)
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        module: nn.Module
        if branch == "audio":
            module = self.audio_branch
        elif branch == "video":
            module = self.video_branch
        elif branch == "sync":
            module = self.sync_branch
        else:
            raise ValueError(f"Unknown branch '{branch}'.")
        load_state_partial(module, state, branch=branch, strict=strict)
