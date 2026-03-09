from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


@dataclass
class WavLMConfig:
    model_name: str = "microsoft/wavlm-base-plus-sv"
    num_classes: int = 2
    dropout: float = 0.2
    train_backbone: bool = False
    unfreeze_layers: int = 0
    local_files_only: bool = False


class WavLMClassifier(nn.Module):
    """
    Lightweight wrapper around a pretrained WavLM encoder with a shallow
    classification head. Accepts raw 16 kHz waveform input.
    """

    def __init__(self, config: WavLMConfig) -> None:
        super().__init__()
        self.cfg = config
        hf_config = AutoConfig.from_pretrained(
            config.model_name,
            num_labels=config.num_classes,
            local_files_only=config.local_files_only,
            trust_remote_code=False,
        )
        self.backbone = AutoModel.from_pretrained(
            config.model_name,
            config=hf_config,
            local_files_only=config.local_files_only,
            trust_remote_code=False,
        )
        hidden = self.backbone.config.hidden_size
        self.hidden = hidden
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.num_classes),
        )
        encoder = getattr(self.backbone, "encoder", None)
        if encoder is not None and hasattr(encoder, "layers"):
            self._encoder_layers = list(encoder.layers)
        else:
            self._encoder_layers = []
        self.set_backbone_trainable(config.train_backbone, config.unfreeze_layers)

    def forward(
        self,
        _: torch.Tensor | None = None,
        *,
        waveform: torch.Tensor | None,
        waveform_lengths: torch.Tensor | None = None,
        **__: object,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if waveform is None or waveform.numel() == 0:
            raise ValueError("WavLMClassifier requires waveform input.")
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)
        elif waveform.dim() != 2:
            raise ValueError(f"Unexpected waveform shape {waveform.shape}")
        attention_mask = None
        if waveform_lengths is not None:
            attention_mask = (
                torch.arange(waveform.size(1), device=waveform.device)
                .unsqueeze(0)
                .expand(waveform.size(0), -1)
                < waveform_lengths.unsqueeze(1)
            ).to(dtype=torch.long)
        outputs = self.backbone(
            waveform,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        logits = self.head(pooled)
        dummy_seg = logits.new_zeros((logits.size(0), 1, 1))
        return logits, dummy_seg, pooled

    def set_backbone_trainable(self, enabled: bool, layers: int | None = None) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False
        if not enabled:
            return
        target_layers = layers if layers is not None else self.cfg.unfreeze_layers
        self._unfreeze_layers(target_layers)

    def _unfreeze_layers(self, layers: int) -> None:
        if layers <= 0 or not self._encoder_layers:
            return
        count = min(int(layers), len(self._encoder_layers))
        for block in self._encoder_layers[-count:]:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "layer_norm"):
            for param in self.backbone.layer_norm.parameters():
                param.requires_grad = True
        adapter = getattr(self.backbone, "adapter", None)
        if adapter is not None:
            for param in adapter.parameters():
                param.requires_grad = True

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        return {"backbone": backbone_params, "head": head_params}
