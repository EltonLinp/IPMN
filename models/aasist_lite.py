from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Lightweight 2D convolutional block used for Mel feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = tuple(k // 2 for k in kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.conv(x))))


class GraphSAGELayer(nn.Module):
    """GraphSAGE layer operating on grid-structured Mel graphs."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.norm = nn.LayerNorm(out_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _expand_edge_index(edge_index: torch.Tensor, batch_size: int, num_nodes: int, device: torch.device) -> torch.Tensor:
        """Repeat edge index for batched graphs with disjoint node sets."""
        if edge_index.numel() == 0:
            return edge_index
        src, dst = edge_index
        offsets = torch.arange(batch_size, device=device, dtype=src.dtype).unsqueeze(1) * num_nodes
        src_expanded = (src.unsqueeze(0) + offsets).reshape(-1)
        dst_expanded = (dst.unsqueeze(0) + offsets).reshape(-1)
        return torch.stack([src_expanded, dst_expanded], dim=0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        num_nodes = x.size(1)
        device = x.device
        edge_index_batch = self._expand_edge_index(edge_index, batch_size, num_nodes, device)

        x_flat = x.reshape(batch_size * num_nodes, -1)
        if edge_index_batch.numel() == 0:
            agg = x_flat
            deg = torch.ones((batch_size * num_nodes, 1), device=device, dtype=x_flat.dtype)
        else:
            src, dst = edge_index_batch
            agg = torch.zeros_like(x_flat)
            agg.index_add_(0, dst, x_flat[src])
            deg = torch.zeros((batch_size * num_nodes, 1), device=device, dtype=x_flat.dtype)
            deg.index_add_(0, dst, torch.ones((dst.size(0), 1), device=device, dtype=x_flat.dtype))
        neigh = agg / deg.clamp_min(1.0)

        out = self.lin_self(x_flat) + self.lin_neigh(neigh) + self.bias
        out = self.dropout(self.activation(self.norm(out)))
        return out.view(batch_size, num_nodes, -1)


class AttentiveStatsPooling(nn.Module):
    """Attentive statistics pooling used to summarise node embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        attn = self.attention(x)
        weights = torch.softmax(attn, dim=1)
        mean = torch.sum(weights * x, dim=1)
        diff = x - mean.unsqueeze(1)
        var = torch.sum(weights * diff * diff, dim=1)
        std = torch.sqrt(var.clamp_min(1e-6))
        return torch.cat([mean, std], dim=1)


class WaveformEncoder(nn.Module):
    """Lightweight 1D convolutional encoder for raw waveform snippets."""

    def __init__(
        self,
        channels: Tuple[int, int, int] = (32, 64, 96),
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        self.strides: list[int] = []
        in_ch = 1
        for idx, out_ch in enumerate(channels):
            stride = 2 if idx < 2 else 1
            dilation = 2 ** idx
            padding = ((kernel_size - 1) // 2) * dilation
            layers.extend(
                [
                    nn.Conv1d(
                        in_ch,
                        out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.PReLU(out_ch),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = out_ch
            self.strides.append(stride)
        self.net = nn.Sequential(*layers)
        self.output_dim = channels[-1]
        self.proj = nn.Sequential(
            nn.Conv1d(self.output_dim, self.output_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.output_dim),
            nn.PReLU(self.output_dim),
        )

    def forward(self, waveform: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if waveform.dim() == 2:
            x = waveform.unsqueeze(1)
        elif waveform.dim() == 3:
            x = waveform
        else:
            raise ValueError(f"Unexpected waveform shape: {tuple(waveform.shape)}")
        x = self.net(x)
        x = self.proj(x)
        if lengths is not None:
            orig_lengths = lengths.to(x.device, dtype=torch.float32)
            eff_lengths = orig_lengths.clone()
            for stride in self.strides:
                eff_lengths = torch.floor((eff_lengths + stride - 1) / stride)
            eff_lengths = eff_lengths.clamp_min(1.0)
            max_len = x.size(-1)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < eff_lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).float()
            summed = (x * mask).sum(dim=-1)
            denom = mask.sum(dim=-1).clamp_min(1.0)
            pooled = summed / denom
            pooled[orig_lengths <= 0] = 0.0
            return pooled
        return x.mean(dim=-1)


def _build_grid_edge_index(freq_bins: int, time_steps: int) -> torch.Tensor:
    edges: List[Tuple[int, int]] = []
    for f in range(freq_bins):
        for t in range(time_steps):
            idx = f * time_steps + t
            neighbours: List[Tuple[int, int]] = []
            # Time neighbours
            if t - 1 >= 0:
                neighbours.append((f, t - 1))
            if t + 1 < time_steps:
                neighbours.append((f, t + 1))
            # Frequency neighbours
            if f - 1 >= 0:
                neighbours.append((f - 1, t))
            if f + 1 < freq_bins:
                neighbours.append((f + 1, t))
            # Diagonal/block neighbours
            for df in (-1, 1):
                nf = f + df
                if 0 <= nf < freq_bins:
                    if t - 1 >= 0:
                        neighbours.append((nf, t - 1))
                    if t + 1 < time_steps:
                        neighbours.append((nf, t + 1))
            for nf, nt in neighbours:
                edges.append((idx, nf * time_steps + nt))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    src, dst = zip(*edges)
    return torch.tensor([src, dst], dtype=torch.long)


class AASISTLite(nn.Module):
    """
    Lightweight AASIST-inspired audio encoder operating on Mel spectrogram graphs.

    Pipeline:
        Mel -> CNN feature extractor -> GraphSAGE stack -> Attentive stats pooling ->
        segment logits aggregation (mean + top-k) -> MLP head.
    """

    def __init__(
        self,
        *,
        num_classes: int = 2,
        mel_bins: int = 80,
        cnn_channels: Tuple[int, int, int] = (48, 96, 160),
        graph_dims: Tuple[int, int, int] = (160, 192, 224),
        graph_dropout: float = 0.1,
        head_dim: int = 256,
        top_k: int = 5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.mel_bins = mel_bins
        self.top_k = max(int(top_k), 1)

        c1, c2, c3 = cnn_channels
        self.cnn_blocks = nn.ModuleList(
            [
                ConvBlock(1, c1, dropout=0.05),
                ConvBlock(c1, c2, stride=(2, 2), dropout=0.1),
                ConvBlock(c2, c3, stride=(2, 2), dropout=0.1),
            ]
        )

        self.pre_graph = nn.Conv2d(c3, graph_dims[0], kernel_size=1, bias=False)
        self.graph_layers = nn.ModuleList()
        in_dim = graph_dims[0]
        for out_dim in graph_dims:
            self.graph_layers.append(GraphSAGELayer(in_dim, out_dim, dropout=graph_dropout))
            in_dim = out_dim

        self.node_norm = nn.LayerNorm(graph_dims[-1])
        self.segment_head = nn.Sequential(
            nn.Linear(graph_dims[-1], graph_dims[-1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(graph_dims[-1], num_classes),
        )
        self.asp = AttentiveStatsPooling(graph_dims[-1], graph_dims[-1])

        self.wave_branch = WaveformEncoder()
        self.wave_proj_dim = 128
        self.wave_proj = nn.Sequential(
            nn.LayerNorm(self.wave_branch.output_dim),
            nn.Linear(self.wave_branch.output_dim, self.wave_proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.wave_gate = nn.Sequential(
            nn.Linear(self.wave_proj_dim + num_classes, self.wave_proj_dim),
            nn.Sigmoid(),
        )

        clip_feat_dim = graph_dims[-1] * 2 + num_classes * 2 + self.wave_proj_dim
        self.embed_proj = nn.Sequential(
            nn.LayerNorm(clip_feat_dim),
            nn.Linear(clip_feat_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(head_dim, num_classes)
        self.feature_dim = head_dim

        self._stage_modules: List[nn.Module] = list(self.cnn_blocks) + [self.pre_graph] + list(self.graph_layers)
        self._edge_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def freeze_stages(self, num_frozen: int) -> None:
        """Freeze the first `num_frozen` backbone stages."""
        for idx, module in enumerate(self._stage_modules):
            requires_grad = idx >= num_frozen
            for param in module.parameters():
                param.requires_grad = requires_grad

    def parameter_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Return parameter groups for optimiser construction."""
        front_params: List[nn.Parameter] = []
        backbone_params: List[nn.Parameter] = []
        head_params: List[nn.Parameter] = []
        seen: set[int] = set()

        for idx, module in enumerate(self._stage_modules):
            target = front_params if idx < 2 else backbone_params
            for param in module.parameters():
                pid = id(param)
                if pid in seen:
                    continue
                seen.add(pid)
                target.append(param)

        for module in [
            self.segment_head,
            self.asp,
            self.embed_proj,
            self.classifier,
            self.node_norm,
            self.wave_branch,
            self.wave_proj,
            self.wave_gate,
        ]:
            for param in module.parameters():
                pid = id(param)
                if pid in seen:
                    continue
                seen.add(pid)
                head_params.append(param)
        return {"front": front_params, "backbone": backbone_params, "head": head_params}

    def _get_edge_index(self, freq: int, time: int, device: torch.device) -> torch.Tensor:
        key = (freq, time)
        if key not in self._edge_cache:
            self._edge_cache[key] = _build_grid_edge_index(freq, time)
        return self._edge_cache[key].to(device)

    def forward(
        self,
        mel: torch.Tensor,
        *,
        waveform: torch.Tensor | None = None,
        waveform_lengths: torch.Tensor | None = None,
        return_speaker: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del return_speaker
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        if mel.dim() != 4:
            raise ValueError(f"Expected mel tensor with 3 or 4 dims, got {mel.shape}")
        batch_size = mel.size(0)

        if mel.size(1) != 1:
            mel = mel.unsqueeze(1) if mel.size(1) != 1 else mel
        if mel.size(2) != self.mel_bins:
            mel = F.interpolate(mel, size=(self.mel_bins, mel.size(-1)), mode="bilinear", align_corners=False)

        x = mel
        for block in self.cnn_blocks:
            x = block(x)
        x = self.pre_graph(x)

        batch_size, channels, freq, time = x.shape
        edge_index = self._get_edge_index(freq, time, x.device)

        nodes = freq * time
        x = x.view(batch_size, channels, nodes).transpose(1, 2).contiguous()
        for layer in self.graph_layers:
            x = layer(x, edge_index, batch_size=batch_size)
        x = self.node_norm(x)

        segment_logits = self.segment_head(x)

        asp_feat = self.asp(x)
        segment_mean = segment_logits.mean(dim=1)
        k = min(self.top_k, segment_logits.size(1))
        if k > 0:
            topk_vals = segment_logits.topk(k, dim=1).values.mean(dim=1)
        else:
            topk_vals = segment_mean

        wave_embed: torch.Tensor
        if waveform is not None and waveform.numel() > 0:
            wave_in = waveform
            if wave_in.dim() == 3 and wave_in.size(1) > 1:
                wave_in = wave_in.mean(dim=1)
            if wave_in.dim() == 3 and wave_in.size(1) == 1:
                wave_in = wave_in.squeeze(1)
            wave_in = wave_in.to(asp_feat.dtype)
            wave_lengths = waveform_lengths.float() if waveform_lengths is not None else None
            base_embed = self.wave_branch(wave_in, wave_lengths)
            wave_embed = self.wave_proj(base_embed)
            gate_input = torch.cat([wave_embed, segment_mean], dim=1)
            wave_embed = wave_embed * self.wave_gate(gate_input)
        else:
            wave_embed = asp_feat.new_zeros((batch_size, self.wave_proj_dim))

        clip_features = torch.cat([asp_feat, segment_mean, topk_vals, wave_embed], dim=1)
        clip_embed = self.embed_proj(clip_features)
        logits = self.classifier(clip_embed)
        return logits, segment_logits, clip_embed


# Backwards-compatible alias for existing imports.
AASISTClassifier = AASISTLite
