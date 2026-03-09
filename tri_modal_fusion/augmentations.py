from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass
class SpecAugParams:
    freq_mask: int = 0
    time_mask: int = 0
    prob: float = 0.0
    noise_std: float = 0.0
    gain_std: float = 0.0
    shift_pct: float = 0.0
    pseudo_fake_prob: float = 0.0
    pseudo_fake_freq: int = 0
    pseudo_fake_time: int = 0


class MelAugmentation:
    """
    Simple SpecAug + gain/noise pipeline operating on [mel_bins, steps] tensors.
    """

    def __init__(self, params: SpecAugParams) -> None:
        self.params = params

    def _mask(self, mel: torch.Tensor, axis: int, width: int) -> None:
        if width <= 0:
            return
        size = mel.size(axis)
        if width >= size:
            mel.zero_()
            return
        start = random.randint(0, size - width)
        slices = [slice(None)] * mel.dim()
        slices[axis] = slice(start, start + width)
        mel[tuple(slices)] = 0.0

    def _apply_spec_aug(self, mel: torch.Tensor, freq_mask: int, time_mask: int, prob: float) -> None:
        if prob <= 0.0:
            return
        if random.random() > prob:
            return
        if freq_mask > 0:
            width = random.randint(1, freq_mask)
            self._mask(mel, axis=-2, width=width)
        if time_mask > 0:
            width = random.randint(1, time_mask)
            self._mask(mel, axis=-1, width=width)

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() == 3:
            mel = mel.squeeze(0)
        augmented = mel.clone()
        steps = augmented.size(-1)
        if self.params.shift_pct > 0.0 and steps > 1:
            max_shift = max(int(steps * self.params.shift_pct), 1)
            shift = random.randint(-max_shift, max_shift)
            if shift != 0:
                augmented = torch.roll(augmented, shifts=shift, dims=-1)
        if self.params.gain_std > 0.0:
            scale = 1.0 + random.gauss(0.0, self.params.gain_std)
            augmented = augmented * scale
        if self.params.noise_std > 0.0:
            noise = torch.randn_like(augmented) * self.params.noise_std
            augmented = augmented + noise
        self._apply_spec_aug(
            augmented,
            freq_mask=self.params.freq_mask,
            time_mask=self.params.time_mask,
            prob=self.params.prob,
        )
        if self.params.pseudo_fake_prob > 0.0 and random.random() < self.params.pseudo_fake_prob:
            self._apply_spec_aug(
                augmented,
                freq_mask=max(self.params.pseudo_fake_freq, self.params.freq_mask),
                time_mask=max(self.params.pseudo_fake_time, self.params.time_mask),
                prob=1.0,
            )
        return augmented.unsqueeze(0)


class VideoAugmentation:
    """
    Operates on [C, T, H, W] video tensors in [-1, 1].
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
        augmented = video
        if self.horizontal_flip_prob > 0.0 and random.random() < self.horizontal_flip_prob:
            augmented = torch.flip(augmented, dims=[-1])
        if self.temporal_jitter > 0 and augmented.size(1) > 1:
            shift = random.randint(-self.temporal_jitter, self.temporal_jitter)
            if shift != 0:
                augmented = torch.roll(augmented, shifts=shift, dims=1)
        if self.brightness > 0.0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            augmented = (augmented * factor).clamp(-1.0, 1.0)
        if self.contrast > 0.0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = augmented.mean(dim=(-1, -2), keepdim=True)
            augmented = ((augmented - mean) * factor + mean).clamp(-1.0, 1.0)
        if self.noise_std > 0.0:
            noise = torch.randn_like(augmented) * self.noise_std
            augmented = (augmented + noise).clamp(-1.0, 1.0)
        return augmented.contiguous()
