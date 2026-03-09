from __future__ import annotations

"""
Utilities for end-to-end tri-modal (audio, sync, video) fusion training.
"""

from .augmentations import MelAugmentation, SpecAugParams, VideoAugmentation
from .dataset import (
    TriModalDatasetConfig,
    FakeAVTriModalDataset,
    TriModalBatch,
    TriModalCollator,
)
from .ema import ModelEMA
from .model import FusionConfig, TriModalFusionModel

__all__ = [
    "MelAugmentation",
    "SpecAugParams",
    "VideoAugmentation",
    "TriModalDatasetConfig",
    "FakeAVTriModalDataset",
    "TriModalBatch",
    "TriModalCollator",
    "ModelEMA",
    "FusionConfig",
    "TriModalFusionModel",
]
