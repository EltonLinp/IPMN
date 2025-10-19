"""Training utilities for TrustFusion."""

from typing import Any

__all__ = ["train_video_detector", "train_audio_detector"]


def train_video_detector(*args: Any, **kwargs: Any) -> Any:
    """
    Lazily import and invoke the video training entry point.

    This avoids importing :mod:`trustfusion.training.video_trainer`
    during package initialisation, preventing runpy warnings when the
    module is executed via ``python -m trustfusion.training.video_trainer``.
    """

    from .video_trainer import main as _main

    return _main(*args, **kwargs)


def train_audio_detector(*args: Any, **kwargs: Any) -> Any:
    """
    Lazily import and invoke the audio training entry point.
    """

    from .audio_trainer import main as _main

    return _main(*args, **kwargs)
