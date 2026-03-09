"""
Model definitions for multimodal deepfake detection.
"""

from .video_backbone import VideoClassifier
from .sync_module import SyncModule
from .multimodal_fusion import MultimodalFusionModel
from .aasist_lite import AASISTLite, AASISTClassifier
from .wavlm_classifier import WavLMClassifier, WavLMConfig

__all__ = [
    "VideoClassifier",
    "SyncModule",
    "MultimodalFusionModel",
    "AASISTLite",
    "AASISTClassifier",
    "WavLMClassifier",
    "WavLMConfig",
]
