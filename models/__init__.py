"""
Model definitions for multimodal deepfake detection.
"""

from .video_backbone import VideoClassifier
from .audio_backbone import AudioClassifier
from .sync_module import SyncModule
from .multimodal_fusion import MultimodalFusionModel

__all__ = ["VideoClassifier", "AudioClassifier", "SyncModule", "MultimodalFusionModel"]
