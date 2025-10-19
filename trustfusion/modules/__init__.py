"""
Model component registry.
"""

from .video_detector import VideoDeepfakeDetector
from .audio_detector import AudioDeepfakeDetector
from .av_consistency import AVConsistencyModule
from .cross_modal_attention import CrossModalAttention
from .biosync import BioSyncAnalyzer
from .diffusion_explainer import DiffusionResidualExplainer
from .sync_transformer import VisionAudioSyncTransformer
from .fusion_model import LinearFusionModel

__all__ = [
    "VideoDeepfakeDetector",
    "AudioDeepfakeDetector",
    "AVConsistencyModule",
    "CrossModalAttention",
    "BioSyncAnalyzer",
    "DiffusionResidualExplainer",
    "VisionAudioSyncTransformer",
    "LinearFusionModel",
]
