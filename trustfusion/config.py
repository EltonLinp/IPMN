
"""
Configuration dataclasses for the TrustFusion prototype.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union


@dataclass
class DatasetConfig:
    """Paths and metadata for training and evaluation datasets."""

    celebdf_root: Path = Path(r"E:\CUHK\Industrial_Project\Celeb-DF-v2")
    video_extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")

    train_manifest: Optional[Path] = Path("manifests/train.csv")
    val_manifest: Optional[Path] = Path("manifests/val.csv")

    test_root: Optional[Path] = None
    test_fake_dir: Optional[Union[str, Sequence[str]]] = "Celeb-synthesis"
    test_real_dir: Optional[Union[str, Sequence[str]]] = ("Celeb-real", "YouTube-real")
    vit_model_dir: Path = Path(r"E:\CUHK\Industrial_Project\vit_model")

    def __post_init__(self) -> None:
        if self.test_root is None:
            self.test_root = self.celebdf_root


@dataclass
class PreprocessConfig:
    """Controls on-the-fly preprocessing / augmentation behaviour."""

    variant: str = "v1"
    video_encoder: str = "hist"  # options: "hist", "vit"
    video_seed: Optional[int] = None
    audio_seed: Optional[int] = None

    enable_video_aug: bool = False
    enable_audio_aug: bool = False
    eval_use_aug: bool = False

    video_flip_prob: float = 0.5
    video_brightness: float = 0.1
    video_contrast: float = 0.1
    video_crop_min_scale: float = 0.85
    video_crop_max_scale: float = 1.0
    video_noise_prob: float = 0.15
    video_noise_std: float = 8.0

    audio_noise_prob: float = 0.3
    audio_noise_std: float = 0.01
    audio_gain_min: float = 0.9
    audio_gain_max: float = 1.1
    audio_time_shift_prob: float = 0.3
    audio_time_shift_max: int = 1600


@dataclass
class DetectorConfig:
    video_model_checkpoint: Optional[Path] = None
    audio_model_checkpoint: Optional[Path] = None
    sync_model_checkpoint: Optional[Path] = None
    device: str = "cuda"
    frame_rate: int = 25
    sample_rate: int = 16000


@dataclass
class BioSyncConfig:
    rppg_window_size: int = 300
    audio_window_size: int = 300
    min_bpm: float = 40.0
    max_bpm: float = 180.0


@dataclass
class DiffusionExplainConfig:
    diffusion_model_path: Optional[Path] = None
    guidance_scale: float = 1.5
    residual_threshold: float = 0.2


@dataclass
class ReportConfig:
    output_dir: Path = Path("./outputs")
    include_pdf: bool = True
    include_json: bool = True
    issuer: str = "TrustFusion Prototype"
    organisation: str = "HSBC Innovation"


@dataclass
class FusionConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    biosync: BioSyncConfig = field(default_factory=BioSyncConfig)
    diffusion: DiffusionExplainConfig = field(default_factory=DiffusionExplainConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    trust_thresholds: Tuple[float, float] = (0.4, 0.7)


