"""
High-level orchestration for the TrustFusion system.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ..config import FusionConfig
from ..data import DeepfakeVideoDataset
from ..modules import (
    AudioDeepfakeDetector,
    AVConsistencyModule,
    BioSyncAnalyzer,
    CrossModalAttention,
    DiffusionResidualExplainer,
    VideoDeepfakeDetector,
)
from ..utils.logger import get_logger
from .fusion import FusionEngine
from .report import TrustReportGenerator
from .risk_analysis import RiskAnalyser, RiskAssessment

LOGGER = get_logger(__name__)


@dataclass
class TrustFusionResult:
    session_id: str
    fusion_output: Dict[str, torch.Tensor]
    risk_assessment: RiskAssessment
    report_paths: Dict[str, Optional[str]]


class TrustFusionSystem:
    """
    Coordinates detectors, fusion logic, risk analysis, and report generation.
    """

    def __init__(self, config: FusionConfig) -> None:
        self.config = config
        self.dataset = DeepfakeVideoDataset(
            root_dir=config.dataset.test_root,
            video_extensions=config.dataset.video_extensions,
            real_dir_name=config.dataset.test_real_dir,
            fake_dir_name=config.dataset.test_fake_dir,
        )
        self.video_detector = VideoDeepfakeDetector()
        self.audio_detector = AudioDeepfakeDetector()
        self.av_module = AVConsistencyModule(feature_dim=64)
        self.attention = CrossModalAttention(d_video=1, d_audio=64, d_out=64)
        self.biosync = BioSyncAnalyzer(
            min_bpm=config.biosync.min_bpm, max_bpm=config.biosync.max_bpm
        )
        self.diffusion = DiffusionResidualExplainer(
            guidance_scale=config.diffusion.guidance_scale,
            residual_threshold=config.diffusion.residual_threshold,
        )
        self.fusion_engine = FusionEngine(
            video_detector=self.video_detector,
            audio_detector=self.audio_detector,
            av_module=self.av_module,
            attention=self.attention,
            biosync=self.biosync,
            diffusion=self.diffusion,
            trust_thresholds=config.trust_thresholds,
        )
        self.risk_analyser = RiskAnalyser(config.trust_thresholds)
        self.report_generator = TrustReportGenerator(config.report)

    def dataset_summary(self) -> Dict[str, object]:
        """
        Return dataset statistics for the configured data root.
        """
        return self.dataset.summary()

    def run(
        self,
        video_tensor: torch.Tensor,
        audio_tensor: torch.Tensor,
        lip_feats: torch.Tensor,
        audio_feats: torch.Tensor,
        rppg_trace: torch.Tensor,
        audio_energy: torch.Tensor,
        sample_rate: int,
        session_id: Optional[str] = None,
        generate_report: bool = True,
    ) -> TrustFusionResult:
        session_id = session_id or str(uuid.uuid4())
        LOGGER.info("Running TrustFusion for session %s", session_id)
        fusion_output = self.fusion_engine.forward(
            video_tensor=video_tensor,
            audio_tensor=audio_tensor,
            lip_feats=lip_feats,
            audio_feats=audio_feats,
            rppg_input=rppg_trace,
            audio_energy=audio_energy,
            sample_rate=sample_rate,
        )
        risk_assessment = self.risk_analyser.analyse(
            trust_score=fusion_output.trust_score,
            modality_scores=fusion_output.modality_scores,
        )
        report_paths: Dict[str, Optional[str]] = {}
        if generate_report:
            paths = self.report_generator.generate(risk_assessment, session_id)
            report_paths = {key: str(path) for key, path in paths.items()}
        return TrustFusionResult(
            session_id=session_id,
            fusion_output={
                "trust_score": fusion_output.trust_score,
                "modality_scores": fusion_output.modality_scores,
                "biosync_score": fusion_output.biosync_score,
                "sync_score": fusion_output.sync_score,
                "diffusion_confidence": fusion_output.diffusion_confidence,
            },
            risk_assessment=risk_assessment,
            report_paths=report_paths,
        )


def build_default_system() -> TrustFusionSystem:
    """
    Factory helper to get a ready-to-run system with default configuration.
    """
    config = FusionConfig()
    return TrustFusionSystem(config)
