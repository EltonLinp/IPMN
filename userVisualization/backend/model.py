
from __future__ import annotations

import logging
import pathlib
import threading
import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Mapping

import torch

LOGGER = logging.getLogger(__name__)


def _ensure_posix_path_support() -> None:
    """
    Torch checkpoints may contain pathlib.PosixPath instances that raise
    ``NotImplementedError`` on Windows. Replace the class with a PurePosixPath
    shim so torch.load can succeed.
    """

    try:
        pathlib.PosixPath("dummy")
    except NotImplementedError:
        class _CompatPosixPath(pathlib.PurePosixPath):  # type: ignore[misc]
            pass

        pathlib.PosixPath = _CompatPosixPath  # type: ignore[attr-defined]
        try:
            from torch.serialization import add_safe_globals  # type: ignore
        except ImportError:
            add_safe_globals = None
        if add_safe_globals is not None:
            add_safe_globals([_CompatPosixPath])


_ensure_posix_path_support()

from models import WavLMConfig
from tri_modal_fusion.model import FusionConfig, TriModalFusionModel

from .deepfake_scoring import compute_deepfake_score
from .preprocess_api import TriModalWebPreprocessor, build_default_preprocessor
from .runtime_config import get_setting


@dataclass
class BranchPrediction:
    label: str
    real_prob: float
    fake_prob: float
    confidence: float

    def as_dict(self) -> Dict[str, float | str]:
        return {
            "label": self.label,
            "real": self.real_prob,
            "fake": self.fake_prob,
            "confidence": self.confidence,
        }


class TriModalDetectionService:
    """
    Loads the tri-modal fusion model once and offers a simple `analyze` method for uploads.
    """

    def __init__(
        self,
        *,
        checkpoint_path: Path | None = None,
        device: str | None = None,
        preprocessor: TriModalWebPreprocessor | None = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        self.checkpoint = checkpoint_path or repo_root / "res" / "best_tri_modal.pt"
        self.vit_path = repo_root / "vit_model"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.preprocessor = preprocessor or build_default_preprocessor()
        self._invert_sync = _truthy_env("SYNC_LABEL_INVERT", False)
        self._video_temp = _float_env("VIDEO_LOGIT_TEMP", 1.0)
        self._video_bias = _float_env("VIDEO_LOGIT_BIAS", 0.0)
        self._audio_temp = _float_env("AUDIO_LOGIT_TEMP", 1.0)
        self._audio_bias = _float_env("AUDIO_LOGIT_BIAS", 0.0)
        self._sync_temp = _float_env("SYNC_LOGIT_TEMP", 1.0)
        self._sync_bias = _float_env("SYNC_LOGIT_BIAS", 0.0)
        self._sync_uncertainty_alpha = _sync_uncertainty_alpha_cfg(0.3)
        self._sync_mismatch_penalty = _sync_mismatch_penalty_cfg(1.0)
        legacy_use_gated = _truthy_env("USE_GATED_FINAL", False)
        strategy_raw = os.environ.get("FINAL_SCORE_STRATEGY")
        if strategy_raw is None:
            strategy_raw = get_setting("FINAL_SCORE_STRATEGY", None)
        strategy = str(strategy_raw).strip().lower() if strategy_raw is not None else ""
        if strategy not in {"raw", "gated", "calibrated"}:
            strategy = "gated" if legacy_use_gated else "calibrated"
        self._final_score_strategy = strategy
        self._final_fake_threshold = float(max(0.0, min(1.0, _float_env("FINAL_FAKE_THRESHOLD", 0.37))))
        self._final_score_raw_weight = float(max(0.0, _float_env("FINAL_SCORE_RAW_WEIGHT", 0.6)))
        self._final_score_audio_weight = float(max(0.0, _float_env("FINAL_SCORE_AUDIO_WEIGHT", 0.4)))
        self._model = self._build_model()
        self._lock = threading.Lock()

    def _build_model(self) -> TriModalFusionModel:
        local_override = os.environ.get("WAVLM_MODEL_PATH")
        use_local = False
        wavlm_model_id = "microsoft/wavlm-base-plus-sv"
        if local_override:
            local_path = Path(local_override)
            if _is_valid_wavlm_dir(local_path):
                wavlm_model_id = str(local_path)
                use_local = True
            else:
                raise FileNotFoundError(
                    f"WAVLM_MODEL_PATH is invalid or missing required files: {local_path}"
                )
        if not use_local:
            cache_root = Path(
                os.environ.get("HUGGINGFACE_HUB_CACHE")
                or os.environ.get("HF_HOME", "")
                or (Path.home() / ".cache" / "huggingface" / "hub")
            )
            snapshots_dir = cache_root / "models--microsoft--wavlm-base-plus-sv" / "snapshots"
            if snapshots_dir.exists():
                candidates = sorted(
                    [p for p in snapshots_dir.iterdir() if p.is_dir()],
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                for candidate in candidates:
                    if _is_valid_wavlm_dir(candidate):
                        wavlm_model_id = str(candidate)
                        use_local = True
                        break
            if not use_local:
                LOGGER.warning("No valid local WavLM snapshot found; falling back to remote download.")
        wavlm_cfg = WavLMConfig(
            model_name=wavlm_model_id,
            dropout=0.2,
            num_classes=2,
            train_backbone=False,
            unfreeze_layers=8,
            local_files_only=use_local,
        )
        fusion_cfg = FusionConfig(
            num_classes=2,
            fusion_dim=512,
            cross_heads=4,
            cross_layers=2,
            cross_attn_layers=1,
            dropout=0.2,
            sync_vit_path=self.vit_path,
            sync_audio_dim=self.preprocessor.cfg.mel_bins,
            sync_transformer_heads=8,
            sync_temporal_layers=3,
            video_backbone="r3d18",
            video_pretrained=False,
            video_dropout=0.3,
            wavlm=wavlm_cfg,
        )
        model = TriModalFusionModel(fusion_cfg).to(self.device)
        self._load_checkpoint(model)
        model.eval()
        return model

    def _load_checkpoint(self, model: TriModalFusionModel) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"Tri-modal checkpoint not found: {self.checkpoint}")
        LOGGER.info("Loading tri-modal weights from %s", self.checkpoint)
        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            for key in ("model_state", "state_dict"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        compat = model.load_state_dict(state, strict=False)
        if compat.missing_keys:
            LOGGER.warning("Missing %d parameters when loading checkpoint.", len(compat.missing_keys))
        if compat.unexpected_keys:
            LOGGER.warning("Encountered %d unexpected parameters when loading checkpoint.", len(compat.unexpected_keys))

    def analyze(
        self,
        video_path: Path,
        debug: bool = False,
        *,
        ablate: Mapping[str, object] | None = None,
        ablate_audio: bool = False,
        ablate_video: bool = False,
        ablate_sync: bool = False,
        sync_uncertainty_alpha: float | None = None,
        sync_mismatch_penalty: float | None = None,
    ) -> Dict[str, object]:
        ablation = self._resolve_ablation(
            ablate=ablate,
            ablate_audio=ablate_audio,
            ablate_video=ablate_video,
            ablate_sync=ablate_sync,
        )
        debug_enabled = bool(debug or _truthy_env("DEBUG_TRIMODAL", False))
        start = perf_counter()
        prepared = self.preprocessor(video_path, debug=debug_enabled)
        prepared = prepared.to_device(self.device)
        with self._lock, torch.no_grad():
            outputs = self._model(
                waveform=prepared.waveform,
                waveform_lengths=prepared.waveform_length,
                mel_sync=prepared.mel_sync,
                video=prepared.video,
                video_sync=prepared.video_sync,
            )
        duration = perf_counter() - start
        audio = self._format_branch(
            outputs["audio_logits"],
            temp=self._audio_temp,
            bias=self._audio_bias,
        ).as_dict()
        video = self._format_branch(
            outputs["video_logits"],
            temp=self._video_temp,
            bias=self._video_bias,
        ).as_dict()
        sync = self._format_branch(
            outputs["sync_logits"],
            invert=self._invert_sync,
            temp=self._sync_temp,
            bias=self._sync_bias,
        ).as_dict()
        sync_quality = self._normalize_sync_quality(prepared.sync_quality)
        sync_before_clamp = dict(sync)
        sync, sync_quality = self._apply_sync_uncertainty_clamp(
            sync=sync,
            sync_quality=sync_quality,
            sync_uncertainty_alpha=sync_uncertainty_alpha,
        )
        final_raw = self._format_branch(outputs["logits"]).as_dict()
        gated, fusion_meta = self._gated_fusion(
            audio=audio,
            video=video,
            sync=sync,
            ablation=ablation,
            sync_quality=sync_quality,
            sync_mismatch_penalty=sync_mismatch_penalty,
        )
        weighted_weights = self._apply_ablation_to_weights(self._deepfake_weights(), ablation)
        weighted_weights = self._apply_sync_quality_penalty_to_weights(
            weighted_weights,
            fusion_meta.get("sync_quality"),
        )
        weighted = compute_deepfake_score(
            {
                "audio": audio,
                "video": video,
                "sync": sync,
            },
            weights=weighted_weights,
        )
        final_served = self._compose_final_prediction(
            final_raw=final_raw,
            audio=audio,
            gated=gated,
        )
        formatted = {
            "final": final_served,
            "final_raw": final_raw,
            "final_gated": gated,
            "audio": audio,
            "video": video,
            "sync": sync,
            "sync_quality": fusion_meta.get("sync_quality"),
            "elapsed": duration,
        }
        if debug_enabled:
            formatted["debug"] = self._build_debug_payload(
                prepared_debug=prepared.debug,
                outputs=outputs,
                audio=audio,
                video=video,
                sync_before_clamp=sync_before_clamp,
                sync=sync,
                final_raw=final_raw,
                final_served=final_served,
                gated=gated,
                fusion_meta=fusion_meta,
                ablation=ablation,
                deepfake_weighted=weighted,
                deepfake_weights_effective=weighted_weights,
            )
        return formatted

    def _build_debug_payload(
        self,
        *,
        prepared_debug: dict[str, object] | None,
        outputs: dict[str, torch.Tensor],
        audio: Dict[str, object],
        video: Dict[str, object],
        sync_before_clamp: Dict[str, object],
        sync: Dict[str, object],
        final_raw: Dict[str, object],
        final_served: Dict[str, object],
        gated: Dict[str, object],
        fusion_meta: dict[str, object],
        ablation: dict[str, bool],
        deepfake_weighted: dict[str, object],
        deepfake_weights_effective: dict[str, float],
    ) -> dict[str, object]:
        logits_raw = {
            "audio_raw": self._binary_logits(outputs.get("audio_logits")),
            "video_raw": self._binary_logits(outputs.get("video_logits")),
            "sync_raw": self._binary_logits(outputs.get("sync_logits")),
        }
        calibration = {
            "audio": {
                "temp": float(self._audio_temp),
                "bias": float(self._audio_bias),
                "prob_real": _safe_float(audio.get("real")),
                "prob_fake": _safe_float(audio.get("fake")),
            },
            "video": {
                "temp": float(self._video_temp),
                "bias": float(self._video_bias),
                "prob_real": _safe_float(video.get("real")),
                "prob_fake": _safe_float(video.get("fake")),
            },
            "sync": {
                "temp": float(self._sync_temp),
                "bias": float(self._sync_bias),
                "invert": bool(self._invert_sync),
                "prob_real_before": _safe_float(sync_before_clamp.get("real")),
                "prob_fake_before": _safe_float(sync_before_clamp.get("fake")),
                "prob_real_after": _safe_float(sync.get("real")),
                "prob_fake_after": _safe_float(sync.get("fake")),
                "prob_real": _safe_float(sync.get("real")),
                "prob_fake": _safe_float(sync.get("fake")),
            },
        }
        model_gate = self._model_gate(outputs.get("gates"))
        preprocess_payload: dict[str, object] = {}
        if isinstance(prepared_debug, dict):
            preprocess_payload = dict(prepared_debug)

        return {
            "preprocess": preprocess_payload,
            "sync_quality": fusion_meta.get("sync_quality"),
            "raw_logits": logits_raw,
            "calibration": calibration,
            "gated_fusion": {
                "branch_confidence": {
                    "audio": _safe_float(audio.get("confidence")),
                    "video": _safe_float(video.get("confidence")),
                    "sync": _safe_float(sync.get("confidence")),
                },
                "branch_weights": gated.get("weights"),
                "branch_prob_fake": {
                    "audio": _safe_float(audio.get("fake")),
                    "video": _safe_float(video.get("fake")),
                    "sync": _safe_float(sync.get("fake")),
                },
                "branch_prob_real": {
                    "audio": _safe_float(audio.get("real")),
                    "video": _safe_float(video.get("real")),
                    "sync": _safe_float(sync.get("real")),
                },
                "raw_weights": fusion_meta.get("raw_weights"),
                "effective_weights_raw": fusion_meta.get("effective_weights_raw"),
                "effective_weights_norm": fusion_meta.get("effective_weights_norm"),
                "ignored_branches": fusion_meta.get("ignored_branches"),
                "missing_branches": fusion_meta.get("missing_branches"),
                "missing_reason": fusion_meta.get("missing_reason"),
                "sync_weight_before_penalty": fusion_meta.get("sync_weight_before_penalty"),
                "sync_weight_after_penalty": fusion_meta.get("sync_weight_after_penalty"),
                "model_gate": model_gate,
                "final_raw": final_raw,
                "final_served": final_served,
                "final_prob": gated,
            },
            "final_score": {
                "strategy": self._final_score_strategy,
                "threshold": float(self._final_fake_threshold),
                "raw_weight": float(self._final_score_raw_weight),
                "audio_weight": float(self._final_score_audio_weight),
                "prob_fake_raw": _safe_float(final_raw.get("fake")),
                "prob_fake_audio": _safe_float(audio.get("fake")),
                "prob_fake_served": _safe_float(final_served.get("fake")),
                "label_served": final_served.get("label"),
            },
            "ablation": {
                "requested": ablation,
                "enabled": any(ablation.values()),
                "ignored_branches": fusion_meta.get("ignored_branches"),
                "missing_branches": fusion_meta.get("missing_branches"),
                "effective_fusion_weights": fusion_meta.get("effective_weights_norm"),
                "effective_weighted_weights": deepfake_weights_effective,
            },
            "deepfake_weighted": deepfake_weighted,
        }

    @staticmethod
    def _binary_logits(logits: torch.Tensor | None) -> dict[str, object] | None:
        if logits is None or not isinstance(logits, torch.Tensor) or logits.numel() == 0:
            return None
        values = logits.detach().float().cpu()
        if values.dim() == 2:
            values = values[0]
        vector = [float(v) for v in values.tolist()]
        payload: dict[str, object] = {"vector": vector}
        if len(vector) >= 2:
            payload["real"] = vector[0]
            payload["fake"] = vector[1]
        return payload

    @staticmethod
    def _model_gate(gates: torch.Tensor | None) -> dict[str, object] | None:
        if gates is None or not isinstance(gates, torch.Tensor) or gates.numel() == 0:
            return None
        values = gates.detach().float().cpu()
        if values.dim() == 2:
            values = values[0]
        vector = [float(v) for v in values.tolist()]
        payload: dict[str, object] = {"vector": vector}
        if len(vector) >= 3:
            payload.update(
                {
                    "audio": vector[0],
                    "sync": vector[1],
                    "video": vector[2],
                }
            )
        return payload

    @staticmethod
    def _deepfake_weights() -> dict[str, float]:
        return {
            "video": _float_env("DEEPFAKE_WEIGHT_VIDEO", 0.7),
            "audio": _float_env("DEEPFAKE_WEIGHT_AUDIO", 0.15),
            "sync": _float_env("DEEPFAKE_WEIGHT_SYNC", 0.15),
        }

    def _format_branch(
        self,
        logits: torch.Tensor,
        *,
        invert: bool = False,
        temp: float = 1.0,
        bias: float = 0.0,
    ) -> BranchPrediction:
        adjusted = logits
        if temp and temp != 1.0:
            adjusted = adjusted / float(max(temp, 1e-3))
        if bias:
            adjusted = adjusted.clone()
            adjusted[:, 1] += float(bias)
        probs = torch.softmax(adjusted, dim=1)[0].detach().cpu()
        real = float(probs[0].item())
        fake = float(probs[1].item())
        if invert:
            real, fake = fake, real
        label = "Fake" if fake >= real else "Real"
        confidence = abs(fake - real)
        return BranchPrediction(label=label, real_prob=real, fake_prob=fake, confidence=confidence)

    def _gated_fusion(
        self,
        *,
        audio: Dict[str, object],
        video: Dict[str, object],
        sync: Dict[str, object],
        ablation: dict[str, bool] | None = None,
        sync_quality: Mapping[str, object] | None = None,
        sync_mismatch_penalty: float | None = None,
    ) -> tuple[Dict[str, object], dict[str, object]]:
        ablation = ablation or {"audio": False, "video": False, "sync": False}
        branches = {"audio": audio, "video": video, "sync": sync}
        raw_weights: dict[str, float] = {}
        effective_raw: dict[str, float] = {}
        effective_norm: dict[str, float] = {}
        missing_branches: list[str] = []
        ignored_branches: list[str] = []
        branch_values: dict[str, tuple[float, float]] = {}

        for name, branch in branches.items():
            real_prob = _safe_float(branch.get("real"))
            fake_prob = _safe_float(branch.get("fake"))
            is_missing = real_prob is None or fake_prob is None
            if is_missing:
                missing_branches.append(name)
                raw_weights[name] = 0.0
                effective_raw[name] = 0.0
                continue
            branch_values[name] = (float(real_prob), float(fake_prob))
            raw = self._branch_weight(branch)
            raw_weights[name] = raw
            if bool(ablation.get(name, False)):
                ignored_branches.append(name)
                effective_raw[name] = 0.0
            else:
                effective_raw[name] = raw

        sync_weight_before_penalty = float(effective_raw.get("sync", 0.0))
        sync_quality_state = self._evaluate_sync_quality(
            sync_quality=sync_quality,
            sync_missing=("sync" in missing_branches),
            sync_ablated=("sync" in ignored_branches),
            sync_weight_before_penalty=sync_weight_before_penalty,
            sync_mismatch_penalty=sync_mismatch_penalty,
        )
        raw_penalty = sync_quality_state.get("applied_penalty", 1.0)
        penalty = 1.0 if raw_penalty is None else float(raw_penalty)
        effective_raw["sync"] = float(max(sync_weight_before_penalty * penalty, 0.0))
        sync_weight_after_penalty = float(effective_raw.get("sync", 0.0))

        total_effective = float(sum(effective_raw.values()))
        if total_effective > 0.0:
            for name, weight in effective_raw.items():
                effective_norm[name] = float(weight / total_effective) if weight > 0 else 0.0
            fake = float(
                sum(branch_values[name][1] * effective_norm.get(name, 0.0) for name in branch_values.keys())
            )
            real = float(
                sum(branch_values[name][0] * effective_norm.get(name, 0.0) for name in branch_values.keys())
            )
            label = "Fake" if fake >= real else "Real"
            confidence = float(abs(fake - real))
            missing_reason = None
        else:
            fake = 0.0
            real = 0.0
            label = "Unknown"
            confidence = 0.0
            effective_norm = {"audio": 0.0, "video": 0.0, "sync": 0.0}
            if branch_values:
                missing_reason = "all_available_branches_were_ablated"
            else:
                missing_reason = "no_valid_branch_probabilities"

        result = {
            "label": label,
            "real": float(real),
            "fake": float(fake),
            "confidence": float(confidence),
            "weights": effective_raw,
        }
        meta = {
            "raw_weights": raw_weights,
            "effective_weights_raw": effective_raw,
            "effective_weights_norm": effective_norm,
            "ignored_branches": ignored_branches,
            "missing_branches": missing_branches,
            "missing_reason": missing_reason,
            "sync_quality": sync_quality_state,
            "sync_weight_before_penalty": sync_weight_before_penalty,
            "sync_weight_after_penalty": sync_weight_after_penalty,
        }
        return result, meta

    @staticmethod
    def _resolve_ablation(
        *,
        ablate: Mapping[str, object] | None,
        ablate_audio: bool,
        ablate_video: bool,
        ablate_sync: bool,
    ) -> dict[str, bool]:
        resolved = {
            "audio": bool(ablate_audio),
            "video": bool(ablate_video),
            "sync": bool(ablate_sync),
        }
        if not isinstance(ablate, Mapping):
            return resolved
        key_map = {
            "audio": "audio",
            "ablate_audio": "audio",
            "video": "video",
            "ablate_video": "video",
            "sync": "sync",
            "ablate_sync": "sync",
        }
        for raw_key, raw_value in ablate.items():
            name = key_map.get(str(raw_key).strip().lower())
            if name is None:
                continue
            resolved[name] = bool(resolved[name] or _coerce_bool(raw_value))
        return resolved

    @staticmethod
    def _apply_ablation_to_weights(weights: dict[str, float], ablation: dict[str, bool]) -> dict[str, float]:
        adjusted = dict(weights)
        for name in ("audio", "video", "sync"):
            if bool(ablation.get(name, False)):
                adjusted[name] = 0.0
        return adjusted

    def _normalize_sync_quality(self, sync_quality: Mapping[str, object] | None) -> dict[str, object]:
        quality = dict(sync_quality) if isinstance(sync_quality, Mapping) else {}
        expected_audio = _safe_int(quality.get("sync_audio_steps_target"))
        actual_audio = _safe_int(quality.get("sync_audio_steps_actual"))
        expected_video = _safe_int(quality.get("sync_video_steps_target"))
        actual_video = _safe_int(quality.get("sync_video_steps_actual"))
        mismatch = _coerce_bool(quality.get("mismatch")) or _coerce_bool(quality.get("t_mismatch"))
        interpolated = _coerce_bool(quality.get("interpolated")) or mismatch
        length_bad = _coerce_bool(quality.get("length_bad"))
        if not length_bad:
            audio_len_invalid = (
                expected_audio is not None and actual_audio is not None and expected_audio != actual_audio
            )
            video_len_invalid = (
                expected_video is not None and actual_video is not None and expected_video != actual_video
            )
            length_bad = bool(audio_len_invalid or video_len_invalid)
        normalized = dict(quality)
        normalized.update(
            {
                "sync_audio_steps_target": expected_audio,
                "sync_audio_steps_actual": actual_audio,
                "sync_video_steps_target": expected_video,
                "sync_video_steps_actual": actual_video,
                "mismatch": bool(mismatch),
                "interpolated": bool(interpolated),
                "length_bad": bool(length_bad),
            }
        )
        return normalized

    def _apply_sync_uncertainty_clamp(
        self,
        *,
        sync: Dict[str, object],
        sync_quality: dict[str, object],
        sync_uncertainty_alpha: float | None,
    ) -> tuple[Dict[str, object], dict[str, object]]:
        adjusted = dict(sync)
        quality = dict(sync_quality)
        alpha_cfg = self._resolve_sync_uncertainty_alpha(sync_uncertainty_alpha)
        trigger_quality = bool(
            _coerce_bool(quality.get("mismatch"))
            or _coerce_bool(quality.get("interpolated"))
            or _coerce_bool(quality.get("length_bad"))
        )
        fake_before = _safe_float(sync.get("fake"))
        real_before = _safe_float(sync.get("real"))
        confidence_before = _safe_float(sync.get("confidence"))
        applied_alpha = 1.0
        if trigger_quality and fake_before is not None:
            applied_alpha = alpha_cfg
            fake_after = 0.5 + (fake_before - 0.5) * applied_alpha
            fake_after = float(max(0.0, min(1.0, fake_after)))
            real_after = float(1.0 - fake_after)
            adjusted["fake"] = fake_after
            adjusted["real"] = real_after
            adjusted["confidence"] = float(max(0.0, min(1.0, abs(fake_after - real_after))))
            adjusted["label"] = "Fake" if fake_after >= real_after else "Real"
        else:
            fake_after = fake_before
            real_after = real_before
        quality.update(
            {
                "clamp_triggered": bool(trigger_quality and fake_before is not None),
                "alpha_config": float(alpha_cfg),
                "applied_alpha": float(applied_alpha),
                "sync_prob_fake_before": fake_before,
                "sync_prob_real_before": real_before,
                "sync_prob_fake_after": fake_after,
                "sync_prob_real_after": real_after,
                "sync_confidence_before": confidence_before,
                "sync_confidence_after": _safe_float(adjusted.get("confidence")),
            }
        )
        return adjusted, quality

    def _evaluate_sync_quality(
        self,
        *,
        sync_quality: Mapping[str, object] | None,
        sync_missing: bool,
        sync_ablated: bool,
        sync_weight_before_penalty: float,
        sync_mismatch_penalty: float | None,
    ) -> dict[str, object]:
        quality = self._normalize_sync_quality(sync_quality)
        expected_audio = _safe_int(quality.get("sync_audio_steps_target"))
        actual_audio = _safe_int(quality.get("sync_audio_steps_actual"))
        expected_video = _safe_int(quality.get("sync_video_steps_target"))
        actual_video = _safe_int(quality.get("sync_video_steps_actual"))
        mismatch = _coerce_bool(quality.get("mismatch"))
        interpolated = _coerce_bool(quality.get("interpolated"))
        length_bad = _coerce_bool(quality.get("length_bad"))
        applied_alpha = _safe_float(quality.get("applied_alpha"))
        if applied_alpha is None:
            applied_alpha = 1.0

        reasons: list[str] = []
        if sync_ablated:
            reasons.append("sync_ablated")
        if sync_missing:
            reasons.append("sync_branch_missing")
        if mismatch:
            reasons.append("sync_mismatch")
        if interpolated:
            reasons.append("sync_interpolated")
        if length_bad:
            reasons.append("sync_length_bad")

        trigger_quality = bool(
            (not sync_ablated)
            and (sync_missing or mismatch or interpolated or length_bad)
        )
        penalty = self._resolve_sync_penalty(sync_mismatch_penalty)
        applied_penalty = float(penalty if trigger_quality else 1.0)
        if not reasons:
            reasons.append("ok")

        return {
            "mismatch": bool(mismatch),
            "interpolated": bool(interpolated),
            "length_bad": bool(length_bad),
            "sync_missing": bool(sync_missing),
            "triggered": bool(trigger_quality),
            "applied_penalty": float(applied_penalty),
            "penalty_config": float(penalty),
            "applied_alpha": float(applied_alpha),
            "alpha_config": _safe_float(quality.get("alpha_config")),
            "audio_steps_expected": expected_audio,
            "audio_steps_actual": actual_audio,
            "video_steps_expected": expected_video,
            "video_steps_actual": actual_video,
            "sync_prob_fake_before": _safe_float(quality.get("sync_prob_fake_before")),
            "sync_prob_fake_after": _safe_float(quality.get("sync_prob_fake_after")),
            "sync_prob_real_before": _safe_float(quality.get("sync_prob_real_before")),
            "sync_prob_real_after": _safe_float(quality.get("sync_prob_real_after")),
            "sync_confidence_before": _safe_float(quality.get("sync_confidence_before")),
            "sync_confidence_after": _safe_float(quality.get("sync_confidence_after")),
            "weight_before_penalty": float(sync_weight_before_penalty),
            "weight_after_penalty": float(sync_weight_before_penalty * applied_penalty),
            "reason": ";".join(reasons),
        }

    def _resolve_sync_uncertainty_alpha(self, override: float | None) -> float:
        if override is None:
            alpha = self._sync_uncertainty_alpha
        else:
            alpha = override
        try:
            value = float(alpha)
        except (TypeError, ValueError):
            value = 1.0
        return float(max(0.0, min(1.0, value)))

    def _resolve_sync_penalty(self, override: float | None) -> float:
        if override is None:
            penalty = self._sync_mismatch_penalty
        else:
            penalty = override
        try:
            value = float(penalty)
        except (TypeError, ValueError):
            value = 0.0
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _apply_sync_quality_penalty_to_weights(
        weights: dict[str, float],
        sync_quality_state: object,
    ) -> dict[str, float]:
        adjusted = dict(weights)
        if not isinstance(sync_quality_state, Mapping):
            return adjusted
        penalty = _safe_float(sync_quality_state.get("applied_penalty"))
        if penalty is None:
            return adjusted
        adjusted["sync"] = float(max(adjusted.get("sync", 0.0) * penalty, 0.0))
        return adjusted

    @staticmethod
    def _branch_weight(branch: Dict[str, object]) -> float:
        confidence = _safe_float(branch.get("confidence"))
        if confidence is None:
            confidence = 0.0
        weight = max(0.1, min(confidence, 1.0))
        if confidence < 0.2:
            weight *= 0.5
        return float(weight)

    def _compose_final_prediction(
        self,
        *,
        final_raw: Dict[str, object],
        audio: Dict[str, object],
        gated: Dict[str, object],
    ) -> Dict[str, object]:
        strategy = self._final_score_strategy
        if strategy == "raw":
            return dict(final_raw)
        if strategy == "gated":
            return dict(gated)

        raw_fake = _safe_float(final_raw.get("fake"))
        audio_fake = _safe_float(audio.get("fake"))
        if raw_fake is None and audio_fake is None:
            return dict(final_raw)
        if raw_fake is None:
            raw_fake = audio_fake
        if audio_fake is None:
            audio_fake = raw_fake
        assert raw_fake is not None
        assert audio_fake is not None

        weight_raw = float(max(self._final_score_raw_weight, 0.0))
        weight_audio = float(max(self._final_score_audio_weight, 0.0))
        total_weight = weight_raw + weight_audio
        if total_weight <= 0.0:
            fake = float(raw_fake)
        else:
            fake = float((weight_raw * raw_fake + weight_audio * audio_fake) / total_weight)
        fake = float(max(0.0, min(1.0, fake)))
        real = float(1.0 - fake)
        return {
            "label": "Fake" if fake >= self._final_fake_threshold else "Real",
            "real": real,
            "fake": fake,
            "confidence": float(abs(fake - real)),
        }


def _is_valid_wavlm_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    weight_files = (
        "pytorch_model.bin",
        "model.safetensors",
        "pytorch_model.bin.index.json",
    )
    return any((path / name).exists() for name in weight_files)


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _sync_mismatch_penalty_cfg(default: float) -> float:
    env_raw = os.environ.get("SYNC_MISMATCH_PENALTY")
    if env_raw is not None:
        parsed = _safe_float(env_raw)
        if parsed is not None:
            return float(max(0.0, min(1.0, parsed)))
    cfg_raw = get_setting("sync_mismatch_penalty", None)
    if cfg_raw is None:
        cfg_raw = get_setting("SYNC_MISMATCH_PENALTY", default)
    parsed = _safe_float(cfg_raw)
    if parsed is None:
        parsed = default
    return float(max(0.0, min(1.0, parsed)))


def _sync_uncertainty_alpha_cfg(default: float) -> float:
    env_raw = os.environ.get("SYNC_UNCERTAINTY_ALPHA")
    if env_raw is not None:
        parsed = _safe_float(env_raw)
        if parsed is not None:
            return float(max(0.0, min(1.0, parsed)))
    cfg_raw = get_setting("sync_uncertainty_alpha", None)
    if cfg_raw is None:
        cfg_raw = get_setting("SYNC_UNCERTAINTY_ALPHA", default)
    parsed = _safe_float(cfg_raw)
    if parsed is None:
        parsed = default
    return float(max(0.0, min(1.0, parsed)))


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        raw = get_setting(name, default)
    if isinstance(raw, (int, float)):
        return float(raw)
    if raw is None:
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _truthy_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        raw = get_setting(name, default)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


_SERVICE: TriModalDetectionService | None = None


def get_service() -> TriModalDetectionService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = TriModalDetectionService()
    return _SERVICE
