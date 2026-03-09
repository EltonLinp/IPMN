from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
from uuid import uuid4
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from preprocessing.fakeav_preprocessor import (
    AudioProcessor,
    AudioSettings,
    VideoProcessor,
    VideoSettings,
)
from .runtime_config import get_setting


@dataclass
class TriModalPreprocessConfig:
    """
    Configuration controlling how web recordings are normalised before inference.
    """

    target_mel_steps: int = 400
    sync_audio_steps: int = 64
    waveform_samples: int = 160000
    video_frames: int = 24
    sync_video_frames: int = 12
    video_size: int = 224
    mel_bins: int = 64
    transcode_to_fakeav: bool = False
    transcode_size: int = 256
    transcode_fps: int = 25
    transcode_crf: int = 23
    transcode_preset: str = "veryfast"
    transcode_audio_rate: int = 16000
    transcode_audio_channels: int = 1
    transcode_container: str = "mp4"


@dataclass
class PreprocessResult:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    mel_sync: torch.Tensor
    video: torch.Tensor
    video_sync: torch.Tensor
    sync_quality: dict[str, object] | None = None
    debug: dict[str, object] | None = None

    def to_device(self, device: torch.device) -> "PreprocessResult":
        return PreprocessResult(
            waveform=self.waveform.to(device),
            waveform_length=self.waveform_length.to(device),
            mel_sync=self.mel_sync.to(device),
            video=self.video.to(device),
            video_sync=self.video_sync.to(device),
            sync_quality=self.sync_quality,
            debug=self.debug,
        )


class TriModalWebPreprocessor:
    """
    Thin wrapper around the project's preprocessing pipeline tailored for single uploads.
    """

    def __init__(
        self,
        *,
        video_settings: VideoSettings | None = None,
        audio_settings: AudioSettings | None = None,
        config: TriModalPreprocessConfig | None = None,
    ) -> None:
        cfg = config or TriModalPreprocessConfig()
        env_transcode = os.environ.get("TRANSCODE_TO_FAKEAV", "").strip().lower()
        if env_transcode in {"1", "true", "yes", "on"}:
            cfg.transcode_to_fakeav = True
        use_face_detector = _env_truthy("USE_FACE_DETECTOR", default=True)
        face_detector = _get_str("FACE_DETECTOR", "insightface")
        video_cfg = video_settings or VideoSettings(
            num_frames=max(cfg.video_frames, cfg.sync_video_frames),
            min_frames=1,
            resize_hw=(cfg.video_size, cfg.video_size),
            sample_strategy="uniform",
            use_face_detector=use_face_detector,
            face_detector=face_detector,
        )
        audio_cfg = audio_settings or AudioSettings(
            sample_rate=16000,
            normalize=True,
            feature_type="mel",
            n_mels=cfg.mel_bins,
            hop_length=160,
        )
        self.cfg = cfg
        self.video_processor = VideoProcessor(video_cfg)
        self.audio_processor = AudioProcessor(audio_cfg)

    def __call__(self, video_path: Path, *, debug: bool = False) -> PreprocessResult:
        source_path = self._maybe_transcode(video_path)
        video_result = self._decode_video(source_path)
        frames = video_result.frames
        audio = self.audio_processor(source_path)
        mel = audio.get("mel")
        waveform = audio.get("waveform")
        if not isinstance(mel, torch.Tensor) or not isinstance(waveform, torch.Tensor):
            raise RuntimeError("Mel spectrogram or waveform could not be extracted from upload.")
        mel_sync, mel_meta = self._prepare_mel_sync(mel)
        video_branch, sync_branch, video_meta = self._prepare_video_views(frames)
        waveform_chunk, waveform_len, waveform_meta = self._prepare_waveform(waveform)
        sync_quality = self._build_sync_quality(
            sync_branch=sync_branch,
            mel_sync=mel_sync,
            sync_video_meta=video_meta,
            sync_audio_meta=mel_meta,
            video_duration_sec=_safe_float(getattr(video_result, "duration_sec", None)),
            audio_samples_raw=_safe_int(waveform_meta.get("raw_samples")),
            audio_sample_rate=_safe_int(audio.get("sample_rate")),
        )
        if source_path != video_path:
            try:
                source_path.unlink(missing_ok=True)
            except Exception:
                pass
        debug_payload = self._build_debug_payload(
            video_result=video_result,
            audio_features=audio,
            video_branch=video_branch,
            sync_branch=sync_branch,
            mel_sync=mel_sync,
            waveform_meta=waveform_meta,
            mel_meta=mel_meta,
            video_meta=video_meta,
            sync_quality=sync_quality,
        ) if debug else None
        return PreprocessResult(
            waveform=waveform_chunk.unsqueeze(0),
            waveform_length=torch.tensor([waveform_len], dtype=torch.long),
            mel_sync=mel_sync.unsqueeze(0),
            video=video_branch.unsqueeze(0),
            video_sync=sync_branch.unsqueeze(0),
            sync_quality=sync_quality,
            debug=debug_payload,
        )

    def _build_sync_quality(
        self,
        *,
        sync_branch: torch.Tensor,
        mel_sync: torch.Tensor,
        sync_video_meta: Mapping[str, object] | None = None,
        sync_audio_meta: Mapping[str, object] | None = None,
        video_duration_sec: float | None = None,
        audio_samples_raw: int | None = None,
        audio_sample_rate: int | None = None,
    ) -> dict[str, object]:
        sync_video_steps = int(sync_branch.size(0))
        sync_audio_steps = int(mel_sync.size(0))
        expected_audio = int(self.cfg.sync_audio_steps)
        expected_video = int(self.cfg.sync_video_frames)
        raw_video_steps = (
            _safe_int(sync_video_meta.get("sync_video_raw_steps"))
            if isinstance(sync_video_meta, Mapping)
            else None
        )
        raw_audio_steps = (
            _safe_int(sync_audio_meta.get("sync_audio_raw_steps"))
            if isinstance(sync_audio_meta, Mapping)
            else None
        )
        if raw_video_steps is None:
            raw_video_steps = sync_video_steps
        if raw_audio_steps is None:
            raw_audio_steps = sync_audio_steps
        video_padded = bool(sync_video_meta.get("sync_video_padded")) if isinstance(sync_video_meta, Mapping) else False
        audio_padded = bool(sync_audio_meta.get("sync_audio_padded")) if isinstance(sync_audio_meta, Mapping) else False
        video_cropped = bool(sync_video_meta.get("sync_video_cropped")) if isinstance(sync_video_meta, Mapping) else False
        audio_cropped = bool(sync_audio_meta.get("sync_audio_cropped")) if isinstance(sync_audio_meta, Mapping) else False
        mel_adjusted = bool(sync_audio_meta.get("mel_padded_or_cropped")) if isinstance(sync_audio_meta, Mapping) else False

        audio_len_invalid = raw_audio_steps < expected_audio
        video_len_invalid = raw_video_steps < expected_video
        interpolated = bool(audio_padded or video_padded or mel_adjusted)
        length_bad = bool(audio_len_invalid or video_len_invalid)

        duration_threshold = _safe_float(os.environ.get("SYNC_DURATION_MISMATCH_RATIO"))
        if duration_threshold is None:
            duration_threshold = _safe_float(get_setting("SYNC_DURATION_MISMATCH_RATIO", 0.2))
        if duration_threshold is None:
            duration_threshold = 0.2
        duration_threshold = float(max(duration_threshold, 0.0))

        video_duration = _safe_float(video_duration_sec)
        audio_duration = None
        if audio_samples_raw is not None and audio_sample_rate is not None and audio_sample_rate > 0:
            audio_duration = float(audio_samples_raw) / float(audio_sample_rate)
        duration_ratio = None
        mismatch = False
        if video_duration is not None and audio_duration is not None and max(video_duration, audio_duration) > 0.0:
            duration_ratio = abs(video_duration - audio_duration) / max(video_duration, audio_duration)
            mismatch = bool(duration_ratio > duration_threshold)
        if not mismatch and (length_bad or interpolated):
            mismatch = True

        reasons: list[str] = []
        if mismatch:
            reasons.append("sync_mismatch")
        if interpolated:
            reasons.append("sync_interpolated")
        if audio_len_invalid:
            reasons.append(f"sync_audio_steps_insufficient:{raw_audio_steps}<{expected_audio}")
        if video_len_invalid:
            reasons.append(f"sync_video_steps_insufficient:{raw_video_steps}<{expected_video}")
        if mel_adjusted:
            reasons.append("sync_audio_mel_adjusted")
        if duration_ratio is not None:
            reasons.append(f"sync_duration_ratio:{duration_ratio:.3f}")
        if not reasons:
            reasons.append("ok")
        return {
            "mismatch": mismatch,
            "interpolated": interpolated,
            "sync_audio_steps_target": expected_audio,
            "sync_audio_steps_actual": sync_audio_steps,
            "sync_audio_steps_before": raw_audio_steps,
            "sync_video_steps_target": expected_video,
            "sync_video_steps_actual": sync_video_steps,
            "sync_video_steps_before": raw_video_steps,
            "audio_steps_valid": not audio_len_invalid,
            "video_steps_valid": not video_len_invalid,
            "audio_padded": bool(audio_padded),
            "video_padded": bool(video_padded),
            "audio_cropped": bool(audio_cropped),
            "video_cropped": bool(video_cropped),
            "audio_mel_adjusted": bool(mel_adjusted),
            "audio_duration_sec": audio_duration,
            "video_duration_sec": video_duration,
            "duration_mismatch_ratio": duration_ratio,
            "duration_mismatch_threshold": duration_threshold,
            "length_bad": length_bad,
            "reason": ";".join(reasons),
        }

    def _decode_video(self, video_path: Path):
        """
        Decord often fails for browser-recorded WebM clips (broken metadata).
        Retry once by forcing the processor to switch to OpenCV decoding.
        """

        try:
            return self.video_processor(video_path)
        except Exception:
            backend = getattr(self.video_processor, "_backend", "")
            if backend != "opencv":
                self.video_processor._backend = "opencv"
                return self.video_processor(video_path)
            raise

    def _maybe_transcode(self, video_path: Path) -> Path:
        cfg = self.cfg
        if not cfg.transcode_to_fakeav:
            return video_path
        storage_dir = Path(os.environ.get("UPLOAD_DIR", "userVisualization/storage/uploads")).resolve().parent
        trans_dir = storage_dir / "transcoded"
        trans_dir.mkdir(parents=True, exist_ok=True)
        suffix = cfg.transcode_container if cfg.transcode_container.startswith(".") else f".{cfg.transcode_container}"
        target_path = trans_dir / f"{uuid4().hex}{suffix}"
        vf = f"scale={cfg.transcode_size}:{cfg.transcode_size},fps={cfg.transcode_fps}"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            cfg.transcode_preset,
            "-crf",
            str(cfg.transcode_crf),
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            str(cfg.transcode_audio_rate),
            "-ac",
            str(cfg.transcode_audio_channels),
            str(target_path),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg not found on PATH; install or disable transcoding.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg transcode failed: {stderr}") from exc
        return target_path

    def _prepare_video_views(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
        if frames.dim() != 4:
            raise ValueError(f"Unexpected frame tensor shape: {frames.shape}")
        video = frames.float()
        if video.size(-1) != self.cfg.video_size or video.size(-2) != self.cfg.video_size:
            video = F.interpolate(
                video,
                size=(self.cfg.video_size, self.cfg.video_size),
                mode="bilinear",
                align_corners=False,
            )
        video_branch = self._temporal_slice(video, self.cfg.video_frames)
        sync_branch, sync_meta = self._temporal_slice_with_meta(video, self.cfg.sync_video_frames)
        video_branch = video_branch.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
        sync_branch = sync_branch.contiguous()  # [T, C, H, W]
        meta = {
            "sync_video_raw_steps": sync_meta.get("before"),
            "sync_video_steps_after": sync_meta.get("after"),
            "sync_video_padded": sync_meta.get("padded"),
            "sync_video_cropped": sync_meta.get("cropped"),
        }
        return video_branch, sync_branch, meta

    def _prepare_waveform(self, waveform: torch.Tensor) -> tuple[torch.Tensor, int, dict[str, Any]]:
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        total = waveform.size(-1)
        target = max(int(self.cfg.waveform_samples), 1)
        mode = "exact"
        pad = 0
        truncated = 0
        if total >= target:
            start = max((total - target) // 2, 0)
            chunk = waveform[..., start : start + target]
            if total > target:
                mode = "truncated"
                truncated = int(total - target)
        else:
            pad = target - total
            chunk = F.pad(waveform, (0, pad))
            mode = "padded"
        length = min(total, target)
        meta = {
            "raw_samples": int(total),
            "target_samples": int(target),
            "used_samples": int(length),
            "mode": mode,
            "padding_samples": int(pad),
            "truncated_samples": int(truncated),
        }
        return chunk.contiguous(), length, meta

    def _prepare_mel_sync(self, mel: torch.Tensor) -> tuple[torch.Tensor, dict[str, object]]:
        input_shape = list(mel.shape)
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
        mel = mel.float()
        mel = self._maybe_resample_mel(mel)
        mel_steps_before_pad_crop = int(mel.size(-1))
        mel = self._pad_crop_mel(mel)
        mel_padded_shape = list(mel.shape)
        sync_base = mel.squeeze(0).permute(1, 0).contiguous()
        sync, sync_meta = self._temporal_slice_with_meta(sync_base, self.cfg.sync_audio_steps)
        meta = {
            "input_shape": input_shape,
            "mel_shape": mel_padded_shape,
            "sync_shape": list(sync.shape),
            "mel_steps_before_pad_crop": mel_steps_before_pad_crop,
            "mel_steps_after_pad_crop": int(mel.size(-1)),
            "mel_padded_or_cropped": bool(mel_steps_before_pad_crop != int(mel.size(-1))),
            "sync_audio_raw_steps": sync_meta.get("before"),
            "sync_audio_steps_after": sync_meta.get("after"),
            "sync_audio_padded": sync_meta.get("padded"),
            "sync_audio_cropped": sync_meta.get("cropped"),
        }
        return sync, meta

    def _build_debug_payload(
        self,
        *,
        video_result,
        audio_features: dict[str, object],
        video_branch: torch.Tensor,
        sync_branch: torch.Tensor,
        mel_sync: torch.Tensor,
        waveform_meta: dict[str, Any],
        mel_meta: dict[str, object],
        video_meta: dict[str, object],
        sync_quality: dict[str, object],
    ) -> dict[str, object]:
        frame_indices = []
        if hasattr(video_result, "frame_indices") and isinstance(video_result.frame_indices, torch.Tensor):
            frame_indices = [int(v) for v in video_result.frame_indices.detach().cpu().tolist()]
        face_detected = getattr(video_result, "face_detected_frames", None)
        sampled_len = len(frame_indices)
        face_ratio = (
            float(face_detected) / float(sampled_len)
            if isinstance(face_detected, int) and sampled_len > 0
            else None
        )

        audio_sample_rate = audio_features.get("sample_rate")
        audio_has_waveform = isinstance(audio_features.get("waveform"), torch.Tensor)

        sync_video_len = int(sync_branch.size(0))
        sync_audio_len = int(mel_sync.size(0))
        sync_video_before = _safe_int(video_meta.get("sync_video_raw_steps")) or sync_video_len
        sync_audio_before = _safe_int(mel_meta.get("sync_audio_raw_steps")) or sync_audio_len
        mismatch = bool(sync_quality.get("mismatch"))
        interpolated = bool(sync_quality.get("interpolated"))
        length_bad = bool(sync_quality.get("length_bad"))
        return {
            "video": {
                "duration_sec": _safe_float(getattr(video_result, "duration_sec", None)),
                "fps": _safe_float(getattr(video_result, "fps", None)),
                "total_frames": _safe_int(getattr(video_result, "total_frames", None)),
                "sampled_indices_len": sampled_len,
                "sampled_indices": frame_indices,
                "video_branch_frames": int(video_branch.size(1)),
                "sync_video_frames": sync_video_len,
            },
            "face_crop": {
                "use_face_detector": _safe_bool(getattr(video_result, "face_crop_requested", None)),
                "face_detector": getattr(video_result, "face_detector", None),
                "detector_ready": _safe_bool(getattr(video_result, "face_detector_ready", None)),
                "detected_frames": _safe_int(face_detected),
                "detected_ratio": face_ratio,
                "center_crop_frames": _safe_int(getattr(video_result, "center_crop_frames", None)),
                "fallback_to_center_crop": _safe_bool(
                    (getattr(video_result, "center_crop_frames", 0) or 0) > 0
                ),
            },
            "audio": {
                "audio_present": audio_has_waveform,
                "sample_rate": _safe_int(audio_sample_rate),
                "waveform_samples_raw": waveform_meta.get("raw_samples"),
                "waveform_samples_target": waveform_meta.get("target_samples"),
                "waveform_samples_used": waveform_meta.get("used_samples"),
                "waveform_adjustment": waveform_meta.get("mode"),
                "waveform_padding_samples": waveform_meta.get("padding_samples"),
                "waveform_truncated_samples": waveform_meta.get("truncated_samples"),
                "mel_shape": mel_meta.get("mel_shape"),
                "mel_input_shape": mel_meta.get("input_shape"),
                "mel_sync_shape": mel_meta.get("sync_shape"),
            },
            "sync": {
                "sync_audio_steps_target": int(self.cfg.sync_audio_steps),
                "sync_audio_steps_actual": sync_audio_len,
                "sync_audio_steps_before": sync_audio_before,
                "sync_video_steps_target": int(self.cfg.sync_video_frames),
                "sync_video_steps_actual": sync_video_len,
                "sync_video_steps_before": sync_video_before,
                "t_mismatch": mismatch,
                "interpolated": interpolated,
                "audio_steps_valid": not bool(sync_quality.get("audio_padded", False) or sync_quality.get("audio_mel_adjusted", False)),
                "video_steps_valid": not bool(sync_quality.get("video_padded", False)),
                "length_bad": length_bad,
                "length_before": {
                    "video": sync_video_before,
                    "audio": sync_audio_before,
                },
                "length_after": {
                    "video": sync_video_len,
                    "audio": sync_audio_len,
                },
            },
        }

    def _pad_crop_mel(self, mel: torch.Tensor) -> torch.Tensor:
        target = max(int(self.cfg.target_mel_steps), 1)
        steps = mel.size(-1)
        if steps == target:
            return mel
        if steps > target:
            start = max((steps - target) // 2, 0)
            return mel[..., start : start + target]
        pad_total = target - steps
        left = pad_total // 2
        right = pad_total - left
        return F.pad(mel, (left, right))

    def _maybe_resample_mel(self, mel: torch.Tensor) -> torch.Tensor:
        target = self.cfg.mel_bins
        if target is None or mel.size(-2) == target:
            return mel
        mel_4d = mel.unsqueeze(0)  # [1, C, mel_bins, steps]
        resized = F.interpolate(
            mel_4d,
            size=(target, mel.size(-1)),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

    def _temporal_slice(self, seq: torch.Tensor, length: int) -> torch.Tensor:
        sliced, _ = self._temporal_slice_with_meta(seq, length)
        return sliced

    def _temporal_slice_with_meta(self, seq: torch.Tensor, length: int) -> tuple[torch.Tensor, dict[str, object]]:
        if length <= 0:
            return seq, {"before": int(seq.size(0)), "after": int(seq.size(0)), "padded": False, "cropped": False}
        total = seq.size(0)
        if total == length:
            return seq, {"before": int(total), "after": int(total), "padded": False, "cropped": False}
        if total > length:
            start = max((total - length) // 2, 0)
            sliced = seq[start : start + length]
            return sliced, {"before": int(total), "after": int(length), "padded": False, "cropped": True}
        pad = length - total
        pad_before = pad // 2
        pad_after = pad - pad_before
        start_pad = seq[:1].expand(pad_before, *seq.shape[1:]) if pad_before > 0 else None
        end_pad = seq[-1:].expand(pad_after, *seq.shape[1:]) if pad_after > 0 else None
        pieces = [p for p in (start_pad, seq, end_pad) if p is not None]
        padded = torch.cat(pieces, dim=0)
        return padded, {"before": int(total), "after": int(length), "padded": True, "cropped": False}


def build_default_preprocessor() -> TriModalWebPreprocessor:
    """
    Convenience helper that builds a preprocessor with sane defaults for the UI.
    """
    use_face_detector = _env_truthy("USE_FACE_DETECTOR", default=True)
    face_detector = _get_str("FACE_DETECTOR", "insightface")
    video_cfg = VideoSettings(
        num_frames=64,
        min_frames=1,
        resize_hw=(224, 224),
        sample_strategy="uniform",
        use_face_detector=use_face_detector,
        face_detector=face_detector,
        use_cuda=False,
    )
    audio_cfg = AudioSettings(
        sample_rate=16000,
        normalize=True,
        feature_type="mel",
        n_mels=64,
        hop_length=160,
    )
    return TriModalWebPreprocessor(video_settings=video_cfg, audio_settings=audio_cfg)


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


def _safe_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    return bool(value)


def _env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        raw = get_setting(name, default)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        raw = get_setting(name, default)
    if raw is None:
        return default
    text = str(raw).strip()
    return text or default
