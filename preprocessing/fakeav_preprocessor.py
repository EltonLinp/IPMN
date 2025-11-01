from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

try:
    import decord  # type: ignore

    _DECORD_AVAILABLE = True
except ImportError:
    decord = None  # type: ignore
    _DECORD_AVAILABLE = False

try:
    import cv2  # type: ignore

    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False

try:
    from insightface.app import FaceAnalysis  # type: ignore

    _INSIGHTFACE_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    FaceAnalysis = None  # type: ignore
    _INSIGHTFACE_AVAILABLE = False

try:
    from moviepy.editor import VideoFileClip  # type: ignore

    _MOVIEPY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    VideoFileClip = None  # type: ignore
    _MOVIEPY_AVAILABLE = False


Label = int


@dataclass
class VideoSettings:
    num_frames: int = 32
    min_frames: int = 8
    resize_hw: Tuple[int, int] = (224, 224)
    sample_strategy: str = "uniform"
    use_face_detector: bool = False
    face_detector: str = "insightface"
    face_enlarge: float = 1.25
    detection_size: Tuple[int, int] = (640, 640)
    use_cuda: bool = field(default_factory=lambda: torch.cuda.is_available())
    seed: Optional[int] = None


@dataclass
class VideoResult:
    frames: torch.Tensor
    frame_indices: torch.Tensor
    timestamps: torch.Tensor
    fps: float


@dataclass
class AudioSettings:
    sample_rate: int = 16000
    normalize: bool = True
    feature_type: str = "mel"
    n_mels: int = 64
    win_length: int = 400
    hop_length: int = 160
    top_db: Optional[float] = 80.0


@dataclass
class PipelineConfig:
    dataset_root: Path
    output_root: Path
    metadata_csv: str = "meta_data.csv"
    speakers: Optional[int] = 15
    real_per_speaker: int = 4
    fake_per_speaker: int = 4
    seed: int = 1337
    skip_existing: bool = True
    save_waveform: bool = False
    index_file: str = "preprocess_index.jsonl"
    num_workers: int = 0
    log_level: int = logging.INFO
    video: VideoSettings = field(default_factory=VideoSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)

    def __post_init__(self) -> None:
        self.dataset_root = Path(self.dataset_root)
        self.output_root = Path(self.output_root)


@dataclass
class SampleEntry:
    speaker_id: str
    sample_id: str
    media_path: Path
    label: Label
    type_name: str
    meta: Dict[str, str]


class FaceCropper:
    def __init__(
        self,
        detector_name: str,
        use_cuda: bool,
        detection_size: Tuple[int, int],
        enlarge: float,
    ) -> None:
        self.detector_name = detector_name.lower()
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.detection_size = detection_size
        self.enlarge = enlarge
        self._detector = None
        self._init_detector()

    def _init_detector(self) -> None:
        if self.detector_name != "insightface":
            logging.warning("Unsupported face detector %s. Falling back to center crop.", self.detector_name)
            return
        if not _INSIGHTFACE_AVAILABLE:
            logging.warning("insightface is not installed. Falling back to center crop.")
            return
        try:
            detector = FaceAnalysis(name="buffalo_l")  # type: ignore
            ctx_id = 0 if self.use_cuda else -1
            detector.prepare(ctx_id=ctx_id, det_size=self.detection_size)  # type: ignore[arg-type]
            self._detector = detector
        except Exception as exc:  # pragma: no cover - dependent on external lib
            logging.warning("Failed to initialise insightface (%s). Using center crop.", exc)
            self._detector = None

    def __call__(self, frame: np.ndarray) -> Optional[np.ndarray]:
        if self._detector is None:
            return None
        faces = self._detector.get(frame[:, :, ::-1])  # type: ignore[union-attr]
        if not faces:
            return None
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = frame.shape[:2]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = (x2 - x1) * self.enlarge
        bh = (y2 - y1) * self.enlarge
        nx1 = max(int(cx - bw / 2), 0)
        ny1 = max(int(cy - bh / 2), 0)
        nx2 = min(int(cx + bw / 2), w)
        ny2 = min(int(cy + bh / 2), h)
        if nx2 - nx1 <= 1 or ny2 - ny1 <= 1:
            return None
        return frame[ny1:ny2, nx1:nx2]


class VideoProcessor:
    def __init__(self, settings: VideoSettings) -> None:
        self.settings = settings
        seed = settings.seed
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self._backend = self._resolve_backend()
        self._face_cropper = (
            FaceCropper(
                detector_name=settings.face_detector,
                use_cuda=settings.use_cuda,
                detection_size=settings.detection_size,
                enlarge=settings.face_enlarge,
            )
            if settings.use_face_detector
            else None
        )

    def _resolve_backend(self) -> str:
        if _DECORD_AVAILABLE:
            try:
                _ = decord.gpu(0) if self.settings.use_cuda else decord.cpu(0)
                return "decord"
            except Exception:
                pass
        if _CV2_AVAILABLE:
            return "opencv"
        raise ImportError("Neither decord nor opencv-python is available for video decoding.")

    def _uniform_indices(self, total: int) -> np.ndarray:
        num = min(self.settings.num_frames, total)
        if num <= 0:
            return np.array([], dtype=np.int64)
        if total == 1:
            return np.array([0], dtype=np.int64)
        return np.linspace(0, total - 1, num=num, dtype=np.int64)

    def _random_indices(self, total: int) -> np.ndarray:
        num = min(self.settings.num_frames, total)
        if num <= 0:
            return np.array([], dtype=np.int64)
        replace = num > total
        indices = self._rng.choice(total, size=num, replace=replace)
        return np.sort(indices.astype(np.int64))

    def _select_indices(self, total: int) -> np.ndarray:
        strategy = self.settings.sample_strategy.lower()
        if strategy == "uniform":
            indices = self._uniform_indices(total)
        elif strategy == "random":
            indices = self._random_indices(total)
        else:
            raise ValueError(f"Unsupported frame sampling strategy: {self.settings.sample_strategy}")
        return indices

    def _read_frames_decord(self, video_path: Path) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        use_gpu = self.settings.use_cuda and torch.cuda.is_available()
        ctx = decord.gpu(0) if use_gpu else decord.cpu(0)
        try:
            reader = decord.VideoReader(str(video_path), ctx=ctx)
        except Exception as exc:
            if use_gpu:
                logging.warning("Decord GPU 初始化失败，回退到 CPU：%s", exc)
                self.settings.use_cuda = False
                return self._read_frames_decord(video_path)
            raise
        fps = float(reader.get_avg_fps() or 0.0)
        total = len(reader)
        indices = self._select_indices(total)
        if indices.size == 0:
            return np.empty((0, 0, 0, 3), dtype=np.uint8), indices, fps, np.empty(0, dtype=np.float32)
        frames = reader.get_batch(indices).asnumpy()
        timestamps = []
        for idx in indices:
            try:
                start, _ = reader.get_frame_timestamp(int(idx))
                timestamps.append(float(start))
            except Exception:
                timestamps.append(float(idx) / fps if fps > 0 else float(idx) / 25.0)
        return frames, indices, fps, np.array(timestamps, dtype=np.float32)

    def _read_frames_opencv(self, video_path: Path) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        indices = self._select_indices(total if total > 0 else self.settings.num_frames)
        if indices.size == 0:
            capture.release()
            return np.empty((0, 0, 0, 3), dtype=np.uint8), indices, fps, np.empty(0, dtype=np.float32)
        frames: List[np.ndarray] = []
        collected_indices: List[int] = []
        target_set = set(indices.tolist())
        current = 0
        success, frame = capture.read()
        while success:
            if current in target_set:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                collected_indices.append(current)
                if len(frames) == len(target_set):
                    break
            success, frame = capture.read()
            current += 1
        capture.release()
        if not frames:
            raise RuntimeError(f"Could not read frames from {video_path}")
        collected = np.array(collected_indices, dtype=np.int64)
        if fps <= 0:
            fps = 25.0
        timestamps = collected.astype(np.float32) / fps
        return np.stack(frames, axis=0), collected, fps, timestamps

    def _center_crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        side = min(h, w)
        top = (h - side) // 2
        left = (w - side) // 2
        return frame[top : top + side, left : left + side]

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=self.settings.resize_hw,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        tensor = (tensor - 0.5) / 0.5
        return tensor

    def __call__(self, video_path: Path) -> VideoResult:
        if self._backend == "decord":
            frames_np, indices, fps, timestamps_np = self._read_frames_decord(video_path)
        else:
            frames_np, indices, fps, timestamps_np = self._read_frames_opencv(video_path)
        if frames_np.size == 0:
            raise RuntimeError(f"No frames extracted from {video_path}")
        if fps <= 0:
            fps = 25.0
        processed: List[torch.Tensor] = []
        for frame in frames_np:
            crop = self._face_cropper(frame) if self._face_cropper else None
            region = crop if crop is not None else self._center_crop(frame)
            processed.append(self._frame_to_tensor(region))
        stacked = torch.stack(processed)
        if stacked.shape[0] < self.settings.min_frames:
            raise RuntimeError(f"Only {stacked.shape[0]} frames extracted from {video_path}")
        frame_indices = torch.from_numpy(indices.astype(np.int64))
        timestamps = torch.from_numpy(timestamps_np.astype(np.float32))
        return VideoResult(
            frames=stacked,
            frame_indices=frame_indices,
            timestamps=timestamps,
            fps=fps,
        )


class AudioProcessor:
    def __init__(self, settings: AudioSettings) -> None:
        self.settings = settings
        self._mel_transform = None
        self._db_transform = None

    def _mel(self) -> torchaudio.transforms.MelSpectrogram:
        if self._mel_transform is None:
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.settings.sample_rate,
                n_fft=self.settings.win_length,
                win_length=self.settings.win_length,
                hop_length=self.settings.hop_length,
                n_mels=self.settings.n_mels,
            )
        return self._mel_transform

    def _to_db(self) -> torchaudio.transforms.AmplitudeToDB:
        if self._db_transform is None:
            self._db_transform = torchaudio.transforms.AmplitudeToDB(top_db=self.settings.top_db)
        return self._db_transform

    def _load_audio(self, media_path: Path) -> Tuple[torch.Tensor, int]:
        try:
            waveform, sr = self._load_with_stream_reader(media_path)
        except Exception as stream_err:
            try:
                waveform, sr = self._load_with_torchaudio(media_path)
            except Exception as load_err:
                if _MOVIEPY_AVAILABLE:
                    logging.debug("torchaudio failed (%s); falling back to moviepy.", load_err)
                    return self._load_audio_moviepy(media_path)
                raise RuntimeError(
                    f"Failed to decode audio for {media_path}. Install FFmpeg and restart."
                ) from load_err
        return waveform, sr

    def _load_with_stream_reader(self, media_path: Path) -> Tuple[torch.Tensor, int]:
        if not hasattr(torchaudio, "io") or not hasattr(torchaudio.io, "StreamReader"):
            raise AttributeError("torchaudio.io.StreamReader is not available.")
        reader = torchaudio.io.StreamReader(str(media_path))
        reader.add_audio_stream(
            frames_per_chunk=0,
            sample_rate=self.settings.sample_rate,
        )
        chunks: List[torch.Tensor] = []
        for (chunk,) in reader.stream():
            chunks.append(chunk.transpose(0, 1))
        if not chunks:
            raise RuntimeError("StreamReader produced no audio frames.")
        waveform = torch.cat(chunks, dim=1)
        return waveform, self.settings.sample_rate

    def _load_with_torchaudio(self, media_path: Path) -> Tuple[torch.Tensor, int]:
        if not hasattr(torchaudio, "load"):
            raise AttributeError("torchaudio.load is not available.")
        try:
            waveform, sr = torchaudio.load(str(media_path))
        except ImportError as codec_err:
            raise RuntimeError(
                "torchaudio.load requires torchcodec. Install torchcodec or FFmpeg."
            ) from codec_err
        return waveform, sr

    def _load_audio_moviepy(self, media_path: Path) -> Tuple[torch.Tensor, int]:
        if not _MOVIEPY_AVAILABLE:
            raise RuntimeError("moviepy is not available for audio fallback.")
        clip = VideoFileClip(str(media_path))
        try:
            if clip.audio is None:
                raise RuntimeError("No audio stream detected.")
            array = clip.audio.to_soundarray(fps=self.settings.sample_rate)
        finally:
            clip.close()
        if array.ndim == 1:
            array = array[:, None]
        waveform = torch.from_numpy(array.T).float()
        return waveform, self.settings.sample_rate

    def __call__(self, media_path: Path) -> Dict[str, torch.Tensor | int | float]:
        waveform, sr = self._load_audio(media_path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.settings.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.settings.sample_rate)
            sr = self.settings.sample_rate
        if self.settings.normalize:
            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / max_val
        features: Dict[str, torch.Tensor | int | float] = {
            "waveform": waveform,
            "sample_rate": sr,
        }
        if self.settings.feature_type.lower() == "mel":
            mel = self._mel()(waveform)
            mel = self._to_db()(mel)
            features["mel"] = mel
            num_steps = mel.shape[-1]
            mel_times = torch.arange(num_steps, dtype=torch.float32) * (self.settings.hop_length / float(sr))
            features["mel_times"] = mel_times
        features["hop_length"] = self.settings.hop_length
        return features


class FakeAVPreprocessor:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        if self.config.video.seed is None:
            self.config.video.seed = config.seed
        self.output_root = config.output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random(config.seed)
        self.video_processor = VideoProcessor(config.video)
        self.audio_processor = AudioProcessor(config.audio)
        logging.basicConfig(
            level=config.log_level,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    @staticmethod
    def _label_from_type(type_name: str) -> Label:
        return 0 if type_name == "RealVideo-RealAudio" else 1

    def _load_metadata(self) -> pd.DataFrame:
        meta_path = self.config.dataset_root / self.config.metadata_csv
        df = pd.read_csv(meta_path)
        df["speaker_id"] = df["source"]
        df["label"] = df["type"].apply(self._label_from_type)
        df["full_path"] = df.apply(
            lambda row: self.config.dataset_root
            / str(row["type"]).strip()
            / str(row["race"]).strip()
            / str(row["gender"]).strip()
            / str(row["source"]).strip()
            / str(row["path"]).strip(),
            axis=1,
        )
        df = df[df["full_path"].apply(lambda p: Path(p).exists())].copy()
        return df

    def _sample_rows(self, rows: pd.DataFrame, count: int) -> List[pd.Series]:
        records = rows.to_dict("records")
        if len(records) <= count:
            return [pd.Series(r) for r in records]
        selected = self._rng.sample(records, count)
        return [pd.Series(r) for r in selected]

    def _select_samples(self, df: pd.DataFrame) -> List[SampleEntry]:
        samples: List[SampleEntry] = []
        grouped = df.groupby("speaker_id")
        eligible_speakers: List[str] = []
        for speaker, group in grouped:
            real_count = int((group["label"] == 0).sum())
            fake_count = int((group["label"] == 1).sum())
            if real_count > 0 and fake_count > 0:
                eligible_speakers.append(speaker)
        requested = self.config.speakers if (self.config.speakers and self.config.speakers > 0) else len(eligible_speakers)
        if len(eligible_speakers) < requested:
            logging.warning(
                "Only %d speakers meet the sampling criteria (requested %d).",
                len(eligible_speakers),
                requested,
            )
        if not eligible_speakers:
            return samples
        if requested >= len(eligible_speakers):
            selected_speakers = eligible_speakers
        else:
            selected_speakers = self._rng.sample(eligible_speakers, k=requested)
        for speaker in selected_speakers:
            group = grouped.get_group(speaker)
            real_rows = group[group["label"] == 0]
            fake_rows = group[group["label"] == 1]
            real_target = min(len(real_rows), self.config.real_per_speaker)
            fake_target = min(len(fake_rows), self.config.fake_per_speaker)
            if real_target == 0 or fake_target == 0:
                continue
            for row in self._sample_rows(real_rows, real_target):
                sample_id = f"{row['speaker_id']}_real_{Path(row['path']).stem}"
                samples.append(
                    SampleEntry(
                        speaker_id=row["speaker_id"],
                        sample_id=sample_id,
                        media_path=Path(row["full_path"]),
                        label=0,
                        type_name=row["type"],
                        meta={k: row[k] for k in ("method", "category", "race", "gender")},
                    )
                )
            for row in self._sample_rows(fake_rows, fake_target):
                sample_id = f"{row['speaker_id']}_fake_{Path(row['path']).stem}"
                samples.append(
                    SampleEntry(
                        speaker_id=row["speaker_id"],
                        sample_id=sample_id,
                        media_path=Path(row["full_path"]),
                        label=1,
                        type_name=row["type"],
                        meta={k: row[k] for k in ("method", "category", "race", "gender")},
                    )
                )
        return samples

    def _sample_output_path(self, sample_id: str) -> Path:
        return self.output_root / f"{sample_id}.pt"

    def _save_sample(
        self,
        sample: SampleEntry,
        video: VideoResult,
        audio_features: Dict[str, torch.Tensor | int | float],
    ) -> Path:
        audio_payload: Dict[str, torch.Tensor | int | float] = {}
        mel_tensor = audio_features.get("mel")
        if isinstance(mel_tensor, torch.Tensor):
            audio_payload["mel"] = mel_tensor
        waveform_tensor = audio_features.get("waveform")
        if self.config.save_waveform and isinstance(waveform_tensor, torch.Tensor):
            audio_payload["waveform"] = waveform_tensor
        audio_payload["sample_rate"] = audio_features.get("sample_rate")
        audio_payload["hop_length"] = audio_features.get("hop_length")
        if "mel_times" in audio_features and isinstance(audio_features["mel_times"], torch.Tensor):
            audio_payload["mel_times"] = audio_features["mel_times"]
        payload = {
            "speaker_id": sample.speaker_id,
            "label": sample.label,
            "type": sample.type_name,
            "video": video.frames,
            "sync": {
                "frame_indices": video.frame_indices,
                "frame_timestamps": video.timestamps,
                "video_fps": video.fps,
                "mel_timestamps": audio_features.get("mel_times"),
                "audio_sample_rate": audio_features.get("sample_rate"),
                "audio_hop_length": audio_features.get("hop_length"),
            },
            "audio": audio_payload,
        }
        output_path = self._sample_output_path(sample.sample_id)
        torch.save(payload, output_path)
        return output_path

    def run(self) -> None:
        df = self._load_metadata()
        samples = self._select_samples(df)
        if not samples:
            logging.error("No samples selected. Adjust the sampling configuration.")
            return
        index_records: List[Dict[str, object]] = []
        index_path = self.output_root / self.config.index_file
        with tqdm(total=len(samples), desc="Preprocessing samples") as progress:
            for sample in samples:
                output_path = self._sample_output_path(sample.sample_id)
                if self.config.skip_existing and output_path.exists():
                    logging.info("Skipping existing sample %s", sample.sample_id)
                    index_records.append(
                        {
                            "sample_id": sample.sample_id,
                            "speaker_id": sample.speaker_id,
                            "label": sample.label,
                            "type": sample.type_name,
                            "media_path": str(sample.media_path),
                            "output_path": str(output_path),
                            "status": "skipped",
                        }
                    )
                    progress.update(1)
                    continue
                try:
                    video_result = self.video_processor(sample.media_path)
                    audio_features = self.audio_processor(sample.media_path)
                    mel_tensor = audio_features.get("mel")
                    mel_shape = list(mel_tensor.shape) if isinstance(mel_tensor, torch.Tensor) else None
                    waveform_tensor = audio_features.get("waveform") if self.config.save_waveform else None
                    waveform_shape = (
                        list(waveform_tensor.shape) if isinstance(waveform_tensor, torch.Tensor) else None
                    )
                    saved_path = self._save_sample(sample, video_result, audio_features)
                    index_records.append(
                        {
                            "sample_id": sample.sample_id,
                            "speaker_id": sample.speaker_id,
                            "label": sample.label,
                            "type": sample.type_name,
                            "media_path": str(sample.media_path),
                            "output_path": str(saved_path),
                            "video_frames": int(video_result.frames.shape[0]),
                            "video_shape": list(video_result.frames.shape),
                            "mel_shape": mel_shape,
                            "waveform_shape": waveform_shape,
                            "sync": {
                                "frame_count": int(video_result.frames.shape[0]),
                                "video_fps": float(video_result.fps),
                                "mel_steps": mel_shape[-1] if mel_shape else None,
                                "audio_sample_rate": int(audio_features.get("sample_rate", 0) or 0),
                                "duration_sec": float(video_result.timestamps[-1].item()) if video_result.timestamps.numel() else None,
                            },
                            "status": "ok",
                        }
                    )
                except Exception as exc:
                    logging.exception("Failed to process %s: %s", sample.media_path, exc)
                    index_records.append(
                        {
                            "sample_id": sample.sample_id,
                            "speaker_id": sample.speaker_id,
                            "label": sample.label,
                            "type": sample.type_name,
                            "media_path": str(sample.media_path),
                            "status": f"error: {exc}",
                        }
                    )
                progress.update(1)
        with index_path.open("w", encoding="utf-8") as handle:
            for record in index_records:
                handle.write(json.dumps(record) + "\n")
        logging.info("Preprocessing finished. Results saved to %s", self.output_root)


def build_default_config(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    speakers: int | None = 15,
    real_per_speaker: int = 4,
    fake_per_speaker: int = 4,
    seed: int = 1337,
) -> PipelineConfig:
    speakers = speakers if (speakers is None or speakers > 0) else None
    return PipelineConfig(
        dataset_root=Path(dataset_root),
        output_root=Path(output_root),
        speakers=speakers,
        real_per_speaker=real_per_speaker,
        fake_per_speaker=fake_per_speaker,
        seed=seed,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="FakeAVCeleb preprocessing pipeline")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to FakeAVCeleb_v1.2 directory.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory to save processed tensors.")
    parser.add_argument("--speakers", type=int, default=15, help="Number of speaker IDs to sample (<=0 to use all).")
    parser.add_argument("--real-per-speaker", type=int, default=4, help="Real samples per speaker.")
    parser.add_argument("--fake-per-speaker", type=int, default=4, help="Fake samples per speaker.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument("--use-face-detector", action="store_true", help="Enable insightface cropping if available.")
    parser.add_argument("--num-frames", type=int, default=32, help="Number of frames per clip.")
    parser.add_argument("--frame-size", type=int, nargs=2, metavar=("H", "W"), default=(224, 224), help="Video resize.")
    parser.add_argument("--sample-strategy", type=str, default="uniform", choices=("uniform", "random"))
    parser.add_argument("--save-waveform", action="store_true", help="Store raw audio waveform alongside features.")
    parser.add_argument("--skip-existing", action="store_true", default=False, help="Skip already processed samples.")
    args = parser.parse_args()

    video_settings = VideoSettings(
        num_frames=args.num_frames,
        resize_hw=tuple(args.frame_size),
        sample_strategy=args.sample_strategy,
        use_face_detector=args.use_face_detector,
        seed=args.seed,
    )
    speaker_count = args.speakers if args.speakers > 0 else None
    config = PipelineConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        speakers=speaker_count,
        real_per_speaker=args.real_per_speaker,
        fake_per_speaker=args.fake_per_speaker,
        seed=args.seed,
        skip_existing=args.skip_existing,
        save_waveform=args.save_waveform,
        video=video_settings,
    )
    processor = FakeAVPreprocessor(config)
    processor.run()


if __name__ == "__main__":  # pragma: no cover
    main()
