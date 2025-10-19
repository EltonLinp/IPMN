"""
Preprocess Celeb-DF v2 videos into multimodal features (video crops + audio).

Output structure (per video):
data/preprocessed/train/<subject>/<video_stem>/
├── metadata.json
├── full_face/frame_00001.png
├── mouth_roi/frame_00001.png
├── audio/waveform.npy
└── audio/mel.npy
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import audioread
import cv2
import librosa
import numpy as np
from tqdm import tqdm
from mediapipe.python.solutions import face_detection

DATA_ROOT = Path(r"E:\CUHK\Industrial_Project\Celeb-DF-v2")
DEFAULT_MANIFEST = Path("manifests/train.csv")
DEFAULT_OUTPUT_ROOT = Path("data/preprocessed/train")

FRAME_RATE = 25
TARGET_FACE_SIZE = (224, 224)
TARGET_MOUTH_SIZE = (96, 96)
MEL_BINS = 80
SAMPLE_RATE = 16_000
FALLBACK_AUDIO_SECONDS = 0.5
WINDOW_SECONDS = 0.28


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def crop_face(frame: np.ndarray, detection: face_detection.FaceDetection) -> tuple[np.ndarray, np.ndarray]:
    image_height, image_width, _ = frame.shape
    bbox = detection.location_data.relative_bounding_box
    x1 = max(int(bbox.xmin * image_width), 0)
    y1 = max(int(bbox.ymin * image_height), 0)
    x2 = min(int((bbox.xmin + bbox.width) * image_width), image_width)
    y2 = min(int((bbox.ymin + bbox.height) * image_height), image_height)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        face = frame

    mouth_height = int(face.shape[0] * 0.4)
    mouth_width = int(face.shape[1] * 0.6)
    mouth_x1 = int(face.shape[1] * 0.2)
    mouth_y1 = int(face.shape[0] * 0.6)
    mouth = face[mouth_y1:mouth_y1 + mouth_height, mouth_x1:mouth_x1 + mouth_width]
    return face, mouth


def extract_audio_features(video_path: Path, out_dir: Path) -> Dict[str, object]:
    try:
        y, sr = librosa.load(str(video_path), sr=SAMPLE_RATE)
    except (FileNotFoundError, audioread.NoBackendError) as exc:
        print(f"[WARN] Audio backend unavailable for {video_path}: {exc}. Using silent fallback.")
        sr = SAMPLE_RATE
        y = np.zeros(int(sr * FALLBACK_AUDIO_SECONDS), dtype=np.float32)
    except Exception as exc:  # pragma: no cover - unexpected decode errors
        print(f"[WARN] Audio extraction failed for {video_path}: {exc}. Using silent fallback.")
        sr = SAMPLE_RATE
        y = np.zeros(int(sr * FALLBACK_AUDIO_SECONDS), dtype=np.float32)

    if y.size == 0:
        y = np.zeros(int(SAMPLE_RATE * FALLBACK_AUDIO_SECONDS), dtype=np.float32)
        sr = SAMPLE_RATE

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=int(0.025 * sr),
        hop_length=int(0.010 * sr),
        n_mels=MEL_BINS,
    )
    mel_db = librosa.power_to_db(mel).astype(np.float32)

    ensure_dir(out_dir)
    np.save(out_dir / "waveform.npy", y.astype(np.float32))
    np.save(out_dir / "mel.npy", mel_db)

    mel_hop = int(0.010 * SAMPLE_RATE)
    window_hops = int(WINDOW_SECONDS * SAMPLE_RATE / mel_hop)

    return {
        "waveform_len": int(len(y)),
        "sample_rate": int(sr),
        "mel_shape": tuple(int(v) for v in mel_db.shape),
        "window_hops": int(window_hops),
    }


def process_video(video_path: Path, subject: str, label: str, output_root: Path) -> None:
    out_dir = output_root / subject / video_path.stem
    if (out_dir / "metadata.json").exists():
        return

    ensure_dir(out_dir / "full_face")
    ensure_dir(out_dir / "mouth_roi")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video {video_path}")
        return

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 0:
        native_fps = FRAME_RATE
    # Derive a sampling schedule so the exported frames track the target frame rate.
    capture_all_frames = native_fps <= FRAME_RATE + 1e-6
    sampling_interval = native_fps / FRAME_RATE if not capture_all_frames else None
    next_capture_index = 0.0

    frame_idx = 0
    saved_frames = 0
    with face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5,
    ) as detector:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_idx += 1
            if not capture_all_frames:
                current_index = frame_idx - 1
                if current_index + 1e-6 < next_capture_index:
                    continue
                next_capture_index += sampling_interval

            saved_frames += 1
            frame_name = f"frame_{saved_frames:05d}.png"

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)
            if results.detections:
                face, mouth = crop_face(frame, results.detections[0])
            else:
                face = frame
                h, w, _ = face.shape
                mouth = face[int(h * 0.6):, int(w * 0.2):int(w * 0.8)]

            face_resized = cv2.resize(face, TARGET_FACE_SIZE)
            mouth_resized = cv2.resize(mouth, TARGET_MOUTH_SIZE)
            cv2.imwrite(str(out_dir / "full_face" / frame_name), face_resized)
            cv2.imwrite(str(out_dir / "mouth_roi" / frame_name), mouth_resized)

    cap.release()

    if saved_frames == 0:
        print(f"[WARN] No frames saved for {video_path}; check detector configuration.")
        return

    audio_meta = extract_audio_features(video_path, out_dir / "audio")

    metadata = {
        "subject": subject,
        "label": label,
        "frames_extracted": saved_frames,
        "frames_read": frame_idx,
        "input_fps": native_fps,
        "target_fps": FRAME_RATE,
        "audio": audio_meta,
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main(manifest_path: Path = DEFAULT_MANIFEST, output_root: Path = DEFAULT_OUTPUT_ROOT) -> None:
    ensure_dir(output_root)
    with manifest_path.open("r", encoding="utf-8") as f:
        entries = list(csv.DictReader(f))

    for row in tqdm(entries, desc="Processing videos", unit="video"):
        video_rel = Path(row["video_path"])
        subject = row["subject"]
        label = row["label"]
        video_path = DATA_ROOT / video_rel
        if not video_path.exists():
            print(f"[WARN] Missing video {video_path}")
            continue
        process_video(video_path, subject, label, output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Celeb-DF v2 videos into multimodal (video + audio) features.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="CSV manifest listing videos to process.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Destination directory for processed samples.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(manifest_path=args.manifest, output_root=args.output_root)
