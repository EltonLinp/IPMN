"""
Dataset helpers for loading video entries from configured directories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from torch.utils.data import Dataset

from ..utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class DatasetSample:
    path: Path
    label: int
    label_name: str


class DeepfakeVideoDataset(Dataset):
    """Flexible dataset wrapper supporting labelled subdirectories or flat folders."""

    def __init__(
        self,
        root_dir: Path,
        video_extensions: Tuple[str, ...],
        real_dir_name: Optional[Union[str, Sequence[str]]] = None,
        fake_dir_name: Optional[Union[str, Sequence[str]]] = None,
        metadata: Optional[Dict[str, int]] = None,
        transforms: Sequence = (),
    ) -> None:
        self.transforms = transforms
        self.video_extensions = video_extensions
        self.root_dir = Path(root_dir)
        if isinstance(real_dir_name, Sequence) and not isinstance(real_dir_name, (str, bytes)):
            self.real_dir_names: Tuple[str, ...] = tuple(real_dir_name)
        elif isinstance(real_dir_name, (str, bytes)):
            self.real_dir_names = (str(real_dir_name),)
        else:
            self.real_dir_names = tuple()

        if isinstance(fake_dir_name, Sequence) and not isinstance(fake_dir_name, (str, bytes)):
            self.fake_dir_names: Tuple[str, ...] = tuple(fake_dir_name)
        elif isinstance(fake_dir_name, (str, bytes)):
            self.fake_dir_names = (str(fake_dir_name),)
        else:
            self.fake_dir_names = tuple()

        self.metadata = metadata or {}
        self.samples: List[DatasetSample] = []
        self._scan_directories()

    def _scan_directories(self) -> None:
        if not self.root_dir.exists():
            LOGGER.warning("Dataset root %s does not exist; dataset is empty.", self.root_dir)
            return

        used_subdirs = False
        mappings: List[Tuple[str, int]] = []
        if self.real_dir_names:
            mappings.extend((dir_name, 0) for dir_name in self.real_dir_names)
        if self.fake_dir_names:
            mappings.extend((dir_name, 1) for dir_name in self.fake_dir_names)

        if mappings:
            for dir_name, label in mappings:
                dir_path = self.root_dir / dir_name
                if not dir_path.exists():
                    LOGGER.warning("Missing sub-directory %s at %s", dir_name, dir_path)
                    continue
                self._collect_from_directory(dir_path, label)
                used_subdirs = True
        if not used_subdirs:
            for file_path in self.root_dir.glob("**/*"):
                if file_path.is_file() and file_path.suffix.lower() in self.video_extensions:
                    stem_key = file_path.stem
                    label = self.metadata.get(stem_key, -1)
                    label_name = "unknown"
                    if label == 0:
                        label_name = "real"
                    elif label == 1:
                        label_name = "fake"
                    self.samples.append(
                        DatasetSample(
                            path=file_path.resolve(),
                            label=label,
                            label_name=label_name,
                        )
                    )
        if not self.samples:
            LOGGER.warning("No video files found under %s.", self.root_dir)
        else:
            LOGGER.info(
                "Dataset loaded: real=%d, fake=%d, unknown=%d, total=%d.",
                sum(1 for s in self.samples if s.label == 0),
                sum(1 for s in self.samples if s.label == 1),
                sum(1 for s in self.samples if s.label not in (0, 1)),
                len(self.samples),
            )

    def _collect_from_directory(self, directory: Path, label: int) -> None:
        for file_path in directory.glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in self.video_extensions:
                self.samples.append(
                    DatasetSample(
                        path=file_path.resolve(),
                        label=label,
                        label_name="real" if label == 0 else "fake",
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        data: Dict[str, object] = {
            "video_path": str(sample.path),
            "label": sample.label,
            "label_name": sample.label_name,
        }
        for transform in self.transforms:
            data = transform(data)
        return data

    def summary(self) -> Dict[str, int]:
        real_count = sum(1 for s in self.samples if s.label == 0)
        fake_count = sum(1 for s in self.samples if s.label == 1)
        unknown_count = sum(1 for s in self.samples if s.label not in (0, 1))
        return {
            "root_dir": str(self.root_dir),
            "total": len(self.samples),
            "real": real_count,
            "fake": fake_count,
            "unknown": unknown_count,
        }
