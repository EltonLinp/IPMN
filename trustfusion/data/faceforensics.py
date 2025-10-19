"""
Dataset utilities for FaceForensics++ training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Optional
import csv

from torch.utils.data import Dataset

from ..utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass
class FaceForensicsSample:
    path: Path
    label: int
    method: str


class FaceForensicsDataset(Dataset):
    """
    Load original (real) and manipulated (fake) videos from FaceForensics++.
    """

    def __init__(
        self,
        root_dir: Path,
        original_dir: str,
        manipulated_dir: str,
        video_extensions: Tuple[str, ...],
        transforms: Sequence = (),
        manifest_path: Optional[Path] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.original_dir = original_dir
        self.manipulated_dir = manipulated_dir
        self.video_extensions = video_extensions
        self.transforms = transforms
        self.samples: List[FaceForensicsSample] = []
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self._collect_samples()

    def _collect_samples(self) -> None:
        if self.manifest_path and self.manifest_path.exists():
            self._load_from_manifest(self.manifest_path)
            return

        original_root = self.root_dir / self.original_dir
        manipulated_root = self.root_dir / self.manipulated_dir
        if not original_root.exists():
            LOGGER.warning("Original directory %s does not exist.", original_root)
        if not manipulated_root.exists():
            LOGGER.warning("Manipulated directory %s does not exist.", manipulated_root)

        self.samples.extend(self._scan_directory(original_root, label=0, method="original"))
        if manipulated_root.exists():
            for method_dir in manipulated_root.iterdir():
                if method_dir.is_dir():
                    method_name = method_dir.name
                    self.samples.extend(self._scan_directory(method_dir, label=1, method=method_name))

        LOGGER.info(
            "FaceForensics dataset loaded: total=%d, real=%d, fake=%d.",
            len(self.samples),
            sum(1 for sample in self.samples if sample.label == 0),
            sum(1 for sample in self.samples if sample.label == 1),
        )

    def _load_from_manifest(self, manifest: Path) -> None:
        rows: List[FaceForensicsSample] = []
        with manifest.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"video_path", "label", "method"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Manifest {manifest} 缺少必要字段 {required}")
            for row in reader:
                path = Path(row["video_path"])
                if not path.is_absolute():
                    path = (self.root_dir / path).resolve()
                if not path.exists():
                    LOGGER.warning("Skipped missing file listed in manifest: %s", path)
                    continue
                label = int(row.get("label", 0))
                method = row.get("method", "manifest")
                rows.append(FaceForensicsSample(path=path, label=label, method=method))
        self.samples = rows
        LOGGER.info(
            "Manifest loaded (%s): total=%d, real=%d, fake=%d.",
            manifest,
            len(self.samples),
            sum(1 for sample in self.samples if sample.label == 0),
            sum(1 for sample in self.samples if sample.label == 1),
        )

    def _scan_directory(self, directory: Path, label: int, method: str) -> List[FaceForensicsSample]:
        collected: List[FaceForensicsSample] = []
        if not directory.exists():
            return collected
        for file_path in directory.glob("**/*"):
            if file_path.is_file() and file_path.suffix.lower() in self.video_extensions:
                collected.append(
                    FaceForensicsSample(
                        path=file_path.resolve(),
                        label=label,
                        method=method,
                    )
                )
        return collected

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        data: Dict[str, object] = {
            "video_path": str(sample.path),
            "label": sample.label,
            "method": sample.method,
        }
        for transform in self.transforms:
            data = transform(data)
        return data
