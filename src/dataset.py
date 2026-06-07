"""InfraredSolarModules dataset wrapper.

Reads `module_metadata.json` and exposes (image, class_index) pairs.
The canonical CLASS_TO_IDX mapping below is the single source of truth — every
downstream stage (splits, estimator, classifier, evaluation) must import from here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
from torch.utils.data import Dataset


class InfraredSolarDataset(Dataset):
    CLASS_TO_IDX: dict[str, int] = {
        "No-Anomaly": 0,
        "Cell": 1,
        "Cell-Multi": 2,
        "Cracking": 3,
        "Hot-Spot": 4,
        "Hot-Spot-Multi": 5,
        "Shadowing": 6,
        "Diode": 7,
        "Diode-Multi": 8,
        "Vegetation": 9,
        "Soiling": 10,
        "Offline-Module": 11,
    }
    IDX_TO_CLASS: dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

    ENVIRONMENTAL: tuple[str, ...] = ("Soiling", "Shadowing", "Vegetation")
    ELECTRICAL: tuple[str, ...] = (
        "Cell",
        "Cell-Multi",
        "Hot-Spot",
        "Hot-Spot-Multi",
        "Diode",
        "Diode-Multi",
        "Cracking",
        "Offline-Module",
    )

    def __init__(
        self,
        root: str | Path,
        class_filter: Iterable[str] | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform

        meta_path = self.root / "module_metadata.json"
        with meta_path.open() as f:
            raw = json.load(f)

        allowed = set(class_filter) if class_filter is not None else None
        self.samples: list[tuple[Path, int]] = []
        for entry in raw.values():
            cls = entry["anomaly_class"]
            if allowed is not None and cls not in allowed:
                continue
            self.samples.append(
                (self.root / entry["image_filepath"], self.CLASS_TO_IDX[cls])
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
