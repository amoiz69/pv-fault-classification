"""InfraredSolarModules dataset wrapper.

The canonical ``CLASS_TO_IDX`` mapping below is the single source of truth —
every downstream stage (splits, estimator, classifier, evaluation) must import
it from here (Pitfall 4: a class-index mismatch is a silent, catastrophic bug).

Construct from a split CSV produced by ``scripts/make_splits.py``::

    from src.dataset import InfraredSolarDataset
    ds = InfraredSolarDataset("data", "data/splits/fault_train.csv")
    img, label = ds[0]            # img: float tensor [1, 24, 40] in [0, 1]

Pass ``binary=True`` to relabel every sample as 0 (No-Anomaly) / 1 (any fault)
for the Stage-1 / Stage-3a binary head. Pass one or more CSV paths to pool
splits (e.g. the full-20k binary set = fault_test + noanomaly_test).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
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
        splits: str | Path | Iterable[str | Path],
        transform: Callable | None = None,
        binary: bool = False,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.binary = binary

        paths = [splits] if isinstance(splits, (str, Path)) else list(splits)
        frames = [pd.read_csv(p) for p in paths]
        self.table = pd.concat(frames, ignore_index=True)

        self.samples: list[tuple[Path, int]] = []
        for row in self.table.itertuples(index=False):
            label = int(row.class_idx)
            if self.binary:
                label = 0 if label == 0 else 1
            self.samples.append((self.root / row.filepath, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        # IR images are single-channel; force grayscale so a stray RGB JPEG
        # cannot silently change the channel count downstream.
        img = Image.open(path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = self._default_transform(img)
        return img, label

    @staticmethod
    def _default_transform(img: Image.Image) -> torch.Tensor:
        """PIL grayscale -> float tensor [1, H, W] scaled to [0, 1]."""
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W]
        return torch.from_numpy(arr).unsqueeze(0)

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights over the labels present, indexed by label.

        Use as ``alpha`` for the focal / class-weighted loss. For multiclass the
        length is ``max(label)+1``; classes absent from this split get weight 0.
        """
        labels = torch.tensor([lbl for _, lbl in self.samples])
        counts = torch.bincount(labels)
        weights = torch.zeros_like(counts, dtype=torch.float)
        nonzero = counts > 0
        weights[nonzero] = counts.sum().float() / (counts[nonzero].float() * int(nonzero.sum()))
        return weights
