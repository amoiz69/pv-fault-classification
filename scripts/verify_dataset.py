"""Sanity check: confirm dataset on disk matches the spec's class distribution.

Run after downloading the dataset to catch corrupted or partial downloads early.
Counts come straight from pv_fault_classification_context.md.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import InfraredSolarDataset  # noqa: E402

EXPECTED_COUNTS: dict[str, int] = {
    "No-Anomaly": 10000,
    "Cell": 1877,
    "Vegetation": 1639,
    "Diode": 1499,
    "Cell-Multi": 1288,
    "Shadowing": 1056,
    "Cracking": 940,
    "Offline-Module": 827,
    "Hot-Spot": 249,
    "Hot-Spot-Multi": 246,
    "Soiling": 204,
    "Diode-Multi": 175,
}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())


def main() -> int:
    ds = InfraredSolarDataset(ROOT / "data")
    counts = Counter(InfraredSolarDataset.IDX_TO_CLASS[lbl] for _, lbl in ds.samples)

    print(f"Total samples loaded: {len(ds)} (expected {EXPECTED_TOTAL})")
    print()
    print(f"  {'class':<18s} {'actual':>7s}  {'expected':>9s}")
    print(f"  {'-' * 18} {'-' * 7}  {'-' * 9}")

    all_ok = len(ds) == EXPECTED_TOTAL
    for cls, expected in EXPECTED_COUNTS.items():
        actual = counts.get(cls, 0)
        match = actual == expected
        all_ok = all_ok and match
        mark = "OK" if match else "FAIL"
        print(f"  {cls:<18s} {actual:>7d}  {expected:>9d}  [{mark}]")

    # also check the first image actually opens and has the expected shape
    img, _ = ds[0]
    print()
    print(f"Sample image 0: mode={img.mode}, size={img.size} (expected 24x40 grayscale)")
    if img.size != (24, 40):
        print(f"  WARNING: image size {img.size} != expected (24, 40)")
        all_ok = False

    if not all_ok:
        print("\nFAIL: counts or image shape do not match spec. Re-download dataset.")
        return 1
    print("\nPASS: dataset matches spec.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
