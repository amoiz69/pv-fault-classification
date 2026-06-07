"""Generate deterministic, stratified train/val/test splits.

Two non-overlapping split families are written to ``data/splits/``:

* ``fault_{train,val,test}.csv`` — the 10,000 fault images (11 classes), split
  70/15/15 stratified by class. This is the multiclass benchmark split that
  mirrors the 2025 InfraredSolarModules paper's 7000/1500/1500 protocol.
* ``noanomaly_{train,val,test}.csv`` — the 10,000 No-Anomaly images, split with
  the same 70/15/15 ratios. ``noanomaly_train`` is the ONLY data the Stage-1
  baseline estimator may see (Pitfall 2: any fault leakage teaches it to
  suppress fault signatures in the residual).

Because the two families are disjoint, the full-20k binary split is simply the
union of the corresponding fault + No-Anomaly CSVs — leak-free by construction.

Run once; the CSVs are committed to git so every downstream stage shares the
exact same split. Re-running with the same seed reproduces it bit-for-bit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Single source of truth for the class mapping.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import InfraredSolarDataset  # noqa: E402

SEED = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15  # of the whole, taken from the post-test remainder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLITS = DATA / "splits"


def load_metadata() -> pd.DataFrame:
    raw = pd.read_json(DATA / "module_metadata.json", orient="index")
    raw.index.name = "image_id"
    df = raw.reset_index().rename(columns={"image_filepath": "filepath"})
    df["class_idx"] = df["anomaly_class"].map(InfraredSolarDataset.CLASS_TO_IDX)
    if df["class_idx"].isna().any():
        unknown = sorted(df.loc[df["class_idx"].isna(), "anomaly_class"].unique())
        raise ValueError(f"Classes not in CLASS_TO_IDX: {unknown}")
    return df[["image_id", "filepath", "anomaly_class", "class_idx"]]


def split_70_15_15(df: pd.DataFrame, stratify: bool) -> dict[str, pd.DataFrame]:
    strat = df["anomaly_class"] if stratify else None
    train_val, test = train_test_split(
        df, test_size=TEST_FRAC, random_state=SEED, stratify=strat
    )
    strat_tv = train_val["anomaly_class"] if stratify else None
    # val fraction relative to the train_val remainder (0.15 / 0.85)
    train, val = train_test_split(
        train_val,
        test_size=VAL_FRAC / (1.0 - TEST_FRAC),
        random_state=SEED,
        stratify=strat_tv,
    )
    return {"train": train, "val": val, "test": test}


def write(prefix: str, parts: dict[str, pd.DataFrame]) -> None:
    for name, part in parts.items():
        out = SPLITS / f"{prefix}_{name}.csv"
        part.sort_values("image_id").to_csv(out, index=False)
        print(f"  wrote {out.relative_to(ROOT)}  ({len(part)} rows)")


def report(prefix: str, parts: dict[str, pd.DataFrame]) -> None:
    print(f"\n[{prefix}] per-class counts (train / val / test):")
    classes = sorted(
        pd.concat(parts.values())["anomaly_class"].unique(),
        key=lambda c: InfraredSolarDataset.CLASS_TO_IDX[c],
    )
    for cls in classes:
        counts = [int((parts[s]["anomaly_class"] == cls).sum()) for s in ("train", "val", "test")]
        print(f"  {cls:16s} {counts[0]:>5d} / {counts[1]:>4d} / {counts[2]:>4d}")


def main() -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)
    df = load_metadata()
    print(f"Loaded {len(df)} records.")

    fault = df[df["anomaly_class"] != "No-Anomaly"].copy()
    healthy = df[df["anomaly_class"] == "No-Anomaly"].copy()
    print(f"  fault images: {len(fault)} ({fault['anomaly_class'].nunique()} classes)")
    print(f"  No-Anomaly images: {len(healthy)}")

    fault_parts = split_70_15_15(fault, stratify=True)
    healthy_parts = split_70_15_15(healthy, stratify=False)

    write("fault", fault_parts)
    write("noanomaly", healthy_parts)
    report("fault", fault_parts)
    report("noanomaly", healthy_parts)

    # Sanity: no image_id appears in more than one split, anywhere.
    all_parts = list(fault_parts.values()) + list(healthy_parts.values())
    all_ids = pd.concat(all_parts)["image_id"]
    assert all_ids.is_unique, "LEAK: duplicate image_id across splits"
    assert len(all_ids) == len(df), "lost records during splitting"
    print("\nLeak check passed: every image assigned to exactly one split.")


if __name__ == "__main__":
    main()
