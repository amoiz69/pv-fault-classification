"""Train the Stage-1 baseline temperature estimator.

Runs on CUDA (Colab T4), Apple MPS, or CPU — auto-detected. Trains on
No-Anomaly samples ONLY (Pitfall 2); the regression target is each image's own
mean intensity (a proxy for its irradiance-driven baseline level).

    python scripts/train_estimator.py --config configs/estimator.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dataset import InfraredSolarDataset  # noqa: E402
from src.estimator import BaselineEstimator, count_params  # noqa: E402


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def baseline_target(images: torch.Tensor) -> torch.Tensor:
    """Regression target = mean intensity per image, shape [B]."""
    return images.mean(dim=(1, 2, 3))


def run_epoch(model, loader, loss_fn, device, optimizer=None) -> float:
    """Returns mean absolute error over the loader. Trains if optimizer given."""
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    for images, _ in loader:                       # labels unused — regression
        images = images.to(device)
        target = baseline_target(images)
        with torch.set_grad_enabled(train):
            pred = model(images)
            loss = loss_fn(pred, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/estimator.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    torch.manual_seed(cfg["seed"])
    device = pick_device()
    print(f"device: {device}")

    root = cfg["data"]["root"]
    train_ds = InfraredSolarDataset(root, cfg["data"]["train_split"])
    val_ds = InfraredSolarDataset(root, cfg["data"]["val_split"])
    print(f"train: {len(train_ds)} | val: {len(val_ds)} (No-Anomaly only)")

    t = cfg["train"]
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True,
                          num_workers=t["num_workers"])
    val_dl = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False,
                        num_workers=t["num_workers"])

    model = BaselineEstimator(width=cfg["model"]["width"]).to(device)
    print(f"params: {count_params(model):,}")
    loss_fn = nn.L1Loss()                          # MAE
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(1, t["epochs"] + 1):
        tr = run_epoch(model, train_dl, loss_fn, device, opt)
        va = run_epoch(model, val_dl, loss_fn, device)
        flag = ""
        if va < best_val:
            best_val, best_state, patience = va, {k: v.cpu() for k, v in model.state_dict().items()}, 0
            flag = "  <- best"
        else:
            patience += 1
        print(f"epoch {epoch:2d}  train MAE {tr:.4f}  val MAE {va:.4f}{flag}")
        if patience >= t["early_stop_patience"]:
            print(f"early stop at epoch {epoch} (no val improvement for {patience})")
            break

    out = ROOT / cfg["output"]["checkpoint"]
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "width": cfg["model"]["width"],
                "val_mae": best_val, "config": cfg}, out)
    print(f"\nbest val MAE: {best_val:.4f}")
    print(f"saved checkpoint -> {out}")


if __name__ == "__main__":
    main()
