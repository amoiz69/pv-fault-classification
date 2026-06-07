"""Stage-1 baseline temperature estimator.

A small CNN that maps a 40x24 thermal crop to a single scalar in [0, 1] — the
predicted *healthy* baseline level (mean intensity) for that image's operating
condition. Trained on No-Anomaly samples ONLY (Pitfall 2). Its output is what
the Stage-2 residual subtracts to remove the irradiance confound.

Why a small custom CNN rather than MobileNetV3-Small: the input is only 40x24x1.
Pretrained ImageNet backbones expect ~224x224x3 and bring millions of params we
would have to retrain from a domain mismatch; this ~30k-param net is faster,
fully interpretable, and trains in minutes on a T4.
"""
from __future__ import annotations

import torch
from torch import nn


class BaselineEstimator(nn.Module):
    def __init__(self, in_ch: int = 1, width: int = 16) -> None:
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                    # 40x24 -> 20x12
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                    # 20x12 -> 10x6
            nn.Conv2d(2 * w, 4 * w, 3, padding=1), nn.BatchNorm2d(4 * w), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                            # -> [B, 4w, 1, 1]
        )
        self.head = nn.Linear(4 * w, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x).flatten(1)        # [B, 4w]
        # sigmoid bounds the prediction to [0, 1], matching the target range
        # (images are normalized to [0, 1]); avoids the net wandering off-scale.
        return torch.sigmoid(self.head(z)).squeeze(1)   # [B]

    @torch.no_grad()
    def predict_baseline(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper: scalar baseline per image, no gradient."""
        self.eval()
        return self.forward(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = BaselineEstimator()
    x = torch.rand(4, 1, 40, 24)
    y = m(x)
    print(f"params: {count_params(m):,}")
    print(f"input {tuple(x.shape)} -> output {tuple(y.shape)} range [{y.min():.3f}, {y.max():.3f}]")
