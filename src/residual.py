"""Stage-2 residual map generator (deterministic, no learning).

Given the trained Stage-1 estimator, the residual is::

    residual = image - baseline_scalar

where ``baseline_scalar`` is the estimator's prediction broadcast across every
pixel. This removes the irradiance/ambient baseline so what remains is the
fault-driven temperature delta (ΔT). Healthy panels -> residual ~ 0 everywhere;
faults -> localized non-zero regions, regardless of time-of-day irradiance.

No gradients flow through this step during classifier training (Stage 3): it is
a fixed preprocessing transform, exactly as the IEC 62446-3 normalization is a
fixed standard, not a learned one.
"""
from __future__ import annotations

import torch

from .estimator import BaselineEstimator


@torch.no_grad()
def generate_residual(images: torch.Tensor, estimator: BaselineEstimator) -> torch.Tensor:
    """Compute ΔT residual maps for a batch.

    Args:
        images: [B, 1, H, W] float tensor in [0, 1].
        estimator: trained BaselineEstimator (trained on No-Anomaly only).
    Returns:
        residual: [B, 1, H, W], signed (can be negative). Same shape as input.
    """
    estimator.eval()
    baseline = estimator(images)                     # [B]
    baseline_map = baseline.view(-1, 1, 1, 1)        # broadcast to [B,1,H,W]
    return images - baseline_map


def residual_stats(residual: torch.Tensor) -> dict[str, float]:
    """Quick diagnostics used to validate the estimator (healthy -> ~0)."""
    return {
        "mean": float(residual.mean()),
        "abs_mean": float(residual.abs().mean()),
        "std": float(residual.std()),
        "max": float(residual.max()),
        "min": float(residual.min()),
    }
