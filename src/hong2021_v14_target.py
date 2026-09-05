"""Exact common-CIC log-density target used by all V14 simulation suites."""
from __future__ import annotations

import numpy as np


def log_cic_target(
    density: np.ndarray,
    mean_density: float,
    density_scale: float = 4.5,
) -> np.ndarray:
    """Map positive CIC counts to log overdensity without clipping or floors."""
    value = np.asarray(density)
    if (
        value.size == 0
        or not np.isfinite(value).all()
        or np.any(value <= 0)
        or not np.isfinite(mean_density)
        or mean_density <= 0
        or not np.isfinite(density_scale)
        or density_scale <= 0
    ):
        raise ValueError("CIC density, mean, and scale must be finite and positive")
    target = np.log10(value / mean_density) / density_scale
    if not np.isfinite(target).all():
        raise RuntimeError("finite positive CIC density produced a non-finite target")
    return target.astype(np.float32, copy=False)
