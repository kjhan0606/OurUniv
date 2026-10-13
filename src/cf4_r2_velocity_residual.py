"""Conservative TSC operators for scale-matched velocity-residual diagnostics."""

from __future__ import annotations

from itertools import product

import numpy as np


def tsc_stencil(positions, box_size, grid_size):
    """Cell-centred periodic TSC indices/weights for an ``(n, 3)`` array."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or not np.isfinite(pos).all():
        raise ValueError("positions must be a finite (n,3) array")
    if not np.isfinite(box_size) or box_size <= 0 or grid_size < 3:
        raise ValueError("invalid periodic grid geometry")
    q = np.mod(pos, box_size) / (box_size / grid_size) - 0.5
    nearest = np.floor(q + 0.5).astype(np.int64)
    delta = q - nearest
    indices = (nearest[..., None] + np.array([-1, 0, 1])) % grid_size
    weights = np.stack((0.5 * (0.5 - delta) ** 2,
                        0.75 - delta ** 2,
                        0.5 * (0.5 + delta) ** 2), axis=-1)
    return indices, weights


def tsc_interpolate_periodic(vector_grid, positions, box_size):
    """TSC-gather a cell-centred periodic 3-vector grid at positions."""
    grid = np.asarray(vector_grid, dtype=np.float64)
    if grid.ndim != 4 or grid.shape[0] != 3 or len(set(grid.shape[1:])) != 1:
        raise ValueError("vector_grid must have shape (3,n,n,n)")
    if not np.isfinite(grid).all():
        raise ValueError("vector_grid contains nonfinite values")
    n = grid.shape[1]
    idx, weights = tsc_stencil(positions, box_size, n)
    gathered = np.zeros((len(idx), 3), dtype=np.float64)
    for a, b, c in product(range(3), repeat=3):
        w = weights[:, 0, a] * weights[:, 1, b] * weights[:, 2, c]
        values = grid[:, idx[:, 0, a], idx[:, 1, b], idx[:, 2, c]].T
        gathered += values * w[:, None]
    return gathered


def tsc_deposit_2x_moments(moments):
    """TSC-deposit cell-centred 50^3 moments onto aligned 25^3 centres.

    Input channels are mass, three momentum components and three diagonal
    second moments. The fine cell centres lie at +/- one quarter of a target
    cell around each aligned target centre. This periodic operator conserves
    all channel sums; it is neither a force solve nor a particle snapshot.
    """
    data = np.asarray(moments, dtype=np.float64)
    if data.shape != (7, 50, 50, 50) or not np.isfinite(data).all():
        raise ValueError("moments must be finite with shape (7,50,50,50)")
    if np.any(data[0] < 0):
        raise ValueError("mass moment cannot be negative")
    blocks = data.reshape(7, 25, 2, 25, 2, 25, 2)
    result = np.zeros((7, 25, 25, 25), dtype=np.float64)
    offsets = (-1, 0, 1)
    by_parity = {0: (0.28125, 0.6875, 0.03125),
                 1: (0.03125, 0.6875, 0.28125)}
    for px, py, pz in product(range(2), repeat=3):
        block = blocks[:, :, px, :, py, :, pz]
        for a, ox in enumerate(offsets):
            for b, oy in enumerate(offsets):
                for c, oz in enumerate(offsets):
                    weight = by_parity[px][a] * by_parity[py][b] * by_parity[pz][c]
                    if weight:
                        result += weight * np.roll(
                            block, shift=(ox, oy, oz), axis=(1, 2, 3))
    return result


def residual_summary(residual):
    """Summarize equal-weight object residuals without retaining identities."""
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError("residuals must be a nonempty (n,3) array")
    if not np.isfinite(values).all():
        raise ValueError("residuals contain nonfinite values")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / len(values)
    component_sigma = np.sqrt(np.diag(covariance))
    one_dimensional_sigma = float(np.sqrt(np.trace(covariance) / 3.0))
    absolute_components = np.abs(centered)
    speed = np.linalg.norm(centered, axis=1)
    return {
        "objects": int(len(values)),
        "mean_km_s": mean.tolist(),
        "covariance_km2_s2": covariance.tolist(),
        "component_sigma_km_s": component_sigma.tolist(),
        "isotropic_1d_sigma_km_s": one_dimensional_sigma,
        "abs_component_quantiles_km_s": np.quantile(
            absolute_components, [0.5, 0.68, 0.9, 0.95, 0.99], axis=0).T.tolist(),
        "speed_quantiles_km_s": np.quantile(
            speed, [0.5, 0.68, 0.9, 0.95, 0.99]).tolist(),
        "fraction_abs_component_gt_100_km_s": float(np.mean(absolute_components > 100.0)),
        "fraction_abs_component_gt_200_km_s": float(np.mean(absolute_components > 200.0)),
    }
