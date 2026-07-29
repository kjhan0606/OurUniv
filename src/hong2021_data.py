#!/usr/bin/env python
"""Data definitions and validation for the Hong et al. (2021) reproduction.

The training files are HDF5 containers with unaugmented, non-overlapping
sub-cubes:

  input   (sample, 2, N, N, N), float32
  target  (sample, 1, N, N, N), float32

Input channel 0 is the integer number of target galaxies per voxel.  Channel 1
is the mean radial peculiar velocity in km/s (zero in empty voxels).  Target is
``log10(rho_dm/rho_dm_mean)/density_scale`` and must lie in [-1, 1].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REQUIRED_ATTRIBUTES = {
    "paper": "Hong et al. 2021 ApJ 913 76",
    "voxel_mpc_h": 0.3125,
    "channels": "Ngal,mean_radial_vpec_kms",
    "galactic_mask_abs_b_deg": 10.0,
}


def grid_galaxy_observables(
    relative_position_mpc_h: np.ndarray,
    relative_velocity_kms: np.ndarray,
    grid: int,
    box_mpc_h: float,
    mask_abs_b_deg: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Make the two published input grids using nearest-voxel assignment."""
    pos = np.asarray(relative_position_mpc_h, dtype=np.float64)
    vel = np.asarray(relative_velocity_kms, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or vel.shape != pos.shape:
        raise ValueError("position and velocity must both have shape (galaxy,3)")
    half = box_mpc_h / 2.0
    inside = np.all((pos >= -half) & (pos < half), axis=1)
    radius = np.linalg.norm(pos, axis=1)
    sin_b = np.divide(pos[:, 2], radius, out=np.zeros_like(radius), where=radius > 0)
    latitude = np.degrees(np.arcsin(np.clip(sin_b, -1.0, 1.0)))
    keep = inside & (radius > 0) & (np.abs(latitude) >= mask_abs_b_deg)
    pos = pos[keep]
    vel = vel[keep]

    cell = np.floor((pos + half) * (grid / box_mpc_h)).astype(np.int64)
    cell = np.clip(cell, 0, grid - 1)
    flat = np.ravel_multi_index(cell.T, (grid, grid, grid))
    radial_hat = pos / np.linalg.norm(pos, axis=1)[:, None]
    radial_v = np.einsum("ij,ij->i", vel, radial_hat)
    counts = np.bincount(flat, minlength=grid**3).astype(np.float32)
    velocity_sum = np.bincount(
        flat, weights=radial_v, minlength=grid**3
    ).astype(np.float64)
    mean_velocity = np.divide(
        velocity_sum,
        counts,
        out=np.zeros_like(velocity_sum),
        where=counts > 0,
    ).astype(np.float32)
    return counts.reshape((grid,) * 3), mean_velocity.reshape((grid,) * 3)


def grid_uncertainty_aware_observables(
    relative_position_mpc_h: np.ndarray,
    relative_velocity_kms: np.ndarray,
    radial_velocity_error_kms: np.ndarray,
    grid: int,
    box_mpc_h: float,
    mask_abs_b_deg: float = 10.0,
    sigma_nonlinear_kms: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid count, precision-weighted mean velocity, its error, and scatter.

    ``sigma_mean`` is the propagated observational uncertainty
    ``1/sqrt(sum(1/sigma_i^2))``.  ``sample_scatter`` is the unbiased weighted
    within-cell velocity dispersion and is zero for cells with fewer than two
    galaxies.  At the paper voxel size the latter is expected to be sparse, so
    it is kept as an optional fourth channel rather than conflated with the
    measurement uncertainty.
    """
    pos = np.asarray(relative_position_mpc_h, dtype=np.float64)
    vel = np.asarray(relative_velocity_kms, dtype=np.float64)
    error = np.asarray(radial_velocity_error_kms, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or vel.shape != pos.shape:
        raise ValueError("position and velocity must both have shape (galaxy,3)")
    if error.shape != (pos.shape[0],):
        raise ValueError("radial_velocity_error_kms must have shape (galaxy,)")
    if np.any(~np.isfinite(error)) or np.any(error <= 0):
        raise ValueError("radial velocity errors must be finite and positive")

    half = box_mpc_h / 2.0
    inside = np.all((pos >= -half) & (pos < half), axis=1)
    radius = np.linalg.norm(pos, axis=1)
    sin_b = np.divide(pos[:, 2], radius, out=np.zeros_like(radius), where=radius > 0)
    latitude = np.degrees(np.arcsin(np.clip(sin_b, -1.0, 1.0)))
    keep = inside & (radius > 0) & (np.abs(latitude) >= mask_abs_b_deg)
    pos, vel, error = pos[keep], vel[keep], error[keep]

    cell = np.floor((pos + half) * (grid / box_mpc_h)).astype(np.int64)
    cell = np.clip(cell, 0, grid - 1)
    flat = np.ravel_multi_index(cell.T, (grid, grid, grid))
    radial_hat = pos / np.linalg.norm(pos, axis=1)[:, None]
    radial_v = np.einsum("ij,ij->i", vel, radial_hat)
    variance = error**2 + float(sigma_nonlinear_kms) ** 2
    weight = 1.0 / variance
    size = grid**3
    counts = np.bincount(flat, minlength=size).astype(np.float64)
    sum_w = np.bincount(flat, weights=weight, minlength=size)
    sum_w2 = np.bincount(flat, weights=weight**2, minlength=size)
    sum_wv = np.bincount(flat, weights=weight * radial_v, minlength=size)
    sum_wv2 = np.bincount(flat, weights=weight * radial_v**2, minlength=size)
    mean = np.divide(sum_wv, sum_w, out=np.zeros(size), where=sum_w > 0)
    sigma_mean = np.sqrt(
        np.divide(1.0, sum_w, out=np.zeros(size), where=sum_w > 0)
    )
    scatter_numerator = sum_wv2 - 2.0 * mean * sum_wv + mean**2 * sum_w
    scatter_denominator = sum_w - np.divide(
        sum_w2, sum_w, out=np.zeros(size), where=sum_w > 0
    )
    scatter_variance = np.divide(
        np.maximum(scatter_numerator, 0.0),
        scatter_denominator,
        out=np.zeros(size),
        where=scatter_denominator > 0,
    )
    shape = (grid,) * 3
    return tuple(
        field.astype(np.float32).reshape(shape)
        for field in (counts, mean, sigma_mean, np.sqrt(scatter_variance))
    )


def cyclic_flip_transform(
    array: np.ndarray, permutation_index: int, flip_bits: int
) -> np.ndarray:
    """One of the paper's 3 cyclic axis permutations x 8 axis flips."""
    if permutation_index not in (0, 1, 2) or not 0 <= flip_bits < 8:
        raise ValueError("permutation_index=0..2 and flip_bits=0..7 required")
    spatial = array.ndim - 3
    axes = tuple(range(spatial)) + tuple(
        spatial + v
        for v in ((0, 1, 2), (1, 2, 0), (2, 0, 1))[permutation_index]
    )
    out = np.transpose(array, axes)
    for dim in range(3):
        if flip_bits & (1 << dim):
            out = np.flip(out, axis=spatial + dim)
    return np.ascontiguousarray(out)


def inspect_training_file(
    path: str | Path,
    expected_channels: int = 2,
    expected_channel_label: str = REQUIRED_ATTRIBUTES["channels"],
) -> dict[str, Any]:
    """Validate a prepared file and return a JSON-serializable report."""
    path = Path(path)
    report: dict[str, Any] = {"path": str(path), "pass": True, "failures": []}
    if not path.is_file():
        return {"path": str(path), "pass": False, "failures": ["file_missing"]}
    with h5py.File(path, "r") as handle:
        missing = [name for name in ("input", "target") if name not in handle]
        if missing:
            return {
                "path": str(path),
                "pass": False,
                "failures": [f"missing_dataset:{v}" for v in missing],
            }
        x, y = handle["input"], handle["target"]
        report.update(input_shape=list(x.shape), target_shape=list(y.shape))
        expected_y = (x.shape[0], 1, *x.shape[-3:])
        if x.ndim != 5 or x.shape[1] != expected_channels:
            report["failures"].append(
                f"input_shape_not_N{expected_channels}SSS"
            )
        if tuple(y.shape) != expected_y:
            report["failures"].append("target_shape_mismatch")
        if x.shape[-1] not in (64, 128) or len(set(x.shape[-3:])) != 1:
            report["failures"].append("grid_not_published_64_or_128")
        required_attributes = {
            **REQUIRED_ATTRIBUTES,
            "channels": expected_channel_label,
        }
        for key, value in required_attributes.items():
            got = handle.attrs.get(key)
            if isinstance(got, bytes):
                got = got.decode()
            if isinstance(value, float):
                valid = got is not None and np.isclose(float(got), value)
            else:
                valid = got == value
            if not valid:
                report["failures"].append(f"attribute:{key}")
        if y.size:
            sample = y[: min(4, y.shape[0])]
            report["target_sample_minmax"] = [float(sample.min()), float(sample.max())]
            if not np.isfinite(sample).all():
                report["failures"].append("nonfinite_target")
            if sample.min() < -1.0001 or sample.max() > 1.0001:
                report["failures"].append("target_outside_tanh_range")
        report["n_unaugmented"] = int(x.shape[0])
        report["n_augmented_3x8"] = int(x.shape[0] * 24)
    report["pass"] = not report["failures"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    reports = [inspect_training_file(path) for path in args.files]
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for report in reports:
            print(
                f"{'PASS' if report['pass'] else 'FAIL'} {report['path']} "
                f"{'; '.join(report['failures'])}"
            )
    if not all(report["pass"] for report in reports):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
