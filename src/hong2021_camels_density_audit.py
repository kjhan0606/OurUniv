#!/usr/bin/env python
"""Audit raw CAMELS particle assignment against the legacy CMD target.

Hong/TNG targets are nearest-grid-point (NGP) dark-matter particle counts in
0.3125 Mpc/h cells.  The legacy SIMBA preparation instead consumes CAMELS
Multifield Dataset grids, whose published construction uses an adaptive
32-neighbour particle kernel.  This module measures that operator mismatch on
development data before V14 freezes a common target definition.

The particle depositors are deliberately small, dependency-free reference
implementations.  They stream a snapshot and conserve particle number exactly.
No independent-gate data are needed by this audit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from hong2021_prepare_simba import conservative_resample_cube


BOX_MPC_H = 25.0
GRID = 80
CELL_MPC_H = BOX_MPC_H / GRID
EXPECTED_DM_PARTICLES = 256**3


def _validate_coordinates(coordinates: np.ndarray, box_mpc_h: float) -> np.ndarray:
    value = np.asarray(coordinates, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"expected coordinates with shape (N,3), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("coordinates contain non-finite values")
    # A coordinate exactly at the periodic upper boundary is equivalent to 0.
    # Values farther outside the box indicate a unit or payload error.
    tolerance = 32.0 * np.finfo(np.float32).eps * box_mpc_h
    if np.any(value < -tolerance) or np.any(value > box_mpc_h + tolerance):
        raise ValueError("coordinates lie outside the periodic box")
    return np.mod(value, box_mpc_h)


def deposit_particle_counts(
    coordinates: np.ndarray,
    *,
    grid: int = GRID,
    box_mpc_h: float = BOX_MPC_H,
    assignment: str = "ngp",
) -> np.ndarray:
    """Deposit equal-mass particles on a periodic cell-centred grid.

    ``ngp`` matches the existing TNG/Hong floor-to-cell target.  ``cic`` uses
    the two nearest cell centres per axis.  Both return float64 so chunk-wise
    accumulation is deterministic to integer precision at CAMELS particle
    counts.
    """
    if grid <= 0 or box_mpc_h <= 0:
        raise ValueError("grid and box size must be positive")
    value = _validate_coordinates(coordinates, box_mpc_h)
    scaled = value * (grid / box_mpc_h)
    shape = (grid, grid, grid)

    if assignment == "ngp":
        cell = np.floor(scaled).astype(np.int64) % grid
        flat = (cell[:, 0] * grid + cell[:, 1]) * grid + cell[:, 2]
        return np.bincount(flat, minlength=grid**3).reshape(shape).astype(np.float64)
    if assignment != "cic":
        raise ValueError(f"unsupported assignment {assignment!r}")

    # Cell centres are at i+1/2.  Interpolate between the two nearest centres.
    lower_coordinate = scaled - 0.5
    lower = np.floor(lower_coordinate).astype(np.int64)
    fraction = lower_coordinate - lower
    result = np.zeros(grid**3, dtype=np.float64)
    for x_offset in (0, 1):
        x_index = (lower[:, 0] + x_offset) % grid
        x_weight = fraction[:, 0] if x_offset else 1.0 - fraction[:, 0]
        for y_offset in (0, 1):
            y_index = (lower[:, 1] + y_offset) % grid
            y_weight = fraction[:, 1] if y_offset else 1.0 - fraction[:, 1]
            for z_offset in (0, 1):
                z_index = (lower[:, 2] + z_offset) % grid
                z_weight = fraction[:, 2] if z_offset else 1.0 - fraction[:, 2]
                flat = (x_index * grid + y_index) * grid + z_index
                result += np.bincount(
                    flat,
                    weights=x_weight * y_weight * z_weight,
                    minlength=grid**3,
                )
    return result.reshape(shape)


def smooth_ngp_as_expected_cic(counts: np.ndarray) -> np.ndarray:
    """Apply the expected CIC kernel to periodic NGP cell counts.

    For particles uniformly distributed inside an NGP cell, their mean CIC
    contribution along one axis is ``[1/8, 3/4, 1/8]``.  The separable 3-D
    operator provides an identical, mass-conserving target definition when
    only an existing NGP cache is available (notably the 1.8-TB TNG snapshot).
    This is a fixed assignment operator, not a fitted Fourier correction.
    """
    value = np.asarray(counts, dtype=np.float64)
    if value.ndim != 3 or len(set(value.shape)) != 1:
        raise ValueError(f"expected a cubic 3-D count field, got {value.shape}")
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError("counts must be finite and non-negative")
    result = value
    for axis in range(3):
        result = (
            0.125 * np.roll(result, 1, axis=axis)
            + 0.75 * result
            + 0.125 * np.roll(result, -1, axis=axis)
        )
    if not np.isclose(result.sum(), value.sum(), rtol=0.0, atol=1.0e-8):
        raise RuntimeError("periodic expected-CIC smoothing changed total mass")
    return result


def stream_snapshot_counts(
    snapshot: Path,
    assignments: Iterable[str] = ("ngp", "cic"),
    block_particles: int = 1_000_000,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read one CAMELS z=0 snapshot and return particle-conserving grids."""
    requested = tuple(assignments)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("assignments must be a non-empty unique sequence")
    counts = {name: np.zeros((GRID,) * 3, dtype=np.float64) for name in requested}
    with h5py.File(snapshot, "r") as handle:
        header = handle["Header"].attrs
        box = float(header["BoxSize"]) / 1000.0
        redshift = float(header["Redshift"])
        coordinates = handle["PartType1/Coordinates"]
        if not np.isclose(box, BOX_MPC_H, rtol=0.0, atol=1.0e-8):
            raise ValueError(f"unexpected BoxSize {box} Mpc/h")
        if abs(redshift) > 1.0e-6:
            raise ValueError(f"snapshot is not z=0: z={redshift}")
        if coordinates.shape != (EXPECTED_DM_PARTICLES, 3):
            raise ValueError(f"unexpected DM coordinate shape {coordinates.shape}")
        for begin in range(0, len(coordinates), block_particles):
            end = min(begin + block_particles, len(coordinates))
            position = np.asarray(coordinates[begin:end], dtype=np.float64) / 1000.0
            for name in requested:
                counts[name] += deposit_particle_counts(position, assignment=name)

        audit = {
            "box_mpc_h": box,
            "redshift": redshift,
            "hubble_param": float(header["HubbleParam"]),
            "omega_m": float(header["Omega0"]),
            # Legacy SIMBA snapshots omit OmegaBaryon even though the public
            # CAMELS reader example exposes it for other suites.
            "omega_b": (
                float(header["OmegaBaryon"])
                if "OmegaBaryon" in header
                else None
            ),
            "dm_particles": len(coordinates),
            "coordinate_dtype": str(coordinates.dtype),
        }
    for name, field in counts.items():
        if not np.isclose(field.sum(dtype=np.float64), EXPECTED_DM_PARTICLES, rtol=0, atol=1e-6):
            raise RuntimeError(f"{name} deposit does not conserve particle number")
    return counts, audit


def field_summary(field: np.ndarray) -> dict[str, Any]:
    value = np.asarray(field, dtype=np.float64)
    mean = float(value.mean())
    positive = value[value > 0]
    normalized = value / mean
    return {
        "mean": mean,
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "zero_cells": int(np.count_nonzero(value == 0)),
        "zero_volume_fraction": float(np.mean(value == 0)),
        "minimum_positive": float(positive.min()) if len(positive) else None,
        "normalized_percentiles": {
            str(percentile): float(np.percentile(normalized, percentile))
            for percentile in (0, 0.1, 1, 5, 50, 95, 99, 99.9, 100)
        },
        "normalized_standard_deviation": float(normalized.std()),
    }


def log_density(field: np.ndarray, zero_floor_count: float | None = None) -> np.ndarray:
    """Return log10 density contrast, optionally censoring zero-count cells."""
    value = np.asarray(field, dtype=np.float64)
    if zero_floor_count is not None:
        if zero_floor_count <= 0:
            raise ValueError("zero floor must be positive")
        value = np.maximum(value, zero_floor_count)
    if np.any(value <= 0) or not np.isfinite(value).all():
        raise ValueError("log-density field must be finite and positive")
    return np.log10(value / value.mean(dtype=np.float64))


def isotropic_power_bands(
    field: np.ndarray,
    *,
    box_mpc_h: float = BOX_MPC_H,
    edges: tuple[float, ...] = (0.3, 1.0, 3.0, 6.0, np.pi / CELL_MPC_H),
) -> dict[str, float]:
    """Mean discrete power in the four field-gate Fourier bands."""
    value = np.asarray(field, dtype=np.float64)
    if value.shape != (GRID,) * 3:
        raise ValueError(f"expected {(GRID,) * 3}, got {value.shape}")
    centered = value - value.mean(dtype=np.float64)
    spectrum = np.fft.rfftn(centered)
    power = np.abs(spectrum) ** 2
    frequency = 2.0 * np.pi * np.fft.fftfreq(GRID, d=box_mpc_h / GRID)
    frequency_z = 2.0 * np.pi * np.fft.rfftfreq(GRID, d=box_mpc_h / GRID)
    radius = np.sqrt(
        frequency[:, None, None] ** 2
        + frequency[None, :, None] ** 2
        + frequency_z[None, None, :] ** 2
    )
    result: dict[str, float] = {}
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (radius >= lower) & (radius < upper)
        if not np.any(selected):
            raise RuntimeError(f"empty Fourier band {lower}-{upper}")
        result[f"{lower:g}-{upper:g}_h_mpc"] = float(power[selected].mean())
    return result


def compare_log_fields(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference_log = log_density(reference)
    candidate_log = log_density(candidate)
    difference = candidate_log - reference_log
    reference_power = isotropic_power_bands(reference_log)
    candidate_power = isotropic_power_bands(candidate_log)
    return {
        "voxel_pearson_r": float(
            np.corrcoef(reference_log.ravel(), candidate_log.ravel())[0, 1]
        ),
        "candidate_minus_reference_mean_dex": float(difference.mean()),
        "candidate_minus_reference_rms_dex": float(
            np.sqrt(np.mean(difference**2))
        ),
        "candidate_over_reference_power": {
            name: candidate_power[name] / reference_power[name]
            for name in reference_power
        },
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    snapshot = Path(args.snapshot)
    counts, header = stream_snapshot_counts(
        snapshot, block_particles=args.block_particles
    )
    counts["ngp_expected_cic"] = smooth_ngp_as_expected_cic(counts["ngp"])
    summaries = {name: field_summary(value) for name, value in counts.items()}
    cmd_summary = None
    comparisons = None
    if args.cmd_grid:
        grids = np.load(args.cmd_grid, mmap_mode="r")
        expected = (27, 256, 256, 256)
        if grids.shape != expected:
            raise ValueError(f"unexpected CMD shape {grids.shape}, expected {expected}")
        cmd = conservative_resample_cube(grids[args.realization], GRID)
        cmd_summary = field_summary(cmd)
        comparisons = {
            "cic_vs_legacy_cmd": compare_log_fields(cmd, counts["cic"]),
            "ngp_expected_cic_vs_legacy_cmd": compare_log_fields(
                cmd, counts["ngp_expected_cic"]
            ),
            "ngp_half_particle_floor_vs_legacy_cmd": compare_log_fields(
                cmd, np.maximum(counts["ngp"], 0.5)
            ),
        }

    report = {
        "schema": "hong2021-camels-target-operator-audit-v1",
        "simulation": args.simulation,
        "realization": args.realization,
        "snapshot": str(snapshot.resolve()),
        "snapshot_bytes": snapshot.stat().st_size,
        "header": header,
        "grid": GRID,
        "voxel_mpc_h": CELL_MPC_H,
        "particle_assignments": summaries,
        "legacy_cmd_adaptive_32_neighbor": cmd_summary,
        "log_density_comparisons": comparisons,
        "interpretation_boundary": (
            "Development-data operator audit only; no independent-gate truth "
            "or model-selection metric is read."
        ),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, destination)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--simulation", default="CAMELS-SIMBA-CV")
    parser.add_argument("--realization", type=int, required=True)
    parser.add_argument("--cmd-grid")
    parser.add_argument("--block-particles", type=int, default=1_000_000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
