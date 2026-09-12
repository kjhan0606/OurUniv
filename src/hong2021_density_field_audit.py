#!/usr/bin/env python
"""Test whether observer-centered Hong target cubes are void-biased."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import ks_2samp

from hong2021_prepare_tng import extract_periodic_cube


THRESHOLDS_LOW = (0.1, 0.2, 0.5, 1.0)
THRESHOLDS_HIGH = (2.0, 10.0, 100.0)


def cube_metrics(count: np.ndarray, mean_count: float) -> dict[str, float]:
    density = np.asarray(count, dtype=np.float64) / mean_count
    log_density = np.log10(density)
    result = {
        "linear_density_mean": float(density.mean()),
        "log10_density_mean": float(log_density.mean()),
        "log10_density_std": float(log_density.std()),
    }
    total_mass = float(density.sum())
    for threshold in THRESHOLDS_LOW:
        selected = density < threshold
        label = f"density_lt_{threshold:g}"
        result[f"volume_fraction_{label}"] = float(selected.mean())
        result[f"mass_fraction_{label}"] = float(density[selected].sum() / total_mass)
    for threshold in THRESHOLDS_HIGH:
        selected = density > threshold
        label = f"density_gt_{threshold:g}"
        result[f"volume_fraction_{label}"] = float(selected.mean())
        result[f"mass_fraction_{label}"] = float(density[selected].sum() / total_mass)
    return result


def measure_origins(
    counts: np.ndarray,
    origins: np.ndarray,
    mean_count: float,
) -> dict[str, np.ndarray]:
    rows = [
        cube_metrics(extract_periodic_cube(counts, origin), mean_count)
        for origin in origins
    ]
    return {
        key: np.asarray([row[key] for row in rows], dtype=np.float64)
        for key in rows[0]
    }


def summarize(values: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p05_p16_p50_p84_p95": np.percentile(
            values, (5, 16, 50, 84, 95)
        ).tolist(),
    }


def comparison(
    values: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in values:
        first = values[key]
        second = reference[key]
        pooled = np.sqrt(0.5 * (first.var() + second.var()))
        output[key] = {
            "standardized_mean_difference": float(
                (first.mean() - second.mean()) / pooled
            ) if pooled > 0 else 0.0,
            "ks_distance": float(ks_2samp(first, second).statistic),
        }
    return output


def coverage(origins: np.ndarray, grid: int, cube: int) -> dict[str, float]:
    visits = np.zeros((grid, grid, grid), dtype=np.uint16)
    for origin in origins:
        axes = [(np.arange(cube) + int(value)) % grid for value in origin]
        visits[np.ix_(*axes)] += 1
    covered = visits > 0
    cube_fraction = cube**3 / grid**3
    return {
        "cube_count": int(len(origins)),
        "sum_cube_volume_over_box_volume": float(len(origins) * cube**3 / grid**3),
        "unique_box_volume_fraction_covered": float(covered.mean()),
        "unique_volume_in_equivalent_cube_volumes": float(
            covered.mean() / cube_fraction
        ),
        "mean_visits_per_covered_voxel": float(visits[covered].mean()),
        "maximum_visits": int(visits.max()),
    }


def read_origins(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["cube_origin_cell"], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dm-grid", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-cubes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    counts = np.load(args.dm_grid, mmap_mode="r")
    if counts.ndim != 3 or len(set(counts.shape)) != 1:
        raise SystemExit("DM grid must be cubic")
    grid = int(counts.shape[0])
    mean_count = float(counts.sum(dtype=np.uint64) / counts.size)
    with h5py.File(args.metadata, "r") as handle:
        candidate_origins = np.asarray(handle["cube_origin_cell"], dtype=np.int64)
    train_origins = read_origins(args.train)
    validation_origins = read_origins(args.validation)
    cube = 64
    generator = np.random.default_rng(args.seed)
    random_origins = generator.integers(
        0, grid, size=(args.random_cubes, 3), dtype=np.int64
    )

    print("[audit] full periodic volume", flush=True)
    full = cube_metrics(counts, mean_count)
    groups = {}
    for name, origins in (
        ("random_volume", random_origins),
        ("all_988_observer_candidates", candidate_origins),
        ("training", train_origins),
        ("validation", validation_origins),
    ):
        print(f"[audit] {name} cubes={len(origins)}", flush=True)
        groups[name] = measure_origins(counts, origins, mean_count)

    random_reference = groups["random_volume"]
    report = {
        "schema": "hong2021-density-field-environment-audit-v1",
        "dm_grid": str(args.dm_grid),
        "grid": grid,
        "cube_grid": cube,
        "cube_mpc_h": 20.0,
        "mean_particles_per_voxel": mean_count,
        "seed": args.seed,
        "random_cubes": args.random_cubes,
        "full_periodic_volume": full,
        "groups": {
            name: {
                "cubes": int(len(next(iter(metrics.values())))),
                "metrics": {key: summarize(value) for key, value in metrics.items()},
                "versus_random_volume": comparison(metrics, random_reference),
            }
            for name, metrics in groups.items()
        },
        "direct_comparisons": {
            "training_versus_validation": comparison(
                groups["training"], groups["validation"]
            ),
            "training_versus_all_988_observer_candidates": comparison(
                groups["training"], groups["all_988_observer_candidates"]
            ),
        },
        "coverage": {
            "all_988_observer_candidates": coverage(candidate_origins, grid, cube),
            "training": coverage(train_origins, grid, cube),
            "validation": coverage(validation_origins, grid, cube),
            "train_and_validation": coverage(
                np.concatenate((train_origins, validation_origins)), grid, cube
            ),
        },
        "interpretation_rule": {
            "representative": "Observer-candidate and training metrics overlap the random-volume cube distributions without large SMD/KS shifts",
            "void_biased": "Observer-candidate or training void fractions are systematically higher than random-volume cubes",
            "selection_biased_not_void_biased": "Differences exist but observer-centered cubes are denser and less void-rich than random-volume cubes"
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
