#!/usr/bin/env python
"""Decompose Hong validation error by true-density environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


EDGES = (0.0, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, np.inf)
DENSITY_SCALE = 4.5


def label(low: float, high: float) -> str:
    upper = "inf" if np.isinf(high) else f"{high:g}"
    return f"{low:g}_to_{upper}_rho_mean"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with h5py.File(args.data, "r") as handle:
        truth_y = np.asarray(handle["target"][:, 0], dtype=np.float32)
    with h5py.File(args.predictions, "r") as handle:
        prediction_y = np.asarray(handle[args.label], dtype=np.float32)
    if prediction_y.shape != truth_y.shape:
        raise SystemExit("prediction and truth shapes differ")

    truth_density = np.power(10.0, DENSITY_SCALE * truth_y.astype(np.float64))
    residual_y = prediction_y.astype(np.float64) - truth_y.astype(np.float64)
    squared = np.square(residual_y)
    absolute = np.abs(residual_y)
    total_squared = float(squared.sum())
    total_mass = float(truth_density.sum())
    bins = {}
    for low, high in zip(EDGES[:-1], EDGES[1:], strict=True):
        selected = (truth_density >= low) & (truth_density < high)
        count = int(selected.sum())
        truth_values = truth_y[selected].astype(np.float64)
        prediction_values = prediction_y[selected].astype(np.float64)
        covariance = np.corrcoef(truth_values, prediction_values)[0, 1]
        bins[label(low, high)] = {
            "voxels": count,
            "volume_fraction": count / truth_density.size,
            "truth_mass_fraction": float(truth_density[selected].sum() / total_mass),
            "mse_y": float(squared[selected].mean()),
            "absolute_error_y_mean": float(absolute[selected].mean()),
            "fraction_of_total_squared_error": float(
                squared[selected].sum() / total_squared
            ),
            "mean_log10_prediction_over_truth_dex": float(
                DENSITY_SCALE * residual_y[selected].mean()
            ),
            "voxel_pearson_y": float(covariance),
        }
    report = {
        "schema": "hong2021-density-stratified-validation-v1",
        "data": str(args.data),
        "predictions": str(args.predictions),
        "label": args.label,
        "samples": int(truth_y.shape[0]),
        "total_voxel_mse_y": float(squared.mean()),
        "bins": bins,
        "interpretation": (
            "Uniform voxel MSE weights a bin in proportion to its volume and "
            "per-voxel error, not its mass or scientific filament importance."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
