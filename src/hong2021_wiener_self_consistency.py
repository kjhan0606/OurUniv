#!/usr/bin/env python
"""Falsify a train-fitted Fourier density likelihood on TNG validation mocks."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
from typing import Any

import h5py
import numpy as np


SCHEMA = "hong2021-density-likelihood-wiener-self-consistency-e2-v1"


def fourier_geometry(
    grid: int, voxel: float, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kxy = 2.0 * np.pi * np.fft.fftfreq(grid, d=voxel)
    kz = 2.0 * np.pi * np.fft.rfftfreq(grid, d=voxel)
    magnitude = np.sqrt(
        kxy[:, None, None] ** 2
        + kxy[None, :, None] ** 2
        + kz[None, None, :] ** 2
    )
    index = np.digitize(magnitude.ravel(), edges) - 1
    valid = (index >= 0) & (index < len(edges) - 1)
    one_weight = np.ones(len(kz), dtype=np.float64)
    if grid % 2 == 0:
        one_weight[1:-1] = 2.0
    else:
        one_weight[1:] = 2.0
    mode_weight = np.broadcast_to(
        one_weight[None, None, :], magnitude.shape
    ).ravel().copy()
    counts = np.bincount(
        index[valid], weights=mode_weight[valid], minlength=len(edges) - 1
    )
    mean_k = np.bincount(
        index[valid], weights=mode_weight[valid] * magnitude.ravel()[valid],
        minlength=len(edges) - 1,
    ) / counts
    return index, valid, mode_weight, np.column_stack((mean_k, counts))


def posterior_parameters(
    truth_power: np.ndarray,
    transfer: np.ndarray,
    noise_power: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = transfer * transfer * truth_power + noise_power
    gain = truth_power * transfer / denominator
    variance = truth_power * noise_power / denominator
    return gain, variance


def read_pair(
    data: h5py.File, means: h5py.File, sample: int
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(data["target"][sample, 0], dtype=np.float64)
    prediction = np.asarray(means["conditional_mean"][sample, 0], dtype=np.float64)
    truth -= truth.mean()
    prediction -= prediction.mean()
    return truth, prediction


def fit_likelihood(
    data_path: Path,
    mean_path: Path,
    bin_index: np.ndarray,
    valid: np.ndarray,
    mode_weight: np.ndarray,
    bins: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sums = np.zeros((bins, 3), dtype=np.float64)
    counts = np.zeros(bins, dtype=np.float64)
    with h5py.File(data_path, "r") as data, h5py.File(mean_path, "r") as means:
        samples = len(data["target"])
        if samples != len(means["conditional_mean"]):
            raise ValueError("calibration data/mean sample mismatch")
        per_sample_count = np.bincount(
            bin_index[valid], weights=mode_weight[valid], minlength=bins
        )
        for sample in range(samples):
            truth, prediction = read_pair(data, means, sample)
            truth_fft = np.fft.rfftn(truth, norm="ortho").ravel()
            mean_fft = np.fft.rfftn(prediction, norm="ortho").ravel()
            values = (
                np.abs(truth_fft) ** 2,
                np.real(mean_fft * np.conj(truth_fft)),
                np.abs(mean_fft) ** 2,
            )
            for column, value in enumerate(values):
                sums[:, column] += np.bincount(
                    bin_index[valid],
                    weights=mode_weight[valid] * value[valid],
                    minlength=bins,
                )
            counts += per_sample_count
            if (sample + 1) % 50 == 0 or sample + 1 == samples:
                print(f"[fit] {sample + 1}/{samples}", flush=True)
    spectra = sums / counts[:, None]
    truth_power = spectra[:, 0]
    transfer = spectra[:, 1] / truth_power
    noise_power = spectra[:, 2] - spectra[:, 1] ** 2 / truth_power
    coherence = spectra[:, 1] ** 2 / (spectra[:, 0] * spectra[:, 2])
    if np.any(noise_power <= 0):
        raise ValueError("fitted likelihood has nonpositive noise power")
    gain, posterior_variance = posterior_parameters(
        truth_power, transfer, noise_power
    )
    result = {
        "truth_power": truth_power,
        "mean_truth_cross_power": spectra[:, 1],
        "mean_power": spectra[:, 2],
        "transfer": transfer,
        "noise_power": noise_power,
        "noise_over_truth": noise_power / truth_power,
        "coherence_squared": coherence,
        "wiener_gain": gain,
        "posterior_variance": posterior_variance,
    }
    metadata = {
        "data": str(data_path.resolve()),
        "conditional_mean": str(mean_path.resolve()),
        "samples": samples,
    }
    return result, metadata


def correlation(sxy: float, sxx: float, syy: float) -> float:
    return float(sxy / math.sqrt(sxx * syy))


def evaluate(
    data_path: Path,
    mean_path: Path,
    fitted: dict[str, np.ndarray],
    bin_index: np.ndarray,
    valid: np.ndarray,
    mode_weight: np.ndarray,
    grid: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bins = len(fitted["transfer"])
    shape = (grid, grid, grid // 2 + 1)
    index_grid = bin_index.reshape(shape)
    valid_grid = valid.reshape(shape)
    gain_grid = np.zeros(shape, dtype=np.float64)
    transfer_grid = np.zeros(shape, dtype=np.float64)
    noise_grid = np.ones(shape, dtype=np.float64)
    posterior_variance_grid = np.zeros(shape, dtype=np.float64)
    for band in range(bins):
        mask = valid_grid & (index_grid == band)
        gain_grid[mask] = fitted["wiener_gain"][band]
        transfer_grid[mask] = fitted["transfer"][band]
        noise_grid[mask] = fitted["noise_power"][band]
        posterior_variance_grid[mask] = fitted["posterior_variance"][band]
    posterior_voxel_sigma = math.sqrt(
        float(np.sum(mode_weight.reshape(shape) * posterior_variance_grid))
        / grid**3
    )
    z68 = NormalDist().inv_cdf(0.84)
    z95 = NormalDist().inv_cdf(0.975)
    coverage68 = 0
    coverage95 = 0
    voxels = 0
    chi_sum = np.zeros(bins, dtype=np.float64)
    chi_count = np.zeros(bins, dtype=np.float64)
    sums = {
        "truth2": 0.0,
        "posterior2": 0.0,
        "mean2": 0.0,
        "truth_posterior": 0.0,
        "truth_mean": 0.0,
        "posterior_error2": 0.0,
        "mean_error2": 0.0,
    }
    with h5py.File(data_path, "r") as data, h5py.File(mean_path, "r") as means:
        samples = len(data["target"])
        if samples != len(means["conditional_mean"]):
            raise ValueError("evaluation data/mean sample mismatch")
        for sample in range(samples):
            truth, prediction = read_pair(data, means, sample)
            truth_fft = np.fft.rfftn(truth, norm="ortho")
            mean_fft = np.fft.rfftn(prediction, norm="ortho")
            truth_band = np.fft.irfftn(
                np.where(valid_grid, truth_fft, 0.0), s=truth.shape,
                axes=(0, 1, 2), norm="ortho"
            ).real
            mean_band = np.fft.irfftn(
                np.where(valid_grid, mean_fft, 0.0), s=truth.shape,
                axes=(0, 1, 2), norm="ortho"
            ).real
            posterior = np.fft.irfftn(
                gain_grid * mean_fft, s=truth.shape,
                axes=(0, 1, 2), norm="ortho"
            ).real
            error = truth_band - posterior
            coverage68 += int(np.count_nonzero(np.abs(error) <= z68 * posterior_voxel_sigma))
            coverage95 += int(np.count_nonzero(np.abs(error) <= z95 * posterior_voxel_sigma))
            voxels += error.size
            sums["truth2"] += float(np.sum(truth_band * truth_band))
            sums["posterior2"] += float(np.sum(posterior * posterior))
            sums["mean2"] += float(np.sum(mean_band * mean_band))
            sums["truth_posterior"] += float(np.sum(truth_band * posterior))
            sums["truth_mean"] += float(np.sum(truth_band * mean_band))
            sums["posterior_error2"] += float(np.sum(error * error))
            sums["mean_error2"] += float(np.sum((truth_band - mean_band) ** 2))
            residual = mean_fft.ravel() - transfer_grid.ravel() * truth_fft.ravel()
            normalized = np.abs(residual) ** 2 / noise_grid.ravel()
            for band in range(bins):
                mask = valid & (bin_index == band)
                chi_sum[band] += float(np.sum(mode_weight[mask] * normalized[mask]))
                chi_count[band] += float(np.sum(mode_weight[mask]))
            if (sample + 1) % 25 == 0 or sample + 1 == samples:
                print(f"[evaluate] {sample + 1}/{samples}", flush=True)
    metrics = {
        "posterior_voxel_sigma": posterior_voxel_sigma,
        "posterior_voxel_coverage_68": coverage68 / voxels,
        "posterior_voxel_coverage_95": coverage95 / voxels,
        "reduced_chi2_by_band": (chi_sum / chi_count).tolist(),
        "posterior_mean_mse": sums["posterior_error2"] / voxels,
        "deterministic_mean_mse": sums["mean_error2"] / voxels,
        "posterior_mean_mse_over_deterministic_mean_mse": (
            sums["posterior_error2"] / sums["mean_error2"]
        ),
        "posterior_mean_correlation": correlation(
            sums["truth_posterior"], sums["truth2"], sums["posterior2"]
        ),
        "deterministic_mean_correlation": correlation(
            sums["truth_mean"], sums["truth2"], sums["mean2"]
        ),
        "truth_band_variance": sums["truth2"] / voxels,
    }
    metadata = {
        "data": str(data_path.resolve()),
        "conditional_mean": str(mean_path.resolve()),
        "samples": samples,
        "evaluated_voxels": voxels,
    }
    return metrics, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--train-mean", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--validation-mean", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k-edges", default="0.3,0.6,1.0,1.5,2.0")
    args = parser.parse_args()
    edges = np.asarray([float(value) for value in args.k_edges.split(",")])
    with h5py.File(args.train_data, "r") as handle:
        grid = int(handle["target"].shape[-1])
        voxel = float(handle.attrs["voxel_mpc_h"])
    bin_index, valid, mode_weight, geometry = fourier_geometry(grid, voxel, edges)
    fitted, calibration_metadata = fit_likelihood(
        args.train_data, args.train_mean, bin_index, valid,
        mode_weight, len(edges) - 1,
    )
    metrics, evaluation_metadata = evaluate(
        args.validation_data, args.validation_mean, fitted,
        bin_index, valid, mode_weight, grid,
    )
    chi_pass = all(0.9 <= value <= 1.1 for value in metrics["reduced_chi2_by_band"])
    coverage68_pass = 0.55 <= metrics["posterior_voxel_coverage_68"] <= 0.85
    coverage95_pass = 0.85 <= metrics["posterior_voxel_coverage_95"] <= 0.99
    mse_pass = metrics["posterior_mean_mse_over_deterministic_mean_mse"] <= 1.0
    correlation_pass = (
        metrics["posterior_mean_correlation"]
        >= metrics["deterministic_mean_correlation"] - 1.0e-12
    )
    checks = {
        "reduced_chi2_each_band_0.9_1.1": chi_pass,
        "posterior_coverage_68_in_0.55_0.85": coverage68_pass,
        "posterior_coverage_95_in_0.85_0.99": coverage95_pass,
        "posterior_mean_mse_not_worse_than_deterministic": mse_pass,
        "posterior_mean_correlation_not_lower_than_deterministic": correlation_pass,
    }
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "scope": "linearized present-density self-consistency; not IC validation",
        "k_edges_h_mpc": edges.tolist(),
        "k_mean_h_mpc": geometry[:, 0].tolist(),
        "full_fourier_mode_count": geometry[:, 1].astype(int).tolist(),
        "calibration": calibration_metadata,
        "evaluation": evaluation_metadata,
        "fitted_likelihood": {key: value.tolist() for key, value in fitted.items()},
        "metrics": metrics,
        "frozen_gate": {
            "checks": checks,
            "pass": all(checks.values()),
            "decision": (
                "advance to differentiable forward-mock density-plus-velocity E3"
                if all(checks.values())
                else "remove the frozen Hong density likelihood from the IC path"
            ),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "wiener_self_consistency.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report["metrics"], indent=2), flush=True)
    print(json.dumps(report["frozen_gate"], indent=2), flush=True)
    print(f"[out] {output}", flush=True)


if __name__ == "__main__":
    main()
