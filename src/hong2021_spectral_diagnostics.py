#!/usr/bin/env python
"""Fourier and normalization-resolved 2pCF diagnostics for Hong predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from hong2021_evaluate import (
    DENSITY_SCALE,
    OpenBoundaryTwoPoint,
    summarize_ks,
)


TWO_POINT_MODES = {
    "cosmic_mean": "delta_truth=rho_truth-1; delta_prediction=rho_prediction-1",
    "shared_truth_ensemble_mean": (
        "both fields divided by the truth validation-ensemble mean"
    ),
    "own_ensemble_mean": "each field divided by its own validation-ensemble mean",
    "shared_truth_cube_mean": "both fields divided by each truth cube mean",
    "own_cube_mean": "each field divided by its own cube mean",
}
K_RANGES = ((0.3, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 10.1))


def density_ratio(normalized_log_density: np.ndarray) -> np.ndarray:
    return np.power(
        np.float32(10.0),
        np.float32(DENSITY_SCALE) * np.asarray(normalized_log_density, dtype=np.float32),
    )


def contrast_pair(
    truth: np.ndarray,
    prediction: np.ndarray,
    mode: str,
    truth_ensemble_mean: float,
    prediction_ensemble_mean: float,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "cosmic_mean":
        return truth - 1.0, prediction - 1.0
    if mode == "shared_truth_ensemble_mean":
        return (
            truth / truth_ensemble_mean - 1.0,
            prediction / truth_ensemble_mean - 1.0,
        )
    if mode == "own_ensemble_mean":
        return (
            truth / truth_ensemble_mean - 1.0,
            prediction / prediction_ensemble_mean - 1.0,
        )
    truth_cube_mean = float(np.mean(truth, dtype=np.float64))
    if mode == "shared_truth_cube_mean":
        return truth / truth_cube_mean - 1.0, prediction / truth_cube_mean - 1.0
    if mode == "own_cube_mean":
        prediction_cube_mean = float(np.mean(prediction, dtype=np.float64))
        return (
            truth / truth_cube_mean - 1.0,
            prediction / prediction_cube_mean - 1.0,
        )
    raise ValueError(f"unknown 2pCF normalization mode: {mode}")


def fourier_diagnostics(
    truth_field: np.ndarray,
    prediction_field: np.ndarray,
    voxel_mpc_h: float,
    field_mode: str = "linear_density",
) -> dict[str, Any]:
    if truth_field.shape != prediction_field.shape or truth_field.ndim != 4:
        raise ValueError("Fourier inputs must match (sample,grid,grid,grid)")
    if field_mode not in {"linear_density", "centered_log_density"}:
        raise ValueError(f"unknown Fourier field mode: {field_mode}")
    samples, grid, second, third = truth_field.shape
    if grid != second or grid != third:
        raise ValueError("Fourier inputs must be cubic")
    box = grid * voxel_mpc_h
    fundamental = 2.0 * np.pi / box
    nyquist = np.pi / voxel_mpc_h
    kx = 2.0 * np.pi * np.fft.fftfreq(grid, d=voxel_mpc_h)
    ky = 2.0 * np.pi * np.fft.fftfreq(grid, d=voxel_mpc_h)
    kz = 2.0 * np.pi * np.fft.rfftfreq(grid, d=voxel_mpc_h)
    kmagnitude = np.sqrt(
        kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    )
    edges = np.arange(0.5, grid / 2.0 + 1.5) * fundamental
    shell = np.digitize(kmagnitude.ravel(), edges) - 1
    valid = (kmagnitude.ravel() > 0) & (kmagnitude.ravel() <= nyquist)
    valid &= (shell >= 0) & (shell < len(edges) - 1)
    shell = shell[valid]
    rfft_weight = np.full(len(kz), 2.0)
    rfft_weight[0] = 1.0
    if grid % 2 == 0:
        rfft_weight[-1] = 1.0
    weights = np.broadcast_to(
        rfft_weight[None, None, :], kmagnitude.shape
    ).ravel()[valid]
    mode_count = np.bincount(shell, weights=weights, minlength=len(edges) - 1)
    ksum = np.bincount(
        shell,
        weights=weights * kmagnitude.ravel()[valid],
        minlength=len(edges) - 1,
    )
    kcenter = np.divide(ksum, mode_count, out=np.zeros_like(ksum), where=mode_count > 0)
    window_1d = np.hanning(grid)
    window = (
        window_1d[:, None, None]
        * window_1d[None, :, None]
        * window_1d[None, None, :]
    )
    window /= np.sqrt(np.mean(np.square(window)))
    truth_power = np.empty((samples, len(kcenter)), dtype=np.float64)
    prediction_power = np.empty_like(truth_power)
    cross_power = np.empty_like(truth_power)
    residual_power = np.empty_like(truth_power)
    for sample in range(samples):
        truth = truth_field[sample].astype(np.float64)
        prediction = prediction_field[sample].astype(np.float64)
        if field_mode == "linear_density":
            truth = truth / truth.mean() - 1.0
            prediction = prediction / prediction.mean() - 1.0
            definition = "linear density contrast normalized by each cube mean"
        else:
            truth = truth - truth.mean()
            prediction = prediction - prediction.mean()
            definition = "centered log10(rho/rho0)"
        truth_fft = np.fft.rfftn(truth * window)
        prediction_fft = np.fft.rfftn(prediction * window)
        residual_fft = prediction_fft - truth_fft
        spectra = (
            np.square(np.abs(truth_fft)),
            np.square(np.abs(prediction_fft)),
            np.real(prediction_fft * truth_fft.conjugate()),
            np.square(np.abs(residual_fft)),
        )
        destinations = (
            truth_power,
            prediction_power,
            cross_power,
            residual_power,
        )
        for spectrum, destination in zip(spectra, destinations, strict=True):
            summed = np.bincount(
                shell,
                weights=weights * spectrum.ravel()[valid],
                minlength=len(kcenter),
            )
            destination[sample] = np.divide(
                summed,
                mode_count,
                out=np.full_like(summed, np.nan),
                where=mode_count > 0,
            )
    keep = mode_count > 0
    kcenter = kcenter[keep]
    mode_count = mode_count[keep]
    truth_power = truth_power[:, keep]
    prediction_power = prediction_power[:, keep]
    cross_power = cross_power[:, keep]
    residual_power = residual_power[:, keep]
    mean_truth = truth_power.mean(axis=0)
    mean_prediction = prediction_power.mean(axis=0)
    mean_cross = cross_power.mean(axis=0)
    mean_residual = residual_power.mean(axis=0)
    transfer = np.sqrt(mean_prediction / mean_truth)
    correlation = mean_cross / np.sqrt(mean_prediction * mean_truth)
    residual_fraction = mean_residual / mean_truth
    sample_transfer = np.sqrt(prediction_power / truth_power)
    sample_correlation = cross_power / np.sqrt(prediction_power * truth_power)
    scale_summary: dict[str, Any] = {}
    for low, high in K_RANGES:
        selected = (kcenter >= low) & (kcenter < min(high, nyquist + 1.0e-9))
        if not np.any(selected):
            continue
        label = f"{low:g}-{min(high, nyquist):g}_h_mpc"
        scale_summary[label] = {
            "modes": int(np.sum(mode_count[selected])),
            "transfer_mode_weighted_mean": float(
                np.average(transfer[selected], weights=mode_count[selected])
            ),
            "cross_correlation_mode_weighted_mean": float(
                np.average(correlation[selected], weights=mode_count[selected])
            ),
            "residual_to_truth_power_mode_weighted_mean": float(
                np.average(residual_fraction[selected], weights=mode_count[selected])
            ),
        }
    return {
        "definition": (
            f"{definition}; separable Hann window; ensemble-mean auto/cross spectra"
        ),
        "box_mpc_h": box,
        "fundamental_h_mpc": fundamental,
        "nyquist_h_mpc": nyquist,
        "k_h_mpc": kcenter.tolist(),
        "mode_count": mode_count.astype(int).tolist(),
        "transfer_sqrt_Ppred_over_Ptruth": transfer.tolist(),
        "cross_correlation_r": correlation.tolist(),
        "residual_to_truth_power": residual_fraction.tolist(),
        "sample_transfer_p16_p50_p84": np.percentile(
            sample_transfer, [16, 50, 84], axis=0
        ).tolist(),
        "sample_cross_correlation_p16_p50_p84": np.percentile(
            sample_correlation, [16, 50, 84], axis=0
        ).tolist(),
        "by_scale": scale_summary,
    }


def plot_diagnostics(
    path: Path,
    truth_density: np.ndarray,
    prediction_density: np.ndarray,
    fourier_linear: dict[str, Any],
    fourier_log: dict[str, Any],
    two_point: dict[str, Any],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    k = np.asarray(fourier_linear["k_h_mpc"])
    linear_transfer = np.asarray(
        fourier_linear["transfer_sqrt_Ppred_over_Ptruth"]
    )
    log_transfer = np.asarray(fourier_log["transfer_sqrt_Ppred_over_Ptruth"])
    axes[0, 0].plot(k, linear_transfer, label="linear density")
    axes[0, 0].plot(k, log_transfer, label="log density")
    axes[0, 0].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_ylim(0, 1.1 * max(np.max(linear_transfer), np.max(log_transfer)))
    axes[0, 0].set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    axes[0, 0].set_ylabel(r"$T(k)=\sqrt{P_{pred}/P_{truth}}$")
    axes[0, 0].legend()
    axes[0, 0].set_title("Density transfer function")

    axes[0, 1].plot(
        k, fourier_linear["cross_correlation_r"], label="r(k), linear"
    )
    axes[0, 1].plot(k, fourier_log["cross_correlation_r"], label="r(k), log")
    axes[0, 1].plot(
        k,
        fourier_linear["residual_to_truth_power"],
        label=r"$P_{res}/P_{truth}$, linear",
        alpha=0.7,
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("symlog", linthresh=0.1)
    axes[0, 1].set_xlabel(r"$k\ [h\,{\rm Mpc}^{-1}]$")
    axes[0, 1].legend()
    axes[0, 1].set_title("Phase correlation and residual power")

    radius = np.asarray(next(iter(two_point.values()))["ks"]["radius_mpc_h"])
    for mode, result in two_point.items():
        axes[1, 0].plot(radius, result["ks"]["ks_at_radius"], label=mode)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel(r"$r\ [h^{-1}{\rm Mpc}]$")
    axes[1, 0].set_ylabel("2pCF distribution KS")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].set_title("Finite-cube normalization sensitivity")

    bins = np.linspace(-3.0, 3.0, 150)
    axes[1, 1].hist(
        np.log10(truth_density).ravel(),
        bins=bins,
        density=True,
        histtype="step",
        label="truth",
    )
    axes[1, 1].hist(
        np.log10(prediction_density).ravel(),
        bins=bins,
        density=True,
        histtype="step",
        label="prediction",
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel(r"$\log_{10}(\rho/\rho_0)$")
    axes[1, 1].legend()
    axes[1, 1].set_title("One-point density PDF")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--voxel-mpc-h", type=float, default=0.3125)
    parser.add_argument("--rmax-mpc-h", type=float, default=10.0)
    args = parser.parse_args()
    with h5py.File(args.data, "r") as handle:
        truth_y = np.asarray(handle["target"][:, 0], dtype=np.float32)
    with h5py.File(args.predictions, "r") as handle:
        if args.label not in handle:
            raise SystemExit(f"prediction label missing: {args.label}")
        prediction_y = np.asarray(handle[args.label], dtype=np.float32)
    if truth_y.shape != prediction_y.shape:
        raise SystemExit("truth and prediction shapes differ")
    truth_density = density_ratio(truth_y)
    prediction_density = density_ratio(prediction_y)
    truth_mean = float(np.mean(truth_density, dtype=np.float64))
    prediction_mean = float(np.mean(prediction_density, dtype=np.float64))
    estimator = OpenBoundaryTwoPoint(
        truth_y.shape[-1], args.voxel_mpc_h, args.rmax_mpc_h
    )
    two_point: dict[str, Any] = {}
    profile_payload: dict[str, np.ndarray] = {
        "radius_mpc_h": estimator.radius_mpc_h
    }
    for mode, definition in TWO_POINT_MODES.items():
        print(f"[2pcf] mode={mode}", flush=True)
        truth_profiles = []
        prediction_profiles = []
        for sample in range(len(truth_density)):
            truth_delta, prediction_delta = contrast_pair(
                truth_density[sample],
                prediction_density[sample],
                mode,
                truth_mean,
                prediction_mean,
            )
            truth_profiles.append(estimator(truth_delta))
            prediction_profiles.append(estimator(prediction_delta))
        truth_profiles_array = np.asarray(truth_profiles)
        prediction_profiles_array = np.asarray(prediction_profiles)
        two_point[mode] = {
            "definition": definition,
            "ks": summarize_ks(
                truth_profiles_array,
                prediction_profiles_array,
                estimator.radius_mpc_h,
            ),
        }
        profile_payload[f"{mode}_truth"] = truth_profiles_array
        profile_payload[f"{mode}_prediction"] = prediction_profiles_array

    print("[fourier] linear-density transfer and cross-correlation", flush=True)
    fourier_linear = fourier_diagnostics(
        truth_density,
        prediction_density,
        args.voxel_mpc_h,
        field_mode="linear_density",
    )
    print("[fourier] log-density transfer and cross-correlation", flush=True)
    fourier_log = fourier_diagnostics(
        DENSITY_SCALE * truth_y,
        DENSITY_SCALE * prediction_y,
        args.voxel_mpc_h,
        field_mode="centered_log_density",
    )
    report = {
        "schema": "hong2021-spectral-diagnostics-v1",
        "data": str(args.data),
        "predictions": str(args.predictions),
        "label": args.label,
        "samples": int(len(truth_density)),
        "grid": int(truth_density.shape[-1]),
        "density_means": {
            "truth_cosmic_units": truth_mean,
            "prediction_cosmic_units": prediction_mean,
            "prediction_over_truth": prediction_mean / truth_mean,
        },
        "paper_definition_note": (
            "Hong et al. define xi=<delta(x)delta(x+r)> but do not publish "
            "the finite-cube mean or boundary convention"
        ),
        "two_point": two_point,
        "fourier": {
            "linear_density": fourier_linear,
            "log10_density": fourier_log,
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "spectral_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    np.savez_compressed(args.out / "normalized_two_point_profiles.npz", **profile_payload)
    plot_diagnostics(
        args.out / "spectral_diagnostics.png",
        truth_density,
        prediction_density,
        fourier_linear,
        fourier_log,
        two_point,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
