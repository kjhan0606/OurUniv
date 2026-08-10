#!/usr/bin/env python
"""Evaluate conditional residual ensembles against held-out TNG cubes."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from hong2021_evaluate import OpenBoundaryTwoPoint, ks_statistic
from hong2021_residual_diffusion import highpass_mask, highpass_numpy
from hong2021_residual_v6 import SCHEMA as V6_SCHEMA, laplacian_residual


DENSITY_SCALE = 4.5
BANDS = ((0.3, 1.0), (1.0, 3.0), (3.0, 6.0), (6.0, 10.0531))
LAPLACIAN_SCHEMAS = {
    V6_SCHEMA,
    "hong2021-conditional-laplacian-residual-v8-observable-context",
    "hong2021-conditional-laplacian-residual-v9-tail-balanced",
    "hong2021-two-component-residual-v10-lowpass-mean-v8-edm",
}
CENTERED_SCHEMAS = {
    "hong2021-corrected-mean-fullband-residual-v11-observable-context-edm",
    "hong2021-gaussianized-fullband-residual-v12-observable-context-edm",
    "hong2021-v13-dc-corrected-v12-ensemble",
    "hong2021-v14-multiscale-location-scale-edm-ensemble-v1",
    "hong2021-v28-train-only-empirical-joint-residual-ensemble-v1",
    "hong2021-v29-direct-physical-residual-transport-ensemble-v1",
    "hong2021-v31-physical-conditional-copula-ensemble-v1",
    "hong2021-v37-query-aligned-conditional-copula-ensemble-v1",
    "hong2021-v38-gaussian-copula-innovation-ensemble-v1",
    "hong2021-v39-bijective-local-patch-copula-ensemble-v1",
    "hong2021-v41-two-stage-structure-amplitude-ensemble-v1",
    "hong2021-v42-within-block-tail-body-ensemble-v1",
    "hong2021-v44-query-local-mixture-copula-ensemble-v1",
    "hong2021-v45-identifiable-query-local-mixture-copula-ensemble-v1",
    "hong2021-v48-identifiable-query-local-Gaussian-mixture-copula-ensemble-v1",
}


def band_key(low: float, high: float) -> str:
    return f"{low:g}-{high:g}_h_mpc"


@dataclass
class SpectralBinner:
    grid: int
    voxel_mpc_h: float

    def __post_init__(self) -> None:
        kxy = 2.0 * np.pi * np.fft.fftfreq(self.grid, d=self.voxel_mpc_h)
        kz = 2.0 * np.pi * np.fft.rfftfreq(self.grid, d=self.voxel_mpc_h)
        magnitude = np.sqrt(
            kxy[:, None, None] ** 2
            + kxy[None, :, None] ** 2
            + kz[None, None, :] ** 2
        )
        self.edges = np.linspace(0.0, np.pi / self.voxel_mpc_h, self.grid // 2 + 1)
        self.bin = np.digitize(magnitude.ravel(), self.edges) - 1
        self.valid = (self.bin >= 0) & (self.bin < self.grid // 2)
        self.count = np.bincount(
            self.bin[self.valid], minlength=self.grid // 2
        ).astype(np.int64)
        weighted_k = np.bincount(
            self.bin[self.valid],
            weights=magnitude.ravel()[self.valid],
            minlength=self.grid // 2,
        )
        self.k = np.divide(
            weighted_k,
            self.count,
            out=np.full(self.grid // 2, np.nan),
            where=self.count > 0,
        )
        window1 = np.hanning(self.grid).astype(np.float32)
        self.window = (
            window1[:, None, None]
            * window1[None, :, None]
            * window1[None, None, :]
        )

    def transform(self, field: np.ndarray) -> np.ndarray:
        value = np.asarray(field, dtype=np.float32)
        value = value - value.mean(axis=(-3, -2, -1), keepdims=True)
        return np.fft.rfftn(value * self.window, axes=(-3, -2, -1))

    def radial_sum(self, value: np.ndarray) -> np.ndarray:
        flat = np.asarray(value).reshape((-1, self.bin.size))
        rows = [
            np.bincount(
                self.bin[self.valid],
                weights=row[self.valid],
                minlength=self.grid // 2,
            )
            for row in flat
        ]
        return np.asarray(rows)

    def power(self, field: np.ndarray) -> np.ndarray:
        transformed = self.transform(field)
        return self.radial_sum(np.abs(transformed) ** 2)

    def cross_correlation(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        first_k = self.transform(first)
        second_k = self.transform(second)
        cross = self.radial_sum(np.real(first_k * second_k.conjugate()))
        first_power = self.radial_sum(np.abs(first_k) ** 2)
        second_power = self.radial_sum(np.abs(second_k) ** 2)
        return np.divide(
            cross,
            np.sqrt(first_power * second_power),
            out=np.full_like(cross, np.nan),
            where=(first_power > 0) & (second_power > 0),
        )


def band_summary(
    k: np.ndarray, mode_count: np.ndarray, value: np.ndarray
) -> dict[str, float]:
    result: dict[str, float] = {}
    for low, high in BANDS:
        selected = (k >= low) & (k < high) & np.isfinite(value)
        result[band_key(low, high)] = float(
            np.average(value[selected], weights=mode_count[selected])
        )
    return result


def density_statistics(density: np.ndarray) -> dict[str, float]:
    local_maximum = density == maximum_filter(density, size=3, mode="nearest")
    local_minimum = density == minimum_filter(density, size=3, mode="nearest")
    return {
        "volume_fraction_rho_lt_0.1": float(np.mean(density < 0.1)),
        "volume_fraction_rho_lt_0.5": float(np.mean(density < 0.5)),
        "volume_fraction_rho_gt_10": float(np.mean(density > 10.0)),
        "volume_fraction_rho_gt_100": float(np.mean(density > 100.0)),
        "local_peak_count_rho_gt_10": float(
            np.count_nonzero(local_maximum & (density > 10.0))
        ),
        "local_peak_count_rho_gt_100": float(
            np.count_nonzero(local_maximum & (density > 100.0))
        ),
        "local_void_count_rho_lt_0.1": float(
            np.count_nonzero(local_minimum & (density < 0.1))
        ),
    }


def average_statistic(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in rows[0]
    }


def two_point_ks(
    truth: np.ndarray, generated: np.ndarray, radius: np.ndarray
) -> dict[str, Any]:
    at_radius = np.asarray(
        [ks_statistic(truth[:, i], generated[:, i]) for i in range(truth.shape[1])]
    )
    return {
        "ks_at_radius": at_radius.tolist(),
        "by_scale": {
            f"{low:g}-{high:g}_mpc_h": {
                "mean": float(at_radius[(radius >= low) & (radius < high)].mean()),
                "std": float(at_radius[(radius >= low) & (radius < high)].std()),
            }
            for low, high in ((0.0, 1.0), (1.0, 3.0), (3.0, 10.0))
        },
    }


def evaluate_candidate(
    label: str, path: Path, voxel_mpc_h: float
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with h5py.File(path, "r") as handle:
        source_indices = handle["source_index"][:].astype(np.int64)
        generated_ds = handle["sample"]
        conditional_mean = handle["conditional_mean"][:, 0].astype(np.float32)
        truth = handle["truth"][:, 0].astype(np.float32)
        objects, ensemble = generated_ds.shape[:2]
        grid = generated_ds.shape[-1]
        schema = str(handle.attrs.get("schema", ""))
        centered_schema = schema in CENTERED_SCHEMAS
        if schema in LAPLACIAN_SCHEMAS:
            sigma_cells = float(handle.attrs["sigma_cells"])
            # Half-amplitude scale of 1-exp[-(k sigma)^2/2].  This is a
            # diagnostic boundary, not a sharp preservation cutoff.
            k_low = float(
                np.sqrt(2.0 * np.log(2.0)) / (sigma_cells * voxel_mpc_h)
            )
            k_high = None
            x0_clip = None
            residual_operator = {
                "name": "reflect-boundary Gaussian-Laplacian",
                "sigma_cells": sigma_cells,
                "half_amplitude_k_h_mpc": k_low,
                "applied_again_at_sampling": False,
            }
        elif centered_schema:
            sigma_cells = None
            k_low = float(handle.attrs["diagnostic_k_h_mpc"])
            k_high = None
            x0_clip = None
            residual_operator = {
                "name": "full-band corrected-mean residual with cube DC null",
                "diagnostic_low_k_h_mpc": k_low,
                "applied_again_at_sampling": False,
            }
        else:
            sigma_cells = None
            k_low = float(handle.attrs["k_low_h_mpc"])
            k_high = float(handle.attrs["k_high_h_mpc"])
            x0_clip = float(handle.attrs["x0_clip_standardized_residual"])
            hp_mask = highpass_mask(grid, voxel_mpc_h, k_low, k_high)
            residual_operator = {
                "name": "periodic soft Fourier high-pass",
                "k_transition_h_mpc": [k_low, k_high],
                "applied_again_at_sampling": True,
            }

        binner = SpectralBinner(grid, voxel_mpc_h)
        estimator = OpenBoundaryTwoPoint(grid, voxel_mpc_h, 10.0)
        histogram_edges = np.linspace(-4.0, 6.0, 401)
        hist_truth = np.zeros(len(histogram_edges) - 1, dtype=np.int64)
        hist_mean = np.zeros_like(hist_truth)
        hist_generated = np.zeros_like(hist_truth)
        truth_power_rows: list[np.ndarray] = []
        mean_power_rows: list[np.ndarray] = []
        generated_power_rows: list[np.ndarray] = []
        truth_residual_power_rows: list[np.ndarray] = []
        generated_residual_power_rows: list[np.ndarray] = []
        ensemble_mean_cross_rows: list[np.ndarray] = []
        truth_2pcf: list[np.ndarray] = []
        mean_2pcf: list[np.ndarray] = []
        generated_2pcf: list[np.ndarray] = []
        truth_environment: list[dict[str, float]] = []
        mean_environment: list[dict[str, float]] = []
        generated_environment: list[dict[str, float]] = []
        rank_histogram = np.zeros(ensemble + 1, dtype=np.int64)
        coverage68_hits = coverage95_hits = coverage_voxels = 0
        low_k_power = total_residual_power = 0.0
        residual_mean_max_abs = 0.0
        generated_residual_square = truth_residual_square = 0.0
        generated_residual_voxels = truth_residual_voxels = 0

        for object_index in range(objects):
            generated = generated_ds[object_index, :, 0].astype(np.float32)
            mean = conditional_mean[object_index]
            target = truth[object_index]
            residual = generated - mean[None]
            if sigma_cells is not None:
                target_residual = laplacian_residual(
                    target - mean, sigma_cells=sigma_cells
                )
            elif centered_schema:
                target_residual = target - mean
                target_residual -= target_residual.mean(
                    axis=(-3, -2, -1), keepdims=True
                )
            else:
                target_residual = highpass_numpy(target - mean, hp_mask)

            residual_mean_max_abs = max(
                residual_mean_max_abs,
                float(np.max(np.abs(residual.mean(axis=(-3, -2, -1))))),
            )
            residual_fourier = np.fft.rfftn(residual, axes=(-3, -2, -1))
            magnitude2 = np.abs(residual_fourier) ** 2
            kxy = 2.0 * np.pi * np.fft.fftfreq(grid, d=voxel_mpc_h)
            kz = 2.0 * np.pi * np.fft.rfftfreq(grid, d=voxel_mpc_h)
            kmag = np.sqrt(
                kxy[:, None, None] ** 2
                + kxy[None, :, None] ** 2
                + kz[None, None, :] ** 2
            )
            low_k_power += float(magnitude2[:, kmag < k_low].sum())
            total_residual_power += float(magnitude2.sum())

            lower68, upper68 = np.quantile(residual, [0.16, 0.84], axis=0)
            lower95, upper95 = np.quantile(residual, [0.025, 0.975], axis=0)
            coverage68_hits += int(
                np.count_nonzero((target_residual >= lower68) & (target_residual <= upper68))
            )
            coverage95_hits += int(
                np.count_nonzero((target_residual >= lower95) & (target_residual <= upper95))
            )
            coverage_voxels += target_residual.size
            rank = np.sum(residual < target_residual[None], axis=0)
            rank_histogram += np.bincount(rank.ravel(), minlength=ensemble + 1)
            generated_residual_square += float(np.square(residual.astype(np.float64)).sum())
            generated_residual_voxels += residual.size
            truth_residual_square += float(np.square(target_residual.astype(np.float64)).sum())
            truth_residual_voxels += target_residual.size

            truth_power_rows.append(binner.power(target[None])[0])
            mean_power_rows.append(binner.power(mean[None])[0])
            generated_power_rows.extend(binner.power(generated))
            truth_residual_power_rows.append(binner.power(target_residual[None])[0])
            generated_residual_power_rows.extend(binner.power(residual))
            ensemble_mean_cross_rows.append(
                binner.cross_correlation(generated.mean(axis=0)[None], target[None])[0]
            )

            target_density = np.power(10.0, DENSITY_SCALE * target, dtype=np.float64)
            mean_density = np.power(10.0, DENSITY_SCALE * mean, dtype=np.float64)
            truth_2pcf.append(estimator(target_density - 1.0))
            mean_2pcf.append(estimator(mean_density - 1.0))
            truth_environment.append(density_statistics(target_density))
            mean_environment.append(density_statistics(mean_density))
            hist_truth += np.histogram(DENSITY_SCALE * target, histogram_edges)[0]
            hist_mean += np.histogram(DENSITY_SCALE * mean, histogram_edges)[0]
            for member in generated:
                density = np.power(10.0, DENSITY_SCALE * member, dtype=np.float64)
                if not np.isfinite(density).all():
                    raise RuntimeError(f"non-finite generated density in {label}")
                generated_2pcf.append(estimator(density - 1.0))
                generated_environment.append(density_statistics(density))
                hist_generated += np.histogram(
                    DENSITY_SCALE * member, histogram_edges
                )[0]

    truth_power = np.mean(truth_power_rows, axis=0)
    mean_power = np.mean(mean_power_rows, axis=0)
    generated_power = np.mean(generated_power_rows, axis=0)
    truth_residual_power = np.mean(truth_residual_power_rows, axis=0)
    generated_residual_power = np.mean(generated_residual_power_rows, axis=0)
    total_power_ratio = np.divide(
        generated_power, truth_power, out=np.full_like(truth_power, np.nan), where=truth_power > 0
    )
    deterministic_power_ratio = np.divide(
        mean_power, truth_power, out=np.full_like(truth_power, np.nan), where=truth_power > 0
    )
    residual_power_ratio = np.divide(
        generated_residual_power,
        truth_residual_power,
        out=np.full_like(truth_residual_power, np.nan),
        where=truth_residual_power > 0,
    )
    normalized_truth_hist = hist_truth / hist_truth.sum()
    normalized_mean_hist = hist_mean / hist_mean.sum()
    normalized_generated_hist = hist_generated / hist_generated.sum()
    expected_rank = rank_histogram.sum() / len(rank_histogram)
    expected_coverage68 = finite_ensemble_interval_expectation(
        ensemble, 0.16, 0.84
    )
    expected_coverage95 = finite_ensemble_interval_expectation(
        ensemble, 0.025, 0.975
    )
    measured_coverage68 = coverage68_hits / coverage_voxels
    measured_coverage95 = coverage95_hits / coverage_voxels
    result = {
        "label": label,
        "path": str(path),
        "objects": objects,
        "ensemble_per_object": ensemble,
        "source_indices": source_indices.tolist(),
        "residual_operator": residual_operator,
        "x0_clip_training_residual_rms": x0_clip,
        "low_k_preservation": {
            "maximum_abs_cube_mean_generated_residual": residual_mean_max_abs,
            "diagnostic_k_h_mpc": k_low,
            "power_fraction_below_diagnostic_k": low_k_power
            / total_residual_power,
        },
        "fourier_log_density": {
            "generated_total_power_over_truth": band_summary(
                binner.k, binner.count, total_power_ratio
            ),
            "deterministic_total_power_over_truth": band_summary(
                binner.k, binner.count, deterministic_power_ratio
            ),
            "generated_residual_power_over_truth_residual": band_summary(
                binner.k, binner.count, residual_power_ratio
            ),
            "ensemble_mean_cross_correlation_with_truth": band_summary(
                binner.k,
                binner.count,
                np.nanmean(ensemble_mean_cross_rows, axis=0),
            ),
        },
        "residual_calibration": {
            "generated_rms": math_sqrt_ratio(generated_residual_square, generated_residual_voxels),
            "truth_residual_rms": math_sqrt_ratio(
                truth_residual_square, truth_residual_voxels
            ),
            "generated_over_truth_rms": math_sqrt_ratio(generated_residual_square, generated_residual_voxels)
            / math_sqrt_ratio(truth_residual_square, truth_residual_voxels),
            "voxel_coverage_68": measured_coverage68,
            "voxel_coverage_95": measured_coverage95,
            "finite_ensemble_expected_coverage_68": expected_coverage68,
            "finite_ensemble_expected_coverage_95": expected_coverage95,
            "coverage_68_minus_finite_ensemble_expected": measured_coverage68
            - expected_coverage68,
            "coverage_95_minus_finite_ensemble_expected": measured_coverage95
            - expected_coverage95,
            "rank_histogram": rank_histogram.tolist(),
            "rank_histogram_total_variation_from_uniform": float(
                0.5 * np.sum(np.abs(rank_histogram - expected_rank)) / rank_histogram.sum()
            ),
        },
        "density_pdf": {
            "log10_density_total_variation_generated_truth": float(
                0.5 * np.abs(normalized_generated_hist - normalized_truth_hist).sum()
            ),
            "log10_density_total_variation_deterministic_truth": float(
                0.5 * np.abs(normalized_mean_hist - normalized_truth_hist).sum()
            ),
        },
        "two_point_cosmic_mean": {
            "generated_vs_truth_ks": two_point_ks(
                np.asarray(truth_2pcf), np.asarray(generated_2pcf), estimator.radius_mpc_h
            ),
            "deterministic_vs_truth_ks": two_point_ks(
                np.asarray(truth_2pcf), np.asarray(mean_2pcf), estimator.radius_mpc_h
            ),
        },
        "environment": {
            "truth": average_statistic(truth_environment),
            "deterministic": average_statistic(mean_environment),
            "generated": average_statistic(generated_environment),
        },
    }
    arrays = {
        "k": binner.k,
        "total_power_ratio": total_power_ratio,
        "deterministic_power_ratio": deterministic_power_ratio,
        "residual_power_ratio": residual_power_ratio,
        "pdf_centers": 0.5 * (histogram_edges[:-1] + histogram_edges[1:]),
        "pdf_truth": normalized_truth_hist,
        "pdf_deterministic": normalized_mean_hist,
        "pdf_generated": normalized_generated_hist,
        "rank_histogram": rank_histogram / rank_histogram.sum(),
    }
    return result, arrays


def math_sqrt_ratio(total: float, count: int) -> float:
    return float(np.sqrt(total / count))


def finite_ensemble_interval_expectation(
    ensemble: int, lower_quantile: float, upper_quantile: float
) -> float:
    """Expected rank coverage for linearly indexed empirical quantiles.

    For an exchangeable truth draw and ``ensemble`` predictive draws, the
    expected CDF locations of the interpolated order indices differ by
    ``(q_hi-q_lo)*(n-1)/(n+1)``.  This makes explicit why 16-member empirical
    16--84 and 2.5--97.5 percent intervals should not be compared directly
    with 0.68 and 0.95.  The rank histogram remains the distribution-free
    primary calibration diagnostic.
    """
    return (upper_quantile - lower_quantile) * (ensemble - 1) / (ensemble + 1)


def plot_results(path: Path, results: dict[str, Any], arrays: dict[str, dict[str, np.ndarray]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for label, value in arrays.items():
        axes[0, 0].plot(value["k"], value["total_power_ratio"], label=f"{label} generated")
        axes[0, 0].plot(value["k"], value["deterministic_power_ratio"], "--", label=f"{label} deterministic")
        axes[0, 1].plot(value["k"], value["residual_power_ratio"], label=label)
        axes[1, 0].plot(value["pdf_centers"], value["pdf_generated"], label=f"{label} generated")
        rank_x = np.arange(len(value["rank_histogram"]))
        axes[1, 1].plot(rank_x, value["rank_histogram"], "o-", label=label)
    first = next(iter(arrays.values()))
    axes[1, 0].plot(first["pdf_centers"], first["pdf_truth"], "k", label="truth")
    axes[1, 0].plot(first["pdf_centers"], first["pdf_deterministic"], "k--", label="deterministic")
    axes[0, 0].axhline(1.0, color="k", linewidth=1)
    axes[0, 1].axhline(1.0, color="k", linewidth=1)
    axes[0, 0].set(xscale="log", yscale="log", xlabel="k [h/Mpc]", ylabel="P/P_truth", title="Total log-density power")
    axes[0, 1].set(xscale="log", yscale="log", xlabel="k [h/Mpc]", ylabel="P_res/P_truth,res", title="High-pass residual power")
    axes[1, 0].set(yscale="log", xlabel="log10(rho/rho0)", ylabel="voxel probability", title="One-point density PDF")
    axes[1, 1].axhline(1.0 / len(first["rank_histogram"]), color="k", linewidth=1)
    axes[1, 1].set(xlabel="truth rank in ensemble", ylabel="fraction", title="Ensemble rank histogram")
    for axis in axes.ravel():
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--voxel-mpc-h", type=float, default=0.3125)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    reference_indices: list[int] | None = None
    for label, path in args.candidate:
        print(f"[evaluate] {label}: {path}", flush=True)
        result, candidate_arrays = evaluate_candidate(label, path, args.voxel_mpc_h)
        if reference_indices is None:
            reference_indices = result["source_indices"]
        elif result["source_indices"] != reference_indices:
            raise RuntimeError("candidate source indices differ")
        results[label] = result
        arrays[label] = candidate_arrays
        (args.out / f"metrics_{label}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    report = {
        "schema": "hong2021-stochastic-residual-ensemble-evaluation-v1",
        "voxel_mpc_h": args.voxel_mpc_h,
        "candidates": results,
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_results(args.out / "diagnostics.png", results, arrays)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
