#!/usr/bin/env python
"""Measure cross-code transfer and noise of a frozen density predictor."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCHEMA = "hong2021-density-likelihood-cross-code-e1-v1"
STAT_NAMES = ("truth_auto", "mean_truth_cross", "mean_auto")


@dataclass(frozen=True)
class Source:
    label: str
    data: Path
    mean: Path
    grouping: str


def tukey(size: int, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Tukey alpha must be in [0, 1]")
    if alpha == 0.0:
        return np.ones(size, dtype=np.float64)
    x = np.linspace(0.0, 1.0, size)
    value = np.ones(size, dtype=np.float64)
    left = x < alpha / 2.0
    right = x > 1.0 - alpha / 2.0
    value[left] = 0.5 * (
        1.0 + np.cos(np.pi * (2.0 * x[left] / alpha - 1.0))
    )
    value[right] = 0.5 * (
        1.0
        + np.cos(np.pi * (2.0 * x[right] / alpha - 2.0 / alpha + 1.0))
    )
    return value


def windows(
    grid: int, voxel: float, alpha: float
) -> tuple[list[str], np.ndarray]:
    one = tukey(grid, alpha)
    edge = one[:, None, None] * one[None, :, None] * one[None, None, :]
    coordinate = (np.arange(grid) + 0.5 - grid / 2.0) * voxel
    radius = np.sqrt(
        coordinate[:, None, None] ** 2
        + coordinate[None, :, None] ** 2
        + coordinate[None, None, :] ** 2
    )
    names = ["global", "inner_0_5", "middle_5_10", "outer_10_plus"]
    masks = [
        np.ones_like(radius, dtype=bool),
        radius < 5.0,
        (radius >= 5.0) & (radius < 10.0),
        radius >= 10.0,
    ]
    result = np.stack([edge * mask for mask in masks])
    if np.any(result.sum(axis=(1, 2, 3)) <= 0):
        raise ValueError("an analysis window is empty")
    return names, result


def k_geometry(
    grid: int, voxel: float, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kx = 2.0 * np.pi * np.fft.fftfreq(grid, d=voxel)
    kz = 2.0 * np.pi * np.fft.rfftfreq(grid, d=voxel)
    magnitude = np.sqrt(
        kx[:, None, None] ** 2 + kx[None, :, None] ** 2 + kz[None, None, :] ** 2
    )
    index = np.digitize(magnitude.ravel(), edges) - 1
    valid = (index >= 0) & (index < len(edges) - 1)
    counts = np.bincount(index[valid], minlength=len(edges) - 1)
    mean_k = np.bincount(
        index[valid], weights=magnitude.ravel()[valid], minlength=len(edges) - 1
    ) / counts
    return index, valid, np.column_stack((mean_k, counts))


def binned_spectra(
    truth: np.ndarray,
    mean: np.ndarray,
    analysis_windows: np.ndarray,
    bin_index: np.ndarray,
    valid_modes: np.ndarray,
    bins: int,
) -> np.ndarray:
    result = np.empty((len(analysis_windows), bins, 3), dtype=np.float64)
    for region, window in enumerate(analysis_windows):
        weight = float(window.sum())
        truth_centered = truth - float(np.sum(window * truth) / weight)
        mean_centered = mean - float(np.sum(window * mean) / weight)
        normalization = np.sqrt(float(np.sum(window * window)))
        truth_fft = np.fft.rfftn(window * truth_centered).ravel() / normalization
        mean_fft = np.fft.rfftn(window * mean_centered).ravel() / normalization
        mode_values = (
            np.abs(truth_fft) ** 2,
            np.real(mean_fft * np.conj(truth_fft)),
            np.abs(mean_fft) ** 2,
        )
        for column, values in enumerate(mode_values):
            result[region, :, column] = np.bincount(
                bin_index[valid_modes],
                weights=values[valid_modes],
                minlength=bins,
            )
    return result


def source_group_spectra(
    entries: list[Source],
    analysis_windows: np.ndarray,
    bin_index: np.ndarray,
    valid_modes: np.ndarray,
    bins: int,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    grouped_sum: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((len(analysis_windows), bins, 3), dtype=np.float64)
    )
    grouped_count: dict[str, int] = defaultdict(int)
    files: list[dict[str, Any]] = []
    for entry in entries:
        with h5py.File(entry.data, "r") as data, h5py.File(entry.mean, "r") as means:
            if len(data["target"]) != len(means["conditional_mean"]):
                raise ValueError(f"sample mismatch: {entry.data} and {entry.mean}")
            if entry.grouping == "realization":
                if "realization" not in data:
                    raise ValueError(f"realization dataset absent: {entry.data}")
                groups = np.asarray(data["realization"], dtype=np.int64)
            elif entry.grouping == "sample":
                groups = np.arange(len(data["target"]), dtype=np.int64)
            else:
                raise ValueError(f"unknown grouping: {entry.grouping}")
            files.append(
                {
                    "data": str(entry.data.resolve()),
                    "mean": str(entry.mean.resolve()),
                    "samples": len(groups),
                    "grouping": entry.grouping,
                }
            )
            prefix = entry.data.stem
            for sample, group in enumerate(groups):
                truth = np.asarray(data["target"][sample, 0], dtype=np.float64)
                prediction = np.asarray(
                    means["conditional_mean"][sample, 0], dtype=np.float64
                )
                key = f"{prefix}:{int(group)}"
                grouped_sum[key] += binned_spectra(
                    truth, prediction, analysis_windows,
                    bin_index, valid_modes, bins,
                )
                grouped_count[key] += 1
                if (sample + 1) % 25 == 0 or sample + 1 == len(groups):
                    print(
                        f"[spectra] {entry.label} {entry.data.name} "
                        f"{sample + 1}/{len(groups)}",
                        flush=True,
                    )
    keys = sorted(grouped_sum)
    spectra = np.stack([grouped_sum[key] / grouped_count[key] for key in keys])
    metadata = {
        "files": files,
        "groups": len(keys),
        "samples": int(sum(grouped_count.values())),
        "samples_per_group": {key: grouped_count[key] for key in keys},
    }
    return spectra, keys, metadata


def derived(statistics: np.ndarray) -> dict[str, np.ndarray]:
    truth = statistics[..., 0]
    cross = statistics[..., 1]
    mean = statistics[..., 2]
    transfer = cross / truth
    noise = mean - cross * cross / truth
    coherence = cross * cross / (truth * mean)
    return {
        "truth_power": truth,
        "cross_power": cross,
        "mean_power": mean,
        "transfer": transfer,
        "noise_power": noise,
        "noise_over_truth": noise / truth,
        "coherence_squared": coherence,
    }


def bootstrap_statistics(
    spectra: np.ndarray, resamples: int, generator: np.random.Generator
) -> np.ndarray:
    groups = len(spectra)
    weights = generator.multinomial(
        groups, np.full(groups, 1.0 / groups), size=resamples
    ).astype(np.float64)
    weights /= groups
    return np.einsum("bg,grks->brks", weights, spectra, optimize=True)


def serializable_metrics(values: dict[str, np.ndarray]) -> dict[str, Any]:
    return {key: value.tolist() for key, value in values.items()}


def plot_report(
    path: Path,
    k: np.ndarray,
    region_names: list[str],
    source_values: dict[str, dict[str, np.ndarray]],
    transfer_ratio: np.ndarray,
    noise_ratio: np.ndarray,
) -> None:
    colors = {"tng": "#2166ac", "simba": "#b2182b"}
    figure, axes = plt.subplots(3, len(region_names), figsize=(16, 10), sharex=True)
    for column, region in enumerate(region_names):
        for label, values in source_values.items():
            axes[0, column].plot(
                k, values["transfer"][column], marker="o", color=colors[label],
                label=label.upper(),
            )
            axes[1, column].plot(
                k, values["noise_over_truth"][column], marker="o",
                color=colors[label], label=label.upper(),
            )
        axes[2, column].plot(k, transfer_ratio[column], marker="o", label="T ratio")
        axes[2, column].plot(k, noise_ratio[column], marker="s", label="Pn ratio")
        axes[0, column].axhline(1.0, color="0.5", lw=1)
        axes[2, column].axhline(1.0, color="0.5", lw=1)
        axes[2, column].axhspan(0.7, 1.3, color="#92c5de", alpha=0.2)
        axes[2, column].axhline(2.0, color="#b2182b", ls="--", lw=1)
        axes[0, column].set_title(region)
        axes[2, column].set_xlabel(r"$k\ [h\,\mathrm{Mpc}^{-1}]$")
        for row in range(3):
            axes[row, column].set_xscale("log")
            axes[row, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel(r"$T_\mathrm{eff}$")
    axes[1, 0].set_ylabel(r"$P_n/P_\delta$")
    axes[2, 0].set_ylabel("SIMBA / TNG")
    axes[0, 0].legend()
    axes[2, 0].legend()
    figure.suptitle("Frozen density-likelihood cross-code transfer (development only)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def parse_source(values: list[str]) -> Source:
    label, data, mean, grouping = values
    return Source(label, Path(data), Path(mean), grouping)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", nargs=4, metavar=("LABEL", "DATA", "MEAN", "GROUPING"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--k-edges", default="0.3,0.6,1.0,1.5,2.0,3.0,4.0,6.0,8.0,10.1"
    )
    parser.add_argument("--tukey-alpha", type=float, default=0.25)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=482021)
    parser.add_argument("--max-transfer-drift", type=float, default=0.3)
    parser.add_argument("--max-noise-inflation", type=float, default=2.0)
    args = parser.parse_args()

    sources = [parse_source(value) for value in args.source]
    labels = {source.label for source in sources}
    if labels != {"tng", "simba"}:
        raise ValueError("sources must use exactly the labels 'tng' and 'simba'")
    first = sources[0]
    with h5py.File(first.data, "r") as handle:
        grid = int(handle["target"].shape[-1])
        voxel = float(handle.attrs["voxel_mpc_h"])
    edges = np.asarray([float(value) for value in args.k_edges.split(",")])
    if not np.all(np.diff(edges) > 0):
        raise ValueError("k edges must be strictly increasing")
    region_names, analysis_windows = windows(grid, voxel, args.tukey_alpha)
    bin_index, valid_modes, geometry = k_geometry(grid, voxel, edges)
    if np.any(geometry[:, 1] == 0):
        raise ValueError("a k bin has no Fourier modes")

    by_label: dict[str, list[Source]] = {
        label: [source for source in sources if source.label == label]
        for label in sorted(labels)
    }
    spectra: dict[str, np.ndarray] = {}
    source_metadata: dict[str, Any] = {}
    for label, entries in by_label.items():
        spectra[label], _, source_metadata[label] = source_group_spectra(
            entries, analysis_windows, bin_index, valid_modes, len(edges) - 1
        )
    point_stats = {label: value.mean(axis=0) for label, value in spectra.items()}
    point = {label: derived(value) for label, value in point_stats.items()}
    transfer_ratio = point["simba"]["transfer"] / point["tng"]["transfer"]
    noise_ratio = point["simba"]["noise_power"] / point["tng"]["noise_power"]

    generator = np.random.default_rng(args.seed)
    boot = {
        label: derived(bootstrap_statistics(value, args.bootstrap, generator))
        for label, value in spectra.items()
    }
    boot_transfer_ratio = boot["simba"]["transfer"] / boot["tng"]["transfer"]
    boot_noise_ratio = boot["simba"]["noise_power"] / boot["tng"]["noise_power"]
    intervals = {
        "transfer_ratio_simba_over_tng_95": np.quantile(
            boot_transfer_ratio, [0.025, 0.975], axis=0
        ).transpose(1, 2, 0),
        "noise_power_ratio_simba_over_tng_95": np.quantile(
            boot_noise_ratio, [0.025, 0.975], axis=0
        ).transpose(1, 2, 0),
    }

    primary = [
        index for index in range(len(edges) - 1)
        if edges[index] >= 0.6 and edges[index + 1] <= 2.0
    ]
    global_index = region_names.index("global")
    checks_by_bin: list[dict[str, Any]] = []
    for index in primary:
        transfer = float(transfer_ratio[global_index, index])
        noise = float(noise_ratio[global_index, index])
        positive_noise = bool(
            point["tng"]["noise_power"][global_index, index] > 0
            and point["simba"]["noise_power"][global_index, index] > 0
        )
        checks_by_bin.append(
            {
                "k_low_h_mpc": float(edges[index]),
                "k_high_h_mpc": float(edges[index + 1]),
                "transfer_ratio_simba_over_tng": transfer,
                "noise_power_ratio_simba_over_tng": noise,
                "positive_noise_power": positive_noise,
                "transfer_pass": abs(transfer - 1.0) <= args.max_transfer_drift,
                "noise_pass": positive_noise and noise <= args.max_noise_inflation,
            }
        )
    gate_pass = all(
        row["transfer_pass"] and row["noise_pass"] for row in checks_by_bin
    )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "locked_test_used": False,
        "grid": grid,
        "voxel_mpc_h": voxel,
        "field": "log10(rho/rho_bar)/4.5",
        "edge_window": {"kind": "separable_tukey", "alpha": args.tukey_alpha},
        "regions": region_names,
        "k_edges_h_mpc": edges.tolist(),
        "k_mean_h_mpc": geometry[:, 0].tolist(),
        "mode_count": geometry[:, 1].astype(int).tolist(),
        "source_metadata": source_metadata,
        "source_metrics": {
            label: serializable_metrics(value) for label, value in point.items()
        },
        "cross_code": {
            "transfer_ratio_simba_over_tng": transfer_ratio.tolist(),
            "noise_power_ratio_simba_over_tng": noise_ratio.tolist(),
            **serializable_metrics(intervals),
        },
        "bootstrap": {
            "resamples": args.bootstrap,
            "seed": args.seed,
            "tng_note": "observer-resampling only; all cubes share TNG100",
            "simba_note": "independent-realization resampling after within-realization averaging",
        },
        "frozen_gate": {
            "region": "global",
            "primary_k_range_h_mpc": [0.6, 2.0],
            "maximum_abs_transfer_ratio_minus_one": args.max_transfer_drift,
            "maximum_simba_over_tng_absolute_noise_power": args.max_noise_inflation,
            "checks_by_bin": checks_by_bin,
            "pass": gate_pass,
            "decision": (
                "advance frozen deterministic mean to TNG mock likelihood E2"
                if gate_pass
                else "reject frozen deterministic mean as a cross-code IC density likelihood"
            ),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "transfer_likelihood.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    plot_report(
        args.out / "transfer_likelihood.png", geometry[:, 0], region_names,
        point, transfer_ratio, noise_ratio,
    )
    print(json.dumps(report["frozen_gate"], indent=2), flush=True)
    print(f"[out] {output}", flush=True)


if __name__ == "__main__":
    main()
