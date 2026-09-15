#!/usr/bin/env python
"""Train-only diagnostic for the low-pass residual omitted by V6--V9."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d


def source_bin_fit(
    data_path: Path, cache_path: Path, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    count = np.zeros(len(edges) - 1, dtype=np.int64)
    total = np.zeros(len(edges) - 1, dtype=np.float64)
    with h5py.File(data_path, "r") as data, h5py.File(cache_path, "r") as cache:
        for index in range(len(data["target"])):
            truth = np.asarray(data["target"][index, 0], dtype=np.float32)
            mean = np.asarray(cache["conditional_mean"][index, 0], dtype=np.float32)
            laplacian = np.asarray(cache["laplacian_residual"][index, 0], dtype=np.float32)
            omitted = truth - mean - laplacian
            bin_index = np.clip(np.digitize(mean, edges) - 1, 0, len(count) - 1)
            count += np.bincount(bin_index.ravel(), minlength=len(count))
            total += np.bincount(
                bin_index.ravel(), weights=omitted.ravel(), minlength=len(count)
            )
    return count, total


def equal_source_bias(
    source_rows: list[tuple[np.ndarray, np.ndarray]], smooth_sigma_bins: float
) -> np.ndarray:
    means = []
    valid = []
    for count, total in source_rows:
        means.append(np.divide(total, count, out=np.zeros_like(total), where=count > 0))
        valid.append(count > 0)
    means_array = np.asarray(means)
    valid_array = np.asarray(valid)
    source_count = valid_array.sum(axis=0)
    combined = np.divide(
        (means_array * valid_array).sum(axis=0),
        source_count,
        out=np.zeros(means_array.shape[1]),
        where=source_count > 0,
    )
    populated = np.flatnonzero(source_count > 0)
    if len(populated) < 2:
        raise ValueError("not enough populated bins")
    missing = source_count == 0
    combined[missing] = np.interp(
        np.flatnonzero(missing), populated, combined[populated]
    )
    return gaussian_filter1d(combined, smooth_sigma_bins, mode="nearest")


def transform_ensemble(
    source_path: Path, output_path: Path, edges: np.ndarray, bias: np.ndarray
) -> None:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.exists() or temporary.exists():
        raise SystemExit(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(source_path, "r") as source, h5py.File(temporary, "w") as out:
            for key in ("source_index", "conditional_mean", "truth"):
                source.copy(key, out)
            generated_source = source["sample"]
            generated_out = out.create_dataset(
                "sample", shape=generated_source.shape, dtype="f4",
                chunks=generated_source.chunks, compression="lzf",
            )
            for object_index in range(len(generated_source)):
                mean = np.asarray(source["conditional_mean"][object_index, 0])
                index = np.clip(np.digitize(mean, edges) - 1, 0, len(bias) - 1)
                correction = bias[index].astype(np.float32)
                correction -= correction.mean(dtype=np.float64)
                generated_out[object_index] = (
                    np.asarray(generated_source[object_index], dtype=np.float32)
                    + correction[None, None]
                )
            for key, value in source.attrs.items():
                out.attrs[key] = value
            out.attrs["schema"] = str(source.attrs["schema"])
            out.attrs["train_only_lowpass_bias_diagnostic"] = True
            out.attrs["source_ensemble"] = str(source_path.resolve())
            out.attrs["complete"] = True
        os.replace(temporary, output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tng-data", type=Path, required=True)
    parser.add_argument("--tng-cache", type=Path, required=True)
    parser.add_argument("--simba-data", type=Path, required=True)
    parser.add_argument("--simba-cache", type=Path, required=True)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--minimum", type=float, default=-0.4)
    parser.add_argument("--maximum", type=float, default=1.0)
    parser.add_argument("--smooth-sigma-bins", type=float, default=2.0)
    args = parser.parse_args()
    edges = np.linspace(args.minimum, args.maximum, args.bins + 1)
    rows = [
        source_bin_fit(args.tng_data, args.tng_cache, edges),
        source_bin_fit(args.simba_data, args.simba_cache, edges),
    ]
    bias = equal_source_bias(rows, args.smooth_sigma_bins)
    transform_ensemble(args.ensemble, args.out, edges, bias)
    report = {
        "schema": "hong2021-train-only-lowpass-bias-diagnostic-v1",
        "uses_validation_truth_for_fit": False,
        "uses_historical_simba_cv0_15": False,
        "uses_eagle": False,
        "fit_sources": [
            {"data": str(args.tng_data.resolve()), "cache": str(args.tng_cache.resolve())},
            {"data": str(args.simba_data.resolve()), "cache": str(args.simba_cache.resolve())},
        ],
        "equal_source_weighting": True,
        "edges": edges.tolist(),
        "counts": [row[0].tolist() for row in rows],
        "smoothed_bias": bias.tolist(),
        "exact_cube_dc_projection": True,
        "source_ensemble": str(args.ensemble.resolve()),
        "output_ensemble": str(args.out.resolve()),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
