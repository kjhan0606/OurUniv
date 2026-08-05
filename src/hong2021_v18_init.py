#!/usr/bin/env python
"""Frozen train-prior measurement and spectral initialization for V18-E6."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from hong2021_v14_multiscale import BAND_EDGES_H_MPC, fourier_band_masks
from hong2021_v14_residual_cache import STANDARDIZED_SCHEMA


SCHEMA = "hong2021-v18-prior-matched-spectral-initialization-v1"
MEASUREMENT_SCHEMA = "hong2021-v18-source-balanced-standardized-residual-band-variance-v1"
EXPECTED_GRID = 64
EXPECTED_VOXEL_MPC_H = 0.3125
EXPECTED_MODE_COUNTS = (146, 3596, 25296, 233105)
EXPECTED_NON_DC_MODES = 262143
EXPECTED_MODE_COUNTS_BY_GRID = {
    64: EXPECTED_MODE_COUNTS,
    80: (250, 6872, 49928, 454949),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_masks(masks: np.ndarray, grid: int) -> None:
    if masks.shape != (4, grid, grid, grid):
        raise ValueError("unexpected Fourier-band mask shape")
    expected_counts = EXPECTED_MODE_COUNTS_BY_GRID.get(grid)
    if expected_counts is None:
        raise ValueError("V18 initializer received an unregistered grid")
    counts = tuple(int(mask.sum()) for mask in masks)
    if counts != expected_counts or sum(counts) != grid**3 - 1:
        raise ValueError("Fourier-band mode counts differ from the frozen V18 grid")
    negative = (-np.arange(grid)) % grid
    for mask in masks:
        if not np.array_equal(mask, mask[np.ix_(negative, negative, negative)]):
            raise ValueError("Fourier-band mask is not conjugate symmetric")


def measure_band_mode_variances(
    sources: Mapping[str, Mapping[str, Any]],
    *,
    parseval_relative_tolerance: float = 1.0e-9,
    maximum_absolute_ortho_dc: float = 1.0e-6,
) -> dict[str, Any]:
    """Measure source-balanced per-mode variance from frozen train caches."""
    if tuple(sources) != ("TNG100", "SIMBA", "Swift"):
        raise ValueError("V18 measurement sources must be TNG100, SIMBA, Swift in order")
    masks = fourier_band_masks(EXPECTED_GRID, EXPECTED_VOXEL_MPC_H)
    _validate_masks(masks, EXPECTED_GRID)
    per_domain: dict[str, list[float]] = {}
    maximum_parseval_error = 0.0
    maximum_dc = 0.0
    for label, specification in sources.items():
        path = Path(str(specification["path"])).resolve()
        if sha256_file(path) != specification["sha256"]:
            raise ValueError(f"{label} train-cache SHA256 mismatch")
        expected_domain = str(specification["domain_attribute"])
        sums = np.zeros(4, dtype=np.float64)
        with h5py.File(path, "r") as handle:
            if str(handle.attrs.get("schema")) != STANDARDIZED_SCHEMA:
                raise ValueError(f"{label} has the wrong standardized-cache schema")
            if not bool(handle.attrs.get("complete", False)):
                raise ValueError(f"{label} standardized cache is incomplete")
            if str(handle.attrs.get("domain")) != expected_domain:
                raise ValueError(f"{label} cache domain attribute mismatch")
            if bool(handle.attrs.get("scale_prediction_uses_target", True)):
                raise ValueError(f"{label} scale prediction uses the target")
            if float(handle.attrs.get("voxel_mpc_h", -1)) != EXPECTED_VOXEL_MPC_H:
                raise ValueError(f"{label} cache voxel size mismatch")
            dataset = handle["standardized_residual"]
            expected_shape = (
                int(specification["objects"]), 1,
                EXPECTED_GRID, EXPECTED_GRID, EXPECTED_GRID,
            )
            if tuple(dataset.shape) != expected_shape:
                raise ValueError(f"{label} standardized-residual shape mismatch")
            for row in dataset:
                value = np.asarray(row[0], dtype=np.float64)
                if not np.isfinite(value).all():
                    raise ValueError(f"{label} cache contains non-finite residuals")
                spectrum = np.fft.fftn(value, norm="ortho")
                power = spectrum.real**2 + spectrum.imag**2
                spatial_power = np.square(value, dtype=np.float64).sum(dtype=np.float64)
                spectral_power = power.sum(dtype=np.float64)
                relative = abs(spectral_power - spatial_power) / max(
                    spatial_power, np.finfo(np.float64).tiny
                )
                maximum_parseval_error = max(maximum_parseval_error, float(relative))
                maximum_dc = max(maximum_dc, float(abs(spectrum[0, 0, 0])))
                sums += np.asarray(
                    [power[mask].mean(dtype=np.float64) for mask in masks],
                    dtype=np.float64,
                )
            per_domain[label] = [float(value) for value in sums / len(dataset)]
    if maximum_parseval_error > parseval_relative_tolerance:
        raise ValueError("V18 train-cache Parseval self-check failed")
    if maximum_dc > maximum_absolute_ortho_dc:
        raise ValueError("V18 train-cache DC self-check failed")
    balanced = np.mean(np.asarray(list(per_domain.values()), dtype=np.float64), axis=0)
    if not np.isfinite(balanced).all() or np.any(balanced <= 0):
        raise ValueError("V18 source-balanced variances are not finite and positive")
    return {
        "per_domain": per_domain,
        "source_balanced": [float(value) for value in balanced],
        "mode_counts": list(EXPECTED_MODE_COUNTS),
        "non_dc_modes": EXPECTED_NON_DC_MODES,
        "maximum_parseval_relative_error": maximum_parseval_error,
        "maximum_absolute_ortho_dc": maximum_dc,
    }


def prior_matched_spectral_std(
    grid: int,
    voxel_mpc_h: float,
    sigma_first_step: float,
    band_mode_variances: list[float] | tuple[float, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    if grid not in EXPECTED_MODE_COUNTS_BY_GRID or voxel_mpc_h != EXPECTED_VOXEL_MPC_H:
        raise ValueError("V18 initializer requires a frozen 64^3 or 80^3 grid")
    variance = np.asarray(band_mode_variances, dtype=np.float64)
    if variance.shape != (4,) or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("V18 band variances must be four finite positive values")
    if not np.isfinite(sigma_first_step) or sigma_first_step <= 0:
        raise ValueError("V18 first sigma must be finite and positive")
    masks = fourier_band_masks(grid, voxel_mpc_h)
    _validate_masks(masks, grid)
    spectral_std = np.full((grid, grid, grid), np.nan, dtype=np.float64)
    spectral_std[0, 0, 0] = sigma_first_step
    for mask, value in zip(masks, variance, strict=True):
        spectral_std[mask] = np.sqrt(sigma_first_step**2 + value)
    if not np.isfinite(spectral_std).all():
        raise ValueError("V18 spectral standard deviation left unassigned modes")
    negative = (-np.arange(grid)) % grid
    if not np.array_equal(
        spectral_std, spectral_std[np.ix_(negative, negative, negative)]
    ):
        raise ValueError("V18 spectral standard deviation is not conjugate symmetric")
    return torch.as_tensor(spectral_std, dtype=torch.float64, device=device)


def prior_matched_init(
    noise: torch.Tensor, spectral_std: torch.Tensor
) -> tuple[torch.Tensor, float]:
    if noise.ndim != 5 or noise.shape[1] != 1 or len(set(noise.shape[-3:])) != 1:
        raise ValueError("V18 initializer expects (batch,1,N,N,N) noise")
    if spectral_std.shape != noise.shape[-3:] or spectral_std.dtype != torch.float64:
        raise ValueError("V18 spectral standard deviation shape or dtype mismatch")
    if spectral_std.device != noise.device or not torch.isfinite(spectral_std).all():
        raise ValueError("V18 spectral standard deviation device or values mismatch")
    with torch.autocast(device_type=noise.device.type, enabled=False):
        spectrum = torch.fft.fftn(noise.double(), dim=(-3, -2, -1), norm="ortho")
        initialized = torch.fft.ifftn(
            spectrum * spectral_std[None, None], dim=(-3, -2, -1), norm="ortho"
        )
        real = initialized.real
        real_rms = torch.sqrt(torch.mean(real.square()))
        if not torch.isfinite(real_rms) or float(real_rms) <= 0:
            raise ValueError("V18 initialized field has invalid real RMS")
        imaginary_ratio = float(initialized.imag.abs().max() / real_rms)
    if not np.isfinite(imaginary_ratio):
        raise ValueError("V18 initialized field has invalid imaginary leakage")
    return real.to(dtype=noise.dtype), imaginary_ratio


class PriorMatchedInitializer:
    def __init__(
        self, spectral_std: torch.Tensor, *, maximum_imaginary_ratio: float
    ) -> None:
        self.spectral_std = spectral_std
        self.maximum_imaginary_ratio = float(maximum_imaginary_ratio)
        self.maximum_observed_imaginary_ratio = 0.0

    def __call__(self, noise: torch.Tensor) -> torch.Tensor:
        initialized, ratio = prior_matched_init(noise, self.spectral_std)
        self.maximum_observed_imaginary_ratio = max(
            self.maximum_observed_imaginary_ratio, ratio
        )
        if ratio > self.maximum_imaginary_ratio:
            raise ValueError("V18 initialization imaginary leakage exceeds tolerance")
        return initialized


def _parse_source(value: str) -> tuple[str, dict[str, Any]]:
    fields = value.split("=", 1)
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("source must be LABEL=PATH")
    label, path = fields
    domain = {"TNG100": "TNG100", "SIMBA": "SIMBA", "Swift": "Swift-EAGLE"}.get(label)
    if domain is None:
        raise argparse.ArgumentTypeError("source label must be TNG100, SIMBA, or Swift")
    with h5py.File(path, "r") as handle:
        objects = int(handle["standardized_residual"].shape[0])
    return label, {
        "path": str(Path(path).resolve()), "sha256": sha256_file(path),
        "objects": objects, "domain_attribute": domain,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_parse_source, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sources = dict(args.source)
    measured = measure_band_mode_variances(sources)
    report = {
        "schema": MEASUREMENT_SCHEMA,
        "sources": sources,
        **measured,
        # Keep the loader-facing name used by the frozen, independently
        # audited measurement record as well as the concise function key.
        "source_balanced_per_band_mode_variance": measured["source_balanced"],
    }
    if args.out.exists() or args.out.with_suffix(args.out.suffix + ".partial").exists():
        raise RuntimeError(f"refusing to overwrite V18 measurement: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
