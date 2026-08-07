#!/usr/bin/env python
"""Sample the development-only V26 conditional Gaussian-copula control.

This is a mechanism audit, not a frozen V26 candidate.  It removes the EDM
trajectory while retaining the train-only V14 conditional location/band-scale
map and the V21 voxel-conditional marginal map.  The resulting distribution is
an explicit conditional Gaussian-copula likelihood in the non-DC subspace.

Astrid and historical EAGLE paths are intentionally absent from this module.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from hong2021_residual_v12_gaussianized import inverse_gaussianize_torch
from hong2021_v14_edm import ENSEMBLE_SCHEMA, V14ResidualDataset
from hong2021_v14_multiscale import fourier_band_masks, inverse_standardized_residual
from hong2021_v18_edm import _indices
from hong2021_v18_init import sha256_file
from hong2021_v21_conditional_affine import invert_profile_torch


SCHEMA = "hong2021-v26-development-gaussian-copula-control-v1"
DOMAIN_KEYS = {"TNG100": "TNG100", "SIMBA": "SIMBA", "Swift": "Swift"}
CACHE_KEYS = {
    "TNG100": "TNG100_validation",
    "SIMBA": "SIMBA_validation",
    "Swift": "Swift_validation",
}


def latent_spectral_standard_deviation(
    grid: int,
    voxel_mpc_h: float,
    band_mode_variances: list[float] | tuple[float, ...],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return the exact non-DC latent Gaussian standard deviation by mode."""
    variance = np.asarray(band_mode_variances, dtype=np.float64)
    masks = fourier_band_masks(grid, voxel_mpc_h)
    if variance.shape != (len(masks),):
        raise ValueError("one latent variance is required per Fourier band")
    if not np.isfinite(variance).all() or np.any(variance <= 0.0):
        raise ValueError("latent band variances must be finite and positive")
    result = np.zeros((grid, grid, grid), dtype=np.float64)
    for mask, value in zip(masks, variance, strict=True):
        result[mask] = np.sqrt(value)
    if result[0, 0, 0] != 0.0 or np.count_nonzero(result) != grid**3 - 1:
        raise RuntimeError("latent Gaussian control has an invalid DC subspace")
    negative = (-np.arange(grid)) % grid
    if not np.array_equal(result, result[np.ix_(negative, negative, negative)]):
        raise RuntimeError("latent Gaussian spectral scale is not Hermitian-symmetric")
    return torch.as_tensor(result, dtype=torch.float64, device=device)


def sample_latent_gaussian(
    *,
    ensemble: int,
    grid: int,
    spectral_std: torch.Tensor,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Draw a real, exactly centered GRF without changing the RNG draw count."""
    if ensemble <= 0 or spectral_std.shape != (grid, grid, grid):
        raise ValueError("invalid Gaussian-control ensemble specification")
    noise = torch.randn(
        (ensemble, 1, grid, grid, grid),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    with torch.autocast(device_type=device.type, enabled=False):
        spectrum = torch.fft.fftn(
            noise.double(), dim=(-3, -2, -1), norm="ortho"
        )
        complex_field = torch.fft.ifftn(
            spectrum * spectral_std[None, None],
            dim=(-3, -2, -1),
            norm="ortho",
        )
        real = complex_field.real
        rms = torch.sqrt(torch.mean(real.square())).clamp_min(
            torch.finfo(torch.float64).tiny
        )
        imaginary_ratio = float(complex_field.imag.abs().max() / rms)
        real -= real.mean(dim=(-3, -2, -1), keepdim=True)
    result = real.float()
    if not torch.isfinite(result).all():
        raise RuntimeError("latent Gaussian control produced non-finite values")
    return result, imaginary_ratio


def _load_inputs(
    repo: Path,
    domain: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[int]]:
    artifacts_path = repo / "config/hong2021_v21_derived_artifacts.json"
    registry_path = repo / "config/hong2021_v20_development_program.json"
    artifacts = json.loads(artifacts_path.read_text())
    registry = json.loads(registry_path.read_text())
    experiment = registry["e8_gaussianized_marginal_retrain"]
    source = DOMAIN_KEYS[domain]
    inputs = experiment["data"][source]
    data = inputs["validation_data"]
    cache = artifacts["caches"][CACHE_KEYS[domain]]
    for row in (data, cache, artifacts["profile"], artifacts["gaussianization"]):
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"frozen V21 input hash mismatch: {row['path']}")
    indices = _indices(experiment["development_objects"][source], repo)
    return artifacts, data, cache, indices


@torch.inference_mode()
def write_control(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    artifacts, data, cache, indices = _load_inputs(repo, args.domain)
    output = args.out.resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite Gaussian-control ensemble: {output}")
    dataset = V14ResidualDataset(data["path"], cache["path"], False)
    if dataset.grid != 64 or dataset.voxel_mpc_h != 0.3125:
        raise ValueError("V26 Gaussian control requires the frozen 64^3 V21 grid")
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    transform = json.loads(Path(artifacts["gaussianization"]["path"]).read_text())
    device = torch.device(args.device)
    spectral_std = latent_spectral_standard_deviation(
        dataset.grid,
        dataset.voxel_mpc_h,
        artifacts["initialization"]["source_balanced_band_mode_variance"],
        device=device,
    )
    centers = torch.as_tensor(profile["centers"], dtype=torch.float64, device=device)
    mu = torch.as_tensor(profile["mu"], dtype=torch.float64, device=device)
    log_sigma = torch.as_tensor(
        profile["log_sigma"], dtype=torch.float64, device=device
    )
    z_knots = torch.as_tensor(
        transform["z_knots"], dtype=torch.float32, device=device
    )
    residual_knots = torch.as_tensor(
        transform["residual_value_knots"], dtype=torch.float32, device=device
    )
    seed = int(args.seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    maximum_imaginary_ratio = 0.0
    try:
        with h5py.File(partial, "w") as handle:
            field_shape = (len(indices), 1, dataset.grid, dataset.grid, dataset.grid)
            sample_shape = (
                len(indices), args.ensemble, 1,
                dataset.grid, dataset.grid, dataset.grid,
            )
            sample_ds = handle.create_dataset(
                "sample", shape=sample_shape, dtype="f4",
                chunks=(1, 1, 1, dataset.grid, dataset.grid, dataset.grid),
                compression="lzf",
            )
            mean_ds = handle.create_dataset(
                "conditional_mean", shape=field_shape, dtype="f4", compression="lzf"
            )
            truth_ds = handle.create_dataset(
                "truth", shape=field_shape, dtype="f4", compression="lzf"
            )
            handle.create_dataset(
                "source_index", data=np.asarray(indices, dtype=np.int64)
            )
            location_ds = handle.create_dataset(
                "predicted_residual_dc", shape=(len(indices),), dtype="f4"
            )
            scale_ds = handle.create_dataset(
                "predicted_band_scales", shape=(len(indices), 4), dtype="f4"
            )
            for output_index, data_index in enumerate(indices):
                _, _, corrected_mean, truth = dataset[data_index]
                location, scales = dataset.predicted_location_scales(data_index)
                latent, imaginary_ratio = sample_latent_gaussian(
                    ensemble=args.ensemble,
                    grid=dataset.grid,
                    spectral_std=spectral_std,
                    generator=generator,
                    device=device,
                )
                maximum_imaginary_ratio = max(
                    maximum_imaginary_ratio, imaginary_ratio
                )
                u = inverse_gaussianize_torch(latent, z_knots, residual_knots)
                mean_batch = corrected_mean[None].to(device).expand(
                    args.ensemble, -1, -1, -1, -1
                )
                standardized = invert_profile_torch(
                    u, mean_batch, centers, mu, log_sigma
                )
                standardized_numpy = standardized[:, 0].float().cpu().numpy()
                physical = np.stack(
                    [
                        inverse_standardized_residual(
                            value,
                            predicted_location=location,
                            predicted_scales=scales,
                            voxel_mpc_h=dataset.voxel_mpc_h,
                        )
                        for value in standardized_numpy
                    ]
                ).astype(np.float32)
                mean_with_location = corrected_mean.numpy() + np.float32(location)
                sample_ds[output_index, :, 0] = corrected_mean.numpy()[0] + physical
                mean_ds[output_index] = mean_with_location
                truth_ds[output_index] = truth.numpy()
                location_ds[output_index] = location
                scale_ds[output_index] = scales
                print(
                    f"[sample] Gaussian control {args.domain} "
                    f"{output_index + 1}/{len(indices)}",
                    flush=True,
                )
            handle.attrs.update(
                {
                    "schema": ENSEMBLE_SCHEMA,
                    "method": "conditional_gaussian_copula_control",
                    "mechanism_audit_schema": SCHEMA,
                    "source_cache": str(Path(cache["path"]).resolve()),
                    "source_cache_sha256": cache["sha256"],
                    "source_data_sha256": data["sha256"],
                    "v21_profile_sha256": artifacts["profile"]["sha256"],
                    "v21_gaussianization_sha256": artifacts["gaussianization"]["sha256"],
                    "latent_band_mode_variance_json": json.dumps(
                        artifacts["initialization"][
                            "source_balanced_band_mode_variance"
                        ]
                    ),
                    "latent_distribution": "zero-DC multiband Gaussian random field",
                    "conditional_path": (
                        "observable-predicted V14 mean/location/band scales plus "
                        "V21 voxel-conditional inverse marginal"
                    ),
                    "ensemble_members": args.ensemble,
                    "seed": seed,
                    "diagnostic_k_h_mpc": 1.0,
                    "location_scale_uses_target": False,
                    "maximum_imaginary_over_real_rms": maximum_imaginary_ratio,
                    "Astrid_accessed": False,
                    "historical_EAGLE_accessed": False,
                    "complete": True,
                }
            )
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    print(f"[out] {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--domain", choices=tuple(DOMAIN_KEYS), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ensemble", type=int, default=16)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    write_control(build_parser().parse_args())
