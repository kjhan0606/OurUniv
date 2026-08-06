#!/usr/bin/env python
"""V21 voxel-conditional affine profile and Gaussianization primitives.

The profile is fitted only from the three frozen training source caches.  This
module intentionally contains no sampling or Astrid/EAGLE access.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from hong2021_v20_gaussianize import fit_knots
from hong2021_residual_v12_gaussianized import gaussianize_numpy, inverse_gaussianize_numpy


DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
PROFILE_SCHEMA = "hong2021-v21-voxel-conditional-affine-profile-v1"
TRANSFORM_SCHEMA = "hong2021-v21-equal-source-conditional-affine-gaussianization-v1"


def _rows(path: Path):
    with h5py.File(path, "r") as handle:
        for mean, residual in zip(handle["conditional_mean"], handle["standardized_residual"]):
            yield np.asarray(mean[0], dtype=np.float32), np.asarray(residual[0], dtype=np.float32)


def fit_profile(paths: Mapping[str, Path], *, bins: int = 10, floor: float = 0.25) -> dict:
    """Fit frozen equal-source voxel profile from train caches."""
    if tuple(paths) != DOMAIN_ORDER or bins != 10 or floor != 0.25:
        raise ValueError("V21 profile requires the frozen three-source 10-bin specification")
    means = []
    for domain in DOMAIN_ORDER:
        with h5py.File(paths[domain], "r") as handle:
            means.append(np.asarray(handle["conditional_mean"][:, 0], dtype=np.float32).reshape(-1))
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    # Equal-source pooled quantiles: average each source quantile curve, rather
    # than weighting a source by its number of voxels.
    source_quantiles = np.asarray(
        [np.quantile(value.astype(np.float64), quantiles, method="linear") for value in means]
    )
    edges = source_quantiles.mean(axis=0)
    edges[0] = float(np.min(source_quantiles[:, 0])); edges[-1] = float(np.max(source_quantiles[:, -1]))
    source_means = np.zeros((3, bins), dtype=np.float64)
    source_sigmas = np.zeros((3, bins), dtype=np.float64)
    counts = np.zeros((3, bins), dtype=np.int64)
    for source_index, domain in enumerate(DOMAIN_ORDER):
        values_m, values_r = [], []
        for mean, residual in _rows(paths[domain]):
            values_m.append(mean.reshape(-1)); values_r.append(residual.reshape(-1))
        m = np.concatenate(values_m).astype(np.float64)
        r = np.concatenate(values_r).astype(np.float64)
        assignment = np.clip(np.searchsorted(edges, m, side="right") - 1, 0, bins - 1)
        for index in range(bins):
            selected = assignment == index
            if not np.any(selected):
                raise ValueError(f"empty V21 profile bin {domain} {index}")
            counts[source_index, index] = int(selected.sum())
            source_means[source_index, index] = float(np.mean(m[selected]))
    mu = source_means.mean(axis=0)
    for source_index, domain in enumerate(DOMAIN_ORDER):
        chunks_m, chunks_r = [], []
        for mean, residual in _rows(paths[domain]):
            chunks_m.append(mean.reshape(-1)); chunks_r.append(residual.reshape(-1))
        m = np.concatenate(chunks_m).astype(np.float64); r = np.concatenate(chunks_r).astype(np.float64)
        assignment = np.clip(np.searchsorted(edges, m, side="right") - 1, 0, bins - 1)
        for index in range(bins):
            selected = assignment == index
            source_sigmas[source_index, index] = float(np.mean(np.square(r[selected] - mu[index])))
    sigma = np.sqrt(np.mean(source_sigmas, axis=0))
    sigma = np.maximum(sigma, float(floor))
    centers = source_means.mean(axis=0)
    return {
        "schema": PROFILE_SCHEMA, "bins": bins, "floor": floor,
        "edges": edges.tolist(), "centers": centers.tolist(),
        "mu": mu.tolist(), "sigma": sigma.tolist(),
        "source_bin_counts": counts.tolist(), "source_bin_means": source_means.tolist(),
        "source_bin_second_central_moments": source_sigmas.tolist(),
        "fit_sources": list(DOMAIN_ORDER), "source_weights": [1/3, 1/3, 1/3],
        "train_only": True,
    }


def apply_profile(residual: np.ndarray, mean: np.ndarray, profile: Mapping) -> np.ndarray:
    """Apply the frozen piecewise-linear affine map in float32."""
    m = np.asarray(mean, dtype=np.float32)
    r = np.asarray(residual, dtype=np.float32)
    centers = np.asarray(profile["centers"], dtype=np.float64)
    mu = np.asarray(profile["mu"], dtype=np.float64)
    sigma = np.asarray(profile["sigma"], dtype=np.float64)
    mf = m.astype(np.float64)
    mapped_mu = np.interp(mf, centers, mu, left=mu[0], right=mu[-1])
    mapped_sigma = np.interp(mf, centers, sigma, left=sigma[0], right=sigma[-1])
    return ((r.astype(np.float64) - mapped_mu) / mapped_sigma).astype(np.float32)


def invert_profile(value: np.ndarray, mean: np.ndarray, profile: Mapping) -> np.ndarray:
    """Invert the affine map in float32."""
    u = np.asarray(value, dtype=np.float32); m = np.asarray(mean, dtype=np.float32)
    centers = np.asarray(profile["centers"], dtype=np.float64)
    mu = np.asarray(profile["mu"], dtype=np.float64); sigma = np.asarray(profile["sigma"], dtype=np.float64)
    mf = m.astype(np.float64)
    mapped_mu = np.interp(mf, centers, mu, left=mu[0], right=mu[-1])
    mapped_sigma = np.interp(mf, centers, sigma, left=sigma[0], right=sigma[-1])
    return (u.astype(np.float64) * mapped_sigma + mapped_mu).astype(np.float32)


def transform_cube(residual: np.ndarray, mean: np.ndarray, profile: Mapping, transform: Mapping) -> tuple[np.ndarray, float, dict]:
    """Apply affine map, V20-style G21, per-cube centering and DC projection."""
    from hong2021_v20_gaussianize import exact_zero_dc_projection
    u = apply_profile(residual, mean, profile)
    latent = gaussianize_numpy(u, dict(transform))
    latent_dc = float(latent.mean(dtype=np.float64))
    projected, projection = exact_zero_dc_projection((latent - latent_dc).astype(np.float32))
    return projected, latent_dc, projection


def inverse_cube(latent: np.ndarray, mean: np.ndarray, profile: Mapping, transform: Mapping) -> np.ndarray:
    u = inverse_gaussianize_numpy(np.asarray(latent, dtype=np.float32), dict(transform))
    return invert_profile(u, mean, profile)


def prepare_cache(source: Path, output: Path, profile: Mapping, transform: Mapping) -> dict:
    """Materialize one V21 cache with the inherited exact-zero-DC projection."""
    from hong2021_v20_gaussianize import exact_zero_dc_projection
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise RuntimeError(f"refusing to overwrite V21 cache: {output}")
    partial = output.with_suffix(output.suffix + ".partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    maximum_dc = 0.0; maximum_adjustment = 0.0; total = 0
    with h5py.File(source, "r") as src, h5py.File(partial, "w") as dst:
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst.attrs["schema"] = "hong2021-v21-conditional-affine-standardized-residual-cache-v1"
        dst.attrs["complete"] = False
        dst.attrs["v21_profile_schema"] = str(profile["schema"])
        dst.attrs["v21_transform_schema"] = str(transform["schema"])
        for name, dataset in src.items():
            if name == "standardized_residual":
                dst.create_dataset(name, shape=dataset.shape, dtype="f4", chunks=dataset.chunks, compression=dataset.compression)
            else:
                src.copy(name, dst)
        latent_dc = dst.create_dataset("latent_dc", shape=(src["standardized_residual"].shape[0],), dtype="f8")
        projection_dc = dst.create_dataset("dc_projection_ortho", shape=(src["standardized_residual"].shape[0],), dtype="f8")
        for index in range(src["standardized_residual"].shape[0]):
            residual = np.asarray(src["standardized_residual"][index, 0], dtype=np.float32)
            mean = np.asarray(src["conditional_mean"][index, 0], dtype=np.float32)
            value, dc, audit = transform_cube(residual, mean, profile, transform)
            dst["standardized_residual"][index, 0] = value
            latent_dc[index] = dc; projection_dc[index] = audit["final_ortho_dc"]
            maximum_dc = max(maximum_dc, float(audit["final_ortho_dc"]))
            maximum_adjustment = max(maximum_adjustment, abs(float(audit["adjustment"])))
            total += 1
        dst.attrs["complete"] = True
        dst.attrs["objects"] = total
        dst.attrs["maximum_postprojection_ortho_dc"] = maximum_dc
        dst.attrs["maximum_absolute_adjustment"] = maximum_adjustment
    partial.replace(output)
    import hashlib
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {"path": str(output), "sha256": digest, "objects": total, "maximum_postprojection_ortho_dc": maximum_dc, "maximum_absolute_adjustment": maximum_adjustment}


def fit_v21_transform(train_paths: Mapping[str, Path], profile: Mapping, *, histogram_bins: int = 131072, knots: int = 8193, z_limit: float = 5.0) -> dict:
    """Fit G21 by streaming affine-normalized training voxels through temporary caches.

    The caller may provide materialized temporary source caches; no validation data
    are accepted by this API.
    """
    if tuple(train_paths) != DOMAIN_ORDER:
        raise ValueError("V21 transform sources must be TNG100, SIMBA, Swift")
    # fit_knots consumes standardized_residual.  Build an in-memory-compatible
    # temporary HDF5 file per source so its exact V20 histogram/CDF recipe is reused.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="hong2021_v21_fit_") as directory:
        paths = {}
        for domain in DOMAIN_ORDER:
            out = Path(directory) / f"{domain}.h5"; paths[domain] = out
            with h5py.File(train_paths[domain], "r") as src, h5py.File(out, "w") as dst:
                shape = src["standardized_residual"].shape
                ds = dst.create_dataset("standardized_residual", shape=shape, dtype="f4")
                dst.attrs["schema"] = "hong2021-v14-multiscale-standardized-residual-cache-v1"
                dst.attrs["complete"] = True
                for index in range(shape[0]):
                    ds[index] = apply_profile(src["standardized_residual"][index], src["conditional_mean"][index], profile)
        return fit_knots(paths, histogram_bins=histogram_bins, knots=knots, z_limit=z_limit)


__all__ = ["fit_profile", "apply_profile", "invert_profile", "transform_cube", "inverse_cube", "fit_v21_transform"]
