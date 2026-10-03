#!/usr/bin/env python
"""Fit, prepare, and audit the frozen V20-E8 Gaussianized residual caches."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from scipy.special import ndtr

from hong2021_residual_v12_gaussianized import (
    gaussianize_numpy,
    inverse_gaussianize_numpy,
)
from hong2021_v18_init import measure_band_mode_variances, sha256_file


REGISTRY_SCHEMA = "hong2021-v20-predeclared-gaussianized-marginal-program-v1"
FROZEN_REGISTRY_SHA256 = (
    "4c9f8cbb992bb2475e3731ca915b6d0795ba0a76256923f973770288dc468bc0"
)
SOURCE_CACHE_SCHEMA = "hong2021-v14-multiscale-standardized-residual-cache-v1"
CACHE_SCHEMA = (
    "hong2021-v20-gaussianized-multiscale-standardized-residual-cache-v1"
)
TRANSFORM_SCHEMA = "hong2021-v20-e8-equal-source-gaussianization-v1"
MEASUREMENT_SCHEMA = (
    "hong2021-v20-e8-source-balanced-gaussianized-band-variance-v1"
)
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
DOMAIN_ATTRIBUTES = {"TNG100": "TNG100", "SIMBA": "SIMBA", "Swift": "Swift-EAGLE"}
SPLITS = ("train", "validation")
EXPECTED_GRID = 64


def _resolve(value: str, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_frozen_registry(path: Path, repo: Path) -> dict[str, Any]:
    path = path.resolve()
    if sha256_file(path) != FROZEN_REGISTRY_SHA256:
        raise ValueError("V20 registry differs from its frozen pre-implementation hash")
    registry = json.loads(path.read_text())
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("invalid V20 development registry")
    if registry.get("status") != (
        "frozen_after_train_only_derivation_before_v20_e8_implementation_or_execution"
    ):
        raise ValueError("V20 registry is not in its frozen pre-E8 state")
    parent = registry["parent_evidence"]
    if sha256_file(_resolve(parent["v19_registry"], repo)) != parent[
        "v19_registry_sha256"
    ]:
        raise ValueError("V20 parent V19 registry hash mismatch")
    audit = registry["independent_audit"]
    if sha256_file(Path(audit["record"]).resolve()) != audit["record_sha256"]:
        raise ValueError("V20 independent audit record hash mismatch")
    return registry


def _cache_specs(experiment: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    data = experiment["data"]
    derived = experiment["derived_caches"]
    return {
        domain: {
            split: {
                "source": Path(data[domain][f"source_{split}_cache"]["path"]).resolve(),
                "source_sha256": data[domain][f"source_{split}_cache"]["sha256"],
                **derived[domain][split],
            }
            for split in SPLITS
        }
        for domain in DOMAIN_ORDER
    }


def _extrema(path: Path) -> tuple[float, float, int]:
    low, high, count = float("inf"), float("-inf"), 0
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("schema")) != SOURCE_CACHE_SCHEMA:
            raise ValueError(f"not a frozen V14 source cache: {path}")
        if not bool(handle.attrs.get("complete", False)):
            raise ValueError(f"incomplete V14 source cache: {path}")
        for row in handle["standardized_residual"]:
            value = np.asarray(row[0], dtype=np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite V14 residual: {path}")
            low = min(low, float(value.min()))
            high = max(high, float(value.max()))
            count += value.size
    return low, high, count


def _histogram(path: Path, edges: np.ndarray) -> tuple[np.ndarray, int]:
    result = np.zeros(len(edges) - 1, dtype=np.int64)
    count = 0
    with h5py.File(path, "r") as handle:
        for row in handle["standardized_residual"]:
            value = np.asarray(row[0], dtype=np.float32)
            result += np.histogram(value, bins=edges)[0]
            count += value.size
    if int(result.sum()) != count:
        raise RuntimeError("V20 histogram did not contain every residual voxel")
    return result, count


def fit_knots(
    train_paths: Mapping[str, Path], *, histogram_bins: int, knots: int,
    z_limit: float,
) -> dict[str, Any]:
    """Return the exact V12-style equal-source three-domain knot fit."""
    if tuple(train_paths) != DOMAIN_ORDER:
        raise ValueError("V20 fit sources must be TNG100, SIMBA, Swift in order")
    extrema = {domain: _extrema(path) for domain, path in train_paths.items()}
    low = min(row[0] for row in extrema.values())
    high = max(row[1] for row in extrema.values())
    padding = max(1.0e-6, (high - low) * 1.0e-6)
    edges = np.linspace(low - padding, high + padding, histogram_bins + 1)
    probabilities = []
    for domain, path in train_paths.items():
        histogram, count = _histogram(path, edges)
        if count != extrema[domain][2]:
            raise RuntimeError("V20 source voxel count changed between fit passes")
        probabilities.append(histogram.astype(np.float64) / count)
    probability_mass = np.mean(np.stack(probabilities), axis=0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cdf_midpoint = np.cumsum(probability_mass) - 0.5 * probability_mass
    z_knots = np.linspace(-z_limit, z_limit, knots)
    value_knots = np.interp(
        ndtr(z_knots), cdf_midpoint, centers,
        left=centers[0], right=centers[-1],
    )
    value_knots = np.maximum.accumulate(value_knots)
    for index in range(1, len(value_knots)):
        if value_knots[index] <= value_knots[index - 1]:
            value_knots[index] = np.nextafter(value_knots[index - 1], np.inf)
    return {
        "z_knots": z_knots,
        "residual_value_knots": value_knots,
        "histogram_range": [float(edges[0]), float(edges[-1])],
        "histogram_padding": float(padding),
        "extrema": extrema,
    }


def exact_zero_dc_projection(
    value: np.ndarray, *, maximum_ortho_dc: float = 1.0e-9,
    maximum_adjustment: float = 0.03, maximum_fsum_difference: float = 2.0e-11,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the frozen Addendum-B one-voxel float32 DC projection."""
    result = np.asarray(value, dtype=np.float32).copy(order="C")
    if result.ndim != 3 or len(set(result.shape)) != 1:
        raise ValueError("V20 DC projection requires one cubic float32 field")
    sqrt_voxels = math.sqrt(result.size)
    flat = result.reshape(-1)
    initial_sum = float(np.sum(flat, dtype=np.float64))
    fsum_difference = abs(initial_sum - math.fsum(map(float, flat)))
    if fsum_difference > maximum_fsum_difference:
        raise ValueError("V20 float64 sum and math.fsum cross-check failed")
    index, passes, adjustment = -1, 0, 0.0
    if initial_sum != 0.0:
        index = int(np.argmin(np.abs(flat)))
        original = float(flat[index])
        for _ in range(2):
            residue = float(np.sum(flat, dtype=np.float64))
            if abs(residue) / sqrt_voxels <= maximum_ortho_dc:
                break
            flat[index] = np.float32(float(flat[index]) - residue)
            passes += 1
        adjustment = float(flat[index]) - original
    final_ortho_dc = abs(float(np.sum(flat, dtype=np.float64))) / sqrt_voxels
    if final_ortho_dc > maximum_ortho_dc:
        raise ValueError("V20 exact-zero-DC projection did not converge")
    if abs(adjustment) > maximum_adjustment:
        raise ValueError("V20 exact-zero-DC projection exceeded its materiality bound")
    if passes > 2:
        raise AssertionError("V20 projection exceeded its fixed pass count")
    return result, {
        "initial_sum": initial_sum,
        "fsum_difference": fsum_difference,
        "index": index,
        "passes": passes,
        "adjustment": adjustment,
        "final_ortho_dc": final_ortho_dc,
    }


def transform_cube(
    source: np.ndarray, transform: Mapping[str, Any],
) -> tuple[np.ndarray, float, dict[str, Any]]:
    latent = gaussianize_numpy(np.asarray(source, dtype=np.float32), dict(transform))
    latent_dc = float(latent.mean(dtype=np.float64))
    centered = (latent - latent_dc).astype(np.float32)
    projected, projection = exact_zero_dc_projection(centered)
    return projected, latent_dc, projection


def _validate_knots_and_roundtrip(
    transform: Mapping[str, Any], fitted: Mapping[str, Any],
) -> dict[str, float]:
    z = np.asarray(transform["z_knots"], dtype=np.float64)
    value = np.asarray(transform["residual_value_knots"], dtype=np.float64)
    if not np.array_equal(z, fitted["z_knots"]):
        raise ValueError("V20 refit z knots differ from the frozen transform")
    if not np.array_equal(value, fitted["residual_value_knots"]):
        raise ValueError("V20 refit value knots differ from the frozen transform")
    if not np.all(np.diff(z) > 0) or not np.all(np.diff(value) > 0):
        raise ValueError("V20 transform knots are not strictly increasing")
    knot_error = float(
        np.max(np.abs(np.interp(np.interp(value, value, z), z, value) - value))
    )
    return {"float64_knot_max_absolute": knot_error}


def _central_moments(raw: np.ndarray) -> dict[str, float | list[float]]:
    mean, second, third, fourth = [float(value) for value in raw]
    variance = second - mean**2
    standard_deviation = math.sqrt(variance)
    central_third = third - 3.0 * mean * second + 2.0 * mean**3
    central_fourth = (
        fourth - 4.0 * mean * third + 6.0 * mean**2 * second - 3.0 * mean**4
    )
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "skewness": central_third / standard_deviation**3,
        "excess_kurtosis": central_fourth / variance**2 - 3.0,
        "raw_moments": [mean, second, third, fourth],
    }


def audit_transform_application(
    train_paths: Mapping[str, Path], transform: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every Addendum-A train-voxel check from source bytes."""
    z_knots = np.asarray(transform["z_knots"], dtype=np.float64)
    value_knots = np.asarray(transform["residual_value_knots"], dtype=np.float64)
    slopes = np.diff(value_knots) / np.diff(z_knots)
    per_source: dict[str, Any] = {}
    pooled_raw = np.zeros(4, dtype=np.float64)
    global_float32 = global_float64 = 0.0
    for domain, path in train_paths.items():
        sums = np.zeros(4, dtype=np.float64)
        count = clipped = 0
        maximum_cube_mean = maximum_float32 = sum_error_squared = 0.0
        maximum_ratio = maximum_float64 = 0.0
        with h5py.File(path, "r") as handle:
            for row in handle["standardized_residual"]:
                source = np.asarray(row[0], dtype=np.float32)
                inside = (source >= value_knots[0]) & (source <= value_knots[-1])
                clipped += int(source.size - inside.sum())
                latent = gaussianize_numpy(source, dict(transform))
                maximum_cube_mean = max(
                    maximum_cube_mean, abs(float(latent.mean(dtype=np.float64)))
                )
                latent64 = latent.astype(np.float64)
                sums += np.asarray(
                    [np.sum(latent64**power, dtype=np.float64) for power in (1, 2, 3, 4)]
                )
                count += latent.size
                source_inside = source[inside]
                latent_inside = latent[inside]
                reconstructed = inverse_gaussianize_numpy(
                    latent_inside, dict(transform)
                )
                error = np.abs(
                    reconstructed.astype(np.float64)
                    - source_inside.astype(np.float64)
                )
                maximum_float32 = max(maximum_float32, float(error.max()))
                sum_error_squared += float(np.square(error).sum())
                interval = np.clip(
                    np.searchsorted(
                        z_knots, latent_inside.astype(np.float64), side="right"
                    ) - 1,
                    0, len(slopes) - 1,
                )
                effective_slope = np.maximum.reduce([
                    slopes[np.clip(interval - 1, 0, len(slopes) - 1)],
                    slopes[interval],
                    slopes[np.clip(interval + 1, 0, len(slopes) - 1)],
                ])
                tolerance = 4.0 * (
                    np.spacing(
                        np.maximum(np.abs(source_inside), np.float32(1.0)).astype(
                            np.float32
                        )
                    ).astype(np.float64)
                    + effective_slope
                    * np.spacing(
                        np.maximum(np.abs(latent_inside), np.float32(1.0)).astype(
                            np.float32
                        )
                    ).astype(np.float64)
                )
                maximum_ratio = max(maximum_ratio, float(np.max(error / tolerance)))
                source64 = source_inside.astype(np.float64)
                exact_latent = np.interp(source64, value_knots, z_knots)
                exact_reconstructed = np.interp(exact_latent, z_knots, value_knots)
                maximum_float64 = max(
                    maximum_float64,
                    float(np.max(
                        np.abs(exact_reconstructed - source64)
                        / np.maximum(np.abs(source64), 1.0)
                    )),
                )
        row = _central_moments(sums / count)
        row.update({
            "maximum_absolute_uncentered_cube_mean": maximum_cube_mean,
            "clipped_voxels": clipped,
            "clipped_fraction": clipped / count,
            "float32_max_absolute_roundtrip": maximum_float32,
            "float32_rms_roundtrip": math.sqrt(sum_error_squared / (count - clipped)),
            "float32_max_error_to_ulp_budget_ratio": maximum_ratio,
            "float64_max_scaled_data_roundtrip": maximum_float64,
        })
        per_source[domain] = row
        pooled_raw += np.asarray(row["raw_moments"], dtype=np.float64) / 3.0
        global_float32 = max(global_float32, maximum_float32)
        global_float64 = max(global_float64, maximum_float64)
    pooled = _central_moments(pooled_raw)
    limits = transform["predeclared_limits"]
    if (
        abs(float(pooled["mean"])) > limits["pooled_absolute_mean"]
        or not limits["pooled_std_min"]
        <= float(pooled["standard_deviation"])
        <= limits["pooled_std_max"]
        or abs(float(pooled["skewness"])) > limits["pooled_absolute_skew"]
        or abs(float(pooled["excess_kurtosis"]))
        > limits["pooled_absolute_excess_kurtosis"]
        or global_float32 > limits["float32_absolute_roundtrip"]
        or global_float64 > limits["float64_scaled_data_roundtrip"]
    ):
        raise ValueError("V20 full train-voxel transform audit failed")
    stored = transform["latent_statistics"]
    for domain in DOMAIN_ORDER:
        for key, value in per_source[domain].items():
            expected = stored["per_source"][domain][key]
            if isinstance(value, list):
                if value != expected:
                    raise ValueError(f"V20 stored latent statistic differs: {domain} {key}")
            elif float(value) != float(expected):
                raise ValueError(f"V20 stored latent statistic differs: {domain} {key}")
    for key, value in pooled.items():
        expected = stored["equal_source_pooled"][key]
        if isinstance(value, list):
            if value != expected:
                raise ValueError(f"V20 stored pooled latent statistic differs: {key}")
        elif float(value) != float(expected):
            raise ValueError(f"V20 stored pooled latent statistic differs: {key}")
    return {
        "per_source": per_source,
        "equal_source_pooled": pooled,
        "float32_global_max_absolute": global_float32,
        "float64_global_max_scaled": global_float64,
    }


def _validate_cache(
    *, source_path: Path, output_path: Path, source_sha256: str,
    output_sha256: str, transform: Mapping[str, Any], transform_path: Path,
    transform_sha256: str, expected_objects: int,
) -> dict[str, Any]:
    if sha256_file(source_path) != source_sha256:
        raise ValueError("V20 source-cache SHA256 mismatch")
    if sha256_file(output_path) != output_sha256:
        raise ValueError("V20 derived-cache SHA256 mismatch")
    sum_squares = 0.0
    voxels = 0
    maximum_ortho_dc = 0.0
    maximum_adjustment = 0.0
    maximum_fsum = 0.0
    pass_2_count = 0
    adjusted = 0
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "r") as out:
        exact_attributes = {
            "schema": CACHE_SCHEMA,
            "complete": True,
            "source_standardized_cache": str(source_path),
            "source_standardized_cache_sha256": source_sha256,
            "gaussianization_transform": str(transform_path),
            "gaussianization_transform_sha256": transform_sha256,
            "gaussianization_train_only": True,
            "latent_dc_restored_at_sampling": False,
            "dc_projection_schema": "hong2021-v20-addendum-b-exact-float32-zero-dc-v1",
        }
        for key, expected in exact_attributes.items():
            actual = out.attrs.get(key)
            if isinstance(actual, np.generic):
                actual = actual.item()
            if actual != expected:
                raise ValueError(f"V20 derived-cache attribute differs: {key}")
        dataset = source["standardized_residual"]
        if tuple(dataset.shape) != (expected_objects, 1, 64, 64, 64):
            raise ValueError("V20 source-cache object count or grid differs")
        if out["standardized_residual"].shape != dataset.shape:
            raise ValueError("V20 derived standardized-residual shape differs")
        for index, row in enumerate(dataset):
            expected, latent_dc, projection = transform_cube(row[0], transform)
            actual = np.asarray(out["standardized_residual"][index, 0])
            if not np.array_equal(actual, expected):
                raise ValueError(f"V20 derived target differs at object {index}")
            if float(out["latent_dc"][index]) != latent_dc:
                raise ValueError(f"V20 latent DC differs at object {index}")
            stored_projection = (
                float(out["dc_projection_adjustment"][index]),
                int(out["dc_projection_index"][index]),
                int(out["dc_projection_passes"][index]),
            )
            expected_projection = (
                projection["adjustment"], projection["index"], projection["passes"],
            )
            if stored_projection != expected_projection:
                raise ValueError(f"V20 DC projection audit differs at object {index}")
            sum_squares += float(np.square(actual, dtype=np.float64).sum())
            voxels += actual.size
            maximum_ortho_dc = max(
                maximum_ortho_dc, projection["final_ortho_dc"]
            )
            maximum_adjustment = max(
                maximum_adjustment, abs(projection["adjustment"])
            )
            maximum_fsum = max(maximum_fsum, projection["fsum_difference"])
            pass_2_count += int(projection["passes"] == 2)
            adjusted += int(projection["passes"] > 0)
        for name in (
            "conditional_mean", "observable_context_features",
            "predicted_band_scales", "predicted_residual_dc",
        ):
            if source[name].shape != out[name].shape:
                raise ValueError(f"V20 copied dataset shape differs: {name}")
            for index in range(len(source[name])):
                if not np.array_equal(source[name][index], out[name][index]):
                    raise ValueError(f"V20 copied dataset differs: {name} {index}")
        rms = math.sqrt(sum_squares / voxels)
        if rms != float(out.attrs["standardized_residual_rms"]):
            raise ValueError("V20 derived-cache RMS differs from its bytes")
    return {
        "objects": expected_objects,
        "voxels": voxels,
        "rms": rms,
        "maximum_absolute_ortho_dc": maximum_ortho_dc,
        "maximum_absolute_adjustment": maximum_adjustment,
        "maximum_fsum_crosscheck": maximum_fsum,
        "pass_2_count": pass_2_count,
        "adjusted_cubes": adjusted,
    }


def verify_frozen_artifacts(registry_path: Path, repo: Path) -> dict[str, Any]:
    registry = load_frozen_registry(registry_path, repo)
    experiment = registry["e8_gaussianized_marginal_retrain"]
    specification = experiment["gaussianization"]
    transform_path = Path(specification["path"]).resolve()
    if sha256_file(transform_path) != specification["sha256"]:
        raise ValueError("V20 Gaussianization transform SHA256 mismatch")
    transform = json.loads(transform_path.read_text())
    if transform.get("schema") != TRANSFORM_SCHEMA:
        raise ValueError("V20 transform schema mismatch")
    cache_specs = _cache_specs(experiment)
    train_paths = {
        domain: cache_specs[domain]["train"]["source"] for domain in DOMAIN_ORDER
    }
    fitted = fit_knots(
        train_paths,
        histogram_bins=int(specification["histogram_bins"]),
        knots=int(specification["knots"]),
        z_limit=float(specification["z_support"][1]),
    )
    roundtrip = _validate_knots_and_roundtrip(transform, fitted)
    transform_application = audit_transform_application(train_paths, transform)
    cache_report: dict[str, dict[str, Any]] = {}
    for domain in DOMAIN_ORDER:
        cache_report[domain] = {}
        for split in SPLITS:
            row = cache_specs[domain][split]
            cache_report[domain][split] = _validate_cache(
                source_path=row["source"], output_path=Path(row["path"]).resolve(),
                source_sha256=row["source_sha256"], output_sha256=row["sha256"],
                transform=transform, transform_path=transform_path,
                transform_sha256=specification["sha256"],
                expected_objects=int(row["objects"]),
            )
    initialization = experiment["initialization_and_normalization"]
    measurement_path = Path(initialization["measurement_report"]).resolve()
    if sha256_file(measurement_path) != initialization["measurement_report_sha256"]:
        raise ValueError("V20 initialization measurement SHA256 mismatch")
    measurement = json.loads(measurement_path.read_text())
    if measurement.get("schema") != MEASUREMENT_SCHEMA:
        raise ValueError("V20 initialization measurement schema mismatch")
    sources = {
        domain: {
            "path": cache_specs[domain]["train"]["path"],
            "sha256": cache_specs[domain]["train"]["sha256"],
            "objects": cache_specs[domain]["train"]["objects"],
            "domain_attribute": DOMAIN_ATTRIBUTES[domain],
        }
        for domain in DOMAIN_ORDER
    }
    measured = measure_band_mode_variances(
        sources,
        parseval_relative_tolerance=1.0e-12,
        maximum_absolute_ortho_dc=1.0e-9,
        allowed_schemas=(CACHE_SCHEMA,),
    )
    expected = np.asarray(
        initialization["source_balanced_band_mode_variance"], dtype=np.float64
    )
    actual = np.asarray(measured["source_balanced"], dtype=np.float64)
    relative = float(np.max(np.abs(actual - expected) / expected))
    if relative > 1.0e-9:
        raise ValueError("V20 band-variance remeasurement differs from registry")
    sigma_data = math.sqrt(
        np.mean([cache_report[domain]["train"]["rms"] ** 2 for domain in DOMAIN_ORDER])
    )
    if sigma_data != float(initialization["sigma_data"]):
        raise ValueError("V20 sigma_data remeasurement differs from registry")
    return {
        "schema": "hong2021-v20-full-derived-artifact-audit-v1",
        "registry_sha256": FROZEN_REGISTRY_SHA256,
        "transform_sha256": specification["sha256"],
        "roundtrip": roundtrip,
        "transform_application": transform_application,
        "caches": cache_report,
        "measurement_sha256": initialization["measurement_report_sha256"],
        "sigma_data": sigma_data,
        "band_variance_remeasurement": measured,
        "maximum_relative_band_variance_difference": relative,
        "complete": True,
    }


def _copy_dataset_layout(source: h5py.Dataset) -> dict[str, Any]:
    result: dict[str, Any] = {
        "shape": source.shape, "dtype": "float32", "chunks": source.chunks,
    }
    if source.compression is not None:
        result.update(
            compression=source.compression, compression_opts=source.compression_opts
        )
    if source.shuffle:
        result["shuffle"] = True
    if source.fletcher32:
        result["fletcher32"] = True
    return result


def prepare_inference_cache(
    source_path: Path, output_path: Path, transform_path: Path,
) -> dict[str, Any]:
    """Apply the sealed V20 transform to one post-seal inference cache."""
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    transform_path = transform_path.resolve()
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    if output_path.exists() or partial.exists():
        raise RuntimeError(f"V20 refuses a pre-existing inference cache: {output_path}")
    transform = json.loads(transform_path.read_text())
    if transform.get("schema") != TRANSFORM_SCHEMA:
        raise ValueError("V20 inference transform schema mismatch")
    transform_sha256 = sha256_file(transform_path)
    source_sha256 = sha256_file(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(source_path, "r") as source, h5py.File(partial, "x") as target:
            if str(source.attrs.get("schema")) != SOURCE_CACHE_SCHEMA:
                raise ValueError("V20 inference source is not a V14 standardized cache")
            if not bool(source.attrs.get("complete", False)):
                raise ValueError("V20 inference source cache is incomplete")
            for key, value in source.attrs.items():
                target.attrs[key] = value
            target.attrs.update({
                "complete": False,
                "schema": CACHE_SCHEMA,
                "source_standardized_cache": str(source_path),
                "source_standardized_cache_sha256": source_sha256,
                "gaussianization_transform": str(transform_path),
                "gaussianization_transform_sha256": transform_sha256,
                "gaussianization_train_only": True,
                "target": "sealed-inference exact-zero-DC equal-source Gaussianized V14 standardized residual",
                "latent_dc_restored_at_sampling": False,
                "dc_projection_schema": "hong2021-v20-addendum-b-exact-float32-zero-dc-v1",
            })
            for name in source:
                if name != "standardized_residual":
                    source.copy(name, target, name=name)
            source_ds = source["standardized_residual"]
            grid = int(source_ds.shape[-1])
            target_ds = target.create_dataset(
                "standardized_residual", **_copy_dataset_layout(source_ds)
            )
            objects = len(source_ds)
            latent_dc_ds = target.create_dataset("latent_dc", (objects,), dtype="float64")
            adjustment_ds = target.create_dataset(
                "dc_projection_adjustment", (objects,), dtype="float64"
            )
            index_ds = target.create_dataset(
                "dc_projection_index", (objects,), dtype="int64"
            )
            passes_ds = target.create_dataset(
                "dc_projection_passes", (objects,), dtype="uint8"
            )
            sum_squares = 0.0
            voxels = 0
            maximum_ortho_dc = maximum_adjustment = maximum_fsum = 0.0
            pass_2_count = adjusted = 0
            for index, row in enumerate(source_ds):
                value, latent_dc, projection = transform_cube(row[0], transform)
                target_ds[index, 0] = value
                latent_dc_ds[index] = latent_dc
                adjustment_ds[index] = projection["adjustment"]
                index_ds[index] = projection["index"]
                passes_ds[index] = projection["passes"]
                sum_squares += float(np.square(value, dtype=np.float64).sum())
                voxels += value.size
                maximum_ortho_dc = max(maximum_ortho_dc, projection["final_ortho_dc"])
                maximum_adjustment = max(
                    maximum_adjustment, abs(projection["adjustment"])
                )
                maximum_fsum = max(maximum_fsum, projection["fsum_difference"])
                pass_2_count += int(projection["passes"] == 2)
                adjusted += int(projection["passes"] > 0)
            target.attrs.update({
                "standardized_residual_rms": math.sqrt(sum_squares / voxels),
                "maximum_absolute_ortho_dc": maximum_ortho_dc,
                "maximum_absolute_dc_projection_adjustment": maximum_adjustment,
                "maximum_fsum_crosscheck": maximum_fsum,
                "dc_projection_pass_2_count": pass_2_count,
                "dc_projection_adjusted_cubes": adjusted,
                "complete": True,
            })
            target.flush()
        os.replace(partial, output_path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "transform_sha256": transform_sha256,
        "objects": objects,
        "grid": grid,
        "maximum_absolute_ortho_dc": maximum_ortho_dc,
        "maximum_absolute_adjustment": maximum_adjustment,
        "pass_2_count": pass_2_count,
    }


def prepare_frozen_caches(registry_path: Path, repo: Path, output: Path) -> None:
    """Rebuild the six frozen cache byte streams into a new empty directory."""
    registry = load_frozen_registry(registry_path, repo)
    experiment = registry["e8_gaussianized_marginal_retrain"]
    specification = experiment["gaussianization"]
    reference_transform = Path(specification["path"]).resolve()
    transform = json.loads(reference_transform.read_text())
    cache_specs = _cache_specs(experiment)
    fitted = fit_knots(
        {domain: cache_specs[domain]["train"]["source"] for domain in DOMAIN_ORDER},
        histogram_bins=int(specification["histogram_bins"]),
        knots=int(specification["knots"]),
        z_limit=float(specification["z_support"][1]),
    )
    _validate_knots_and_roundtrip(transform, fitted)
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"V20 refuses a pre-existing preparation output: {output}")
    output.mkdir(parents=True)
    partials: list[Path] = []
    try:
        shutil.copyfile(reference_transform, output / "gaussianization.json")
        for domain in DOMAIN_ORDER:
            for split in SPLITS:
                row = cache_specs[domain][split]
                source_path = row["source"]
                final = output / Path(row["path"]).name
                partial = final.with_suffix(final.suffix + ".partial")
                partials.append(partial)
                with h5py.File(source_path, "r") as source, h5py.File(
                    partial, "x"
                ) as target:
                    for key, value in source.attrs.items():
                        target.attrs[key] = value
                    target.attrs.update({
                        "complete": False,
                        "schema": CACHE_SCHEMA,
                        "source_standardized_cache": str(source_path),
                        "source_standardized_cache_sha256": row["source_sha256"],
                        "gaussianization_transform": str(reference_transform),
                        "gaussianization_transform_sha256": specification["sha256"],
                        "gaussianization_train_only": True,
                        "target": "deterministically exact-zero-DC equal-source Gaussianized V14 standardized residual",
                        "latent_dc_restored_at_sampling": False,
                        "dc_projection_schema": "hong2021-v20-addendum-b-exact-float32-zero-dc-v1",
                    })
                    for name in source:
                        if name != "standardized_residual":
                            source.copy(name, target, name=name)
                    source_ds = source["standardized_residual"]
                    target_ds = target.create_dataset(
                        "standardized_residual", **_copy_dataset_layout(source_ds)
                    )
                    objects = len(source_ds)
                    latent_dc_ds = target.create_dataset("latent_dc", (objects,), dtype="float64")
                    adjustment_ds = target.create_dataset(
                        "dc_projection_adjustment", (objects,), dtype="float64"
                    )
                    index_ds = target.create_dataset(
                        "dc_projection_index", (objects,), dtype="int64"
                    )
                    passes_ds = target.create_dataset(
                        "dc_projection_passes", (objects,), dtype="uint8"
                    )
                    sum_squares = 0.0
                    voxels = 0
                    maximum_ortho_dc = maximum_adjustment = maximum_fsum = 0.0
                    pass_2_count = adjusted = 0
                    for index, source_row in enumerate(source_ds):
                        value, latent_dc, projection = transform_cube(
                            source_row[0], transform
                        )
                        target_ds[index, 0] = value
                        latent_dc_ds[index] = latent_dc
                        adjustment_ds[index] = projection["adjustment"]
                        index_ds[index] = projection["index"]
                        passes_ds[index] = projection["passes"]
                        sum_squares += float(np.square(value, dtype=np.float64).sum())
                        voxels += value.size
                        maximum_ortho_dc = max(
                            maximum_ortho_dc, projection["final_ortho_dc"]
                        )
                        maximum_adjustment = max(
                            maximum_adjustment, abs(projection["adjustment"])
                        )
                        maximum_fsum = max(
                            maximum_fsum, projection["fsum_difference"]
                        )
                        pass_2_count += int(projection["passes"] == 2)
                        adjusted += int(projection["passes"] > 0)
                    target.attrs.update({
                        "standardized_residual_rms": math.sqrt(sum_squares / voxels),
                        "maximum_absolute_ortho_dc": maximum_ortho_dc,
                        "maximum_absolute_dc_projection_adjustment": maximum_adjustment,
                        "maximum_fsum_crosscheck": maximum_fsum,
                        "dc_projection_pass_2_count": pass_2_count,
                        "dc_projection_adjusted_cubes": adjusted,
                        "complete": True,
                    })
                    target.flush()
                if sha256_file(partial) != row["sha256"]:
                    raise ValueError(f"V20 rebuilt cache SHA256 differs: {domain} {split}")
                os.replace(partial, final)
        reference_measurement = Path(
            experiment["initialization_and_normalization"]["measurement_report"]
        ).resolve()
        shutil.copyfile(reference_measurement, output / reference_measurement.name)
    except BaseException:
        for partial in partials:
            if partial.exists():
                partial.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--registry", type=Path, required=True)
    verify.add_argument("--repo", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--registry", type=Path, required=True)
    prepare.add_argument("--repo", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "verify":
        print(json.dumps(verify_frozen_artifacts(args.registry, args.repo), indent=2))
    else:
        prepare_frozen_caches(args.registry, args.repo, args.out)


if __name__ == "__main__":
    main()
