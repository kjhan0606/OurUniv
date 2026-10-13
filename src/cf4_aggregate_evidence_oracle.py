#!/usr/bin/env python3
"""Exact cached geometry primitives for aggregate CF4 peak evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cf4_peak_evidence_phase_cache import (
    covariance_for_point_sets,
    restriction_adjoint_spectrum,
)


FINE_N = 576
COARSE_N = 192
BOX_SIZE_MPC_H = 384.0
DX_MPC_H = BOX_SIZE_MPC_H / FINE_N
HALF_SEPARATION_CELLS = 3
SHELL_RADIUS_CELLS = 2
PRODUCTION_PARENT_SEEDS = tuple(range(3193, 3449))
PRODUCTION_REPLICATE_MASTER_SEEDS = (
    2026082301,
    2026082302,
    2026082303,
    2026082304,
)
PRODUCTION_PARTICLE_COUNT = 2048


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_vector(vector: np.ndarray, *, normalize: bool) -> np.ndarray:
    """Apply the frozen lowest-max-index sign convention."""
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("canonical vector must be a finite three-vector")
    norm = float(np.linalg.norm(value))
    if norm == 0.0:
        raise ValueError("canonical vector cannot be zero")
    result = value / norm if normalize else value.copy()
    index = int(np.argmax(np.abs(result)))
    if result[index] < 0.0:
        result = -result
    return result


def canonical_axis(axis: np.ndarray) -> np.ndarray:
    return canonical_vector(axis, normalize=True)


def canonical_axis_offset(
    axis: np.ndarray,
    half_separation_cells: int = HALF_SEPARATION_CELLS,
) -> np.ndarray:
    if half_separation_cells <= 0:
        raise ValueError("half separation must be positive")
    unit = canonical_axis(axis)
    raw = np.rint(float(half_separation_cells) * unit).astype(np.int64)
    canonical = canonical_vector(raw, normalize=False)
    return canonical.astype(np.int64)


def geometry_key(
    midpoint_mpc_h: np.ndarray,
    axis: np.ndarray,
    *,
    fine_n: int = FINE_N,
    dx_mpc_h: float = DX_MPC_H,
    half_separation_cells: int = HALF_SEPARATION_CELLS,
) -> tuple[int, int, int, int, int, int]:
    q = np.asarray(midpoint_mpc_h, dtype=np.float64)
    if q.shape != (3,) or not np.all(np.isfinite(q)):
        raise ValueError("midpoint must be a finite three-vector")
    if fine_n <= 0 or not np.isfinite(dx_mpc_h) or dx_mpc_h <= 0.0:
        raise ValueError("mesh geometry must be positive and finite")
    midpoint = np.full(3, fine_n // 2, dtype=np.int64)
    midpoint += np.rint(q / float(dx_mpc_h)).astype(np.int64)
    midpoint = np.mod(midpoint, fine_n)
    offset = canonical_axis_offset(axis, half_separation_cells)
    return tuple(int(value) for value in np.concatenate((midpoint, offset)))


def geometry_key_from_grid_axis(
    midpoint_grid: np.ndarray,
    axis: np.ndarray,
    *,
    fine_n: int = FINE_N,
    half_separation_cells: int = HALF_SEPARATION_CELLS,
) -> tuple[int, int, int, int, int, int]:
    midpoint = np.asarray(midpoint_grid, dtype=np.int64)
    if midpoint.shape != (3,):
        raise ValueError("midpoint grid must be a three-vector")
    midpoint = np.mod(midpoint, fine_n)
    offset = canonical_axis_offset(axis, half_separation_cells)
    return tuple(int(value) for value in np.concatenate((midpoint, offset)))


def covariance_key(
    key: Iterable[int],
    *,
    refinement_ratio: int = 3,
) -> tuple[int, int, int, int, int, int]:
    value = np.asarray(tuple(key), dtype=np.int64)
    if value.shape != (6,) or refinement_ratio <= 0:
        raise ValueError("invalid likelihood key or refinement ratio")
    return tuple(int(item) for item in np.concatenate((
        np.mod(value[:3], refinement_ratio), value[3:],
    )))


def points_from_geometry_key(
    key: Iterable[int],
    *,
    fine_n: int = FINE_N,
    shell_radius_cells: int = SHELL_RADIUS_CELLS,
) -> np.ndarray:
    value = np.asarray(tuple(key), dtype=np.int64)
    if value.shape != (6,):
        raise ValueError("likelihood key must contain six integers")
    midpoint = np.mod(value[:3], fine_n)
    offset = value[3:]
    if np.all(offset == 0):
        raise ValueError("axis offset cannot be zero")
    centres = np.vstack((midpoint - offset, midpoint + offset))
    unit = np.eye(3, dtype=np.int64) * int(shell_radius_cells)
    points = []
    for centre in centres:
        points.append(centre)
        points.extend(centre + delta for delta in np.vstack((unit, -unit)))
    return np.mod(np.asarray(points, dtype=np.int64), fine_n)


def target_vector(
    centre_target: float,
    shell_target: float,
) -> np.ndarray:
    result = np.full(14, float(shell_target), dtype=np.float64)
    result[[0, 7]] = float(centre_target)
    return result


def logmeanexp_parent(log_z: np.ndarray) -> np.ndarray:
    values = np.asarray(log_z, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("parent evidence must be finite with parents on the last axis")
    maximum = np.max(values, axis=-1, keepdims=True)
    return (
        np.squeeze(maximum, axis=-1)
        + np.log(np.sum(np.exp(values - maximum), axis=-1))
        - math.log(values.shape[-1])
    )


@dataclass(frozen=True)
class AtlasBounds:
    relative_min: tuple[int, int, int]
    relative_max: tuple[int, int, int]
    padded_min: tuple[int, int, int]
    padded_max: tuple[int, int, int]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(
            high - low + 1
            for low, high in zip(self.padded_min, self.padded_max)
        )


def response_atlas_bounds(
    prior_mean_mpc_h: np.ndarray,
    prior_sigma_mpc_h: np.ndarray,
    *,
    dx_mpc_h: float = DX_MPC_H,
    sigma_extent: float = 10.0,
    padding_cells: int = 5,
) -> AtlasBounds:
    mean = np.asarray(prior_mean_mpc_h, dtype=np.float64)
    sigma = np.asarray(prior_sigma_mpc_h, dtype=np.float64)
    if mean.shape != (3,) or sigma.shape != (3,):
        raise ValueError("atlas prior vectors must have length three")
    if (
        not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(sigma))
        or np.any(sigma <= 0.0)
        or not np.isfinite(dx_mpc_h)
        or dx_mpc_h <= 0.0
        or sigma_extent <= 0.0
        or padding_cells < 0
    ):
        raise ValueError("invalid atlas bounds input")
    low = np.ceil(
        (mean - sigma_extent * sigma) / dx_mpc_h - 0.5
    ).astype(np.int64)
    high = np.floor(
        (mean + sigma_extent * sigma) / dx_mpc_h + 0.5
    ).astype(np.int64)
    padding = int(padding_cells)
    return AtlasBounds(
        tuple(int(value) for value in low),
        tuple(int(value) for value in high),
        tuple(int(value) for value in low - padding),
        tuple(int(value) for value in high + padding),
    )


def parent_response_grid(
    coarse: np.ndarray,
    filter_full: np.ndarray,
) -> np.ndarray:
    """Return the existing exact A R* y response as a float64 real grid."""
    coarse = np.asarray(coarse, dtype=np.float64)
    filter_full = np.asarray(filter_full)
    if coarse.ndim != 3 or not (
        coarse.shape[0] == coarse.shape[1] == coarse.shape[2]
    ):
        raise ValueError("coarse response field must be cubic")
    fine_n = int(filter_full.shape[0])
    if filter_full.shape != (fine_n, fine_n, fine_n):
        raise ValueError("full density filter must be cubic")
    if fine_n % coarse.shape[0] != 0:
        raise ValueError("fine/coarse response ratio must be integral")
    coarse_fft = np.fft.fftn(coarse, norm="ortho")
    fine_mean_fft = restriction_adjoint_spectrum(coarse_fft, fine_n)
    response = np.fft.ifftn(filter_full * fine_mean_fft, norm="ortho")
    real_rms = float(np.sqrt(np.mean(response.real ** 2)))
    imaginary_rms = float(np.sqrt(np.mean(response.imag ** 2)))
    if imaginary_rms / max(real_rms, np.finfo(float).tiny) > 1e-12:
        raise RuntimeError("parent response broke Hermitian symmetry")
    return np.asarray(response.real, dtype=np.float64)


def extract_response_atlas(
    response: np.ndarray,
    bounds: AtlasBounds,
) -> np.ndarray:
    response = np.asarray(response, dtype=np.float64)
    if response.ndim != 3 or not (
        response.shape[0] == response.shape[1] == response.shape[2]
    ):
        raise ValueError("response must be a cubic grid")
    centre = response.shape[0] // 2
    indices = [
        np.mod(np.arange(low, high + 1) + centre, response.shape[0])
        for low, high in zip(bounds.padded_min, bounds.padded_max)
    ]
    return np.asarray(response[np.ix_(*indices)], dtype=np.float64)


def atlas_point_indices(
    points: np.ndarray,
    bounds: AtlasBounds,
    *,
    fine_n: int = FINE_N,
) -> tuple[np.ndarray, np.ndarray]:
    """Map periodic fine-grid points to atlas indices and an in-atlas mask."""
    value = np.mod(np.asarray(points, dtype=np.int64), fine_n)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("points must have shape (n,3)")
    centre = fine_n // 2
    output = np.zeros_like(value)
    inside = np.ones(len(value), dtype=bool)
    for dimension, (low, high) in enumerate(zip(
        bounds.padded_min, bounds.padded_max
    )):
        candidates = np.arange(low, high + 1, dtype=np.int64)
        periodic = np.mod(candidates + centre, fine_n)
        inverse = {int(cell): index for index, cell in enumerate(periodic)}
        for row, cell in enumerate(value[:, dimension]):
            index = inverse.get(int(cell))
            if index is None:
                inside[row] = False
            else:
                output[row, dimension] = index
    return output, inside


def lookup_response_atlas(
    atlas: np.ndarray,
    points: np.ndarray,
    bounds: AtlasBounds,
    *,
    fine_n: int = FINE_N,
) -> np.ndarray:
    atlas = np.asarray(atlas)
    if atlas.shape != bounds.shape or atlas.dtype != np.float64:
        raise ValueError("atlas shape or dtype does not match its frozen bounds")
    indices, inside = atlas_point_indices(points, bounds, fine_n=fine_n)
    if not np.all(inside):
        raise KeyError("one or more response points lie outside the atlas")
    return np.asarray(atlas[tuple(indices.T)], dtype=np.float64)


def sorted_unique_geometry_keys(
    keys: Iterable[Iterable[int]],
) -> list[tuple[int, int, int, int, int, int]]:
    result = set()
    for key in keys:
        value = tuple(int(item) for item in key)
        if len(value) != 6:
            raise ValueError("every geometry key must contain six integers")
        result.add(value)
    return sorted(result)


def point_sets_from_keys(
    keys: Iterable[Iterable[int]],
    *,
    fine_n: int = FINE_N,
    shell_radius_cells: int = SHELL_RADIUS_CELLS,
) -> np.ndarray:
    values = sorted_unique_geometry_keys(keys)
    if not values:
        return np.empty((0, 14, 3), dtype=np.int64)
    return np.stack([
        points_from_geometry_key(
            key, fine_n=fine_n, shell_radius_cells=shell_radius_cells
        )
        for key in values
    ])


def covariance_terms_for_keys(
    filter_full: np.ndarray,
    keys: Iterable[Iterable[int]],
    *,
    coarse_n: int = COARSE_N,
    fine_n: int = FINE_N,
    sigma_delta: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate exact covariance terms with a temporary cache."""
    cache = ExactCovarianceCache(
        filter_full,
        coarse_n=coarse_n,
        fine_n=fine_n,
        sigma_delta=sigma_delta,
    )
    return cache.terms(keys)


class ExactCovarianceCache:
    """Persist exact phase/offset Cholesky terms across oracle calls."""

    def __init__(
        self,
        filter_full: np.ndarray,
        *,
        coarse_n: int = COARSE_N,
        fine_n: int = FINE_N,
        sigma_delta: float = 0.25,
    ):
        self.filter_full = np.asarray(filter_full)
        self.coarse_n = int(coarse_n)
        self.fine_n = int(fine_n)
        self.sigma_delta = float(sigma_delta)
        if self.filter_full.shape != (self.fine_n,) * 3:
            raise ValueError("covariance cache filter shape mismatch")
        self._terms: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}
        self.evaluation_batches = 0
        self.evaluated_covariance_keys = 0

    def terms(
        self,
        keys: Iterable[Iterable[int]],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Evaluate one exact covariance per previously unseen cache key."""
        values = sorted_unique_geometry_keys(keys)
        if not values:
            return (
                np.empty((0, 14, 14), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                {
                    "geometry_key_count": 0,
                    "unique_covariance_key_count": 0,
                    "new_covariance_key_count": 0,
                    "cached_covariance_key_count": len(self._terms),
                },
            )
        covariance_keys = [covariance_key(key) for key in values]
        representative: dict[tuple[int, ...], tuple[int, ...]] = {}
        for key, cov_key in zip(values, covariance_keys):
            if cov_key not in self._terms:
                representative.setdefault(cov_key, key)
        missing = sorted(representative)
        phase_diagnostics: dict[str, Any] = {}
        if missing:
            representative_points = [
                points_from_geometry_key(
                    representative[key], fine_n=self.fine_n
                )
                for key in missing
            ]
            covariance, phase_diagnostics = covariance_for_point_sets(
                self.filter_full, self.coarse_n, representative_points
            )
            observation = covariance + np.eye(14)[None] * self.sigma_delta**2
            cholesky = np.linalg.cholesky(observation)
            logdet = 2.0 * np.sum(
                np.log(np.diagonal(cholesky, axis1=1, axis2=2)), axis=1
            )
            for index, key in enumerate(missing):
                self._terms[key] = (
                    np.asarray(cholesky[index], dtype=np.float64),
                    float(logdet[index]),
                )
            self.evaluation_batches += 1
            self.evaluated_covariance_keys += len(missing)
        output_cholesky = np.stack([
            self._terms[key][0] for key in covariance_keys
        ])
        output_logdet = np.asarray([
            self._terms[key][1] for key in covariance_keys
        ])
        diagnostics = dict(phase_diagnostics)
        diagnostics.update({
            "geometry_key_count": len(values),
            "unique_covariance_key_count": len(set(covariance_keys)),
            "new_covariance_key_count": len(missing),
            "cached_covariance_key_count": len(self._terms),
            "evaluation_batches": self.evaluation_batches,
        })
        return output_cholesky, output_logdet, diagnostics


def vectorized_log_evidence(
    means: np.ndarray,
    targets: np.ndarray,
    cholesky: np.ndarray,
    log_determinants: np.ndarray,
) -> np.ndarray:
    means = np.asarray(means, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    cholesky = np.asarray(cholesky, dtype=np.float64)
    log_determinants = np.asarray(log_determinants, dtype=np.float64)
    if means.ndim != 2 or targets.shape != means.shape:
        raise ValueError("means and targets must be aligned matrices")
    count, dimension = means.shape
    if cholesky.shape != (count, dimension, dimension):
        raise ValueError("batched Cholesky shape mismatch")
    if log_determinants.shape != (count,):
        raise ValueError("log-determinant shape mismatch")
    residual = targets - means
    whitened = np.linalg.solve(cholesky, residual[..., None])[..., 0]
    quadratic = np.einsum("gi,gi->g", whitened, whitened)
    return -0.5 * (
        dimension * math.log(2.0 * math.pi) + log_determinants + quadratic
    )


def _atlas_index_maps(
    bounds: AtlasBounds,
    fine_n: int,
) -> np.ndarray:
    maps = np.full((3, fine_n), -1, dtype=np.int16)
    centre = fine_n // 2
    for dimension, (low, high) in enumerate(zip(
        bounds.padded_min, bounds.padded_max
    )):
        relative = np.arange(low, high + 1, dtype=np.int64)
        periodic = np.mod(relative + centre, fine_n)
        if len(np.unique(periodic)) != len(periodic):
            raise ValueError("atlas interval cannot wrap onto itself")
        maps[dimension, periodic] = np.arange(len(relative), dtype=np.int16)
    return maps


def _evaluate_log_z_from_atlases(
    keys: Iterable[Iterable[int]],
    atlas_entries: list[dict[str, Any]],
    bounds: AtlasBounds,
    filter_full: np.ndarray,
    targets: np.ndarray,
    *,
    coarse_n: int = COARSE_N,
    fine_n: int = FINE_N,
    sigma_delta: float = 0.25,
    verify_atlas_hashes: bool = True,
    covariance_cache: ExactCovarianceCache | None = None,
) -> tuple[list[tuple[int, int, int, int, int, int]], np.ndarray, dict[str, Any]]:
    """Evaluate exact parent log evidence for sorted unique geometry keys."""
    values = sorted_unique_geometry_keys(keys)
    target = np.asarray(targets, dtype=np.float64)
    if target.shape != (14,) or not np.all(np.isfinite(target)):
        raise ValueError("the exact peak target must contain 14 finite values")
    if not values or not atlas_entries:
        raise ValueError("atlas evidence needs geometry keys and parent entries")
    points = np.stack([
        points_from_geometry_key(key, fine_n=fine_n) for key in values
    ])
    maps = _atlas_index_maps(bounds, fine_n)
    atlas_indices = np.stack([
        maps[dimension, points[..., dimension]] for dimension in range(3)
    ], axis=-1)
    inside = np.all(atlas_indices >= 0, axis=(1, 2))
    if covariance_cache is None:
        raise ValueError("a persistent exact covariance cache is required")
    if (
        covariance_cache.coarse_n != coarse_n
        or covariance_cache.fine_n != fine_n
        or covariance_cache.sigma_delta != sigma_delta
        or covariance_cache.filter_full is not np.asarray(filter_full)
    ):
        raise ValueError("covariance cache contract differs from the oracle")
    cholesky, logdet, covariance_diagnostics = covariance_cache.terms(values)
    repeated_target = np.broadcast_to(target, (len(values), 14))
    log_z = np.empty((len(values), len(atlas_entries)), dtype=np.float64)
    for parent_index, entry in enumerate(atlas_entries):
        atlas_path = Path(entry["atlas"])
        if verify_atlas_hashes:
            if "atlas_sha256" not in entry:
                raise RuntimeError("atlas entry lacks its mandatory hash")
            if sha256_file(atlas_path) != entry["atlas_sha256"]:
                raise RuntimeError(f"atlas hash mismatch for seed {entry['seed']}")
        atlas = np.load(atlas_path, mmap_mode="r", allow_pickle=False)
        if atlas.shape != bounds.shape or atlas.dtype != np.float64:
            raise RuntimeError("atlas shard shape or dtype mismatch")
        means = np.empty((len(values), 14), dtype=np.float64)
        if np.any(inside):
            index = atlas_indices[inside]
            means[inside] = atlas[index[..., 0], index[..., 1], index[..., 2]]
        if np.any(~inside):
            parent_path = Path(entry["parent_field"])
            if "parent_field_sha256" not in entry:
                raise RuntimeError("atlas entry lacks its mandatory parent hash")
            if sha256_file(parent_path) != entry["parent_field_sha256"]:
                raise RuntimeError(
                    f"parent hash mismatch for outside-atlas seed {entry['seed']}"
                )
            with np.load(parent_path, allow_pickle=False) as item:
                if int(item["sample_seed"]) != int(entry["seed"]):
                    raise RuntimeError("outside-atlas parent seed mismatch")
                coarse = item["s_out"].astype(np.float32)
            full_response = parent_response_grid(coarse, filter_full)
            outside_points = points[~inside]
            means[~inside] = full_response[
                outside_points[..., 0],
                outside_points[..., 1],
                outside_points[..., 2],
            ]
        log_z[:, parent_index] = vectorized_log_evidence(
            means, repeated_target, cholesky, logdet
        )
    if not np.all(np.isfinite(log_z)):
        raise RuntimeError("exact atlas evidence contains nonfinite values")
    diagnostics = {
        "geometry_key_count": len(values),
        "parent_count": len(atlas_entries),
        "inside_atlas_key_count": int(np.count_nonzero(inside)),
        "outside_atlas_key_count": int(np.count_nonzero(~inside)),
        "covariance": covariance_diagnostics,
    }
    return values, log_z, diagnostics


def evaluate_log_z_from_atlases(
    keys: Iterable[Iterable[int]],
    atlas_entries: list[dict[str, Any]],
    bounds: AtlasBounds,
    filter_full: np.ndarray,
    targets: np.ndarray,
    *,
    coarse_n: int = COARSE_N,
    fine_n: int = FINE_N,
    sigma_delta: float = 0.25,
    covariance_cache: ExactCovarianceCache | None = None,
) -> tuple[list[tuple[int, int, int, int, int, int]], np.ndarray, dict[str, Any]]:
    """Public exact evaluator; all atlas and outside-parent hashes are mandatory."""
    return _evaluate_log_z_from_atlases(
        keys,
        atlas_entries,
        bounds,
        filter_full,
        targets,
        coarse_n=coarse_n,
        fine_n=fine_n,
        sigma_delta=sigma_delta,
        verify_atlas_hashes=True,
        covariance_cache=covariance_cache,
    )


def load_verified_atlas_manifest(
    manifest_path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], AtlasBounds]:
    """Hard-gate the canonical 256-parent production atlas lineage."""
    manifest_path = Path(manifest_path)
    if sha256_file(manifest_path) != expected_sha256:
        raise RuntimeError("response-atlas manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "ouruniv-cf4-parent-response-atlas-manifest-v1":
        raise RuntimeError("unexpected response-atlas manifest schema")
    if manifest.get("status") != "complete_exact_parent_response_atlas":
        raise RuntimeError("response-atlas manifest status did not pass")
    if manifest.get("parent_count") != 256 or manifest.get("dtype") != "float64":
        raise RuntimeError("response-atlas parent count or dtype mismatch")
    bounds_record = manifest.get("bounds", {})
    bounds = AtlasBounds(
        tuple(bounds_record.get("relative_min", ())),
        tuple(bounds_record.get("relative_max", ())),
        tuple(bounds_record.get("padded_min", ())),
        tuple(bounds_record.get("padded_max", ())),
    )
    expected_bounds = response_atlas_bounds(
        PRIOR_MEAN_MPC_H_ORACLE,
        PRIOR_SIGMA_MPC_H_ORACLE,
    )
    if bounds != expected_bounds or bounds.shape != (101, 101, 101):
        raise RuntimeError("response-atlas bounds differ from the frozen design")
    entries = manifest.get("entries", [])
    if len(entries) != 256 or tuple(
        entry.get("seed") for entry in entries
    ) != PRODUCTION_PARENT_SEEDS:
        raise RuntimeError("response-atlas seeds are not exactly 3193 through 3448")
    atlas_paths = []
    for entry in entries:
        required = {
            "seed",
            "parent_field",
            "parent_field_sha256",
            "atlas",
            "atlas_sha256",
            "shape",
            "dtype",
        }
        if not required.issubset(entry):
            raise RuntimeError("response-atlas entry lacks mandatory lineage")
        if entry["shape"] != [101, 101, 101] or entry["dtype"] != "float64":
            raise RuntimeError("response-atlas entry shape or dtype mismatch")
        atlas_path = Path(entry["atlas"])
        parent_path = Path(entry["parent_field"])
        if sha256_file(atlas_path) != entry["atlas_sha256"]:
            raise RuntimeError(f"atlas hash mismatch for seed {entry['seed']}")
        if sha256_file(parent_path) != entry["parent_field_sha256"]:
            raise RuntimeError(f"parent hash mismatch for seed {entry['seed']}")
        atlas = np.load(atlas_path, mmap_mode="r", allow_pickle=False)
        if atlas.shape != (101, 101, 101) or atlas.dtype != np.float64:
            raise RuntimeError("response-atlas shard header mismatch")
        atlas_paths.append(str(atlas_path.resolve()))
    if len(set(atlas_paths)) != 256:
        raise RuntimeError("response-atlas shard paths must be unique")
    return manifest, bounds


PRIOR_MEAN_MPC_H_ORACLE = np.asarray([0.0, -6.0, 4.0], dtype=np.float64)
PRIOR_SIGMA_MPC_H_ORACLE = np.asarray([3.0, 3.0, 3.0], dtype=np.float64)


class ExactAtlasEvidenceEvaluator:
    """Verified production-grade exact atlas evaluator with persistent cache."""

    def __init__(
        self,
        manifest_path: Path,
        manifest_sha256: str,
        filter_path: Path,
        filter_sha256: str,
        physical_model_path: Path,
        physical_model_sha256: str,
    ):
        self.manifest, self.bounds = load_verified_atlas_manifest(
            manifest_path, manifest_sha256
        )
        filter_path = Path(filter_path)
        if sha256_file(filter_path) != filter_sha256:
            raise RuntimeError("density-filter hash mismatch")
        filter_rfft = np.load(filter_path, allow_pickle=False)
        if (
            filter_rfft.shape != (576, 576, 289)
            or filter_rfft.dtype != np.float32
            or not np.all(np.isfinite(filter_rfft))
        ):
            raise RuntimeError("density-filter shape, dtype, or finite gate failed")
        from cf4_peak_evidence_phase_cache import full_spectrum_from_rfft

        self.filter_full = full_spectrum_from_rfft(filter_rfft)
        physical_model_path = Path(physical_model_path)
        if sha256_file(physical_model_path) != physical_model_sha256:
            raise RuntimeError("physical-model hash mismatch")
        model = json.loads(physical_model_path.read_text())
        peak = model["peak_constraints"]
        if float(peak["likelihood_sigma_delta"]) != 0.25:
            raise RuntimeError("physical peak sigma changed")
        self.targets = target_vector(
            float(peak["centre_target_delta_linear"]),
            float(peak["six_shell_target_delta_linear"]),
        )
        self.covariance_cache = ExactCovarianceCache(self.filter_full)

    def __call__(self, keys):
        values, log_z, _ = _evaluate_log_z_from_atlases(
            keys,
            self.manifest["entries"],
            self.bounds,
            self.filter_full,
            self.targets,
            covariance_cache=self.covariance_cache,
            verify_atlas_hashes=False,
        )
        if log_z.shape != (len(values), 256):
            raise RuntimeError("verified atlas evaluator broke parent width")
        return values, log_z


class AggregateEvidenceControllerOracle:
    """Firewall parent-specific evidence from the adaptive SMC controller."""

    def __init__(self, evaluator):
        self._evaluator = evaluator
        self._cache: dict[tuple[int, ...], np.ndarray] = {}
        self._terminal_histories: dict[int, np.ndarray] = {}
        self._sealed = False

    def evaluate(
        self,
        midpoint_mpc_h: np.ndarray,
        axis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._sealed:
            raise RuntimeError("evidence evaluation is closed after terminal seal")
        midpoint = np.asarray(midpoint_mpc_h, dtype=np.float64)
        axes = np.asarray(axis, dtype=np.float64)
        if midpoint.ndim != 2 or midpoint.shape[1] != 3 or axes.shape != midpoint.shape:
            raise ValueError("controller geometry arrays must have shape (n,3)")
        keys = [geometry_key(q, a) for q, a in zip(midpoint, axes)]
        missing = sorted(set(keys).difference(self._cache))
        if missing:
            evaluated_keys, log_z = self._evaluator(missing)
            if (
                evaluated_keys != missing
                or log_z.shape != (len(missing), 256)
                or not np.all(np.isfinite(log_z))
            ):
                raise RuntimeError("evidence evaluator broke its sorted-key contract")
            for key, row in zip(evaluated_keys, log_z):
                if key in self._cache and not np.array_equal(self._cache[key], row):
                    raise RuntimeError("evidence cache collision is not bitwise stable")
                self._cache[key] = np.asarray(row, dtype=np.float64)
        parent_log_z = np.stack([self._cache[key] for key in keys])
        aggregate = logmeanexp_parent(parent_log_z)
        return np.asarray(keys, dtype=np.int16), aggregate

    def register_terminal_history(self, master_seed: int, keys: np.ndarray) -> None:
        if self._sealed:
            raise RuntimeError("terminal histories are already irreversibly sealed")
        seed = int(master_seed)
        if seed not in PRODUCTION_REPLICATE_MASTER_SEEDS:
            raise ValueError("terminal history seed is not one of the four frozen seeds")
        if seed in self._terminal_histories:
            raise RuntimeError("duplicate terminal history seed")
        value = self._validated_terminal_keys(keys)
        tuples = [tuple(int(item) for item in row) for row in value]
        if any(key not in self._cache for key in tuples):
            raise KeyError("terminal history contains an unevaluated key")
        self._terminal_histories[seed] = value.copy()

    @staticmethod
    def _validated_terminal_keys(keys: np.ndarray) -> np.ndarray:
        value = np.asarray(keys)
        if value.dtype != np.dtype(np.int16):
            raise TypeError("terminal keys must retain the exact int16 dtype")
        if value.shape != (PRODUCTION_PARTICLE_COUNT, 6):
            raise ValueError("terminal history must contain exactly 2048 keys")
        midpoint = value[:, :3]
        offset = value[:, 3:]
        if np.any(midpoint < 0) or np.any(midpoint >= FINE_N):
            raise ValueError("terminal midpoint key is outside [0,576)")
        if np.any(np.abs(offset) > HALF_SEPARATION_CELLS) or np.any(
            np.all(offset == 0, axis=1)
        ):
            raise ValueError("terminal axis offset range is invalid")
        maximum_index = np.argmax(np.abs(offset), axis=1)
        selected = offset[np.arange(len(offset)), maximum_index]
        if np.any(selected <= 0):
            raise ValueError("terminal axis offset is not canonical")
        return value

    def seal_terminal_histories(self) -> None:
        if self._sealed:
            raise RuntimeError("terminal histories are already sealed")
        if tuple(sorted(self._terminal_histories)) != PRODUCTION_REPLICATE_MASTER_SEEDS:
            raise RuntimeError("all four frozen terminal histories are required")
        self._sealed = True
        self._evaluator = None

    def terminal_parent_log_z(
        self,
        master_seed: int,
        keys: np.ndarray,
    ) -> np.ndarray:
        if not self._sealed:
            raise RuntimeError("parent evidence is closed until all histories are sealed")
        seed = int(master_seed)
        if seed not in self._terminal_histories:
            raise ValueError("terminal seed was not registered")
        value = self._validated_terminal_keys(keys)
        if not np.array_equal(value, self._terminal_histories[seed]):
            raise RuntimeError("terminal accessor accepts only the registered history")
        tuples = [tuple(int(item) for item in row) for row in value]
        return np.stack([self._cache[key] for key in tuples])
