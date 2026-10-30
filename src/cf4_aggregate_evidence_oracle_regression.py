#!/usr/bin/env python3
"""Prospective exact regression suite for the aggregate-evidence oracle."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from cf4_aggregate_evidence_oracle import (
    AtlasBounds,
    ExactAtlasEvidenceEvaluator,
    ExactCovarianceCache,
    atlas_point_indices,
    covariance_key,
    evaluate_log_z_from_atlases,
    geometry_key,
    geometry_key_from_grid_axis,
    load_verified_atlas_manifest,
    logmeanexp_parent,
    parent_response_grid,
    points_from_geometry_key,
    sorted_unique_geometry_keys,
    target_vector,
    vectorized_log_evidence,
)
from cf4_peak_evidence import normalized_gaussian_logpdf
from cf4_peak_evidence_phase_cache import (
    _phase_response_grid_with_diagnostics,
    full_spectrum_from_rfft,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_DESIGN = ROOT / "config/cf4_aggregate_evidence_oracle_regression_design.json"
CANONICAL_PROGRAM = (
    ROOT / "config/cf4_aggregate_evidence_oracle_regression_program.json"
)
FROZEN_DESIGN_SHA256 = (
    "b735918021d6898bfbb4b29f7f4f3f732fea2804205a813761a52cc4b7616dd0"
)
EXPECTED_PARENT_SEEDS = np.arange(3193, 3449, dtype=np.int32)
EXPECTED_POINT_KINDS = np.asarray(
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8
)


_DIRECT_FILTER: np.ndarray | None = None
_DIRECT_POINTS: np.ndarray | None = None
_DIRECT_TARGETS: np.ndarray | None = None
_DIRECT_CHOLESKY: np.ndarray | None = None
_DIRECT_LOGDET: np.ndarray | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _rooted(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _require_hash(path: Path, expected: str, label: str) -> None:
    if sha256_file(path) != expected:
        raise RuntimeError(f"{label} hash mismatch")


def _load_json_with_status(item: dict[str, Any], label: str) -> dict[str, Any]:
    path = _rooted(item["path"])
    _require_hash(path, item["sha256"], label)
    record = json.loads(path.read_text())
    required = item.get("required_status")
    if required is not None and record.get("status") != required:
        raise RuntimeError(f"{label} status mismatch")
    return record


def validate_frozen_design(design: dict[str, Any], design_path: Path) -> None:
    design_path = Path(design_path).resolve()
    if design_path != FROZEN_DESIGN.resolve():
        raise RuntimeError("oracle-regression design path is not canonical")
    _require_hash(design_path, FROZEN_DESIGN_SHA256, "oracle-regression design")
    if design.get("status") != "frozen_scientific_design_before_implementation":
        raise RuntimeError("oracle-regression scientific design is not frozen")
    authorization = design.get("authorization", {})
    if authorization.get("implementation_and_tests_authorized") is not True:
        raise RuntimeError("oracle-regression implementation is not authorized")
    for key in (
        "regression_execution_authorized",
        "production_SMC_authorized",
        "conditional_field_bank_authorized",
        "parent_or_seed_selection_authorized",
        "PM_or_halo_finder_authorized",
        "RAMSES_authorized",
    ):
        if authorization.get(key) is not False:
            raise RuntimeError(f"frozen design opened forbidden authorization: {key}")

    lineage = design["fixed_lineage"]
    for key in (
        "atlas_result_record",
        "reference_calibration",
        "physical_model",
        "phase_control_record",
    ):
        _load_json_with_status(lineage[key], key)
    for key in ("atlas_manifest", "atlas_complete"):
        item = lineage[key]
        _require_hash(Path(item["path"]), item["sha256"], key)
    for bank_name in ("adaptation_2048", "adaptation_8192"):
        bank = lineage[bank_name]
        for path_key, hash_key in (
            ("record", "record_sha256"),
            ("result", "result_sha256"),
            ("arrays", "arrays_sha256"),
            ("complete", "complete_sha256"),
        ):
            _require_hash(_rooted(bank[path_key]), bank[hash_key], f"{bank_name} {path_key}")
    density_filter = lineage["density_filter"]
    filter_path = Path(density_filter["path"])
    _require_hash(filter_path, density_filter["sha256"], "density filter")
    filter_value = np.load(filter_path, mmap_mode="r", allow_pickle=False)
    if (
        list(filter_value.shape) != density_filter["shape"]
        or str(filter_value.dtype) != density_filter["dtype"]
        or not np.all(np.isfinite(filter_value))
    ):
        raise RuntimeError("density-filter shape, dtype, or finite gate failed")
    for label, item in lineage["pinned_existing_sources"].items():
        _require_hash(_rooted(item["path"]), item["sha256"], label)


def _candidate_rng(master_seed: int, branch: int, index: int) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        int(master_seed), spawn_key=(int(branch), int(index))
    )
    return np.random.Generator(np.random.PCG64DXSM(sequence))


def _isotropic_axis(rng: np.random.Generator) -> np.ndarray:
    z = 2.0 * float(rng.random()) - 1.0
    phi = 2.0 * math.pi * float(rng.random())
    radius = math.sqrt(max(0.0, 1.0 - z * z))
    return np.asarray(
        [radius * math.cos(phi), radius * math.sin(phi), z], dtype=np.float64
    )


def _sorted_selection(
    accepted: list[tuple[tuple[int, ...], int]],
) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(accepted, key=lambda row: row[0])
    keys = np.asarray([row[0] for row in ordered], dtype=np.int16)
    candidate_index = np.asarray([row[1] for row in ordered], dtype=np.int32)
    return keys, candidate_index


def deterministic_atlas_keys(
    bounds: AtlasBounds,
    *,
    master_seed: int = 2026082501,
    inside_count: int = 1024,
    outside_count: int = 64,
    candidate_limit: int = 1_000_000,
    fine_n: int = 576,
) -> dict[str, np.ndarray]:
    centre = fine_n // 2
    core_low = np.asarray(bounds.relative_min, dtype=np.int64)
    core_high = np.asarray(bounds.relative_max, dtype=np.int64)
    padded_low = np.asarray(bounds.padded_min, dtype=np.int64)
    padded_high = np.asarray(bounds.padded_max, dtype=np.int64)

    inside: list[tuple[tuple[int, ...], int]] = []
    seen_inside: set[tuple[int, ...]] = set()
    for index in range(candidate_limit):
        rng = _candidate_rng(master_seed, 0, index)
        relative = np.asarray([
            rng.integers(int(low), int(high) + 1)
            for low, high in zip(core_low, core_high)
        ], dtype=np.int64)
        axis = _isotropic_axis(rng)
        key = geometry_key_from_grid_axis(centre + relative, axis, fine_n=fine_n)
        points = points_from_geometry_key(key, fine_n=fine_n)
        _, point_inside = atlas_point_indices(points, bounds, fine_n=fine_n)
        if key not in seen_inside and np.all(point_inside):
            seen_inside.add(key)
            inside.append((key, index))
            if len(inside) == inside_count:
                break
    if len(inside) != inside_count:
        raise RuntimeError("inside-atlas candidate limit exhausted")

    outside: list[tuple[tuple[int, ...], int]] = []
    seen_outside: set[tuple[int, ...]] = set()
    for index in range(candidate_limit):
        rng = _candidate_rng(master_seed, 1, index)
        face = int(rng.integers(0, 6))
        dimension = face // 2
        excursion = int(rng.integers(1, 10))
        relative = np.zeros(3, dtype=np.int64)
        for other in range(3):
            if other != dimension:
                relative[other] = rng.integers(
                    int(core_low[other]), int(core_high[other]) + 1
                )
        if face % 2 == 0:
            relative[dimension] = padded_low[dimension] - excursion
        else:
            relative[dimension] = padded_high[dimension] + excursion
        axis = _isotropic_axis(rng)
        key = geometry_key_from_grid_axis(centre + relative, axis, fine_n=fine_n)
        points = points_from_geometry_key(key, fine_n=fine_n)
        _, point_inside = atlas_point_indices(points, bounds, fine_n=fine_n)
        if key not in seen_outside and not np.all(point_inside):
            seen_outside.add(key)
            outside.append((key, index))
            if len(outside) == outside_count:
                break
    if len(outside) != outside_count:
        raise RuntimeError("outside-atlas candidate limit exhausted")

    inside_keys, inside_index = _sorted_selection(inside)
    outside_keys, outside_index = _sorted_selection(outside)
    if set(map(tuple, inside_keys)).intersection(map(tuple, outside_keys)):
        raise RuntimeError("inside and outside regression keys overlap")
    return {
        "inside_keys": inside_keys,
        "inside_candidate_index": inside_index,
        "outside_keys": outside_keys,
        "outside_candidate_index": outside_index,
    }


def _block_pair(points: np.ndarray) -> tuple[tuple[tuple[int, ...], ...], ...]:
    value = np.asarray(points, dtype=np.int64)
    if value.shape != (14, 3):
        raise RuntimeError("historical point array has the wrong shape")
    return tuple(sorted(
        tuple(tuple(int(cell) for cell in point) for point in value[start:start + 7])
        for start in (0, 7)
    ))


def reconstruct_historical_keys(
    midpoint_offset_mpc_h: np.ndarray,
    axis: np.ndarray,
    midpoint_grid: np.ndarray,
    points: np.ndarray,
    point_kinds: np.ndarray,
    *,
    fine_n: int = 576,
) -> np.ndarray:
    offsets = np.asarray(midpoint_offset_mpc_h)
    axes = np.asarray(axis)
    midpoints = np.asarray(midpoint_grid)
    historical_points = np.asarray(points)
    kinds = np.asarray(point_kinds)
    count = len(offsets)
    if (
        offsets.dtype != np.float64
        or axes.dtype != np.float64
        or midpoints.dtype != np.int16
        or historical_points.dtype != np.int16
        or kinds.dtype != np.int8
        or offsets.shape != (count, 3)
        or axes.shape != (count, 3)
        or midpoints.shape != (count, 3)
        or historical_points.shape != (count, 14, 3)
        or kinds.shape != (count, 14)
    ):
        raise RuntimeError("historical geometry arrays are not aligned")
    if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(axes)):
        raise RuntimeError("historical continuous geometry is nonfinite")
    if not np.all(kinds == EXPECTED_POINT_KINDS):
        raise RuntimeError("historical point-kind contract changed")
    keys = np.empty((count, 6), dtype=np.int16)
    for index, (offset, direction, midpoint, old_points) in enumerate(zip(
        offsets, axes, midpoints, historical_points
    )):
        discrete = geometry_key_from_grid_axis(
            midpoint, direction, fine_n=fine_n
        )
        continuous = geometry_key(offset, direction, fine_n=fine_n)
        if discrete != continuous:
            raise RuntimeError("continuous and stored-grid geometry keys differ")
        canonical_points = points_from_geometry_key(discrete, fine_n=fine_n)
        if _block_pair(canonical_points) != _block_pair(old_points):
            raise RuntimeError("historical points differ from the canonical key")
        keys[index] = discrete
    return keys


def unique_keys_and_inverse(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(keys)
    if value.ndim != 2 or value.shape[1] != 6:
        raise ValueError("geometry keys must have shape (n,6)")
    unique = sorted_unique_geometry_keys(value)
    mapping = {key: index for index, key in enumerate(unique)}
    inverse = np.asarray([mapping[tuple(map(int, row))] for row in value], dtype=np.int32)
    return np.asarray(unique, dtype=np.int16), inverse


def dense_phase_keys(*, fine_n: int = 576) -> tuple[np.ndarray, np.ndarray]:
    phases = np.asarray(list(itertools.product(range(3), repeat=3)), dtype=np.int8)
    centre = fine_n // 2
    keys = np.asarray([
        (centre + int(px), centre + int(py), centre + int(pz), 3, 0, 0)
        for px, py, pz in phases
    ], dtype=np.int16)
    return phases, keys


def explicit_dense_covariance(
    filter_full: np.ndarray,
    coarse_n: int,
    keys: Iterable[Iterable[int]],
) -> tuple[np.ndarray, dict[str, Any]]:
    value = sorted_unique_geometry_keys(keys)
    point_sets = [points_from_geometry_key(key, fine_n=filter_full.shape[0]) for key in value]
    fine_n = int(filter_full.shape[0])
    ratio = fine_n // int(coarse_n)
    if ratio != 3 or fine_n % int(coarse_n):
        raise RuntimeError("dense covariance requires exact refinement ratio three")
    matrices = np.full((len(point_sets), 14, 14), np.nan, dtype=np.float64)
    assignment_count = np.zeros((len(point_sets), 14, 14), dtype=np.int16)
    maximum_relative = 0.0
    maximum_absolute = 0.0
    phases_used = []
    for phase_tuple in itertools.product(range(ratio), repeat=3):
        phase = np.asarray(phase_tuple, dtype=np.int64)
        response, diagnostics = _phase_response_grid_with_diagnostics(
            filter_full, coarse_n, phase
        )
        imaginary_relative = diagnostics["imaginary_relative_RMS"]
        imaginary_absolute = diagnostics["maximum_absolute_imaginary"]
        if (
            not np.all(np.isfinite(response))
            or not np.isfinite(imaginary_relative)
            or not np.isfinite(imaginary_absolute)
        ):
            raise RuntimeError("dense phase response or diagnostic is nonfinite")
        phases_used.append(phase_tuple)
        maximum_relative = max(
            maximum_relative, imaginary_relative
        )
        maximum_absolute = max(
            maximum_absolute, imaginary_absolute
        )
        for geometry_index, points in enumerate(point_sets):
            for column in range(14):
                if tuple(np.mod(points[column], ratio)) != phase_tuple:
                    continue
                translation = points[column] - phase
                for row in range(14):
                    location = tuple(np.mod(points[row] - translation, fine_n))
                    matrices[geometry_index, row, column] = response[location]
                    assignment_count[geometry_index, row, column] += 1
        del response
    validate_dense_phase_coverage(phases_used, assignment_count, ratio=ratio)
    maximum_asymmetry = float(np.max(np.abs(
        matrices - np.swapaxes(matrices, 1, 2)
    )))
    matrices = 0.5 * (matrices + np.swapaxes(matrices, 1, 2))
    return matrices, {
        "phase_count": len(phases_used),
        "phases": [list(phase) for phase in phases_used],
        "response_grids_held_simultaneously": 1,
        "maximum_phase_response_imaginary_relative_RMS": maximum_relative,
        "maximum_phase_response_absolute_imaginary": maximum_absolute,
        "maximum_pre_symmetrization_asymmetry": maximum_asymmetry,
    }


def validate_dense_phase_coverage(
    phases: Iterable[Iterable[int]],
    assignment_count: np.ndarray,
    *,
    ratio: int,
) -> None:
    expected = list(itertools.product(range(3), repeat=3))
    actual = [tuple(int(value) for value in phase) for phase in phases]
    counts = np.asarray(assignment_count)
    if ratio != 3:
        raise RuntimeError("dense phase coverage requires refinement ratio three")
    if actual != expected:
        raise RuntimeError("dense phase set or lexicographic order is incomplete")
    if counts.shape != (27, 14, 14) or not np.all(counts == 1):
        raise RuntimeError("dense covariance entries were not assigned exactly once")


def dense_phase_control(
    filter_full: np.ndarray,
    target: np.ndarray,
    parent_entries: list[dict[str, Any]],
    *,
    coarse_n: int = 192,
    fine_n: int = 576,
    sigma_delta: float = 0.25,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    phases, keys = dense_phase_keys(fine_n=fine_n)
    cache = ExactCovarianceCache(
        filter_full, coarse_n=coarse_n, fine_n=fine_n, sigma_delta=sigma_delta
    )
    cached_cholesky, cached_logdet, cached_diagnostics = cache.terms(keys)
    identity = np.eye(14, dtype=np.float64)
    cached_signal = (
        cached_cholesky @ np.swapaxes(cached_cholesky, 1, 2)
        - sigma_delta ** 2 * identity[None]
    )
    direct_signal, direct_diagnostics = explicit_dense_covariance(
        filter_full, coarse_n, keys
    )
    direct_observation = direct_signal + sigma_delta ** 2 * identity[None]
    direct_cholesky = np.linalg.cholesky(direct_observation)
    direct_logdet = 2.0 * np.sum(
        np.log(np.diagonal(direct_cholesky, axis1=1, axis2=2)), axis=1
    )
    control_seeds = np.asarray([3193, 3429], dtype=np.int32)
    entries_by_seed = {int(entry["seed"]): entry for entry in parent_entries}
    point_sets = np.stack([points_from_geometry_key(key, fine_n=fine_n) for key in keys])
    cached_log_z = np.empty((27, 2), dtype=np.float64)
    direct_log_z = np.empty((27, 2), dtype=np.float64)
    repeated_target = np.broadcast_to(target, (27, 14))
    for parent_index, seed in enumerate(control_seeds):
        entry = entries_by_seed[int(seed)]
        parent_path = Path(entry["parent_field"])
        _require_hash(parent_path, entry["parent_field_sha256"], f"dense parent {seed}")
        with np.load(parent_path, allow_pickle=False) as item:
            if int(item["sample_seed"]) != int(seed):
                raise RuntimeError("dense parent internal seed mismatch")
            coarse = item["s_out"].astype(np.float32)
        response = parent_response_grid(coarse, filter_full)
        means = response[
            point_sets[..., 0], point_sets[..., 1], point_sets[..., 2]
        ]
        del response
        cached_log_z[:, parent_index] = vectorized_log_evidence(
            means, repeated_target, cached_cholesky, cached_logdet
        )
        for geometry_index in range(27):
            direct_log_z[geometry_index, parent_index] = normalized_gaussian_logpdf(
                target, means[geometry_index], direct_observation[geometry_index]
            )[0]

    signal_difference = np.abs(cached_signal - direct_signal)
    relative = float(
        np.linalg.norm(cached_signal - direct_signal)
        / max(np.linalg.norm(direct_signal), np.finfo(float).tiny)
    )
    metrics = {
        "phase_count": direct_diagnostics["phase_count"],
        "phases": direct_diagnostics["phases"],
        "unique_covariance_key_count": len({covariance_key(row) for row in keys}),
        "response_grids_held_simultaneously": direct_diagnostics[
            "response_grids_held_simultaneously"
        ],
        "maximum_phase_response_imaginary_relative_RMS": direct_diagnostics[
            "maximum_phase_response_imaginary_relative_RMS"
        ],
        "maximum_phase_response_absolute_imaginary": direct_diagnostics[
            "maximum_phase_response_absolute_imaginary"
        ],
        "maximum_pre_symmetrization_asymmetry": direct_diagnostics[
            "maximum_pre_symmetrization_asymmetry"
        ],
        "signal_covariance_max_abs_difference": float(np.max(signal_difference)),
        "signal_covariance_relative_Frobenius_difference": relative,
        "cholesky_max_abs_difference": float(np.max(np.abs(
            cached_cholesky - direct_cholesky
        ))),
        "logdet_max_abs_difference": float(np.max(np.abs(
            cached_logdet - direct_logdet
        ))),
        "normalized_log_Z_max_abs_difference": float(np.max(np.abs(
            cached_log_z - direct_log_z
        ))),
        "cached_covariance_diagnostics": cached_diagnostics,
    }
    arrays = {
        "dense_phase": phases,
        "dense_keys": keys,
        "dense_signal_cached": cached_signal,
        "dense_signal_direct": direct_signal,
        "dense_cholesky_cached": cached_cholesky,
        "dense_cholesky_direct": direct_cholesky,
        "dense_logdet_cached": cached_logdet,
        "dense_logdet_direct": direct_logdet,
        "dense_control_parent_seed": control_seeds,
        "dense_log_Z_cached": cached_log_z,
        "dense_log_Z_direct": direct_log_z,
    }
    return arrays, metrics


def _direct_parent_block(
    task: tuple[int, list[dict[str, Any]]],
) -> tuple[int, np.ndarray, int]:
    if any(value is None for value in (
        _DIRECT_FILTER,
        _DIRECT_POINTS,
        _DIRECT_TARGETS,
        _DIRECT_CHOLESKY,
        _DIRECT_LOGDET,
    )):
        raise RuntimeError("direct parent worker was not initialized")
    start, entries = task
    output = np.empty((len(_DIRECT_POINTS), len(entries)), dtype=np.float64)
    repeated_target = np.broadcast_to(_DIRECT_TARGETS, (len(_DIRECT_POINTS), 14))
    for column, entry in enumerate(entries):
        parent_path = Path(entry["parent_field"])
        _require_hash(
            parent_path, entry["parent_field_sha256"], f"direct parent {entry['seed']}"
        )
        with np.load(parent_path, allow_pickle=False) as item:
            if int(item["sample_seed"]) != int(entry["seed"]):
                raise RuntimeError("direct parent internal seed mismatch")
            coarse = item["s_out"].astype(np.float32)
        if coarse.shape != (192, 192, 192) or not np.all(np.isfinite(coarse)):
            raise RuntimeError("direct parent shape or finite gate failed")
        response = parent_response_grid(coarse, _DIRECT_FILTER)
        means = response[
            _DIRECT_POINTS[..., 0],
            _DIRECT_POINTS[..., 1],
            _DIRECT_POINTS[..., 2],
        ]
        del response
        output[:, column] = vectorized_log_evidence(
            means, repeated_target, _DIRECT_CHOLESKY, _DIRECT_LOGDET
        )
    return start, output, len(entries)


def reassemble_parent_blocks(
    blocks: list[tuple[int, np.ndarray, int]],
    *,
    geometry_count: int,
    parent_count: int,
    parent_block_size: int,
) -> tuple[np.ndarray, int]:
    ordered = sorted(blocks, key=lambda row: row[0])
    expected_starts = list(range(0, parent_count, parent_block_size))
    if [row[0] for row in ordered] != expected_starts:
        raise RuntimeError("direct parent blocks were reassembled out of order")
    if any(
        row[1].shape != (geometry_count, min(parent_block_size, parent_count - row[0]))
        or row[2] != row[1].shape[1]
        for row in ordered
    ):
        raise RuntimeError("direct parent block shape or count changed")
    output = np.concatenate([row[1] for row in ordered], axis=1)
    evaluations = sum(row[2] for row in ordered)
    if output.shape != (geometry_count, parent_count):
        raise RuntimeError("direct parent block assembly shape mismatch")
    return output, evaluations


def direct_log_z_all_parents(
    keys: Iterable[Iterable[int]],
    parent_entries: list[dict[str, Any]],
    filter_full: np.ndarray,
    target: np.ndarray,
    covariance_cache: ExactCovarianceCache,
    *,
    worker_processes: int = 8,
    parent_block_size: int = 32,
    fine_n: int = 576,
) -> tuple[list[tuple[int, ...]], np.ndarray, int]:
    values = sorted_unique_geometry_keys(keys)
    if len(parent_entries) != 256:
        raise RuntimeError("direct control requires all 256 parents")
    cholesky, logdet, _ = covariance_cache.terms(values)
    points = np.stack([points_from_geometry_key(key, fine_n=fine_n) for key in values])
    tasks = [
        (start, parent_entries[start:start + parent_block_size])
        for start in range(0, len(parent_entries), parent_block_size)
    ]
    if len(tasks) != worker_processes or any(len(block) != 32 for _, block in tasks):
        raise RuntimeError("direct parent blocks differ from eight blocks of 32")
    global _DIRECT_FILTER, _DIRECT_POINTS, _DIRECT_TARGETS
    global _DIRECT_CHOLESKY, _DIRECT_LOGDET
    _DIRECT_FILTER = filter_full
    _DIRECT_POINTS = points
    _DIRECT_TARGETS = np.asarray(target, dtype=np.float64)
    _DIRECT_CHOLESKY = cholesky
    _DIRECT_LOGDET = logdet
    context = mp.get_context("fork")
    with context.Pool(processes=worker_processes) as pool:
        blocks = pool.map(_direct_parent_block, tasks, chunksize=1)
    _DIRECT_FILTER = None
    _DIRECT_POINTS = None
    _DIRECT_TARGETS = None
    _DIRECT_CHOLESKY = None
    _DIRECT_LOGDET = None
    output, evaluations = reassemble_parent_blocks(
        blocks,
        geometry_count=len(values),
        parent_count=len(parent_entries),
        parent_block_size=parent_block_size,
    )
    if output.shape != (len(values), 256) or not np.all(np.isfinite(output)):
        raise RuntimeError("direct parent log evidence failed shape or finite gate")
    return values, output, evaluations


def evaluate_unique_batched(
    evaluator: ExactAtlasEvidenceEvaluator,
    keys: np.ndarray,
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(keys, dtype=np.int16)
    chunks = []
    before = evaluator.covariance_cache.evaluated_covariance_keys
    for start in range(0, len(values), batch_size):
        block = values[start:start + batch_size]
        evaluated, log_z = evaluator(block)
        if evaluated != [tuple(map(int, row)) for row in block]:
            raise RuntimeError("batched evaluator changed lexicographic key order")
        chunks.append(log_z)
    first = np.concatenate(chunks, axis=0)
    after_first = evaluator.covariance_cache.evaluated_covariance_keys
    repeated = []
    for start in range(0, len(values), batch_size):
        block = values[start:start + batch_size]
        _, log_z = evaluator(block)
        repeated.append(log_z)
    second = np.concatenate(repeated, axis=0)
    after_second = evaluator.covariance_cache.evaluated_covariance_keys
    if not np.array_equal(first, second) or after_second != after_first:
        raise RuntimeError("second identical oracle evaluation was not cache-stable")
    return first, {
        "new_covariance_keys_first_evaluation": after_first - before,
        "new_covariance_keys_second_evaluation": after_second - after_first,
        "second_evaluation_bitwise_identical": True,
    }


def historical_bank_regression(
    bank_name: str,
    arrays_path: Path,
    evaluator: ExactAtlasEvidenceEvaluator,
    *,
    draw_count: int,
    unique_geometry_count: int,
    unique_covariance_count: int,
    batch_size: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    required = {
        "parent_seed",
        "log_Z_peak",
        "midpoint_offset_mpc_h",
        "axis",
        "midpoint_grid",
        "points",
        "point_kinds",
        "logmean_parent_Z_by_geometry",
        "covariance_log_determinant",
    }
    with np.load(arrays_path, allow_pickle=False) as source:
        if not required.issubset(source.files):
            raise RuntimeError(f"{bank_name} lacks required historical arrays")
        parent_seed = source["parent_seed"]
        historical_log_z = source["log_Z_peak"]
        midpoint_offset = source["midpoint_offset_mpc_h"]
        axis = source["axis"]
        midpoint_grid = source["midpoint_grid"]
        points = source["points"]
        point_kinds = source["point_kinds"]
        historical_log_z_bar = source["logmean_parent_Z_by_geometry"]
        historical_logdet = source["covariance_log_determinant"]
        exact_contract = {
            "parent_seed": (parent_seed, np.dtype("int32"), (256,)),
            "log_Z_peak": (
                historical_log_z, np.dtype("float64"), (256, draw_count)
            ),
            "midpoint_offset_mpc_h": (
                midpoint_offset, np.dtype("float64"), (draw_count, 3)
            ),
            "axis": (axis, np.dtype("float64"), (draw_count, 3)),
            "midpoint_grid": (
                midpoint_grid, np.dtype("int16"), (draw_count, 3)
            ),
            "points": (points, np.dtype("int16"), (draw_count, 14, 3)),
            "point_kinds": (
                point_kinds, np.dtype("int8"), (draw_count, 14)
            ),
            "logmean_parent_Z_by_geometry": (
                historical_log_z_bar, np.dtype("float64"), (draw_count,)
            ),
            "covariance_log_determinant": (
                historical_logdet, np.dtype("float64"), (draw_count,)
            ),
        }
        for name, (value, dtype, shape) in exact_contract.items():
            if value.dtype != dtype or value.shape != shape:
                raise RuntimeError(f"{bank_name} historical dtype or shape: {name}")
        keys = reconstruct_historical_keys(
            midpoint_offset, axis, midpoint_grid, points, point_kinds
        )
    if (
        not np.array_equal(parent_seed, EXPECTED_PARENT_SEEDS)
        or not all(np.all(np.isfinite(value)) for value in (
            historical_log_z,
            midpoint_offset,
            axis,
            historical_log_z_bar,
            historical_logdet,
        ))
    ):
        raise RuntimeError(f"{bank_name} historical input contract failed")
    unique, inverse = unique_keys_and_inverse(keys)
    actual_covariance_count = len({covariance_key(row) for row in unique})
    if len(unique) != unique_geometry_count or actual_covariance_count != unique_covariance_count:
        raise RuntimeError(f"{bank_name} frozen unique-key counts changed")
    unique_log_z, cache_diagnostics = evaluate_unique_batched(
        evaluator, unique, batch_size=batch_size
    )
    new_log_z = unique_log_z[inverse].T
    new_log_z_bar = logmeanexp_parent(unique_log_z[inverse])
    _, unique_logdet, _ = evaluator.covariance_cache.terms(unique)
    new_logdet = unique_logdet[inverse]
    if not all(np.all(np.isfinite(value)) for value in (
        new_log_z, new_log_z_bar, new_logdet
    )):
        raise RuntimeError(f"{bank_name} regenerated values are nonfinite")
    metrics = {
        "draw_count": draw_count,
        "unique_geometry_key_count": len(unique),
        "unique_covariance_key_count": actual_covariance_count,
        "historical_log_Z_max_abs_difference": float(np.max(np.abs(
            new_log_z - historical_log_z
        ))),
        "historical_log_Z_bar_max_abs_difference": float(np.max(np.abs(
            new_log_z_bar - historical_log_z_bar
        ))),
        "historical_covariance_logdet_max_abs_difference": float(np.max(np.abs(
            new_logdet - historical_logdet
        ))),
        "cache": cache_diagnostics,
    }
    return {
        f"{bank_name}_keys": keys,
        f"{bank_name}_new_log_Z": new_log_z,
        f"{bank_name}_new_log_Z_bar": new_log_z_bar,
    }, metrics


def _relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(left) - np.asarray(right))
        / max(np.linalg.norm(np.asarray(right)), np.finfo(float).tiny)
    )


def _gate_results(
    design: dict[str, Any],
    dense: dict[str, Any],
    atlas: dict[str, Any],
    banks: dict[str, dict[str, Any]],
) -> tuple[dict[str, bool], str | None]:
    finite_pass = all(_all_finite_metrics(value) for value in (
        dense, atlas, banks
    ))
    dense_gate = design["dense_27_phase_control"]["gates"]
    dense_pass = bool(
        finite_pass
        and dense["phase_count"] == dense_gate["phase_count_exact"]
        and dense["phases"] == [
            list(phase) for phase in itertools.product(range(3), repeat=3)
        ]
        and dense["unique_covariance_key_count"] == 27
        and dense["response_grids_held_simultaneously"]
        == dense_gate["response_grids_held_simultaneously"]
        and dense["maximum_phase_response_imaginary_relative_RMS"]
        <= dense_gate["phase_imaginary_relative_RMS_max"]
        and dense["maximum_phase_response_absolute_imaginary"]
        <= dense_gate["phase_imaginary_absolute_max"]
        and dense["maximum_pre_symmetrization_asymmetry"]
        <= dense_gate["pre_symmetry_max_abs"]
        and dense["signal_covariance_max_abs_difference"]
        <= dense_gate["signal_covariance_max_abs_difference"]
        and dense["signal_covariance_relative_Frobenius_difference"]
        <= dense_gate["signal_covariance_relative_Frobenius_difference"]
        and dense["cholesky_max_abs_difference"]
        <= dense_gate["cholesky_max_abs_difference"]
        and dense["logdet_max_abs_difference"]
        <= dense_gate["logdet_max_abs_difference"]
        and dense["normalized_log_Z_max_abs_difference"]
        <= dense_gate["normalized_log_Z_max_abs_difference"]
    )
    atlas_gate = design["atlas_vs_direct_control"]["gates"]
    inside_pass = bool(
        finite_pass
        and atlas["inside_diagnostics"]["inside_atlas_key_count"] == 1024
        and atlas["inside_diagnostics"]["outside_atlas_key_count"] == 0
        and atlas["inside_log_Z_max_abs_difference"]
        <= atlas_gate["inside_log_Z_max_abs_difference"]
    )
    outside_pass = bool(
        finite_pass
        and atlas["outside_diagnostics"]["inside_atlas_key_count"] == 0
        and atlas["outside_diagnostics"]["outside_atlas_key_count"] == 64
        and atlas["outside_slow_path_full_response_parent_evaluations"] == 256
        and atlas["direct_full_response_parent_evaluations"] == 256
        and atlas["outside_log_Z_max_abs_difference"]
        <= atlas_gate["outside_log_Z_max_abs_difference"]
    )
    bank_gate = design["historical_bank_regression"]["gates_each_bank"]
    bank_pass = {}
    for label, metrics in banks.items():
        bank_pass[label] = bool(
            finite_pass
            and metrics["historical_log_Z_max_abs_difference"]
            <= bank_gate["historical_log_Z_max_abs_difference"]
            and metrics["historical_log_Z_bar_max_abs_difference"]
            <= bank_gate["historical_log_Z_bar_max_abs_difference"]
            and metrics["historical_covariance_logdet_max_abs_difference"]
            <= bank_gate["historical_covariance_logdet_max_abs_difference"]
            and metrics["cache"]["second_evaluation_bitwise_identical"]
            and metrics["cache"]["new_covariance_keys_second_evaluation"] == 0
        )
    gates = {
        "all_lineage_and_input_contracts": True,
        "all_values_finite": finite_pass,
        "dense_27_phase": dense_pass,
        "inside_atlas": inside_pass,
        "outside_slow_path": outside_pass,
        "historical_2048": bank_pass["bank_2048"],
        "historical_8192": bank_pass["bank_8192"],
    }
    failure = None if finite_pass else "nonfinite_or_numerical_failure"
    for gate_name, failure_name in (
        ("dense_27_phase", "dense_27_phase_mismatch"),
        ("inside_atlas", "inside_atlas_mismatch"),
        ("outside_slow_path", "outside_slow_path_or_lineage_mismatch"),
        ("historical_2048", "historical_2048_mismatch"),
        ("historical_8192", "historical_8192_mismatch"),
    ):
        if failure is None and not gates[gate_name]:
            failure = failure_name
            break
    gates["oracle_regression_pass"] = failure is None
    return gates, failure


def _all_finite_metrics(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite_metrics(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite_metrics(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, (float, np.floating, int, np.integer)) \
            and not isinstance(value, (bool, np.bool_)):
        return bool(np.isfinite(value))
    return True


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=_json_default)
        stream.write("\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_program(program: dict[str, Any], program_path: Path) -> dict[str, Any]:
    program_path = Path(program_path).resolve()
    if program_path != CANONICAL_PROGRAM.resolve():
        raise PermissionError("oracle-regression program path is not canonical")
    if program.get("schema") != "ouruniv-cf4-aggregate-evidence-oracle-regression-program-v1" \
            or program.get("status") != "frozen_before_exact_oracle_regression":
        raise RuntimeError("oracle-regression program is not frozen")
    design_item = program.get("design", {})
    if (
        _rooted(design_item.get("path", "")).resolve() != FROZEN_DESIGN.resolve()
        or design_item.get("sha256") != FROZEN_DESIGN_SHA256
    ):
        raise RuntimeError("oracle-regression program changed the frozen design")
    design = json.loads(FROZEN_DESIGN.read_text())
    validate_frozen_design(design, FROZEN_DESIGN)
    implementation = program.get("implementation", {})
    source_path = Path(__file__).resolve()
    if (
        _rooted(implementation.get("path", "")).resolve() != source_path
        or sha256_file(source_path) != implementation.get("sha256")
    ):
        raise RuntimeError("oracle-regression implementation hash mismatch")
    execution = program.get("execution", {})
    frozen_execution = design["execution"]
    if execution != {
        "host": frozen_execution["host"],
        "device": frozen_execution["device"],
        "worker_processes": frozen_execution["worker_processes"],
        "threads_per_worker": frozen_execution["threads_per_worker"],
        "multiprocessing_start_method": frozen_execution[
            "multiprocessing_start_method"
        ],
        "parent_block_size": 32,
        "geometry_batch_size": frozen_execution["geometry_batch_size"],
        "process_table_polling": False,
        "automatic_retry_or_scaling": False,
    }:
        raise RuntimeError("oracle-regression execution contract changed")
    storage = program.get("storage", {})
    frozen_storage = design["storage"]
    data = Path(frozen_storage["data_directory"])
    if storage != {
        "program": str(CANONICAL_PROGRAM.resolve()),
        "data_directory": str(data),
        "state_directory": frozen_storage["state_directory"],
        "result": str(data / frozen_storage["result"]),
        "arrays": str(data / frozen_storage["arrays"]),
        "manifest": str(data / frozen_storage["manifest"]),
        "exclusive_create_and_atomic_publication": True,
    }:
        raise RuntimeError("oracle-regression storage contract changed")
    decision = program.get("decision", {})
    if decision.get("regression_execution_authorized") is not True:
        raise PermissionError("canonical program did not authorize regression execution")
    for key in (
        "production_SMC_authorized",
        "conditional_field_bank_authorized",
        "parent_or_seed_selection_authorized",
        "PM_or_halo_finder_authorized",
        "RAMSES_authorized",
        "automatic_follow_on",
    ):
        if decision.get(key) is not False:
            raise RuntimeError(f"oracle-regression program opened {key}")
    return design


def _run_regression_core(
    design: dict[str, Any],
    *,
    worker_processes: int = 8,
    geometry_batch_size: int = 256,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.monotonic()
    lineage = design["fixed_lineage"]
    manifest, bounds = load_verified_atlas_manifest(
        Path(lineage["atlas_manifest"]["path"]),
        lineage["atlas_manifest"]["sha256"],
    )
    entries = manifest["entries"]
    if [entry["seed"] for entry in entries] != EXPECTED_PARENT_SEEDS.tolist():
        raise RuntimeError("atlas parent order changed")
    filter_item = lineage["density_filter"]
    filter_rfft = np.load(filter_item["path"], allow_pickle=False)
    if (
        filter_rfft.shape != tuple(filter_item["shape"])
        or filter_rfft.dtype != np.float32
        or not np.all(np.isfinite(filter_rfft))
    ):
        raise RuntimeError("density-filter runtime contract failed")
    filter_full = full_spectrum_from_rfft(filter_rfft)
    model = json.loads(_rooted(lineage["physical_model"]["path"]).read_text())
    peak = model["peak_constraints"]
    if float(peak["likelihood_sigma_delta"]) != 0.25:
        raise RuntimeError("physical-model likelihood sigma changed")
    target = target_vector(
        float(peak["centre_target_delta_linear"]),
        float(peak["six_shell_target_delta_linear"]),
    )

    selection_spec = design["random_atlas_control"]
    selection = deterministic_atlas_keys(
        bounds,
        master_seed=selection_spec["master_seed"],
        inside_count=selection_spec["inside_count"],
        outside_count=selection_spec["outside_count"],
        candidate_limit=selection_spec["candidate_limit_each"],
    )
    dense_arrays, dense_metrics = dense_phase_control(
        filter_full, target, entries
    )

    control_cache = ExactCovarianceCache(filter_full)
    inside_values, inside_cached, inside_diagnostics = evaluate_log_z_from_atlases(
        selection["inside_keys"], entries, bounds, filter_full, target,
        covariance_cache=control_cache,
    )
    outside_values, outside_slow, outside_diagnostics = evaluate_log_z_from_atlases(
        selection["outside_keys"], entries, bounds, filter_full, target,
        covariance_cache=control_cache,
    )
    if (
        inside_values != [tuple(map(int, row)) for row in selection["inside_keys"]]
        or outside_values != [tuple(map(int, row)) for row in selection["outside_keys"]]
    ):
        raise RuntimeError("atlas control evaluator changed selected-key order")
    combined = np.asarray(sorted_unique_geometry_keys([
        *inside_values, *outside_values
    ]), dtype=np.int16)
    direct_values, combined_direct, direct_evaluations = direct_log_z_all_parents(
        combined,
        entries,
        filter_full,
        target,
        control_cache,
        worker_processes=worker_processes,
    )
    direct_index = {key: index for index, key in enumerate(direct_values)}
    inside_direct = np.stack([
        combined_direct[direct_index[key]] for key in inside_values
    ])
    outside_direct = np.stack([
        combined_direct[direct_index[key]] for key in outside_values
    ])
    atlas_metrics = {
        "inside_diagnostics": inside_diagnostics,
        "outside_diagnostics": outside_diagnostics,
        "outside_slow_path_full_response_parent_evaluations": (
            len(entries) if outside_diagnostics["outside_atlas_key_count"] else 0
        ),
        "direct_full_response_parent_evaluations": direct_evaluations,
        "inside_log_Z_max_abs_difference": float(np.max(np.abs(
            inside_cached - inside_direct
        ))),
        "outside_log_Z_max_abs_difference": float(np.max(np.abs(
            outside_slow - outside_direct
        ))),
    }

    evaluator = ExactAtlasEvidenceEvaluator(
        Path(lineage["atlas_manifest"]["path"]),
        lineage["atlas_manifest"]["sha256"],
        Path(filter_item["path"]),
        filter_item["sha256"],
        _rooted(lineage["physical_model"]["path"]),
        lineage["physical_model"]["sha256"],
    )
    bank_arrays = {}
    bank_metrics = {}
    for label, lineage_key, count, unique_count, covariance_count in (
        ("bank_2048", "adaptation_2048", 2048, 2016, 1146),
        ("bank_8192", "adaptation_8192", 8192, 7756, 1755),
    ):
        arrays, metrics = historical_bank_regression(
            label,
            Path(lineage[lineage_key]["arrays"]),
            evaluator,
            draw_count=count,
            unique_geometry_count=unique_count,
            unique_covariance_count=covariance_count,
            batch_size=geometry_batch_size,
        )
        bank_arrays.update(arrays)
        bank_metrics[label] = metrics

    arrays = {
        "parent_seed": EXPECTED_PARENT_SEEDS,
        **selection,
        "inside_cached_log_Z": inside_cached,
        "inside_direct_log_Z": inside_direct,
        "outside_slow_log_Z": outside_slow,
        "outside_direct_log_Z": outside_direct,
        **bank_arrays,
        **dense_arrays,
    }
    expected_contract = design["arrays_contract"]
    for key, (dtype, shape) in expected_contract.items():
        if key not in arrays:
            raise RuntimeError(f"regression arrays lack {key}")
        value = np.asarray(arrays[key])
        if str(value.dtype) != dtype or list(value.shape) != shape \
                or not np.all(np.isfinite(value)):
            raise RuntimeError(f"regression array contract failed: {key}")
    gates, failure = _gate_results(
        design, dense_metrics, atlas_metrics, bank_metrics
    )
    passed = gates["oracle_regression_pass"]
    result = {
        "schema": design["result_schema"]["schema"],
        "status": (
            "complete_pass_exact_oracle_regression"
            if passed else "complete_fail_exact_oracle_regression"
        ),
        "lineage": {
            "design": str(FROZEN_DESIGN),
            "design_sha256": FROZEN_DESIGN_SHA256,
            "atlas_manifest_sha256": lineage["atlas_manifest"]["sha256"],
            "density_filter_sha256": filter_item["sha256"],
            "physical_model_sha256": lineage["physical_model"]["sha256"],
            "adaptation_2048_arrays_sha256": lineage["adaptation_2048"][
                "arrays_sha256"
            ],
            "adaptation_8192_arrays_sha256": lineage["adaptation_8192"][
                "arrays_sha256"
            ],
        },
        "execution": {
            "host_required": design["execution"]["host"],
            "worker_processes": worker_processes,
            "threads_per_worker": 1,
            "geometry_batch_size": geometry_batch_size,
            "elapsed_seconds": time.monotonic() - started,
        },
        "selection": {
            "master_seed": selection_spec["master_seed"],
            "inside_count": len(selection["inside_keys"]),
            "outside_count": len(selection["outside_keys"]),
            "inside_selection_sha256": array_sha256(
                selection["inside_keys"], selection["inside_candidate_index"]
            ),
            "outside_selection_sha256": array_sha256(
                selection["outside_keys"], selection["outside_candidate_index"]
            ),
        },
        "dense_phase": dense_metrics,
        "atlas_direct": atlas_metrics,
        "historical_banks": bank_metrics,
        "gates": gates,
        "failure_class": failure,
        "information_firewall": design["information_firewall"],
        "decision": {
            "oracle_regression_pass": passed,
            "production_SMC_program_design_authorized": passed,
            "production_SMC_execution_authorized": False,
            "conditional_field_bank_authorized": False,
            "parent_or_seed_selection_authorized": False,
            "PM_or_halo_finder_authorized": False,
            "RAMSES_authorized": False,
            "automatic_follow_on": False,
        },
    }
    return arrays, result


def execute_program(program_path: Path) -> dict[str, Any]:
    program_path = Path(program_path).resolve()
    if program_path != CANONICAL_PROGRAM.resolve():
        raise PermissionError("oracle-regression program path is not canonical")
    program = json.loads(program_path.read_text())
    design = validate_program(program, program_path)
    data_directory = Path(program["storage"]["data_directory"])
    if data_directory.exists():
        raise FileExistsError("refusing to reuse oracle-regression data directory")
    data_directory.mkdir(parents=True, exist_ok=False)
    arrays, result = _run_regression_core(
        design,
        worker_processes=program["execution"]["worker_processes"],
        geometry_batch_size=program["execution"]["geometry_batch_size"],
    )
    arrays_path = Path(program["storage"]["arrays"])
    result_path = Path(program["storage"]["result"])
    manifest_path = Path(program["storage"]["manifest"])
    atomic_npz(arrays_path, arrays)
    arrays_sha = sha256_file(arrays_path)
    result["lineage"].update({
        "program": str(program_path),
        "program_sha256": sha256_file(program_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "arrays": str(arrays_path),
        "arrays_sha256": arrays_sha,
    })
    atomic_json(result_path, result)
    result_sha = sha256_file(result_path)
    manifest = {
        "schema": "ouruniv-cf4-aggregate-evidence-oracle-regression-manifest-v1",
        "status": result["status"],
        "manifest": str(manifest_path),
        "program": str(program_path),
        "program_sha256": sha256_file(program_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "arrays": str(arrays_path),
        "arrays_sha256": arrays_sha,
        "result": str(result_path),
        "result_sha256": result_sha,
        "arrays_shape_dtype": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "gates": result["gates"],
        "information_firewall": result["information_firewall"],
        "decision": result["decision"],
    }
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    args = parser.parse_args()
    manifest = execute_program(Path(args.program))
    print(
        f"[oracle-regression] status={manifest['status']} "
        f"manifest={manifest['manifest']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
