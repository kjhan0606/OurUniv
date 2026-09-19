#!/usr/bin/env python3
"""Fail-closed parallel exact oracle and immutable evidence-cache primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable

import numpy as np

from cf4_aggregate_evidence_oracle import (
    AggregateEvidenceControllerOracle,
    ExactAtlasEvidenceEvaluator,
    PRODUCTION_PARENT_SEEDS,
    atlas_point_indices,
    logmeanexp_parent,
    parent_response_grid,
    points_from_geometry_key,
    sha256_file,
    sorted_unique_geometry_keys,
    vectorized_log_evidence,
)


WORKER_PROCESSES = 8
PARENTS_PER_BLOCK = 32
INSIDE_CONTROL_ROWS = np.asarray(
    [0, 68, 136, 204, 272, 341, 409, 477, 545, 613, 682, 750,
     818, 886, 954, 1023],
    dtype=np.int64,
)
OUTSIDE_CONTROL_ROWS = np.asarray(
    [0, 9, 18, 27, 36, 45, 54, 63], dtype=np.int64
)
REGRESSION_ARRAYS_SHA256 = (
    "40c52c48ae7899219476fe6c9f0308b7f68e70ac2493c9fdea26cd37923572e2"
)


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def fixed_parent_blocks(entries: Iterable[dict[str, Any]]) -> tuple[tuple[int, tuple[dict[str, Any], ...]], ...]:
    values = tuple(entries)
    seeds = tuple(int(row.get("seed", -1)) for row in values)
    if seeds != PRODUCTION_PARENT_SEEDS:
        raise RuntimeError("parents must be exactly seeds 3193 through 3448 in order")
    blocks = tuple(
        (start, values[start:start + PARENTS_PER_BLOCK])
        for start in range(0, len(values), PARENTS_PER_BLOCK)
    )
    if len(blocks) != WORKER_PROCESSES or any(
        len(block) != PARENTS_PER_BLOCK for _, block in blocks
    ):
        raise RuntimeError("parallel oracle requires eight parent blocks of 32")
    return blocks


def _validated_sorted_keys(keys: Iterable[Iterable[int]]) -> np.ndarray:
    original = [tuple(int(item) for item in row) for row in keys]
    values = np.asarray(sorted_unique_geometry_keys(original), dtype=np.int16)
    if values.ndim != 2 or values.shape[1:] != (6,) or len(values) == 0:
        raise ValueError("oracle keys must be nonempty sorted unique int16[n,6]")
    if not np.array_equal(values, np.asarray(original, dtype=np.int16)):
        raise ValueError("oracle input keys must already be sorted and unique")
    return values


def evaluate_fixed_parent_blocks(
    keys: np.ndarray,
    entries: Iterable[dict[str, Any]],
    block_worker: Callable[[tuple[int, tuple[dict[str, Any], ...], np.ndarray]], tuple[int, tuple[int, ...], np.ndarray]],
    *,
    parallel: bool,
) -> np.ndarray:
    """Evaluate and reassemble exactly eight fixed blocks in parent-seed order."""
    values = _validated_sorted_keys(keys)
    tasks = tuple((start, block, values) for start, block in fixed_parent_blocks(entries))
    if parallel:
        context = mp.get_context("fork")
        with context.Pool(processes=WORKER_PROCESSES) as pool:
            rows = pool.map(block_worker, tasks, chunksize=1)
    else:
        rows = [block_worker(task) for task in tasks]
    rows = sorted(rows, key=lambda row: row[0])
    expected_starts = list(range(0, 256, PARENTS_PER_BLOCK))
    if [row[0] for row in rows] != expected_starts:
        raise RuntimeError("parallel parent blocks were not reassembled in order")
    for start, seeds, block in rows:
        if seeds != PRODUCTION_PARENT_SEEDS[start:start + PARENTS_PER_BLOCK]:
            raise RuntimeError("parallel parent block seed order changed")
        if block.shape != (len(values), PARENTS_PER_BLOCK) \
                or block.dtype != np.float64 or not np.all(np.isfinite(block)):
            raise RuntimeError("parallel parent block output contract failed")
    output = np.concatenate([row[2] for row in rows], axis=1)
    if output.shape != (len(values), 256):
        raise RuntimeError("parallel oracle parent width changed")
    return output


_EXACT_STATIC_CONTEXT: dict[str, Any] | None = None


def _exact_parent_block_worker(task):
    start, keys, dynamic = task
    context = _EXACT_STATIC_CONTEXT
    if context is None:
        raise RuntimeError("forked exact-oracle static context is missing")
    entries = context["entries"][start:start + PARENTS_PER_BLOCK]
    points = dynamic["points"]
    atlas_indices = dynamic["atlas_indices"]
    inside = dynamic["inside"]
    log_z = np.empty((len(keys), len(entries)), dtype=np.float64)
    for column, entry in enumerate(entries):
        atlas = np.load(entry["atlas"], mmap_mode="r", allow_pickle=False)
        if atlas.shape != context["bounds"].shape or atlas.dtype != np.float64:
            raise RuntimeError("response-atlas shard header changed after preflight")
        means = np.empty((len(keys), 14), dtype=np.float64)
        if np.any(inside):
            index = atlas_indices[inside]
            means[inside] = atlas[index[..., 0], index[..., 1], index[..., 2]]
        if np.any(~inside):
            with np.load(entry["parent_field"], allow_pickle=False) as item:
                if int(item["sample_seed"]) != int(entry["seed"]):
                    raise RuntimeError("outside-atlas parent seed changed")
                coarse = item["s_out"].astype(np.float32)
            response = parent_response_grid(coarse, context["filter_full"])
            outside_points = points[~inside]
            means[~inside] = response[
                outside_points[..., 0],
                outside_points[..., 1],
                outside_points[..., 2],
            ]
        log_z[:, column] = vectorized_log_evidence(
            means,
            dynamic["targets"],
            dynamic["cholesky"],
            dynamic["logdet"],
        )
    return start, tuple(int(row["seed"]) for row in entries), log_z


class ParallelExactAtlasEvaluator:
    """Eight-process exact evaluator with one persistent covariance cache."""

    def __init__(
        self,
        manifest_path: Path,
        manifest_sha256: str,
        filter_path: Path,
        filter_sha256: str,
        physical_model_path: Path,
        physical_model_sha256: str,
    ):
        verified = ExactAtlasEvidenceEvaluator(
            manifest_path,
            manifest_sha256,
            filter_path,
            filter_sha256,
            physical_model_path,
            physical_model_sha256,
        )
        self.manifest = verified.manifest
        self.bounds = verified.bounds
        self.filter_full = verified.filter_full
        self.filter_full.flags.writeable = False
        self.targets = verified.targets
        self.covariance_cache = verified.covariance_cache
        fixed_parent_blocks(self.manifest["entries"])
        global _EXACT_STATIC_CONTEXT
        _EXACT_STATIC_CONTEXT = {
            "entries": tuple(self.manifest["entries"]),
            "bounds": self.bounds,
            "filter_full": self.filter_full,
        }
        self._pool = mp.get_context("fork").Pool(processes=WORKER_PROCESSES)

    def close(self) -> None:
        global _EXACT_STATIC_CONTEXT
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        _EXACT_STATIC_CONTEXT = None
        self.manifest = None
        self.bounds = None
        self.filter_full = None
        self.targets = None
        self.covariance_cache = None

    def __enter__(self):
        if self._pool is None:
            raise RuntimeError("parallel exact evaluator is already closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __call__(self, keys):
        if self._pool is None:
            raise RuntimeError("parallel exact evaluator is closed")
        values = _validated_sorted_keys(keys)
        points = np.stack([points_from_geometry_key(row) for row in values])
        flattened, inside_points = atlas_point_indices(
            points.reshape(-1, 3), self.bounds
        )
        atlas_indices = flattened.reshape(points.shape)
        inside = inside_points.reshape(len(values), 14).all(axis=1)
        cholesky, logdet, _ = self.covariance_cache.terms(values)
        dynamic = {
            "points": points,
            "atlas_indices": atlas_indices,
            "inside": inside,
            "targets": np.broadcast_to(self.targets, (len(values), 14)),
            "cholesky": cholesky,
            "logdet": logdet,
        }
        tasks = tuple(
            (start, values, dynamic)
            for start, _ in fixed_parent_blocks(self.manifest["entries"])
        )
        rows = sorted(
            self._pool.map(_exact_parent_block_worker, tasks, chunksize=1),
            key=lambda row: row[0],
        )
        if [row[0] for row in rows] != list(range(0, 256, 32)):
            raise RuntimeError("parallel parent blocks were not reassembled in order")
        for start, seeds, block in rows:
            if seeds != PRODUCTION_PARENT_SEEDS[start:start + 32] \
                    or block.shape != (len(values), 32) \
                    or block.dtype != np.float64 \
                    or not np.all(np.isfinite(block)):
                raise RuntimeError("parallel exact parent block contract failed")
        log_z = np.concatenate([row[2] for row in rows], axis=1)
        if log_z.shape != (len(values), 256):
            raise RuntimeError("parallel exact oracle parent width changed")
        return [tuple(int(item) for item in row) for row in values], log_z


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class CacheShard:
    path: str
    sha256: str
    row_count: int


class AppendOnlyEvidenceCache:
    """One-shot cache: no existing directory, import, resume, or replacement."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=False, exist_ok=False)
        self._keys: set[tuple[int, ...]] = set()
        self._shards: list[CacheShard] = []
        self._sealed = False

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    def append(self, keys: np.ndarray, log_z: np.ndarray) -> CacheShard:
        if self._sealed:
            raise RuntimeError("evidence cache is already sealed")
        values = np.asarray(keys)
        evidence = np.asarray(log_z)
        if values.dtype != np.int16 or values.ndim != 2 or values.shape[1:] != (6,):
            raise TypeError("cache keys must be int16[n,6]")
        tuples = [tuple(int(item) for item in row) for row in values]
        if tuples != sorted(set(tuples)):
            raise ValueError("cache shard keys must be sorted and unique")
        if any(key in self._keys for key in tuples):
            raise RuntimeError("cross-shard duplicate evidence key")
        if evidence.dtype != np.float64 or evidence.shape != (len(values), 256) \
                or not np.all(np.isfinite(evidence)):
            raise ValueError("cache log_Z must be finite float64[n,256]")
        log_z_bar = logmeanexp_parent(evidence)
        index = len(self._shards)
        path = self.directory / f"shard_{index:06d}.npz"
        _atomic_npz(path, {
            "keys": values,
            "log_Z": evidence,
            "log_Z_bar": np.asarray(log_z_bar, dtype=np.float64),
        })
        shard = CacheShard(str(path.resolve()), sha256_file(path), len(values))
        self._shards.append(shard)
        self._keys.update(tuples)
        return shard

    def seal(self) -> tuple[Path, str]:
        if self._sealed:
            raise RuntimeError("evidence cache manifest is already sealed")
        if any(
            not Path(row.path).is_file() or sha256_file(Path(row.path)) != row.sha256
            for row in self._shards
        ):
            raise RuntimeError("evidence cache shard changed before manifest seal")
        manifest = self.directory / "manifest.json"
        value = {
            "schema": "ouruniv-cf4-aggregate-evidence-cache-manifest-v1",
            "status": "complete_immutable_append_only_cache",
            "restart_or_checkpoint_imported": False,
            "shard_count": len(self._shards),
            "total_row_count": sum(row.row_count for row in self._shards),
            "shards": [row.__dict__ for row in self._shards],
        }
        _atomic_json(manifest, value)
        self._sealed = True
        return manifest, sha256_file(manifest)


@dataclass(frozen=True)
class RegressionControlResult:
    inside_max_abs_difference: float
    outside_max_abs_difference: float
    selection_sha256: str
    summary_path: str
    summary_sha256: str
    control_cache_discarded: bool
    control_evaluator_discarded: bool
    covariance_cache_identity_distinct: bool
    production_covariance_cached_key_count: int
    production_covariance_evaluation_batches: int
    production_cache_empty: bool


def run_sealed_regression_control(
    evaluator_factory,
    regression_arrays: Path,
    namespace_root: Path,
) -> tuple[RegressionControlResult, Any, AppendOnlyEvidenceCache]:
    """Run only the frozen control rows, discard them, and return an empty cache."""
    regression_arrays = Path(regression_arrays)
    if sha256_file(regression_arrays) != REGRESSION_ARRAYS_SHA256:
        raise RuntimeError("sealed regression arrays hash mismatch")
    root = Path(namespace_root)
    root.mkdir(parents=False, exist_ok=False)
    control_path = root / "control_cache"
    production_path = root / "production_cache"
    control = AppendOnlyEvidenceCache(control_path)
    with np.load(regression_arrays, allow_pickle=False) as item:
        if item["parent_seed"].dtype != np.int32 \
                or item["parent_seed"].shape != (256,) \
                or not np.array_equal(
                    item["parent_seed"], np.asarray(PRODUCTION_PARENT_SEEDS)
                ):
            raise RuntimeError("sealed control parent order is not 3193 through 3448")
        inside_keys = item["inside_keys"][INSIDE_CONTROL_ROWS]
        outside_keys = item["outside_keys"][OUTSIDE_CONTROL_ROWS]
        if inside_keys.dtype != np.int16 or inside_keys.shape != (16, 6) \
                or outside_keys.dtype != np.int16 or outside_keys.shape != (8, 6):
            raise RuntimeError("sealed control key dtype or shape changed")
        unsorted_keys = np.concatenate((inside_keys, outside_keys))
        unsorted_expected = np.concatenate((
            item["inside_direct_log_Z"][INSIDE_CONTROL_ROWS],
            item["outside_direct_log_Z"][OUTSIDE_CONTROL_ROWS],
        ))
        if unsorted_expected.dtype != np.float64 \
                or unsorted_expected.shape != (24, 256):
            raise RuntimeError("sealed control parent matrix contract changed")
        unsorted_membership = np.concatenate((
            np.ones(len(inside_keys), dtype=np.bool_),
            np.zeros(len(outside_keys), dtype=np.bool_),
        ))
        unsorted_source_row = np.concatenate((
            INSIDE_CONTROL_ROWS.astype(np.int64),
            OUTSIDE_CONTROL_ROWS.astype(np.int64),
        ))
        order = np.lexsort(tuple(unsorted_keys[:, column] for column in range(5, -1, -1)))
        keys = unsorted_keys[order]
        expected = unsorted_expected[order]
        membership = unsorted_membership[order]
        source_row = unsorted_source_row[order]
    if len(np.unique(keys, axis=0)) != 24:
        raise RuntimeError("sealed control keys are not globally unique")
    selection_sha = array_sha256(keys, membership, source_row)
    control_evaluator = evaluator_factory()
    control_evaluator_discarded = False
    try:
        if not hasattr(control_evaluator, "covariance_cache"):
            raise RuntimeError("control evaluator lacks a persistent covariance cache")
        if not callable(getattr(control_evaluator, "close", None)):
            raise RuntimeError("control evaluator lacks an explicit pool close")
        control_covariance = control_evaluator.covariance_cache
        if control_covariance.evaluated_covariance_keys != 0 \
                or control_covariance.evaluation_batches != 0:
            raise RuntimeError("control evaluator covariance cache did not start empty")
        evaluated, actual = control_evaluator(keys)
        expected_keys = [tuple(int(value) for value in row) for row in keys]
        if evaluated != expected_keys or actual.dtype != np.float64 \
                or actual.shape != (24, 256) or not np.all(np.isfinite(actual)):
            raise RuntimeError("sealed oracle control changed key or parent order")
        inside_difference = float(np.max(
            np.abs(actual[membership] - expected[membership])
        ))
        outside_difference = float(np.max(
            np.abs(actual[~membership] - expected[~membership])
        ))
        if max(inside_difference, outside_difference) > 1e-10:
            raise RuntimeError("sealed exact-oracle control failed")
        control.append(
            np.asarray(keys, dtype=np.int16),
            np.asarray(actual, dtype=np.float64),
        )
        control_manifest, control_manifest_sha = control.seal()
        summary_path = root / "sealed_oracle_control_summary.json"
        _atomic_json(summary_path, {
            "schema": "ouruniv-cf4-sealed-oracle-production-control-summary-v1",
            "status": "complete_pass_exact_24_row_control",
            "selection_sha256": selection_sha,
            "inside_source_rows": INSIDE_CONTROL_ROWS.tolist(),
            "outside_source_rows": OUTSIDE_CONTROL_ROWS.tolist(),
            "inside_row_count": int(np.count_nonzero(membership)),
            "outside_row_count": int(np.count_nonzero(~membership)),
            "global_unique_key_count": 24,
            "parent_seed_first": 3193,
            "parent_seed_last": 3448,
            "parent_count": 256,
            "inside_max_abs_difference": inside_difference,
            "outside_max_abs_difference": outside_difference,
            "control_cache_manifest_sha256": control_manifest_sha,
            "control_cache_reuse_authorized": False,
        })
    finally:
        close = getattr(control_evaluator, "close", None)
        if callable(close):
            close()
            control_evaluator_discarded = True
    control_evaluator = None
    shutil.rmtree(control_path)
    if control_path.exists():
        raise RuntimeError("control cache namespace was not discarded")
    production_evaluator = evaluator_factory()
    if not hasattr(production_evaluator, "covariance_cache"):
        raise RuntimeError("production evaluator lacks a persistent covariance cache")
    production_covariance = production_evaluator.covariance_cache
    distinct = production_covariance is not control_covariance
    production_cached = int(production_covariance.evaluated_covariance_keys)
    production_batches = int(production_covariance.evaluation_batches)
    if not distinct or production_cached != 0 or production_batches != 0:
        if hasattr(production_evaluator, "close"):
            production_evaluator.close()
        raise RuntimeError("fresh production covariance-cache hard gate failed")
    control_covariance = None
    production = AppendOnlyEvidenceCache(production_path)
    result = RegressionControlResult(
        inside_difference,
        outside_difference,
        selection_sha,
        str(summary_path.resolve()),
        sha256_file(summary_path),
        control_evaluator_discarded,
        True,
        distinct,
        production_cached,
        production_batches,
        production.shard_count == 0 and not any(production_path.iterdir()),
    )
    if not result.production_cache_empty:
        raise RuntimeError("production evidence cache did not start empty")
    return result, production_evaluator, production


class ShardedControllerOracle(AggregateEvidenceControllerOracle):
    """Controller firewall which persists only newly evaluated parent rows."""

    def __init__(self, evaluator, cache: AppendOnlyEvidenceCache):
        super().__init__(evaluator)
        self._shard_cache = cache

    def evaluate(self, midpoint_mpc_h, axis):
        before = set(self._cache)
        keys, aggregate = super().evaluate(midpoint_mpc_h, axis)
        new = sorted(set(self._cache).difference(before))
        if new:
            self._shard_cache.append(
                np.asarray(new, dtype=np.int16),
                np.stack([self._cache[key] for key in new]).astype(np.float64),
            )
        return keys, aggregate
