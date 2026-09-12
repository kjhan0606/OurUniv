"""Fail-closed future wiring for v6-open shared-schedule production.

The public entry is deliberately unauthorized.  Private helpers implement the
audited factory, lease, immutable-artifact, and postcheck contracts so they can
be tested without reserving or touching a science namespace.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
from types import MappingProxyType
from typing import Any, Callable, Mapping, NoReturn, Sequence

import numpy as np

import cf4_aggregate_evidence_smc_capability as base_capability
import cf4_aggregate_evidence_smc_v6_open_shared_schedule_production as capability
from cf4_aggregate_evidence_oracle import (
    AggregateEvidenceControllerOracle,
    PRODUCTION_PARENT_SEEDS,
    canonical_axis,
    geometry_key,
    logmeanexp_parent,
)
from cf4_aggregate_evidence_parallel_oracle import (
    ParallelExactAtlasEvaluator,
    run_sealed_regression_control,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PROGRAM = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_program.json"
PROGRAM_SHA256 = "54ffb61a9053a6e7935a7355a6d5a948184c4ded8dc585cf85b45d209a9b2dbc"
EXECUTION_DESIGN = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_design.json"
EXECUTION_DESIGN_SHA256 = "08d99219b88a232dc809b3a2c945381cbbcda1fac0c7202c1c2681a09be609aa"
EXECUTION_DESIGN_COMMIT = "e9c3489b0f1f26f0a2594ac0291566e62aade63e"
DATA_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1")
STATE_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_run")
RECEIPT_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_receipts")
CACHE_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_cache")
SCIENTIFIC_STATUSES = {
    "complete_pass_production_smc",
    "complete_scientific_fail_production_smc",
}
CONTROL_SELECTION_SHA256 = (
    "6902ab3f1a6c7fb2e5d9416d49ee956380c83eacc5d6b1b4fd2b678f64a59198"
)
PROGRAM_KEYS = {
    "schema", "status", "frozen_date", "purpose", "execution_design",
    "capability", "storage", "fixed_science", "evaluator_factory",
    "resource_contract", "artifact_contracts", "authorization", "next_step",
}
AUTHORIZATION_KEYS = {
    "grant_present_and_valid", "release_present_and_valid",
    "receipt_creation_authorized", "cache_population_authorized",
    "production_execution_authorized", "Slurm_submission_authorized",
    "retry_resume_retune_or_scale_up_authorized",
    "conditional_bank_authorized", "candidate_selection_authorized",
    "PM_authorized", "HOP_authorized", "RAMSES_authorized",
    "downstream_execution_authorized", "automatic_follow_on_authorized",
}
REPLICATE_ARRAY_KEYS = {
    "master_seed", "midpoint_mpc_h", "axis", "keys", "weights",
    "log_Z_bar", "ancestor_labels", "beta_history",
    "conditional_ESS_history", "particle_ESS_history",
    "log_normalizer_increment", "log_I_bar", "genealogical_ESS",
    "resampling_ancestors", "move_proposal_count", "move_acceptance_count",
    "q_scale_proposal_count", "q_scale_acceptance_count",
    "axis_scale_proposal_count", "axis_scale_acceptance_count",
}


class ExecutionContractError(RuntimeError):
    """Invalid provenance, lifecycle, or artifact state; never scientific."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExecutionContractError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ExecutionContractError(f"JSON root is not an object: {path}")
    return value


def _resolved(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _require_file(path: Path, digest: str, mode: str) -> None:
    if (
        not path.is_file()
        or sha256_file(path) != digest
        or f"{stat.S_IMODE(path.stat().st_mode):04o}" != mode
    ):
        raise ExecutionContractError(f"pinned file changed: {path}")


def load_canonical_program(*, verify_file_hashes: bool = True) -> dict[str, Any]:
    """Validate the unique unauthorized program and all inherited hard pins."""
    if sha256_file(CANONICAL_PROGRAM) != PROGRAM_SHA256:
        raise ExecutionContractError("canonical program hash changed")
    program = _read_json(CANONICAL_PROGRAM)
    if (
        set(program) != PROGRAM_KEYS
        or program.get("schema") != "ouruniv-cf4-v6-open-shared-schedule-production-program-v1"
        or program.get("status") != "frozen_unauthorized_v6_open_shared_schedule_production_program"
    ):
        raise ExecutionContractError("canonical program schema or keyset changed")
    design_row = program.get("execution_design", {})
    if design_row != {
        "path": str(EXECUTION_DESIGN.relative_to(ROOT)),
        "commit": EXECUTION_DESIGN_COMMIT,
        "sha256": EXECUTION_DESIGN_SHA256,
        "mode": "0644",
    }:
        raise ExecutionContractError("execution design lineage changed")
    if verify_file_hashes:
        _require_file(EXECUTION_DESIGN, EXECUTION_DESIGN_SHA256, "0644")
    design = _read_json(EXECUTION_DESIGN)
    if (
        design.get("schema") != "ouruniv-cf4-v6-open-shared-schedule-production-execution-design-v1"
        or design.get("status") != "prospective_execution_design_only_all_runtime_unauthorized"
        or design.get("lineage", {}).get("design_parent_commit")
        != "39127f775c9b9a8b218e3fd1ce6e903996ccd2a8"
    ):
        raise ExecutionContractError("execution design identity changed")
    if verify_file_hashes:
        for row in design.get("hard_pins", {}).values():
            if not isinstance(row, dict) or set(row) != {"path", "sha256", "mode", "role"}:
                raise ExecutionContractError("execution design hard-pin schema changed")
            _require_file(_resolved(row["path"]), row["sha256"], row["mode"])
    authorization = program.get("authorization", {})
    if set(authorization) != AUTHORIZATION_KEYS or any(
        value is not False for value in authorization.values()
    ):
        raise ExecutionContractError("runtime authorization is not fully closed")
    storage = program.get("storage", {})
    expected_storage = {
        "data_root": str(DATA_ROOT), "state_root": str(STATE_ROOT),
        "receipt_root": str(RECEIPT_ROOT), "cache_root": str(CACHE_ROOT),
        "grant_path": "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_grant.json",
        "release_path": "/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_execution_release.json",
        "release_manifest_path": "/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_execution_manifest.json",
        "all_runtime_paths_must_be_absent": True,
        "restart_checkpoint_or_cache_import": False,
    }
    if storage != expected_storage:
        raise ExecutionContractError("program storage contract changed")
    fixed = program.get("fixed_science", {})
    if (
        fixed.get("master_seeds") != list(capability.MASTER_SEEDS)
        or fixed.get("particles_per_replicate") != capability.PARTICLES
        or fixed.get("parent_seed_range_inclusive") != [3193, 3448]
        or fixed.get("parent_count") != 256
        or tuple(fixed.get("beta", ())) != capability.SHARED_BETA
        or fixed.get("stage_count_exact") != 5
        or fixed.get("MH_sweeps_per_stage") != 4
        or fixed.get("resampling_ESS_fraction") != 0.5
        or fixed.get("P_pool") != "evidence_weighted"
        or fixed.get("adaptive_beta") is not False
        or fixed.get("runtime_override") is not False
    ):
        raise ExecutionContractError("program fixed-science contract changed")
    resources = program.get("resource_contract", {})
    if (
        resources.get("backend") != "Lageunha_local_one_shot_CPU_runner"
        or resources.get("host_casefold_exact") != "lageunha"
        or resources.get("cpus_required") != 8
        or resources.get("required_MemAvailable_GiB") != 80
        or resources.get("minimum_free_disk_GiB") != 40
        or resources.get("timeout_command")
        != "/usr/bin/timeout --foreground --signal=TERM --kill-after=300s 12h"
        or resources.get("Slurm_submission") is not False
        or resources.get("syn101_execution") is not False
        or resources.get("process_table_polling") is not False
    ):
        raise ExecutionContractError("program resource contract changed")
    if verify_file_hashes:
        capability.load_frozen_contract()
    return program


def run_production_execution(*args: Any, **kwargs: Any) -> NoReturn:
    """Refuse before program loading, factory creation, or filesystem mutation."""
    raise PermissionError("v6-open shared-schedule production execution is unauthorized")


class _MemoizedAggregateOracle:
    """Expose only aggregate evidence to SMC while retaining terminal rows."""

    def __init__(
        self, evaluator: Any, shared_evidence: dict[tuple[int, ...], np.ndarray],
        expected_master_seed: int,
    ):
        self._evaluator = evaluator
        self._cache = shared_evidence
        self._expected_master_seed = int(expected_master_seed)
        self._terminal_calls = 0

    def evaluate(self, midpoint_mpc_h: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        midpoint = np.asarray(midpoint_mpc_h, dtype=np.float64)
        axes = np.asarray(axis, dtype=np.float64)
        if midpoint.ndim != 2 or midpoint.shape[1:] != (3,) or axes.shape != midpoint.shape:
            raise ValueError("controller geometry arrays must be aligned n-by-3")
        keys = [geometry_key(q, a) for q, a in zip(midpoint, axes)]
        missing = sorted(set(keys).difference(self._cache))
        if missing:
            evaluated_keys, log_z = self._evaluator(missing)
            if evaluated_keys != missing:
                raise ExecutionContractError("evaluator key order changed")
            values = np.asarray(log_z)
            if values.dtype != np.float64 or values.shape != (len(missing), 256) \
                    or not np.all(np.isfinite(values)):
                raise ExecutionContractError("evaluator parent evidence contract changed")
            for key, row in zip(evaluated_keys, values):
                if key in self._cache and not np.array_equal(self._cache[key], row):
                    raise ExecutionContractError("memoized parent evidence collision")
                frozen = np.asarray(row, dtype=np.float64).copy()
                frozen.flags.writeable = False
                self._cache[key] = frozen
        parent_log_z = np.stack([self._cache[key] for key in keys])
        return np.asarray(keys, dtype=np.int16), logmeanexp_parent(parent_log_z)

    def terminal_parent_log_z(self, master_seed: int, keys: np.ndarray) -> np.ndarray:
        if int(master_seed) != self._expected_master_seed or self._terminal_calls != 0:
            raise ExecutionContractError("terminal accessor seed or call count changed")
        value = AggregateEvidenceControllerOracle._validated_terminal_keys(keys)
        tuples = [tuple(int(item) for item in row) for row in value]
        if any(key not in self._cache for key in tuples):
            raise ExecutionContractError("terminal history contains unevaluated keys")
        self._terminal_calls += 1
        return np.stack([self._cache[key] for key in tuples])


@dataclass
class ExactEvaluatorLease:
    oracle: _MemoizedAggregateOracle
    evaluator: Any
    parent_seeds: np.ndarray
    provenance: capability.FreshReplicateProvenance
    closed: bool = False
    close_count: int = 0

    def terminal_parent_log_z(self, master_seed: int, keys: np.ndarray) -> np.ndarray:
        if self.closed:
            raise ExecutionContractError("terminal accessor called after lease close")
        return self.oracle.terminal_parent_log_z(master_seed, keys)

    def close(self) -> None:
        self.close_count += 1
        if self.close_count != 1:
            raise ExecutionContractError("lease close count exceeded one")
        close = getattr(self.evaluator, "close", None)
        if not callable(close):
            raise ExecutionContractError("evaluator lacks explicit close")
        close()
        self.closed = True


class FreshExactLeaseFactory:
    """Create one evaluator/covariance cache per replicate with shared evidence only."""

    def __init__(self, evaluator_builder: Callable[[], Any], cache_namespace: str):
        self._evaluator_builder = evaluator_builder
        self._cache_namespace = cache_namespace
        self._shared_evidence: dict[tuple[int, ...], np.ndarray] = {}
        self._lease_ids: set[int] = set()
        self._evaluator_ids: set[int] = set()
        self.created_leases: list[ExactEvaluatorLease] = []

    def __call__(self, master_seed: int, contract: capability.FrozenProductionContract) -> ExactEvaluatorLease:
        if master_seed not in capability.MASTER_SEEDS or contract.cache_namespace != self._cache_namespace:
            raise ExecutionContractError("factory seed or namespace changed")
        evaluator = self._evaluator_builder()
        oracle = _MemoizedAggregateOracle(
            evaluator, self._shared_evidence, master_seed
        )
        lease = ExactEvaluatorLease(
            oracle=oracle,
            evaluator=evaluator,
            parent_seeds=np.asarray(PRODUCTION_PARENT_SEEDS, dtype=np.int64),
            provenance=capability.FreshReplicateProvenance(
                master_seed=master_seed,
                fresh_token=f"v6-production-{master_seed}-{id(evaluator)}",
                evaluator_namespace=f"v6-production-evaluator-{master_seed}-{id(evaluator)}",
                cache_namespace=self._cache_namespace,
                pilot_state_reused=False,
                v5_state_reused=False,
                pilot_cache_reused=False,
                pilot_particles_reused=False,
                pilot_rng_state_reused=False,
                evaluator_closed=False,
                evaluator_close_count=0,
            ),
        )
        if id(lease) in self._lease_ids or id(evaluator) in self._evaluator_ids:
            lease.close()
            raise ExecutionContractError("factory reused a lease or evaluator")
        self._lease_ids.add(id(lease))
        self._evaluator_ids.add(id(evaluator))
        self.created_leases.append(lease)
        return lease

    def seal_cache(self, cache_directory: Path) -> tuple[Path, Path]:
        cache_directory = Path(cache_directory)
        if not cache_directory.is_dir() or any(cache_directory.iterdir()):
            raise ExecutionContractError("cache seal requires an empty reserved directory")
        keys = np.asarray(sorted(self._shared_evidence), dtype=np.int16)
        if keys.ndim != 2 or keys.shape[1:] != (6,) or len(keys) == 0:
            raise ExecutionContractError("production evidence memo is empty")
        log_z = np.stack([self._shared_evidence[tuple(row)] for row in keys])
        shard = cache_directory / "shard_000000.npz"
        _atomic_npz(shard, {
            "keys": keys,
            "log_Z": np.asarray(log_z, dtype=np.float64),
            "log_Z_bar": logmeanexp_parent(log_z),
        })
        _seal_read_only(shard)
        manifest = cache_directory / "manifest.json"
        _atomic_json(manifest, {
            "schema": "ouruniv-cf4-aggregate-evidence-cache-manifest-v1",
            "status": "complete_immutable_evidence_cache",
            "row_count": int(len(keys)),
            "shards": [_artifact_record(shard, lineage=PROGRAM_SHA256)],
        })
        _seal_read_only(manifest)
        return shard, manifest


def _actual_evaluator_builder(program: Mapping[str, Any]) -> Callable[[], Any]:
    del program
    design = _read_json(EXECUTION_DESIGN)
    pins = design["hard_pins"]

    def build() -> ParallelExactAtlasEvaluator:
        return ParallelExactAtlasEvaluator(
            _resolved(pins["response_atlas_manifest"]["path"]),
            pins["response_atlas_manifest"]["sha256"],
            _resolved(pins["density_filter"]["path"]),
            pins["density_filter"]["sha256"],
            _resolved(pins["physical_model"]["path"]),
            pins["physical_model"]["sha256"],
        )

    return build


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(dict(value), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    np.savez(temporary, **arrays)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _npz_array_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with np.load(path, allow_pickle=False) as item:
        for name in item.files:
            value = np.asarray(item[name])
            records.append({
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "finite": bool(np.all(np.isfinite(value))),
            })
    return records


def _artifact_record(path: Path, *, lineage: str) -> dict[str, Any]:
    path = Path(path)
    common = {
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
        "kind": "NPZ" if path.suffix == ".npz" else "JSON",
        "byte_count": path.stat().st_size, "lineage": lineage,
    }
    if path.suffix == ".npz":
        common["arrays"] = _npz_array_records(path)
    else:
        value = _read_json(path)
        common["schema"] = value.get("schema")
        common["status"] = value.get("status")
    return common


def _seal_read_only(path: Path) -> None:
    Path(path).chmod(0o444)


def _jsonable(value: Any) -> Any:
    """Convert frozen numerical records to an exact JSON-compatible tree."""
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _provenance_records(
    summary: capability.TerminalSummary,
) -> list[dict[str, Any]]:
    return [
        {
            "master_seed": row.master_seed,
            "fresh_token": row.fresh_token,
            "evaluator_namespace": row.evaluator_namespace,
            "cache_namespace": row.cache_namespace,
            "pilot_state_reused": row.pilot_state_reused,
            "v5_state_reused": row.v5_state_reused,
            "pilot_cache_reused": row.pilot_cache_reused,
            "pilot_particles_reused": row.pilot_particles_reused,
            "pilot_rng_state_reused": row.pilot_rng_state_reused,
            "evaluator_closed": row.evaluator_closed,
            "evaluator_close_count": row.evaluator_close_count,
        }
        for row in summary.provenance
    ]


def _publish_terminal_products(
    data_directory: Path,
    products: Sequence[capability.FreshReplicateProduct],
    summary: capability.TerminalSummary,
) -> tuple[list[Path], Path]:
    data_directory = Path(data_directory)
    replicate_paths = []
    for index, product in enumerate(products):
        arrays = base_capability._replicate_arrays(product.replicate)
        path = data_directory / f"replicate_{index}.npz"
        _atomic_npz(path, arrays)
        _seal_read_only(path)
        replicate_paths.append(path)
    terminal = data_directory / "terminal_parent_frozen.npz"
    _atomic_npz(terminal, {
        "master_seed": np.asarray(summary.master_seed, dtype=np.int64),
        "parent_seed": np.asarray(PRODUCTION_PARENT_SEEDS, dtype=np.int32),
        "log_I_bar": np.asarray(summary.log_I_bar, dtype=np.float64),
        "P_rep": np.asarray(summary.P_rep, dtype=np.float64),
        "P_pool": np.asarray(summary.P_pool, dtype=np.float64),
    })
    _seal_read_only(terminal)
    return replicate_paths, terminal


def _validated_science_artifacts(
    replicate_paths: Sequence[Path], terminal_path: Path
) -> tuple[Mapping[str, bool], list[dict[str, np.ndarray]], dict[str, np.ndarray]]:
    if len(replicate_paths) != 4 or any(
        not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o444
        for path in (*replicate_paths, terminal_path)
    ):
        raise ExecutionContractError("terminal artifacts are not sealed")
    replicate_log_i = []
    replicate_arrays: list[dict[str, np.ndarray]] = []
    for expected_seed, path in zip(capability.MASTER_SEEDS, replicate_paths):
        with np.load(path, allow_pickle=False) as item:
            if set(item.files) != REPLICATE_ARRAY_KEYS:
                raise ExecutionContractError("replicate artifact keyset changed")
            arrays = {name: np.asarray(item[name]) for name in item.files}
        exact = {
            "master_seed": (np.dtype("int64"), ()),
            "midpoint_mpc_h": (np.dtype("float64"), (2048, 3)),
            "axis": (np.dtype("float64"), (2048, 3)),
            "keys": (np.dtype("int16"), (2048, 6)),
            "weights": (np.dtype("float64"), (2048,)),
            "log_Z_bar": (np.dtype("float64"), (2048,)),
            "ancestor_labels": (np.dtype("int64"), (2048,)),
            "beta_history": (np.dtype("float64"), (6,)),
            "conditional_ESS_history": (np.dtype("float64"), (5,)),
            "particle_ESS_history": (np.dtype("float64"), (6,)),
            "log_normalizer_increment": (np.dtype("float64"), (5,)),
            "log_I_bar": (np.dtype("float64"), ()),
            "genealogical_ESS": (np.dtype("float64"), ()),
            "move_proposal_count": (np.dtype("int64"), (5, 4, 4)),
            "move_acceptance_count": (np.dtype("int64"), (5, 4, 4)),
            "q_scale_proposal_count": (np.dtype("int64"), (5, 4, 3)),
            "q_scale_acceptance_count": (np.dtype("int64"), (5, 4, 3)),
            "axis_scale_proposal_count": (np.dtype("int64"), (5, 4, 3)),
            "axis_scale_acceptance_count": (np.dtype("int64"), (5, 4, 3)),
        }
        if any(
            arrays[name].dtype != dtype or arrays[name].shape != shape
            for name, (dtype, shape) in exact.items()
        ):
            raise ExecutionContractError("replicate artifact dtype or shape changed")
        resampling = arrays["resampling_ancestors"]
        if resampling.dtype != np.int64 or resampling.ndim != 2 \
                or resampling.shape[1:] != (2048,):
            raise ExecutionContractError("replicate resampling shape changed")
        if not all(np.all(np.isfinite(value)) for value in arrays.values()):
            raise ExecutionContractError("replicate artifact contains nonfinite values")
        reconstructed = np.asarray([
            geometry_key(q, a)
            for q, a in zip(arrays["midpoint_mpc_h"], arrays["axis"])
        ], dtype=np.int16)
        recomputed_gess = base_capability.genealogical_ess(
            arrays["ancestor_labels"], 2048
        )
        if (
            int(arrays["master_seed"]) != expected_seed
            or not np.array_equal(arrays["beta_history"], np.asarray(capability.SHARED_BETA))
            or not np.array_equal(arrays["keys"], reconstructed)
            or not np.allclose(
                np.linalg.norm(arrays["axis"], axis=1), 1.0,
                rtol=0.0, atol=3e-15,
            )
            or not np.allclose(
                arrays["axis"],
                np.stack([canonical_axis(value) for value in arrays["axis"]]),
                rtol=0.0, atol=3e-15,
            )
            or np.any(arrays["weights"] < 0.0)
            or not math.isclose(float(arrays["weights"].sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
            or np.any(arrays["ancestor_labels"] < 0)
            or np.any(arrays["ancestor_labels"] >= 2048)
            or np.any(arrays["conditional_ESS_history"] <= 0.0)
            or np.any(arrays["conditional_ESS_history"] > 2048.0 * (1.0 + 1e-12))
            or np.any(arrays["particle_ESS_history"] <= 0.0)
            or np.any(arrays["particle_ESS_history"] > 2048.0 * (1.0 + 1e-12))
            or not math.isclose(
                float(arrays["log_normalizer_increment"].sum()),
                float(arrays["log_I_bar"]), rel_tol=1e-12, abs_tol=1e-12,
            )
            or not math.isclose(
                recomputed_gess, float(arrays["genealogical_ESS"]),
                rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            raise ExecutionContractError("replicate artifact history contract changed")
        expected_resampling = int(np.count_nonzero(
            arrays["particle_ESS_history"][1:] < 1024.0
        ))
        move_proposal = arrays["move_proposal_count"]
        move_acceptance = arrays["move_acceptance_count"]
        q_proposal = arrays["q_scale_proposal_count"]
        q_acceptance = arrays["q_scale_acceptance_count"]
        axis_proposal = arrays["axis_scale_proposal_count"]
        axis_acceptance = arrays["axis_scale_acceptance_count"]
        if (
            resampling.shape[0] != expected_resampling
            or np.any(resampling < 0) or np.any(resampling >= 2048)
            or np.any(move_proposal < 0) or np.any(move_acceptance < 0)
            or np.any(q_proposal < 0) or np.any(q_acceptance < 0)
            or np.any(axis_proposal < 0) or np.any(axis_acceptance < 0)
            or np.any(move_acceptance > move_proposal)
            or np.any(q_acceptance > q_proposal)
            or np.any(axis_acceptance > axis_proposal)
            or not np.all(move_proposal.sum(axis=2) == 2048)
            or not np.array_equal(
                q_proposal.sum(axis=2), move_proposal[..., 0] + move_proposal[..., 2]
            )
            or not np.array_equal(
                axis_proposal.sum(axis=2), move_proposal[..., 1] + move_proposal[..., 2]
            )
            or not np.array_equal(
                q_acceptance.sum(axis=2), move_acceptance[..., 0] + move_acceptance[..., 2]
            )
            or not np.array_equal(
                axis_acceptance.sum(axis=2), move_acceptance[..., 1] + move_acceptance[..., 2]
            )
        ):
            raise ExecutionContractError("replicate move or resampling history changed")
        replicate_log_i.append(float(arrays["log_I_bar"]))
        replicate_arrays.append(arrays)
    with np.load(terminal_path, allow_pickle=False) as item:
        if set(item.files) != {"master_seed", "parent_seed", "log_I_bar", "P_rep", "P_pool"}:
            raise ExecutionContractError("terminal parent artifact keyset changed")
        terminal = {name: np.asarray(item[name]) for name in item.files}
    terminal_exact = {
        "master_seed": (np.dtype("int64"), (4,)),
        "parent_seed": (np.dtype("int32"), (256,)),
        "log_I_bar": (np.dtype("float64"), (4,)),
        "P_rep": (np.dtype("float64"), (4, 256)),
        "P_pool": (np.dtype("float64"), (256,)),
    }
    if any(
        terminal[name].dtype != dtype or terminal[name].shape != shape
        or not np.all(np.isfinite(terminal[name]))
        for name, (dtype, shape) in terminal_exact.items()
    ):
        raise ExecutionContractError("terminal parent dtype shape or finite contract changed")
    expected_pool, _ = base_capability.pool_parent_probabilities(
        terminal["log_I_bar"], terminal["P_rep"]
    )
    if (
        not np.array_equal(terminal["master_seed"], np.asarray(capability.MASTER_SEEDS))
        or not np.array_equal(terminal["parent_seed"], np.asarray(PRODUCTION_PARENT_SEEDS, dtype=np.int32))
        or not np.array_equal(terminal["log_I_bar"], np.asarray(replicate_log_i))
        or np.any(terminal["P_rep"] < 0.0) or np.any(terminal["P_pool"] < 0.0)
        or not np.allclose(terminal["P_rep"].sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        or not np.allclose(terminal["P_pool"], expected_pool, rtol=0.0, atol=1e-12)
    ):
        raise ExecutionContractError("terminal parent normalization or pooling changed")
    gates = MappingProxyType({name: True for name in capability.VALIDITY_GATE_KEYS})
    return gates, replicate_arrays, terminal


def _validated_validity_gates(
    replicate_paths: Sequence[Path], terminal_path: Path
) -> Mapping[str, bool]:
    gates, _, _ = _validated_science_artifacts(replicate_paths, terminal_path)
    return gates


def _reload_terminal_summary(
    replicate_paths: Sequence[Path], terminal_path: Path,
    provenance_rows: Any, contract: capability.FrozenProductionContract,
) -> tuple[
    capability.TerminalSummary, Mapping[str, bool], list[dict[str, np.ndarray]]
]:
    validity, replicates, terminal = _validated_science_artifacts(
        replicate_paths, terminal_path
    )
    if not isinstance(provenance_rows, list) or len(provenance_rows) != 4:
        raise ExecutionContractError("replicate provenance count changed")
    expected_keys = {
        "master_seed", "fresh_token", "evaluator_namespace", "cache_namespace",
        "pilot_state_reused", "v5_state_reused", "pilot_cache_reused",
        "pilot_particles_reused", "pilot_rng_state_reused", "evaluator_closed",
        "evaluator_close_count",
    }
    provenance = []
    for expected_seed, row in zip(capability.MASTER_SEEDS, provenance_rows):
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise ExecutionContractError("replicate provenance schema changed")
        value = capability.FreshReplicateProvenance(**row)
        if (
            value.master_seed != expected_seed
            or value.cache_namespace != contract.cache_namespace
            or any((
                value.pilot_state_reused, value.v5_state_reused,
                value.pilot_cache_reused, value.pilot_particles_reused,
                value.pilot_rng_state_reused,
            ))
            or value.evaluator_closed is not True
            or value.evaluator_close_count != 1
            or not value.fresh_token or not value.evaluator_namespace
        ):
            raise ExecutionContractError("replicate provenance contract changed")
        token_prefix = f"v6-production-{expected_seed}-"
        evaluator_prefix = f"v6-production-evaluator-{expected_seed}-"
        token_suffix = value.fresh_token.removeprefix(token_prefix)
        evaluator_suffix = value.evaluator_namespace.removeprefix(evaluator_prefix)
        if (
            not value.fresh_token.startswith(token_prefix)
            or not value.evaluator_namespace.startswith(evaluator_prefix)
            or not token_suffix.isdecimal()
            or evaluator_suffix != token_suffix
        ):
            raise ExecutionContractError("replicate provenance token changed")
        provenance.append(value)
    if (
        len({row.fresh_token for row in provenance}) != 4
        or len({row.evaluator_namespace for row in provenance}) != 4
    ):
        raise ExecutionContractError("replicate provenance is not independent")
    pooled, pooled_log_i = base_capability.pool_parent_probabilities(
        terminal["log_I_bar"], terminal["P_rep"]
    )
    if not np.allclose(pooled, terminal["P_pool"], rtol=0.0, atol=1e-12):
        raise ExecutionContractError("reloaded evidence-weighted pool changed")
    summary = capability.TerminalSummary(
        master_seed=terminal["master_seed"],
        beta_history=np.stack([row["beta_history"] for row in replicates]),
        log_I_bar=terminal["log_I_bar"],
        P_rep=terminal["P_rep"],
        P_pool=terminal["P_pool"],
        P_rep_arithmetic_mean_diagnostic_only=np.mean(terminal["P_rep"], axis=0),
        pooled_log_I_bar=float(pooled_log_i),
        genealogical_ESS=np.asarray(
            [row["genealogical_ESS"] for row in replicates], dtype=np.float64
        ),
        provenance=tuple(provenance),
    )
    capability._validate_terminal_summary(summary, contract)
    return summary, validity, replicates


def _build_manifest(
    program: Mapping[str, Any], data_directory: Path, result_path: Path,
    artifact_paths: Sequence[Path], outcome: capability.ProductionDecision,
) -> dict[str, Any]:
    records = [_artifact_record(path, lineage=PROGRAM_SHA256) for path in artifact_paths]
    result = _read_json(result_path)
    return {
        "schema": "ouruniv-cf4-aggregate-evidence-smc-production-manifest-v1",
        "status": "complete_immutable_production_manifest",
        "science_status": result["status"],
        "program_sha256": PROGRAM_SHA256,
        "execution_design_sha256": EXECUTION_DESIGN_SHA256,
        "outcome_kind": outcome.outcome_kind,
        "failure_class": outcome.primary_failure,
        "failed_channels": list(outcome.failed_channels),
        "validity_gates": result["validity_gates"],
        "pre_CF4_gates": result["pre_CF4_gates"],
        "CF4_gates": result["CF4_gates"],
        "all_gates": result["all_gates"],
        "pre_CF4_metrics": result["pre_CF4_metrics"],
        "CF4_metrics": result["CF4_metrics"],
        "replicate_provenance": result["replicate_provenance"],
        "artifacts": records,
        "result": _artifact_record(result_path, lineage=PROGRAM_SHA256),
        "authorization": dict(program["authorization"]),
    }


def _execute_reserved_synthetic_test_only(
    program: Mapping[str, Any], contract: capability.FrozenProductionContract,
    data_directory: Path,
    *, lease_factory: capability.FreshReplicateFactory,
    control_runner: Callable[[], Mapping[str, Any]],
    cf4_gate_provider: Callable[[capability.TerminalSummary], Mapping[str, Any]],
    cache_directory: Path,
) -> dict[str, Any]:
    """Injected synthetic-test executor; never a canonical science entry."""
    if dict(program) != load_canonical_program(verify_file_hashes=False):
        raise ExecutionContractError("private executor program differs from canonical")
    capability._require_contract_identity(contract)
    data_directory = Path(data_directory)
    cache_directory = Path(cache_directory)
    if (
        not data_directory.is_dir() or any(data_directory.iterdir())
        or not cache_directory.is_dir() or any(cache_directory.iterdir())
        or data_directory.resolve() == cache_directory.resolve()
    ):
        raise ExecutionContractError("private executor requires distinct empty reserved directories")
    control = dict(control_runner())
    if not {
        key: control.get(key) for key in (
            "control_rows", "control_evaluator_discarded",
            "control_cache_discarded", "production_cache_empty",
        )
    } == {
        "control_rows": 24,
        "control_evaluator_discarded": True,
        "control_cache_discarded": True,
        "production_cache_empty": True,
    }:
        raise ExecutionContractError("sealed oracle control contract failed")
    control_path = data_directory / "sealed_oracle_control_summary.json"
    control_payload = control.get("sealed_summary")
    if control_payload is None:
        control_payload = {
            "schema": "ouruniv-cf4-aggregate-evidence-sealed-oracle-control-v1",
            "status": "complete_pass_sealed_oracle_control",
            **control,
        }
    elif (
        not isinstance(control_payload, dict)
        or control_payload.get("schema")
        != "ouruniv-cf4-sealed-oracle-production-control-summary-v1"
        or control_payload.get("status") != "complete_pass_exact_24_row_control"
        or control_payload.get("global_unique_key_count") != 24
        or control_payload.get("parent_count") != 256
        or max(
            float(control_payload.get("inside_max_abs_difference", np.inf)),
            float(control_payload.get("outside_max_abs_difference", np.inf)),
        ) > 1e-10
    ):
        raise ExecutionContractError("canonical sealed control summary changed")
    _atomic_json(control_path, control_payload)
    _seal_read_only(control_path)
    products = capability._run_four_fresh_replicates(lease_factory, contract)
    summary = capability.build_terminal_summary(products, contract)
    replicate_paths, terminal_path = _publish_terminal_products(
        data_directory, products, summary
    )
    validity = _validated_validity_gates(replicate_paths, terminal_path)
    pre_cf4 = capability.evaluate_pre_cf4_diagnostics(summary, contract)
    cf4 = dict(cf4_gate_provider(summary))
    if set(cf4) != {"gates", "parent_seed", "deviance", "metrics"}:
        raise ExecutionContractError("CF4 provider result contract changed")
    cf4_gates = dict(cf4["gates"])
    decision = capability.classify_complete_gate_set(
        validity, pre_cf4, cf4_gates, contract
    )
    if decision.outcome_kind == "invalid":
        raise ExecutionContractError("invalid gate outcome cannot be scientific COMPLETE")
    science_status = (
        "complete_pass_production_smc"
        if decision.outcome_kind == "pass"
        else "complete_scientific_fail_production_smc"
    )
    cf4_arrays_path = data_directory / "post_terminal_cf4_gates.npz"
    parent_seed = np.asarray(cf4["parent_seed"])
    deviance = np.asarray(cf4["deviance"])
    if parent_seed.dtype != np.int32 or parent_seed.shape != (256,) \
            or not np.array_equal(parent_seed, np.asarray(PRODUCTION_PARENT_SEEDS, dtype=np.int32)) \
            or deviance.dtype != np.float64 or deviance.shape != (256,) \
            or not np.all(np.isfinite(deviance)):
        raise ExecutionContractError("CF4 arrays contract changed")
    _atomic_npz(cf4_arrays_path, {
        "parent_seed": parent_seed,
        "deviance": deviance,
        "P_pool": np.asarray(summary.P_pool, dtype=np.float64),
    })
    _seal_read_only(cf4_arrays_path)
    cf4_path = data_directory / "post_terminal_cf4_gates.json"
    _atomic_json(cf4_path, {
        "schema": "ouruniv-cf4-aggregate-evidence-post-terminal-cf4-gates-v1",
        "status": science_status,
        "terminal_parent_frozen": str(terminal_path.resolve()),
        "terminal_parent_frozen_sha256": sha256_file(terminal_path),
        "post_terminal_arrays": str(cf4_arrays_path.resolve()),
        "post_terminal_arrays_sha256": sha256_file(cf4_arrays_path),
        "pre_CF4_metrics": dict(pre_cf4.metrics),
        "CF4_metrics": dict(cf4["metrics"]),
        "gates": cf4_gates,
        "failure_class": decision.primary_failure,
    })
    _seal_read_only(cf4_path)
    seal_cache = getattr(lease_factory, "seal_cache", None)
    if not callable(seal_cache):
        raise ExecutionContractError("lease factory lacks immutable cache seal")
    cache_shard_path, cache_manifest_path = seal_cache(cache_directory)
    result_path = data_directory / "result.json"
    result = {
        "schema": "ouruniv-cf4-aggregate-evidence-smc-production-result-v1",
        "status": science_status,
        "outcome_kind": decision.outcome_kind,
        "failure_class": decision.primary_failure,
        "failed_channels": list(decision.failed_channels),
        "validity_gates": _jsonable(validity),
        "pre_CF4_gates": _jsonable(pre_cf4.gates),
        "CF4_gates": _jsonable(cf4_gates),
        "all_gates": _jsonable(decision.all_gates),
        "pre_CF4_metrics": _jsonable(pre_cf4.metrics),
        "CF4_metrics": _jsonable(cf4["metrics"]),
        "replicate_provenance": _provenance_records(summary),
        "program_sha256": PROGRAM_SHA256,
        "execution_design_sha256": EXECUTION_DESIGN_SHA256,
        "authorization": dict(program["authorization"]),
    }
    _atomic_json(result_path, result)
    _seal_read_only(result_path)
    artifacts = [
        control_path, cache_shard_path, cache_manifest_path,
        *replicate_paths, terminal_path, cf4_arrays_path, cf4_path,
    ]
    manifest = _build_manifest(program, data_directory, result_path, artifacts, decision)
    manifest_path = data_directory / "manifest.json"
    _atomic_json(manifest_path, manifest)
    _seal_read_only(manifest_path)
    validate_published_bundle(data_directory)
    return result


def _canonical_regression_arrays() -> Path:
    design = _read_json(EXECUTION_DESIGN)
    row = design["hard_pins"]["oracle_regression_postmortem"]
    record_path = _resolved(row["path"])
    _require_file(record_path, row["sha256"], row["mode"])
    record = _read_json(record_path)
    arrays = record.get("immutable_original_artifacts", {}).get("arrays", {})
    path = Path(str(arrays.get("path", "")))
    digest = str(arrays.get("sha256", ""))
    if not path.is_file() or sha256_file(path) != digest:
        raise ExecutionContractError("sealed regression arrays lineage changed")
    return path


def _canonical_cf4_gate_provider(
    summary: capability.TerminalSummary,
) -> Mapping[str, Any]:
    calibration = base_capability._load_pinned_calibration()
    evaluated = base_capability.evaluate_cf4_gates(
        calibration.deviance,
        np.asarray(summary.P_pool, dtype=np.float64),
        calibration.reference_q99,
        calibration.reference_q99p5,
    )
    return {
        "gates": dict(evaluated["gates"]),
        "metrics": dict(evaluated["metrics"]),
        "parent_seed": np.asarray(calibration.parent_seed, dtype=np.int32),
        "deviance": np.asarray(calibration.deviance, dtype=np.float64),
    }


def _execute_reserved_canonical_private(
    program: Mapping[str, Any], contract: capability.FrozenProductionContract,
    data_directory: Path, cache_directory: Path,
) -> dict[str, Any]:
    """Canonical no-override science wiring, still unreachable publicly."""
    if dict(program) != load_canonical_program(verify_file_hashes=True):
        raise ExecutionContractError("canonical executor program changed")
    data_directory = Path(data_directory)
    control_root = data_directory / ".sealed_control_runtime"
    evaluator_builder = _actual_evaluator_builder(program)

    def canonical_control() -> Mapping[str, Any]:
        returned_evaluator = None
        try:
            control, returned_evaluator, production_cache = run_sealed_regression_control(
                evaluator_builder, _canonical_regression_arrays(), control_root
            )
            if (
                not control.control_evaluator_discarded
                or not control.control_cache_discarded
                or not control.covariance_cache_identity_distinct
                or control.production_covariance_cached_key_count != 0
                or control.production_covariance_evaluation_batches != 0
                or not control.production_cache_empty
                or production_cache.shard_count != 0
            ):
                raise ExecutionContractError("sealed 24-row control did not isolate production")
            summary = _read_json(Path(control.summary_path))
            return {
                "control_rows": 24,
                "control_evaluator_discarded": True,
                "control_cache_discarded": True,
                "production_cache_empty": True,
                "sealed_summary": summary,
            }
        finally:
            if returned_evaluator is not None:
                close = getattr(returned_evaluator, "close", None)
                if not callable(close):
                    raise ExecutionContractError("control return evaluator lacks close")
                close()
            if control_root.exists():
                shutil.rmtree(control_root)

    lease_factory = FreshExactLeaseFactory(
        evaluator_builder, contract.cache_namespace
    )
    return _execute_reserved_synthetic_test_only(
        program,
        contract,
        data_directory,
        lease_factory=lease_factory,
        control_runner=canonical_control,
        cf4_gate_provider=_canonical_cf4_gate_provider,
        cache_directory=Path(cache_directory),
    )


def _validate_record(record: Mapping[str, Any]) -> None:
    common = {"path", "sha256", "mode", "kind", "byte_count", "lineage"}
    if (
        not common.issubset(record)
        or record.get("mode") != "0444"
        or record.get("lineage") != PROGRAM_SHA256
    ):
        raise ExecutionContractError("manifest artifact common record changed")
    path = Path(str(record["path"]))
    if (
        path.is_symlink() or not path.is_file()
        or sha256_file(path) != record["sha256"]
        or path.stat().st_size != record["byte_count"]
        or f"{stat.S_IMODE(path.stat().st_mode):04o}" != record["mode"]
    ):
        raise ExecutionContractError("manifest artifact no longer matches disk")
    if record["kind"] == "NPZ":
        if set(record) != common | {"arrays"} or record["arrays"] != _npz_array_records(path):
            raise ExecutionContractError("manifest NPZ record changed")
    elif record["kind"] == "JSON":
        value = _read_json(path)
        if set(record) != common | {"schema", "status"} \
                or record["schema"] != value.get("schema") \
                or record["status"] != value.get("status"):
            raise ExecutionContractError("manifest JSON record changed")
    else:
        raise ExecutionContractError("manifest artifact kind changed")


def validate_published_bundle(data_directory: Path) -> dict[str, Any]:
    """Independently reconstruct every scientific decision from sealed arrays."""
    data_directory = Path(data_directory)
    result_path = data_directory / "result.json"
    manifest_path = data_directory / "manifest.json"
    if not result_path.is_file() or not manifest_path.is_file():
        raise ExecutionContractError("result or manifest is missing")
    if any(stat.S_IMODE(path.stat().st_mode) != 0o444 for path in (result_path, manifest_path)):
        raise ExecutionContractError("result or manifest is writable")
    result = _read_json(result_path)
    manifest = _read_json(manifest_path)
    if (
        result.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-production-result-v1"
        or result.get("status") not in SCIENTIFIC_STATUSES
        or result.get("outcome_kind") not in {"pass", "scientific_fail"}
        or result.get("program_sha256") != PROGRAM_SHA256
        or result.get("execution_design_sha256") != EXECUTION_DESIGN_SHA256
        or manifest.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-production-manifest-v1"
        or manifest.get("status") != "complete_immutable_production_manifest"
        or manifest.get("program_sha256") != PROGRAM_SHA256
        or manifest.get("execution_design_sha256") != EXECUTION_DESIGN_SHA256
        or manifest.get("science_status") != result.get("status")
        or manifest.get("outcome_kind") != result.get("outcome_kind")
        or manifest.get("failure_class") != result.get("failure_class")
        or result.get("authorization") != manifest.get("authorization")
        or set(result.get("authorization", {})) != AUTHORIZATION_KEYS
        or any(result.get("authorization", {}).values())
        or manifest.get("failed_channels") != result.get("failed_channels")
    ):
        raise ExecutionContractError("result or manifest semantic contract changed")
    records = manifest.get("artifacts")
    if not isinstance(records, list) or len(records) != 10:
        raise ExecutionContractError("manifest artifact count changed")
    for record in records:
        _validate_record(record)
    _validate_record(manifest.get("result", {}))
    if Path(str(manifest["result"]["path"])).resolve() != result_path.resolve():
        raise ExecutionContractError("manifest result path is noncanonical")
    by_path: dict[Path, Mapping[str, Any]] = {}
    for row in records:
        path = Path(str(row["path"])).resolve()
        if path in by_path:
            raise ExecutionContractError("manifest contains duplicate artifact paths")
        by_path[path] = row
    cache_manifests = [
        row for row in records
        if row.get("kind") == "JSON"
        and row.get("schema") == "ouruniv-cf4-aggregate-evidence-cache-manifest-v1"
    ]
    cache_shards = [
        row for row in records
        if row.get("kind") == "NPZ"
        and Path(str(row.get("path"))).name.startswith("shard_")
    ]
    if len(cache_manifests) != 1 or len(cache_shards) != 1:
        raise ExecutionContractError("cache manifest or shard count changed")
    expected_inside = {
        (data_directory / "sealed_oracle_control_summary.json").resolve(),
        (data_directory / "terminal_parent_frozen.npz").resolve(),
        (data_directory / "post_terminal_cf4_gates.npz").resolve(),
        (data_directory / "post_terminal_cf4_gates.json").resolve(),
    }
    expected_inside.update(
        (data_directory / f"replicate_{index}.npz").resolve()
        for index in range(4)
    )
    cache_manifest_path = Path(str(cache_manifests[0]["path"])).resolve()
    cache_shard_path = Path(str(cache_shards[0]["path"])).resolve()
    if (
        {path for path in by_path if data_directory.resolve() in path.parents}
        != expected_inside
        or {path for path in by_path if data_directory.resolve() not in path.parents}
        != {cache_manifest_path, cache_shard_path}
        or cache_manifest_path.name != "manifest.json"
        or cache_shard_path.name != "shard_000000.npz"
        or cache_manifest_path.parent != cache_shard_path.parent
    ):
        raise ExecutionContractError("manifest artifact paths are noncanonical")
    cache_value = _read_json(Path(cache_manifests[0]["path"]))
    if (
        cache_value.get("status") != "complete_immutable_evidence_cache"
        or cache_value.get("shards") != cache_shards
        or cache_value.get("row_count") != cache_shards[0]["arrays"][0]["shape"][0]
    ):
        raise ExecutionContractError("cache manifest does not bind its shard")
    with np.load(cache_shard_path, allow_pickle=False) as item:
        if set(item.files) != {"keys", "log_Z", "log_Z_bar"}:
            raise ExecutionContractError("cache shard keyset changed")
        cache_keys = np.asarray(item["keys"])
        cache_log_z = np.asarray(item["log_Z"])
        cache_log_z_bar = np.asarray(item["log_Z_bar"])
    row_count = len(cache_keys)
    sorted_keys = np.asarray(sorted(tuple(int(x) for x in row) for row in cache_keys), dtype=np.int16)
    if (
        cache_keys.dtype != np.int16 or cache_keys.ndim != 2 or cache_keys.shape[1:] != (6,)
        or row_count == 0 or len(np.unique(cache_keys, axis=0)) != row_count
        or not np.array_equal(cache_keys, sorted_keys)
        or cache_log_z.dtype != np.float64 or cache_log_z.shape != (row_count, 256)
        or cache_log_z_bar.dtype != np.float64 or cache_log_z_bar.shape != (row_count,)
        or not np.all(np.isfinite(cache_log_z)) or not np.all(np.isfinite(cache_log_z_bar))
        or not np.allclose(
            cache_log_z_bar, logmeanexp_parent(cache_log_z), rtol=0.0, atol=1e-12
        )
        or cache_value.get("row_count") != row_count
    ):
        raise ExecutionContractError("cache shard scientific contract changed")
    cache_directory = Path(cache_manifests[0]["path"]).parent
    expected_cache_entries = {
        Path(cache_manifests[0]["path"]).resolve(),
        Path(cache_shards[0]["path"]).resolve(),
    }
    actual_cache_entries = set(cache_directory.rglob("*"))
    if (
        cache_directory.is_symlink()
        or actual_cache_entries != expected_cache_entries
        or any(path.is_symlink() or not path.is_file() for path in expected_cache_entries)
    ):
        raise ExecutionContractError("cache directory contains an unbound artifact")
    recorded = {Path(row["path"]).resolve() for row in records}
    recorded.add(result_path.resolve())
    recorded.add(manifest_path.resolve())
    if data_directory.is_symlink():
        raise ExecutionContractError("completed data directory is a symlink")
    actual = set(data_directory.rglob("*"))
    recorded_inside = {
        path for path in recorded if path == data_directory.resolve()
        or data_directory.resolve() in path.parents
    }
    if actual != recorded_inside or any(
        path.is_symlink() or not path.is_file() for path in recorded_inside
    ):
        raise ExecutionContractError("completed bundle contains an unbound artifact")

    contract = capability.load_frozen_contract()
    control = _read_json(data_directory / "sealed_oracle_control_summary.json")
    if (
        control.get("schema")
        != "ouruniv-cf4-sealed-oracle-production-control-summary-v1"
        or control.get("status") != "complete_pass_exact_24_row_control"
        or control.get("selection_sha256") != CONTROL_SELECTION_SHA256
        or control.get("inside_source_rows")
        != [0, 68, 136, 204, 272, 341, 409, 477, 545, 613, 682, 750, 818, 886, 954, 1023]
        or control.get("outside_source_rows") != [0, 9, 18, 27, 36, 45, 54, 63]
        or control.get("inside_row_count") != 16
        or control.get("outside_row_count") != 8
        or control.get("global_unique_key_count") != 24
        or control.get("parent_seed_first") != 3193
        or control.get("parent_seed_last") != 3448
        or control.get("parent_count") != 256
        or float(control.get("inside_max_abs_difference", np.inf)) > 1e-10
        or float(control.get("outside_max_abs_difference", np.inf)) > 1e-10
        or control.get("control_cache_reuse_authorized") is not False
        or not isinstance(control.get("control_cache_manifest_sha256"), str)
        or len(control["control_cache_manifest_sha256"]) != 64
    ):
        raise ExecutionContractError("sealed 24-row control artifact changed")
    replicate_paths = [data_directory / f"replicate_{index}.npz" for index in range(4)]
    terminal_path = data_directory / "terminal_parent_frozen.npz"
    summary, validity, replicate_arrays = _reload_terminal_summary(
        replicate_paths, terminal_path, result.get("replicate_provenance"), contract
    )
    cache_lookup = {
        tuple(int(x) for x in key): value
        for key, value in zip(cache_keys, cache_log_z_bar)
    }
    cache_parent_lookup = {
        tuple(int(x) for x in key): value
        for key, value in zip(cache_keys, cache_log_z)
    }
    for replicate_index, arrays in enumerate(replicate_arrays):
        keys = [tuple(int(x) for x in row) for row in arrays["keys"]]
        if any(key not in cache_lookup for key in keys) or not np.allclose(
            arrays["log_Z_bar"],
            np.asarray([cache_lookup[key] for key in keys]),
            rtol=0.0, atol=1e-12,
        ):
            raise ExecutionContractError("replicate evidence is not bound to cache")
        parent_log_z = np.stack([cache_parent_lookup[key] for key in keys])
        recomputed_parent_probability = base_capability.replicate_parent_probability(
            arrays["weights"], parent_log_z
        )
        if not np.allclose(
            recomputed_parent_probability,
            summary.P_rep[replicate_index],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ExecutionContractError(
                "terminal parent posterior is not bound to raw cache evidence"
            )

    cf4_arrays_path = data_directory / "post_terminal_cf4_gates.npz"
    with np.load(cf4_arrays_path, allow_pickle=False) as item:
        if set(item.files) != {"parent_seed", "deviance", "P_pool"}:
            raise ExecutionContractError("CF4 array keyset changed")
        cf4_parent = np.asarray(item["parent_seed"])
        cf4_deviance = np.asarray(item["deviance"])
        cf4_pool = np.asarray(item["P_pool"])
    calibration = base_capability._load_pinned_calibration()
    if (
        cf4_parent.dtype != np.int32 or cf4_parent.shape != (256,)
        or cf4_deviance.dtype != np.float64 or cf4_deviance.shape != (256,)
        or cf4_pool.dtype != np.float64 or cf4_pool.shape != (256,)
        or not np.array_equal(cf4_parent, calibration.parent_seed)
        or not np.array_equal(cf4_deviance, calibration.deviance)
        or not np.allclose(cf4_pool, summary.P_pool, rtol=0.0, atol=1e-12)
    ):
        raise ExecutionContractError("CF4 arrays are not the pinned calibration join")
    cf4_recomputed = base_capability.evaluate_cf4_gates(
        cf4_deviance, cf4_pool,
        calibration.reference_q99, calibration.reference_q99p5,
    )
    pre_cf4 = capability.evaluate_pre_cf4_diagnostics(summary, contract)
    decision = capability.classify_complete_gate_set(
        validity, pre_cf4, cf4_recomputed["gates"], contract
    )
    if decision.outcome_kind == "invalid":
        raise ExecutionContractError("recomputed completion is invalid")
    expected_status = (
        "complete_pass_production_smc"
        if decision.outcome_kind == "pass"
        else "complete_scientific_fail_production_smc"
    )
    expected = {
        "status": expected_status,
        "outcome_kind": decision.outcome_kind,
        "failure_class": decision.primary_failure,
        "failed_channels": list(decision.failed_channels),
        "validity_gates": _jsonable(validity),
        "pre_CF4_gates": _jsonable(pre_cf4.gates),
        "CF4_gates": _jsonable(cf4_recomputed["gates"]),
        "all_gates": _jsonable(decision.all_gates),
        "pre_CF4_metrics": _jsonable(pre_cf4.metrics),
        "CF4_metrics": _jsonable(cf4_recomputed["metrics"]),
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ExecutionContractError("result scientific decision changed")
    manifest_expected = dict(expected)
    manifest_expected.pop("status")
    manifest_expected["science_status"] = expected_status
    if any(manifest.get(key) != value for key, value in manifest_expected.items()):
        raise ExecutionContractError("manifest scientific decision changed")
    if manifest.get("replicate_provenance") != result.get("replicate_provenance"):
        raise ExecutionContractError("manifest replicate provenance changed")
    cf4_json = _read_json(data_directory / "post_terminal_cf4_gates.json")
    if (
        cf4_json.get("status") != expected_status
        or cf4_json.get("terminal_parent_frozen") != str(terminal_path.resolve())
        or cf4_json.get("terminal_parent_frozen_sha256") != sha256_file(terminal_path)
        or cf4_json.get("post_terminal_arrays") != str(cf4_arrays_path.resolve())
        or cf4_json.get("post_terminal_arrays_sha256") != sha256_file(cf4_arrays_path)
        or cf4_json.get("pre_CF4_metrics") != expected["pre_CF4_metrics"]
        or cf4_json.get("CF4_metrics") != expected["CF4_metrics"]
        or cf4_json.get("gates") != expected["CF4_gates"]
        or cf4_json.get("failure_class") != decision.primary_failure
    ):
        raise ExecutionContractError("post-terminal CF4 decision changed")
    return {
        "status": expected_status,
        "outcome_kind": decision.outcome_kind,
        "failure_class": decision.primary_failure,
        "valid_scientific_complete": True,
    }
