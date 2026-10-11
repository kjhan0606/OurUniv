#!/usr/bin/env python3
"""Fail-closed wiring for the frozen, presently unauthorized production SMC."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from cf4_aggregate_evidence_parallel_oracle import (
    AppendOnlyEvidenceCache,
    ParallelExactAtlasEvaluator,
    RegressionControlResult,
    ShardedControllerOracle,
    run_sealed_regression_control,
)
from cf4_aggregate_evidence_smc_capability import _run_fixed_capability_core
from cf4_aggregate_evidence_smc_capability import (
    GATE_FAILURE_PRIORITY,
    classify_failure,
)
from cf4_aggregate_evidence_smc_validation import run_validation
from cf4_aggregate_evidence_oracle import logmeanexp_parent


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PROGRAM = (
    ROOT / "config/cf4_aggregate_evidence_smc_production_program.json"
)
PROGRAM_SHA256 = (
    "74cd10fdff0171daff6984ebc8db13cfd82d6dc495891ff585b81ac9eb0129c5"
)
SOURCE_COMMIT = "6630b6b04ab02e513d47f1667617384894eb349f"
CAPABILITY_COMMIT = "22587e47232782feb4c08768d8f64d853d76e62b"
DATA_DIRECTORY = Path(
    "/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v1"
)
STATE_DIRECTORY = Path(
    "/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v1_run"
)
ARCHITECTURE_FAILURES = {
    "SMC_temperature_stagnation",
    "SMC_maximum_temperature_stages",
}
SCIENTIFIC_STATUSES = {
    "complete_pass_production_smc",
    "complete_scientific_fail_production_smc",
}
CAPABILITY_DECISION_KEYS = {
    "production_SMC_execution_authorized",
    "conditional_field_bank_authorized",
    "parent_or_seed_selection_authorized",
    "PM_authorized",
    "HOP_authorized",
    "RAMSES_authorized",
    "downstream_execution_authorized",
    "automatic_follow_on",
}
RESULT_DECISION_KEYS = CAPABILITY_DECISION_KEYS | {
    "candidate_generation_authorized",
    "automatic_retry_scale_or_retune",
}
STABLE_SCIENTIFIC_FAILURES = {
    "replicate_log_I_bar_range",
    "replicate_log_I_bar_sample_SE",
    "replicate_parent_probability_L1",
    "genealogical_ESS",
    "pooled_parent_ESS",
    "maximum_pooled_parent_probability",
    "weighted_CF4_Q99_exceedance_mass",
    "weighted_CF4_Q90",
    "weighted_CF4_one_sided_KS_permutation",
}
CACHE_LOGMEAN_TOLERANCE = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
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


def _resolved_input_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _require_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeError(f"{label} is not a full lowercase SHA256")
    return text


def validate_program_document(
    program: dict[str, Any],
    *,
    verify_file_hashes: bool,
) -> None:
    """Validate frozen structure; file hashing is a preflight-only option."""
    if program.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-production-program-v1" \
            or program.get("status") != "frozen_unauthorized_production_program" \
            or program.get("source_commit") != SOURCE_COMMIT \
            or program.get("capability_commit") != CAPABILITY_COMMIT:
        raise RuntimeError("production program identity or commit lineage changed")
    storage = program.get("storage", {})
    if storage.get("data_directory") != str(DATA_DIRECTORY) \
            or storage.get("state_directory") != str(STATE_DIRECTORY) \
            or storage.get("result") != str(DATA_DIRECTORY / "result.json") \
            or storage.get("manifest") != str(DATA_DIRECTORY / "manifest.json") \
            or storage.get("exclusive_reservation") is not True \
            or storage.get("restart_or_checkpoint_import") is not False:
        raise RuntimeError("production storage contract changed")
    fixed = program.get("fixed_execution", {})
    expected_fixed = {
        "synthetic_validation_first": True,
        "sealed_control_rows": 24,
        "control_cache_reuse": False,
        "fresh_production_covariance_cached_keys": 0,
        "fresh_production_covariance_evaluation_batches": 0,
        "worker_processes": 8,
        "parents_per_worker_block": 32,
        "threads_per_worker": 1,
        "replicate_count": 4,
        "replicates_sequential": True,
        "particles_per_replicate": 2048,
        "automatic_retry": False,
        "automatic_scale_up": False,
        "automatic_retune": False,
        "automatic_follow_on": False,
        "runtime_override_allowed": False,
    }
    if fixed != expected_fixed:
        raise RuntimeError("fixed production execution contract changed")
    environment = program.get("execution_environment", {})
    if environment != {
        "host": "lageunha",
        "device": "CPU",
        "CUDA_VISIBLE_DEVICES": "",
        "minimum_free_disk_GiB": 40,
        "minimum_MemAvailable_GiB": 64,
        "multiprocessing_start_method": "fork",
    }:
        raise RuntimeError("production environment contract changed")
    authorization = program.get("authorization", {})
    allowed = "production_program_and_runner_design_and_implementation_authorized"
    if authorization.get(allowed) is not True or any(
        value is not False for key, value in authorization.items() if key != allowed
    ) or set(authorization) != {
        allowed,
        "production_execution_authorized",
        "oracle_cache_population_authorized",
        "conditional_field_bank_authorized",
        "candidate_generation_authorized",
        "parent_or_seed_selection_authorized",
        "PM_authorized",
        "HOP_authorized",
        "RAMSES_authorized",
        "downstream_execution_authorized",
        "automatic_follow_on",
    }:
        raise RuntimeError("production authorization map is not fail closed")
    audited = program.get("audited_capability_files", [])
    pinned = program.get("pinned_local_files", [])
    if len(audited) != 5 or len(pinned) != 16:
        raise RuntimeError("production local pin count changed")
    audited_map = {row.get("path"): row.get("sha256") for row in audited}
    pinned_map = {row.get("path"): row.get("sha256") for row in pinned}
    if len(audited_map) != 5 or len(pinned_map) != 16 \
            or any(pinned_map.get(path) != digest for path, digest in audited_map.items()):
        raise RuntimeError("audited capability pins are missing or duplicated")
    for label, rows in (("audited capability", audited), ("local", pinned)):
        for row in rows:
            digest = _require_sha256(row.get("sha256"), f"{label} pin")
            path = _resolved_input_path(str(row.get("path")))
            if verify_file_hashes and (
                not path.is_file() or sha256_file(path) != digest
            ):
                raise RuntimeError(f"{label} hash mismatch: {row.get('path')}")
    expected_external = {
        "response_atlas_manifest": (
            "/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_parent_response_atlas_v1/manifest.json",
            "47049d0047aa626912652c82ac34757f01ebe4adc0654d07f674d6b943db4211",
        ),
        "oracle_regression_arrays": (
            "/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_oracle_regression_v1/arrays.npz",
            "40c52c48ae7899219476fe6c9f0308b7f68e70ac2493c9fdea26cd37923572e2",
        ),
        "density_filter": (
            "/gpfs/kjhan/CF4/recon/linear_cr/peak_evidence_phase_control_v2/density_filter_rfft.npy",
            "1e1d2ce4b022c908b8a3a64257e82fb8621c40cf8f77b1fd206f1347cdd2f59a",
        ),
        "physical_model": (
            "config/p2_lg_z0_forward_importance_v8.json",
            "6a89f5027f253282e18f21201146dde384837f0d689d725a25022def8ea7e6f2",
        ),
        "reference_calibration": (
            "/gpfs/kjhan/CF4/recon/linear_cr/v8_cf4_mode_release_reference/calibration.json",
            "c9edb6d0a108746fe18fa75295ab73f53286a25f9aa2725d132a0560375cb988",
        ),
    }
    external = program.get("external_inputs", {})
    if set(external) != set(expected_external):
        raise RuntimeError("production external input set changed")
    for name, (expected_path, expected_sha) in expected_external.items():
        row = external[name]
        if row.get("path") != expected_path \
                or _require_sha256(row.get("sha256"), name) != expected_sha:
            raise RuntimeError(f"production external pin changed: {name}")
        path = _resolved_input_path(expected_path)
        if verify_file_hashes and (
            not path.is_file() or sha256_file(path) != expected_sha
        ):
            raise RuntimeError(f"production external hash mismatch: {name}")
    parent = program.get("parent_lineage", {})
    if parent.get("seed_range_inclusive") != [3193, 3448] \
            or parent.get("parent_count") != 256 \
            or parent.get("full_sha256_required") is not True:
        raise RuntimeError("production parent lineage contract changed")
    if verify_file_hashes:
        atlas = json.loads(Path(expected_external["response_atlas_manifest"][0]).read_text())
        calibration = json.loads(Path(expected_external["reference_calibration"][0]).read_text())
        atlas_rows = atlas.get("entries", [])
        calibration_rows = calibration.get("rows", [])
        if len(atlas_rows) != 256 or len(calibration_rows) != 256:
            raise RuntimeError("production parent lineage count changed")
        for seed, atlas_row, calibration_row in zip(
            range(3193, 3449), atlas_rows, calibration_rows
        ):
            if atlas_row.get("seed") != seed \
                    or calibration_row.get("seed") != seed \
                    or atlas_row.get("parent_field") != calibration_row.get("field") \
                    or _require_sha256(
                        atlas_row.get("parent_field_sha256"), "atlas parent"
                    ) != calibration_row.get("field_sha256") \
                    or len(_require_sha256(
                        atlas_row.get("atlas_sha256"), "atlas shard"
                    )) != 64:
                raise RuntimeError("atlas and calibration parent lineage disagree")


def load_canonical_program(*, verify_file_hashes: bool) -> dict[str, Any]:
    if sha256_file(CANONICAL_PROGRAM) != PROGRAM_SHA256:
        raise RuntimeError("canonical production program hash mismatch")
    program = json.loads(CANONICAL_PROGRAM.read_text())
    validate_program_document(program, verify_file_hashes=verify_file_hashes)
    return program


def _json_artifact(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "kind": "json",
        "schema": value.get("schema"),
        "status": value.get("status"),
        "byte_count": Path(path).stat().st_size,
    }


def _npz_artifact(path: Path) -> dict[str, Any]:
    arrays = []
    with np.load(path, allow_pickle=False) as item:
        for name in item.files:
            value = item[name]
            finite = bool(
                not np.issubdtype(value.dtype, np.number)
                or np.all(np.isfinite(value))
            )
            if not finite:
                raise RuntimeError(f"nonfinite NPZ artifact array: {path}:{name}")
            arrays.append({
                "name": name,
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "finite": finite,
            })
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "kind": "npz",
        "byte_count": Path(path).stat().st_size,
        "arrays": arrays,
    }


def _recorded_artifact(path: Path) -> dict[str, Any]:
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise RuntimeError(f"required production artifact is absent or empty: {path}")
    return _npz_artifact(path) if Path(path).suffix == ".npz" else _json_artifact(path)


def _array_map(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in record.get("arrays", [])}


def _validate_cache_shard_record(record: dict[str, Any]) -> None:
    arrays = _array_map(record)
    if set(arrays) != {"keys", "log_Z", "log_Z_bar"}:
        raise RuntimeError("production cache shard key set changed")
    row_count = arrays["keys"]["shape"][0] if len(arrays["keys"]["shape"]) == 2 else -1
    if row_count <= 0 \
            or arrays["keys"]["dtype"] != "int16" \
            or arrays["keys"]["shape"] != [row_count, 6] \
            or arrays["log_Z"]["dtype"] != "float64" \
            or arrays["log_Z"]["shape"] != [row_count, 256] \
            or arrays["log_Z_bar"]["dtype"] != "float64" \
            or arrays["log_Z_bar"]["shape"] != [row_count]:
        raise RuntimeError("production cache shard dtype or shape changed")


def _validate_cache_bundle(
    cache_manifest_path: Path,
    cache_records: list[dict[str, Any]],
) -> dict[str, Any]:
    cache_manifest_path = Path(cache_manifest_path)
    value = json.loads(cache_manifest_path.read_text())
    if set(value) != {
        "schema", "status", "restart_or_checkpoint_imported", "shard_count",
        "total_row_count", "shards",
    } or value.get("schema") != (
        "ouruniv-cf4-aggregate-evidence-cache-manifest-v1"
    ) or value.get("status") != "complete_immutable_append_only_cache" \
            or value.get("restart_or_checkpoint_imported") is not False \
            or value.get("shard_count") != len(cache_records) \
            or not isinstance(value.get("total_row_count"), int) \
            or not isinstance(value.get("shards"), list) \
            or len(value["shards"]) != len(cache_records):
        raise RuntimeError("production cache manifest exact contract changed")
    seen: set[tuple[int, ...]] = set()
    total = 0
    for index, (row, record) in enumerate(zip(value["shards"], cache_records)):
        expected_path = cache_manifest_path.parent / f"shard_{index:06d}.npz"
        if set(row) != {"path", "sha256", "row_count"} \
                or row.get("path") != str(expected_path.resolve()) \
                or Path(record.get("path", "")).resolve() != expected_path.resolve() \
                or row.get("sha256") != record.get("sha256") \
                or not isinstance(row.get("row_count"), int) \
                or row["row_count"] <= 0 \
                or not expected_path.is_file() \
                or sha256_file(expected_path) != row["sha256"]:
            raise RuntimeError("production cache shard order, SHA, or row count changed")
        _validate_cache_shard_record(record)
        with np.load(expected_path, allow_pickle=False) as item:
            keys = item["keys"]
            log_z = item["log_Z"]
            stored_bar = item["log_Z_bar"]
            if len(keys) != row["row_count"]:
                raise RuntimeError("production cache shard stored row count changed")
            tuples = [tuple(int(element) for element in key) for key in keys]
            if tuples != sorted(set(tuples)):
                raise RuntimeError("production cache shard keys are not sorted unique")
            if any(key in seen for key in tuples):
                raise RuntimeError("production cache has a cross-shard duplicate key")
            recomputed_bar = logmeanexp_parent(log_z)
            if np.max(np.abs(stored_bar - recomputed_bar)) > CACHE_LOGMEAN_TOLERANCE:
                raise RuntimeError("production cache log_Z_bar is inconsistent")
            seen.update(tuples)
            total += len(keys)
    if total != value["total_row_count"]:
        raise RuntimeError("production cache total row count changed")
    actual_paths = {
        path.resolve() for path in cache_manifest_path.parent.glob("shard_*.npz")
    }
    if actual_paths != {
        Path(row["path"]).resolve() for row in value["shards"]
    }:
        raise RuntimeError("production cache shard set changed")
    return value


def _validate_smc_artifact_records(
    records: list[dict[str, Any]], outcome_kind: str
) -> None:
    names = [Path(row["path"]).name for row in records]
    if outcome_kind == "architecture_stop":
        if names != ["capability_result.json"] \
                or records[0]["kind"] != "json" \
                or records[0]["schema"] != (
                    "ouruniv-cf4-aggregate-evidence-smc-capability-result-v1"
                ) \
                or records[0]["status"] != "complete_scientific_fail":
            raise RuntimeError("architecture-stop artifact set changed")
        return
    expected = [
        "replicate_0.npz", "replicate_1.npz", "replicate_2.npz",
        "replicate_3.npz", "terminal_parent_frozen.npz",
        "post_terminal_cf4_gates.npz", "post_terminal_cf4_gates.json",
    ]
    if names != expected:
        raise RuntimeError("terminal SMC artifact set changed")
    replicate_keys = {
        "master_seed", "midpoint_mpc_h", "axis", "keys", "weights",
        "log_Z_bar", "ancestor_labels", "beta_history",
        "conditional_ESS_history", "particle_ESS_history",
        "log_normalizer_increment", "log_I_bar", "genealogical_ESS",
        "resampling_ancestors", "move_proposal_count",
        "move_acceptance_count", "q_scale_proposal_count",
        "q_scale_acceptance_count", "axis_scale_proposal_count",
        "axis_scale_acceptance_count",
    }
    expected_masters = (2026082301, 2026082302, 2026082303, 2026082304)
    for replicate_index, record in enumerate(records[:4]):
        arrays = _array_map(record)
        if set(arrays) != replicate_keys \
                or arrays["master_seed"]["dtype"] != "int64" \
                or arrays["master_seed"]["shape"] != [] \
                or arrays["midpoint_mpc_h"]["dtype"] != "float64" \
                or arrays["midpoint_mpc_h"]["shape"] != [2048, 3] \
                or arrays["axis"]["dtype"] != "float64" \
                or arrays["axis"]["shape"] != [2048, 3] \
                or arrays["keys"]["dtype"] != "int16" \
                or arrays["keys"]["shape"] != [2048, 6] \
                or arrays["weights"]["dtype"] != "float64" \
                or arrays["weights"]["shape"] != [2048] \
                or arrays["log_Z_bar"]["dtype"] != "float64" \
                or arrays["log_Z_bar"]["shape"] != [2048] \
                or arrays["ancestor_labels"]["dtype"] != "int64" \
                or arrays["ancestor_labels"]["shape"] != [2048]:
            raise RuntimeError("replicate artifact fixed dtype or shape changed")
        stages = arrays["conditional_ESS_history"]["shape"]
        if len(stages) != 1 or not 1 <= stages[0] <= 256 \
                or arrays["beta_history"]["dtype"] != "float64" \
                or arrays["conditional_ESS_history"]["dtype"] != "float64" \
                or arrays["particle_ESS_history"]["dtype"] != "float64" \
                or arrays["log_normalizer_increment"]["dtype"] != "float64" \
                or arrays["log_I_bar"]["dtype"] != "float64" \
                or arrays["log_I_bar"]["shape"] != [] \
                or arrays["genealogical_ESS"]["dtype"] != "float64" \
                or arrays["genealogical_ESS"]["shape"] != [] \
                or arrays["resampling_ancestors"]["dtype"] != "int64" \
                or arrays["beta_history"]["shape"] != [stages[0] + 1] \
                or arrays["particle_ESS_history"]["shape"] != [stages[0] + 1] \
                or arrays["log_normalizer_increment"]["shape"] != [stages[0]] \
                or arrays["resampling_ancestors"]["shape"][1:] != [2048] \
                or arrays["move_proposal_count"]["shape"] != [stages[0], 4, 4] \
                or arrays["move_acceptance_count"]["shape"] != [stages[0], 4, 4] \
                or arrays["q_scale_proposal_count"]["shape"] != [stages[0], 4, 3] \
                or arrays["q_scale_acceptance_count"]["shape"] != [stages[0], 4, 3] \
                or arrays["axis_scale_proposal_count"]["shape"] != [stages[0], 4, 3] \
                or arrays["axis_scale_acceptance_count"]["shape"] != [stages[0], 4, 3]:
            raise RuntimeError("replicate artifact history shape changed")
        for name in (
            "move_proposal_count", "move_acceptance_count",
            "q_scale_proposal_count", "q_scale_acceptance_count",
            "axis_scale_proposal_count", "axis_scale_acceptance_count",
        ):
            if arrays[name]["dtype"] != "int64":
                raise RuntimeError("replicate move matrix dtype changed")
        with np.load(record["path"], allow_pickle=False) as item:
            master_seed = int(item["master_seed"])
            weights = item["weights"]
            beta = item["beta_history"]
            conditional_ess = item["conditional_ESS_history"]
            particle_ess = item["particle_ESS_history"]
            increments = item["log_normalizer_increment"]
            resampling = item["resampling_ancestors"]
            ancestor_labels = item["ancestor_labels"]
            if master_seed != expected_masters[replicate_index]:
                raise RuntimeError("replicate master seed order changed")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) \
                    or not np.isclose(
                        weights.sum(), 1.0, rtol=0.0, atol=1e-12
                    ):
                raise RuntimeError("replicate weights are not normalized")
            if beta[0] != 0.0 or beta[-1] != 1.0 \
                    or np.any(np.diff(beta) <= 0.0) \
                    or np.any(conditional_ess <= 0.0) \
                    or np.any(conditional_ess > 2048.0 * (1.0 + 1e-12)) \
                    or np.any(particle_ess <= 0.0) \
                    or np.any(particle_ess > 2048.0 * (1.0 + 1e-12)) \
                    or len(conditional_ess) != len(beta) - 1 \
                    or len(particle_ess) != len(beta) \
                    or len(increments) != len(beta) - 1:
                raise RuntimeError("replicate beta or history relation changed")
            expected_resampling_rows = int(np.count_nonzero(
                particle_ess[1:] < 1024.0
            ))
            if resampling.dtype != np.int64 \
                    or resampling.shape != (expected_resampling_rows, 2048) \
                    or np.any(resampling < 0) or np.any(resampling >= 2048) \
                    or np.any(ancestor_labels < 0) \
                    or np.any(ancestor_labels >= 2048):
                raise RuntimeError("replicate resampling history changed")
            if not np.isclose(
                increments.sum(), float(item["log_I_bar"]),
                rtol=1e-12, atol=1e-12,
            ):
                raise RuntimeError("replicate normalizer history changed")
    terminal = _array_map(records[4])
    if {
        name: (row["dtype"], row["shape"]) for name, row in terminal.items()
    } != {
        "master_seed": ("int64", [4]),
        "parent_seed": ("int32", [256]),
        "log_I_bar": ("float64", [4]),
        "P_rep": ("float64", [4, 256]),
        "P_pool": ("float64", [256]),
    }:
        raise RuntimeError("terminal parent artifact dtype or shape changed")
    with np.load(records[4]["path"], allow_pickle=False) as item:
        if not np.array_equal(item["master_seed"], expected_masters) \
                or not np.array_equal(
                    item["parent_seed"], np.arange(3193, 3449, dtype=np.int32)
                ):
            raise RuntimeError("terminal master or parent seed lineage changed")
        p_rep = item["P_rep"]
        p_pool = item["P_pool"]
        if not np.all(np.isfinite(p_rep)) or not np.all(np.isfinite(p_pool)) \
                or np.any(p_rep < 0.0) or np.any(p_pool < 0.0) \
                or not np.allclose(
                    p_rep.sum(axis=1), 1.0, rtol=0.0, atol=1e-12
                ) \
                or not np.isclose(
                    p_pool.sum(), 1.0, rtol=0.0, atol=1e-12
                ):
            raise RuntimeError("terminal parent probabilities are not normalized")
    cf4 = _array_map(records[5])
    if {
        name: (row["dtype"], row["shape"]) for name, row in cf4.items()
    } != {
        "parent_seed": ("int32", [256]),
        "deviance": ("float64", [256]),
        "P_pool": ("float64", [256]),
    } or records[6]["kind"] != "json" \
            or records[6]["schema"] != (
                "ouruniv-cf4-aggregate-evidence-post-terminal-cf4-gates-v1"
            ) \
            or records[6]["status"] not in {
                "complete_pass", "complete_scientific_fail"
            }:
        raise RuntimeError("post-terminal CF4 artifact dtype or shape changed")
    with np.load(records[5]["path"], allow_pickle=False) as item:
        if not np.array_equal(
            item["parent_seed"], np.arange(3193, 3449, dtype=np.int32)
        ) or not np.all(np.isfinite(item["P_pool"])) \
                or np.any(item["P_pool"] < 0.0) \
                or not np.isclose(
                    item["P_pool"].sum(), 1.0, rtol=0.0, atol=1e-12
                ):
            raise RuntimeError("post-terminal CF4 parent lineage changed")
        with np.load(records[4]["path"], allow_pickle=False) as terminal_item:
            if not np.array_equal(item["P_pool"], terminal_item["P_pool"]):
                raise RuntimeError("post-terminal CF4 pooled probability changed")


def _validate_capability_summary(summary: dict[str, Any]) -> tuple[str, str]:
    schema = summary.get("schema")
    status = summary.get("status")
    failure = summary.get("failure_class")
    if status == "complete_pass":
        gates = summary.get("gates")
        if schema != "ouruniv-cf4-aggregate-evidence-post-terminal-cf4-gates-v1" \
                or failure is not None \
                or not isinstance(gates, dict) \
                or set(gates) != {name for name, _ in GATE_FAILURE_PRIORITY} \
                or any(value is not True for value in gates.values()):
            raise RuntimeError("passing capability result contract changed")
        return "complete_pass_production_smc", "terminal"
    if status != "complete_scientific_fail" or not isinstance(failure, str):
        raise RuntimeError("capability core did not return a valid scientific status")
    if failure in ARCHITECTURE_FAILURES:
        if schema != "ouruniv-cf4-aggregate-evidence-smc-capability-result-v1" \
                or summary.get("valid_scientific_architecture_stop") is not True \
                or summary.get("gates") is not None:
            raise RuntimeError("architecture stop lacks its valid provenance flag")
        return "complete_scientific_fail_production_smc", "architecture_stop"
    gates = summary.get("gates")
    if schema != "ouruniv-cf4-aggregate-evidence-post-terminal-cf4-gates-v1" \
            or failure not in STABLE_SCIENTIFIC_FAILURES \
            or not isinstance(gates, dict) \
            or set(gates) != {name for name, _ in GATE_FAILURE_PRIORITY} \
            or any(not isinstance(value, bool) for value in gates.values()) \
            or classify_failure(gates) != failure:
        raise RuntimeError("scientific gate failure contract changed")
    return "complete_scientific_fail_production_smc", "terminal"


def _require_downstream_closed(
    summary: dict[str, Any], expected_keys: set[str]
) -> None:
    decision = summary.get("decision", {})
    if not isinstance(decision, dict) or set(decision) != expected_keys \
            or any(value is not False for value in decision.values()):
        raise RuntimeError("capability result opened a forbidden downstream decision")


class _ExactlyOnceCloseProxy:
    """Track evaluator ownership without changing its callable API."""

    def __init__(self, evaluator):
        self._evaluator = evaluator
        self.close_count = 0

    def __getattr__(self, name):
        return getattr(self._evaluator, name)

    def __call__(self, keys):
        return self._evaluator(keys)

    def close(self) -> None:
        if self.close_count != 0:
            raise RuntimeError("evaluator pool close was requested more than once")
        self.close_count = 1
        self._evaluator.close()


def _build_manifest(
    program: dict[str, Any],
    data_directory: Path,
    lifecycle_status: str,
    outcome_kind: str,
    control: RegressionControlResult,
    cache_manifest_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    cache_value = json.loads(cache_manifest_path.read_text())
    cache_records = []
    for row in cache_value.get("shards", []):
        path = Path(row.get("path", ""))
        record = _recorded_artifact(path)
        cache_records.append(record)
    _validate_cache_bundle(cache_manifest_path, cache_records)
    smc_directory = data_directory / "smc"
    smc_records = []
    if outcome_kind == "terminal":
        required = [
            *(smc_directory / f"replicate_{index}.npz" for index in range(4)),
            smc_directory / "terminal_parent_frozen.npz",
            smc_directory / "post_terminal_cf4_gates.npz",
            smc_directory / "post_terminal_cf4_gates.json",
        ]
    else:
        required = [smc_directory / "capability_result.json"]
        forbidden = [
            *(smc_directory / f"replicate_{index}.npz" for index in range(4)),
            smc_directory / "terminal_parent_frozen.npz",
            smc_directory / "post_terminal_cf4_gates.npz",
            smc_directory / "post_terminal_cf4_gates.json",
        ]
        if any(path.exists() for path in forbidden):
            raise RuntimeError("architecture stop published forbidden terminal artifacts")
    smc_records.extend(_recorded_artifact(path) for path in required)
    _validate_smc_artifact_records(smc_records, outcome_kind)
    actual_smc_files = {path.resolve() for path in smc_directory.iterdir() if path.is_file()}
    if actual_smc_files != {Path(row["path"]).resolve() for row in smc_records}:
        raise RuntimeError("SMC output directory contains an unbound artifact")
    summary_record = _recorded_artifact(Path(control.summary_path))
    if summary_record["sha256"] != control.summary_sha256:
        raise RuntimeError("sealed control summary changed before manifest")
    local_lineage = {
        row["path"]: row["sha256"] for row in program["pinned_local_files"]
    }
    external_lineage = {
        name: row["sha256"] for name, row in program["external_inputs"].items()
    }
    return {
        "schema": "ouruniv-cf4-aggregate-evidence-smc-production-manifest-v1",
        "status": "complete_immutable_production_manifest",
        "scientific_status": lifecycle_status,
        "outcome_kind": outcome_kind,
        "program": str(CANONICAL_PROGRAM.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "source_commit": SOURCE_COMMIT,
        "capability_commit": CAPABILITY_COMMIT,
        "local_lineage_sha256": local_lineage,
        "external_lineage_sha256": external_lineage,
        "synthetic_validation": _recorded_artifact(
            data_directory / "synthetic_validation.json"
        ),
        "control": summary_record,
        "control_selection_sha256": control.selection_sha256,
        "control_cache_discarded": control.control_cache_discarded,
        "control_evaluator_discarded": control.control_evaluator_discarded,
        "production_covariance_cached_key_count_at_open": control.production_covariance_cached_key_count,
        "production_covariance_evaluation_batches_at_open": control.production_covariance_evaluation_batches,
        "production_cache_manifest": _recorded_artifact(cache_manifest_path),
        "production_cache_shards": cache_records,
        "SMC_artifacts": smc_records,
        "result": _recorded_artifact(result_path),
        "all_NPZ_arrays_finite": True,
        "automatic_retry_scale_retune_or_follow_on": False,
    }


def _execute_into_reserved_directory(
    program: dict[str, Any],
    data_directory: Path,
    *,
    validation_runner: Callable[[], dict[str, Any]],
    evaluator_factory: Callable[[], Any],
    control_runner: Callable[..., tuple[RegressionControlResult, Any, AppendOnlyEvidenceCache]],
    capability_core: Callable[[Any, Path], dict[str, Any]],
) -> dict[str, Any]:
    data_directory = Path(data_directory)
    if not data_directory.is_dir() or any(data_directory.iterdir()):
        raise RuntimeError("production data reservation is absent or not empty")
    validation = validation_runner()
    if validation.get("status") != "complete_pass" \
            or validation.get("all_pass") is not True:
        raise RuntimeError("synthetic SMC validation did not pass")
    validation_path = data_directory / "synthetic_validation.json"
    _atomic_json(validation_path, validation)
    external = program["external_inputs"]
    production_evaluator = None
    production_cache = None
    owned_evaluators: list[_ExactlyOnceCloseProxy] = []

    def owned_factory():
        evaluator = _ExactlyOnceCloseProxy(evaluator_factory())
        owned_evaluators.append(evaluator)
        return evaluator

    try:
        control, production_evaluator, production_cache = control_runner(
            owned_factory,
            Path(external["oracle_regression_arrays"]["path"]),
            data_directory / "oracle",
        )
        if not control.control_cache_discarded \
                or not control.control_evaluator_discarded \
                or not control.covariance_cache_identity_distinct \
                or control.production_covariance_cached_key_count != 0 \
                or control.production_covariance_evaluation_batches != 0 \
                or not control.production_cache_empty \
                or production_cache.shard_count != 0:
            raise RuntimeError("sealed control did not return a fresh production cache")
        covariance = getattr(production_evaluator, "covariance_cache", None)
        if covariance is None \
                or covariance.evaluated_covariance_keys != 0 \
                or covariance.evaluation_batches != 0:
            raise RuntimeError("fresh production evaluator hard gate failed")
        oracle = ShardedControllerOracle(production_evaluator, production_cache)
        capability_summary = capability_core(oracle, data_directory / "smc")
        lifecycle_status, outcome_kind = _validate_capability_summary(
            capability_summary
        )
        _require_downstream_closed(
            capability_summary, CAPABILITY_DECISION_KEYS
        )
        cache_manifest_path, cache_manifest_sha = production_cache.seal()
    finally:
        if production_evaluator is not None:
            close = getattr(production_evaluator, "close", None)
            if not callable(close):
                raise RuntimeError("production evaluator lacks explicit pool close")
            close()
            production_evaluator = None
        for evaluator in owned_evaluators:
            if evaluator.close_count == 0:
                evaluator.close()
        if any(evaluator.close_count != 1 for evaluator in owned_evaluators):
            raise RuntimeError("evaluator pools were not closed exactly once")
    result_path = data_directory / "result.json"
    result = {
        "schema": "ouruniv-cf4-aggregate-evidence-smc-production-result-v1",
        "status": lifecycle_status,
        "outcome_kind": outcome_kind,
        "failure_class": capability_summary.get("failure_class"),
        "program_sha256": PROGRAM_SHA256,
        "program": str(CANONICAL_PROGRAM.resolve()),
        "source_commit": SOURCE_COMMIT,
        "capability_commit": CAPABILITY_COMMIT,
        "synthetic_validation": str(validation_path.resolve()),
        "synthetic_validation_sha256": sha256_file(validation_path),
        "sealed_control_summary": control.summary_path,
        "sealed_control_summary_sha256": control.summary_sha256,
        "production_cache_manifest": str(cache_manifest_path.resolve()),
        "production_cache_manifest_sha256": cache_manifest_sha,
        "production_evaluator_closed": True,
        "capability_status": capability_summary.get("status"),
        "capability_gates": capability_summary.get("gates"),
        "decision": {
            "production_SMC_execution_authorized": False,
            "conditional_field_bank_authorized": False,
            "candidate_generation_authorized": False,
            "parent_or_seed_selection_authorized": False,
            "PM_authorized": False,
            "HOP_authorized": False,
            "RAMSES_authorized": False,
            "downstream_execution_authorized": False,
            "automatic_follow_on": False,
            "automatic_retry_scale_or_retune": False,
        },
    }
    _atomic_json(result_path, result)
    manifest = _build_manifest(
        program,
        data_directory,
        lifecycle_status,
        outcome_kind,
        control,
        cache_manifest_path,
        result_path,
    )
    manifest_path = data_directory / "manifest.json"
    _atomic_json(manifest_path, manifest)
    validate_published_bundle(data_directory)
    return result


def _actual_evaluator_factory(program: dict[str, Any]) -> Callable[[], Any]:
    external = program["external_inputs"]

    def factory():
        return ParallelExactAtlasEvaluator(
            Path(external["response_atlas_manifest"]["path"]),
            external["response_atlas_manifest"]["sha256"],
            Path(external["density_filter"]["path"]),
            external["density_filter"]["sha256"],
            _resolved_input_path(external["physical_model"]["path"]),
            external["physical_model"]["sha256"],
        )

    return factory


def _execute_authorized_program(program: dict[str, Any]) -> dict[str, Any]:
    return _execute_into_reserved_directory(
        program,
        DATA_DIRECTORY,
        validation_runner=run_validation,
        evaluator_factory=_actual_evaluator_factory(program),
        control_runner=run_sealed_regression_control,
        capability_core=_run_fixed_capability_core,
    )


def run_production(program_path: Path) -> dict[str, Any]:
    """Sole public entry; no runtime injection or override is accepted."""
    path = Path(program_path).resolve()
    if path != CANONICAL_PROGRAM.resolve():
        raise PermissionError("production SMC accepts only the canonical program path")
    if sha256_file(path) != PROGRAM_SHA256:
        raise RuntimeError("canonical production program hash mismatch")
    program = json.loads(path.read_text())
    validate_program_document(program, verify_file_hashes=False)
    if program["authorization"].get("production_execution_authorized") is not True:
        raise PermissionError("production SMC execution is not authorized")
    validate_program_document(program, verify_file_hashes=True)
    return _execute_authorized_program(program)


def validate_published_bundle(data_directory: Path) -> dict[str, Any]:
    root = Path(data_directory).resolve()
    result_path = root / "result.json"
    manifest_path = root / "manifest.json"
    if not result_path.is_file() or not manifest_path.is_file():
        raise RuntimeError("production result or manifest is absent")
    result = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    status = result.get("status")
    if sha256_file(CANONICAL_PROGRAM) != PROGRAM_SHA256 \
            or result.get("schema") != (
                "ouruniv-cf4-aggregate-evidence-smc-production-result-v1"
            ) \
            or manifest.get("schema") != (
                "ouruniv-cf4-aggregate-evidence-smc-production-manifest-v1"
            ) \
            or status not in SCIENTIFIC_STATUSES \
            or manifest.get("status") != "complete_immutable_production_manifest" \
            or manifest.get("scientific_status") != status \
            or manifest.get("program") != str(CANONICAL_PROGRAM.resolve()) \
            or manifest.get("program_sha256") != PROGRAM_SHA256 \
            or manifest.get("source_commit") != SOURCE_COMMIT \
            or manifest.get("capability_commit") != CAPABILITY_COMMIT \
            or result.get("program") != str(CANONICAL_PROGRAM.resolve()) \
            or result.get("program_sha256") != PROGRAM_SHA256 \
            or result.get("source_commit") != SOURCE_COMMIT \
            or result.get("capability_commit") != CAPABILITY_COMMIT \
            or result.get("production_evaluator_closed") is not True \
            or manifest.get("all_NPZ_arrays_finite") is not True \
            or manifest.get("automatic_retry_scale_retune_or_follow_on") is not False:
        raise RuntimeError("production result/manifest lifecycle contract failed")
    outcome_kind = result.get("outcome_kind")
    failure = result.get("failure_class")
    if manifest.get("outcome_kind") != outcome_kind \
            or outcome_kind not in {"terminal", "architecture_stop"} \
            or (status == "complete_pass_production_smc" and failure is not None) \
            or (status == "complete_scientific_fail_production_smc" and not isinstance(failure, str)) \
            or (outcome_kind == "architecture_stop" and failure not in ARCHITECTURE_FAILURES) \
            or (outcome_kind == "terminal" and failure in ARCHITECTURE_FAILURES):
        raise RuntimeError("production scientific completion classification changed")
    if manifest.get("control_cache_discarded") is not True \
            or manifest.get("control_evaluator_discarded") is not True \
            or manifest.get("production_covariance_cached_key_count_at_open") != 0 \
            or manifest.get("production_covariance_evaluation_batches_at_open") != 0:
        raise RuntimeError("production control-to-cache transition changed")
    program = load_canonical_program(verify_file_hashes=True)
    if manifest.get("local_lineage_sha256") != {
        row["path"]: row["sha256"] for row in program["pinned_local_files"]
    } or manifest.get("external_lineage_sha256") != {
        name: row["sha256"] for name, row in program["external_inputs"].items()
    }:
        raise RuntimeError("production manifest frozen lineage changed")
    synthetic_record = manifest.get("synthetic_validation", {})
    control_record = manifest.get("control", {})
    cache_record = manifest.get("production_cache_manifest", {})
    if synthetic_record.get("path") != str(
        (root / "synthetic_validation.json").resolve()
    ) \
            or control_record.get("path") != str(
                (root / "oracle/sealed_oracle_control_summary.json").resolve()
            ) \
            or cache_record.get("path") != str(
                (root / "oracle/production_cache/manifest.json").resolve()
            ) \
            or manifest.get("result", {}).get("path") != str(result_path.resolve()) \
            or result.get("synthetic_validation") != synthetic_record.get("path") \
            or result.get("synthetic_validation_sha256") != synthetic_record.get("sha256") \
            or result.get("sealed_control_summary") != control_record.get("path") \
            or result.get("sealed_control_summary_sha256") != control_record.get("sha256") \
            or result.get("production_cache_manifest") != cache_record.get("path") \
            or result.get("production_cache_manifest_sha256") != cache_record.get("sha256") \
            or synthetic_record.get("schema") != (
                "ouruniv-cf4-aggregate-evidence-smc-synthetic-validation-v1"
            ) \
            or synthetic_record.get("status") != "complete_pass" \
            or control_record.get("schema") != (
                "ouruniv-cf4-sealed-oracle-production-control-summary-v1"
            ) \
            or control_record.get("status") != "complete_pass_exact_24_row_control":
        raise RuntimeError("production result-to-manifest artifact binding changed")
    records = [
        manifest.get("synthetic_validation"),
        manifest.get("control"),
        manifest.get("production_cache_manifest"),
        *manifest.get("production_cache_shards", []),
        *manifest.get("SMC_artifacts", []),
        manifest.get("result"),
    ]
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("production manifest lost an artifact record")
        path = Path(record.get("path", "")).resolve()
        if root not in path.parents or not path.is_file() \
                or sha256_file(path) != record.get("sha256"):
            raise RuntimeError("production manifest artifact hash changed")
        actual = _recorded_artifact(path)
        if record.get("kind") != actual.get("kind"):
            raise RuntimeError("production manifest artifact kind changed")
        if record["kind"] == "npz" \
                and record.get("arrays") != actual.get("arrays"):
            raise RuntimeError("production manifest NPZ dtype/shape/finite changed")
        if record["kind"] == "json" and (
            record.get("schema") != actual.get("schema")
            or record.get("status") != actual.get("status")
        ):
            raise RuntimeError("production manifest JSON schema or status changed")
    smc_records = manifest.get("SMC_artifacts", [])
    if len(smc_records) != (7 if outcome_kind == "terminal" else 1):
        raise RuntimeError("production manifest SMC artifact count changed")
    _validate_smc_artifact_records(smc_records, outcome_kind)
    smc_directory = root / "smc"
    if {path.resolve() for path in smc_directory.iterdir() if path.is_file()} != {
        Path(row["path"]).resolve() for row in smc_records
    }:
        raise RuntimeError("SMC artifact set changed after publication")
    capability_path = Path(
        smc_records[-1 if outcome_kind == "terminal" else 0]["path"]
    )
    capability_summary = json.loads(capability_path.read_text())
    capability_status, capability_outcome = _validate_capability_summary(
        capability_summary
    )
    _require_downstream_closed(
        capability_summary, CAPABILITY_DECISION_KEYS
    )
    if outcome_kind == "terminal":
        terminal_record = smc_records[4]
        cf4_record = smc_records[5]
        if capability_summary.get("terminal_parent_frozen") != terminal_record["path"] \
                or capability_summary.get("terminal_parent_frozen_sha256") != terminal_record["sha256"] \
                or capability_summary.get("post_terminal_arrays") != cf4_record["path"] \
                or capability_summary.get("post_terminal_arrays_sha256") != cf4_record["sha256"]:
            raise RuntimeError("terminal capability artifact binding changed")
    elif capability_summary.get("CF4_calibration_opened") is not False \
            or capability_summary.get(
                "automatic_retry_retune_or_scale_up_authorized"
            ) is not False:
        raise RuntimeError("architecture-stop closed-state flags changed")
    if capability_status != status or capability_outcome != outcome_kind \
            or result.get("capability_status") != capability_summary.get("status") \
            or result.get("capability_gates") != capability_summary.get("gates") \
            or failure != capability_summary.get("failure_class"):
        raise RuntimeError("production capability status or gates changed")
    _validate_cache_bundle(
        Path(manifest["production_cache_manifest"]["path"]),
        manifest.get("production_cache_shards", []),
    )
    if manifest["result"]["sha256"] != sha256_file(result_path):
        raise RuntimeError("production manifest result binding changed")
    control_summary = json.loads(Path(control_record["path"]).read_text())
    manifest_selection_sha = _require_sha256(
        manifest.get("control_selection_sha256"),
        "production manifest control selection",
    )
    control_selection_sha = _require_sha256(
        control_summary.get("selection_sha256"),
        "sealed control selection",
    )
    if manifest_selection_sha != control_selection_sha:
        raise RuntimeError("sealed control selection binding changed")
    _require_downstream_closed(result, RESULT_DECISION_KEYS)
    return {
        "status": status,
        "outcome_kind": outcome_kind,
        "failure_class": failure,
        "valid_scientific_complete": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    args = parser.parse_args()
    result = run_production(args.program)
    print(
        f"[aggregate-evidence-smc] status={result['status']} "
        f"failure_class={result['failure_class']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
