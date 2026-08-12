#!/usr/bin/env python
"""Apply the frozen V79 complete gate to one separately approved candidate."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import hong2021_v75_rank_coverage_exact_null as v75
import hong2021_v78_global_conditional_null as v78
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v20_development_gate import marginal_diagnostics
from hong2021_v63_train import _is_ancestor
from hong2021_v6_gate import ENVIRONMENT_FIELDS, field_gate
from hong2021_v72_sqt import scalar_energy_score


PROGRAM_SCHEMA = "hong2021-v79-complete-candidate-agnostic-gate-program-v1"
PROGRAM_STATUS = (
    "frozen_before_V79_implementation_candidate_design_sampling_or_fresh_payload_access"
)
PROGRAM_SHA256 = "1234ac864343b41cb08f6788892f7dcb2db60b68843bc00edad5d15284e3956d"
PROGRAM_FREEZE_COMMIT = "4eb30e76072fb0bb0e9370d81b4d100377c71c9c"
MANIFEST_SCHEMA = "hong2021-v79-single-use-execution-manifest-v1"
CANDIDATE_PROGRAM_SCHEMA = "hong2021-v79-single-candidate-program-v1"
CANDIDATE_PROGRAM_STATUS = (
    "frozen_and_pushed_before_V79_sampling_or_selected_payload_access"
)
RESULT_SCHEMA = "hong2021-v79-complete-single-use-gate-decision-v1"
SEAL_SCHEMA = "hong2021-v79-complete-single-use-terminal-seal-v1"
EVALUATION_SCHEMA = "hong2021-stochastic-residual-ensemble-evaluation-v1"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
MEMBERS = 16
QUERIES = 32
FIELD_COUNT = MEMBERS + 1
SAMPLE_SHAPE = (QUERIES, MEMBERS, 1, 64, 64, 64)
FIELD_SHAPE = (QUERIES, 1, 64, 64, 64)
SOURCE_SHAPE = (QUERIES,)
POWER_BANDS = ("3-6_h_mpc", "6-10.0531_h_mpc")
TWO_POINT_SCALES = ("0-1_mpc_h", "1-3_mpc_h", "3-10_mpc_h")
RELABELINGS = 99_999
BLOCK_ALPHA = 0.025
GLOBAL_ALPHA = 0.05
MAXIMUM_RESIDUAL_DC = 1.0e-7


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def resolve_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"V79 {label} keys differ")


def _hash_row(value: dict[str, Any], label: str) -> tuple[Path, str]:
    _exact_keys(value, {"path", "sha256"}, label)
    path = Path(str(value["path"])).resolve()
    digest = str(value["sha256"])
    if len(digest) != 64 or sha256_file(path) != digest:
        raise ValueError(f"V79 {label} hash differs")
    return path, digest


def _finite_tree(value: Any, label: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{label}[{index}]")
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError(f"V79 nonfinite value at {label}")


def frozen_indices(program: dict[str, Any]) -> dict[str, list[int]]:
    selection = program["single_use_fresh_selection"]
    output = {domain: list(map(int, selection[domain])) for domain in DOMAIN_ORDER}
    limits = {"TNG100": 93, "SIMBA": 64, "Swift": 148}
    for domain, indices in output.items():
        if (
            len(indices) != QUERIES
            or len(set(indices)) != QUERIES
            or min(indices) < 0
            or max(indices) >= limits[domain]
        ):
            raise ValueError(f"V79 frozen {domain} indices differ")
    return output


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    path = path.resolve()
    repo = repo.resolve()
    if sha256_file(path) != PROGRAM_SHA256:
        raise ValueError("V79 program hash differs")
    program = strict_json(path)
    authorization = program.get("implementation_only_authorization", {})
    decision = program.get("complete_decision", {})
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or authorization.get("implement_and_unit_test_candidate_agnostic_gate") is not True
        or authorization.get("read_any_selected_validation_input_or_target") is not False
        or authorization.get("construct_train_sample_or_evaluate_candidate") is not False
        or authorization.get("run_V79_gate") is not False
        or float(decision.get("global_alpha", -1)) != GLOBAL_ALPHA
    ):
        raise ValueError("V79 schema, firewall, or alpha differs")

    parents = program["parent_evidence"]
    measurements = program["measurement_sources"]
    references = program["frozen_reference_inputs"]
    paths: dict[str, Path] = {}
    for key in (
        "V78_result_record",
        "V78_program",
        "V78_source",
        "V77_result_record",
        "V76_result_record",
    ):
        paths[key] = resolve_path(repo, parents[key])
        if sha256_file(paths[key]) != parents[f"{key}_sha256"]:
            raise ValueError(f"V79 parent differs: {key}")
    for key in (
        "ensemble_evaluator",
        "field_gate_diagnostics",
        "V72_measurement_helpers",
        "exact_label_helpers",
        "global_null_helpers",
    ):
        paths[key] = resolve_path(repo, measurements[key])
        if sha256_file(paths[key]) != measurements[f"{key}_sha256"]:
            raise ValueError(f"V79 measurement source differs: {key}")
    for key in ("V77_calibration_arrays", "V77_verification_arrays"):
        paths[key] = Path(references[key]).resolve()
        if sha256_file(paths[key]) != references[f"{key}_sha256"]:
            raise ValueError(f"V79 reference differs: {key}")

    v78_result = strict_json(paths["V78_result_record"])
    if (
        v78_result.get("decision", {}).get("classification")
        != parents["V78_selection_required"]
        or v78_result.get("decision", {}).get(
            "complete_candidate_agnostic_specification_may_be_frozen"
        )
        is not True
        or v78_result.get("decision", {}).get(
            "candidate_or_fresh_payload_execution_authorized"
        )
        is not False
    ):
        raise ValueError("V79 V78 selection or authorization differs")
    frozen_indices(program)
    return program, paths


def load_candidate_program(
    row: dict[str, Any], repo: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Any], Path, str, str]:
    _exact_keys(row, {"path", "sha256", "freeze_commit"}, "candidate program row")
    path = resolve_path(repo, str(row["path"]))
    digest = str(row["sha256"])
    freeze_commit = str(row["freeze_commit"])
    if sha256_file(path) != digest or len(freeze_commit) != 40:
        raise ValueError("V79 candidate program binding differs")
    candidate = strict_json(path)
    authorization = candidate.get("authorization", {})
    if (
        candidate.get("schema") != CANDIDATE_PROGRAM_SCHEMA
        or candidate.get("status") != CANDIDATE_PROGRAM_STATUS
        or candidate.get("V79_program_sha256") != PROGRAM_SHA256
        or candidate.get("candidate_count") != 1
        or authorization.get(
            "explicit_user_approval_for_candidate_design_and_single_use_execution"
        )
        is not True
        or authorization.get("V79_single_use_gate_execution") is not True
        or authorization.get("post_disclosure_candidate_change_or_retry") is not False
        or candidate.get("single_use_fresh_selection")
        != manifest.get("source_indices")
        or candidate.get("frozen_execution_provenance")
        != manifest.get("provenance")
        or candidate.get("frozen_domain_execution_contracts")
        != {
            domain: {
                key: manifest["domains"][domain][key]
                for key in (
                    "candidate_expected_attrs",
                    "control_expected_attrs",
                    "pairing",
                )
            }
            for domain in DOMAIN_ORDER
        }
    ):
        raise ValueError("V79 candidate program or explicit authorization differs")
    return candidate, path, digest, freeze_commit


def validate_frozen_provenance(
    provenance: dict[str, Any], manifest_domains: dict[str, Any], repo: Path,
    candidate_freeze_commit: str,
) -> dict[str, str]:
    _exact_keys(
        provenance,
        {
            "candidate_count",
            "queries_per_domain",
            "members_per_query",
            "sampler",
            "seeds_by_domain",
            "frozen_code_configuration_checkpoint_and_cache_artifacts",
            "innovation_pairing_digests",
            "post_disclosure_sampler_seed_member_checkpoint_or_cache_change",
        },
        "frozen provenance",
    )
    if (
        provenance["candidate_count"] != 1
        or provenance["queries_per_domain"] != QUERIES
        or provenance["members_per_query"] != MEMBERS
        or not isinstance(provenance["sampler"], dict)
        or not provenance["sampler"]
        or provenance[
            "post_disclosure_sampler_seed_member_checkpoint_or_cache_change"
        ]
        is not False
        or set(provenance["seeds_by_domain"]) != set(DOMAIN_ORDER)
        or set(provenance["innovation_pairing_digests"]) != set(DOMAIN_ORDER)
    ):
        raise ValueError("V79 frozen execution provenance differs")
    for domain in DOMAIN_ORDER:
        seeds = provenance["seeds_by_domain"][domain]
        _exact_keys(seeds, {"candidate", "control"}, f"{domain} seeds")
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in seeds.values()):
            raise ValueError(f"V79 {domain} seed binding differs")
        digest = str(provenance["innovation_pairing_digests"][domain])
        if (
            len(digest) != 64
            or manifest_domains[domain]["pairing"]["innovation_pairing_digest"]
            != digest
        ):
            raise ValueError(f"V79 {domain} pairing provenance differs")
    artifacts = provenance[
        "frozen_code_configuration_checkpoint_and_cache_artifacts"
    ]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("V79 frozen artifact provenance is empty")
    required_artifacts = {
        "candidate_sampling_source",
        "control_sampling_source",
        "candidate_configuration",
        "control_configuration",
        "candidate_checkpoint_or_frozen_state",
        "control_checkpoint_or_frozen_state",
        *{
            f"{domain}_{suffix}"
            for domain in DOMAIN_ORDER
            for suffix in ("source_data", "source_cache")
        },
    }
    if not required_artifacts.issubset(artifacts):
        raise ValueError("V79 required code/configuration/input artifacts are missing")
    hashes = {
        label: _hash_row(row, f"provenance artifact {label}")[1]
        for label, row in artifacts.items()
    }
    required_attrs = {
        "ensemble_members",
        "sampler",
        "sampler_steps",
        "sampling_batch_size",
        "noise_seed",
        "checkpoint_sha256",
        "source_data_sha256",
        "source_cache_sha256",
        "sampling_code_commit",
        "innovation_pairing_digest",
        "worktree_clean_at_sampling",
        "complete",
    }
    for domain in DOMAIN_ORDER:
        for arm in ("candidate", "control"):
            attrs = manifest_domains[domain][f"{arm}_expected_attrs"]
            if not required_attrs.issubset(attrs):
                raise ValueError(f"V79 {domain} {arm} sampler metadata is incomplete")
            expected_checkpoint = hashes[f"{arm}_checkpoint_or_frozen_state"]
            if (
                attrs["ensemble_members"] != MEMBERS
                or attrs["noise_seed"] != provenance["seeds_by_domain"][domain][arm]
                or attrs["checkpoint_sha256"] != expected_checkpoint
                or attrs["source_data_sha256"] != hashes[f"{domain}_source_data"]
                or attrs["source_cache_sha256"] != hashes[f"{domain}_source_cache"]
                or attrs["innovation_pairing_digest"]
                != provenance["innovation_pairing_digests"][domain]
                or attrs["worktree_clean_at_sampling"] is not True
                or attrs["complete"] is not True
                or not isinstance(attrs["sampler_steps"], int)
                or attrs["sampler_steps"] <= 0
                or not isinstance(attrs["sampling_batch_size"], int)
                or attrs["sampling_batch_size"] <= 0
                or not _is_ancestor(
                    repo, str(attrs["sampling_code_commit"]), candidate_freeze_commit
                )
            ):
                raise ValueError(f"V79 {domain} {arm} frozen sampler provenance differs")
    return hashes


def load_manifest(
    path: Path, program: dict[str, Any], repo: Path, gate_commit: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = strict_json(path.resolve())
    _exact_keys(
        manifest,
        {
            "schema",
            "status",
            "V79",
            "candidate_program",
            "preflight",
            "source_indices",
            "domains",
            "provenance",
            "single_use",
            "output_root",
        },
        "manifest",
    )
    if (
        manifest["schema"] != MANIFEST_SCHEMA
        or manifest["status"] != "complete_single_use_artifacts_bound_before_gate"
        or manifest["source_indices"] != frozen_indices(program)
        or set(manifest["domains"]) != set(DOMAIN_ORDER)
    ):
        raise ValueError("V79 manifest identity or selection differs")
    _exact_keys(
        manifest["V79"],
        {"program_sha256", "gate_source", "gate_implementation_commit"},
        "V79 binding",
    )
    source_path, source_sha = _hash_row(manifest["V79"]["gate_source"], "gate source")
    expected_source = (repo / "src/hong2021_v79_complete_gate.py").resolve()
    implementation_commit = str(manifest["V79"]["gate_implementation_commit"])
    if (
        manifest["V79"]["program_sha256"] != PROGRAM_SHA256
        or source_path != expected_source
        or len(implementation_commit) != 40
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, implementation_commit)
        or not _is_ancestor(repo, implementation_commit, gate_commit)
    ):
        raise ValueError("V79 gate implementation binding differs")

    candidate, candidate_path, candidate_sha, freeze_commit = load_candidate_program(
        manifest["candidate_program"], repo, manifest
    )
    if (
        not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, freeze_commit)
        or not _is_ancestor(repo, freeze_commit, gate_commit)
    ):
        raise ValueError("V79 candidate freeze ancestry differs")

    preflight_path, preflight_sha = _hash_row(manifest["preflight"]["artifact"], "preflight")
    _exact_keys(
        manifest["preflight"],
        {
            "artifact",
            "selected_payload_accessed_before_candidate_freeze",
            "candidate_program_frozen_and_pushed_before_sampling",
        },
        "preflight row",
    )
    preflight = strict_json(preflight_path)
    if (
        manifest["preflight"]["selected_payload_accessed_before_candidate_freeze"] is not False
        or manifest["preflight"]["candidate_program_frozen_and_pushed_before_sampling"] is not True
        or preflight.get("selected_payload_accessed_before_candidate_freeze") is not False
        or preflight.get("candidate_program_sha256") != candidate_sha
        or preflight.get("V79_program_sha256") != PROGRAM_SHA256
    ):
        raise ValueError("V79 preflight firewall differs")

    _exact_keys(
        manifest["single_use"],
        {"candidate_count", "prior_gate_disclosure", "retry_authorized", "post_disclosure_change_authorized"},
        "single-use row",
    )
    if manifest["single_use"] != {
        "candidate_count": 1,
        "prior_gate_disclosure": False,
        "retry_authorized": False,
        "post_disclosure_change_authorized": False,
    }:
        raise ValueError("V79 single-use contract differs")
    provenance_hashes = validate_frozen_provenance(
        manifest["provenance"],
        manifest["domains"],
        repo,
        freeze_commit,
    )
    _finite_tree(manifest)
    return manifest, {
        "gate_source_sha256": source_sha,
        "gate_implementation_commit": implementation_commit,
        "candidate_program": candidate,
        "candidate_program_path": candidate_path,
        "candidate_program_sha256": candidate_sha,
        "candidate_freeze_commit": freeze_commit,
        "preflight_path": preflight_path,
        "preflight_sha256": preflight_sha,
        "provenance_artifact_sha256": provenance_hashes,
    }


def _attr_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    return value.item() if isinstance(value, np.generic) else value


def _bitwise_equal(first: np.ndarray, second: np.ndarray) -> bool:
    first = np.asarray(first)
    second = np.asarray(second)
    return (
        first.shape == second.shape
        and first.dtype.str == second.dtype.str
        and first.tobytes(order="C") == second.tobytes(order="C")
    )


def _validate_expected_attrs(
    handle: h5py.File, expected: dict[str, Any], label: str
) -> None:
    if not expected:
        raise ValueError(f"V79 {label} has no bound sampler metadata")
    for key, value in expected.items():
        if _attr_value(handle.attrs.get(key)) != value:
            raise ValueError(f"V79 {label} metadata differs: {key}")


def validate_ensemble_pair(
    candidate_path: Path,
    control_path: Path,
    expected_indices: list[int],
    candidate_attrs: dict[str, Any],
    control_attrs: dict[str, Any],
    pairing: dict[str, Any],
) -> dict[str, Any]:
    _exact_keys(
        pairing,
        {"rule", "innovation_pairing_digest", "candidate_control_pairing_proven"},
        "pairing",
    )
    pairing_digest = str(pairing["innovation_pairing_digest"])
    if (
        not pairing["rule"]
        or len(pairing_digest) != 64
        or pairing["candidate_control_pairing_proven"] is not True
    ):
        raise ValueError("V79 innovation pairing proof differs")
    maxima = {}
    with h5py.File(candidate_path, "r") as candidate, h5py.File(control_path, "r") as control:
        for label, handle, attrs in (
            ("candidate", candidate, candidate_attrs),
            ("control", control, control_attrs),
        ):
            if (
                tuple(handle["sample"].shape) != SAMPLE_SHAPE
                or tuple(handle["truth"].shape) != FIELD_SHAPE
                or tuple(handle["conditional_mean"].shape) != FIELD_SHAPE
                or tuple(handle["source_index"].shape) != SOURCE_SHAPE
                or handle["source_index"].dtype.kind not in "iu"
                or not np.array_equal(
                    handle["source_index"][:],
                    np.asarray(expected_indices, dtype=handle["source_index"].dtype),
                )
            ):
                raise ValueError(f"V79 {label} ensemble shape or index differs")
            _validate_expected_attrs(handle, attrs, label)
            if _attr_value(handle.attrs.get("innovation_pairing_digest")) != pairing_digest:
                raise ValueError(f"V79 {label} innovation digest differs")
        for dataset in ("truth", "conditional_mean", "source_index"):
            if not _bitwise_equal(candidate[dataset][:], control[dataset][:]):
                raise ValueError(f"V79 candidate/control {dataset} differs")
        if ("input" in candidate) != ("input" in control):
            raise ValueError("V79 candidate/control input storage differs")
        if "input" in candidate:
            candidate_input = candidate["input"][:]
            control_input = control["input"][:]
            if (
                not _bitwise_equal(candidate_input, control_input)
                or not np.isfinite(candidate_input).all()
                or not np.isfinite(control_input).all()
            ):
                raise ValueError("V79 candidate/control input differs or is nonfinite")
        for label, handle in (("candidate", candidate), ("control", control)):
            maximum = 0.0
            for query in range(QUERIES):
                mean = np.asarray(handle["conditional_mean"][query, 0], dtype=np.float64)
                sample = np.asarray(handle["sample"][query, :, 0], dtype=np.float64)
                truth = np.asarray(handle["truth"][query, 0], dtype=np.float64)
                if (
                    not np.isfinite(mean).all()
                    or not np.isfinite(sample).all()
                    or not np.isfinite(truth).all()
                ):
                    raise ValueError(f"V79 {label} ensemble is nonfinite")
                dc = np.abs((sample - mean).mean(axis=(-3, -2, -1), dtype=np.float64))
                maximum = max(maximum, float(dc.max()))
            maxima[label] = maximum
    return {
        "candidate_maximum_absolute_residual_DC": maxima["candidate"],
        "control_maximum_absolute_residual_DC": maxima["control"],
        "maximum_absolute_residual_DC": max(maxima.values()),
        "residual_DC_pass": max(maxima.values()) <= MAXIMUM_RESIDUAL_DC,
        "truth_conditional_mean_source_index_and_optional_input_bitwise_equal": True,
        "innovation_pairing_digest": pairing_digest,
        "candidate_control_pairing_proven": True,
    }


def load_metrics(path: Path, ensemble_path: Path, expected_indices: list[int]) -> dict[str, Any]:
    payload = strict_json(path)
    candidates = payload.get("candidates")
    if (
        payload.get("schema") != EVALUATION_SCHEMA
        or float(payload.get("voxel_mpc_h", -1)) != 0.3125
        or not isinstance(candidates, dict)
        or len(candidates) != 1
    ):
        raise ValueError("V79 evaluator schema or candidate count differs")
    metrics = next(iter(candidates.values()))
    if (
        Path(str(metrics.get("path"))).resolve() != ensemble_path.resolve()
        or int(metrics.get("objects", -1)) != QUERIES
        or int(metrics.get("ensemble_per_object", -1)) != MEMBERS
        or list(map(int, metrics.get("source_indices", []))) != expected_indices
    ):
        raise ValueError("V79 metrics-to-ensemble binding differs")
    _finite_tree(metrics, "metrics")
    return metrics


def environment_improves(metrics: dict[str, Any]) -> bool:
    environment = metrics["environment"]
    return all(
        abs(environment["generated"][field]["mean"] - environment["truth"][field]["mean"])
        < abs(environment["deterministic"][field]["mean"] - environment["truth"][field]["mean"])
        for field in ENVIRONMENT_FIELDS
    )


def cube_maximum_energy_score(path: Path) -> dict[str, Any]:
    per_query = []
    with h5py.File(path, "r") as handle:
        for query in range(QUERIES):
            generated = 4.5 * np.asarray(
                handle["sample"][query, :, 0], dtype=np.float64
            ).max(axis=(-3, -2, -1))
            truth = 4.5 * float(
                np.asarray(handle["truth"][query, 0], dtype=np.float64).max()
            )
            per_query.append(scalar_energy_score(generated, truth))
    return {
        "mean_unbiased_energy_score": float(np.mean(per_query)),
        "per_query_unbiased_energy_score": per_query,
        "truth_cubes": QUERIES,
        "generated_members_per_truth_query": MEMBERS,
        "unequal_sample_global_maximum_used": False,
    }


def physical_energy_observation(
    domains: dict[str, dict[str, Any]], reference_paths: dict[str, Path]
) -> tuple[float, dict[str, Any]]:
    phase: dict[str, np.ndarray] = {}
    measurements: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        row = domains[domain]
        marginal = marginal_diagnostics(row["candidate_ensemble"])
        candidate_energy = cube_maximum_energy_score(row["candidate_ensemble"])
        control_energy = cube_maximum_energy_score(row["control_ensemble"])
        energy_delta = (
            candidate_energy["mean_unbiased_energy_score"]
            - control_energy["mean_unbiased_energy_score"]
        )
        metrics = row["candidate_metrics"]
        power = metrics["fourier_log_density"]["generated_total_power_over_truth"]
        two_point = metrics["two_point_cosmic_mean"]["generated_vs_truth_ks"]["by_scale"]
        values = {
            "q_delta": marginal["delta_q99_999_dex"],
            "q4_ratio": marginal["generated_over_truth_mean_delta_squared"],
            "power_3_6": power[POWER_BANDS[0]],
            "power_6_10": power[POWER_BANDS[1]],
            "rms_ratio": metrics["residual_calibration"]["generated_over_truth_rms"],
            "oracle_pdf_tv": metrics["density_pdf"]["log10_density_total_variation_generated_truth"],
            "oracle_ks_0_1": two_point[TWO_POINT_SCALES[0]]["mean"],
            "oracle_ks_1_3": two_point[TWO_POINT_SCALES[1]]["mean"],
            "oracle_ks_3_10": two_point[TWO_POINT_SCALES[2]]["mean"],
            "environment": int(environment_improves(metrics)),
            "energy_A_minus_B": energy_delta,
        }
        for key, value in values.items():
            phase[f"{domain}__{key}"] = np.asarray([value], dtype=np.float64)
        measurements[domain] = {
            **values,
            "marginal_diagnostics": marginal,
            "candidate_energy": candidate_energy,
            "control_energy": control_energy,
            "old_field_gate_diagnostic": field_gate(metrics),
            "control_old_field_gate_diagnostic": field_gate(row["control_metrics"]),
        }
    calibration = v78.arrays_dict(reference_paths["V77_calibration_arrays"])
    reference = v78.arrays_dict(reference_paths["V77_verification_arrays"])
    fit = v78.fit_physical_energy(calibration, reference)
    block, components = v78.physical_energy_p(phase, fit)
    family_p = {
        family: float(components["family_p"][0, index])
        for index, family in enumerate(v78.FAMILY_ORDER)
    }
    result = {
        "measurements": measurements,
        "family_p_values": family_p,
        "sparse_statistic": float(components["sparse"][0]),
        "dense_statistic": float(components["dense"][0]),
        "sparse_p": float(components["sparse_p"][0]),
        "dense_p": float(components["dense_p"][0]),
        "block_p": float(block[0]),
        "block_alpha": BLOCK_ALPHA,
    }
    _finite_tree(result, "physical-energy")
    return float(block[0]), result


def ensemble_label_table(path: Path, priority_seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(priority_seed)
    tables = []
    with h5py.File(path, "r") as handle:
        if (
            tuple(handle["sample"].shape) != SAMPLE_SHAPE
            or tuple(handle["truth"].shape) != FIELD_SHAPE
            or tuple(handle["conditional_mean"].shape) != FIELD_SHAPE
        ):
            raise ValueError("V79 exact-label ensemble shape differs")
        for query in range(QUERIES):
            mean = np.asarray(handle["conditional_mean"][query, 0], dtype=np.float64)
            generated = np.asarray(handle["sample"][query, :, 0], dtype=np.float64) - mean
            truth = np.asarray(handle["truth"][query, 0], dtype=np.float64) - mean
            generated -= generated.mean(axis=(-3, -2, -1), keepdims=True)
            truth -= truth.mean()
            tables.append(v75.query_label_table(np.concatenate((generated, truth[None])), rng))
            print(f"[v79-exact-label] query {query + 1}/{QUERIES}", flush=True)
    return v75.stack_query_tables(tables)


def rank_coverage_observation(
    domains: dict[str, dict[str, Any]]
) -> tuple[float, dict[str, Any]]:
    result = {}
    individual = []
    for domain in DOMAIN_ORDER:
        ensemble_sha = domains[domain]["candidate_ensemble_sha256"]
        tie_seed = v75.derived_seed(PROGRAM_SHA256, ensemble_sha, domain, "tie_priority")
        label_seed = v75.derived_seed(PROGRAM_SHA256, ensemble_sha, domain, "relabel")
        table = ensemble_label_table(domains[domain]["candidate_ensemble"], tie_seed)
        observed = v75.observed_statistics(table, label=MEMBERS)
        labels = np.random.default_rng(label_seed).integers(
            0, FIELD_COUNT, size=(RELABELINGS, QUERIES), dtype=np.int8
        )
        null = v75.assignment_statistics(table, labels)
        p_values = v75.conditional_p_values(observed, null)
        individual.extend((p_values["rank_tv"], p_values["coverage_deviation"]))
        result[domain] = {
            "tie_priority_seed": tie_seed,
            "relabel_seed": label_seed,
            "random_relabelings": RELABELINGS,
            "adjacent_tie_fraction": table["adjacent_tie_fraction"],
            "observed": observed,
            "conditional_p_values": {
                "rank_tv": p_values["rank_tv"],
                "coverage_deviation": p_values["coverage_deviation"],
            },
            "null_quantiles": {
                key: np.quantile(null[key], [0.025, 0.5, 0.975]).tolist()
                for key in ("rank_tv", "coverage_deviation")
            },
        }
    block = min(1.0, 6.0 * min(individual))
    output = {
        "domains": result,
        "six_individual_p_values": individual,
        "block_p": block,
        "block_alpha": BLOCK_ALPHA,
        "individual_effective_threshold": 1.0 / 240.0,
    }
    _finite_tree(output, "rank-coverage")
    return block, output


def _artifact_rows(domain: dict[str, Any], label: str) -> dict[str, tuple[Path, str]]:
    _exact_keys(
        domain,
        {
            "candidate_ensemble",
            "control_ensemble",
            "candidate_metrics",
            "control_metrics",
            "candidate_expected_attrs",
            "control_expected_attrs",
            "pairing",
        },
        f"{label} domain",
    )
    return {
        key: _hash_row(domain[key], f"{label} {key}")
        for key in ("candidate_ensemble", "control_ensemble", "candidate_metrics", "control_metrics")
    }


def evaluate_manifest(
    manifest: dict[str, Any], program: dict[str, Any], reference_paths: dict[str, Path]
) -> dict[str, Any]:
    expected = frozen_indices(program)
    domains: dict[str, dict[str, Any]] = {}
    for domain in DOMAIN_ORDER:
        spec = manifest["domains"][domain]
        artifacts = _artifact_rows(spec, domain)
        candidate_ensemble, candidate_sha = artifacts["candidate_ensemble"]
        control_ensemble, control_sha = artifacts["control_ensemble"]
        candidate_metrics_path, candidate_metrics_sha = artifacts["candidate_metrics"]
        control_metrics_path, control_metrics_sha = artifacts["control_metrics"]
        numerical = validate_ensemble_pair(
            candidate_ensemble,
            control_ensemble,
            expected[domain],
            spec["candidate_expected_attrs"],
            spec["control_expected_attrs"],
            spec["pairing"],
        )
        domains[domain] = {
            "candidate_ensemble": candidate_ensemble,
            "candidate_ensemble_sha256": candidate_sha,
            "control_ensemble": control_ensemble,
            "control_ensemble_sha256": control_sha,
            "candidate_metrics_path": candidate_metrics_path,
            "candidate_metrics_sha256": candidate_metrics_sha,
            "control_metrics_path": control_metrics_path,
            "control_metrics_sha256": control_metrics_sha,
            "candidate_metrics": load_metrics(candidate_metrics_path, candidate_ensemble, expected[domain]),
            "control_metrics": load_metrics(control_metrics_path, control_ensemble, expected[domain]),
            "numerical_and_pairing": numerical,
        }
    physical_p, physical = physical_energy_observation(domains, reference_paths)
    rank_p, rank = rank_coverage_observation(domains)
    global_p = float(v78.global_p_value(np.asarray([physical_p]), np.asarray([rank_p]))[0])
    deterministic_pass = all(
        row["numerical_and_pairing"]["residual_DC_pass"] for row in domains.values()
    )
    statistical_pass = global_p > GLOBAL_ALPHA
    candidate_pass = bool(statistical_pass and deterministic_pass)
    provenance = {
        domain: {
            key: value
            for key, value in row.items()
            if key.endswith("sha256") or key == "numerical_and_pairing"
        }
        for domain, row in domains.items()
    }
    result = {
        "physical_energy_global_block": physical,
        "rank_coverage_global_block": rank,
        "complete_global_p": global_p,
        "global_alpha": GLOBAL_ALPHA,
        "statistical_pass": statistical_pass,
        "deterministic_numerical_and_pairing_pass": deterministic_pass,
        "candidate_pass": candidate_pass,
        "domain_provenance": provenance,
        "classification": (
            "single_target_unseen_compatible_population_screen_passed"
            if candidate_pass
            else "single_target_unseen_compatible_population_screen_failed_and_sealed"
        ),
    }
    _finite_tree(result, "decision")
    return result


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(partial, path)


def run(program_path: Path, manifest_path: Path, repo: Path, output_root: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, references = load_program(program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V79 requires a clean frozen Lageunha checkout")
    manifest, bindings = load_manifest(manifest_path.resolve(), program, repo, commit)
    if output_root.resolve() != Path(str(manifest["output_root"])).resolve():
        raise ValueError("V79 output root differs from the manifest")
    decision_path = output_root / "decision.json"
    seal_path = output_root / "sealed_result.json"
    if output_root.exists() or decision_path.exists() or seal_path.exists():
        raise FileExistsError("V79 refuses an existing single-use output")
    output_root.mkdir(parents=True)
    try:
        evaluation = evaluate_manifest(manifest, program, references)
        terminal_reason = "pass_await_new_user_decision" if evaluation["candidate_pass"] else "failure_no_retry"
        status = "complete_single_use_gate_sealed"
    except Exception as error:
        evaluation = {
            "candidate_pass": False,
            "classification": "deterministic_provenance_or_measurement_failure_sealed_no_retry",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        terminal_reason = "failure_no_retry"
        status = "failed_single_use_gate_sealed"
    decision: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": status,
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path.resolve()),
        "gate_code_commit": commit,
        "gate_source_sha256": bindings["gate_source_sha256"],
        "gate_implementation_commit": bindings["gate_implementation_commit"],
        "candidate_program": str(bindings["candidate_program_path"]),
        "candidate_program_sha256": bindings["candidate_program_sha256"],
        "candidate_freeze_commit": bindings["candidate_freeze_commit"],
        "preflight": str(bindings["preflight_path"]),
        "preflight_sha256": bindings["preflight_sha256"],
        "frozen_provenance": manifest["provenance"],
        "provenance_artifact_sha256": bindings["provenance_artifact_sha256"],
        "evaluation": evaluation,
        "terminal_reason": terminal_reason,
        "single_use": True,
        "retry_seed_metric_partition_or_candidate_change_authorized": False,
        "V72_stage_B_accessed": False,
        "Astrid_accessed": False,
        "historical_or_independent_EAGLE_accessed": False,
        "V72_verdict_changed": False,
    }
    decision["decision_digest_sha256"] = canonical_digest(decision)
    _write_json_atomic(decision_path, decision)
    seal: dict[str, Any] = {
        "schema": SEAL_SCHEMA,
        "status": "terminal_single_use_result_sealed",
        "decision": str(decision_path.resolve()),
        "decision_sha256": sha256_file(decision_path),
        "decision_digest_sha256": decision["decision_digest_sha256"],
        "candidate_pass": bool(evaluation["candidate_pass"]),
        "terminal_reason": terminal_reason,
        "retry_authorized": False,
        "next": (
            "await_new_explicit_user_decision_about_independent_confirmation_or_CF4"
            if evaluation["candidate_pass"]
            else "stop_no_retry_or_alternative_fresh_subset"
        ),
    }
    seal["seal_digest_sha256"] = canonical_digest(seal)
    _write_json_atomic(seal_path, seal)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.program, args.manifest, args.repo, args.output_root), indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
