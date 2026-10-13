#!/usr/bin/env python
"""Audit V74 gate attainability under the actual fresh 32-query group design."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import hong2021_v73_gate_attainability as v73
import hong2021_v74_gate_redesign as v74
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor


PROGRAM_SCHEMA = "hong2021-v77-fresh32-partition-compatibility-audit-program-v1"
PROGRAM_STATUS = "frozen_before_V77_implementation_or_compatible_design_bootstrap_draws"
PROGRAM_SHA256 = "43ea962c8a1f1143731c0e968a1e416bdc92c7c269e1a09f9aa51a23d365c2e1"
PROGRAM_FREEZE_COMMIT = "afaa4125d2e03392ce81436416758cfd47b5cd5a"
RESULT_SCHEMA = "hong2021-v77-fresh32-partition-compatibility-audit-result-v1"
DOMAIN_ORDER = v73.DOMAIN_ORDER
QUERY_COUNT = 32
V76_FAMILY_FAILURE_UPPER = 0.05


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def resolve_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    path = path.resolve()
    repo = repo.resolve()
    if sha256_file(path) != PROGRAM_SHA256:
        raise ValueError("V77 program hash differs")
    program = strict_json(path)
    limits = program.get("scope_limits", {})
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or limits.get("this_is_not_a_generator") is not True
        or limits.get("validation_input_or_target_payload_access") is not False
        or limits.get("physical_morphology_threshold_relaxation") is not False
        or limits.get("V76_rule_change") is not False
    ):
        raise ValueError("V77 schema or firewall differs")
    parent = program["parent_evidence"]
    local_keys = (
        "V76_result_record",
        "V74_result_record",
        "V73_result_record",
        "V72_result_record",
        "V72_feasibility_record",
        "V74_source",
    )
    paths = {key: resolve_path(repo, parent[key]) for key in local_keys}
    paths.update(
        {
            key: Path(parent[key]).resolve()
            for key in ("V73_summary_record", "V73_summary_cache")
        }
    )
    for key, bound_path in paths.items():
        if sha256_file(bound_path) != parent[f"{key}_sha256"]:
            raise ValueError(f"V77 parent differs: {key}")
    v76 = strict_json(paths["V76_result_record"])
    if (
        v76["decision"]["classification"]
        != "separate_exact_label_rank_coverage_rule_selected"
        or v76["authorization"]["freeze_a_complete_prospective_gate_specification"]
        is not True
    ):
        raise ValueError("V77 V76 authorization differs")
    summary = strict_json(paths["V73_summary_record"])
    if summary.get("summary_cache_sha256") != parent["V73_summary_cache_sha256"]:
        raise ValueError("V77 summary provenance differs")
    return program, paths


def sample_compatible_queries(
    domain: str, groups: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if domain == "TNG100":
        if len(unique) != 4:
            raise ValueError("V77 TNG group count differs")
        quota_values = rng.permutation(np.asarray([6, 8, 9, 9]))
        quota = dict(zip(map(int, unique), map(int, quota_values)))
    elif domain == "SIMBA":
        if len(unique) != 8:
            raise ValueError("V77 SIMBA group count differs")
        chosen = rng.choice(unique, size=3, replace=False)
        quota_values = rng.permutation(np.asarray([7, 10, 15]))
        quota = dict(zip(map(int, chosen), map(int, quota_values)))
    elif domain == "Swift":
        if len(unique) != 20:
            raise ValueError("V77 Swift group count differs")
        chosen = rng.choice(unique, size=7, replace=False)
        five = set(map(int, rng.choice(chosen, size=4, replace=False)))
        quota = {int(group): (5 if int(group) in five else 4) for group in chosen}
    else:
        raise ValueError("unknown V77 domain")
    selected: list[int] = []
    for group, number in quota.items():
        pool = np.flatnonzero(groups == group)
        if len(pool) <= number:
            raise ValueError("V77 needs at least one non-query oracle donor per group")
        selected.extend(rng.choice(pool, size=number, replace=False).tolist())
    output = np.asarray(selected, dtype=np.int64)
    if len(output) != QUERY_COUNT or len(np.unique(output)) != QUERY_COUNT:
        raise ValueError("V77 compatible query count differs")
    return output


def rows_to_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def run_phase(
    summaries: dict[str, dict[str, np.ndarray]],
    k: np.ndarray,
    count: np.ndarray,
    radius: np.ndarray,
    trials: int,
    seed: int,
    phase_name: str,
) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    output = {}
    for domain in DOMAIN_ORDER:
        summary = summaries[domain]
        rows = []
        for trial in range(trials):
            queries = sample_compatible_queries(domain, summary["group"], rng)
            oracle_a = v73.sample_same_group_oracle(summary["group"], queries, rng)
            oracle_b = v73.sample_same_group_oracle(summary["group"], queries, rng)
            row = v73.trial_metrics(
                summary, queries, summary, oracle_a, k, count, radius
            )
            row["energy_A_minus_B"] = v74.energy_delta(
                summary, queries, oracle_a, oracle_b
            )
            rows.append(row)
            if (trial + 1) % 2000 == 0:
                print(
                    f"[v77-{phase_name}] {domain} {trial + 1}/{trials}",
                    flush=True,
                )
        output[domain] = rows_to_arrays(rows)
    return output


def flatten_phase(phase: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        f"{domain}__{key}": value
        for domain, arrays in phase.items()
        for key, value in arrays.items()
    }


def phase_summary(phase: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    components = (
        "q99_999",
        "Q4",
        "high_k_power",
        "residual_RMS",
        "absolute_core",
        "density_PDF",
        "two_point",
        "environment",
        "morphology_core",
        "joint",
    )
    domains = {
        domain: {key: v74.probability_row(values[key]) for key in components}
        for domain, values in phase.items()
    }
    aligned = np.logical_and.reduce([phase[domain]["joint"] for domain in DOMAIN_ORDER])
    return {
        "domains": domains,
        "all_three_domain_aligned_joint": v74.probability_row(aligned),
    }


def energy_margins(
    calibration: dict[str, dict[str, np.ndarray]], quantile: float
) -> dict[str, float]:
    return {
        domain: float(
            np.quantile(values["energy_A_minus_B"], quantile, method="linear")
        )
        for domain, values in calibration.items()
    }


def energy_verification(
    verification: dict[str, dict[str, np.ndarray]], margins: dict[str, float]
) -> dict[str, Any]:
    domains = {}
    pass_rows = []
    for domain in DOMAIN_ORDER:
        delta = verification[domain]["energy_A_minus_B"]
        passed = delta <= margins[domain]
        pass_rows.append(passed)
        domains[domain] = {
            "upper_margin": margins[domain],
            "pass": v74.probability_row(passed),
            "false_rejection": v74.probability_row(~passed),
            "delta_quantiles_2p5_50_97p5": np.quantile(
                delta, [0.025, 0.5, 0.975]
            ).tolist(),
        }
    family_pass = np.logical_and.reduce(pass_rows)
    family_failure = v74.probability_row(~family_pass)
    return {
        "domains": domains,
        "all_three_domain_aligned_pass": v74.probability_row(family_pass),
        "family_wise_false_rejection": family_failure,
        "calibrated": bool(
            family_failure["probability"] <= 0.06
            and family_failure["Wilson_95"][1] <= 0.065
        ),
    }


def conservative_complete_lower(
    physical_joint_wilson_lower: float,
    energy_family_wilson_upper: float,
    rank_coverage_family_upper: float = V76_FAMILY_FAILURE_UPPER,
) -> dict[str, float | bool]:
    physical_failure = 1.0 - physical_joint_wilson_lower
    total_failure = physical_failure + energy_family_wilson_upper + rank_coverage_family_upper
    lower = max(0.0, 1.0 - total_failure)
    return {
        "physical_morphology_failure_upper": physical_failure,
        "energy_failure_upper": energy_family_wilson_upper,
        "rank_coverage_failure_upper": rank_coverage_family_upper,
        "union_bound_total_failure_upper": total_failure,
        "complete_gate_pass_lower": lower,
        "required_lower": 0.8,
        "pass": lower >= 0.8,
    }


def verify_fresh_selection(
    program: dict[str, Any], v72_result: dict[str, Any], feasibility: dict[str, Any]
) -> dict[str, Any]:
    spec = program["metadata_only_selection"]
    historical = feasibility["historical_development_consumption"]
    consumed = v72_result["fresh_stage_A"]["source_indices"]
    output = {}
    for domain in DOMAIN_ORDER:
        row = spec[domain]
        path = Path(row["validation_path"])
        with h5py.File(path, "r") as handle:
            total = int(handle["input"].shape[0])
            if handle["target"].shape[0] != total:
                raise ValueError("V77 validation metadata length differs")
            if domain == "TNG100":
                position = np.asarray(handle["center_position_mpc_h"])
                groups = 2 * (position[:, 0] >= 37.5).astype(np.int64) + (
                    position[:, 2] >= 37.5
                ).astype(np.int64)
            else:
                groups = np.asarray(handle["realization"], dtype=np.int64)
        excluded = set(map(int, historical[domain])) | set(map(int, consumed[domain]))
        available = np.asarray([index for index in range(total) if index not in excluded])
        available_counts = {
            str(group): int(np.count_nonzero(groups[available] == int(group)))
            for group in row["available_counts"]
        }
        selected = np.asarray(row["selected_indices"], dtype=np.int64)
        quota = {key: int(value) for key, value in row["quota"].items()}
        observed_quota = {
            key: int(np.count_nonzero(groups[selected] == int(key))) for key in quota
        }
        passed = bool(
            len(selected) == QUERY_COUNT
            and len(np.unique(selected)) == QUERY_COUNT
            and not (set(map(int, selected)) & excluded)
            and set(map(int, selected)).issubset(set(map(int, available)))
            and available_counts == {
                key: int(value) for key, value in row["available_counts"].items()
            }
            and observed_quota == quota
        )
        output[domain] = {
            "objects": total,
            "excluded_historical": len(historical[domain]),
            "excluded_consumed_V72_stage_A": len(consumed[domain]),
            "untouched_available": len(available),
            "available_counts": available_counts,
            "observed_selected_quota": observed_quota,
            "selected_indices": selected.tolist(),
            "pass": passed,
        }
    output["all_domains_pass"] = all(output[d]["pass"] for d in DOMAIN_ORDER)
    output["input_voxels_read"] = False
    output["target_voxels_read"] = False
    return output


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial, path)


def run(program_path: Path, repo: Path, output_root: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, parents = load_program(program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V77 requires a clean frozen Lageunha checkout")
    paths = {
        key: Path(program["outputs"][key]).resolve()
        for key in ("root", "calibration_arrays", "verification_arrays", "audit_result")
    }
    if output_root.resolve() != paths["root"]:
        raise ValueError("V77 output root differs")
    if output_root.exists() or any(paths[key].exists() for key in paths if key != "root"):
        raise FileExistsError("V77 refuses an existing output")
    output_root.mkdir(parents=True)
    fresh = verify_fresh_selection(
        program,
        strict_json(parents["V72_result_record"]),
        strict_json(parents["V72_feasibility_record"]),
    )
    with np.load(parents["V73_summary_cache"], allow_pickle=False) as cache:
        summaries = {domain: v73._domain_summary(cache, domain) for domain in DOMAIN_ORDER}
        k = np.asarray(cache["fourier_k"], dtype=np.float64)
        count = np.asarray(cache["fourier_mode_count"], dtype=np.int64)
        radius = np.asarray(cache["radius_mpc_h"], dtype=np.float64)
    monte = program["independent_monte_carlo"]
    calibration = run_phase(
        summaries, k, count, radius,
        int(monte["calibration"]["trials_per_domain"]),
        int(monte["calibration"]["seed"]), "calibration",
    )
    margins = energy_margins(
        calibration, float(program["maximum_energy_recalibration"]["calibration_quantile"])
    )
    write_npz(paths["calibration_arrays"], flatten_phase(calibration))
    verification = run_phase(
        summaries, k, count, radius,
        int(monte["verification"]["trials_per_domain"]),
        int(monte["verification"]["seed"]), "verification",
    )
    write_npz(paths["verification_arrays"], flatten_phase(verification))
    verification_summary = phase_summary(verification)
    physical_row = verification_summary["all_three_domain_aligned_joint"]
    domain_absolute = {
        domain: verification_summary["domains"][domain]["absolute_core"]
        for domain in DOMAIN_ORDER
    }
    physical_pass = bool(
        physical_row["Wilson_95"][0] >= 0.8
        and all(row["Wilson_95"][0] >= 0.8 for row in domain_absolute.values())
    )
    energy = energy_verification(verification, margins)
    complete = conservative_complete_lower(
        physical_row["Wilson_95"][0],
        energy["family_wise_false_rejection"]["Wilson_95"][1],
    )
    selected = bool(
        fresh["all_domains_pass"]
        and physical_pass
        and energy["calibrated"]
        and complete["pass"]
    )
    classification = (
        "fresh32_partition_compatible_complete_gate_may_be_frozen"
        if selected
        else "fresh32_partition_or_gate_requires_additional_redesign"
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete_V77_fresh32_partition_compatibility_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "audit_code_commit": commit,
        "worktree_clean": clean,
        "fresh_metadata_selection": fresh,
        "calibration": phase_summary(calibration),
        "verification": verification_summary,
        "physical_morphology_attainability": {
            "aligned_joint": physical_row,
            "domain_absolute_core": domain_absolute,
            "pass": physical_pass,
        },
        "maximum_energy_recalibration": {
            "upper_margins": margins,
            "verification": energy,
        },
        "conservative_complete_gate_attainability": complete,
        "decision": {
            "compatible_design_sufficient": selected,
            "classification": classification,
            "complete_candidate_agnostic_gate_may_be_frozen": selected,
            "candidate_or_fresh_payload_execution_authorized": False,
            "next": (
                "freeze_complete_candidate_agnostic_V78_gate_then_await_explicit_approval"
                if selected
                else "stop_before_complete_gate_or_candidate_and_report_failed_requirement"
            ),
        },
        "artifacts": {
            "calibration_arrays": str(paths["calibration_arrays"]),
            "calibration_arrays_sha256": sha256_file(paths["calibration_arrays"]),
            "verification_arrays": str(paths["verification_arrays"]),
            "verification_arrays_sha256": sha256_file(paths["verification_arrays"]),
        },
        "validation_input_or_target_payload_accessed": False,
        "training_or_model_sampling_performed": False,
        "raw_fit_train_truth_reread": False,
        "V72_stage_B_accessed": False,
        "Astrid_accessed": False,
        "historical_or_independent_EAGLE_accessed": False,
        "new_candidate_authorized": False,
        "V72_verdict_changed": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    partial = paths["audit_result"].with_suffix(".json.partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, paths["audit_result"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.program, args.repo, args.output_root), indent=2), flush=True)


if __name__ == "__main__":
    main()
