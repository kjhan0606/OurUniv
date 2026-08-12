#!/usr/bin/env python
"""Audit a nonmasking separate-metric replacement for the V75 scalar gate."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor
from hong2021_v74_gate_redesign import wilson_interval
from hong2021_v75_rank_coverage_exact_null import (
    DOMAIN_ORDER,
    FIELD_COUNT,
    _alternative_fields,
    assignment_statistics,
    build_table_from_fields,
    conditional_p_values,
    observed_statistics,
    tail_p_from_reference,
)


PROGRAM_SCHEMA = "hong2021-v76-separate-rank-coverage-nonmasking-audit-program-v1"
PROGRAM_STATUS = "frozen_before_V76_implementation_or_power_synthetic_draws"
PROGRAM_SHA256 = "f72e2b0ff7ae7bd23acea3e120f0fb6e89b4088c57b6b7feb1405eef973796c0"
PROGRAM_FREEZE_COMMIT = "974a450195dded9688b4ff4e7ad0a40fdb117e97"
RESULT_SCHEMA = "hong2021-v76-separate-rank-coverage-nonmasking-audit-result-v1"
METRICS = ("rank_tv", "coverage_deviation")
PER_TEST_ALPHA = 1.0 / 120.0
INDIVIDUAL_POINT_LIMIT = 0.01
INDIVIDUAL_WILSON_UPPER_LIMIT = 0.011
FAMILY_DIAGNOSTIC_LIMIT = 0.055


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
        raise ValueError("V76 program hash differs")
    program = strict_json(path)
    limits = program.get("scope_limits", {})
    rule = program.get("scientific_rule", {})
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or limits.get("this_is_not_a_generator") is not True
        or limits.get("validation_or_fresh_partition_access") is not False
        or limits.get("V72_ensemble_or_array_access") is not False
        or rule.get("per_test_alpha") != PER_TEST_ALPHA
        or rule.get("number_of_tests") != 6
    ):
        raise ValueError("V76 schema, firewall, or multiplicity differs")
    parents = program["parent_evidence"]
    bindings = {
        "V75_result_record": resolve_path(repo, parents["V75_result_record"]),
        "V75_program": resolve_path(repo, parents["V75_program"]),
        "V75_source": resolve_path(repo, parents["V75_source"]),
        "V75_audit_result": Path(parents["V75_audit_result"]).resolve(),
        "V75_synthetic_arrays": Path(parents["V75_synthetic_arrays"]).resolve(),
    }
    for key, bound_path in bindings.items():
        if sha256_file(bound_path) != parents[f"{key}_sha256"]:
            raise ValueError(f"V76 parent differs: {key}")
    return program, bindings


def probability_row(values: np.ndarray) -> dict[str, Any]:
    boolean = np.asarray(values, dtype=bool).reshape(-1)
    lower, upper = wilson_interval(int(boolean.sum()), len(boolean))
    return {
        "successes": int(boolean.sum()),
        "trials": int(len(boolean)),
        "probability": float(boolean.mean()),
        "Wilson_95": [float(lower), float(upper)],
    }


def separate_pass(rank_p: float, coverage_p: float) -> bool:
    return bool(rank_p > PER_TEST_ALPHA and coverage_p > PER_TEST_ALPHA)


def null_calibration(
    program: dict[str, Any], arrays_path: Path
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    spec = program["existing_null_calibration_audit"]
    result: dict[str, Any] = {}
    p_arrays: dict[str, np.ndarray] = {}
    with np.load(arrays_path) as arrays:
        for scenario in spec["scenarios"]:
            p_values = {}
            metric_rows = {}
            for metric in METRICS:
                prefix = f"null__{scenario}"
                reference = np.asarray(
                    arrays[f"{prefix}__calibration__{metric}"], dtype=np.float64
                )
                verification = np.asarray(
                    arrays[f"{prefix}__verification__{metric}"], dtype=np.float64
                )
                if (
                    reference.shape != (100000,)
                    or verification.shape != (100000,)
                    or not np.isfinite(reference).all()
                    or not np.isfinite(verification).all()
                ):
                    raise ValueError("V76 V75 null array differs")
                current = tail_p_from_reference(reference, verification)
                rejected = current <= PER_TEST_ALPHA
                row = probability_row(rejected)
                row["calibrated"] = bool(
                    row["probability"] <= INDIVIDUAL_POINT_LIMIT
                    and row["Wilson_95"][1] <= INDIVIDUAL_WILSON_UPPER_LIMIT
                )
                row["p_value_quantiles_0_2p5_50_97p5_100"] = np.quantile(
                    current, [0.0, 0.025, 0.5, 0.975, 1.0]
                ).tolist()
                metric_rows[metric] = row
                p_values[metric] = current
                p_arrays[f"null__{scenario}__{metric}_p"] = current
            domain_union = (p_values["rank_tv"] <= PER_TEST_ALPHA) | (
                p_values["coverage_deviation"] <= PER_TEST_ALPHA
            )
            domain_row = probability_row(domain_union)
            extrapolated = float(1.0 - (1.0 - domain_row["probability"]) ** 3)
            result[scenario] = {
                "metrics": metric_rows,
                "within_domain_union_rejection": domain_row,
                "three_independent_domain_extrapolated_rejection": extrapolated,
                "family_diagnostic_calibrated": extrapolated
                <= FAMILY_DIAGNOSTIC_LIMIT,
            }
    individual_calibrated = all(
        result[scenario]["metrics"][metric]["calibrated"]
        for scenario in spec["scenarios"]
        for metric in METRICS
    )
    family_calibrated = all(
        result[scenario]["family_diagnostic_calibrated"]
        for scenario in spec["scenarios"]
    )
    return {
        "scenarios": result,
        "per_test_alpha": PER_TEST_ALPHA,
        "mathematical_Bonferroni_FWER_upper": 6 * PER_TEST_ALPHA,
        "individual_calibrated": individual_calibrated,
        "family_diagnostic_calibrated": family_calibrated,
    }, p_arrays


def nonmasking_audit(program: dict[str, Any]) -> dict[str, Any]:
    grid = np.asarray(
        program["deterministic_nonmasking_audit"]["p_value_grid"], dtype=np.float64
    )
    comparisons = 0
    mismatches = []
    for rank_p in grid:
        for coverage_p in grid:
            measured = separate_pass(float(rank_p), float(coverage_p))
            expected = bool(
                rank_p > PER_TEST_ALPHA and coverage_p > PER_TEST_ALPHA
            )
            comparisons += 1
            if measured != expected:
                mismatches.append([float(rank_p), float(coverage_p)])
    rank_only_detected = not separate_pass(PER_TEST_ALPHA / 2.0, 1.0)
    coverage_only_detected = not separate_pass(1.0, PER_TEST_ALPHA / 2.0)
    passed = not mismatches and rank_only_detected and coverage_only_detected
    return {
        "grid": grid.tolist(),
        "comparisons": comparisons,
        "mismatches": mismatches,
        "rank_only_signal_detected": rank_only_detected,
        "coverage_only_signal_detected": coverage_only_detected,
        "nonmasking_invariant_pass": passed,
    }


def power_audit(
    program: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    spec = program["independent_synthetic_power_audit"]
    rng = np.random.default_rng(int(spec["seed"]))
    replications = int(spec["replications_per_alternative"])
    queries = int(spec["queries"])
    relabelings = int(spec["random_relabelings_per_replication"])
    result = {}
    arrays = {}
    for alternative in spec["alternatives"]:
        rank_p = np.empty(replications, dtype=np.float64)
        coverage_p = np.empty(replications, dtype=np.float64)
        for replication in range(replications):
            fields = _alternative_fields(
                alternative,
                queries,
                int(spec["voxels_per_field"]),
                rng,
            )
            table = build_table_from_fields(
                fields, int(rng.integers(0, np.iinfo(np.int64).max))
            )
            observed = observed_statistics(table)
            labels = rng.integers(
                0, FIELD_COUNT, size=(relabelings, queries), dtype=np.int8
            )
            null = assignment_statistics(table, labels)
            p = conditional_p_values(observed, null)
            rank_p[replication] = p["rank_tv"]
            coverage_p[replication] = p["coverage_deviation"]
        rank_detected = rank_p <= PER_TEST_ALPHA
        coverage_detected = coverage_p <= PER_TEST_ALPHA
        union_detected = rank_detected | coverage_detected
        union_row = probability_row(union_detected)
        sufficient = bool(
            union_row["probability"] >= 0.8
            and union_row["Wilson_95"][0] >= 0.7
        )
        result[alternative] = {
            "rank_detection": probability_row(rank_detected),
            "coverage_detection": probability_row(coverage_detected),
            "union_detection": union_row,
            "rank_p_quantiles_2p5_50_97p5": np.quantile(
                rank_p, [0.025, 0.5, 0.975]
            ).tolist(),
            "coverage_p_quantiles_2p5_50_97p5": np.quantile(
                coverage_p, [0.025, 0.5, 0.975]
            ).tolist(),
            "power_sufficient": sufficient,
        }
        arrays[f"power__{alternative}__rank_p"] = rank_p
        arrays[f"power__{alternative}__coverage_p"] = coverage_p
        print(f"[v76-power] {alternative}", flush=True)
    return {
        "alternatives": result,
        "power_sufficient": all(
            result[alternative]["power_sufficient"]
            for alternative in spec["alternatives"]
        ),
    }, arrays


def consumed_v75_diagnostic(v75_result: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    masking_cases = []
    source = v75_result["consumed_V72_diagnostic"]
    for arm in ("candidate", "control"):
        output[arm] = {}
        for domain in DOMAIN_ORDER:
            old = source[arm][domain]
            p = old["conditional_p_values"]
            corrected_pass = separate_pass(p["rank_tv"], p["coverage_deviation"])
            scalar_pass = bool(p["composite"] > 1.0 / 60.0)
            masked = bool(scalar_pass and not corrected_pass)
            output[arm][domain] = {
                "rank_p": p["rank_tv"],
                "coverage_p": p["coverage_deviation"],
                "V75_scalar_composite_p": p["composite"],
                "V75_scalar_pass": scalar_pass,
                "V76_separate_pass": corrected_pass,
                "scalar_masking_detected": masked,
            }
            if masked:
                masking_cases.append(f"{arm}/{domain}")
    output["scalar_masking_cases"] = masking_cases
    output["V72_verdict_changed"] = False
    output["selection_role"] = False
    return output


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial, path)


def run(program_path: Path, repo: Path, output_root: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, bindings = load_program(program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V76 requires a clean frozen Lageunha checkout")
    paths = {
        key: Path(program["outputs"][key]).resolve()
        for key in ("root", "power_arrays", "audit_result")
    }
    if output_root.resolve() != paths["root"]:
        raise ValueError("V76 output root differs")
    if output_root.exists() or paths["power_arrays"].exists() or paths["audit_result"].exists():
        raise FileExistsError("V76 refuses an existing output")
    output_root.mkdir(parents=True)
    calibration, calibration_p_arrays = null_calibration(
        program, bindings["V75_synthetic_arrays"]
    )
    nonmasking = nonmasking_audit(program)
    power, power_arrays = power_audit(program)
    write_npz(paths["power_arrays"], {**calibration_p_arrays, **power_arrays})
    v75_result = strict_json(bindings["V75_audit_result"])
    diagnostic = consumed_v75_diagnostic(v75_result)
    selected = bool(
        calibration["individual_calibrated"]
        and calibration["family_diagnostic_calibrated"]
        and nonmasking["nonmasking_invariant_pass"]
        and power["power_sufficient"]
    )
    classification = (
        "separate_exact_label_rank_coverage_rule_selected"
        if selected
        else "rank_coverage_rule_requires_additional_redesign"
    )
    next_step = (
        "freeze_a_complete_prospective_32_query_gate_then_await_explicit_candidate_approval"
        if selected
        else "stop_before_a_complete_gate_or_new_candidate_and_report_failed_requirement"
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete_V76_separate_rank_coverage_nonmasking_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "audit_code_commit": commit,
        "worktree_clean": clean,
        "existing_null_calibration": calibration,
        "deterministic_nonmasking": nonmasking,
        "independent_synthetic_power": power,
        "consumed_V75_diagnostic": diagnostic,
        "decision": {
            "separate_rule_selected": selected,
            "classification": classification,
            "complete_prospective_gate_may_be_frozen": selected,
            "complete_gate_or_new_candidate_execution_authorized": False,
            "next": next_step,
        },
        "artifacts": {
            "power_arrays": str(paths["power_arrays"]),
            "power_arrays_sha256": sha256_file(paths["power_arrays"]),
        },
        "training_or_model_sampling_performed": False,
        "raw_fit_train_truth_accessed": False,
        "validation_or_fresh_payload_accessed": False,
        "V72_ensemble_or_array_accessed": False,
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
    result = run(args.program, args.repo, args.output_root)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
