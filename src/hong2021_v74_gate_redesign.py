#!/usr/bin/env python
"""Query-count and maximum-energy redesign audit for the V72/V73 gate."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np

import hong2021_v73_gate_attainability as v73
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor
from hong2021_v72_sqt import scalar_energy_score


PROGRAM_SCHEMA = "hong2021-v74-query-count-energy-gate-redesign-program-v1"
PROGRAM_STATUS = "frozen_before_V74_implementation_or_new_bootstrap_draws"
PROGRAM_SHA256 = "7b08cf433396b673430909dd8caa676da669e7ef61c8dab39d6ae9d0037850fb"
PROGRAM_FREEZE_COMMIT = "f1b07a27d630bdbde2db3cff09a56856d4eeae00"
RESULT_SCHEMA = "hong2021-v74-query-count-energy-gate-redesign-result-v1"
DOMAIN_ORDER = v73.DOMAIN_ORDER
QUERY_COUNTS = (16, 32)
V73_REFERENCE = {
    "TNG100": {"absolute_core": 0.8482, "joint": 0.7675},
    "SIMBA": {"absolute_core": 0.9989, "joint": 0.9981},
    "Swift": {"absolute_core": 0.9982, "joint": 0.9962},
    "all_domain_joint": 0.7633,
}
WILSON_Z = 1.959963984540054


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def resolve_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    path = path.resolve()
    repo = repo.resolve()
    if sha256_file(path) != PROGRAM_SHA256:
        raise ValueError("V74 program hash differs")
    program = strict_json(path)
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or program.get("scope_limits", {}).get("this_is_not_a_generator") is not True
        or program.get("scope_limits", {}).get("validation_or_fresh_partition_access")
        is not False
        or program.get("scope_limits", {}).get("physical_threshold_relaxation")
        is not False
    ):
        raise ValueError("V74 schema or firewall differs")
    parents = program["parent_evidence"]
    for key in ("V73_result_record", "V73_program", "V73_source"):
        if sha256_file(resolve_path(repo, parents[key])) != parents[f"{key}_sha256"]:
            raise ValueError(f"V74 local parent differs: {key}")
    for key in ("V73_audit_result", "V73_summary_record", "V73_summary_cache"):
        if sha256_file(Path(parents[key])) != parents[f"{key}_sha256"]:
            raise ValueError(f"V74 GPFS parent differs: {key}")
    result = strict_json(Path(parents["V73_audit_result"]))
    digest = result.pop("decision_digest_sha256")
    if digest != parents["V73_decision_digest_sha256"] or canonical_digest(result) != digest:
        raise ValueError("V74 V73 decision digest differs")
    summary_record = strict_json(Path(parents["V73_summary_record"]))
    if (
        summary_record.get("program_sha256") != parents["V73_program_sha256"]
        or summary_record.get("summary_cache_sha256")
        != parents["V73_summary_cache_sha256"]
        or summary_record.get("validation_or_fresh_payload_accessed") is not False
    ):
        raise ValueError("V74 V73 summary provenance differs")
    return program


def wilson_interval(
    successes: int | np.integer, trials: int, z: float = WILSON_Z
) -> tuple[float, float]:
    if trials <= 0 or not 0 <= int(successes) <= trials or z <= 0:
        raise ValueError("invalid Wilson inputs")
    probability = float(successes) / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half = (
        z
        * np.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return float(center - half), float(center + half)


def probability_row(values: np.ndarray) -> dict[str, Any]:
    boolean = np.asarray(values, dtype=bool).reshape(-1)
    successes = int(boolean.sum())
    interval = wilson_interval(successes, len(boolean))
    return {
        "successes": successes,
        "trials": int(len(boolean)),
        "probability": float(boolean.mean()),
        "Wilson_95": list(interval),
    }


def sample_queries_count(
    domain: str,
    groups: np.ndarray,
    query_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    quota: dict[int, int]
    if domain == "TNG100":
        if len(unique) != 4 or query_count not in QUERY_COUNTS:
            raise ValueError("V74 TNG query design differs")
        quota = {int(group): query_count // 4 for group in unique}
    elif domain == "SIMBA":
        if len(unique) != 8 or query_count not in QUERY_COUNTS:
            raise ValueError("V74 SIMBA query design differs")
        quota = {int(group): query_count // 8 for group in unique}
    elif domain == "Swift":
        if len(unique) != 20 or query_count not in QUERY_COUNTS:
            raise ValueError("V74 Swift query design differs")
        if query_count == 16:
            chosen = set(map(int, rng.choice(unique, size=16, replace=False)))
            quota = {int(group): 1 for group in unique if int(group) in chosen}
        else:
            doubled = set(map(int, rng.choice(unique, size=12, replace=False)))
            quota = {
                int(group): (2 if int(group) in doubled else 1) for group in unique
            }
    else:
        raise ValueError("unknown V74 domain")
    selected: list[int] = []
    for group, number in quota.items():
        pool = np.flatnonzero(groups == group)
        if len(pool) < number:
            raise ValueError("V74 query group is too small")
        selected.extend(rng.choice(pool, size=number, replace=False).tolist())
    output = np.asarray(selected, dtype=np.int64)
    if len(output) != query_count or len(np.unique(output)) != query_count:
        raise ValueError("V74 query count differs")
    return output


def energy_delta(
    summary: dict[str, np.ndarray],
    queries: np.ndarray,
    oracle_A: np.ndarray,
    oracle_B: np.ndarray,
) -> float:
    truth = summary["truth_max"][queries]
    first = np.mean(
        [
            scalar_energy_score(summary["truth_max"][oracle_A[row]], truth[row])
            for row in range(len(queries))
        ]
    )
    second = np.mean(
        [
            scalar_energy_score(summary["truth_max"][oracle_B[row]], truth[row])
            for row in range(len(queries))
        ]
    )
    return float(first - second)


def rows_to_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def run_phase(
    summaries: dict[str, dict[str, np.ndarray]],
    k: np.ndarray,
    count: np.ndarray,
    radius: np.ndarray,
    trials: int,
    seed: int,
    phase: str,
) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    output: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    for query_count in QUERY_COUNTS:
        output[query_count] = {}
        for domain in DOMAIN_ORDER:
            summary = summaries[domain]
            rows: list[dict[str, Any]] = []
            for trial in range(trials):
                queries = sample_queries_count(
                    domain, summary["group"], query_count, rng
                )
                oracle_A = v73.sample_same_group_oracle(
                    summary["group"], queries, rng
                )
                oracle_B = v73.sample_same_group_oracle(
                    summary["group"], queries, rng
                )
                row = v73.trial_metrics(
                    summary,
                    queries,
                    summary,
                    oracle_A,
                    k,
                    count,
                    radius,
                )
                row["energy_A_minus_B"] = energy_delta(
                    summary, queries, oracle_A, oracle_B
                )
                rows.append(row)
                if (trial + 1) % 2000 == 0:
                    print(
                        f"[v74-{phase}] q{query_count} {domain} "
                        f"{trial + 1}/{trials}",
                        flush=True,
                    )
            output[query_count][domain] = rows_to_arrays(rows)
    return output


def flatten_phase(
    phase: dict[int, dict[str, dict[str, np.ndarray]]]
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for query_count, domains in phase.items():
        for domain, values in domains.items():
            for key, value in values.items():
                arrays[f"q{query_count}__{domain}__{key}"] = value
    return arrays


def phase_summary(
    phase: dict[int, dict[str, dict[str, np.ndarray]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for query_count, domains in phase.items():
        domain_result = {}
        for domain, values in domains.items():
            domain_result[domain] = {
                key: probability_row(values[key])
                for key in (
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
            }
        aligned = np.logical_and.reduce(
            [domains[domain]["joint"] for domain in DOMAIN_ORDER]
        )
        result[str(query_count)] = {
            "domains": domain_result,
            "all_three_domain_aligned_joint": probability_row(aligned),
        }
    return result


def calibration_margins(
    calibration: dict[int, dict[str, dict[str, np.ndarray]]], quantile: float
) -> dict[int, dict[str, float]]:
    return {
        query_count: {
            domain: float(
                np.quantile(
                    domains[domain]["energy_A_minus_B"], quantile, method="linear"
                )
            )
            for domain in DOMAIN_ORDER
        }
        for query_count, domains in calibration.items()
    }


def energy_verification(
    verification: dict[int, dict[str, dict[str, np.ndarray]]],
    margins: dict[int, dict[str, float]],
) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for query_count, domains in verification.items():
        domain_rows = {}
        pass_rows = []
        for domain in DOMAIN_ORDER:
            delta = domains[domain]["energy_A_minus_B"]
            passed = delta <= margins[query_count][domain]
            pass_rows.append(passed)
            domain_rows[domain] = {
                "upper_margin": margins[query_count][domain],
                "pass": probability_row(passed),
                "false_rejection": probability_row(~passed),
                "delta_quantiles_2p5_50_97p5": np.quantile(
                    delta, [0.025, 0.5, 0.975]
                ).tolist(),
            }
        family_pass = np.logical_and.reduce(pass_rows)
        output[query_count] = {
            "domains": domain_rows,
            "all_three_domain_aligned_pass": probability_row(family_pass),
            "family_wise_false_rejection": probability_row(~family_pass),
        }
    return output


def v73_reproduction(
    verification: dict[int, dict[str, dict[str, np.ndarray]]]
) -> dict[str, Any]:
    domains = verification[16]
    rows: dict[str, Any] = {}
    differences = []
    for domain in DOMAIN_ORDER:
        rows[domain] = {}
        for key in ("absolute_core", "joint"):
            observed = float(domains[domain][key].mean())
            reference = float(V73_REFERENCE[domain][key])
            difference = observed - reference
            differences.append(abs(difference))
            rows[domain][key] = {
                "V74_verification": observed,
                "V73_reference": reference,
                "difference": difference,
            }
    all_joint = float(
        np.logical_and.reduce([domains[domain]["joint"] for domain in DOMAIN_ORDER]).mean()
    )
    difference = all_joint - float(V73_REFERENCE["all_domain_joint"])
    differences.append(abs(difference))
    return {
        "domains": rows,
        "all_domain_joint": {
            "V74_verification": all_joint,
            "V73_reference": V73_REFERENCE["all_domain_joint"],
            "difference": difference,
        },
        "maximum_absolute_probability_difference": max(differences),
        "tolerance": 0.02,
        "pass": max(differences) <= 0.02,
    }


def select_query_count(
    verification: dict[int, dict[str, dict[str, np.ndarray]]],
    reproduction_pass: bool,
) -> tuple[int | None, dict[str, Any]]:
    rows: dict[str, Any] = {}
    selected: int | None = None
    for query_count in QUERY_COUNTS:
        domains = verification[query_count]
        aligned = np.logical_and.reduce(
            [domains[domain]["joint"] for domain in DOMAIN_ORDER]
        )
        all_joint = probability_row(aligned)
        absolute = {
            domain: probability_row(domains[domain]["absolute_core"])
            for domain in DOMAIN_ORDER
        }
        attainable = bool(
            all_joint["Wilson_95"][0] >= 0.8
            and all(row["Wilson_95"][0] >= 0.8 for row in absolute.values())
        )
        rows[str(query_count)] = {
            "all_three_domain_aligned_joint": all_joint,
            "domain_absolute_core": absolute,
            "attainable": attainable,
        }
        if selected is None and attainable:
            selected = query_count
    if not reproduction_pass:
        selected = None
    return selected, rows


def energy_calibrated(row: dict[str, Any]) -> bool:
    false_rejection = row["family_wise_false_rejection"]
    return bool(
        false_rejection["probability"] <= 0.06
        and false_rejection["Wilson_95"][1] <= 0.065
    )


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial, path)


def run(program_path: Path, repo: Path, output_root: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program = load_program(program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V74 requires a clean frozen Lageunha checkout")
    expected_root = Path(program["outputs"]["root"]).resolve()
    if output_root.resolve() != expected_root:
        raise ValueError("V74 output root differs")
    paths = {
        key: Path(program["outputs"][key])
        for key in ("calibration_arrays", "verification_arrays", "audit_result")
    }
    if output_root.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError("V74 refuses an existing output")
    output_root.mkdir(parents=True)
    with np.load(
        Path(program["parent_evidence"]["V73_summary_cache"]), allow_pickle=False
    ) as cache:
        summaries = {domain: v73._domain_summary(cache, domain) for domain in DOMAIN_ORDER}
        k = np.asarray(cache["fourier_k"], dtype=np.float64)
        count = np.asarray(cache["fourier_mode_count"], dtype=np.int64)
        radius = np.asarray(cache["radius_mpc_h"], dtype=np.float64)
    monte_carlo = program["independent_monte_carlo_phases"]
    calibration_spec = monte_carlo["calibration"]
    verification_spec = monte_carlo["verification"]
    calibration = run_phase(
        summaries,
        k,
        count,
        radius,
        int(calibration_spec["trials_per_query_count_and_domain"]),
        int(calibration_spec["seed"]),
        "calibration",
    )
    margins = calibration_margins(
        calibration, float(program["energy_redesign"]["calibration_quantile"])
    )
    write_npz(paths["calibration_arrays"], flatten_phase(calibration))
    verification = run_phase(
        summaries,
        k,
        count,
        radius,
        int(verification_spec["trials_per_query_count_and_domain"]),
        int(verification_spec["seed"]),
        "verification",
    )
    write_npz(paths["verification_arrays"], flatten_phase(verification))
    reproduction = v73_reproduction(verification)
    selected_query_count, query_decision = select_query_count(
        verification, bool(reproduction["pass"])
    )
    energy = energy_verification(verification, margins)
    selected_energy_calibrated = bool(
        selected_query_count is not None
        and energy_calibrated(energy[selected_query_count])
    )
    v73_result = strict_json(Path(program["parent_evidence"]["V73_audit_result"]))
    v72_energy = {
        domain: {
            "mean_candidate_minus_control": v73_result[
                "consumed_V72_energy_stability"
            ][domain]["mean_candidate_minus_control"],
            "V74_q16_upper_margin": margins[16][domain],
            "prospective_no_detectable_inferiority_pass": bool(
                v73_result["consumed_V72_energy_stability"][domain][
                    "mean_candidate_minus_control"
                ]
                <= margins[16][domain]
            ),
        }
        for domain in DOMAIN_ORDER
    }
    if selected_query_count is None:
        classification = "additional_metric_level_gate_redesign_required"
        next_step = "stop_before_a_complete_gate_or_new_candidate"
    elif not selected_energy_calibrated:
        classification = "query_count_sufficient_but_energy_redesign_not_calibrated"
        next_step = "redesign_energy_before_a_complete_gate_or_new_candidate"
    else:
        classification = "query_count_and_energy_redesign_sufficient_full_gate_incomplete"
        next_step = "audit_rank_and_coverage_exact_conditional_null_before_freezing_a_complete_gate"
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete_V74_query_count_and_energy_gate_redesign_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "audit_code_commit": commit,
        "worktree_clean": clean,
        "calibration": phase_summary(calibration),
        "verification": phase_summary(verification),
        "V73_sixteen_query_reproduction": reproduction,
        "query_count_decision": {
            "rows": query_decision,
            "selected_lowest_attainable_query_count": selected_query_count,
            "unchanged_physical_and_morphology_thresholds": True,
            "query_count_redesign_sufficient": selected_query_count is not None,
        },
        "energy_redesign": {
            "calibration_quantile": program["energy_redesign"][
                "calibration_quantile"
            ],
            "upper_margins": {
                str(query_count): margins[query_count]
                for query_count in QUERY_COUNTS
            },
            "verification": {
                str(query_count): energy[query_count]
                for query_count in QUERY_COUNTS
            },
            "selected_query_count_energy_calibrated": selected_energy_calibrated,
            "V72_diagnostic_under_q16_margin": v72_energy,
            "V72_verdict_changed": False,
        },
        "decision": {
            "classification": classification,
            "selected_query_count": selected_query_count,
            "energy_redesign_calibrated": selected_energy_calibrated,
            "full_prospective_gate_complete": False,
            "rank_and_coverage_exact_conditional_null_audit_required": True,
            "next": next_step,
        },
        "artifacts": {
            "calibration_arrays": str(paths["calibration_arrays"].resolve()),
            "calibration_arrays_sha256": sha256_file(paths["calibration_arrays"]),
            "verification_arrays": str(paths["verification_arrays"].resolve()),
            "verification_arrays_sha256": sha256_file(paths["verification_arrays"]),
        },
        "training_or_model_sampling_performed": False,
        "raw_train_truth_reread": False,
        "validation_or_fresh_payload_accessed": False,
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
