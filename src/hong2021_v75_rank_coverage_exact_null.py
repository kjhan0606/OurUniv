#!/usr/bin/env python
"""Exact truth-label null audit for voxel rank and empirical coverage gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_residual_evaluate import finite_ensemble_interval_expectation
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor
from hong2021_v74_gate_redesign import wilson_interval


PROGRAM_SCHEMA = "hong2021-v75-rank-coverage-exact-label-null-audit-program-v1"
PROGRAM_STATUS = "frozen_before_V75_implementation_synthetic_draws_or_V72_payload_read"
PROGRAM_SHA256 = "6f0eaf9d06c151e429e7d378dc4f1c4460d1ea170214367fce0262ca559a13b6"
PROGRAM_FREEZE_COMMIT = "218b516d25616f5004de4ded066f893142740164"
RESULT_SCHEMA = "hong2021-v75-rank-coverage-exact-label-null-audit-result-v1"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
FIELD_COUNT = 17
MEMBERS = 16
EXPECTED_68 = finite_ensemble_interval_expectation(MEMBERS, 0.16, 0.84)
EXPECTED_95 = finite_ensemble_interval_expectation(MEMBERS, 0.025, 0.975)
DOMAIN_ALPHA = 1.0 / 60.0
QUANTILE_SPEC = {
    "lower68": (2, 3, 0.4),
    "upper68": (12, 13, 0.6),
    "lower95": (0, 1, 0.375),
    "upper95": (14, 15, 0.625),
}


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
        raise ValueError("V75 program hash differs")
    program = strict_json(path)
    limits = program.get("scope_limits", {})
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or limits.get("this_is_not_a_generator") is not True
        or limits.get("validation_or_fresh_partition_access") is not False
        or limits.get("V72_verdict_change") is not False
    ):
        raise ValueError("V75 schema or firewall differs")
    parents = program["parent_evidence"]
    for key in ("V74_result_record", "ensemble_evaluator", "field_gate", "V72_gate"):
        if sha256_file(resolve_path(repo, parents[key])) != parents[f"{key}_sha256"]:
            raise ValueError(f"V75 parent differs: {key}")
    for arm in ("candidate", "control"):
        for domain in DOMAIN_ORDER:
            row = program["consumed_V72_diagnostic"][arm][domain]
            if sha256_file(Path(row["path"])) != row["sha256"]:
                raise ValueError(f"V75 consumed V72 artifact differs: {arm}/{domain}")
    return program


def derived_seed(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _remaining_value(
    sorted_values: np.ndarray, removed_rank: np.ndarray, remaining_index: int
) -> np.ndarray:
    voxel = np.arange(sorted_values.shape[1])
    source_index = remaining_index + (remaining_index >= removed_rank)
    return sorted_values[source_index, voxel]


def query_label_table(
    fields: np.ndarray, priority_rng: np.random.Generator
) -> dict[str, np.ndarray | float | int]:
    """Measure all seventeen possible truth designations for one query."""
    values = np.asarray(fields, dtype=np.float64).reshape(FIELD_COUNT, -1)
    if not np.isfinite(values).all() or values.shape[1] == 0:
        raise ValueError("V75 field table input differs")
    priority = priority_rng.random(values.shape)
    order = np.lexsort((priority, values), axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks,
        order,
        np.arange(FIELD_COUNT, dtype=np.int64)[:, None],
        axis=0,
    )
    histogram = np.zeros((FIELD_COUNT, FIELD_COUNT), dtype=np.int64)
    coverage68 = np.zeros(FIELD_COUNT, dtype=np.int64)
    coverage95 = np.zeros(FIELD_COUNT, dtype=np.int64)
    for label in range(FIELD_COUNT):
        removed_rank = ranks[label]
        histogram[label] = np.bincount(
            removed_rank, minlength=FIELD_COUNT
        )
        quantiles = {}
        for name, (lower_index, upper_index, fraction) in QUANTILE_SPEC.items():
            lower = _remaining_value(sorted_values, removed_rank, lower_index)
            upper = _remaining_value(sorted_values, removed_rank, upper_index)
            quantiles[name] = (1.0 - fraction) * lower + fraction * upper
        truth = values[label]
        coverage68[label] = np.count_nonzero(
            (truth >= quantiles["lower68"]) & (truth <= quantiles["upper68"])
        )
        coverage95[label] = np.count_nonzero(
            (truth >= quantiles["lower95"]) & (truth <= quantiles["upper95"])
        )
    tie_pairs = int(np.count_nonzero(sorted_values[1:] == sorted_values[:-1]))
    return {
        "histogram": histogram,
        "coverage68": coverage68,
        "coverage95": coverage95,
        "voxels": int(values.shape[1]),
        "adjacent_tie_pairs": tie_pairs,
        "adjacent_tie_fraction": float(
            tie_pairs / ((FIELD_COUNT - 1) * values.shape[1])
        ),
    }


def stack_query_tables(tables: list[dict[str, Any]]) -> dict[str, np.ndarray | int | float]:
    voxels = {int(table["voxels"]) for table in tables}
    if len(voxels) != 1:
        raise ValueError("V75 query voxel counts differ")
    return {
        "histogram": np.stack([table["histogram"] for table in tables]),
        "coverage68": np.stack([table["coverage68"] for table in tables]),
        "coverage95": np.stack([table["coverage95"] for table in tables]),
        "voxels": voxels.pop(),
        "adjacent_tie_pairs": int(sum(table["adjacent_tie_pairs"] for table in tables)),
        "adjacent_tie_fraction": float(
            np.mean([table["adjacent_tie_fraction"] for table in tables])
        ),
    }


def assignment_statistics(
    table: dict[str, Any], assignments: np.ndarray, chunk: int = 5000
) -> dict[str, np.ndarray]:
    labels = np.asarray(assignments, dtype=np.int64)
    queries = table["histogram"].shape[0]
    if labels.ndim != 2 or labels.shape[1] != queries or np.any(
        (labels < 0) | (labels >= FIELD_COUNT)
    ):
        raise ValueError("V75 label assignment differs")
    output = {
        "rank_tv": np.empty(len(labels), dtype=np.float64),
        "coverage68": np.empty(len(labels), dtype=np.float64),
        "coverage95": np.empty(len(labels), dtype=np.float64),
        "coverage_deviation": np.empty(len(labels), dtype=np.float64),
        "composite": np.empty(len(labels), dtype=np.float64),
    }
    query_index = np.arange(queries)[None, :]
    total_voxels = queries * int(table["voxels"])
    expected_rank = total_voxels / FIELD_COUNT
    for start in range(0, len(labels), chunk):
        stop = min(start + chunk, len(labels))
        current = labels[start:stop]
        histogram = table["histogram"][query_index, current].sum(axis=1)
        rank_tv = (
            0.5 * np.abs(histogram - expected_rank).sum(axis=1) / total_voxels
        )
        coverage68 = (
            table["coverage68"][query_index, current].sum(axis=1) / total_voxels
        )
        coverage95 = (
            table["coverage95"][query_index, current].sum(axis=1) / total_voxels
        )
        coverage_deviation = np.maximum(
            np.abs(coverage68 - EXPECTED_68),
            np.abs(coverage95 - EXPECTED_95),
        )
        output["rank_tv"][start:stop] = rank_tv
        output["coverage68"][start:stop] = coverage68
        output["coverage95"][start:stop] = coverage95
        output["coverage_deviation"][start:stop] = coverage_deviation
        output["composite"][start:stop] = np.maximum(
            rank_tv / 0.05, coverage_deviation / 0.03
        )
    return output


def observed_statistics(table: dict[str, Any], label: int = 16) -> dict[str, float]:
    assignment = np.full((1, table["histogram"].shape[0]), label, dtype=np.int8)
    return {
        key: float(value[0])
        for key, value in assignment_statistics(table, assignment).items()
    }


def conditional_p_values(
    observed: dict[str, float], null: dict[str, np.ndarray]
) -> dict[str, float]:
    keys = ("rank_tv", "coverage_deviation", "composite")
    return {
        key: float((1 + np.count_nonzero(null[key] >= observed[key])) / (len(null[key]) + 1))
        for key in keys
    }


def probability_row(values: np.ndarray) -> dict[str, Any]:
    boolean = np.asarray(values, dtype=bool).reshape(-1)
    interval = wilson_interval(int(boolean.sum()), len(boolean))
    return {
        "successes": int(boolean.sum()),
        "trials": int(len(boolean)),
        "probability": float(boolean.mean()),
        "Wilson_95": list(interval),
    }


def _synthetic_fields(
    scenario: str,
    queries: int,
    voxels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if scenario == "independent_normal_voxels":
        return rng.normal(size=(queries, FIELD_COUNT, voxels))
    if scenario == "spatially_correlated_normal":
        global_value = rng.normal(size=(queries, FIELD_COUNT, 1))
        innovation = rng.normal(size=(queries, FIELD_COUNT, voxels))
        return 0.9 * global_value + np.sqrt(1.0 - 0.9**2) * innovation
    if scenario == "perfect_within_field_dependence":
        value = rng.normal(size=(queries, FIELD_COUNT, 1))
        return np.broadcast_to(value, (queries, FIELD_COUNT, voxels)).copy()
    if scenario == "rounded_heavy_tail_with_ties":
        global_value = rng.standard_t(3, size=(queries, FIELD_COUNT, 1))
        innovation = rng.standard_t(3, size=(queries, FIELD_COUNT, voxels))
        value = 0.85 * global_value + np.sqrt(1.0 - 0.85**2) * innovation
        return np.round(value, 1)
    raise ValueError("unknown V75 synthetic scenario")


def build_table_from_fields(fields: np.ndarray, priority_seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(priority_seed)
    return stack_query_tables(
        [query_label_table(fields[index], rng) for index in range(fields.shape[0])]
    )


def tail_p_from_reference(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_reference = np.sort(np.asarray(reference, dtype=np.float64))
    current = np.asarray(values, dtype=np.float64)
    lower = np.searchsorted(sorted_reference, current, side="left")
    exceed = len(sorted_reference) - lower
    return (1.0 + exceed) / (len(sorted_reference) + 1.0)


def synthetic_null_audit(program: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    spec = program["synthetic_exact_null_audit"]
    scenarios = list(spec["scenarios"])
    field_rng = np.random.default_rng(int(spec["field_seed"]))
    calibration_rng = np.random.default_rng(int(spec["calibration_label_seed"]))
    verification_rng = np.random.default_rng(int(spec["verification_label_seed"]))
    assignments = int(spec["label_assignments_per_phase_and_scenario"])
    query_count = int(spec["queries"])
    result: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for scenario_index, scenario in enumerate(scenarios):
        fields = _synthetic_fields(
            scenario,
            query_count,
            int(spec["voxels_per_field"]),
            field_rng,
        )
        table = build_table_from_fields(fields, 751000 + scenario_index)
        calibration_labels = calibration_rng.integers(
            0, FIELD_COUNT, size=(assignments, query_count), dtype=np.int8
        )
        verification_labels = verification_rng.integers(
            0, FIELD_COUNT, size=(assignments, query_count), dtype=np.int8
        )
        calibration = assignment_statistics(table, calibration_labels)
        verification = assignment_statistics(table, verification_labels)
        composite_p = tail_p_from_reference(
            calibration["composite"], verification["composite"]
        )
        rejected = composite_p <= DOMAIN_ALPHA
        old_rank = verification["rank_tv"] <= 0.05
        old_coverage = verification["coverage_deviation"] <= 0.03
        result[scenario] = {
            "adjacent_tie_fraction": table["adjacent_tie_fraction"],
            "old_hard_rank_pass": probability_row(old_rank),
            "old_hard_coverage_pass": probability_row(old_coverage),
            "old_hard_joint_pass": probability_row(old_rank & old_coverage),
            "replacement_rejection": probability_row(rejected),
            "replacement_p_value_quantiles_2p5_50_97p5": np.quantile(
                composite_p, [0.025, 0.5, 0.975]
            ).tolist(),
            "rank_tv_quantiles_2p5_50_97p5": np.quantile(
                verification["rank_tv"], [0.025, 0.5, 0.975]
            ).tolist(),
            "coverage_deviation_quantiles_2p5_50_97p5": np.quantile(
                verification["coverage_deviation"], [0.025, 0.5, 0.975]
            ).tolist(),
        }
        for key, value in calibration.items():
            arrays[f"null__{scenario}__calibration__{key}"] = value
        for key, value in verification.items():
            arrays[f"null__{scenario}__verification__{key}"] = value
        arrays[f"null__{scenario}__verification__composite_p"] = composite_p
        print(f"[v75-null] {scenario}", flush=True)
    old_invalid = any(
        result[scenario][key]["probability"] < 0.8
        for scenario in scenarios
        for key in ("old_hard_rank_pass", "old_hard_coverage_pass")
    )
    replacement_calibrated = all(
        result[scenario]["replacement_rejection"]["probability"] <= 0.02
        and result[scenario]["replacement_rejection"]["Wilson_95"][1] <= 0.022
        for scenario in scenarios
    )
    return {
        "scenarios": result,
        "old_hard_gate_invalid": old_invalid,
        "replacement_calibrated": replacement_calibrated,
    }, arrays


def coverage_counterexample(program: dict[str, Any]) -> dict[str, Any]:
    spec = program["synthetic_exact_null_audit"]["coverage_distribution_counterexample"]
    rng = np.random.default_rng(int(spec["seed"]))
    total = int(spec["sets_per_distribution"])
    output = {}
    for distribution in spec["distributions"]:
        hit68 = hit95 = 0
        for start in range(0, total, 25000):
            number = min(25000, total - start)
            if distribution == "uniform":
                values = rng.random((number, FIELD_COUNT))
            elif distribution == "normal":
                values = rng.normal(size=(number, FIELD_COUNT))
            elif distribution == "lognormal":
                values = rng.lognormal(size=(number, FIELD_COUNT))
            elif distribution == "Student-t-3":
                values = rng.standard_t(3, size=(number, FIELD_COUNT))
            else:
                raise ValueError("unknown V75 coverage distribution")
            ensemble = values[:, :MEMBERS]
            truth = values[:, MEMBERS]
            lower68, upper68 = np.quantile(ensemble, [0.16, 0.84], axis=1)
            lower95, upper95 = np.quantile(ensemble, [0.025, 0.975], axis=1)
            hit68 += int(np.count_nonzero((truth >= lower68) & (truth <= upper68)))
            hit95 += int(np.count_nonzero((truth >= lower95) & (truth <= upper95)))
        coverage68 = hit68 / total
        coverage95 = hit95 / total
        output[distribution] = {
            "coverage68": coverage68,
            "old_expected68": EXPECTED_68,
            "difference68": coverage68 - EXPECTED_68,
            "coverage95": coverage95,
            "old_expected95": EXPECTED_95,
            "difference95": coverage95 - EXPECTED_95,
        }
    return output


def _alternative_fields(
    alternative: str,
    queries: int,
    voxels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    global_value = rng.normal(size=(queries, FIELD_COUNT, 1))
    innovation = rng.normal(size=(queries, FIELD_COUNT, voxels))
    base = 0.7 * global_value + np.sqrt(1.0 - 0.7**2) * innovation
    if alternative == "location_bias":
        base[:, :MEMBERS] += 0.5
    elif alternative == "underdispersion":
        base[:, :MEMBERS] *= 0.6
    else:
        raise ValueError("unknown V75 power alternative")
    return base


def synthetic_power_audit(program: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    spec = program["synthetic_power_audit"]
    rng = np.random.default_rng(int(spec["seed"]))
    query_count = int(spec["queries"])
    relabelings = int(spec["random_relabelings_per_replication"])
    replications = int(spec["replications_per_alternative"])
    result = {}
    arrays = {}
    for alternative in spec["alternatives"]:
        p_values = np.empty(replications, dtype=np.float64)
        observed_composite = np.empty(replications, dtype=np.float64)
        for replication in range(replications):
            fields = _alternative_fields(
                alternative,
                query_count,
                int(spec["voxels_per_field"]),
                rng,
            )
            table = build_table_from_fields(
                fields, int(rng.integers(0, np.iinfo(np.int64).max))
            )
            observed = observed_statistics(table)
            labels = rng.integers(
                0, FIELD_COUNT, size=(relabelings, query_count), dtype=np.int8
            )
            null = assignment_statistics(table, labels)
            p_values[replication] = conditional_p_values(observed, null)["composite"]
            observed_composite[replication] = observed["composite"]
        detected = p_values <= DOMAIN_ALPHA
        row = probability_row(detected)
        result[alternative] = {
            "detection": row,
            "p_value_quantiles_2p5_50_97p5": np.quantile(
                p_values, [0.025, 0.5, 0.975]
            ).tolist(),
            "observed_composite_quantiles_2p5_50_97p5": np.quantile(
                observed_composite, [0.025, 0.5, 0.975]
            ).tolist(),
        }
        arrays[f"power__{alternative}__p_value"] = p_values
        arrays[f"power__{alternative}__observed_composite"] = observed_composite
        print(f"[v75-power] {alternative}", flush=True)
    sufficient = all(
        result[alternative]["detection"]["probability"] >= float(spec["minimum_power"])
        and result[alternative]["detection"]["Wilson_95"][0] >= 0.7
        for alternative in spec["alternatives"]
    )
    return {"alternatives": result, "replacement_power_sufficient": sufficient}, arrays


def v72_table(path: Path, priority_seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(priority_seed)
    tables = []
    with h5py.File(path, "r") as handle:
        if (
            tuple(handle["sample"].shape) != (16, 16, 1, 64, 64, 64)
            or tuple(handle["truth"].shape) != (16, 1, 64, 64, 64)
            or tuple(handle["conditional_mean"].shape) != (16, 1, 64, 64, 64)
        ):
            raise ValueError("V75 V72 ensemble shape differs")
        for query in range(16):
            mean = np.asarray(handle["conditional_mean"][query, 0], dtype=np.float64)
            generated = np.asarray(handle["sample"][query, :, 0], dtype=np.float64) - mean
            truth = np.asarray(handle["truth"][query, 0], dtype=np.float64) - mean
            generated -= generated.mean(axis=(-3, -2, -1), keepdims=True)
            truth -= truth.mean()
            fields = np.concatenate((generated, truth[None]), axis=0)
            tables.append(query_label_table(fields, rng))
            print(f"[v75-V72-table] {path.parent.parent.parent.name} {query + 1}/16", flush=True)
    return stack_query_tables(tables)


def v72_diagnostic(
    program: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    spec = program["consumed_V72_diagnostic"]
    relabelings = int(program["prospective_exact_label_test"]["random_relabelings_per_domain"])
    result: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for arm in ("candidate", "control"):
        result[arm] = {}
        for domain in DOMAIN_ORDER:
            row = spec[arm][domain]
            tie_seed = derived_seed(PROGRAM_SHA256, row["sha256"], domain, "tie_priority")
            label_seed = derived_seed(PROGRAM_SHA256, row["sha256"], domain, "relabel")
            table = v72_table(Path(row["path"]), tie_seed)
            observed = observed_statistics(table)
            rng = np.random.default_rng(label_seed)
            labels = rng.integers(
                0,
                FIELD_COUNT,
                size=(relabelings, int(spec["queries_per_domain"])),
                dtype=np.int8,
            )
            null = assignment_statistics(table, labels)
            p_values = conditional_p_values(observed, null)
            result[arm][domain] = {
                "ensemble": row["path"],
                "ensemble_sha256": row["sha256"],
                "queries": int(spec["queries_per_domain"]),
                "adjacent_tie_fraction": table["adjacent_tie_fraction"],
                "observed": observed,
                "old_hard_rank_pass": observed["rank_tv"] <= 0.05,
                "old_hard_coverage_pass": observed["coverage_deviation"] <= 0.03,
                "conditional_p_values": p_values,
                "prospective_composite_pass": p_values["composite"] > DOMAIN_ALPHA,
                "domain_alpha": DOMAIN_ALPHA,
                "null_composite_quantiles_95_98p333_99": np.quantile(
                    null["composite"], [0.95, 1.0 - DOMAIN_ALPHA, 0.99]
                ).tolist(),
            }
            for key, value in null.items():
                arrays[f"V72__{arm}__{domain}__null__{key}"] = value
            print(f"[v75-V72] {arm} {domain}", flush=True)
    return result, arrays


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
        raise RuntimeError("V75 requires a clean frozen Lageunha checkout")
    expected_root = Path(program["outputs"]["root"]).resolve()
    if output_root.resolve() != expected_root:
        raise ValueError("V75 output root differs")
    paths = {
        key: Path(program["outputs"][key])
        for key in ("synthetic_arrays", "V72_diagnostic_arrays", "audit_result")
    }
    if output_root.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError("V75 refuses an existing output")
    output_root.mkdir(parents=True)
    null, null_arrays = synthetic_null_audit(program)
    counterexample = coverage_counterexample(program)
    power, power_arrays = synthetic_power_audit(program)
    write_npz(paths["synthetic_arrays"], {**null_arrays, **power_arrays})
    diagnostic, diagnostic_arrays = v72_diagnostic(program)
    write_npz(paths["V72_diagnostic_arrays"], diagnostic_arrays)
    replacement_selected = bool(
        null["old_hard_gate_invalid"]
        and null["replacement_calibrated"]
        and power["replacement_power_sufficient"]
    )
    if replacement_selected:
        classification = "exact_label_randomization_rank_coverage_replacement_selected"
        next_step = "freeze_a_complete_prospective_32_query_gate_then_await_explicit_candidate_approval"
    else:
        classification = "rank_coverage_requires_additional_redesign"
        next_step = "stop_before_a_complete_gate_or_new_candidate"
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete_V75_rank_coverage_exact_label_null_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "audit_code_commit": commit,
        "worktree_clean": clean,
        "old_fixed_expectations": {
            "coverage68": EXPECTED_68,
            "coverage95": EXPECTED_95,
        },
        "synthetic_exact_null": null,
        "physical_coverage_distribution_counterexample": counterexample,
        "synthetic_power": power,
        "consumed_V72_diagnostic": diagnostic,
        "decision": {
            "old_hard_gate_invalid": null["old_hard_gate_invalid"],
            "exact_label_replacement_calibrated": null[
                "replacement_calibrated"
            ],
            "exact_label_replacement_power_sufficient": power[
                "replacement_power_sufficient"
            ],
            "replacement_selected": replacement_selected,
            "classification": classification,
            "complete_prospective_gate_may_be_frozen": replacement_selected,
            "complete_gate_or_new_candidate_execution_authorized": False,
            "next": next_step,
        },
        "artifacts": {
            "synthetic_arrays": str(paths["synthetic_arrays"].resolve()),
            "synthetic_arrays_sha256": sha256_file(paths["synthetic_arrays"]),
            "V72_diagnostic_arrays": str(paths["V72_diagnostic_arrays"].resolve()),
            "V72_diagnostic_arrays_sha256": sha256_file(
                paths["V72_diagnostic_arrays"]
            ),
        },
        "training_or_model_sampling_performed": False,
        "raw_fit_train_truth_accessed": False,
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
