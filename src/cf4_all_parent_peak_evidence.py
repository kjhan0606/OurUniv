#!/usr/bin/env python3
"""All-256-parent evidence-only feasibility audit for the LG peak model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import numpy as np

from cf4_lg_peak_cr import (
    draw_protohalo_midpoint_offset,
    two_peak_points,
)
from cf4_peak_evidence_phase_cache import (
    covariance_for_point_sets,
    full_spectrum_from_rfft,
    parent_mean_at_point_sets,
)


ROOT = Path(__file__).resolve().parents[1]
_WORKER_FILTER: np.ndarray | None = None
_WORKER_POINT_SETS: list[np.ndarray] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values))
    return maximum + math.log(float(np.exp(values - maximum).sum()))


def normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    log_weights = np.asarray(log_weights, dtype=np.float64)
    weights = np.exp(log_weights - np.max(log_weights))
    return weights / np.sum(weights)


def effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    return float(1.0 / np.sum(weights ** 2))


def joint_importance_diagnostics(
    log_importance: np.ndarray,
) -> dict[str, Any]:
    """Measure support over the complete parent-by-geometry proposal bank."""
    values = np.asarray(log_importance, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("joint log importance must be a finite matrix")
    joint_weights = normalized_weights(values.ravel())
    geometry_ess = np.asarray([
        effective_sample_size(normalized_weights(row)) for row in values
    ])
    return {
        "joint_parent_geometry_ESS": effective_sample_size(joint_weights),
        "maximum_joint_parent_geometry_weight": float(np.max(joint_weights)),
        "parent_geometry_ESS": geometry_ess,
    }


def weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or values.ndim != 1:
        raise ValueError("values and weights must be aligned vectors")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0,1]")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order]) / np.sum(weights)
    index = min(int(np.searchsorted(cumulative, probability, side="left")), len(values) - 1)
    return float(sorted_values[index])


def one_sided_weighted_ks_statistic(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return sup_d(F_uniform(d) - F_weighted(d)), grouping tied values."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape != weights.shape or values.ndim != 1 or len(values) == 0:
        raise ValueError("values and weights must be aligned nonempty vectors")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("values and weights must be finite")
    if np.any(weights < 0.0) or not float(np.sum(weights)) > 0.0:
        raise ValueError("weights must be nonnegative with positive mass")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    endpoints = np.flatnonzero(np.r_[sorted_values[1:] != sorted_values[:-1], True])
    uniform_cdf = (endpoints + 1) / len(values)
    weighted_cdf = np.cumsum(weights[order])[endpoints] / np.sum(weights)
    return float(max(0.0, np.max(uniform_cdf - weighted_cdf)))


def weighted_ks_permutation_test(
    values: np.ndarray,
    weights: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Calibrate the one-sided CF4 deviance shift by weight permutation."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    observed = one_sided_weighted_ks_statistic(values, weights)
    rng = np.random.Generator(np.random.PCG64DXSM(int(seed)))
    exceedances = 0
    for _ in range(int(iterations)):
        statistic = one_sided_weighted_ks_statistic(
            values, rng.permutation(weights)
        )
        exceedances += int(statistic >= observed)
    return {
        "statistic": observed,
        "permutation_pvalue": (exceedances + 1) / (int(iterations) + 1),
        "exceedances": exceedances,
        "iterations": int(iterations),
        "seed": int(seed),
        "rng": "NumPy Generator PCG64DXSM",
        "tail": "one-sided greater-or-equal",
    }


def failure_classification(
    lineage_pass: bool,
    finite_pass: bool,
    stability_pass: bool,
    integration_support_pass: bool,
    parent_support_pass: bool,
    cf4_pass: bool,
) -> str | None:
    """Apply the preregistered hierarchy without claiming tension from bad MC."""
    if not lineage_pass or not finite_pass:
        return "invalid_numerical_or_lineage"
    if not stability_pass or not integration_support_pass:
        return "Monte_Carlo_or_proposal_instability"
    if not parent_support_pass or not cf4_pass:
        return "parent_support_or_CF4_compatibility"
    return None


def validate_program_contract(
    program: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    if program.get("status") != "frozen_before_all_256_parent_evidence_only_audit":
        raise RuntimeError("all-parent program is not in the frozen status")
    canonical = Path(program["storage"]["canonical_output"]).resolve()
    if output_path.resolve() != canonical:
        raise RuntimeError("output path is not the frozen canonical output")
    authorization = program["authorization"]
    record_path = (ROOT / authorization["phase_cache_result_record"]).resolve()
    if sha256_file(record_path) != authorization["phase_cache_result_record_sha256"]:
        raise RuntimeError("phase-control authorization hash mismatch")
    with record_path.open() as stream:
        record = json.load(stream)
    if record.get("status") != authorization["required_status"]:
        raise RuntimeError("phase-control prerequisite status did not pass")
    if not record.get("decision", {}).get(
        "freeze_all_256_parent_evidence_program_authorized", False
    ):
        raise RuntimeError("phase-control record did not authorize this program")
    return record


def validate_reference_calibration(
    calibration: dict[str, Any],
    program: dict[str, Any],
) -> None:
    required = program["reference_calibration"]["required_status"]
    if calibration.get("status") != required:
        raise RuntimeError("reference calibration prerequisite status did not pass")
    if not calibration.get("two_chain_audit", {}).get("all_pass", False):
        raise RuntimeError("reference calibration two-chain audit did not pass")
    if not calibration.get("decision", {}).get(
        "authorize_opening_V8_projection_CF4_metrics", False
    ):
        raise RuntimeError("reference calibration did not authorize CF4 metrics")


def geometry_row(
    program: dict[str, Any],
    peak: dict[str, Any],
    midpoint_seed: int,
    axis_seed: int,
) -> dict[str, Any]:
    n = int(program["mesh"]["fine_N"])
    box = float(program["mesh"]["box_size_mpc_h"])
    dx = box / n
    offset, source = draw_protohalo_midpoint_offset(peak, midpoint_seed)
    midpoint = np.full(3, n // 2, dtype=np.int64) + np.rint(
        offset / dx
    ).astype(np.int64)
    axis = np.random.default_rng(axis_seed).normal(size=3)
    points, kinds = two_peak_points(
        n,
        midpoint,
        axis,
        int(round(float(peak["protohalo_separation_mpc_h"]) / dx)),
        int(round(float(peak["shell_radius_mpc_h"]) / dx)),
    )
    targets = np.where(
        kinds == 1,
        float(peak["centre_target_delta_linear"]),
        float(peak["six_shell_target_delta_linear"]),
    )
    return {
        "midpoint_seed": midpoint_seed,
        "axis_seed": axis_seed,
        "midpoint_offset_mpc_h": offset,
        "midpoint_grid": np.mod(midpoint, n),
        "axis": axis / np.linalg.norm(axis),
        "points": points,
        "kinds": kinds,
        "targets": targets,
        "log_midpoint_target_over_proposal": float(
            source["log_target_prior_over_sampling_proposal"]
        ),
    }


def _worker_parent_means(case: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_FILTER is None or _WORKER_POINT_SETS is None:
        raise RuntimeError("worker arrays were not initialized before fork")
    path = Path(case["path"])
    actual_hash = sha256_file(path)
    if actual_hash != case["sha256"]:
        raise RuntimeError(f"parent hash mismatch for seed {case['seed']}")
    with np.load(path, allow_pickle=False) as item:
        internal_seed = int(item["sample_seed"])
        field = item["s_out"].astype(np.float32)
    if internal_seed != int(case["seed"]):
        raise RuntimeError("parent internal seed mismatch")
    means = parent_mean_at_point_sets(
        field, _WORKER_FILTER, _WORKER_POINT_SETS
    )
    return {
        "seed": int(case["seed"]),
        "field_sha256": actual_hash,
        "means": np.asarray(means),
    }


def prepare_parent_cases(calibration: dict[str, Any], program: dict[str, Any]) -> list[dict[str, Any]]:
    expected = list(range(
        int(program["parents"]["seed_range_inclusive"][0]),
        int(program["parents"]["seed_range_inclusive"][1]) + 1,
    ))
    cases = sorted(calibration["reference_field_hashes"], key=lambda row: int(row["seed"]))
    seeds = [int(row["seed"]) for row in cases]
    if seeds != expected or len(cases) != int(program["parents"]["count"]):
        raise RuntimeError("reference parent lineage is incomplete or reordered")
    return cases


def evidence_from_means(
    means: np.ndarray,
    geometries: list[dict[str, Any]],
    cholesky: list[np.ndarray],
    log_determinants: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    log_importance = []
    rows = []
    dimension = len(geometries[0]["targets"])
    constant = dimension * math.log(2.0 * math.pi)
    for index, geometry in enumerate(geometries):
        residual = np.asarray(geometry["targets"]) - means[index]
        whitened = np.linalg.solve(cholesky[index], residual)
        quadratic = float(whitened @ whitened)
        log_z = -0.5 * (constant + float(log_determinants[index]) + quadratic)
        log_ratio = float(geometry["log_midpoint_target_over_proposal"])
        log_importance.append(log_z + log_ratio)
        rows.append({
            "log_Z_peak": log_z,
            "log_midpoint_target_over_proposal": log_ratio,
            "log_importance": log_z + log_ratio,
            "quadratic": quadratic,
            "log_determinant": float(log_determinants[index]),
        })
    return np.asarray(log_importance), rows


def run(program: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    with (ROOT / program["Local_Group_model"]["source_program"]).open() as stream:
        v8 = json.load(stream)
    peak = v8["peak_constraints"]
    calibration_path = Path(program["reference_calibration"]["path"])
    if sha256_file(calibration_path) != program["reference_calibration"]["sha256"]:
        raise RuntimeError("reference calibration hash mismatch")
    with calibration_path.open() as stream:
        calibration = json.load(stream)
    validate_reference_calibration(calibration, program)
    parent_cases = prepare_parent_cases(calibration, program)

    integration = program["integration"]
    count = int(integration["draw_count"])
    midpoint_start = int(integration["midpoint_seed_range_inclusive"][0])
    axis_start = int(integration["axis_seed_range_inclusive"][0])
    geometries = [
        geometry_row(program, peak, midpoint_start + index, axis_start + index)
        for index in range(count)
    ]
    point_sets = [row["points"] for row in geometries]

    filter_path = Path(program["density_filter"]["path"])
    if sha256_file(filter_path) != program["density_filter"]["sha256"]:
        raise RuntimeError("density filter hash mismatch")
    filter_full = full_spectrum_from_rfft(np.load(filter_path, allow_pickle=False))
    covariance, cache = covariance_for_point_sets(
        filter_full, int(program["mesh"]["coarse_N"]), point_sets
    )
    sigma = float(peak["likelihood_sigma_delta"])
    observation = [
        matrix + np.eye(len(matrix)) * sigma ** 2 for matrix in covariance
    ]
    cholesky = [np.linalg.cholesky(matrix) for matrix in observation]
    log_determinants = np.asarray([
        2.0 * np.sum(np.log(np.diag(factor))) for factor in cholesky
    ])

    global _WORKER_FILTER, _WORKER_POINT_SETS
    _WORKER_FILTER = filter_full
    _WORKER_POINT_SETS = point_sets
    workers = int(program["execution"]["worker_processes"])
    parent_rows = []
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        for number, result in enumerate(
            pool.imap(_worker_parent_means, parent_cases, chunksize=1), start=1
        ):
            log_values, evidence_rows = evidence_from_means(
                result["means"], geometries, cholesky, log_determinants
            )
            half = count // 2
            parent_rows.append({
                "seed": result["seed"],
                "field_sha256": result["field_sha256"],
                "log_importance_by_draw": log_values,
                "log_parent_marginal_evidence": logsumexp(log_values) - math.log(count),
                "log_parent_marginal_first_half": logsumexp(log_values[:half]) - math.log(half),
                "log_parent_marginal_second_half": logsumexp(log_values[half:]) - math.log(count - half),
                "evidence_terms": evidence_rows,
            })
            print(
                f"[all-parent] {number}/{len(parent_cases)} seed={result['seed']}",
                flush=True,
            )
    _WORKER_FILTER = None
    _WORKER_POINT_SETS = None

    seeds = np.asarray([row["seed"] for row in parent_rows])
    if not np.array_equal(seeds, np.arange(seeds[0], seeds[0] + len(seeds))):
        raise RuntimeError("worker results lost parent order")
    log_parent = np.asarray([
        row["log_parent_marginal_evidence"] for row in parent_rows
    ])
    log_first = np.asarray([
        row["log_parent_marginal_first_half"] for row in parent_rows
    ])
    log_second = np.asarray([
        row["log_parent_marginal_second_half"] for row in parent_rows
    ])
    weights = normalized_weights(log_parent)
    first_weights = normalized_weights(log_first)
    second_weights = normalized_weights(log_second)
    ess = effective_sample_size(weights)
    ess_first = effective_sample_size(first_weights)
    ess_second = effective_sample_size(second_weights)
    half_l1 = float(np.sum(np.abs(first_weights - second_weights)))
    half_ess_relative = float(
        abs(ess_first - ess_second) / max(ess_first, ess_second)
    )
    log_joint = np.stack([
        np.asarray(row["log_importance_by_draw"], dtype=np.float64)
        for row in parent_rows
    ])
    joint = joint_importance_diagnostics(log_joint)
    parent_geometry_ess = joint["parent_geometry_ESS"]

    reference_by_seed = {int(row["seed"]): row for row in calibration["rows"]}
    deviances = np.asarray([
        float(reference_by_seed[int(seed)]["marginal_deviance"]) for seed in seeds
    ])
    thresholds = calibration["L3_reference_thresholds"]
    weighted_q90 = weighted_quantile(deviances, weights, 0.90)
    weighted_q99_mass = float(np.sum(
        weights[deviances > float(thresholds["deviance_Q99"])]
    ))
    ks_program = program["CF4_weighted_shift_test"]
    weighted_ks = weighted_ks_permutation_test(
        deviances,
        weights,
        int(ks_program["permutations"]),
        int(ks_program["seed"]),
    )

    gate = program["gates"]
    lineage_pass = bool(
        len(parent_rows) == int(program["parents"]["count"])
        and all(row["field_sha256"] == case["sha256"]
                for row, case in zip(parent_rows, parent_cases))
    )
    finite_pass = bool(all(
        np.all(np.isfinite(row["log_importance_by_draw"]))
        and all(
            np.isfinite(term["log_Z_peak"])
            and np.isfinite(term["quadratic"])
            and np.isfinite(term["log_determinant"])
            for term in row["evidence_terms"]
        )
        for row in parent_rows
    ))
    stability_pass = bool(
        half_l1 <= gate["half_bank_parent_weight_L1_max"]
        and half_ess_relative <= gate["half_bank_ESS_relative_difference_max"]
    )
    integration_support_pass = bool(
        joint["joint_parent_geometry_ESS"]
        >= gate["joint_parent_geometry_ESS_min"]
        and joint["maximum_joint_parent_geometry_weight"]
        <= gate["maximum_joint_parent_geometry_weight_max"]
    )
    support_pass = bool(
        ess >= gate["parent_ESS_min"]
        and float(np.max(weights)) <= gate["maximum_parent_weight_max"]
    )
    cf4_pass = bool(
        weighted_q99_mass <= gate["weighted_CF4_reference_Q99_mass_max"]
        and weighted_q90 <= float(thresholds["deviance_Q99p5"])
        and weighted_ks["permutation_pvalue"]
        >= gate["weighted_CF4_one_sided_KS_permutation_p_min"]
    )
    passed = bool(
        lineage_pass and finite_pass and stability_pass
        and integration_support_pass and support_pass and cf4_pass
    )
    for row, weight, deviance, geometry_ess in zip(
        parent_rows, weights, deviances, parent_geometry_ess
    ):
        row["normalized_parent_weight"] = float(weight)
        row["CF4_marginal_deviance"] = float(deviance)
        row["geometry_importance_ESS"] = float(geometry_ess)

    order = np.argsort(weights)[::-1]
    return {
        "schema": "ouruniv-cf4-all-parent-peak-evidence-result-v1",
        "status": (
            "complete_pass_all_parent_peak_evidence_feasibility"
            if passed else "complete_fail_all_parent_peak_evidence_feasibility"
        ),
        "information_firewall": program["information_firewall"],
        "mesh": program["mesh"],
        "execution": {
            "worker_processes": workers,
            "elapsed_seconds": time.monotonic() - started,
        },
        "phase_cache": cache,
        "geometry_rows": geometries,
        "parent_rows": parent_rows,
        "summary": {
            "parent_ESS": ess,
            "maximum_parent_weight": float(np.max(weights)),
            "first_half_parent_ESS": ess_first,
            "second_half_parent_ESS": ess_second,
            "half_bank_parent_weight_L1": half_l1,
            "half_bank_ESS_relative_difference": half_ess_relative,
            "joint_parent_geometry_ESS": joint["joint_parent_geometry_ESS"],
            "maximum_joint_parent_geometry_weight": joint[
                "maximum_joint_parent_geometry_weight"
            ],
            "minimum_parent_geometry_ESS": float(np.min(parent_geometry_ess)),
            "median_parent_geometry_ESS": float(np.median(parent_geometry_ess)),
            "parent_geometry_ESS_Q05": float(np.quantile(
                parent_geometry_ess, 0.05, method="linear"
            )),
            "weighted_CF4_deviance_Q90": weighted_q90,
            "reference_CF4_deviance_Q99p5": float(thresholds["deviance_Q99p5"]),
            "weighted_CF4_reference_Q99_exceedance_mass": weighted_q99_mass,
            "weighted_CF4_one_sided_KS": weighted_ks,
            "top_parent_weights": [
                {"seed": int(seeds[index]), "weight": float(weights[index])}
                for index in order[:10]
            ],
        },
        "gates": {
            "all_parent_lineage": lineage_pass,
            "finite_normalized_evidence": finite_pass,
            "half_bank_stability": stability_pass,
            "joint_integration_support": integration_support_pass,
            "parent_support": support_pass,
            "weighted_CF4_compatibility": cf4_pass,
            "feasibility_pass": passed,
        },
        "decision": {
            "conditional_field_bank_authorized": passed,
            "candidate_generation_authorized": False,
            "PM_or_RAMSES_authorized": False,
            "failure_class": failure_classification(
                lineage_pass,
                finite_pass,
                stability_pass,
                integration_support_pass,
                support_pass,
                cf4_pass,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    program_path = Path(args.program).resolve()
    output_path = Path(args.out).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    with program_path.open() as stream:
        program = json.load(stream)
    validate_program_contract(program, output_path)
    for item in program["pinned_local_files"]:
        path = (ROOT / item["path"]).resolve()
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"local hash mismatch: {item['path']}")
    result = run(program)
    result["lineage"] = {
        "program": str(program_path),
        "program_sha256": sha256_file(program_path),
        "implementation_sha256": program["implementation"]["sha256"],
        "density_filter_sha256": program["density_filter"]["sha256"],
        "reference_calibration_sha256": program["reference_calibration"]["sha256"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, default=json_default)
        stream.write("\n")
    temporary.replace(output_path)
    print(f"[all-parent] status={result['status']}", flush=True)


if __name__ == "__main__":
    main()
