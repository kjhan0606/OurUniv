#!/usr/bin/env python3
"""Independent 2048-geometry adaptation bank for CF4 peak evidence."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from cf4_adaptive_geometry_proposal import (
    draw_adaptation_geometry,
    fit_cross_validated_mixture,
    normalized_log_weights,
    run_synthetic_validation,
)
from cf4_all_parent_peak_evidence import (
    effective_sample_size,
    json_default,
    prepare_parent_cases,
    sha256_file,
    validate_reference_calibration,
)
from cf4_lg_peak_cr import two_peak_points
from cf4_peak_evidence_phase_cache import (
    covariance_for_point_sets,
    full_spectrum_from_rfft,
    parent_mean_at_point_sets,
)


ROOT = Path(__file__).resolve().parents[1]
_WORKER_FILTER: np.ndarray | None = None
_WORKER_POINT_SETS: list[np.ndarray] | None = None


def logsumexp_over_parents(log_values: np.ndarray) -> np.ndarray:
    values = np.asarray(log_values, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("parent log values must be a finite matrix")
    maximum = np.max(values, axis=0)
    return maximum + np.log(np.sum(np.exp(values - maximum), axis=0))


def vectorized_log_evidence(
    means: np.ndarray,
    targets: np.ndarray,
    cholesky: np.ndarray,
    log_determinants: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(means, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    cholesky = np.asarray(cholesky, dtype=np.float64)
    log_determinants = np.asarray(log_determinants, dtype=np.float64)
    if means.shape != targets.shape or means.ndim != 2:
        raise ValueError("means and targets must be aligned matrices")
    draw_count, dimension = means.shape
    if cholesky.shape != (draw_count, dimension, dimension):
        raise ValueError("batched Cholesky shape mismatch")
    if log_determinants.shape != (draw_count,):
        raise ValueError("log-determinant shape mismatch")
    residual = targets - means
    whitened = np.linalg.solve(cholesky, residual[..., None])[..., 0]
    quadratic = np.einsum("gi,gi->g", whitened, whitened)
    log_z = -0.5 * (
        dimension * math.log(2.0 * math.pi)
        + log_determinants
        + quadratic
    )
    return log_z, quadratic


def adaptation_failure_classification(
    lineage_pass: bool,
    finite_pass: bool,
    numerical_control_pass: bool,
    support_pass: bool,
    fit_pass: bool,
    cv_pass: bool,
    synthetic_pass: bool,
) -> str | None:
    if not lineage_pass or not finite_pass or not numerical_control_pass:
        return "invalid_numerical_or_lineage"
    if not support_pass:
        return "insufficient_adaptation_support_fallback_authorized"
    if not fit_pass or not cv_pass or not synthetic_pass:
        return "proposal_family_fit_or_validation_failure"
    return None


def geometry_bank(
    program: dict[str, Any],
    peak: dict[str, Any],
) -> dict[str, np.ndarray]:
    count = int(program["adaptation_bank"]["draw_count"])
    master_seed = int(program["adaptation_bank"]["master_seed"])
    n = int(program["mesh"]["fine_N"])
    box = float(program["mesh"]["box_size_mpc_h"])
    dx = box / n
    midpoint_offset = np.empty((count, 3), dtype=np.float64)
    axis = np.empty((count, 3), dtype=np.float64)
    midpoint_grid = np.empty((count, 3), dtype=np.int16)
    points = np.empty((count, 14, 3), dtype=np.int16)
    kinds = np.empty((count, 14), dtype=np.int8)
    targets = np.empty((count, 14), dtype=np.float64)
    branch = np.empty(count, dtype=np.int8)
    component = np.empty(count, dtype=np.int8)
    log_target = np.empty(count, dtype=np.float64)
    log_sampling = np.empty(count, dtype=np.float64)
    log_ratio = np.empty(count, dtype=np.float64)
    separation = int(round(float(peak["protohalo_separation_mpc_h"]) / dx))
    shell = int(round(float(peak["shell_radius_mpc_h"]) / dx))
    for index in range(count):
        row = draw_adaptation_geometry(peak, master_seed, index)
        q = np.asarray(row["midpoint_offset_mpc_h"], dtype=np.float64)
        a = np.asarray(row["axis"], dtype=np.float64)
        midpoint = np.full(3, n // 2, dtype=np.int64) + np.rint(q / dx).astype(np.int64)
        row_points, row_kinds = two_peak_points(
            n, midpoint, a, separation, shell
        )
        midpoint_offset[index] = q
        axis[index] = a
        midpoint_grid[index] = np.mod(midpoint, n)
        points[index] = row_points
        kinds[index] = row_kinds
        targets[index] = np.where(
            row_kinds == 1,
            float(peak["centre_target_delta_linear"]),
            float(peak["six_shell_target_delta_linear"]),
        )
        branch[index] = row["proposal_branch"]
        component[index] = row["proposal_component"]
        log_target[index] = row["log_target_geometry_density"]
        log_sampling[index] = row["log_sampling_geometry_density"]
        log_ratio[index] = row["log_target_over_proposal"]
    return {
        "midpoint_offset_mpc_h": midpoint_offset,
        "axis": axis,
        "midpoint_grid": midpoint_grid,
        "points": points,
        "point_kinds": kinds,
        "targets": targets,
        "proposal_branch": branch,
        "proposal_component": component,
        "log_target_geometry_density": log_target,
        "log_sampling_geometry_density": log_sampling,
        "log_target_over_proposal": log_ratio,
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
        "means": np.asarray(means, dtype=np.float64),
    }


def validate_program_contract(
    program: dict[str, Any],
    output_path: Path,
    arrays_path: Path,
    proposal_path: Path,
) -> None:
    if program.get("status") != "frozen_before_independent_2048_adaptation":
        raise RuntimeError("adaptation program is not frozen")
    expected = program["storage"]
    actual_paths = (output_path.resolve(), arrays_path.resolve(), proposal_path.resolve())
    frozen_paths = (
        Path(expected["canonical_output"]).resolve(),
        Path(expected["canonical_arrays"]).resolve(),
        Path(expected["canonical_proposal"]).resolve(),
    )
    if actual_paths != frozen_paths:
        raise RuntimeError("adaptation output path is not canonical")
    design = program["scientific_design"]
    design_path = (ROOT / design["path"]).resolve()
    if sha256_file(design_path) != design["sha256"]:
        raise RuntimeError("adaptive scientific-design hash mismatch")
    with design_path.open() as stream:
        design_record = json.load(stream)
    if design_record.get("status") != design["required_status"]:
        raise RuntimeError("adaptive scientific design is not frozen")
    source = design_record["authorization"]
    if not source.get("adaptation_execution_authorized", False):
        raise RuntimeError("adaptive scientific design did not authorize execution")
    if source.get("final_execution_authorized_before_adaptation_pass") is not False:
        raise RuntimeError("adaptive design opened the final bank prematurely")
    for key in (
        "conditional_field_bank_authorized",
        "candidate_generation_authorized",
        "parent_or_seed_selection_authorized",
        "PM_or_halo_finder_authorized",
        "RAMSES_authorized",
    ):
        if source.get(key) is not False:
            raise RuntimeError("adaptive design opened a forbidden downstream action")
    source_path = (ROOT / source["source_result_record"]).resolve()
    if sha256_file(source_path) != source["source_result_record_sha256"]:
        raise RuntimeError("source V1 result-record hash mismatch")
    with source_path.open() as stream:
        source_record = json.load(stream)
    if source_record.get("status") != source["source_result_record_required_status"]:
        raise RuntimeError("source V1 result does not authorize adaptation")
    if not source_record.get("decision", {}).get(
        "new_independent_integration_design_authorized", False
    ):
        raise RuntimeError("source V1 decision did not authorize adaptation")


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
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
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=json_default)
        stream.write("\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(
    program: dict[str, Any],
    arrays_path: Path,
    proposal_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    v8_path = ROOT / program["Local_Group_model"]["source_program"]
    with v8_path.open() as stream:
        v8 = json.load(stream)
    peak = v8["peak_constraints"]

    calibration_path = Path(program["reference_calibration"]["path"])
    if sha256_file(calibration_path) != program["reference_calibration"]["sha256"]:
        raise RuntimeError("reference calibration hash mismatch")
    with calibration_path.open() as stream:
        calibration = json.load(stream)
    validate_reference_calibration(calibration, program)
    parent_cases = prepare_parent_cases(calibration, program)

    geometry = geometry_bank(program, peak)
    point_sets = [row.astype(np.int64) for row in geometry["points"]]
    filter_path = Path(program["density_filter"]["path"])
    if sha256_file(filter_path) != program["density_filter"]["sha256"]:
        raise RuntimeError("density filter hash mismatch")
    filter_full = full_spectrum_from_rfft(np.load(filter_path, allow_pickle=False))
    covariance, phase_cache = covariance_for_point_sets(
        filter_full, int(program["mesh"]["coarse_N"]), point_sets
    )
    sigma = float(peak["likelihood_sigma_delta"])
    observation = np.asarray([
        matrix + np.eye(len(matrix)) * sigma**2 for matrix in covariance
    ])
    cholesky = np.linalg.cholesky(observation)
    log_determinants = 2.0 * np.sum(
        np.log(np.diagonal(cholesky, axis1=1, axis2=2)), axis=1
    )

    draw_count = int(program["adaptation_bank"]["draw_count"])
    parent_count = int(program["parents"]["count"])
    log_z = np.empty((parent_count, draw_count), dtype=np.float64)
    quadratic = np.empty((parent_count, draw_count), dtype=np.float64)
    parent_seeds = np.empty(parent_count, dtype=np.int32)
    parent_hashes = []
    vectorization_control: dict[str, Any] | None = None

    global _WORKER_FILTER, _WORKER_POINT_SETS
    _WORKER_FILTER = filter_full
    _WORKER_POINT_SETS = point_sets
    workers = int(program["execution"]["worker_processes"])
    context = mp.get_context("fork")
    with context.Pool(processes=workers) as pool:
        for index, result in enumerate(
            pool.imap(_worker_parent_means, parent_cases, chunksize=1)
        ):
            row_log_z, row_quadratic = vectorized_log_evidence(
                result["means"], geometry["targets"], cholesky, log_determinants
            )
            log_z[index] = row_log_z
            quadratic[index] = row_quadratic
            if index == 0:
                scalar_log_z = []
                for draw in range(draw_count):
                    residual = geometry["targets"][draw] - result["means"][draw]
                    whitened = np.linalg.solve(cholesky[draw], residual)
                    scalar_quadratic = float(whitened @ whitened)
                    scalar_log_z.append(-0.5 * (
                        len(residual) * math.log(2.0 * math.pi)
                        + log_determinants[draw]
                        + scalar_quadratic
                    ))
                scalar_log_z = np.asarray(scalar_log_z)
                log_z_difference = float(np.max(np.abs(
                    scalar_log_z - row_log_z
                )))
                log_weight_difference = float(np.max(np.abs(
                    (scalar_log_z + geometry["log_target_over_proposal"])
                    - (row_log_z + geometry["log_target_over_proposal"])
                )))
                vectorization_control = {
                    "parent_seed": result["seed"],
                    "draw_count": draw_count,
                    "scalar_vectorized_log_Z_max_difference": log_z_difference,
                    "scalar_vectorized_log_weight_max_difference": log_weight_difference,
                    "tolerance": 1e-10,
                    "pass": bool(
                        log_z_difference <= 1e-10
                        and log_weight_difference <= 1e-10
                    ),
                }
            parent_seeds[index] = result["seed"]
            parent_hashes.append({
                "seed": result["seed"],
                "sha256": result["field_sha256"],
            })
            print(
                f"[adaptation] {index + 1}/{parent_count} seed={result['seed']}",
                flush=True,
            )
    _WORKER_FILTER = None
    _WORKER_POINT_SETS = None

    expected_seeds = np.arange(
        int(program["parents"]["seed_range_inclusive"][0]),
        int(program["parents"]["seed_range_inclusive"][1]) + 1,
    )
    lineage_pass = bool(
        np.array_equal(parent_seeds, expected_seeds)
        and all(
            row["sha256"] == case["sha256"]
            for row, case in zip(parent_hashes, parent_cases)
        )
    )
    log_importance = log_z + geometry["log_target_over_proposal"][None, :]
    finite_pass = bool(
        np.all(np.isfinite(log_z))
        and np.all(np.isfinite(log_importance))
        and np.all(np.isfinite(quadratic))
        and np.all(np.isfinite(log_determinants))
    )
    if not finite_pass:
        raise RuntimeError("adaptation evidence contains nonfinite values")
    numerical_control_pass = bool(
        vectorization_control is not None and vectorization_control["pass"]
    )
    logmean_z = logsumexp_over_parents(log_z) - math.log(parent_count)
    geometry_log_weight = logmean_z + geometry["log_target_over_proposal"]
    geometry_weights = normalized_log_weights(geometry_log_weight)
    geometry_ess = effective_sample_size(geometry_weights)
    maximum_geometry_weight = float(np.max(geometry_weights))
    gate = program["gates"]
    support_pass = bool(
        geometry_ess >= gate["geometry_marginal_ESS_min"]
        and maximum_geometry_weight <= gate["maximum_normalized_geometry_weight_max"]
    )

    arrays = {
        "parent_seed": parent_seeds,
        "log_Z_peak": log_z,
        "log_importance": log_importance,
        "quadratic": quadratic,
        "midpoint_offset_mpc_h": geometry["midpoint_offset_mpc_h"],
        "axis": geometry["axis"],
        "midpoint_grid": geometry["midpoint_grid"],
        "points": geometry["points"],
        "point_kinds": geometry["point_kinds"],
        "proposal_branch": geometry["proposal_branch"],
        "proposal_component": geometry["proposal_component"],
        "log_target_geometry_density": geometry["log_target_geometry_density"],
        "log_sampling_geometry_density": geometry["log_sampling_geometry_density"],
        "log_target_over_proposal": geometry["log_target_over_proposal"],
        "logmean_parent_Z_by_geometry": logmean_z,
        "normalized_geometry_weight": geometry_weights,
        "covariance_log_determinant": log_determinants,
    }
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(arrays_path, arrays)
    arrays_sha = sha256_file(arrays_path)

    fit_result: dict[str, Any] | None = None
    synthetic: dict[str, Any] | None = None
    fit_failure: str | None = None
    cv_pass = False
    synthetic_pass = False
    if lineage_pass and finite_pass and numerical_control_pass and support_pass:
        try:
            fit_result = fit_cross_validated_mixture(
                geometry["midpoint_offset_mpc_h"],
                geometry["axis"],
                geometry_log_weight,
                geometry["log_target_geometry_density"],
                int(program["weighted_EM"]["master_seed"]),
            )
            cv_pass = bool(fit_result["all_holdout_delta_nonnegative"])
            synthetic = run_synthetic_validation(
                peak["protohalo_midpoint_prior"],
                fit_result["full_fit"]["parameters"],
                int(program["synthetic_validation"]["master_seed"]),
                int(program["synthetic_validation"]["sampling_draw_count"]),
            )
            synthetic_pass = bool(synthetic["all_pass"])
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            fit_failure = str(error)

    passed = bool(
        lineage_pass and finite_pass and numerical_control_pass and support_pass
        and fit_result is not None and cv_pass and synthetic_pass
    )
    proposal_sha = None
    if passed:
        proposal_artifact = {
            "schema": "ouruniv-cf4-defensive-geometry-proposal-v1",
            "status": "frozen_pass_adaptation_for_independent_final_bank",
            "target": "pi0(q,a)=p(q)/(4*pi)",
            "target_midpoint_prior": peak["protohalo_midpoint_prior"],
            "unchanged_peak_likelihood_sigma_delta": float(
                peak["likelihood_sigma_delta"]
            ),
            "lineage": {
                "adaptation_implementation": program["implementation"],
                "proposal_implementation": next(
                    row for row in program["pinned_local_files"]
                    if row["path"] == "src/cf4_adaptive_geometry_proposal.py"
                ),
                "scientific_design": program["scientific_design"],
                "Local_Group_source_program": {
                    "path": program["Local_Group_model"]["source_program"],
                    "sha256": program["Local_Group_model"]["source_program_sha256"],
                },
            },
            "proposal": "g_F=0.5*pi0+0.5*h_K4",
            "analytic_target_over_proposal_bound": 2.0,
            "parameters": fit_result["full_fit"]["parameters"],
            "adaptation_arrays": str(arrays_path),
            "adaptation_arrays_sha256": arrays_sha,
            "adaptation_draw_count": draw_count,
            "adaptation_master_seed": int(program["adaptation_bank"]["master_seed"]),
            "EM_master_seed": int(program["weighted_EM"]["master_seed"]),
            "fit_summary": {
                "selected_restart": fit_result["full_fit"]["selected_restart"],
                "objective": fit_result["full_fit"]["objective"],
                "iterations": fit_result["full_fit"]["iterations"],
                "component_effective_membership": fit_result["full_fit"][
                    "component_effective_membership"
                ],
                "holdout_delta": [row["holdout_delta"] for row in fit_result["folds"]],
            },
        }
        atomic_json(proposal_path, proposal_artifact)
        proposal_sha = sha256_file(proposal_path)

    failure_class = adaptation_failure_classification(
        lineage_pass,
        finite_pass,
        numerical_control_pass,
        support_pass,
        fit_result is not None,
        cv_pass,
        synthetic_pass,
    )

    return {
        "schema": "ouruniv-cf4-peak-evidence-adaptation-result-v1",
        "status": (
            "complete_pass_freeze_defensive_final_proposal"
            if passed else "complete_fail_adaptation"
        ),
        "information_firewall": program["information_firewall"],
        "execution": {
            "worker_processes": workers,
            "elapsed_seconds": time.monotonic() - started,
        },
        "mesh": program["mesh"],
        "phase_cache": phase_cache,
        "lineage": {
            "parent_field_hashes": parent_hashes,
            "arrays": str(arrays_path),
            "arrays_sha256": arrays_sha,
            "arrays_shape_dtype": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in arrays.items()
            },
            "proposal": str(proposal_path) if passed else None,
            "proposal_sha256": proposal_sha,
        },
        "summary": {
            "parent_count": parent_count,
            "geometry_draw_count": draw_count,
            "normalized_log_Z_row_count": int(log_z.size),
            "geometry_marginal_ESS": geometry_ess,
            "maximum_normalized_geometry_weight": maximum_geometry_weight,
            "geometry_weight_entropy": float(
                -np.sum(geometry_weights * np.log(geometry_weights))
            ),
            "CV_holdout_delta": (
                [row["holdout_delta"] for row in fit_result["folds"]]
                if fit_result is not None else None
            ),
            "fit_failure": fit_failure,
        },
        "fit": fit_result,
        "synthetic_validation": {
            "proposal_controls": synthetic,
            "real_evidence_vectorization_control": vectorization_control,
        },
        "gates": {
            "all_parent_lineage": lineage_pass,
            "all_log_Z_and_importance_finite": finite_pass,
            "real_evidence_scalar_vectorized_control": numerical_control_pass,
            "geometry_integration_support": support_pass,
            "all_EM_fits_converged": fit_result is not None,
            "all_CV_holdout_delta_nonnegative": cv_pass,
            "synthetic_validation": synthetic_pass,
            "adaptation_pass": passed,
        },
        "decision": {
            "final_proposal_frozen": passed,
            "independent_8192_final_bank_authorized": passed,
            "fallback_8192_adaptation_bank_authorized": bool(
                lineage_pass and finite_pass and numerical_control_pass
                and not support_pass
            ),
            "conditional_field_bank_authorized": False,
            "candidate_generation_authorized": False,
            "parent_or_seed_selection_authorized": False,
            "PM_or_RAMSES_authorized": False,
            "failure_class": failure_class,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arrays-out", required=True)
    parser.add_argument("--proposal-out", required=True)
    args = parser.parse_args()
    program_path = Path(args.program).resolve()
    output_path = Path(args.out).resolve()
    arrays_path = Path(args.arrays_out).resolve()
    proposal_path = Path(args.proposal_out).resolve()
    for path in (output_path, arrays_path, proposal_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    with program_path.open() as stream:
        program = json.load(stream)
    validate_program_contract(program, output_path, arrays_path, proposal_path)
    for item in program["pinned_local_files"]:
        path = (ROOT / item["path"]).resolve()
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"local hash mismatch: {item['path']}")
    result = run(program, arrays_path, proposal_path)
    result["lineage"].update({
        "program": str(program_path),
        "program_sha256": sha256_file(program_path),
        "implementation_sha256": program["implementation"]["sha256"],
        "scientific_design_sha256": program["scientific_design"]["sha256"],
        "density_filter_sha256": program["density_filter"]["sha256"],
        "reference_calibration_sha256": program["reference_calibration"]["sha256"],
    })
    atomic_json(output_path, result)
    print(f"[adaptation] status={result['status']}", flush=True)


if __name__ == "__main__":
    main()
