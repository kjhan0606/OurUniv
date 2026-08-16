#!/usr/bin/env python3
"""Full N576 control for exact 27-phase peak-evidence covariance caching."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from cf4_lg_peak_cr import (
    draw_protohalo_midpoint_offset,
    linear_density_filter,
    two_peak_points,
)
from cf4_peak_evidence import normalized_gaussian_logpdf
from cf4_peak_evidence_phase_cache import (
    covariance_for_point_sets,
    full_spectrum_from_rfft,
    parent_mean_at_point_sets,
    phase_cache_metadata,
)


ROOT = Path(__file__).resolve().parents[1]


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


def load_parent(case: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
    path = Path(case["path"])
    if sha256_file(path) != case["sha256"]:
        raise RuntimeError(f"parent hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as item:
        if int(item["sample_seed"]) != int(case["parent_seed"]):
            raise RuntimeError("parent internal seed mismatch")
        field = item["s_out"].astype(np.float32)
        cosmology = {
            "Om": float(item["Om"]),
            "Ob": float(item["Ob"]),
            "h": float(item["hh"]),
            "A_s_1e9": float(item["A_s_1e9"]),
            "ns": float(item["ns"]),
        }
    return field, cosmology


def geometry_row(
    program: dict[str, Any],
    peak: dict[str, Any],
    case: dict[str, int],
) -> dict[str, Any]:
    n = int(program["mesh"]["fine_N"])
    box = float(program["mesh"]["box_size_mpc_h"])
    dx = box / n
    offset, source = draw_protohalo_midpoint_offset(
        peak, int(case["midpoint_seed"])
    )
    midpoint = np.full(3, n // 2, dtype=np.int64) + np.rint(
        offset / dx
    ).astype(np.int64)
    axis = np.random.default_rng(int(case["axis_seed"])).normal(size=3)
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
        "midpoint_seed": int(case["midpoint_seed"]),
        "axis_seed": int(case["axis_seed"]),
        "midpoint_offset_draw_mpc_h": offset,
        "midpoint_source": source,
        "midpoint_grid": np.mod(midpoint, n),
        "axis": axis / np.linalg.norm(axis),
        "points": points,
        "kinds": kinds,
        "targets": targets,
    }


def relative_matrix_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(np.asarray(left) - np.asarray(right))
        / max(np.linalg.norm(np.asarray(right)), np.finfo(float).tiny)
    )


def run(program: dict[str, Any], filter_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    parent_cases = program["parent_controls"]
    first_parent, cosmology = load_parent(parent_cases[0])
    fine_n = int(program["mesh"]["fine_N"])
    coarse_n = int(program["mesh"]["coarse_N"])
    box = float(program["mesh"]["box_size_mpc_h"])
    with (ROOT / program["Local_Group_model"]["source_program"]).open() as stream:
        v8 = json.load(stream)
    peak = v8["peak_constraints"]

    filter_rfft = linear_density_filter(
        fine_n, box, float(peak["gaussian_radius_mpc_h"]), cosmology
    )
    filter_path.parent.mkdir(parents=True, exist_ok=True)
    with filter_path.open("xb") as stream:
        np.save(stream, filter_rfft, allow_pickle=False)
    filter_sha = sha256_file(filter_path)
    filter_full = full_spectrum_from_rfft(filter_rfft)
    del filter_rfft
    print(f"[phase-control] filter={filter_sha}", flush=True)

    geometries = [
        geometry_row(program, peak, case)
        for case in program["geometry_controls"]
    ]
    translation = np.asarray(
        program["coarse_translation_control"]["fine_cell_shift"],
        dtype=np.int64,
    )
    translated_points = np.mod(
        geometries[0]["points"] + translation,
        fine_n,
    )
    point_sets = [row["points"] for row in geometries] + [translated_points]
    covariance, cache = covariance_for_point_sets(
        filter_full, coarse_n, point_sets
    )
    translation_error = relative_matrix_error(covariance[-1], covariance[0])

    parents = [(parent_cases[0], first_parent)]
    for case in parent_cases[1:]:
        field, other_cosmology = load_parent(case)
        if other_cosmology != cosmology:
            raise RuntimeError("parent cosmology mismatch")
        parents.append((case, field))
    evidence_rows = []
    minimum_signal_eigenvalue = math.inf
    maximum_observation_condition = 0.0
    for geometry_index, geometry in enumerate(geometries):
        signal = covariance[geometry_index]
        minimum_signal_eigenvalue = min(
            minimum_signal_eigenvalue,
            float(np.min(np.linalg.eigvalsh(signal))),
        )
        observation = signal + np.eye(len(signal)) * float(
            peak["likelihood_sigma_delta"]
        ) ** 2
        maximum_observation_condition = max(
            maximum_observation_condition,
            float(np.linalg.cond(observation)),
        )
        np.linalg.cholesky(observation)
        for parent_case, parent in parents:
            mean = parent_mean_at_point_sets(
                parent, filter_full, [geometry["points"]]
            )[0]
            log_evidence, terms = normalized_gaussian_logpdf(
                geometry["targets"], mean, observation
            )
            evidence_rows.append({
                "geometry_index": geometry_index,
                "parent_seed": int(parent_case["parent_seed"]),
                "predicted_mean": mean,
                "normalized_log_evidence": log_evidence,
                **terms,
            })
            print(
                f"[phase-control] geometry={geometry_index} "
                f"parent={parent_case['parent_seed']} logZ={log_evidence:.6f}",
                flush=True,
            )

    gate = program["gates"]
    finite = bool(all(
        np.isfinite(row["normalized_log_evidence"])
        and np.isfinite(row["quadratic"])
        and np.isfinite(row["log_determinant"])
        for row in evidence_rows
    ))
    covariance_pass = bool(
        cache["maximum_pre_symmetrization_asymmetry"]
        <= gate["covariance_asymmetry_max"]
        and minimum_signal_eigenvalue >= gate["signal_minimum_eigenvalue_min"]
        and maximum_observation_condition
        <= gate["observation_condition_number_max"]
    )
    translation_pass = bool(
        translation_error <= gate["coarse_translation_relative_error_max"]
    )
    phase_pass = bool(
        cache["phase_count_used"] <= cache["maximum_possible_phase_count"]
        and cache["response_grids_held_simultaneously"] == 1
    )
    passed = bool(finite and covariance_pass and translation_pass and phase_pass)
    return {
        "schema": "ouruniv-cf4-peak-evidence-phase-control-result-v1",
        "status": (
            "complete_pass_exact_N576_phase_cache"
            if passed else "complete_fail_exact_N576_phase_cache"
        ),
        "information_firewall": program["information_firewall"],
        "mesh": program["mesh"],
        "phase_cache": cache,
        "filter": {
            "path": str(filter_path),
            "sha256": filter_sha,
            "shape_rfft": [fine_n, fine_n, fine_n // 2 + 1],
            "dtype": "float32",
            "radius_mpc_h": float(peak["gaussian_radius_mpc_h"]),
        },
        "geometries": geometries,
        "evidence_rows": evidence_rows,
        "diagnostics": {
            "coarse_translation_relative_error": translation_error,
            "minimum_signal_covariance_eigenvalue": minimum_signal_eigenvalue,
            "maximum_observation_condition_number": maximum_observation_condition,
            "elapsed_seconds": time.monotonic() - started,
        },
        "gates": {
            "finite_normalized_evidence": finite,
            "covariance_numerics": covariance_pass,
            "coarse_translation_identity": translation_pass,
            "bounded_phase_cache": phase_pass,
            "phase_cache_pass": passed,
        },
        "decision": {
            "exact_phase_cache_engineering_pass": passed,
            "freeze_all_parent_evidence_program_authorized": passed,
            "candidate_generation_authorized": False,
            "PM_or_RAMSES_authorized": False,
        },
        "implementation_metadata": phase_cache_metadata(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--filter-out", required=True)
    args = parser.parse_args()
    program_path = Path(args.program).resolve()
    output_path = Path(args.out).resolve()
    filter_path = Path(args.filter_out).resolve()
    if output_path.exists() or filter_path.exists():
        raise FileExistsError("refusing to overwrite control output or filter")
    with program_path.open() as stream:
        program = json.load(stream)
    for key in ("implementation", "phase_cache", "projection_contract"):
        item = program[key]
        path = (ROOT / item["path"]).resolve()
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"{key} hash mismatch")
    pinned = [
        (
            "architecture_design",
            program["authorization"]["architecture_design"],
            program["authorization"]["architecture_design_sha256"],
        ),
        (
            "peak_geometry_implementation",
            program["peak_geometry_implementation"]["path"],
            program["peak_geometry_implementation"]["sha256"],
        ),
        (
            "Local_Group_model",
            program["Local_Group_model"]["source_program"],
            program["Local_Group_model"]["source_program_sha256"],
        ),
    ]
    for label, relative_path, expected_hash in pinned:
        if sha256_file((ROOT / relative_path).resolve()) != expected_hash:
            raise RuntimeError(f"{label} hash mismatch")
    result = run(program, filter_path)
    result["lineage"] = {
        "program": str(program_path),
        "program_sha256": sha256_file(program_path),
        "implementation_sha256": program["implementation"]["sha256"],
        "phase_cache_sha256": program["phase_cache"]["sha256"],
        "projection_contract_sha256": program["projection_contract"]["sha256"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, default=json_default)
        stream.write("\n")
    temporary.replace(output_path)
    print(f"[phase-control] status={result['status']}", flush=True)


if __name__ == "__main__":
    main()
