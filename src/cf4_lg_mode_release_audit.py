#!/usr/bin/env python3
"""Prospective all-256 V8 N64-to-N192 CF4 mode-release audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kurtosis, skew

from cf4_lg_mode_release_reference import (
    mahalanobis_distance,
    parse_shell_edges,
    profile_gaussian_nuisance,
    radial_residual_metrics,
    released_shell_geometry,
    released_shell_metrics,
    summary_coordinates,
)
from cf4_lg_peak_cr import free_rfft_mask
from cf4_linear_cr import build_forward, prepare_catalog
from cf4_make_ic import fourier_resample_white_field


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


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


def validate_source(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{label} hash mismatch: {actual} != {expected_sha256}")


def right_sided_ks(reference: np.ndarray, proposal: np.ndarray) -> float:
    """Return sup_t(F_reference(t)-F_proposal(t)), including exact ties."""
    reference = np.asarray(reference, dtype=np.float64)
    proposal = np.asarray(proposal, dtype=np.float64)
    if reference.ndim != 1 or proposal.ndim != 1:
        raise ValueError("KS samples must be one-dimensional")
    if reference.size == 0 or proposal.size == 0:
        raise ValueError("KS samples must be nonempty")
    values = np.concatenate((reference, proposal))
    labels = np.concatenate((np.ones(reference.size), np.zeros(proposal.size)))
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    cumulative_reference = np.cumsum(labels[order])
    ends = np.r_[np.flatnonzero(sorted_values[1:] != sorted_values[:-1]), values.size - 1]
    total = ends + 1
    nref = cumulative_reference[ends]
    difference = nref / reference.size - (total - nref) / proposal.size
    return float(max(0.0, np.max(difference)))


def _permuted_right_sided_ks(
    values: np.ndarray, permutations: np.ndarray, reference_size: int
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    permutations = np.asarray(permutations, dtype=np.int64)
    nrow = values.size
    proposal_size = nrow - reference_size
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ends = np.r_[np.flatnonzero(sorted_values[1:] != sorted_values[:-1]), nrow - 1]
    labels = np.zeros((permutations.shape[0], nrow), dtype=np.uint8)
    rows = np.arange(permutations.shape[0])[:, None]
    labels[rows, permutations[:, :reference_size]] = 1
    cumulative_reference = np.cumsum(labels[:, order], axis=1, dtype=np.int32)
    nref = cumulative_reference[:, ends]
    total = (ends + 1)[None, :]
    difference = nref / reference_size - (total - nref) / proposal_size
    return np.maximum(0.0, np.max(difference, axis=1))


def l3_max_stat_permutation(
    reference: np.ndarray,
    proposal: np.ndarray,
    fixed_reference_q99: float,
    iterations: int,
    seed: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Common-label one-sided max-stat permutation for six fields and D shifts."""
    reference = np.asarray(reference, dtype=np.float64)
    proposal = np.asarray(proposal, dtype=np.float64)
    if reference.ndim != 2 or proposal.ndim != 2:
        raise ValueError("L3 samples must be matrices")
    if reference.shape[1] != 6 or proposal.shape[1] != 6:
        raise ValueError("L3 requires D plus five radial-RMS coordinates")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(proposal)):
        raise ValueError("L3 samples must be finite")
    nref, nprop = reference.shape[0], proposal.shape[0]
    pooled = np.vstack((reference, proposal))
    observed = np.array(
        [right_sided_ks(reference[:, j], proposal[:, j]) for j in range(6)]
        + [
            np.median(proposal[:, 0]) - np.median(reference[:, 0]),
            np.quantile(proposal[:, 0], 0.9, method="linear")
            - np.quantile(reference[:, 0], 0.9, method="linear"),
            np.mean(proposal[:, 0] > fixed_reference_q99)
            - np.mean(reference[:, 0] > fixed_reference_q99),
        ],
        dtype=np.float64,
    )
    statistics = np.empty((iterations, 9), dtype=np.float32)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        count = stop - start
        permutations = np.stack([rng.permutation(nref + nprop) for _ in range(count)])
        for coordinate in range(6):
            statistics[start:stop, coordinate] = _permuted_right_sided_ks(
                pooled[:, coordinate], permutations, nref
            )
        dref = pooled[permutations[:, :nref], 0]
        dprop = pooled[permutations[:, nref:], 0]
        statistics[start:stop, 6] = (
            np.quantile(dprop, 0.5, axis=1, method="linear")
            - np.quantile(dref, 0.5, axis=1, method="linear")
        )
        statistics[start:stop, 7] = (
            np.quantile(dprop, 0.9, axis=1, method="linear")
            - np.quantile(dref, 0.9, axis=1, method="linear")
        )
        statistics[start:stop, 8] = (
            np.mean(dprop > fixed_reference_q99, axis=1)
            - np.mean(dref > fixed_reference_q99, axis=1)
        )
    null_mean = statistics.astype(np.float64).mean(axis=0)
    null_scale = statistics.astype(np.float64).std(axis=0, ddof=1)
    if np.any(~np.isfinite(null_scale)) or np.any(null_scale <= 0.0):
        raise RuntimeError("degenerate L3 permutation studentization")
    observed_z = (observed - null_mean) / null_scale
    permuted_maximum = np.max(
        (statistics.astype(np.float64) - null_mean[None, :])
        / null_scale[None, :],
        axis=1,
    )
    observed_maximum = float(np.max(observed_z))
    pvalue = float(
        (1 + np.count_nonzero(permuted_maximum >= observed_maximum))
        / (iterations + 1)
    )
    return {
        "coordinate_names": [
            "right_KS_D_per_CF4_row",
            "right_KS_radial_RMS_bin_1",
            "right_KS_radial_RMS_bin_2",
            "right_KS_radial_RMS_bin_3",
            "right_KS_radial_RMS_bin_4",
            "right_KS_radial_RMS_bin_5",
            "D_median_shift",
            "D_Q90_shift",
            "D_reference_Q99_exceedance_fraction_shift",
        ],
        "reference_rows": nref,
        "proposal_rows": nprop,
        "iterations": iterations,
        "rng": "NumPy Generator PCG64DXSM",
        "seed": seed,
        "observed_coordinates": observed,
        "permutation_null_mean": null_mean,
        "permutation_null_sample_sd": null_scale,
        "observed_studentized_coordinates": observed_z,
        "observed_maximum_studentized_statistic": observed_maximum,
        "one_sided_max_stat_pvalue": pvalue,
    }


def simultaneous_envelope(
    values: np.ndarray, calibration: dict[str, Any], critical: float
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    q99 = np.asarray(calibration["coordinate_q99"], dtype=np.float64)
    observed = summary_coordinates(values, q99)
    reference = np.asarray(calibration["reference_summary"], dtype=np.float64)
    scale = np.asarray(calibration["bootstrap_studentization_scale"], dtype=np.float64)
    if observed.shape != reference.shape or reference.shape != scale.shape:
        raise RuntimeError("simultaneous calibration shape mismatch")
    upper = reference + float(critical) * scale
    standardized = (observed - reference) / scale
    finite = bool(np.all(np.isfinite(observed)))
    passed = finite and bool(np.all(observed <= upper))
    return {
        "proposal_summary": observed,
        "reference_summary": reference,
        "simultaneous_upper_envelope": upper,
        "studentized_difference": standardized,
        "maximum_studentized_difference": float(np.max(standardized)),
        "critical_value": float(critical),
        "finite": finite,
        "pass": passed,
    }


def frozen_mode_errors(
    field: np.ndarray, parent: np.ndarray, frozen_mask: np.ndarray
) -> dict[str, float]:
    field_fft = np.fft.rfftn(np.asarray(field, dtype=np.float64), norm="ortho")
    parent_fft = np.fft.rfftn(np.asarray(parent, dtype=np.float64), norm="ortho")
    reference = parent_fft[frozen_mask]
    difference = field_fft[frozen_mask] - reference
    denominator = float(np.sqrt(np.mean(np.abs(reference) ** 2)))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("degenerate frozen-mode normalization")
    return {
        "relative_RMS": float(np.sqrt(np.mean(np.abs(difference) ** 2)) / denominator),
        "maximum_normalized_error": float(
            np.max(np.abs(difference), initial=0.0) / denominator
        ),
    }


def projection_errors(stored: np.ndarray, recomputed: np.ndarray) -> dict[str, float]:
    stored = np.asarray(stored, dtype=np.float64)
    recomputed = np.asarray(recomputed, dtype=np.float64)
    difference = stored - recomputed
    denominator = float(np.sqrt(np.mean(recomputed**2)))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise RuntimeError("degenerate projection normalization")
    return {
        "relative_RMS": float(np.sqrt(np.mean(difference**2)) / denominator),
        "maximum_normalized_error": float(
            np.max(np.abs(difference), initial=0.0) / denominator
        ),
    }


def global_field_statistics(field: np.ndarray) -> dict[str, float]:
    values = np.asarray(field, dtype=np.float64).ravel()
    return {
        "std": float(values.std()),
        "skew": float(skew(values)),
        "excess_kurtosis": float(kurtosis(values)),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, default=json_default)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    program = json.loads(args.program.read_text())
    if program.get("status") != "frozen_after_reference_pass_before_V8_metrics":
        raise RuntimeError("V8 mode-release audit program is not frozen")
    output = Path(program["storage"]["canonical_output"])
    if args.out.resolve() != output.resolve():
        raise RuntimeError("output path differs from the frozen canonical output")
    if output.exists():
        raise FileExistsError(f"immutable audit output already exists: {output}")

    implementation = (ROOT / program["implementation"]["path"]).resolve()
    if implementation != Path(__file__).resolve():
        raise RuntimeError("frozen implementation path does not name this program")
    validate_source(
        implementation, program["implementation"]["sha256"], "audit implementation"
    )
    loaded: dict[str, tuple[Path, dict[str, Any]]] = {}
    for name, spec in program["inputs"].items():
        path = Path(spec["path"])
        if not path.is_absolute():
            path = ROOT / path
        validate_source(path, spec["sha256"], name)
        loaded[name] = (path, json.loads(path.read_text()))
    for name, spec in program["source_dependencies"].items():
        validate_source(ROOT / spec["path"], spec["sha256"], name)

    reference_program = loaded["reference_program"][1]
    reference_result = loaded["reference_result_record"][1]
    calibration = loaded["reference_calibration"][1]
    terminal = loaded["V8_terminal_record"][1]
    autopsy = loaded["V8_autopsy_result_record"][1]
    proposal_path, proposal = loaded["proposal_manifest"]
    projection_path, projection = loaded["projection_manifest"]
    required_status = program["required_status"]
    status_checks = {
        "reference_result_pass": reference_result["status"]
        == required_status["reference_result_record"],
        "reference_calibration_pass": calibration["status"]
        == required_status["reference_calibration"],
        "reference_authorizes_projection_metrics": bool(
            calibration["decision"]["authorize_opening_V8_projection_CF4_metrics"]
        ),
        "V8_terminal_closed": terminal["status"] == required_status["V8_terminal_record"],
        "V8_autopsy_pass": autopsy["status"]
        == required_status["V8_autopsy_result_record"],
    }

    expected_proposal_seeds = list(range(*program["proposal_seed_range_python"]))
    proposal_by_seed = {int(row["proposal_seed"]): row for row in proposal["entries"]}
    projection_by_seed = {
        int(row["proposal_seed"]): row for row in projection["entries"]
    }
    reference_hashes = {
        int(row["seed"]): row for row in calibration["reference_field_hashes"]
    }
    expected_reference_seeds = list(range(*program["reference_seed_range_python"]))
    lineage_checks = {
        **status_checks,
        "proposal_seed_set_exact": sorted(proposal_by_seed) == expected_proposal_seeds,
        "projection_seed_set_exact": sorted(projection_by_seed)
        == expected_proposal_seeds,
        "reference_seed_set_exact": sorted(reference_hashes)
        == expected_reference_seeds,
        "proposal_entry_order_exact": [
            int(row["proposal_seed"]) for row in proposal["entries"]
        ] == expected_proposal_seeds,
        "projection_entry_order_exact": [
            int(row["proposal_seed"]) for row in projection["entries"]
        ] == expected_proposal_seeds,
        "projection_output_order_exact": projection["outputs"] == [
            row["field"] for row in projection["entries"]
        ],
        "geometry_exact": (
            int(proposal["mesh_size"]) == int(program["geometry"]["proposal_N"])
            and int(proposal["frozen_mode_mesh_size"])
            == int(program["geometry"]["frozen_N"])
            and int(projection["configuration"]["N"])
            == int(program["geometry"]["projection_N"])
            and np.isclose(
                float(proposal["box_size_mpc_h"]),
                float(program["geometry"]["box_size_mpc_h"]),
            )
            and np.isclose(
                float(projection["configuration"]["box_size"]),
                float(program["geometry"]["box_size_mpc_h"]),
            )
        ),
        "proposal_manifest_source_config_hash": proposal["config_sha256"]
        == program["proposal_config_sha256"],
        "projection_manifest_source_hash": projection[
            "source_proposal_manifest_sha256"
        ]
        == sha256_file(proposal_path),
        "proposal_parent_is_reference_parent3429": (
            proposal["parent_field"] == program["parent3429"]["path"]
            and proposal["parent_field_sha256"]
            == program["parent3429"]["sha256"]
        ),
        "reference_program_hash_matches_calibration": calibration["program_sha256"]
        == sha256_file(loaded["reference_program"][0]),
        "reference_result_pins_calibration": reference_result["lineage"][
            "calibration_sha256"
        ]
        == sha256_file(loaded["reference_calibration"][0]),
    }
    mismatches: list[dict[str, Any]] = []
    projection_rows = []

    for number, seed in enumerate(expected_reference_seeds, 1):
        spec = reference_hashes.get(seed)
        if spec is None:
            mismatches.append({"kind": "reference_manifest_seed_missing", "seed": seed})
            continue
        path = Path(spec["path"])
        actual = sha256_file(path) if path.is_file() else "missing"
        if actual != spec["sha256"]:
            mismatches.append({
                "kind": "reference_field_hash",
                "seed": seed,
                "path": str(path),
                "expected": spec["sha256"],
                "actual": actual,
            })
        else:
            with np.load(path, allow_pickle=False) as item:
                internal_seed = int(item["sample_seed"])
            if internal_seed != seed:
                mismatches.append({"kind": "reference_internal_seed", "seed": seed})
        if number % 32 == 0:
            print(f"[L0] reference hashes {number}/{len(expected_reference_seeds)}", flush=True)

    l0_gate = reference_program["gates"]["L0_lineage"]
    for number, seed in enumerate(expected_proposal_seeds, 1):
        proposal_row = proposal_by_seed.get(seed)
        projection_row = projection_by_seed.get(seed)
        if proposal_row is None or projection_row is None:
            mismatches.append({
                "kind": "proposal_or_projection_manifest_seed_missing",
                "seed": seed,
            })
            continue
        proposal_field = Path(proposal_row["field"])
        projection_field = Path(projection_row["field"])
        proposal_hash = sha256_file(proposal_field) if proposal_field.is_file() else "missing"
        projection_hash = (
            sha256_file(projection_field) if projection_field.is_file() else "missing"
        )
        if proposal_hash != proposal_row["field_sha256"]:
            mismatches.append({
                "kind": "N576_field_hash", "seed": seed,
                "expected": proposal_row["field_sha256"], "actual": proposal_hash,
            })
        if projection_hash != projection_row["field_sha256"]:
            mismatches.append({
                "kind": "N192_field_hash", "seed": seed,
                "expected": projection_row["field_sha256"], "actual": projection_hash,
            })
        nested = proposal_row.get("parent_projection", {})
        if (
            nested.get("field") != projection_row["field"]
            or nested.get("field_sha256") != projection_row["field_sha256"]
        ):
            mismatches.append({"kind": "nested_projection_hash", "seed": seed})
        if (
            proposal_hash != proposal_row["field_sha256"]
            or projection_hash != projection_row["field_sha256"]
        ):
            continue
        with np.load(proposal_field, allow_pickle=False) as item:
            internal_seed = int(item["proposal_seed"])
            canonical = item["s_conditioned"].astype(np.float32)
        with np.load(projection_field, allow_pickle=False) as item:
            projected_seed = int(item["sample_seed"])
            stored_projection = item["s_out"].astype(np.float32)
        if internal_seed != seed or projected_seed != seed:
            mismatches.append({
                "kind": "proposal_projection_internal_seed", "seed": seed,
                "N576_seed": internal_seed, "N192_seed": projected_seed,
            })
        recomputed = fourier_resample_white_field(canonical, int(
            projection["configuration"]["N"]
        ))
        errors = projection_errors(stored_projection, recomputed)
        errors["seed"] = seed
        projection_rows.append(errors)
        if (
            errors["relative_RMS"] > float(l0_gate["projection_relative_RMS_max"])
            or errors["maximum_normalized_error"]
            > float(l0_gate["projection_maximum_normalized_error_max"])
        ):
            mismatches.append({"kind": "projection_recomputation", **errors})
        del canonical, stored_projection, recomputed
        if number % 8 == 0:
            print(f"[L0] exact N576->N192 {number}/{len(expected_proposal_seeds)}", flush=True)

    lineage_checks["all_768_field_hashes_match"] = not any(
        row["kind"].endswith("field_hash") for row in mismatches
    )
    lineage_checks["all_internal_seed_identities_match"] = not any(
        "internal_seed" in row["kind"] for row in mismatches
    )
    lineage_checks["all_exact_projections_pass"] = (
        len(projection_rows) == len(expected_proposal_seeds)
        and not any(row["kind"] == "projection_recomputation" for row in mismatches)
    )
    l0_pass = all(lineage_checks.values()) and not mismatches
    base_report = {
        "schema": "ouruniv-cf4-lg-v8-mode-release-audit-v1",
        "program": str(args.program.resolve()),
        "program_sha256": sha256_file(args.program),
        "implementation": str(implementation),
        "implementation_sha256": sha256_file(implementation),
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, (path, _) in loaded.items()
        },
        "L0": {
            "checks": lineage_checks,
            "reference_fields_rehashed": len(expected_reference_seeds),
            "N576_fields_rehashed": len(expected_proposal_seeds),
            "N192_fields_rehashed": len(expected_proposal_seeds),
            "exact_projections_recomputed": len(projection_rows),
            "maximum_projection_relative_RMS": max(
                (row["relative_RMS"] for row in projection_rows), default=math.inf
            ),
            "maximum_projection_normalized_error": max(
                (row["maximum_normalized_error"] for row in projection_rows),
                default=math.inf,
            ),
            "mismatches": mismatches,
            "pass": l0_pass,
        },
    }
    if not l0_pass:
        base_report.update({
            "status": "complete_fail_L0_stop_before_V8_metrics",
            "decision": {
                "V8_projection_audit_pass": False,
                "V9_design_authorized": False,
                "seed_promotion_authorized": False,
                "RAMSES_authorized": False,
            },
        })
        write_report(output, base_report)
        return

    reference_manifest_path = Path(calibration["reference_manifest"])
    validate_source(
        reference_manifest_path,
        calibration["reference_manifest_sha256"],
        "reference manifest",
    )
    reference_manifest = json.loads(reference_manifest_path.read_text())
    catalog_path = Path(calibration["catalog"])
    validate_source(catalog_path, calibration["catalog_sha256"], "CF4 catalog")
    if Path(reference_manifest["catalog"]).resolve() != catalog_path.resolve():
        raise RuntimeError("reference manifest catalog path differs from calibration")
    config = argparse.Namespace(**reference_manifest["configuration"])
    data = prepare_catalog(config)
    if np.any(data["holdout"]) or data["raw_idx"].size != calibration["CF4_likelihood_rows"]:
        raise RuntimeError("CF4 likelihood catalog differs from sealed reference")
    forward, _, _, npdtype = build_forward(data["pos"], data["rhat"], config)
    import jax.numpy as jnp

    parent_path = Path(program["parent3429"]["path"])
    validate_source(parent_path, program["parent3429"]["sha256"], "parent3429")
    with np.load(parent_path, allow_pickle=False) as item:
        parent_field = item["s_out"].astype(np.float64)
        reference_mean = item["s_map"].astype(np.float64)
        if sha256_array(item["s_map"]) != calibration["canonical_Wiener_mean"][
            "array_sha256"
        ]:
            raise RuntimeError("canonical Wiener mean hash mismatch")
    reference_mean_fft = np.fft.rfftn(reference_mean, norm="ortho")
    parent_fft = np.fft.rfftn(parent_field, norm="ortho")
    frozen_mask = ~free_rfft_mask(
        int(config.N), int(reference_program["released_modes"]["frozen_mesh_size"])
    )
    geometry = released_shell_geometry(
        int(config.N), float(config.box_size),
        int(reference_program["released_modes"]["frozen_mesh_size"]),
        parse_shell_edges(reference_program["released_modes"]["shell_edges_h_mpc"]),
    )
    radial_edges = np.asarray(
        reference_program["radial_residuals"]["cz_edges_km_s"], dtype=np.float64
    )
    rows = []
    for number, seed in enumerate(expected_proposal_seeds, 1):
        path = Path(projection_by_seed[seed]["field"])
        with np.load(path, allow_pickle=False) as item:
            field = item["s_out"].astype(np.float64)
        frozen = frozen_mode_errors(field, parent_field, frozen_mask)
        prediction = np.asarray(forward(jnp.asarray(field, dtype=npdtype)), dtype=np.float64)
        nuisance = profile_gaussian_nuisance(
            data["vobs"], prediction, data["variance"], data["B"], data["q_std"]
        )
        bias, rms, _ = radial_residual_metrics(
            nuisance["standardized_residual"], data["cz"], radial_edges
        )
        shells = released_shell_metrics(
            field, reference_mean_fft, parent_fft, geometry
        )
        global_stats = global_field_statistics(field)
        rows.append({
            "seed": seed,
            "field": str(path.resolve()),
            "field_sha256": projection_by_seed[seed]["field_sha256"],
            "frozen_relative_RMS": frozen["relative_RMS"],
            "frozen_maximum_normalized_error": frozen["maximum_normalized_error"],
            "marginal_deviance": nuisance["marginal_deviance"],
            "deviance_per_CF4_row": nuisance["marginal_deviance"]
            / data["raw_idx"].size,
            "qhat": nuisance["qhat"],
            "radial_bias": bias,
            "radial_rms": rms,
            "released_Eres": shells["Eres"],
            "released_Pwhite": shells["Pwhite"],
            "released_delta_E_parent3429": shells["delta_E_parent3429"],
            "global_white_field_std": global_stats["std"],
            "global_white_field_skew": global_stats["skew"],
            "global_white_field_excess_kurtosis": global_stats["excess_kurtosis"],
        })
        if number % 16 == 0:
            print(f"[metrics] V8 projection {number}/{len(expected_proposal_seeds)}", flush=True)

    l1_gate = reference_program["gates"]["L1_N64_preservation"]
    l1_relative = np.asarray([row["frozen_relative_RMS"] for row in rows])
    l1_maximum = np.asarray([
        row["frozen_maximum_normalized_error"] for row in rows
    ])
    l1_pass = bool(
        np.all(l1_relative <= float(l1_gate["relative_RMS_max"]))
        and np.all(l1_maximum <= float(l1_gate["maximum_normalized_error_max"]))
    )

    reference_rows = [
        row for row in calibration["rows"]
        if int(row["seed"]) != int(calibration["excluded_parent_seed"])
    ]
    proposal_d = np.asarray([row["deviance_per_CF4_row"] for row in rows])
    proposal_rms = np.asarray([row["radial_rms"] for row in rows])
    reference_d = np.asarray([row["deviance_per_CF4_row"] for row in reference_rows])
    reference_rms = np.asarray([row["radial_rms"] for row in reference_rows])
    l3_gate = reference_program["gates"]["L3_CF4_deviance"]
    fixed_q99_per_row = float(calibration["L3_reference_thresholds"]["deviance_Q99"])
    fixed_q99_per_row /= float(calibration["CF4_likelihood_rows"])
    l3 = l3_max_stat_permutation(
        np.column_stack((reference_d, reference_rms)),
        np.column_stack((proposal_d, proposal_rms)),
        fixed_q99_per_row,
        iterations=int(reference_program["calibration"]["permutation_iterations"]),
        seed=int(reference_program["calibration"]["permutation_seed"]),
        chunk_size=int(reference_program["calibration"]["bootstrap_chunk_size"]),
    )
    proposal_deviance = proposal_d * calibration["CF4_likelihood_rows"]
    l3.update({
        "minimum_p": float(l3_gate["one_sided_permutation_max_stat_FWER_alpha"]),
        "permutation_pass": l3["one_sided_max_stat_pvalue"]
        >= float(l3_gate["one_sided_permutation_max_stat_FWER_alpha"]),
        "proposal_median_deviance": float(np.median(proposal_deviance)),
        "proposal_median_le_reference_Q95": bool(
            np.median(proposal_deviance)
            <= calibration["L3_reference_thresholds"]["deviance_Q95"]
        ),
        "proposal_Q90_deviance": float(
            np.quantile(proposal_deviance, 0.9, method="linear")
        ),
        "proposal_Q90_le_reference_Q99p5": bool(
            np.quantile(proposal_deviance, 0.9, method="linear")
            <= calibration["L3_reference_thresholds"]["deviance_Q99p5"]
        ),
        "proposal_reference_Q99_exceedance_fraction": float(
            np.mean(proposal_d > fixed_q99_per_row)
        ),
        "exceedance_fraction_le_reference_bootstrap_Q99p9": bool(
            np.mean(proposal_d > fixed_q99_per_row)
            <= calibration["L3_reference_thresholds"][
                "Q99_exceedance_fraction_bootstrap_Q99p9"
            ]
        ),
    })
    l3["pass"] = bool(
        l3["permutation_pass"]
        and l3["proposal_median_le_reference_Q95"]
        and l3["proposal_Q90_le_reference_Q99p5"]
        and l3["exceedance_fraction_le_reference_bootstrap_Q99p9"]
    )

    qhat = np.asarray([row["qhat"] for row in rows])
    radial_bias = np.asarray([row["radial_bias"] for row in rows])
    qhat_transform = calibration["L4_transform"]
    l4_matrix = np.column_stack((
        np.abs(qhat - np.asarray(qhat_transform["qhat_reference_median"])[None, :]),
        mahalanobis_distance(
            qhat,
            np.asarray(qhat_transform["qhat_reference_mean"]),
            np.asarray(qhat_transform["qhat_reference_covariance"]),
        ),
        np.abs(
            radial_bias
            - np.asarray(qhat_transform["radial_bias_reference_median"])[None, :]
        ),
        proposal_rms,
    ))
    pwhite = np.asarray([row["released_Pwhite"] for row in rows])
    l5_matrix = np.column_stack((
        np.asarray([row["released_Eres"] for row in rows]),
        np.abs(pwhite - 1.0),
        np.asarray([row["released_delta_E_parent3429"] for row in rows]),
    ))
    bootstrap = calibration["simultaneous_bootstrap_calibration"]
    critical = float(bootstrap["simultaneous_studentized_max_critical"])
    l4 = simultaneous_envelope(
        l4_matrix, bootstrap["families"]["L4_qhat_radial"], critical
    )
    l5 = simultaneous_envelope(
        l5_matrix, bootstrap["families"]["L5_released_modes"], critical
    )
    l5_gate = reference_program["gates"]["L5_released_modes"]
    global_std = np.asarray([row["global_white_field_std"] for row in rows])
    global_skew = np.asarray([row["global_white_field_skew"] for row in rows])
    global_kurtosis = np.asarray([
        row["global_white_field_excess_kurtosis"] for row in rows
    ])
    std_low, std_high = map(float, l5_gate["global_white_field_std_each_projection"])
    shell_tolerance = np.maximum(
        0.05, 3.0 / np.sqrt(np.asarray(geometry["mode_counts_rfft"], dtype=np.float64))
    )
    mean_pwhite = pwhite.mean(axis=0)
    l5.update({
        "global_std_range": [float(global_std.min()), float(global_std.max())],
        "global_std_each_projection_pass": bool(
            np.all((global_std >= std_low) & (global_std <= std_high))
        ),
        "maximum_abs_global_skew": float(np.max(np.abs(global_skew))),
        "global_skew_each_projection_pass": bool(
            np.all(np.abs(global_skew)
                   <= float(l5_gate["global_white_field_max_abs_skew_each_projection"]))
        ),
        "maximum_abs_global_excess_kurtosis": float(
            np.max(np.abs(global_kurtosis))
        ),
        "global_excess_kurtosis_each_projection_pass": bool(
            np.all(
                np.abs(global_kurtosis)
                <= float(l5_gate[
                    "global_white_field_max_abs_excess_kurtosis_each_projection"
                ])
            )
        ),
        "released_shell_mean_Pwhite": mean_pwhite,
        "released_shell_mean_Pwhite_tolerance": shell_tolerance,
        "released_shell_mean_Pwhite_pass": bool(
            np.all(np.abs(mean_pwhite - 1.0) <= shell_tolerance)
        ),
        "all_metrics_finite": bool(all(
            np.all(np.isfinite(value)) for value in (
                l4_matrix, l5_matrix, global_std, global_skew, global_kurtosis
            )
        )),
    })
    l5["pass"] = bool(
        l5["pass"]
        and l5["global_std_each_projection_pass"]
        and l5["global_skew_each_projection_pass"]
        and l5["global_excess_kurtosis_each_projection_pass"]
        and l5["released_shell_mean_Pwhite_pass"]
        and l5["all_metrics_finite"]
    )
    l2_pass = bool(calibration["L2_parent3429"]["pass"])
    all_pass = bool(l0_pass and l1_pass and l2_pass and l3["pass"] and l4["pass"] and l5["pass"])
    base_report.update({
        "status": (
            "complete_pass_authorize_fresh_V9_design_only"
            if all_pass else "complete_fail_current_N64_freeze_architecture_V9_no_go"
        ),
        "L1": {
            "maximum_relative_RMS": float(l1_relative.max()),
            "relative_RMS_limit": float(l1_gate["relative_RMS_max"]),
            "maximum_normalized_error": float(l1_maximum.max()),
            "maximum_normalized_error_limit": float(
                l1_gate["maximum_normalized_error_max"]
            ),
            "pass": l1_pass,
        },
        "L2": calibration["L2_parent3429"],
        "L3": l3,
        "L4": l4,
        "L5": l5,
        "rows": rows,
        "decision": {
            "V8_projection_audit_pass": all_pass,
            "current_N64_freeze_architecture_V9_no_go": not all_pass,
            "V9_design_authorized": all_pass,
            "seed_promotion_authorized": False,
            "RAMSES_authorized": False,
        },
    })
    write_report(output, base_report)
    print(json.dumps({
        "status": base_report["status"],
        "L0": l0_pass,
        "L1": l1_pass,
        "L2": l2_pass,
        "L3": l3["pass"],
        "L4": l4["pass"],
        "L5": l5["pass"],
        "output": str(output.resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
