#!/usr/bin/env python3
"""Audit angular-completeness outliers before a 2M++ count likelihood.

This technical audit reuses the passing V4 equatorial-coordinate input design.
It checks the completeness map assigned to each CF4-disjoint eligible tracer,
preserves every large discrepancy as CSV, and diagnoses the N32 angular
footprint.  It performs no density-field inference and consumes no tracer as a
likelihood datum.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _load_frozen_v4() -> Any:
    path = Path(__file__).with_name("cf4_twompp_disjoint_tracer_pilot_v4.py")
    name = "_cf4_twompp_disjoint_tracer_pilot_frozen_v4_for_outlier_audit"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct the frozen V4 module specification")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v4 = _load_frozen_v4()
base = v4.base

PROGRAM_SCHEMA = "ouruniv-cf4-twompp-completeness-outlier-audit-program-v1"
RESULT_SCHEMA = "ouruniv-cf4-twompp-completeness-outlier-audit-v1"
MANIFEST_SCHEMA = "ouruniv-cf4-twompp-completeness-outlier-audit-manifest-v1"
COMPLETE_SCHEMA = "ouruniv-cf4-twompp-completeness-outlier-audit-complete-v1"
STATUS_PASS = "PASS_COMPLETENESS_OUTLIER_ZERO_EXPOSURE_GATE_NO_FIELD_INFERENCE"
STATUS_FAIL = "FAIL_COMPLETENESS_OUTLIER_ZERO_EXPOSURE_GATE_NO_FIELD_INFERENCE"
EXPECTED_FILES = {"result.json", "outliers.csv", "manifest.json", "COMPLETE"}


def _verify_binding(binding: Mapping[str, Any], label: str) -> Path:
    path = Path(str(binding["path"]))
    if not path.is_file():
        raise base.PilotError(f"bound {label} is absent: {path}")
    if "bytes" in binding and path.stat().st_size != int(binding["bytes"]):
        raise base.PilotError(f"bound {label} size changed")
    if base.sha256_file(path) != binding["sha256"]:
        raise base.PilotError(f"bound {label} hash changed")
    return path


def load_program(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    raw = Path(path).read_bytes()
    program = json.loads(raw)
    if program.get("schema") != PROGRAM_SCHEMA:
        raise base.PilotError("unexpected completeness outlier program schema")
    if not program.get("authorization", {}).get("technical_audit", False):
        raise base.PilotError("completeness outlier audit is not authorized")
    parent_program_path = _verify_binding(program["parent_v4_program"], "V4 program")
    _verify_binding(program["parent_v4_implementation"], "V4 implementation")
    parent_record_path = _verify_binding(
        program["parent_v4_result_record"], "V4 result record"
    )
    parent_result_path = _verify_binding(program["parent_v4_result"], "V4 result")
    effective, _ = v4.load_program(parent_program_path)
    parent_record = json.loads(parent_record_path.read_bytes())
    parent_result = json.loads(parent_result_path.read_bytes())
    if parent_record.get("status") != "TECHNICAL_INPUT_GATE_PASS_NO_FIELD_INFERENCE":
        raise base.PilotError("V4 result record is not passing")
    if parent_result.get("status") != v4.STATUS_PASS:
        raise base.PilotError("V4 pilot result is not passing")
    if parent_result.get("failed_gates") or not all(parent_result["gates"].values()):
        raise base.PilotError("V4 pilot result contains a failed gate")
    return program, effective, parent_result, hashlib.sha256(raw).hexdigest()


def _population_counts(
    selected: np.ndarray,
    apparent_bin: np.ndarray,
    absolute_bin: np.ndarray,
    absolute_edges: list[float],
) -> dict[str, int]:
    counts = {}
    for apparent in (0, 1):
        for absolute in range(len(absolute_edges) - 1):
            key = f"apparent_{apparent}_absolute_{absolute}"
            counts[key] = int(
                np.count_nonzero(
                    selected & (apparent_bin == apparent) & (absolute_bin == absolute)
                )
            )
    return counts


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise base.PilotError("cannot summarize empty or nonfinite values")
    return {
        "minimum": float(np.min(values)),
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def n32_footprint_summary(
    completeness11: np.ndarray,
    completeness12: np.ndarray,
    nside: int,
    design: Mapping[str, Any],
) -> dict[str, Any]:
    """Sample the angular masks at active N32 voxel centres for diagnostics."""

    import healpy as hp
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    grid = int(design["grid_N"])
    box = float(design["box_size_cMpc_h"])
    spacing = box / grid
    axis = (np.arange(grid, dtype=np.float64) + 0.5) * spacing - box / 2.0
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    radius = np.sqrt(x * x + y * y + z * z)
    active = (radius >= float(design["radial_min_cMpc_h"])) & (
        radius <= float(design["radial_max_cMpc_h"])
    )
    selected_radius = radius[active]
    longitude = np.mod(np.arctan2(y[active], x[active]), 2.0 * np.pi)
    latitude = np.arcsin(z[active] / selected_radius)
    sg = SkyCoord(
        sgl=longitude * u.rad,
        sgb=latitude * u.rad,
        distance=selected_radius * u.Mpc,
        frame="supergalactic",
    )
    icrs = sg.icrs
    theta = 0.5 * np.pi - icrs.dec.rad
    phi = np.mod(icrs.ra.rad, 2.0 * np.pi)
    pixels = hp.ang2pix(nside, theta, phi, nest=False)

    output: dict[str, Any] = {
        "grid_N": grid,
        "cell_size_cMpc_h": spacing,
        "active_radial_voxel_count": int(selected_radius.size),
        "centre_sampling_only_not_volume_integrated_selection": True,
    }
    for label, values in (("map_11_5", completeness11), ("map_12_5", completeness12)):
        exposure = values[pixels]
        output[label] = {
            "positive_exposure_voxel_count": int(np.count_nonzero(exposure > 0.0)),
            "zero_exposure_voxel_count": int(np.count_nonzero(exposure <= 0.0)),
            "positive_exposure_fraction": float(np.mean(exposure > 0.0)),
            "exposure_quantiles": _quantiles(exposure),
        }
    return output


def _outlier_rows(
    catalog: Mapping[str, np.ndarray],
    eligible: np.ndarray,
    apparent_bin: np.ndarray,
    absolute_bin: np.ndarray,
    pixels: np.ndarray,
    assigned_exposure: np.ndarray,
    assigned_mark: np.ndarray,
    difference: np.ndarray,
    completeness11: np.ndarray,
    completeness12: np.ndarray,
    nside: int,
    threshold: float,
) -> list[dict[str, Any]]:
    import healpy as hp

    selected = np.flatnonzero(eligible & ((difference > threshold) | (assigned_exposure <= 0.0)))
    rows: list[dict[str, Any]] = []
    for index in selected:
        values = completeness11 if apparent_bin[index] == 0 else completeness12
        neighbours = hp.get_all_neighbours(nside, int(pixels[index]), nest=False)
        valid_neighbours = neighbours[neighbours >= 0]
        candidate_exposures = np.concatenate(
            (np.asarray([assigned_exposure[index]]), values[valid_neighbours])
        )
        nearest_difference = float(
            np.min(np.abs(candidate_exposures - assigned_mark[index]))
        )
        rows.append(
            {
                "recno": int(catalog["recno"][index]),
                "RA_deg": float(catalog["RA"][index]),
                "DEC_deg": float(catalog["DEC"][index]),
                "Ksmag": float(catalog["Ksmag"][index]),
                "apparent_bin": int(apparent_bin[index]),
                "absolute_bin": int(absolute_bin[index]),
                "assigned_map": "11_5" if apparent_bin[index] == 0 else "12_5",
                "pixel_RING": int(pixels[index]),
                "map_exposure": float(assigned_exposure[index]),
                "catalog_mark": float(assigned_mark[index]),
                "absolute_difference": float(difference[index]),
                "zero_exposure": bool(assigned_exposure[index] <= 0.0),
                "nearest_central_or_neighbor_difference": nearest_difference,
                "neighbor_reconciled_at_threshold": bool(
                    nearest_difference <= threshold
                ),
            }
        )
    rows.sort(key=lambda row: (-row["absolute_difference"], row["recno"]))
    return rows


def canonical_csv_bytes(rows: list[Mapping[str, Any]]) -> bytes:
    fields = [
        "recno",
        "RA_deg",
        "DEC_deg",
        "Ksmag",
        "apparent_bin",
        "absolute_bin",
        "assigned_map",
        "pixel_RING",
        "map_exposure",
        "catalog_mark",
        "absolute_difference",
        "zero_exposure",
        "nearest_central_or_neighbor_difference",
        "neighbor_reconciled_at_threshold",
    ]
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def collect_audit(
    program: Mapping[str, Any],
    effective: Mapping[str, Any],
    parent_result: Mapping[str, Any],
    program_sha256: str,
    commit: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise base.PilotError("implementation commit must be lowercase 40-hex")
    inputs = effective["inputs"]
    catalog = v4.v3.load_catalog(inputs["twompp_catalog"]["path"])
    exclusions, _ = base.read_crossmatch_exclusions(
        inputs["cf4_twompp_crossmatch"]["path"],
        int(effective["no_double_counting"]["expected_unique_2Mpp_targets_excluded"]),
    )
    distance, absolute_magnitude = base.distance_and_absolute_magnitude(
        catalog["Vcmb"], catalog["Ksmag"], effective["cosmology"]
    )
    eligible, reason_counts, apparent_bin, absolute_bin = base.classify_disjoint_tracer(
        catalog,
        exclusions,
        distance,
        absolute_magnitude,
        effective["tracer_design"],
    )
    expected_eligible = int(
        parent_result["tracer_voxel_summary"]["eligible_row_count"]
    )

    nside = int(effective["angular_completeness_gate"]["HEALPix_NSIDE"])
    completeness11 = base.load_completeness_map(
        inputs["completeness_11_5"]["path"], nside
    )
    completeness12 = base.load_completeness_map(
        inputs["completeness_12_5"]["path"], nside
    )
    longitude, latitude = v4.equatorial_directions(catalog["RA"], catalog["DEC"])
    import healpy as hp

    pixels = hp.ang2pix(
        nside, 0.5 * np.pi - latitude, np.mod(longitude, 2.0 * np.pi), nest=False
    )
    map11 = completeness11[pixels]
    map12 = completeness12[pixels]
    assigned_exposure = np.where(apparent_bin == 0, map11, map12)
    assigned_mark = np.where(apparent_bin == 0, catalog["c11_5"], catalog["c12_5"])
    if np.any(~np.isfinite(assigned_mark[eligible])):
        raise base.PilotError("eligible tracer has a nonfinite assigned completeness mark")
    difference = np.abs(assigned_exposure - assigned_mark)

    thresholds = program["frozen_gates"]
    large_threshold = float(thresholds["large_absolute_difference_strictly_greater_than"])
    zero = eligible & (assigned_exposure <= 0.0)
    large = eligible & (difference > large_threshold)
    near_zero = eligible & (assigned_exposure > 0.0) & (
        assigned_exposure < float(thresholds["near_zero_exposure_strictly_less_than"])
    )
    usable = eligible & ~zero & ~large
    eligible_count = int(np.count_nonzero(eligible))
    large_count = int(np.count_nonzero(large))
    large_fraction = large_count / eligible_count
    original_populations = _population_counts(
        eligible, apparent_bin, absolute_bin, effective["tracer_design"]["absolute_K_edges"]
    )
    usable_populations = _population_counts(
        usable, apparent_bin, absolute_bin, effective["tracer_design"]["absolute_K_edges"]
    )
    rows = _outlier_rows(
        catalog,
        eligible,
        apparent_bin,
        absolute_bin,
        pixels,
        assigned_exposure,
        assigned_mark,
        difference,
        completeness11,
        completeness12,
        nside,
        large_threshold,
    )
    neighbor_reconciled = sum(
        1 for row in rows if row["neighbor_reconciled_at_threshold"]
    )
    footprint = n32_footprint_summary(
        completeness11, completeness12, nside, effective["tracer_design"]
    )
    gates = {
        "parent_v4_all_gates_pass": not parent_result["failed_gates"]
        and all(parent_result["gates"].values()),
        "eligible_count_exactly_reproduced": eligible_count == expected_eligible,
        "eligible_zero_exposure_count_zero": int(np.count_nonzero(zero)) == 0,
        "eligible_large_difference_fraction_max": large_fraction
        <= float(thresholds["large_difference_fraction_max_inclusive"]),
        "usable_tracer_minimum": int(np.count_nonzero(usable))
        >= int(thresholds["usable_tracer_count_min_inclusive"]),
        "all_six_usable_populations_nonempty": all(
            count > 0 for count in usable_populations.values()
        ),
        "both_N32_angular_footprints_nonempty": all(
            footprint[label]["positive_exposure_voxel_count"] > 0
            for label in ("map_11_5", "map_12_5")
        ),
    }
    failed = sorted(key for key, value in gates.items() if not value)
    result = {
        "schema": RESULT_SCHEMA,
        "status": STATUS_PASS if not failed else STATUS_FAIL,
        "program_sha256": program_sha256,
        "implementation_commit": commit,
        "parent_v4_result_sha256": program["parent_v4_result"]["sha256"],
        "frozen_gates": thresholds,
        "catalog_partition": reason_counts,
        "eligible_assignment": {
            "eligible_count": eligible_count,
            "assigned_exposure_quantiles": _quantiles(assigned_exposure[eligible]),
            "assigned_absolute_difference_quantiles": _quantiles(difference[eligible]),
            "zero_exposure_count": int(np.count_nonzero(zero)),
            "near_zero_exposure_count": int(np.count_nonzero(near_zero)),
            "large_difference_count": large_count,
            "large_difference_fraction": float(large_fraction),
            "usable_after_outlier_exclusion_count": int(np.count_nonzero(usable)),
            "original_six_population_counts": original_populations,
            "usable_six_population_counts": usable_populations,
        },
        "outlier_diagnostics": {
            "row_count": len(rows),
            "neighbor_reconciled_count": neighbor_reconciled,
            "neighbor_reconciled_fraction": (
                float(neighbor_reconciled / len(rows)) if rows else 1.0
            ),
            "individual_rows_artifact": "outliers.csv",
        },
        "N32_angular_footprint": footprint,
        "gates": gates,
        "failed_gates": failed,
        "likelihood_rows_consumed": 0,
        "field_inference_executed": False,
        "joint_information_budget_executed": False,
        "observational_resolution_claim_allowed": False,
        "next_action": (
            "implement_frozen_N32_joint_velocity_density_information_budget"
            if not failed
            else "stop_and_redesign_angular_selection_before_any_likelihood"
        ),
    }
    return result, rows


def publish_audit(
    program_path: str | Path, output: str | Path, implementation_commit: str
) -> dict[str, Any]:
    program, effective, parent_result, program_sha = load_program(program_path)
    target = Path(output)
    stage = target.with_name(f".{target.name}.staging")
    if target.exists() or stage.exists():
        raise base.PilotError("audit output or staging path already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    result, rows = collect_audit(
        program, effective, parent_result, program_sha, implementation_commit
    )
    result_bytes = base.canonical_json_bytes(result)
    outlier_bytes = canonical_csv_bytes(rows)
    (stage / "result.json").write_bytes(result_bytes)
    (stage / "outliers.csv").write_bytes(outlier_bytes)
    if result["status"] != STATUS_PASS:
        failed_record = {
            "schema": "ouruniv-cf4-twompp-completeness-outlier-audit-failed-v1",
            "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
            "outliers_sha256": hashlib.sha256(outlier_bytes).hexdigest(),
            "failed_gates": result["failed_gates"],
            "field_inference_executed": False,
        }
        (stage / "FAILED").write_bytes(base.canonical_json_bytes(failed_record))
        raise base.PilotError(
            f"completeness outlier gate failed: {result['failed_gates']}; diagnostics preserved"
        )

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "files": {
            "result.json": {
                "bytes": len(result_bytes),
                "sha256": hashlib.sha256(result_bytes).hexdigest(),
            },
            "outliers.csv": {
                "bytes": len(outlier_bytes),
                "sha256": hashlib.sha256(outlier_bytes).hexdigest(),
            },
        },
    }
    manifest_bytes = base.canonical_json_bytes(manifest)
    (stage / "manifest.json").write_bytes(manifest_bytes)
    complete = {
        "schema": COMPLETE_SCHEMA,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "outliers_sha256": hashlib.sha256(outlier_bytes).hexdigest(),
        "status": STATUS_PASS,
    }
    (stage / "COMPLETE").write_bytes(base.canonical_json_bytes(complete))
    os.rename(stage, target)
    return validate_audit(target)


def validate_audit(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != EXPECTED_FILES:
        raise base.PilotError("audit artifact file set is not exact")
    raw = {name: (root / name).read_bytes() for name in EXPECTED_FILES}
    result = json.loads(raw["result.json"])
    manifest = json.loads(raw["manifest.json"])
    complete = json.loads(raw["COMPLETE"])
    if result.get("schema") != RESULT_SCHEMA or result.get("status") != STATUS_PASS:
        raise base.PilotError("audit result is not passing")
    if result.get("failed_gates") or not all(result.get("gates", {}).values()):
        raise base.PilotError("audit result contains a failed gate")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise base.PilotError("audit manifest schema changed")
    for name in ("result.json", "outliers.csv"):
        expected = {
            "bytes": len(raw[name]),
            "sha256": hashlib.sha256(raw[name]).hexdigest(),
        }
        if manifest.get("files", {}).get(name) != expected:
            raise base.PilotError(f"audit manifest does not bind {name}")
    if complete.get("schema") != COMPLETE_SCHEMA:
        raise base.PilotError("audit COMPLETE schema changed")
    if complete.get("manifest_sha256") != hashlib.sha256(raw["manifest.json"]).hexdigest():
        raise base.PilotError("audit COMPLETE does not bind manifest")
    if complete.get("result_sha256") != hashlib.sha256(raw["result.json"]).hexdigest():
        raise base.PilotError("audit COMPLETE does not bind result")
    if complete.get("outliers_sha256") != hashlib.sha256(raw["outliers.csv"]).hexdigest():
        raise base.PilotError("audit COMPLETE does not bind outliers")
    if complete.get("status") != STATUS_PASS:
        raise base.PilotError("audit COMPLETE status changed")
    return {
        "status": "PASS",
        "result_sha256": hashlib.sha256(raw["result.json"]).hexdigest(),
        "outliers_sha256": hashlib.sha256(raw["outliers.csv"]).hexdigest(),
        "gates": result["gates"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-audit")
    run.add_argument("--program", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--implementation-commit", required=True)
    validate = subparsers.add_parser("validate-audit")
    validate.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "run-audit":
        result = publish_audit(args.program, args.output, args.implementation_commit)
    else:
        result = validate_audit(args.directory)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
