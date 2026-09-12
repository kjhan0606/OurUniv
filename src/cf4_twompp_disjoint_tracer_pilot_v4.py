#!/usr/bin/env python3
"""V4 coordinate correction for the disjoint 2M++ technical input pilot.

The official ARES example states that its 2M++ angles are equatorial, and its
preparation script writes RA and DEC directly.  V3 incorrectly transformed
those angles to Galactic coordinates before querying the official HEALPix
completeness maps.  V4 corrects only that convention.  It also writes measured
gate diagnostics into the staging directory before raising on a failed gate.
No field inference or information-budget calculation is performed here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _load_frozen_v3() -> Any:
    path = Path(__file__).with_name("cf4_twompp_disjoint_tracer_pilot_v3.py")
    name = "_cf4_twompp_disjoint_tracer_pilot_frozen_v3"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct the frozen V3 module specification")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v3 = _load_frozen_v3()
base = v3.base

PROGRAM_SCHEMA = "ouruniv-cf4-twompp-disjoint-tracer-pilot-program-v4"
RESULT_SCHEMA = "ouruniv-cf4-twompp-disjoint-tracer-pilot-v4"
MANIFEST_SCHEMA = "ouruniv-cf4-twompp-disjoint-tracer-pilot-manifest-v4"
COMPLETE_SCHEMA = "ouruniv-cf4-twompp-disjoint-tracer-pilot-complete-v4"
STATUS_PASS = "PASS_TECHNICAL_INPUT_GATE_NO_FIELD_INFERENCE"
STATUS_FAIL = "FAIL_TECHNICAL_INPUT_GATE_NO_FIELD_INFERENCE"
EXPECTED_FILES = {"result.json", "manifest.json", "COMPLETE"}


def _verify_binding(binding: Mapping[str, Any], label: str) -> Path:
    path = Path(str(binding["path"]))
    if not path.is_file():
        raise base.PilotError(f"bound {label} is absent: {path}")
    if "bytes" in binding and path.stat().st_size != int(binding["bytes"]):
        raise base.PilotError(f"bound {label} size changed")
    if base.sha256_file(path) != binding["sha256"]:
        raise base.PilotError(f"bound {label} hash changed")
    return path


def load_program(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load V4 over the frozen V1 design and verify all correction evidence."""

    correction_path = Path(path)
    raw = correction_path.read_bytes()
    correction = json.loads(raw)
    if correction.get("schema") != PROGRAM_SCHEMA:
        raise base.PilotError("unexpected V4 pilot program schema")

    v1_path = _verify_binding(correction["frozen_v1_program"], "V1 program")
    _verify_binding(correction["failed_v3_program"], "V3 program")
    _verify_binding(correction["failed_v3_implementation"], "V3 implementation")
    _verify_binding(correction["v3_failure_audit"], "V3 failure audit")
    for label, binding in correction["coordinate_convention_evidence"].items():
        _verify_binding(binding, f"coordinate evidence {label}")

    program = copy.deepcopy(json.loads(v1_path.read_bytes()))
    if program.get("schema") != "ouruniv-cf4-twompp-disjoint-tracer-pilot-program-v1":
        raise base.PilotError("frozen V1 pilot program schema changed")
    angular = program["angular_completeness_gate"]
    expected_old = (
        "official map sampled in Galactic coordinates against the per-galaxy "
        "c11.5 or c12.5 catalogue mark"
    )
    if angular.get("comparison") != expected_old:
        raise base.PilotError("frozen V1 angular-coordinate contract changed")
    angular["comparison"] = (
        "official map sampled at equatorial ICRS RA/DEC in radians against the "
        "per-galaxy c11.5 or c12.5 catalogue mark"
    )
    angular["coordinate_convention_source"] = (
        "ARES examples/README.md and examples/prepare_2mpp.py at the frozen commit"
    )

    program["schema"] = PROGRAM_SCHEMA
    program["status"] = correction["status"]
    program["authorization"]["authorization_basis"] = correction[
        "authorization_basis"
    ]
    program["implementation"] = correction["implementation"]
    program["execution"]["output"] = correction["execution"]["output"]
    program["execution"]["maximum_submissions"] = 1
    program["coordinate_correction_v4"] = correction["coordinate_correction_v4"]
    return program, hashlib.sha256(raw).hexdigest()


def equatorial_directions(
    ra_degrees: np.ndarray, dec_degrees: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return HEALPix longitude/latitude from catalogue ICRS RA/DEC."""

    ra = np.asarray(ra_degrees, dtype=np.float64)
    dec = np.asarray(dec_degrees, dtype=np.float64)
    if ra.shape != dec.shape or np.any(~np.isfinite(ra)) or np.any(~np.isfinite(dec)):
        raise base.PilotError("RA/DEC arrays are incompatible or nonfinite")
    if np.any((dec < -90.0) | (dec > 90.0)):
        raise base.PilotError("declination lies outside [-90,90] degrees")
    return np.mod(np.deg2rad(ra), 2.0 * np.pi), np.deg2rad(dec)


def collect_audit(
    program: Mapping[str, Any], program_sha256: str, commit: str
) -> dict[str, Any]:
    """Collect every technical metric before assigning the pass/fail status."""

    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise base.PilotError("implementation commit must be lowercase 40-hex")
    inputs = program["inputs"]
    bound = {
        name: base.verify_file_binding(inputs[name])
        for name in (
            "twompp_catalog",
            "cf4_twompp_crossmatch",
            "cf4_twompp_crossmatch_result",
            "completeness_11_5",
            "completeness_12_5",
            "carrick_density",
            "carrick_readme",
        )
    }
    ares = base.verify_ares_repository(inputs["ares_repository"])
    exclusions, class_counts = base.read_crossmatch_exclusions(
        inputs["cf4_twompp_crossmatch"]["path"],
        int(program["no_double_counting"]["expected_unique_2Mpp_targets_excluded"]),
    )
    catalog = v3.load_catalog(inputs["twompp_catalog"]["path"])
    if catalog["recno"].size != int(program["catalog_gate"]["expected_rows"]):
        raise base.PilotError("2M++ row count changed")

    nside = int(program["angular_completeness_gate"]["HEALPix_NSIDE"])
    completeness11 = base.load_completeness_map(
        inputs["completeness_11_5"]["path"], nside
    )
    completeness12 = base.load_completeness_map(
        inputs["completeness_12_5"]["path"], nside
    )
    longitude, latitude = equatorial_directions(catalog["RA"], catalog["DEC"])
    real = np.char.lower(np.char.strip(catalog["Ref"].astype(str))) != "zoa"
    agreement11 = base.map_catalog_agreement(
        completeness11,
        longitude,
        latitude,
        catalog["c11_5"],
        real & np.isfinite(catalog["c11_5"]),
        nside,
    )
    agreement12 = base.map_catalog_agreement(
        completeness12,
        longitude,
        latitude,
        catalog["c12_5"],
        real & np.isfinite(catalog["c12_5"]),
        nside,
    )

    distance, absolute_magnitude = base.distance_and_absolute_magnitude(
        catalog["Vcmb"], catalog["Ksmag"], program["cosmology"]
    )
    eligible, reason_counts, apparent_bin, absolute_bin = (
        base.classify_disjoint_tracer(
            catalog,
            exclusions,
            distance,
            absolute_magnitude,
            program["tracer_design"],
        )
    )
    unit_vectors = base.supergalactic_unit_vectors(catalog["RA"], catalog["DEC"])
    voxels = base.voxel_summary(
        unit_vectors,
        distance,
        eligible,
        apparent_bin,
        absolute_bin,
        program["tracer_design"],
    )
    carrick = base.carrick_cube_audit(
        inputs["carrick_density"]["path"], program["carrick_reference_gate"]
    )

    angular_gate = program["angular_completeness_gate"]
    gates = {
        "ARES_commit_exact": ares["commit"] == inputs["ares_repository"]["commit"],
        "crossmatch_unique_target_count_exact": len(exclusions)
        == int(program["no_double_counting"]["expected_unique_2Mpp_targets_excluded"]),
        "completeness_11_5_median_abs_difference": agreement11[
            "median_absolute_difference"
        ]
        <= float(angular_gate["median_absolute_difference_max_inclusive"]),
        "completeness_11_5_p95_abs_difference": agreement11[
            "p95_absolute_difference"
        ]
        <= float(angular_gate["p95_absolute_difference_max_inclusive"]),
        "completeness_12_5_median_abs_difference": agreement12[
            "median_absolute_difference"
        ]
        <= float(angular_gate["median_absolute_difference_max_inclusive"]),
        "completeness_12_5_p95_abs_difference": agreement12[
            "p95_absolute_difference"
        ]
        <= float(angular_gate["p95_absolute_difference_max_inclusive"]),
        "eligible_disjoint_tracer_minimum": voxels["eligible_row_count"]
        >= int(program["catalog_gate"]["eligible_disjoint_rows_min_inclusive"]),
        "all_six_populations_nonempty": all(
            value > 0 for value in voxels["six_population_counts"].values()
        ),
        "Carrick_cube_full_finite": carrick["finite_fraction"] == 1.0,
    }
    failed = sorted(key for key, value in gates.items() if not value)
    return {
        "schema": RESULT_SCHEMA,
        "status": STATUS_PASS if not failed else STATUS_FAIL,
        "program_sha256": program_sha256,
        "implementation_commit": commit,
        "bound_inputs": bound,
        "ARES_repository": ares,
        "crossmatch": {
            "class_counts": class_counts,
            "unique_2Mpp_targets_excluded": len(exclusions),
            "policy": (
                "exclude every non-unmatched 2M++ target before density-tracer use"
            ),
        },
        "angular_completeness": {
            "coordinate_convention": "equatorial_ICRS_RA_DEC",
            "HEALPix_NSIDE": nside,
            "ordering": "RING",
            "map_11_5": {
                "minimum": float(np.min(completeness11)),
                "maximum": float(np.max(completeness11)),
                "positive_pixel_fraction": float(np.mean(completeness11 > 0.0)),
                "catalog_agreement": agreement11,
            },
            "map_12_5": {
                "minimum": float(np.min(completeness12)),
                "maximum": float(np.max(completeness12)),
                "positive_pixel_fraction": float(np.mean(completeness12 > 0.0)),
                "catalog_agreement": agreement12,
            },
        },
        "exclusive_catalog_classification": reason_counts,
        "tracer_voxel_summary": voxels,
        "Carrick_density_reference": carrick,
        "gates": gates,
        "failed_gates": failed,
        "coordinate_correction_v4": program["coordinate_correction_v4"],
        "field_inference_executed": False,
        "likelihood_datum_consumed_by_field_inference": False,
        "information_frontier_claim_allowed": False,
        "observational_resolution_claim_allowed": False,
        "next_action_requires_result_record_and_user_approval": True,
    }


def publish_pilot(
    program_path: str | Path, output: str | Path, implementation_commit: str
) -> dict[str, Any]:
    """Publish a pass atomically; preserve measured failures in staging."""

    program, program_sha = load_program(program_path)
    target = Path(output)
    stage = target.with_name(f".{target.name}.staging")
    if target.exists() or stage.exists():
        raise base.PilotError("pilot output or staging path already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    result = collect_audit(program, program_sha, implementation_commit)
    result_bytes = base.canonical_json_bytes(result)
    (stage / "result.json").write_bytes(result_bytes)
    if result["status"] != STATUS_PASS:
        failed_record = {
            "schema": "ouruniv-cf4-twompp-disjoint-tracer-pilot-failed-v4",
            "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
            "failed_gates": result["failed_gates"],
            "field_inference_executed": False,
        }
        (stage / "FAILED").write_bytes(base.canonical_json_bytes(failed_record))
        raise base.PilotError(
            f"technical pilot gate failed: {result['failed_gates']}; diagnostics preserved"
        )

    result_sha = hashlib.sha256(result_bytes).hexdigest()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "files": {"result.json": {"bytes": len(result_bytes), "sha256": result_sha}},
    }
    manifest_bytes = base.canonical_json_bytes(manifest)
    (stage / "manifest.json").write_bytes(manifest_bytes)
    complete = {
        "schema": COMPLETE_SCHEMA,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "result_sha256": result_sha,
        "status": STATUS_PASS,
    }
    (stage / "COMPLETE").write_bytes(base.canonical_json_bytes(complete))
    os.rename(stage, target)
    return validate_pilot(target)


def validate_pilot(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != EXPECTED_FILES:
        raise base.PilotError("pilot artifact file set is not exact")
    result_raw = (root / "result.json").read_bytes()
    manifest_raw = (root / "manifest.json").read_bytes()
    complete_raw = (root / "COMPLETE").read_bytes()
    result = json.loads(result_raw)
    manifest = json.loads(manifest_raw)
    complete = json.loads(complete_raw)
    if result.get("schema") != RESULT_SCHEMA or result.get("status") != STATUS_PASS:
        raise base.PilotError("pilot result status changed")
    if result.get("failed_gates") or not all(result.get("gates", {}).values()):
        raise base.PilotError("pilot result contains a failed gate")
    result_sha = hashlib.sha256(result_raw).hexdigest()
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise base.PilotError("pilot manifest schema changed")
    declared = manifest.get("files", {}).get("result.json", {})
    if declared != {"bytes": len(result_raw), "sha256": result_sha}:
        raise base.PilotError("pilot manifest does not bind result.json")
    if complete.get("schema") != COMPLETE_SCHEMA:
        raise base.PilotError("pilot COMPLETE schema changed")
    if complete.get("manifest_sha256") != manifest_sha:
        raise base.PilotError("pilot COMPLETE does not bind manifest.json")
    if complete.get("result_sha256") != result_sha or complete.get("status") != STATUS_PASS:
        raise base.PilotError("pilot COMPLETE does not bind the passing result")
    return {"status": "PASS", "result_sha256": result_sha, "gates": result["gates"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-pilot")
    run.add_argument("--program", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--implementation-commit", required=True)
    validate = subparsers.add_parser("validate-pilot")
    validate.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "run-pilot":
        result = publish_pilot(args.program, args.output, args.implementation_commit)
    else:
        result = validate_pilot(args.directory)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
