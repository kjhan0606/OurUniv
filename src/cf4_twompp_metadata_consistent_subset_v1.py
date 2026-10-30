#!/usr/bin/env python3
"""Freeze and validate the metadata-consistent CF4-disjoint 2M++ subset.

The exclusion set is exactly the 319 rows in the hash-bound failed outlier
audit.  This module neither adapts the threshold nor examines a density-field
outcome.  It validates positive exposure and six-population support before any
count likelihood or joint information calculation is permitted.
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


def _load_frozen_outlier_audit() -> Any:
    path = Path(__file__).with_name("cf4_twompp_completeness_outlier_audit_v1.py")
    name = "_cf4_twompp_completeness_outlier_audit_frozen_v1_for_subset"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct frozen outlier-audit module specification")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


outlier_audit = _load_frozen_outlier_audit()
v4 = outlier_audit.v4
base = outlier_audit.base

PROGRAM_SCHEMA = "ouruniv-cf4-twompp-metadata-consistent-subset-program-v1"
RESULT_SCHEMA = "ouruniv-cf4-twompp-metadata-consistent-subset-v1"
MANIFEST_SCHEMA = "ouruniv-cf4-twompp-metadata-consistent-subset-manifest-v1"
COMPLETE_SCHEMA = "ouruniv-cf4-twompp-metadata-consistent-subset-complete-v1"
STATUS_PASS = "PASS_METADATA_CONSISTENT_36635_TRACER_SUBSET_NO_FIELD_INFERENCE"
STATUS_FAIL = "FAIL_METADATA_CONSISTENT_TRACER_SUBSET_NO_FIELD_INFERENCE"
EXPECTED_FILES = {"result.json", "excluded_recnos.csv", "manifest.json", "COMPLETE"}


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
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path, str]:
    raw = Path(path).read_bytes()
    program = json.loads(raw)
    if program.get("schema") != PROGRAM_SCHEMA:
        raise base.PilotError("unexpected metadata-consistent subset program schema")
    if not program.get("authorization", {}).get("subset_validation", False):
        raise base.PilotError("metadata-consistent subset validation is not authorized")
    parent_program_path = _verify_binding(
        program["parent_outlier_program"], "outlier program"
    )
    _verify_binding(program["parent_outlier_implementation"], "outlier implementation")
    failure_record_path = _verify_binding(
        program["parent_failure_result_record"], "outlier failure result record"
    )
    failure_result_path = _verify_binding(
        program["parent_failure_result"], "outlier failure result"
    )
    outliers_path = _verify_binding(program["frozen_outliers"], "frozen outlier CSV")
    _verify_binding(program["parent_FAILED_marker"], "outlier FAILED marker")
    _, effective, _, _ = outlier_audit.load_program(parent_program_path)
    failure_record = json.loads(failure_record_path.read_bytes())
    failure_result = json.loads(failure_result_path.read_bytes())
    if failure_record.get("status") != (
        "NO_GO_UNFILTERED_COUNT_LIKELIHOOD_ZERO_EXPOSURE_GATE_FAILED"
    ):
        raise base.PilotError("parent failure result record status changed")
    if failure_result.get("status") != outlier_audit.STATUS_FAIL:
        raise base.PilotError("parent outlier result is not the frozen failure")
    if failure_result.get("failed_gates") != ["eligible_zero_exposure_count_zero"]:
        raise base.PilotError("parent failed-gate set changed")
    return (
        program,
        effective,
        failure_result,
        outliers_path,
        hashlib.sha256(raw).hexdigest(),
    )


def read_frozen_exclusions(
    path: str | Path, expected_count: int
) -> tuple[list[int], dict[int, dict[str, str]]]:
    rows: dict[int, dict[str, str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "recno",
            "absolute_difference",
            "zero_exposure",
            "assigned_map",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise base.PilotError("frozen outlier CSV header changed")
        for row in reader:
            recno = int(row["recno"])
            if recno in rows:
                raise base.PilotError("frozen outlier recno is not unique")
            difference = float(row["absolute_difference"])
            zero = row["zero_exposure"].strip().lower() == "true"
            if not (difference > 0.05 or zero):
                raise base.PilotError("frozen outlier row violates its selection rule")
            rows[recno] = row
    if len(rows) != expected_count:
        raise base.PilotError("frozen outlier count changed")
    recnos = sorted(rows)
    return recnos, rows


def canonical_exclusion_csv_bytes(recnos: list[int], parent_sha256: str) -> bytes:
    handle = io.StringIO(newline="")
    fields = ["recno", "reason", "parent_outliers_sha256"]
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for recno in recnos:
        writer.writerow(
            {
                "recno": recno,
                "reason": "metadata_completeness_absdiff_gt_0p05_or_zero_exposure",
                "parent_outliers_sha256": parent_sha256,
            }
        )
    return handle.getvalue().encode("utf-8")


def collect_validation(
    program: Mapping[str, Any],
    effective: Mapping[str, Any],
    parent_result: Mapping[str, Any],
    outliers_path: Path,
    program_sha256: str,
    commit: str,
) -> tuple[dict[str, Any], bytes]:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise base.PilotError("implementation commit must be lowercase 40-hex")
    frozen = program["frozen_subset"]
    recnos, frozen_rows = read_frozen_exclusions(
        outliers_path, int(frozen["excluded_recno_count_exact"])
    )
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
    frozen_mask = np.isin(catalog["recno"], np.asarray(recnos, dtype=np.int64))
    retained = eligible & ~frozen_mask

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
    exposure = np.where(
        apparent_bin == 0, completeness11[pixels], completeness12[pixels]
    )
    mark = np.where(apparent_bin == 0, catalog["c11_5"], catalog["c12_5"])
    difference = np.abs(exposure - mark)
    threshold = float(frozen["large_absolute_difference_strictly_greater_than"])
    retained_populations = outlier_audit._population_counts(
        retained,
        apparent_bin,
        absolute_bin,
        effective["tracer_design"]["absolute_K_edges"],
    )
    expected_populations = {
        key: int(value) for key, value in frozen["retained_six_population_counts"].items()
    }
    excluded_catalog_count = int(np.count_nonzero(frozen_mask))
    excluded_eligible_count = int(np.count_nonzero(frozen_mask & eligible))
    retained_count = int(np.count_nonzero(retained))
    retained_zero = int(np.count_nonzero(retained & (exposure <= 0.0)))
    retained_large = int(np.count_nonzero(retained & (difference > threshold)))
    excluded_rule_reproduced = all(
        (
            difference[index] > threshold
            or exposure[index] <= 0.0
        )
        for index in np.flatnonzero(frozen_mask & eligible)
    )
    gates = {
        "parent_failure_exact": parent_result["failed_gates"]
        == ["eligible_zero_exposure_count_zero"],
        "excluded_recno_count_exact": len(recnos)
        == int(frozen["excluded_recno_count_exact"]),
        "every_excluded_recno_present_once_in_catalog": excluded_catalog_count == len(recnos),
        "every_excluded_recno_was_parent_eligible": excluded_eligible_count == len(recnos),
        "excluded_rule_exactly_reproduced": excluded_rule_reproduced,
        "retained_count_exact": retained_count
        == int(frozen["retained_tracer_count_exact"]),
        "retained_zero_exposure_count_zero": retained_zero == 0,
        "retained_large_difference_count_zero": retained_large == 0,
        "retained_six_population_counts_exact": retained_populations
        == expected_populations,
        "retained_all_six_populations_nonempty": all(
            count > 0 for count in retained_populations.values()
        ),
    }
    failed = sorted(key for key, value in gates.items() if not value)
    exclusion_bytes = canonical_exclusion_csv_bytes(
        recnos, program["frozen_outliers"]["sha256"]
    )
    result = {
        "schema": RESULT_SCHEMA,
        "status": STATUS_PASS if not failed else STATUS_FAIL,
        "program_sha256": program_sha256,
        "implementation_commit": commit,
        "parent_outliers_sha256": program["frozen_outliers"]["sha256"],
        "catalog_partition_before_metadata_exclusion": reason_counts,
        "metadata_consistency_exclusion": {
            "excluded_recno_count": len(recnos),
            "excluded_catalog_count": excluded_catalog_count,
            "excluded_parent_eligible_count": excluded_eligible_count,
            "zero_exposure_parent_rows_excluded": sum(
                row["zero_exposure"].strip().lower() == "true"
                for row in frozen_rows.values()
            ),
            "canonical_manifest": "excluded_recnos.csv",
            "density_outcomes_used_for_exclusion": False,
        },
        "retained_subset": {
            "count": retained_count,
            "zero_exposure_count": retained_zero,
            "large_difference_count": retained_large,
            "assigned_exposure_quantiles": outlier_audit._quantiles(exposure[retained]),
            "assigned_absolute_difference_quantiles": outlier_audit._quantiles(
                difference[retained]
            ),
            "six_population_counts": retained_populations,
        },
        "gates": gates,
        "failed_gates": failed,
        "likelihood_rows_consumed": 0,
        "field_inference_executed": False,
        "joint_information_budget_executed": False,
        "observational_resolution_claim_allowed": False,
        "next_action": (
            "implement_frozen_N32_joint_velocity_density_information_budget"
            if not failed
            else "stop_before_joint_information_budget"
        ),
    }
    return result, exclusion_bytes


def publish_validation(
    program_path: str | Path, output: str | Path, implementation_commit: str
) -> dict[str, Any]:
    program, effective, parent_result, outliers_path, program_sha = load_program(
        program_path
    )
    target = Path(output)
    stage = target.with_name(f".{target.name}.staging")
    if target.exists() or stage.exists():
        raise base.PilotError("subset output or staging path already exists")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage.mkdir()
    result, exclusions = collect_validation(
        program,
        effective,
        parent_result,
        outliers_path,
        program_sha,
        implementation_commit,
    )
    result_bytes = base.canonical_json_bytes(result)
    (stage / "result.json").write_bytes(result_bytes)
    (stage / "excluded_recnos.csv").write_bytes(exclusions)
    if result["status"] != STATUS_PASS:
        failure = {
            "schema": "ouruniv-cf4-twompp-metadata-consistent-subset-failed-v1",
            "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
            "excluded_recnos_sha256": hashlib.sha256(exclusions).hexdigest(),
            "failed_gates": result["failed_gates"],
            "field_inference_executed": False,
        }
        (stage / "FAILED").write_bytes(base.canonical_json_bytes(failure))
        raise base.PilotError(
            f"metadata-consistent subset gate failed: {result['failed_gates']}; diagnostics preserved"
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "files": {
            "result.json": {
                "bytes": len(result_bytes),
                "sha256": hashlib.sha256(result_bytes).hexdigest(),
            },
            "excluded_recnos.csv": {
                "bytes": len(exclusions),
                "sha256": hashlib.sha256(exclusions).hexdigest(),
            },
        },
    }
    manifest_bytes = base.canonical_json_bytes(manifest)
    (stage / "manifest.json").write_bytes(manifest_bytes)
    complete = {
        "schema": COMPLETE_SCHEMA,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "excluded_recnos_sha256": hashlib.sha256(exclusions).hexdigest(),
        "status": STATUS_PASS,
    }
    (stage / "COMPLETE").write_bytes(base.canonical_json_bytes(complete))
    os.rename(stage, target)
    return validate_subset(target)


def validate_subset(directory: str | Path) -> dict[str, Any]:
    root = Path(directory)
    if not root.is_dir() or {path.name for path in root.iterdir()} != EXPECTED_FILES:
        raise base.PilotError("subset artifact file set is not exact")
    raw = {name: (root / name).read_bytes() for name in EXPECTED_FILES}
    result = json.loads(raw["result.json"])
    manifest = json.loads(raw["manifest.json"])
    complete = json.loads(raw["COMPLETE"])
    if result.get("schema") != RESULT_SCHEMA or result.get("status") != STATUS_PASS:
        raise base.PilotError("subset result is not passing")
    if result.get("failed_gates") or not all(result.get("gates", {}).values()):
        raise base.PilotError("subset result contains a failed gate")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise base.PilotError("subset manifest schema changed")
    for name in ("result.json", "excluded_recnos.csv"):
        expected = {
            "bytes": len(raw[name]),
            "sha256": hashlib.sha256(raw[name]).hexdigest(),
        }
        if manifest.get("files", {}).get(name) != expected:
            raise base.PilotError(f"subset manifest does not bind {name}")
    if complete.get("schema") != COMPLETE_SCHEMA:
        raise base.PilotError("subset COMPLETE schema changed")
    if complete.get("manifest_sha256") != hashlib.sha256(raw["manifest.json"]).hexdigest():
        raise base.PilotError("subset COMPLETE does not bind manifest")
    if complete.get("result_sha256") != hashlib.sha256(raw["result.json"]).hexdigest():
        raise base.PilotError("subset COMPLETE does not bind result")
    if complete.get("excluded_recnos_sha256") != hashlib.sha256(
        raw["excluded_recnos.csv"]
    ).hexdigest():
        raise base.PilotError("subset COMPLETE does not bind exclusion manifest")
    if complete.get("status") != STATUS_PASS:
        raise base.PilotError("subset COMPLETE status changed")
    return {
        "status": "PASS",
        "result_sha256": hashlib.sha256(raw["result.json"]).hexdigest(),
        "excluded_recnos_sha256": hashlib.sha256(
            raw["excluded_recnos.csv"]
        ).hexdigest(),
        "gates": result["gates"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run-validation")
    run.add_argument("--program", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--implementation-commit", required=True)
    validate = subparsers.add_parser("validate-subset")
    validate.add_argument("--directory", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "run-validation":
        result = publish_validation(args.program, args.output, args.implementation_commit)
    else:
        result = validate_subset(args.directory)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
