#!/usr/bin/env python3
"""Last preregistered independent 8192-draw adaptation fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cf4_peak_evidence_adaptation import (
    ROOT,
    atomic_json,
    run,
)
from cf4_all_parent_peak_evidence import sha256_file


def validate_fallback_contract(
    program: dict,
    output_path: Path,
    arrays_path: Path,
    proposal_path: Path,
) -> None:
    if program.get("status") != "frozen_before_independent_8192_adaptation_fallback":
        raise RuntimeError("fallback adaptation program is not frozen")
    if program.get("adaptation_stage") != "fallback_8192":
        raise RuntimeError("fallback adaptation stage mismatch")
    storage = program["storage"]
    actual = (output_path.resolve(), arrays_path.resolve(), proposal_path.resolve())
    canonical = (
        Path(storage["canonical_output"]).resolve(),
        Path(storage["canonical_arrays"]).resolve(),
        Path(storage["canonical_proposal"]).resolve(),
    )
    if actual != canonical:
        raise RuntimeError("fallback output path is not canonical")

    design = program["scientific_design"]
    design_path = (ROOT / design["path"]).resolve()
    if sha256_file(design_path) != design["sha256"]:
        raise RuntimeError("fallback scientific-design hash mismatch")
    with design_path.open() as stream:
        design_record = json.load(stream)
    if design_record.get("status") != design["required_status"]:
        raise RuntimeError("fallback scientific design is not frozen")
    frozen_fallback = design_record["adaptation_bank"]["fallback"]
    if int(frozen_fallback["independent_draw_count"]) != 8192 \
            or int(frozen_fallback["master_seed"]) != 2026082005 \
            or frozen_fallback.get("combine_with_first_adaptation_bank") is not False \
            or int(frozen_fallback["maximum_attempts"]) != 1:
        raise RuntimeError("fallback bank differs from the Fable design")

    prerequisite = program["fallback_prerequisite"]
    record_path = (ROOT / prerequisite["result_record"]).resolve()
    if sha256_file(record_path) != prerequisite["result_record_sha256"]:
        raise RuntimeError("2048 adaptation result-record hash mismatch")
    with record_path.open() as stream:
        record = json.load(stream)
    if record.get("status") != prerequisite["required_record_status"]:
        raise RuntimeError("2048 adaptation result record has the wrong status")
    decision = record.get("decision", {})
    if decision.get("fallback_8192_adaptation_bank_authorized") is not True \
            or decision.get("combine_or_reuse_2048_rows") is not False \
            or decision.get("additional_fallback_after_8192_authorized") is not False:
        raise RuntimeError("2048 result record did not authorize exactly one fallback")
    for key in (
        "final_proposal_frozen",
        "independent_8192_final_bank_authorized",
        "conditional_field_bank_authorized",
        "candidate_generation_authorized",
        "parent_or_seed_selection_authorized",
        "PM_or_halo_finder_authorized",
        "RAMSES_authorized",
    ):
        if decision.get(key) is not False:
            raise RuntimeError("2048 result record opened a forbidden action")

    raw_paths = {
        "result": (
            Path(record["lineage"]["canonical_result"]),
            prerequisite["canonical_result_sha256"],
        ),
        "arrays": (
            Path(record["lineage"]["canonical_arrays"]),
            prerequisite["canonical_arrays_sha256"],
        ),
        "COMPLETE": (
            Path(record["lineage"]["complete_marker"]),
            prerequisite["complete_marker_sha256"],
        ),
    }
    for label, (path, expected_hash) in raw_paths.items():
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"2048 canonical {label} lineage mismatch")
    with raw_paths["result"][0].open() as stream:
        raw_result = json.load(stream)
    raw_decision = raw_result.get("decision", {})
    if raw_result.get("status") != prerequisite["required_raw_status"] \
            or raw_decision.get("failure_class") != prerequisite["required_failure_class"] \
            or raw_decision.get("fallback_8192_adaptation_bank_authorized") is not True \
            or raw_decision.get("final_proposal_frozen") is not False \
            or raw_result.get("lineage", {}).get("proposal") is not None:
        raise RuntimeError("2048 raw result does not satisfy fallback prerequisites")
    old_proposal = Path(prerequisite["forbidden_2048_proposal_path"])
    if old_proposal.exists():
        raise RuntimeError("2048 adaptation unexpectedly has a proposal artifact")


def finalize_fallback_result(result: dict, program: dict) -> dict:
    passed = bool(result["gates"]["adaptation_pass"])
    result["schema"] = "ouruniv-cf4-peak-evidence-adaptation-fallback-result-v1"
    result["status"] = (
        "complete_pass_freeze_defensive_final_proposal_from_fallback"
        if passed else "complete_fail_fallback_adaptation"
    )
    result["adaptation_stage"] = "fallback_8192"
    result["summary"]["master_seed"] = int(
        program["adaptation_bank"]["master_seed"]
    )
    decision = result["decision"]
    decision["fallback_8192_adaptation_bank_authorized"] = False
    decision["additional_adaptation_fallback_authorized"] = False
    if not passed and not result["gates"]["geometry_integration_support"] \
            and result["gates"]["all_parent_lineage"] \
            and result["gates"]["all_log_Z_and_importance_finite"] \
            and result["gates"]["real_evidence_scalar_vectorized_control"]:
        decision["failure_class"] = "insufficient_fallback_adaptation_support_stop"
    return result


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
    validate_fallback_contract(program, output_path, arrays_path, proposal_path)
    for item in program["pinned_local_files"]:
        path = (ROOT / item["path"]).resolve()
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"local hash mismatch: {item['path']}")

    result = finalize_fallback_result(
        run(program, arrays_path, proposal_path), program
    )
    result["lineage"].update({
        "program": str(program_path),
        "program_sha256": sha256_file(program_path),
        "fallback_implementation_sha256": program["implementation"]["sha256"],
        "adaptation_core_sha256": program["adaptation_core"]["sha256"],
        "proposal_implementation_sha256": program[
            "proposal_implementation"
        ]["sha256"],
        "scientific_design_sha256": program["scientific_design"]["sha256"],
        "fallback_prerequisite_result_sha256": program[
            "fallback_prerequisite"
        ]["canonical_result_sha256"],
        "density_filter_sha256": program["density_filter"]["sha256"],
        "reference_calibration_sha256": program["reference_calibration"]["sha256"],
    })
    atomic_json(output_path, result)
    print(f"[fallback-adaptation] status={result['status']}", flush=True)


if __name__ == "__main__":
    main()
