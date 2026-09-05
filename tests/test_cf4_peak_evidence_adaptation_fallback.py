import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from cf4_peak_evidence_adaptation_fallback import (
    finalize_fallback_result,
    validate_fallback_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_program():
    return json.loads((
        ROOT / "config/cf4_peak_evidence_adaptation_fallback_program.json"
    ).read_text())


def test_fallback_program_is_hash_pinned_to_the_failed_2048_bank():
    program = load_program()
    for item in program["pinned_local_files"]:
        assert sha256_file(ROOT / item["path"]) == item["sha256"]
    prerequisite = program["fallback_prerequisite"]
    record_path = ROOT / prerequisite["result_record"]
    assert sha256_file(record_path) == prerequisite["result_record_sha256"]
    record = json.loads(record_path.read_text())
    assert record["status"] == prerequisite["required_record_status"]
    assert record["lineage"]["canonical_result_sha256"] \
        == prerequisite["canonical_result_sha256"]
    assert record["lineage"]["canonical_arrays_sha256"] \
        == prerequisite["canonical_arrays_sha256"]
    assert record["lineage"]["complete_marker_sha256"] \
        == prerequisite["complete_marker_sha256"]
    assert program["implementation"]["imported_adaptation_core"] \
        == program["adaptation_core"]


def test_fallback_contract_uses_canonical_paths_when_gpfs_inputs_exist():
    program = load_program()
    prerequisite = program["fallback_prerequisite"]
    record = json.loads((ROOT / prerequisite["result_record"]).read_text())
    required_paths = [
        Path(record["lineage"]["canonical_result"]),
        Path(record["lineage"]["canonical_arrays"]),
        Path(record["lineage"]["complete_marker"]),
    ]
    if not all(path.exists() for path in required_paths):
        pytest.skip("sealed GPFS prerequisite is unavailable")
    storage = program["storage"]
    validate_fallback_contract(
        program,
        Path(storage["canonical_output"]),
        Path(storage["canonical_arrays"]),
        Path(storage["canonical_proposal"]),
    )
    draft = deepcopy(program)
    draft["status"] = "draft"
    with pytest.raises(RuntimeError, match="not frozen"):
        validate_fallback_contract(
            draft,
            Path(storage["canonical_output"]),
            Path(storage["canonical_arrays"]),
            Path(storage["canonical_proposal"]),
        )
    missing_core = deepcopy(program)
    del missing_core["implementation"]["imported_adaptation_core"]
    with pytest.raises(RuntimeError, match="numeric core"):
        validate_fallback_contract(
            missing_core,
            Path(storage["canonical_output"]),
            Path(storage["canonical_arrays"]),
            Path(storage["canonical_proposal"]),
        )


def base_result():
    return {
        "schema": "old",
        "status": "old",
        "summary": {},
        "gates": {
            "adaptation_pass": False,
            "all_parent_lineage": True,
            "all_log_Z_and_importance_finite": True,
            "real_evidence_scalar_vectorized_control": True,
            "geometry_integration_support": False,
        },
        "decision": {
            "failure_class": "insufficient_adaptation_support_fallback_authorized",
            "fallback_8192_adaptation_bank_authorized": True,
            "additional_adaptation_fallback_authorized": True,
        },
    }


def test_fallback_support_failure_stops_without_recursive_authorization():
    result = finalize_fallback_result(base_result(), load_program())
    assert result["status"] == "complete_fail_fallback_adaptation"
    assert result["adaptation_stage"] == "fallback_8192"
    assert result["summary"]["master_seed"] == 2026082005
    decision = result["decision"]
    assert decision["failure_class"] == "insufficient_fallback_adaptation_support_stop"
    assert decision["fallback_8192_adaptation_bank_authorized"] is False
    assert decision["additional_adaptation_fallback_authorized"] is False


def test_fallback_numerical_failure_remains_invalid_without_fallback():
    value = base_result()
    value["gates"]["real_evidence_scalar_vectorized_control"] = False
    value["decision"]["failure_class"] = "invalid_numerical_or_lineage"
    result = finalize_fallback_result(value, load_program())
    assert result["decision"]["failure_class"] == "invalid_numerical_or_lineage"
    assert result["decision"]["fallback_8192_adaptation_bank_authorized"] is False
    assert result["decision"]["additional_adaptation_fallback_authorized"] is False


def test_fallback_pass_only_freezes_the_final_proposal():
    value = base_result()
    value["gates"]["adaptation_pass"] = True
    value["gates"]["geometry_integration_support"] = True
    value["decision"].update({
        "failure_class": None,
        "final_proposal_frozen": True,
        "independent_8192_final_bank_authorized": True,
        "conditional_field_bank_authorized": False,
        "candidate_generation_authorized": False,
        "parent_or_seed_selection_authorized": False,
        "PM_or_RAMSES_authorized": False,
    })
    result = finalize_fallback_result(value, load_program())
    assert result["status"] \
        == "complete_pass_freeze_defensive_final_proposal_from_fallback"
    assert result["decision"]["final_proposal_frozen"] is True
    assert result["decision"]["independent_8192_final_bank_authorized"] is True
    assert result["decision"]["additional_adaptation_fallback_authorized"] is False


def test_fallback_bank_is_independent_last_chance_with_unchanged_gates():
    program = load_program()
    bank = program["adaptation_bank"]
    assert bank["draw_count"] == 8192
    assert bank["master_seed"] == 2026082005
    assert bank["reuse_or_combine_any_2048_geometry"] is False
    assert bank["additional_fallback_after_this_bank"] is False
    gates = program["gates"]
    assert gates["all_2097152_log_Z_finite"] is True
    assert gates["geometry_marginal_ESS_min"] == 128.0
    assert gates["maximum_normalized_geometry_weight_max"] == 0.025
    firewall = program["information_firewall"]
    assert firewall["independent_of_2048_rows"] is True
    assert firewall["combine_or_reuse_2048_rows"] is False
    assert firewall["conditional_or_candidate_field_generated"] is False
    assert firewall["PM_or_halo_finder_run"] is False
    assert firewall["RAMSES_authorized"] is False


def test_fallback_scripts_are_marker_driven_hash_pinned_and_bounded():
    runner = (ROOT / "scripts/run_cf4_peak_evidence_adaptation_fallback_lageunha.sh").read_text()
    launcher = (ROOT / "scripts/launch_cf4_peak_evidence_adaptation_fallback_lageunha.sh").read_text()
    status = (ROOT / "scripts/status_cf4_peak_evidence_adaptation_fallback.sh").read_text()
    combined = "\n".join((runner, launcher, status)).lower()
    assert "expected_program_sha=" in runner
    assert "expected_implementation_sha=" in runner
    assert "expected_core_sha=" in runner
    assert "expected_proposal_implementation_sha=" in runner
    assert "adaptation_stage=fallback_8192" in runner
    assert "runner_sha256=%s" in runner
    assert "fallback result opened a recursive adaptation" in runner
    assert "worker_processes=8" in runner
    assert "threads_per_worker=1" in runner
    assert "pgrep" not in combined
    assert "postgres" not in combined
    assert "while " not in combined
    assert "sleep " not in combined
