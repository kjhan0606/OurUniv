import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "config/cf4_datum_bearing_z0_twompp_datum_v3_result_record.json"


def test_result_record_binds_published_artifacts_and_lineage():
    record = json.loads(RECORD.read_text())
    assert record["status"] == "PASS_PHASE_A_ACTUAL_COUNT_DATUM_PUBLISHED_STOP_BEFORE_PHASE_B"
    for binding in record["implementation_lineage"].values():
        if not isinstance(binding, dict):
            continue
        path = ROOT / binding["path"]
        assert path.stat().st_size == binding["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    for binding in record["published_artifacts"].values():
        if not isinstance(binding, dict):
            continue
        path = Path(record["published_artifacts"]["directory"]) / next(
            key for key, value in record["published_artifacts"].items() if value is binding
        )
        assert path.stat().st_size == binding["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]


def test_phase_a_gate_and_count_contract_passed_exactly():
    record = json.loads(RECORD.read_text())
    gate = record["gate_result"]
    assert gate["passed_gate_count"] == gate["total_gate_count"] == 12
    assert gate["failed_gates"] == []
    assert sum(gate["population_all"]) == gate["retained_total"] == 36635
    assert gate["unique_recno_total"] == 36635
    assert sum(gate["population_train"]) == gate["train_row_count"] == 29257
    assert sum(gate["population_holdout"]) == gate["holdout_row_count"] == 7378
    assert gate["counts_all_equals_train_plus_holdout"] is True
    assert gate["row_manifest_reconstructs_all_count_arrays"] is True
    assert gate["count_dtype"] == "int64"
    assert gate["count_shape"] == [6, 32, 32, 32]
    assert gate["positive_count_nonpositive_exposure_count"] == 0


def test_selection_is_raw_order6_and_claims_remain_bounded():
    record = json.loads(RECORD.read_text())
    selection = record["selection"]
    assert selection["quadrature_order_per_axis"] == 6
    assert selection["subpoints_per_voxel"] == 216
    assert selection["epsilon_floor_used"] is False
    assert selection["occupied_voxel_override_used"] is False
    assert selection["final_selection_convergence_claim"] is False
    science = record["scientific_disposition"]
    assert science["actual_2Mpp_integer_count_datum"] == "CREATED_AND_PUBLISHED"
    assert science["present_density_posterior"] == "NOT_CREATED"
    assert science["IC_posterior"] == "NOT_CREATED"
    assert science["observational_resolution_claim"] == "NOT_ALLOWED"
    assert science["numerical_0p3_cMpc_h_claim"] == "NOT_ALLOWED"
    assert science["Phase_A"] == "PASS"
    assert science["Phase_B"] == "NOT_AUTHORIZED_BY_THIS_RESULT"


def test_execution_was_slurm_only_and_no_follow_on_occurred():
    execution = json.loads(RECORD.read_text())["execution"]
    assert execution["Slurm_job_id"] == 329833
    assert execution["state"] == "COMPLETED"
    assert execution["exit_code"] == "0:0"
    assert execution["requested_memory_MiB"] >= 1.2 * execution["expected_peak_memory_MiB"]
    assert execution["manual_execution_on_syntax_or_syn101"] is False
    assert execution["automatic_follow_on_executed"] is False
