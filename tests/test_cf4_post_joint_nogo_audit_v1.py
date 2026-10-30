import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "config/cf4_post_joint_nogo_science_gate_and_exact_covariance_audit_v1.json"
JOINT_RESULT = ROOT / "config/cf4_twompp_joint_information_budget_pilot_v1_result_record.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_gate_correction_is_explicitly_outcome_aware_and_non_destructive():
    audit = _load(AUDIT)
    policy = audit["gate_provenance_audit"]["correction_policy"]
    authorization = audit["authorization"]

    assert audit["status"] == "AUDIT_COMPLETE_GATE_ERROR_FOUND_EXACT_COVARIANCE_PILOT_RECOMMENDED"
    assert policy["outcome_aware_correction"] is True
    assert policy["blind_preregistration_claim_allowed"] is False
    assert policy["old_program_or_result_rewrite_allowed"] is False
    assert audit["final_audit_decision"]["old_0p8_NO_GO_preserved"] is True
    assert authorization["new_Slurm_submission"] is False
    assert authorization["observational_CF4_field_inference"] is False
    assert authorization["untouched_256_mock_validation"] is False


def test_covariance_gate_identities_and_hierarchy_are_coherent():
    audit = _load(AUDIT)
    inconsistency = audit["gate_provenance_audit"]["inconsistency_in_covariance_gate"]
    hierarchy = audit["gate_provenance_audit"]["recommended_gate_hierarchy"]

    assert math.isclose(inconsistency["correlation_r_min_0p7_implies_I_min"], 0.7**2)
    assert math.isclose(inconsistency["residual_power_ratio_max_0p5_implies_I_min"], 0.5)
    assert math.isclose(inconsistency["I_min_0p8_implies_correlation_r_min"], math.sqrt(0.8))
    assert math.isclose(inconsistency["I_min_0p8_implies_residual_power_ratio_max"], 0.2)
    assert hierarchy["development_material_constraint"]["information_numerical_lower_min_inclusive"] == 0.5
    assert hierarchy["strong_reconstruction_stretch_goal"]["information_and_lower_min_inclusive"] == 0.8
    assert hierarchy["strong_reconstruction_stretch_goal"]["route_killing_hard_gate"] is False


def test_joint_result_values_are_reclassified_without_mutating_original_status():
    audit = _load(AUDIT)
    original = _load(JOINT_RESULT)
    reclassified = audit["joint_pilot_reclassification"]
    known = reclassified["lowest_bin"]["known_selection_reference_bias"]
    marginalized = reclassified["lowest_bin"]["normalization_marginalized_reference_bias"]

    assert reclassified["preserved_original_status"] == original["status"]
    source_known = original["lowest_bin_information"]["known_selection_reference_bias_ceiling"]
    source_marginalized = original["lowest_bin_information"]["normalization_marginalized_reference_bias"]
    assert known["information"] == source_known["recovered_information_fraction"]
    assert known["numerical_95_lower"] == source_known["numerical_95_lower"]
    assert marginalized["information"] == source_marginalized["recovered_information_fraction"]
    assert marginalized["numerical_95_lower"] == source_marginalized["numerical_95_lower"]
    assert known["original_0p8_stretch_gate"] is False
    assert known["corrected_0p5_material_gate"] is True
    assert marginalized["corrected_0p5_material_gate"] is False


def test_next_pilot_is_single_geometry_matrix_free_and_not_pre_authorized():
    audit = _load(AUDIT)
    covariance = audit["exact_velocity_covariance_audit"]
    pilot = audit["minimum_cost_next_pilot"]

    assert covariance["stored_draw_limit"]["sample_covariance_rank_max"] == 15
    assert covariance["operator_recovery"]["all_inputs_available_without_new_truth"] is True
    assert covariance["matrix_free_implementation"]["materialize_full_covariance"] is False
    assert covariance["matrix_free_implementation"]["mathematical_upper_bound_from_trace_only_available"] is False
    assert pilot["authorization_required_before_submission"] is True
    assert pilot["design"]["geometry_index"] == 0
    assert pilot["design"]["new_truth_or_validation_seed_count"] == 0
    assert pilot["execution_envelope"]["minimum_requested_memory_MiB"] >= math.ceil(
        1.2 * pilot["execution_envelope"]["expected_peak_memory_MiB"]
    )
