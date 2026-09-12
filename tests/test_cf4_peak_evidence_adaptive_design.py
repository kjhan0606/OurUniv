import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_design():
    return json.loads((
        ROOT / "config/cf4_peak_evidence_adaptive_integration_design.json"
    ).read_text())


def test_adaptive_design_is_pinned_to_the_failed_v1_result():
    design = load_design()
    assert design["status"] == "frozen_Fable_scientific_design_before_implementation"
    authorization = design["authorization"]
    record = ROOT / authorization["source_result_record"]
    assert sha256_file(record) == authorization["source_result_record_sha256"]
    result = json.loads(record.read_text())
    assert result["status"] == authorization["source_result_record_required_status"]
    assert authorization["adaptation_execution_authorized"] is True
    assert authorization["final_execution_authorized_before_adaptation_pass"] is False


def test_adaptation_and_final_banks_are_independent_and_firewalled():
    design = load_design()
    adaptation = design["adaptation_bank"]
    final = design["final_bank"]
    assert adaptation["draw_count"] == 2048
    assert adaptation["master_seed"] == 2026082001
    assert final["draw_count"] == 8192
    assert final["master_seed"] == 2026082003
    assert adaptation["master_seed"] != final["master_seed"]
    firewall = design["information_firewall"]
    assert firewall["V1_64_draw_values_used_for_new_fit_or_initialization"] is False
    assert firewall["adaptation_rows_reused_in_final_scientific_gate"] is False
    assert firewall["final_rows_reused_for_proposal_training"] is False
    assert firewall["physical_peak_target_or_sigma_change_allowed"] is False
    forbidden = " ".join(firewall["adaptation_fitter_forbidden_input"])
    assert "parent seed" in forbidden
    assert "CF4 deviance" in forbidden


def test_proposal_and_em_hyperparameters_are_fully_frozen():
    design = load_design()
    component = design["adaptive_component_density"]
    assert component["component_count"] == 4
    assert component["midpoint_covariance"] == "full 3x3"
    assert component["midpoint_covariance_eigenvalue_min_mpc_h_squared"] == 0.75**2
    assert component["midpoint_covariance_eigenvalue_max_mpc_h_squared"] == 6.0**2
    assert component["axis_density"]["kappa_max"] == 20.0
    em = design["weighted_EM"]
    assert em["restart_count"] == 8
    assert em["maximum_iterations"] == 200
    assert em["master_seed"] == 2026082002
    assert "<4" in em["empty_component_failure"]
    assert em["resurrection_merge_or_K_change_allowed"] is False
    assert design["final_proposal"]["analytic_target_over_proposal_bound"] == 2.0


def test_adaptation_and_final_gates_cannot_be_relaxed():
    design = load_design()
    adaptation = design["adaptation_bank"]["gate"]
    assert adaptation["geometry_marginal_ESS_min"] == 128.0
    assert adaptation["maximum_normalized_geometry_weight_max"] == 0.025
    assert adaptation["normalized_log_Z_row_count"] == 256 * 2048
    fallback = design["adaptation_bank"]["fallback"]
    assert fallback["independent_draw_count"] == 8192
    assert fallback["combine_with_first_adaptation_bank"] is False
    assert fallback["maximum_attempts"] == 1
    final = design["final_bank"]["gates"]
    assert final["first_second_4096_parent_weight_L1_max"] == 0.20
    assert final["joint_parent_geometry_ESS_min"] == 128.0
    assert final["geometry_marginal_ESS_min"] == 128.0
    assert final["parent_ESS_min"] == 32.0
    assert final["weighted_CF4_one_sided_KS_permutation_p_min"] == 0.01


def test_all_downstream_field_and_simulation_authorizations_remain_closed():
    decision = load_design()["authorization"]
    assert decision["conditional_field_bank_authorized"] is False
    assert decision["candidate_generation_authorized"] is False
    assert decision["parent_or_seed_selection_authorized"] is False
    assert decision["PM_or_halo_finder_authorized"] is False
    assert decision["RAMSES_authorized"] is False


def test_2048_adaptation_result_record_authorizes_only_the_frozen_fallback():
    record = json.loads((
        ROOT / "config/cf4_peak_evidence_adaptation_v1_result_record.json"
    ).read_text())
    assert record["status"] == "complete_fail_insufficient_adaptation_support"
    assert record["gates"]["all_parent_lineage"] is True
    assert record["gates"]["all_log_Z_and_importance_finite"] is True
    assert record["gates"]["real_evidence_scalar_vectorized_control"] is True
    assert record["gates"]["geometry_integration_support"] is False
    decision = record["decision"]
    assert decision["fallback_8192_adaptation_bank_authorized"] is True
    assert decision["combine_or_reuse_2048_rows"] is False
    assert decision["additional_fallback_after_8192_authorized"] is False
    assert decision["final_proposal_frozen"] is False
    assert decision["independent_8192_final_bank_authorized"] is False
    for key in (
        "conditional_field_bank_authorized",
        "candidate_generation_authorized",
        "parent_or_seed_selection_authorized",
        "PM_or_halo_finder_authorized",
        "RAMSES_authorized",
    ):
        assert decision[key] is False
    for path_key, hash_key in (
        ("canonical_result", "canonical_result_sha256"),
        ("canonical_arrays", "canonical_arrays_sha256"),
        ("complete_marker", "complete_marker_sha256"),
    ):
        path = Path(record["lineage"][path_key])
        if path.exists():
            assert sha256_file(path) == record["lineage"][hash_key]
