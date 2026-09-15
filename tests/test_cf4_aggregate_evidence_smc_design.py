import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_design():
    return json.loads((
        ROOT / "config/cf4_aggregate_evidence_annealed_smc_design.json"
    ).read_text())


def test_smc_design_is_pinned_to_the_closed_iid_architecture():
    design = load_design()
    assert design["status"] == "frozen_scientific_design_before_implementation"
    source = design["source_failure"]
    record_path = ROOT / source["result_record"]
    assert sha256_file(record_path) == source["result_record_sha256"]
    record = json.loads(record_path.read_text())
    assert record["status"] == source["required_status"]
    assert record["decision"]["fixed_proposal_IID_adaptation_architecture_closed"]
    assert record["decision"]["additional_IID_adaptation_bank_authorized"] is False
    assert record["decision"]["retroactive_gate_relaxation_authorized"] is False


def test_physical_model_and_exact_operator_are_unchanged():
    design = load_design()
    fixed = design["fixed_inputs"]
    model_path = ROOT / fixed["physical_model"]["path"]
    assert sha256_file(model_path) == fixed["physical_model"]["sha256"]
    assert fixed["parent_count"] == 256
    assert fixed["parent_seed_range_inclusive"] == [3193, 3448]
    assert fixed["peak_sigma_delta"] == 0.25
    assert fixed["physical_targets_changed"] is False
    covariance = design["exact_covariance"]
    assert covariance["operator"] == "Q = I - R*R"
    assert covariance["maximum_phase_count"] == 27
    assert covariance["stationary_approximation_allowed"] is False


def test_piecewise_geometry_and_antipodal_rules_are_frozen():
    geometry = load_design()["piecewise_geometry"]
    assert "ties to even" in geometry["rounding"]
    assert geometry["canonical_vector_rule"]["index"].startswith("lowest")
    assert geometry["likelihood_cache_key"] == [
        "m_x", "m_y", "m_z", "o_x", "o_y", "o_z"
    ]
    assert len(geometry["covariance_cache_key"]) == 6
    assert "a and -a" in geometry["antipodal_contract"]
    assert "q itself remains unbounded" in geometry["periodicity"]


def test_replicate_temperature_and_resampling_contract_is_frozen():
    design = load_design()
    replicates = design["replicates"]
    assert replicates["count"] == 4
    assert replicates["particles_per_replicate"] == 2048
    assert replicates["master_seeds"] == [
        2026082301, 2026082302, 2026082303, 2026082304
    ]
    tempering = design["adaptive_tempering"]
    assert tempering["target_conditional_ESS"] == 0.8 * 2048
    assert tempering["bisection"]["interval_tolerance"] == 1e-10
    assert tempering["maximum_positive_temperature_stages"] == 256
    resampling = design["resampling"]
    assert resampling["trigger"] == (
        "strictly ESS < 1024.0; equality does not resample"
    )
    assert resampling["method"] == "systematic"
    assert "lowest index" in resampling["CDF_tie_rule"]


def test_mh_kernel_and_information_firewall_are_frozen():
    design = load_design()
    mh = design["MH_rejuvenation"]
    assert mh["sweeps_per_stage"] == 4
    assert mh["move_mixture"] == {
        "q_local": 0.4,
        "axis_local": 0.3,
        "joint_local": 0.2,
        "prior_independence": 0.1,
    }
    assert mh["q_local"]["whitened_scales"] == [0.25, 0.6, 1.5]
    assert mh["axis_local"]["kappa_values"] == [100.0, 10.0, 1.0]
    assert "q remains in R^3" in mh["q_local"]["boundary"]
    firewall = design["information_firewall"]
    assert firewall["temperature_controller_uses_only_log_Z_bar"] is True
    assert firewall["parent_specific_terminal_evaluation_only_after_all_particle_histories_are_frozen"] is True
    assert firewall["old_rows_used_as_particles_or_estimators"] is False
    assert firewall["old_rows_used_to_initialize_cache"] is False


def test_regression_banks_are_pinned_but_forbidden_from_production_cache():
    regression = load_design()["validation"]["old_bank_regression"]
    for prefix in ("adaptation_2048", "adaptation_8192"):
        path = ROOT / regression[f"{prefix}_record"]
        assert sha256_file(path) == regression[f"{prefix}_record_sha256"]
    assert regression["immutable_regression_input_only"] is True
    assert regression["cache_artifacts_reused_in_production"] is False
    assert regression["log_Z_max_difference"] == 1e-10


def test_smc_gates_and_all_downstream_authorizations_are_closed():
    design = load_design()
    gates = design["prospective_gates"]
    assert gates["minimum_genealogical_ESS_each_replicate"] == 128.0
    assert gates["replicate_log_I_bar_range_max_nat"] == 0.2
    assert gates["maximum_pairwise_parent_probability_L1"] == 0.2
    assert gates["pooled_parent_ESS_min"] == 32.0
    assert gates["maximum_pooled_parent_probability"] == 0.1
    assert gates["old_IID_geometry_ESS_or_max_weight_gate_reused"] is False
    authorization = design["authorization"]
    assert authorization["implementation_authorized"] is True
    assert authorization["production_execution_authorized"] is False
    for key in (
        "conditional_field_bank_authorized",
        "parent_or_seed_selection_authorized",
        "PM_or_halo_finder_authorized",
        "RAMSES_authorized",
    ):
        assert authorization[key] is False
