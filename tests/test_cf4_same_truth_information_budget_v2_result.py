import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "config/cf4_same_truth_information_budget_v2_result.json"


def _load() -> dict:
    return json.loads(RECORD.read_text())


def test_result_record_binds_the_frozen_v2_sources() -> None:
    record = _load()
    for name in (
        "program",
        "solver_correction_record",
        "implementation",
        "pilot_runner",
        "member_runner",
        "aggregate_runner",
    ):
        binding = record["bindings"][name]
        path = ROOT / binding["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]


def test_result_record_seals_the_complete_no_go_execution() -> None:
    record = _load()
    assert record["status"] == (
        "COMPLETE_COVARIANCE_ONLY_INFORMATION_BUDGET_NO_GO_"
        "ADD_INDEPENDENT_Z0_DENSITY_TRACERS"
    )
    execution = record["executions"]
    assert execution["v2_technical_pilot"]["validator_status"] == "PASS"
    assert execution["v2_member_array"]["task_count"] == 64
    assert execution["v2_member_array"]["all_task_exit_codes"] == "0:0"
    assert execution["v2_aggregate"]["exit_code"] == "0:0"
    assert not execution["v2_retry_or_replacement"]
    assert not execution["manual_syntax_or_syn101_numerical_execution"]
    artifact = record["bindings"]["aggregate_artifact"]
    assert artifact["validator_status"] == "PASS"
    assert all(len(artifact[key]) == 64 for key in (
        "result_json_sha256",
        "metrics_npz_sha256",
        "manifest_json_sha256",
        "COMPLETE_sha256",
    ))


def test_all_finite_scenarios_fail_the_frozen_lowest_bin_gate() -> None:
    record = _load()
    lowest = record["lowest_joint_bin"]
    assert lowest["required_information_fraction_and_bootstrap_lower"] == 0.8
    assert not any(lowest["strict_pass_pattern"].values())
    assert lowest["all_scenario_joint_frontier_prefix_bin_count"] == 0
    assert lowest["preregistered_diagnostic_code"] == (
        "FINITE_LOW_NOISE_CEILING_INSUFFICIENT_ADD_INDEPENDENT_Z0_DENSITY_TRACERS"
    )
    for metrics in record["scenario_metrics_lowest_bin"].values():
        information = metrics["recovered_information_fraction"]
        assert math.isclose(metrics["expected_response"], information)
        assert math.isclose(metrics["expected_correlation_r"], math.sqrt(information))
        assert math.isclose(metrics["expected_residual_power_ratio"], 1.0 - information)
        assert metrics["recovered_information_bootstrap_95_interval"][0] < 0.8
        assert not metrics["strict_gate"]


def test_contrasts_and_finite_ceiling_interpretation_are_consistent() -> None:
    record = _load()
    metrics = record["scenario_metrics_lowest_bin"]
    contrasts = record["information_contrasts_lowest_bin"]
    baseline = metrics["marginalized_s1"]["recovered_information_fraction"]
    known = metrics["known_s1"]["recovered_information_fraction"]
    known03 = metrics["known_s0p3"]["recovered_information_fraction"]
    known01 = metrics["known_s0p1"]["recovered_information_fraction"]
    assert math.isclose(contrasts["known_minus_marginalized_s1_information"], known - baseline)
    assert math.isclose(contrasts["known_s0p3_minus_s1_information"], known03 - known)
    assert math.isclose(contrasts["known_s0p1_minus_s1_information"], known01 - known)
    assert "not a zero-noise theorem" in record["interpretation"]["finite_ceiling_limit"]


def test_firewalls_and_next_stage_require_new_user_approval() -> None:
    record = _load()
    firewall = record["firewall_audit"]
    assert firewall["covariance_only"]
    assert not firewall["truth_array_generated_or_deserialized"]
    assert not firewall["likelihood_datum_consumed_by_inference"]
    assert firewall["new_truth_seed_count"] == 0
    assert firewall["new_random_seed_count"] == 0
    disposition = record["scientific_disposition"]
    assert disposition["high_resolution_z0_density_map"] == "NOT_ESTABLISHED"
    assert disposition["constrained_IC"] == "NOT_ESTABLISHED"
    assert disposition["0p3_cMpc_h_observational_density_or_velocity_claim"] == "NOT_ALLOWED"
    next_stage = record["recommended_next_stage_requiring_user_approval"]
    assert next_stage["name"] == "independent_z0_density_tracer_information_budget_design"
    assert not next_stage["new_execution_authorized_by_this_record"]
    assert "new Slurm scientific execution" in next_stage["forbidden_without_new_approval"]


def test_memory_request_retained_more_than_twenty_percent_observed_headroom() -> None:
    resource = _load()["resource_audit"]
    assert resource["requested_memory_has_at_least_20_percent_observed_headroom"]
    assert resource["member_requested_memory_MiB"] >= math.ceil(
        1.2 * resource["observed_max_batch_MaxRSS_MiB"]
    )
