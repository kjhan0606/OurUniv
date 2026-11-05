import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "config/cf4_bgc_population_calibration_v1_result.json"


def _record():
    return json.loads(RECORD.read_text())


def test_result_record_binds_repository_inputs_and_immutable_artifact():
    record = _record()
    bindings = record["bindings"]
    for name in (
        "scientific_program",
        "member_source",
        "corrected_aggregate_source",
        "aggregate_correction",
        "successful_runner",
    ):
        item = bindings[name]
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item[
            "sha256"
        ]
    artifact = bindings["aggregate_artifact"]
    directory = Path(artifact["directory"])
    expected = {
        "result.json": (artifact["result_json_sha256"], artifact["result_json_bytes"]),
        "metrics.npz": (artifact["metrics_npz_sha256"], artifact["metrics_npz_bytes"]),
        "manifest.json": (
            artifact["manifest_json_sha256"],
            artifact["manifest_json_bytes"],
        ),
        "COMPLETE": (artifact["COMPLETE_sha256"], artifact["COMPLETE_bytes"]),
    }
    assert {path.name for path in directory.iterdir()} == set(expected)
    for name, (digest, size) in expected.items():
        payload = (directory / name).read_bytes()
        assert len(payload) == size
        assert hashlib.sha256(payload).hexdigest() == digest


def test_all_execution_failures_are_preserved_and_final_job_passed():
    execution = _record()["executions"]
    member = execution["member_array"]
    assert member["Slurm_array_job_id"] == 328686
    assert member["task_count"] == 64
    assert member["all_task_exit_codes"] == "0:0"
    assert [row["Slurm_job_id"] for row in execution["preserved_failed_aggregates"]] == [
        328695,
        328769,
        328782,
    ]
    assert all(
        row["artifact_published"] is False
        for row in execution["preserved_failed_aggregates"]
    )
    final = execution["successful_aggregate"]
    assert final["Slurm_job_id"] == 328809
    assert final["state"] == "COMPLETED"
    assert final["exit_code"] == "0:0"


def test_development_result_is_no_go_despite_heldout_prediction_gain():
    record = _record()
    fidelity = record["population_generator_fidelity"]
    assert fidelity["passing_member_count"] == 41
    assert fidelity["failing_member_count"] == 23
    assert fidelity["all_64_members_pass"] is False
    result = record["calibration_result"]
    assert result["heldout_cumulative_prediction_pass_count"] == 12
    assert result["density_strict_gate_pass_count"] == 0
    assert result["theta_strict_gate_pass_count"] == 0
    assert result["joint_frontier_prefix_bin_count"] == 0
    assert result["development_k_eff_global_h_Mpc_or_null"] is None
    assert result["lowest_bin"]["response"] < 0.8
    assert result["lowest_bin"]["correlation_r"] < 0.7
    assert result["lowest_bin"]["residual_power_ratio"] > 0.5
    disposition = record["scientific_disposition"]
    assert disposition["development_population_mock_calibration"] == "NO_GO"
    assert disposition["current_low_k_parent_posterior_promotion"] == "NO_GO"
    assert disposition["untouched_256_mock_validation"] == "NOT_RUN_AND_STILL_UNAUTHORIZED"
    assert disposition["0p3_cMpc_h_observational_density_or_velocity_claim"] == (
        "NOT_ALLOWED"
    )


def test_next_stage_is_same_truth_ablation_and_requires_new_approval():
    next_stage = _record()["recommended_next_stage_requiring_user_approval"]
    assert next_stage["name"] == "same_truth_nested_likelihood_ablation"
    assert len(next_stage["ordered_arms"]) == 4
    assert "Do not retune" in next_stage["decision"]
    assert next_stage["new_Slurm_execution_authorized_by_this_record"] is False
