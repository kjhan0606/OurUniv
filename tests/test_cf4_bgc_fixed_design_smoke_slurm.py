import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_V1 = ROOT / "config" / "cf4_bgc_fixed_design_smoke_execution_v1.json"
RUNNER_V1 = ROOT / "scripts" / "run_cf4_bgc_fixed_design_smoke_v1.sbatch"
RESULT_V1 = ROOT / "config" / "cf4_bgc_fixed_design_smoke_v1_result.json"
CONFIG = ROOT / "config" / "cf4_bgc_fixed_design_smoke_execution_v2.json"
RUNNER = ROOT / "scripts" / "run_cf4_bgc_fixed_design_smoke_v2.sbatch"
RESULT = ROOT / "config" / "cf4_bgc_fixed_design_smoke_v2_result.json"


def test_failed_v1_is_preserved_and_bound_to_its_original_execution():
    config = json.loads(CONFIG_V1.read_text())
    record = json.loads(RESULT_V1.read_text())
    assert record["status"] == (
        "IMPLEMENTATION_FAIL_THETA_NOT_DERIVED_FROM_STORED_VELOCITY"
    )
    assert record["execution"]["Slurm_job_id"] == 328661
    assert record["bindings"]["execution_config_sha256"] == hashlib.sha256(
        CONFIG_V1.read_bytes()
    ).hexdigest()
    assert record["failed_check"]["full_grid_velocity_divergence_relative_error"] > 0.08
    assert record["failed_check"]["non_Nyquist_velocity_divergence_relative_error"] < 1e-12
    assert record["scientific_disposition"]["science_claim_allowed"] is False
    assert f"config_sha={hashlib.sha256(CONFIG_V1.read_bytes()).hexdigest()}" in (
        RUNNER_V1.read_text()
    )
    assert config["authorization"]["retry_authorized"] is False


def test_correction_execution_contract_is_single_fixed_design_no_claim_smoke():
    config = json.loads(CONFIG.read_text())
    assert config["status"] == (
        "ASSISTANT_ERROR_CORRECTION_SINGLE_FIXED_DESIGN_TRUTH_MOCK_SMOKE"
    )
    authorization = config["authorization"]
    assert authorization["single_Slurm_correction_execution_authorized"] is True
    assert authorization["same_truth_seed_reexecution_authorized"] is True
    assert authorization["new_unique_development_truth_seed_authorized"] is False
    assert authorization["population_selection_mock_authorized"] is False
    assert authorization["development_64_mock_execution_authorized"] is False
    assert authorization["untouched_256_mock_validation_authorized"] is False
    assert authorization["additional_retry_authorized"] is False
    smoke = config["smoke_contract"]
    assert smoke["selection_semantics"] == "observed_grouped_CF4_fixed_design_conditioned"
    assert smoke["mock_datum"] == "u_mock=A*s_truth+B*q_truth+epsilon"
    assert (smoke["grid_N"], smoke["cell_size_cMpc_h"]) == (32, 12.0)
    assert smoke["new_unique_development_truth_seed_count"] == 0
    assert smoke["posterior_draw_count"] == 4
    assert smoke["density_canonical_independent_real_mode_count"] == 8538
    assert smoke["theta_non_Nyquist_canonical_independent_real_mode_count"] == 8535
    assert smoke["Nyquist_plane_modes_excluded_from_theta_metrics"] is True
    assert smoke["science_claim_allowed"] is False
    assert config["failed_predecessor_binding"]["result_record_sha256"] == (
        hashlib.sha256(RESULT_V1.read_bytes()).hexdigest()
    )


def test_v2_inputs_and_unchanged_dependencies_remain_bound():
    config = json.loads(CONFIG.read_text())
    for record in config["input_bindings"].values():
        expected = record.get("sha256", record.get("file_sha256"))
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == expected
    for name, record in config["source_bindings"].items():
        if name == "fixed_design_smoke":
            assert record["sha256"] == (
                "0a1a9348ec4a65ff8a9d10eddc0f3dd251b3a1469ec8b50f874f80f554aa46cb"
            )
            continue
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == (
            record["sha256"]
        )


def test_slurm_runner_matches_resource_and_controller_contract():
    config = json.loads(CONFIG.read_text())
    source = RUNNER.read_text()
    execution = config["execution"]
    assert "#SBATCH --partition=a10" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=1536M" in source
    assert "#SBATCH --time=00:30:00" in source
    assert execution["memory_request_MiB"] >= 1.2 * execution["memory_expected_peak_MiB"]
    assert execution["maximum_correction_submissions"] == 1
    assert execution["maximum_correction_executions"] == 1
    assert "SUBMISSION_CONTROLLER\" == syntax" in source
    assert "host_name\" != syntax" in source
    assert "host_name\" != syn101" in source
    assert "scripts/tripwire/**" in source
    assert "JAX_PLATFORMS=cpu" in source
    assert "--implementation-commit \"$EXPECTED_COMMIT\"" in source
    assert "renameat2" not in source
    assert "pgrep" not in source


def test_runner_binds_exact_execution_config_hash():
    source = RUNNER.read_text()
    expected = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert f"config_sha={expected}" in source


def test_v2_result_record_binds_pass_without_a_science_claim():
    record = json.loads(RESULT.read_text())
    assert record["status"] == "COMPLETE_IMPLEMENTATION_SMOKE_PASS_NO_SCIENCE_CLAIM"
    assert record["execution"]["Slurm_job_id"] == 328664
    assert record["execution"]["new_unique_development_truth_seed_count_consumed"] == 0
    assert record["bindings"]["execution_config_sha256"] == hashlib.sha256(
        CONFIG.read_bytes()
    ).hexdigest()
    assert record["bindings"]["Slurm_runner_sha256"] == hashlib.sha256(
        RUNNER.read_bytes()
    ).hexdigest()
    assert record["bindings"]["failed_v1_result_record_sha256"] == hashlib.sha256(
        RESULT_V1.read_bytes()
    ).hexdigest()
    assert record["bindings"]["portable_validator_source_sha256"] == hashlib.sha256(
        (ROOT / "src" / "cf4_bgc_fixed_design_smoke.py").read_bytes()
    ).hexdigest()
    checks = record["implementation_checks"]
    assert checks["theta_non_Nyquist_canonical_independent_real_mode_count"] == 8535
    assert checks["delta_theta_non_Nyquist_max_relative_error"] < 1e-12
    assert checks["all_non_theta_arrays_identical_to_v1"] is True
    disposition = record["scientific_disposition"]
    assert disposition["fixed_design_conditional_truth_mock_implementation"] == "PASS"
    assert disposition["population_selection_function_validated"] is False
    assert disposition["development_64_mock_execution_performed"] is False
    assert disposition["target_0_3_cMpc_h_reached"] is False
    assert disposition["science_claim_allowed"] is False
