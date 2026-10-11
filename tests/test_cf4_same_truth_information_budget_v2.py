import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cf4_same_truth_information_budget as v1  # noqa: E402
import cf4_same_truth_information_budget_v2 as v2  # noqa: E402


PROGRAM = ROOT / "config/cf4_same_truth_information_budget_program_v2.json"
CORRECTION = ROOT / "config/cf4_same_truth_information_budget_solver_correction_v2.json"
SOURCE = ROOT / "src/cf4_same_truth_information_budget_v2.py"


def test_correction_preserves_failed_v1_and_changes_solver_only():
    correction = json.loads(CORRECTION.read_text())
    assert correction["failed_v1_execution"]["member_array_job_id"] == 329405
    assert correction["failed_v1_execution"]["aggregate_job_id"] == 329406
    assert correction["failed_v1_execution"]["published_member_artifact_count"] == 0
    assert correction["failed_v1_execution"]["scientific_result_produced"] is False
    fix = correction["root_cause_and_correction"]
    assert fix["v1_CG_max_iterations"] == 500
    assert fix["v2_CG_max_iterations"] == v2.CG_MAXITER == 4000
    assert fix["v2_preconditioner_probe_count"] == v2.PRECONDITIONER_PROBES == 16
    assert fix["scenario_or_noise_scale_changed"] is False
    assert fix["truth_or_datum_changed"] is False
    assert fix["random_seed_or_posterior_draw_changed"] is False
    assert fix["estimand_threshold_or_decision_tree_changed"] is False


def test_v2_program_keeps_v1_science_contract_and_exact_solver_fix():
    program = json.loads(PROGRAM.read_text())
    assert tuple(program["design"]["scenario_order"]) == v1.SCENARIOS
    assert program["design"]["known_nuisance_noise_standard_deviation_scales"] == v1.NOISE_SCALES
    assert program["design"]["new_truth_seed_count"] == 0
    assert program["design"]["new_random_seed_count"] == 0
    assert program["solver_correction"] == {
        "failed_v1_member_array_job_id": 329405,
        "failed_v1_aggregate_job_id": 329406,
        "system": "unnormalized_data_covariance_AAT_plus_N",
        "noise_scale_continuation_order": [1.0, 0.3, 0.1],
        "warm_start_previous_noise_scale": True,
        "preconditioner_probe_count": 16,
        "CG_tolerance": 3e-05,
        "CG_max_iterations": 4000,
        "science_or_random_stream_changed": False,
    }


def test_v2_program_binds_all_repo_and_source_files():
    program = json.loads(PROGRAM.read_text())
    for collection in ("repository_bindings", "source_bindings"):
        for record in program[collection].values():
            assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record[
                "sha256"
            ]


def test_v2_source_uses_unnormalized_warm_started_cg_without_new_seed():
    source = SOURCE.read_text()
    assert "noise_variance * values + A_train(AT_train(values))" in source
    assert "x0=beta_start" in source
    assert "beta_start = beta" in source
    assert "PRECONDITIONER_PROBES = 16" in source
    assert "CG_MAXITER = 4000" in source
    assert "base.seed_schedule" not in source
    assert 'seeds["truth"]' not in source


def test_v2_memory_requests_retain_expected_headroom():
    execution = json.loads(PROGRAM.read_text())["execution"]
    assert execution["technical_pilot_requested_memory_MiB"] >= 1.2 * execution[
        "technical_pilot_expected_peak_memory_MiB"
    ]
    assert execution["member_requested_memory_MiB"] >= 1.2 * execution[
        "member_expected_peak_memory_MiB"
    ]
    assert execution["aggregate_requested_memory_MiB"] >= 1.2 * execution[
        "aggregate_expected_peak_memory_MiB"
    ]
