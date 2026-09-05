import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_datum_bearing_z0_phasec_execution_amendment_v5.json"
SOURCE = ROOT / "src/cf4_datum_bearing_z0_phasec_pilot.py"
PREFLIGHT = ROOT / "scripts/preflight_cf4_datum_bearing_z0_phasec_pilot_v5.sbatch"
RUNNER = ROOT / "scripts/run_cf4_datum_bearing_z0_phasec_pilot_v5.sbatch"
AGGREGATOR = ROOT / "scripts/aggregate_cf4_datum_bearing_z0_phasec_pilot_v5.sbatch"
RESULT_RECORD = ROOT / "config/cf4_datum_bearing_z0_phasec_v5_result_record.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    from cf4_datum_bearing_z0_phasec_pilot import load_program

    return load_program(PROGRAM)[0]


def test_phase_c_scope_seed_firewall_and_balanced_assignment():
    program = load()
    authorization = program["authorization"]
    assert authorization["Phase_C_eight_mock_pilot"] is True
    assert authorization["Slurm_GPU_array"] is True
    for key in (
        "actual_observational_field_inference",
        "actual_2Mpp_count_read",
        "actual_CF4_velocity_datum_used",
        "validation_seed_access",
        "Phase_D_or_later",
        "IC_PM_HOP_RAMSES",
    ):
        assert authorization[key] is False
    assignments = program["mock_assignments"]
    assert [row["seed"] for row in assignments] == list(range(2026083000, 2026083008))
    assert [row["arm"] for row in assignments] == list("AABBCCDD")


def test_truth_is_finer_pmwd_and_reuses_coarse_low_modes():
    program = load()
    assert program["grid"] == {
        "box_size_cMpc_h": 384.0,
        "inference_N": 32,
        "inference_cell_size_cMpc_h": 12.0,
        "truth_N": 64,
        "truth_cell_size_cMpc_h": 6.0,
    }
    text = SOURCE.read_text()
    assert "linear_modes" in text
    assert "nbody(" in text
    assert "coarse_non_nyquist_mode_count" in text
    assert "fine_k[np.ix_(fine_idx, fine_idx, fine_idx)]" in text


def test_small_nested_field_inherits_every_non_nyquist_coarse_mode():
    from cf4_datum_bearing_z0_phasec_pilot import nested_white_fields

    fine, coarse, metrics = nested_white_fields(1234, 4, 8, 99)
    assert fine.shape == (8, 8, 8)
    assert coarse.shape == (4, 4, 4)
    assert metrics["inherited_mode_count"] == 27
    assert metrics["max_inherited_complex_coefficient_error"] < 5e-12


def test_block_sum_and_tsc_are_conservative():
    from cf4_datum_bearing_z0_phasec_pilot import block_sum, tsc_deposit_numpy

    field = np.arange(8**3, dtype=np.float64).reshape(8, 8, 8) + 1.0
    coarse = block_sum(field, 4)
    assert coarse.shape == (4, 4, 4)
    assert np.isclose(coarse.sum(), field.sum())
    rng = np.random.default_rng(31)
    positions = rng.uniform(0.0, 16.0, size=(400, 3))
    masses = np.exp(rng.normal(size=400))
    deposited = tsc_deposit_numpy(masses, positions, 4, 16.0)
    assert np.isclose(deposited.sum(), masses.sum(), rtol=2e-15)


def test_stress_arms_and_heldout_rule_are_frozen_before_results():
    program = load()
    assert program["stress"]["A"].startswith("PMWD")
    assert "coherent" in program["stress"]["B"]
    assert "FoG" in program["stress"]["C"]
    assert "overdispersion" in program["stress"]["D"]
    assert program["heldout"]["training_fraction"] == 0.8
    assert program["heldout"]["holdout_fraction"] == 0.2
    assert "selection-only" in program["heldout"]["Phase_C_delta_LPD_semantics"] or \
        "diagnostic" in program["heldout"]["Phase_C_delta_LPD_semantics"]


def test_four_chain_sampler_and_reports_are_frozen():
    program = load()
    sampler = program["sampler"]
    assert sampler["chain_count"] == 4
    assert sampler["warmup_steps"] == 192
    assert sampler["posterior_draws_per_chain"] == 256
    assert sampler["no_result_dependent_retuning_within_eight_mocks"] is True
    reports = " ".join(program["diagnostics"]["required_reports"])
    for required in ("Rhat", "ESS", "heldout", "P0", "P2", "2PCF", "coverage"):
        assert required in reports


def test_actual_observational_arrays_are_never_read():
    source = SOURCE.read_text()
    assert 'datum["raw_selection_exposure"]' in source
    for forbidden in ('datum["counts_all"]', 'datum["counts_train"]', 'datum["counts_holdout"]'):
        assert forbidden not in source
    assert 'design["vobs"]' not in source
    assert "actual_2Mpp_counts_read\": False" in source
    assert "actual_CF4_velocity_datum_used\": False" in source


def test_resources_have_at_least_twenty_percent_host_memory_headroom():
    execution = load()["execution"]
    assert execution["controller"] == "syntax_submit_and_audit_only"
    assert execution["requested_host_memory_MiB_per_GPU_task"] >= 1.2 * execution[
        "expected_peak_host_memory_MiB_per_GPU_task"
    ]
    assert execution["aggregate_requested_host_memory_MiB"] >= 1.2 * execution[
        "aggregate_expected_peak_host_memory_MiB"
    ]
    assert execution["manual_numerical_execution_on_syntax_or_syn101_allowed"] is False
    assert execution["monitoring_loop_allowed"] is False


def test_bound_inputs_and_sources_match_hashes():
    for section in ("input_bindings", "source_bindings"):
        for record in load()[section].values():
            path = Path(record["path"])
            assert path.is_file()
            assert sha256(path) == record["sha256"]


def test_success_cannot_claim_actual_posterior_or_resolution():
    semantics = load()["success_semantics"]
    assert semantics["eight_mock_development_pilot_only"] is True
    assert semantics["actual_present_day_density_posterior_created"] is False
    assert semantics["actual_present_day_velocity_posterior_created"] is False
    assert semantics["observational_frontier_or_resolution_claim"] is False
    assert semantics["target_0p3_cMpc_h_reached"] is False
    assert semantics["validation_opened"] is False
    assert semantics["Phase_D_requires_result_audit_and_new_user_approval"] is True


def test_slurm_runners_are_fail_closed_and_never_use_manual_syn101():
    preflight = PREFLIGHT.read_text()
    runner = RUNNER.read_text()
    aggregate = AGGREGATOR.read_text()
    assert "#SBATCH --array=0-7" in runner
    assert "#SBATCH --mem=10240M" in runner
    assert "#SBATCH --mem=5120M" in preflight
    assert "#SBATCH --mem=1024M" in aggregate
    assert "XLA_FLAGS=--xla_gpu_autotune_level=0" in preflight
    assert "XLA_FLAGS=--xla_gpu_autotune_level=0" in runner
    assert "XLA_PYTHON_CLIENT_PREALLOCATE=false" in runner
    assert "h100" not in preflight.splitlines()[2]
    assert "h100" not in runner.splitlines()[2]
    for text in (preflight, runner, aggregate):
        assert '"$SUBMISSION_CONTROLLER" == syntax' in text
        assert "EXPECTED_UPSTREAM_COMMIT" in text
        assert "scripts/tripwire/**" in text
        assert "renameat2" not in text
        assert "pgrep" not in text


def test_v2_is_execution_only_and_binds_the_v1_failure():
    amendment = json.loads(PROGRAM.read_text())
    assert amendment["authorization"]["execution_only_retry_after_pre_science_failure"] is True
    assert amendment["authorization"]["change_science_contract"] is False
    assert amendment["execution_override"]["maximum_Slurm_submissions"] == 3
    assert amendment["execution_override"]["dependency_order"].startswith("preflight afterok")
    failure = Path(amendment["V1_infrastructure_failure"]["path"])
    assert sha256(failure) == amendment["V1_infrastructure_failure"]["sha256"]
    record = json.loads(failure.read_text())
    assert record["artifact_status"]["science_result_created"] is False
    assert record["execution_only_repair"]["science_contract_change"] is False
    prior = Path(amendment["prior_preflight_failure"]["path"])
    assert sha256(prior) == amendment["prior_preflight_failure"]["sha256"]
    prior_record = json.loads(prior.read_text())
    assert prior_record["failure_boundary"]["joint_likelihood_value_or_gradient_created"] is False
    assert prior_record["failure_boundary"]["posterior_draw_count"] == 0
    assert prior_record["execution_only_repair"]["science_contract_change"] is False


def test_v5_result_record_is_fail_closed_and_preserves_scope_firewall():
    record = json.loads(RESULT_RECORD.read_text())
    assert record["lineage"]["execution_amendment"]["sha256"] == sha256(PROGRAM)
    assert record["lineage"]["implementation"]["sha256"] == sha256(SOURCE)
    summary = record["array_summary"]
    assert summary["requested_tasks"] == 8
    assert summary["validated_complete_artifact_sets"] == 6
    assert summary["Slurm_failed_tasks"] == 2
    assert summary["tasks_passing_all_sampler_usability_flags"] == 0
    assert summary["balanced_arm_summary_allowed"] is False
    assert [row["assignment"]["seed"] for row in record["task_outcomes"]] == list(
        range(2026083000, 2026083008)
    )
    decision = record["decision"]
    for key in (
        "Phase_C_pilot_pass",
        "promote_current_parent_posterior",
        "open_validation_seeds",
        "run_actual_observational_posterior",
        "enter_Phase_D",
        "claim_present_day_density_or_velocity_posterior",
        "claim_0p3_cMpc_h_resolution",
        "automatic_retry_allowed",
    ):
        assert decision[key] is False
    assert decision["new_user_approval_required_for_redesigned_Phase_C"] is True
    assert not any(record["scope_firewall"].values())
