import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_datum_bearing_z0_phasec_redesign_v1.json"
SOURCE = ROOT / "src/cf4_phasec_redesign_generator_gate.py"
RUNNER = ROOT / "scripts/run_cf4_phasec_redesign_generator_gate_v1.sbatch"
AGGREGATOR = ROOT / "scripts/aggregate_cf4_phasec_redesign_generator_gate_v1.sbatch"
RESULT_RECORD = ROOT / "config/cf4_phasec_redesign_generator_v1_result_record.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_program():
    from cf4_phasec_redesign_generator_gate import load_program

    return load_program(PROGRAM)[0]


def test_scope_firewall_and_exact_development_assignments():
    program = load_program()
    authorization = program["authorization"]
    assert authorization["replacement_Phase_C_mock_design_implementation_and_execution"] is True
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


def test_generator_ladder_and_all_seed_gate_are_frozen():
    program = load_program()
    base = json.loads(Path(program["lineage"]["base_program"]["path"]).read_text())
    assert program["rng_tags"] == base["rng_tags"]
    generator = program["generator_gate"]
    candidates = generator["candidates"]
    assert [row["name"] for row in candidates] == [
        "convergence_1_over_128",
        "production_1_over_256",
    ]
    assert [row["a_nbody_maxstep"] for row in candidates] == [1 / 128, 1 / 256]
    assert "every" in generator["aggregate_gate"]
    assert generator["per_candidate_gates"]["density_mean_absolute_error_max"] == 2e-12
    convergence = generator["time_convergence_gates_on_N32_fields"]
    assert convergence == {
        "density_cross_correlation_min": 0.999,
        "density_relative_L2_max": 0.03,
        "velocity_cross_correlation_min": 0.995,
        "velocity_relative_L2_max": 0.05,
    }


def test_sampler_redesign_is_prior_equivalent_and_does_not_relax_gates():
    inference = load_program()["replacement_inference"]
    parameterization = inference["parameterization"]
    assert parameterization["scientific_prior_change"] is False
    assert "all 32792 coordinates are standard normal" in parameterization["joint_prior"]
    sampler = inference["sampler"]
    assert sampler["field_or_nuisance_empirical_mass_adaptation"] is False
    assert sampler["chain_count"] == 4
    assert sampler["warmup_steps"] == 512
    assert sampler["posterior_draws_per_chain"] == 512
    assert sampler["integration_steps"] == 12
    gates = inference["sampler_usability_gates"]
    assert gates["rank_normalized_split_Rhat_max"] == 1.05
    assert gates["bulk_ESS_min"] == 100
    assert gates["tail_ESS_min"] == 100
    assert gates["threshold_relaxation_after_outcomes_allowed"] is False
    derived = inference["derived_state_mechanics_gate"]
    assert derived["evaluate_every_retained_draw"] is True
    assert derived["clipping_allowed"] is False


def test_generator_source_binding_and_lineage_hashes_match():
    program = json.loads(PROGRAM.read_text())
    for binding in program["lineage"].values():
        path = Path(binding["path"])
        assert path.is_file()
        assert sha256(path) == binding["sha256"]


def test_generator_source_never_reads_count_or_velocity_data():
    source = SOURCE.read_text()
    for forbidden in (
        "raw_selection_exposure",
        "counts_all",
        "counts_train",
        "counts_holdout",
        "vobs",
        "prepare_catalog",
        "run_four_chains",
    ):
        assert forbidden not in source
    assert "nested_white_fields" in source
    assert "linear_modes" in source
    assert "nbody(" in source


def test_time_convergence_metric_identity_and_offset_behaviour():
    from cf4_phasec_redesign_generator_gate import correlation_and_relative_l2

    reference = np.arange(27, dtype=np.float64).reshape(3, 3, 3) - 13.0
    correlation, relative_l2 = correlation_and_relative_l2(reference, reference)
    assert np.isclose(correlation, 1.0)
    assert np.isclose(relative_l2, 0.0)
    shifted = reference + 0.1
    shifted_correlation, shifted_l2 = correlation_and_relative_l2(shifted, reference)
    assert np.isclose(shifted_correlation, 1.0)
    assert shifted_l2 > 0.0


def test_slurm_resources_and_fail_closed_controller_contract():
    program = load_program()
    execution = program["staged_execution"]
    assert execution["generator_requested_host_memory_MiB_per_task"] >= 1.2 * execution[
        "generator_expected_peak_host_memory_MiB_per_task"
    ]
    assert execution["generator_aggregate_requested_host_memory_MiB"] >= 1.2 * execution[
        "generator_aggregate_expected_peak_host_memory_MiB"
    ]
    runner = RUNNER.read_text()
    aggregator = AGGREGATOR.read_text()
    assert "#SBATCH --array=0-7" in runner
    assert "#SBATCH --mem=5120M" in runner
    assert "#SBATCH --mem=1024M" in aggregator
    assert "XLA_FLAGS=--xla_gpu_autotune_level=0" in runner
    for text in (runner, aggregator):
        assert '"$SUBMISSION_CONTROLLER" == syntax' in text
        assert "EXPECTED_UPSTREAM_COMMIT" in text
        assert "scripts/tripwire/**" in text
        assert "renameat2" not in text
        assert "pgrep" not in text
        assert "/tmp" not in text


def test_success_semantics_do_not_claim_observations_or_resolution():
    semantics = load_program()["success_semantics"]
    for key in (
        "actual_present_day_density_posterior_created",
        "actual_present_day_velocity_posterior_created",
        "observational_frontier_or_resolution_claim",
        "target_0p3_cMpc_h_reached",
        "validation_opened",
        "Phase_D_started",
    ):
        assert semantics[key] is False
    assert semantics["actual_observational_step_requires_separate_user_approval"] is True


def test_generator_result_record_is_no_go_and_releases_no_sampler():
    record = json.loads(RESULT_RECORD.read_text())
    assert record["lineage"]["program"]["sha256"] == sha256(PROGRAM)
    assert record["lineage"]["implementation"]["sha256"] == sha256(SOURCE)
    assert record["summary"]["valid_task_artifact_count"] == 8
    assert record["summary"]["passing_seed_count"] == 7
    assert record["summary"]["all_seed_gate_pass"] is False
    assert record["summary"]["sampler_mechanics_pilot_started"] is False
    failures = [row for row in record["task_outcomes"] if not row["pass"]]
    assert [row["seed"] for row in failures] == [2026083002]
    assert record["cross_run_evidence"]["seed_2026083002"][
        "nested_white_metrics_identical_between_V5_and_redesign"
    ] is True
    decision = record["decision"]
    assert decision["generator_gate_pass"] is False
    assert decision["run_sampler_mechanics_pilot"] is False
    assert decision["replace_or_drop_seed_2026083002"] is False
    assert decision["relax_generator_thresholds"] is False
    assert not any(record["scope_firewall"].values())
    resource = record["resource_audit"]
    assert resource["next_GPU_job_rounded_request_MiB"] >= 1.2 * resource[
        "largest_sacct_batch_MaxRSS_MiB"
    ]
