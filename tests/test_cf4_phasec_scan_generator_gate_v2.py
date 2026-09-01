import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_phasec_scan_generator_gate_v2.json"
SOURCE = ROOT / "src/cf4_phasec_scan_generator_gate_v2.py"
RUNNER = ROOT / "scripts/run_cf4_phasec_scan_generator_gate_v2.sbatch"
AGGREGATOR = ROOT / "scripts/aggregate_cf4_phasec_scan_generator_gate_v2.sbatch"
RESULT_RECORD = ROOT / "config/cf4_phasec_scan_generator_gate_v2_result_record.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_program():
    from cf4_phasec_scan_generator_gate_v2 import load_program

    return load_program(PROGRAM)[0]


def test_exact_eight_seed_assignment_and_scan_ladder():
    program = load_program()
    assignments = program["assignments"]
    assert [row["seed"] for row in assignments] == list(range(2026083000, 2026083008))
    assert [row["arm"] for row in assignments] == list("AABBCCDD")
    integrator = program["integrator"]
    assert integrator["a_nbody_maxsteps"] == [1 / 128, 1 / 256]
    assert integrator["production_a_nbody_maxstep"] == 1 / 256
    assert "jax.lax.scan" in integrator["loop_implementation"]


def test_scan_diagnostic_pass_is_bound_and_scope_is_generator_only():
    program = load_program()
    for binding in program["lineage"].values():
        path = Path(binding["path"])
        assert path.is_file()
        assert sha256(path) == binding["sha256"]
    assert not any(program["scope_firewall"].values())
    authorization = program["authorization"]
    for key in ("sampler", "actual_observational_data", "validation_seed", "Phase_D_or_later"):
        assert authorization[key] is False


def test_all_seed_gate_cannot_replace_seeds_or_relax_thresholds():
    gates = load_program()["gates"]
    assert gates["all_eight_seeds_must_pass"] is True
    assert gates["seed_replacement_or_threshold_relaxation"] is False
    assert gates["density_cross_correlation_min"] == 0.999
    assert gates["density_relative_L2_max"] == 0.03
    assert gates["velocity_cross_correlation_min"] == 0.995
    assert gates["velocity_relative_L2_max"] == 0.05


def test_source_uses_admitted_scan_candidate_and_not_inference():
    source = SOURCE.read_text()
    assert "build_scan_candidate" in source
    assert "nested_white_fields" in source
    for forbidden in (
        "counts_train",
        "counts_holdout",
        "vobs",
        "prepare_catalog",
        "run_four_chains",
        "blackjax",
    ):
        assert forbidden not in source


def test_slurm_memory_and_controller_contract():
    execution = load_program()["execution"]
    assert execution["requested_host_memory_MiB_per_task"] >= 1.2 * execution[
        "expected_peak_host_memory_MiB_per_task"
    ]
    assert execution["aggregate_requested_host_memory_MiB"] >= 1.2 * execution[
        "aggregate_expected_peak_host_memory_MiB"
    ]
    runner = RUNNER.read_text()
    aggregator = AGGREGATOR.read_text()
    assert "#SBATCH --array=0-7" in runner
    assert "#SBATCH --mem=9216M" in runner
    assert "#SBATCH --mem=1024M" in aggregator
    for text in (runner, aggregator):
        assert '"$SUBMISSION_CONTROLLER" == syntax' in text
        assert "EXPECTED_UPSTREAM_COMMIT" in text
        assert "scripts/tripwire/**" in text
        assert "pgrep" not in text
        assert "renameat2" not in text
        assert "/tmp" not in text


def test_pass_releases_only_two_mock_sampler_mechanics_indices():
    decision = load_program()["decision_rule"]
    assert decision["sampler_mechanics_pilot_indices_after_PASS"] == [0, 6]
    assert decision["actual_observational_posterior_allowed"] is False
    assert decision["validation_or_Phase_D_allowed"] is False


def test_recorded_result_is_no_go_and_revises_the_two_seed_inference():
    result = json.loads(RESULT_RECORD.read_text())
    assert result["lineage"]["commit"] == "b4f33d2a1f408c07ba746a8765d78096a6028e5c"
    summary = result["summary"]
    assert summary["valid_task_artifact_count"] == 8
    assert summary["passing_seed_count"] == 6
    assert summary["failing_seeds"] == [2026083001, 2026083006]
    assert summary["all_seed_gate_pass"] is False
    assert result["decision"]["run_sampler_mechanics_pilot"] is False
    assert "falsifies" in result["audit_revision"]["revised_interpretation"]
    assert result["decision"]["next_allowed_work"].startswith("mock-only")
