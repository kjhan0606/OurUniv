import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_phasec_same_gpu_repeatability_diagnostic_v1.json"
SOURCE = ROOT / "src/cf4_phasec_same_gpu_repeatability_diagnostic.py"
RUNNER = ROOT / "scripts/run_cf4_phasec_same_gpu_repeatability_diagnostic_v1.sbatch"
RESULT_RECORD = ROOT / "config/cf4_phasec_same_gpu_repeatability_diagnostic_v1_result_record.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_program():
    from cf4_phasec_same_gpu_repeatability_diagnostic import load_program

    return load_program(PROGRAM)[0]


def test_program_binds_exact_failures_controls_and_two_repeats():
    program = load_program()
    assert [(row["index"], row["seed"]) for row in program["assignments"]] == [
        (1, 2026083001),
        (6, 2026083006),
        (2, 2026083002),
        (7, 2026083007),
    ]
    assert [row["role"] for row in program["assignments"]] == [
        "scan_gate_failure",
        "scan_gate_failure",
        "scan_gate_pass_control",
        "scan_gate_pass_control",
    ]
    assert program["repetitions"] == [0, 1]


def test_program_lineage_is_frozen_and_scope_remains_mock_only():
    program = load_program()
    for binding in program["lineage"].values():
        path = Path(binding["path"])
        assert path.is_file()
        assert sha256(path) == binding["sha256"]
    assert not any(program["scope_firewall"].values())
    authorization = program["authorization"]
    for key in ("sampler", "actual_observational_data", "validation_seed", "Phase_D_or_later"):
        assert authorization[key] is False


def test_repeat_summary_separates_reproducibility_from_all_pass():
    from cf4_phasec_same_gpu_repeatability_diagnostic import summarize_seed_repeats

    stable_fail = [
        {"artifact_status": "VALID", "pass": False, "result_sha256": "a", "science_signature": {"pass": False}},
        {"artifact_status": "VALID", "pass": False, "result_sha256": "a", "science_signature": {"pass": False}},
    ]
    summary = summarize_seed_repeats(stable_fail)
    assert summary["exact_science_repeat"] is True
    assert summary["all_repeats_pass"] is False
    assert summary["classification"] == "REPEATABLE_FAILURE"

    unstable = [
        {"artifact_status": "VALID", "pass": True, "result_sha256": "a", "science_signature": {"pass": True}},
        {"artifact_status": "VALID", "pass": False, "result_sha256": "b", "science_signature": {"pass": False}},
    ]
    summary = summarize_seed_repeats(unstable)
    assert summary["exact_science_repeat"] is False
    assert summary["classification"] == "NONREPRODUCIBLE_SAME_GPU"


def test_runner_is_one_gpu_serial_fresh_process_execution_with_memory_headroom():
    program = load_program()
    execution = program["execution"]
    assert execution["requested_host_memory_MiB"] >= 1.2 * execution["expected_peak_host_memory_MiB"]
    runner = RUNNER.read_text()
    assert "#SBATCH --gres=gpu:1" in runner
    assert "#SBATCH --mem=9216M" in runner
    assert "#SBATCH --array" not in runner
    assert "for repeat in 0 1" in runner
    assert "for task_index in 1 6 2 7" in runner
    assert '"$python" -I -P "$gate_implementation" run' in runner
    assert '"$SUBMISSION_CONTROLLER" == syntax' in runner
    assert "scripts/tripwire/**" in runner
    for forbidden in ("pgrep", "renameat2", "/tmp", "syn101 --"):
        assert forbidden not in runner


def test_source_records_device_and_never_releases_sampler():
    source = SOURCE.read_text()
    assert "nvidia-smi" in source
    assert "SLURM_JOB_GPUS" in source
    assert '"sampler_allowed": False' in source
    for forbidden in ("counts_train", "counts_holdout", "prepare_catalog", "blackjax", "run_four_chains"):
        assert forbidden not in source


def test_result_record_preserves_exact_no_go_and_all_run_pass_evidence():
    import json

    result = json.loads(RESULT_RECORD.read_text())
    assert result["lineage"]["commit"] == "6638fba5970f7bdd684775604c41f12a1898b450"
    outcome = result["predeclared_outcome"]
    assert outcome["valid_artifact_count"] == 8
    assert outcome["all_eight_numerical_runs_pass"] is True
    assert outcome["exact_science_signature_repeatability_pass"] is False
    assert result["decision"]["run_sampler_mechanics_pilot"] is False
    audit = result["posthoc_difference_audit"]
    assert audit["largest_maximum_normalized_scalar_difference"] < 3e-15
    assert audit["role"].startswith("magnitude characterization only")
    assert "syn06" in result["decision"]["next_allowed_work"]
