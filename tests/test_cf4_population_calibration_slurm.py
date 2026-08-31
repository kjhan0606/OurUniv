import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_bgc_population_calibration_program_v1.json"
SOURCE = ROOT / "src/cf4_population_calibration.py"
MEMBERS = ROOT / "scripts/run_cf4_bgc_population_calibration_members_v1.sbatch"
AGGREGATE = ROOT / "scripts/run_cf4_bgc_population_calibration_aggregate_v1.sbatch"
CORRECTION = ROOT / "config/cf4_bgc_population_calibration_aggregate_correction_v2.json"
AGGREGATE_V2 = ROOT / "scripts/run_cf4_bgc_population_calibration_aggregate_v2.sbatch"


def test_program_binds_current_sources_and_inputs():
    program = json.loads(PROGRAM.read_text())
    assert set(program["inputs"]) == {"catalog", "bin_manifest"}
    for record in program["inputs"].values():
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record[
            "sha256"
        ]
    for record in program["source_bindings"].values():
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record[
            "sha256"
        ]


def test_member_array_is_exactly_64_cpu_members_with_memory_headroom():
    program = json.loads(PROGRAM.read_text())
    execution = program["execution"]
    source = MEMBERS.read_text()
    assert "#SBATCH --array=0-63%8" in source
    assert "#SBATCH --partition=a10" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=2048M" in source
    assert "#SBATCH --time=01:00:00" in source
    assert execution["member_requested_memory_MiB"] >= 1.2 * execution[
        "member_expected_peak_memory_MiB"
    ]
    assert "SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 64" in source
    assert "run-member" in source and "validate-member" in source
    assert "JAX_PLATFORMS=cpu" in source
    assert "host_name\" != syntax" in source
    assert "host_name\" != syn101" in source
    assert "scripts/tripwire/**" in source
    assert "renameat2" not in source
    assert "pgrep" not in source
    assert "--requeue" not in source


def test_aggregate_is_afterok_fail_closed_and_has_memory_headroom():
    program = json.loads(PROGRAM.read_text())
    execution = program["execution"]
    source = AGGREGATE.read_text()
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --mem=4096M" in source
    assert "#SBATCH --time=00:30:00" in source
    assert execution["aggregate_requested_memory_MiB"] >= 1.2 * execution[
        "aggregate_expected_peak_memory_MiB"
    ]
    assert "Dependency=afterok:${MEMBER_ARRAY_JOB_ID}" in source
    assert "aggregate" in source and "validate-aggregate" in source
    assert "[[ ! -e \"$output\" ]]" in source
    assert "JAX_PLATFORMS=cpu" in source
    assert "renameat2" not in source
    assert "pgrep" not in source
    assert "--requeue" not in source


def test_both_runners_pin_exact_program_and_source_hashes():
    program_sha = hashlib.sha256(PROGRAM.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for runner in (MEMBERS, AGGREGATE):
        text = runner.read_text()
        assert f"program_sha={program_sha}" in text
        assert f"source_sha={source_sha}" in text
        assert 'head_commit" == "$EXPECTED_COMMIT' in text
        assert 'upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT' in text


def test_no_untouched_validation_or_downstream_science_is_authorized():
    program = json.loads(PROGRAM.read_text())
    authorization = program["authorization"]
    assert authorization["untouched_256_mock_validation"] is False
    assert authorization["frontier_promotion"] is False
    assert authorization["KF_EXPAND"] is False
    assert authorization["IC_PM_HOP_RAMSES"] is False
    assert authorization["automatic_retry"] is False
    assert authorization["automatic_follow_on_after_aggregate"] is False
    assert program["validation_firewall"]["may_be_opened_or_executed_by_this_program"] is False


def test_v2_preserves_v1_failure_and_changes_only_dependency_authentication():
    correction = json.loads(CORRECTION.read_text())
    assert correction["status"] == "ASSISTANT_ERROR_SINGLE_AGGREGATE_CORRECTION_AUTHORIZED"
    assert correction["failed_v1"]["Slurm_job_id"] == 328695
    assert correction["failed_v1"]["member_array_job_id"] == 328686
    assert correction["failed_v1"]["aggregate_artifact_published"] is False
    assert correction["failed_v1"]["scientific_or_numerical_failure"] is False
    assert correction["correction"]["scientific_program_changed"] is False
    assert correction["correction"]["source_changed"] is False
    assert correction["authorization"]["member_rerun"] is False
    assert correction["authorization"]["untouched_256_mock_validation"] is False
    assert correction["execution"]["requested_memory_MiB"] >= 1.2 * correction[
        "execution"
    ]["expected_peak_memory_MiB"]
    for record in correction["bindings"].values():
        if not isinstance(record, dict):
            continue
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record[
            "sha256"
        ]


def test_v2_runner_uses_persistent_comment_and_no_live_dependency_assertion():
    source = AGGREGATE_V2.read_text()
    correction_sha = hashlib.sha256(CORRECTION.read_bytes()).hexdigest()
    assert f"correction_sha={correction_sha}" in source
    assert "Comment=${expected_comment}" in source
    assert "Dependency=afterok:${MEMBER_ARRAY_JOB_ID}" not in source
    assert '[[ "$MEMBER_ARRAY_JOB_ID" == 328686 ]]' in source
    assert '[[ "$FAILED_V1_JOB_ID" == 328695 ]]' in source
    assert '[[ "$IMPLEMENTATION_COMMIT" == 342bb7c77ac60801a303cb81311a30b3506c8f1d ]]' in source
    assert '--implementation-commit "$IMPLEMENTATION_COMMIT"' in source
    assert "run-member" not in source
    assert "aggregate" in source and "validate-aggregate" in source
    assert "--requeue" not in source
    assert "renameat2" not in source
    assert "pgrep" not in source
