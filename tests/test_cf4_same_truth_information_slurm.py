import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_same_truth_information_budget_program_v1.json"
SOURCE = ROOT / "src/cf4_same_truth_information_budget.py"
MEMBERS = ROOT / "scripts/run_cf4_same_truth_information_members_v1.sbatch"
AGGREGATE = ROOT / "scripts/run_cf4_same_truth_information_aggregate_v1.sbatch"


def test_member_runner_is_exact_cpu_array_with_three_gib_request():
    text = MEMBERS.read_text()
    assert "#SBATCH --array=0-63%8" in text
    assert "#SBATCH --partition=a10" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --mem=3072M" in text
    assert "#SBATCH --time=00:30:00" in text
    assert "SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 64" in text
    assert '[[ ! -e "$output" ]]' in text
    assert "run-member" in text and "validate-member" in text
    assert "JAX_PLATFORMS=cpu" in text


def test_aggregate_runner_is_afterok_target_without_live_dependency_assertion():
    text = AGGREGATE.read_text()
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --mem=3072M" in text
    assert "#SBATCH --time=00:30:00" in text
    assert "MEMBER_ARRAY_JOB_ID" in text
    assert "cf4_same_truth_information_afterok_${MEMBER_ARRAY_JOB_ID}" in text
    assert "Dependency=afterok:" not in text
    assert '[[ ! -e "$output" ]]' in text
    assert "aggregate" in text and "validate-aggregate" in text


def test_runners_pin_hashes_clean_commit_and_controller_only_execution():
    program_sha = hashlib.sha256(PROGRAM.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for runner in (MEMBERS, AGGREGATE):
        text = runner.read_text()
        assert f"program_sha={program_sha}" in text
        assert f"source_sha={source_sha}" in text
        assert 'head_commit" == "$EXPECTED_COMMIT' in text
        assert 'upstream_commit" == "$EXPECTED_UPSTREAM_COMMIT' in text
        assert '"$SUBMISSION_CONTROLLER" == syntax' in text
        assert 'host_name" != syntax' in text
        assert 'host_name" != syn101' in text
        assert "scripts/tripwire/**" in text
        assert "renameat2" not in text
        assert "pgrep" not in text
        assert "--requeue" not in text


def test_program_allows_one_submission_without_automatic_followon():
    program = json.loads(PROGRAM.read_text())
    execution = program["execution"]
    assert execution["maximum_member_array_submissions"] == 1
    assert execution["maximum_aggregate_submissions"] == 1
    assert execution["manual_syntax_or_syn101_execution_allowed"] is False
    assert program["authorization"]["automatic_retry"] is False
    assert program["authorization"]["automatic_follow_on_after_aggregate"] is False
