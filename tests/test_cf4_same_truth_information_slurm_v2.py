import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_same_truth_information_budget_program_v2.json"
SOURCE = ROOT / "src/cf4_same_truth_information_budget_v2.py"
PILOT = ROOT / "scripts/run_cf4_same_truth_information_pilot_v2.sbatch"
MEMBERS = ROOT / "scripts/run_cf4_same_truth_information_members_v2.sbatch"
AGGREGATE = ROOT / "scripts/run_cf4_same_truth_information_aggregate_v2.sbatch"


def test_v2_pilot_is_one_nonarray_cpu_member_and_refuses_overwrite():
    text = PILOT.read_text()
    assert "#SBATCH --array" not in text
    assert "#SBATCH --mem=3072M" in text
    assert "#SBATCH --time=00:30:00" in text
    assert "--mock-index 0" in text
    assert '[[ ! -e "$output" ]]' in text
    assert "validate-member" in text


def test_v2_production_requires_completed_pilot_and_exact_array():
    text = MEMBERS.read_text()
    assert "#SBATCH --array=0-63%8" in text
    assert "#SBATCH --mem=3072M" in text
    assert "PILOT_JOB_ID" in text
    assert '[[ "$pilot_state" == "COMPLETED|0:0" ]]' in text
    assert '[[ ! -e "$output"' in text


def test_v2_aggregate_is_afterok_target_and_no_live_dependency_assertion():
    text = AGGREGATE.read_text()
    assert "#SBATCH --mem=3072M" in text
    assert "MEMBER_ARRAY_JOB_ID" in text and "PILOT_JOB_ID" in text
    assert "cf4_same_truth_information_v2_afterok_${MEMBER_ARRAY_JOB_ID}" in text
    assert "Dependency=afterok:" not in text
    assert '[[ -d "$members_root" && ! -e "$output"' in text


def test_all_v2_runners_pin_hashes_and_forbid_manual_controller_execution():
    program_sha = hashlib.sha256(PROGRAM.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    for runner in (PILOT, MEMBERS, AGGREGATE):
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
