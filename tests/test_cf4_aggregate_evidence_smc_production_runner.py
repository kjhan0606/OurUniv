import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_cf4_aggregate_evidence_smc_production_lageunha.sh"
LAUNCHER = ROOT / "scripts/launch_cf4_aggregate_evidence_smc_production_lageunha.sh"
STATUS = ROOT / "scripts/status_cf4_aggregate_evidence_smc_production.sh"


def _bash_syntax(path):
    subprocess.run(["bash", "-n", str(path)], check=True)


def _status(state, data):
    environment = dict(os.environ)
    environment.update({
        "CF4_AGGREGATE_SMC_STATUS_STATE": str(state),
        "CF4_AGGREGATE_SMC_STATUS_DATA": str(data),
    })
    return subprocess.run(
        ["bash", str(STATUS)],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )


def test_runner_is_fixed_fail_closed_and_has_no_monitoring_or_retry_loop():
    _bash_syntax(RUNNER)
    text = RUNNER.read_text()
    assert "readonly expected_host=lageunha" in text
    assert "readonly source_commit=6630b6b04ab02e513d47f1667617384894eb349f" in text
    assert "readonly capability_commit=22587e47232782feb4c08768d8f64d853d76e62b" in text
    assert "aggregate_evidence_smc_v1" in text
    assert "aggregate_evidence_smc_v1_run" in text
    assert "available_kib < 41943040" in text
    assert "memory_kib < 67108864" in text
    assert 'CUDA_VISIBLE_DEVICES=""' in text
    assert "worker_processes=8" in text
    assert "threads_per_worker=1" in text
    assert "replicates_sequential=true" in text
    assert "flock -n 9" in text
    assert "trap finish EXIT" in text
    assert 'mkdir "$state"' in text and 'mkdir "$data"' in text
    assert "load_canonical_program(verify_file_hashes=True)" in text
    assert "validate_published_bundle" in text
    assert text.count('git -C "$repo" rev-parse HEAD') >= 2
    assert text.count("diff\", \"--quiet\"") >= 2
    assert not re.search(r"\b(?:pgrep|ps)\b", text)
    assert not re.search(r"\b(?:while|until)\b", text)
    assert "sleep " not in text
    assert "retry" not in text.lower().replace(
        "automatic_retry_scale_retune_or_follow_on=false", ""
    ).replace("automatic_retry", "")


def test_runner_marker_lifecycle_is_exclusive_and_scientific_complete_only():
    text = RUNNER.read_text()
    assert 'readonly running="$state/RUNNING"' in text
    assert 'readonly complete="$state/COMPLETE"' in text
    assert 'readonly failed="$state/FAILED"' in text
    assert 'if (( rc == 0 )) && [[ "${validated_complete:-false}" == true' in text
    assert 'validated_complete=${validated_complete,,}' in text
    assert "complete_pass_production_smc" in text
    assert "complete_scientific_fail_production_smc" in text
    assert "invalid_execution_or_postcheck_failure" in text
    assert 'mv "$marker_tmp" "$complete"' in text
    assert 'mv "$marker_tmp" "$failed"' in text
    assert 'rm -f "$running"' in text
    reservation = text.index('started_at=$(date --iso-8601=seconds)')
    preflight = text[:reservation]
    assert 'mkdir "$state"' not in preflight
    assert 'mkdir "$data"' not in preflight


def test_launcher_is_one_fixed_lageunha_tmux_session_without_follow_on():
    _bash_syntax(LAUNCHER)
    text = LAUNCHER.read_text()
    assert "readonly host=lageunha" in text
    assert "readonly session=cf4-aggregate-evidence-smc-v1" in text
    assert "run_cf4_aggregate_evidence_smc_production_lageunha.sh" in text
    assert "production SMC execution remains unauthorized" in text
    assert "tmux has-session" in text
    assert "tmux new-session -d" in text
    assert not re.search(r"\b(?:pgrep|ps)\b", text)
    assert "sleep " not in text


def test_status_fails_closed_for_absent_and_orphan_namespaces(tmp_path):
    _bash_syntax(STATUS)
    absent = _status(tmp_path / "state", tmp_path / "data")
    assert absent.returncode == 3
    assert "not_started_fail_closed" in absent.stdout

    data_only = tmp_path / "data_only"
    data_only.mkdir()
    orphan_data = _status(tmp_path / "missing_state", data_only)
    assert orphan_data.returncode == 65
    assert "orphan_data_without_state" in orphan_data.stdout

    state_only = tmp_path / "state_only"
    state_only.mkdir()
    orphan_state = _status(state_only, tmp_path / "missing_data")
    assert orphan_state.returncode == 65
    assert "orphan_state_without_data" in orphan_state.stdout


def test_status_fails_closed_for_markerless_empty_conflict_and_incomplete(tmp_path):
    cases = ("markerless", "empty", "conflict", "incomplete")
    for case in cases:
        state = tmp_path / f"{case}_state"
        data = tmp_path / f"{case}_data"
        state.mkdir()
        data.mkdir()
        if case == "empty":
            (state / "RUNNING").touch()
        elif case == "conflict":
            (state / "RUNNING").write_text("status=running\n")
            (state / "FAILED").write_text("status=failed\n")
        elif case == "incomplete":
            (state / "COMPLETE").write_text("status=complete\n")
        result = _status(state, data)
        assert result.returncode == 65
        assert "status=invalid" in result.stdout


def test_status_reports_running_without_process_table_scan_and_failed_terminal(tmp_path):
    running_state = tmp_path / "running_state"
    running_data = tmp_path / "running_data"
    running_state.mkdir()
    running_data.mkdir()
    (running_state / "RUNNING").write_text("status=running\nstage=smc\n")
    running = _status(running_state, running_data)
    assert running.returncode == 0
    assert "status=running" in running.stdout

    failed_state = tmp_path / "failed_state"
    failed_data = tmp_path / "failed_data"
    failed_state.mkdir()
    failed_data.mkdir()
    (failed_state / "FAILED").write_text("status=failed\nexit_code=1\n")
    failed = _status(failed_state, failed_data)
    assert failed.returncode == 1
    assert "status=failed" in failed.stdout
    assert not re.search(r"\b(?:pgrep|ps)\b", STATUS.read_text())
