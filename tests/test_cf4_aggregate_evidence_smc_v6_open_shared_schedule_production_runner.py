import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_lageunha.sh"
LAUNCHER = ROOT / "scripts/launch_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_lageunha.sh"
STATUS = ROOT / "scripts/status_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production.sh"


def _status(state, data):
    cache = state.with_name(state.name + "_cache")
    environment = os.environ.copy()
    environment.update({
        "CF4_V6_SHARED_STATUS_STATE": str(state),
        "CF4_V6_SHARED_STATUS_DATA": str(data),
        "CF4_V6_SHARED_STATUS_CACHE": str(cache),
    })
    return subprocess.run(
        [str(STATUS)], text=True, capture_output=True, env=environment,
        check=False,
    )


def test_shell_syntax_modes_and_forbidden_process_surfaces():
    for path in (RUNNER, LAUNCHER, STATUS):
        subprocess.run(["bash", "-n", str(path)], check=True)
        assert path.stat().st_mode & 0o111
    combined = RUNNER.read_text() + LAUNCHER.read_text()
    for forbidden in ("pgrep", "ps -", "squeue", "sbatch", "syn101"):
        assert forbidden not in combined
    assert "tmux new-session" not in LAUNCHER.read_text()
    assert "ssh " not in LAUNCHER.read_text()
    assert "/usr/bin/timeout --foreground --signal=TERM --kill-after=300s 12h" in RUNNER.read_text()


def test_status_absent_orphan_markerless_conflict_and_empty(tmp_path):
    state, data = tmp_path / "state", tmp_path / "data"
    value = _status(state, data)
    assert value.returncode == 3 and "not_started_fail_closed" in value.stdout
    state.mkdir()
    value = _status(state, data)
    assert value.returncode == 65 and "orphan_state" in value.stdout
    data.mkdir()
    value = _status(state, data)
    assert value.returncode == 65 and "markerless" in value.stdout
    (state / "RUNNING").write_text("status=running\n")
    (state / "FAILED").write_text("status=failed\n")
    value = _status(state, data)
    assert value.returncode == 65 and "conflicting" in value.stdout
    (state / "FAILED").unlink()
    (state / "RUNNING").write_text("")
    value = _status(state, data)
    assert value.returncode == 65 and "empty_marker" in value.stdout


@pytest.mark.parametrize(
    "marker,expected_rc", [("RUNNING", 0), ("FAILED", 1)]
)
def test_status_running_and_failed_are_read_only(tmp_path, marker, expected_rc):
    state, data = tmp_path / "state", tmp_path / "data"
    state.mkdir(); data.mkdir()
    (state / marker).write_text(f"status={marker.lower()}\n")
    if marker == "FAILED":
        (state / marker).chmod(0o444)
    before = sorted(path.name for path in state.iterdir())
    value = _status(state, data)
    assert value.returncode == expected_rc
    assert f"status={marker.lower()}" in value.stdout
    assert before == sorted(path.name for path in state.iterdir())


def test_status_complete_requires_result_and_manifest(tmp_path):
    state, data = tmp_path / "state", tmp_path / "data"
    cache = state.with_name(state.name + "_cache")
    state.mkdir(); data.mkdir(); cache.mkdir()
    (state / "COMPLETE").write_text("status=complete\n")
    (state / "COMPLETE").chmod(0o444)
    state.chmod(0o555); data.chmod(0o555); cache.chmod(0o555)
    value = _status(state, data)
    assert value.returncode == 65 and "artifacts_incomplete" in value.stdout
    data.chmod(0o755)
    (data / "result.json").write_text("{}")
    (data / "manifest.json").write_text("{}")
    data.chmod(0o555)
    value = _status(state, data)
    assert value.returncode == 65 and "invalid_complete_postcheck" in value.stdout


def test_status_terminal_modes_fail_closed(tmp_path):
    state, data = tmp_path / "state", tmp_path / "data"
    cache = state.with_name(state.name + "_cache")
    state.mkdir(); data.mkdir(); cache.mkdir()
    (state / "COMPLETE").write_text("status=complete\n")
    value = _status(state, data)
    assert value.returncode == 65 and "mode_contract" in value.stdout
    (state / "COMPLETE").unlink()
    (state / "FAILED").write_text("status=failed\n")
    value = _status(state, data)
    assert value.returncode == 65 and "failed_marker_mode" in value.stdout


def test_runner_and_launcher_gate_before_mutation_or_process_action():
    runner = RUNNER.read_text()
    launcher = LAUNCHER.read_text()
    assert "load_canonical_program(verify_file_hashes=True)" in runner
    assert "production execution remains intentionally unauthorized" in runner
    assert "load_canonical_program(verify_file_hashes=True)" in launcher
    assert "production launcher remains intentionally unauthorized" in launcher
    for mutation in ("mkdir ", "flock ", "rm ", "mv "):
        assert mutation not in runner
        assert mutation not in launcher
