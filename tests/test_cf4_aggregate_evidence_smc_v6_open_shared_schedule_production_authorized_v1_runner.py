from __future__ import annotations

from pathlib import Path
import os
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_authorized_v1_lageunha.sh"
LAUNCHER = ROOT / "scripts/launch_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_authorized_v1_lageunha.sh"
STATUS = ROOT / "scripts/status_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_authorized_v1.sh"


def test_shells_are_syntax_valid_and_have_no_process_scan_or_remote_backend():
    subprocess.run(["bash", "-n", str(RUNNER), str(LAUNCHER), str(STATUS)], check=True)
    combined = "\n".join(path.read_text() for path in (RUNNER, LAUNCHER, STATUS))
    for forbidden in ("pgrep", "squeue", "tmux", "ssh ", "syn101"):
        assert forbidden not in combined
    assert "/usr/bin/timeout --foreground --signal=TERM --kill-after=300s 12h" in RUNNER.read_text()
    assert "CUDA_VISIBLE_DEVICES=" in RUNNER.read_text()
    assert "MALLOC_ARENA_MAX=2" in RUNNER.read_text()
    assert "available_cpus=$(nproc)" in RUNNER.read_text()
    assert "_supervisor_force_failed" in RUNNER.read_text()
    assert "runtime_rc" in RUNNER.read_text()


def test_authorization_precedes_host_resources_and_any_runtime_action():
    runner = RUNNER.read_text()
    gate = runner.index("validate_authorization(load_program())")
    assert gate < runner.index("host=$(hostname)")
    assert gate < runner.index("MemAvailable")
    assert gate < runner.index("/usr/bin/timeout")
    launcher = LAUNCHER.read_text()
    assert launcher.index("validate_authorization(load_program())") < launcher.index("host=$(hostname)")
    assert 'exec "$runner"' in launcher


def test_status_is_read_only_and_fail_closed_without_polling():
    text = STATUS.read_text()
    assert "status=invalid_state_no_marker" in text
    assert "status=invalid_state_conflicting_markers" in text
    assert "status=invalid_marker_type_or_empty" in text
    assert "status=invalid_dangling_receipt_root" in text
    assert "RUNNING COMPLETE FAILED" in text
    assert "_read_only_complete_status" in text
    assert "_read_only_failed_status" in text
    assert not any(token in text for token in ("pgrep", "ps ", "squeue", "while ", "sleep "))


def test_exact_seven_additive_paths_exist_with_modes():
    expected = {
        "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_program_v1.json": 0o644,
        "src/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_v1.py": 0o644,
        "scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_authorized_v1_lageunha.sh": 0o755,
        "scripts/launch_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_authorized_v1_lageunha.sh": 0o755,
        "scripts/status_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_authorized_v1.sh": 0o755,
        "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_v1.py": 0o644,
        "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_authorized_v1_runner.py": 0o644,
    }
    for relative, mode in expected.items():
        path = ROOT / relative
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_mode & 0o777 == mode


def test_actual_runner_signal_trap_terminates_child_and_calls_supervisor(tmp_path: Path):
    runner = RUNNER.read_text()
    start = runner.index("handle_runner_signal() {")
    end = runner.index("\n}\n", start) + 3
    function = runner[start:end]
    ready = tmp_path / "ready"
    sealed = tmp_path / "sealed"
    script = f'''set -Eeuo pipefail
runtime_pid=
seal_supervisor_failed() {{ printf '%s' "$1" > {sealed}; }}
{function}
trap 'handle_runner_signal TERM 143' TERM
sleep 30 &
runtime_pid=$!
printf ready > {ready}
wait "$runtime_pid"
'''
    process = subprocess.Popen(["bash", "-c", script])
    try:
        for _ in range(100):
            if ready.exists():
                break
            time.sleep(0.01)
        assert ready.exists()
        os.kill(process.pid, signal.SIGTERM)
        assert process.wait(timeout=5) == 143
        assert sealed.read_text() == "runner_signal_TERM"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
