from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/run_cf4_lg_v8_mode_release_audit.sh"
LAUNCHER = REPO / "scripts/launch_cf4_lg_v8_mode_release_audit.sh"
MONITOR = REPO / "scripts/monitor_cf4_lg_v8_mode_release_audit.sh"
MONITOR_LAUNCHER = REPO / "scripts/launch_cf4_lg_v8_mode_release_audit_monitor.sh"


def test_audit_runner_is_detached_marker_driven_and_environment_pinned():
    runner = RUNNER.read_text()
    launcher = LAUNCHER.read_text()
    assert "pgrep" not in runner
    assert "while " not in runner
    assert "flock -n 9" in runner
    assert "PYTHONNOUSERSITE=1" in runner
    assert 'PYTHONPATH="$repo/src"' in runner
    assert 'jax.default_backend()' in runner
    assert 'tmux new-session -d -s "$session"' in launcher


def test_audit_monitor_is_low_frequency_marker_only_and_self_terminating():
    monitor = MONITOR.read_text()
    launcher = MONITOR_LAUNCHER.read_text()
    assert "pgrep" not in monitor
    assert 'readonly interval=${MONITOR_INTERVAL_SECONDS:-300}' in monitor
    assert 'sleep "$interval"' in monitor
    assert '"$state/COMPLETE"' in monitor
    assert '"$state/FAILED"' in monitor
    assert 'tmux new-session -d -s "$session"' in launcher
