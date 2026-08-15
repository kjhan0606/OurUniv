from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/run_cf4_lg_v8_mode_release_reference.sh"
LAUNCHER = REPO / "scripts/launch_cf4_lg_v8_mode_release_reference.sh"


def test_runner_is_marker_driven_and_never_polls_processes():
    source = RUNNER.read_text()
    assert "pgrep" not in source
    assert "while " not in source
    assert "flock -n 9" in source
    assert 'trap finish EXIT' in source
    assert 'readonly running="$state/RUNNING"' in source
    assert 'readonly complete="$state/COMPLETE"' in source
    assert 'readonly failed="$state/FAILED"' in source
    assert 'XLA_PYTHON_CLIENT_PREALLOCATE=false' in source
    assert 'PYTHONNOUSERSITE=1' in source
    assert 'PYTHONPATH="$repo/src"' in source
    assert 'jax.default_backend()' in source
    assert 'exec >"$log" 2>&1' in source
    assert 'mv "$marker_tmp" "$complete"' in source
    assert 'mv "$marker_tmp" "$failed"' in source


def test_launcher_detaches_once_with_tmux_and_has_no_wait_loop():
    source = LAUNCHER.read_text()
    assert "pgrep" not in source
    assert "while " not in source
    assert 'readonly session=cf4-v8-ref' in source
    assert 'tmux has-session -t "$session"' in source
    assert 'tmux new-session -d -s "$session"' in source
    assert 'CUDA_VISIBLE_DEVICES=0' in source
