from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v1.sbatch"
STATUS = ROOT / "scripts/status_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v1.sh"
PROGRAM = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_program_v1.json"
SOURCE = ROOT / "src/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_authorized_v1.py"
SOURCE_TEST = ROOT / "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_authorized_v1.py"
RUNNER_TEST = Path(__file__)


def test_sbatch_and_status_are_syntax_valid_and_have_no_forbidden_backend() -> None:
    for path in (SBATCH, STATUS):
        subprocess.run(["bash", "-n", str(path)], check=True)
    combined = SBATCH.read_text() + STATUS.read_text()
    for forbidden in ("pgrep", " pgrep", "ps -", "squeue", "sacct", "tmux", "ssh ", "syn101"):
        assert forbidden not in combined
    assert "while " not in combined
    assert "--requeue" not in combined


def test_sbatch_header_and_timeout_child_argv_are_exact() -> None:
    text = SBATCH.read_text()
    required = {
        "#SBATCH --partition=debug",
        "#SBATCH --nodelist=grammar-debug",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=12",
        "#SBATCH --mem=96G",
        "#SBATCH --time=12:00:00",
        "#SBATCH --signal=B:TERM@300",
        "#SBATCH --no-requeue",
        "#SBATCH --export=NONE",
        "#SBATCH --chdir=/home/kjhan/BACKUP/CF4",
        "#SBATCH --open-mode=truncate",
    }
    assert required <= set(text.splitlines())
    assert "/usr/bin/timeout --foreground --signal=TERM --kill-after=240s 12h" in text
    assert "/usr/bin/env PYTHONDONTWRITEBYTECODE=1" in text
    assert "/home/kjhan/miniconda3/envs/circle/bin/python -m" in text
    assert "cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_authorized_v1" in text
    assert "CUDA_VISIBLE_DEVICES=" in text
    assert text.startswith("#!/bin/bash\n")
    assert "unset BASH_ENV ENV PYTHONSTARTUP LD_PRELOAD LD_LIBRARY_PATH" in text


def test_authorization_and_slurm_provenance_precede_child_and_runtime() -> None:
    text = SBATCH.read_text()
    preflight = text.index("validate_authorization(load_program())")
    clean = text.index("/usr/bin/git -C \"$repo\" status --porcelain=v1 --untracked-files=all")
    child = text.index("/usr/bin/timeout --foreground")
    assert clean < preflight < child
    assert "':(exclude)scripts/tripwire/**'" in text
    assert "_read_scheduler_context(); _require_resources(); _require_runtime_environment()" in text
    assert "runtime_pid=$!" in text
    assert "kill -TERM \"$runtime_pid\"" in text
    assert "killall" not in text and "pkill" not in text


def test_status_is_artifact_only_and_fail_closed() -> None:
    text = STATUS.read_text()
    assert "status=absent" in text
    assert "status=invalid_state_no_marker" in text
    assert "status=invalid_state_conflicting_markers" in text
    assert "_read_only_running_status" in text
    assert "_read_only_complete_status" in text
    assert "_read_only_failed_status" in text
    assert "/proc" not in text and "SLURM_JOB_ID" not in text


def test_exact_six_additive_files_exist_with_frozen_modes() -> None:
    expected = {
        PROGRAM: 0o644,
        SOURCE: 0o644,
        SBATCH: 0o755,
        STATUS: 0o755,
        SOURCE_TEST: 0o644,
        RUNNER_TEST: 0o644,
    }
    assert len(expected) == 6
    for path, wanted in expected.items():
        assert path.is_file() and not path.is_symlink()
        assert os.stat(path).st_mode & 0o777 == wanted


def test_status_script_does_not_mutate_or_submit() -> None:
    text = STATUS.read_text()
    for forbidden in ("sbatch", "scancel", "srun", "mkdir", "chmod", "rm ", "mv ", "touch "):
        assert forbidden not in text


def test_actual_batch_term_trap_kills_only_known_child_and_seals(tmp_path: Path) -> None:
    text = SBATCH.read_text()
    start = text.index("handle_batch_signal() {")
    end = text.index("\n}\n", start) + 3
    function = text[start:end]
    ready = tmp_path / "ready"
    sealed = tmp_path / "sealed"
    script = f'''set -Eeuo pipefail
runtime_pid=
seal_supervisor_failed() {{ printf '%s' "$1" > {sealed}; }}
{function}
trap 'handle_batch_signal TERM 143' TERM
sleep 30 &
runtime_pid=$!
printf ready > {ready}
wait "$runtime_pid"
'''
    process = subprocess.Popen(["/bin/bash", "-c", script])
    try:
        for _ in range(200):
            if ready.exists():
                break
            time.sleep(0.01)
        assert ready.exists()
        os.kill(process.pid, signal.SIGTERM)
        assert process.wait(timeout=5) == 143
        assert sealed.read_text() == "slurm_batch_signal_TERM"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
