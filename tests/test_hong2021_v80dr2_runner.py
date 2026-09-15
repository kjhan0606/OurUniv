from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/hong2021_v80dr2_report_only_lageunha.sh"


def test_v80dr2_runner_is_strictly_report_only() -> None:
    source = RUNNER.read_text()
    assert "program_sha=cc84825d29bf969cce8bba5355356c0da78e6fe2e41198ba8329bc621f2e9db6" in source
    assert "program_freeze=508b62608626225c9997d952740e1c1d72da1b13" in source
    assert "hong2021_v80dr2_report_only.py" in source
    assert "hong2021_v80_evaluate.py" not in source
    assert "hong2021_v80_sample.py" not in source
    assert "hong2021_v80dr_metadata_recovery.py" not in source
    assert "hong2021_v80_manifest.py" not in source
    assert "hong2021_v79_complete_gate.py" not in source
    assert "failed_terminal_V80DR2_no_additional_report_retry" in source


def test_v80dr2_runner_stays_off_ramses_cpus() -> None:
    source = RUNNER.read_text()
    assert source.count("taskset -c 64-95") == 2
    assert "taskset -c 0-" not in source
