from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/hong2021_v84c0_unique_cell_tail_audit_lageunha.sh"


def test_v84c0_runner_is_frozen_cpu_only_and_avoids_ramses() -> None:
    source = RUNNER.read_text()
    assert "program_sha=8ffd8ec74ee0494c195c114069cc1eabccbce50f7d83091f2c3477f372a43489" in source
    assert "freeze_commit=0f958c453a42a135d5063d2276ffb37022b77116" in source
    assert source.count("taskset -c 64-79") == 2
    assert "taskset -c 0-" not in source
    assert "CUDA_VISIBLE_DEVICES" not in source
    assert "refuses existing output" in source


def test_v84c0_runner_audits_without_training_or_outer_evaluation() -> None:
    source = RUNNER.read_text()
    assert "hong2021_v84c0_unique_cell_tail_audit.py" in source
    assert "train.py" not in source
    assert "sample.py" not in source
    assert "group_gate.py" not in source
    assert "before_any_training" in source
