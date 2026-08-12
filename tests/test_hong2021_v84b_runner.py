from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/hong2021_v84b_group_spliced_tail_lageunha.sh"


def test_v84b_runner_is_frozen_and_avoids_ramses_cores() -> None:
    source = RUNNER.read_text()
    assert "program_sha=5a6c89fc032f33a1245f43d731189e5e4114c2e564acef55e96d2d62a45c03bb" in source
    assert "freeze_commit=243fd09640782eaeef44b8dae85ab73fb58b1df4" in source
    assert source.count("taskset -c 64-79") == 4
    assert "taskset -c 0-" not in source
    assert "CUDA_VISIBLE_DEVICES=0" in source
    assert "refuses existing output" in source


def test_v84b_runner_stops_after_group_gate_without_other_payloads() -> None:
    source = RUNNER.read_text()
    assert "hong2021_v84b_preflight.py" in source
    assert "hong2021_v84b_train.py" in source
    assert "hong2021_v84b_group_gate.py" in source
    assert "sample.py" not in source
    assert "validation" not in source
    assert "Astrid" not in source
    assert "historical_EAGLE" not in source
    assert "waiting_result_record_before_any_production_refit" in source
