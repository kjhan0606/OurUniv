from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/hong2021_v84b_group_spliced_tail_lageunha.sh"
AMENDED_RUNNER = REPO / "scripts/hong2021_v84b1_group_spliced_tail_lageunha.sh"


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


def test_v84b1_runner_freezes_amended_program_and_new_outputs() -> None:
    source = AMENDED_RUNNER.read_text()
    assert "program_sha=f07ca34f9e5ab57c2625aba138dba8056193d049bae591abadcb820197896175" in source
    assert "freeze_commit=c0f1ae2498d0f4b75f4b182a94d3e6a51c3953b2" in source
    assert "v84b1_group_spliced_tail" in source
    assert source.count("taskset -c 64-79") == 4
    assert "taskset -c 0-" not in source
    assert "CUDA_VISIBLE_DEVICES=0" in source
    assert "sample.py" not in source
    assert "validation" not in source
