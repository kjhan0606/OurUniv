from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/hong2021_v84a_group_tail_attribution_lageunha.sh"
AMENDED_RUNNER = REPO / "scripts/hong2021_v84a1_group_tail_attribution_lageunha.sh"


def test_v84a_runner_is_frozen_and_avoids_ramses_cores() -> None:
    source = RUNNER.read_text()
    assert "program_sha=549439e97d5c90665d5a3f40068b5d403459bbc40b61ed8f54e8155467a322c2" in source
    assert "freeze_commit=ec1daa3b1a06b74adbe4fe76b6e059b2efa43fb3" in source
    assert source.count("taskset -c 64-79") == 2
    assert "taskset -c 0-" not in source
    assert "CUDA_VISIBLE_DEVICES=0" in source
    assert "refuses existing output" in source
    assert "complete_V84A_waiting_result_record_and_review" in source


def test_v84a_runner_only_audits_without_training_or_independent_data() -> None:
    source = RUNNER.read_text()
    assert "hong2021_v84a_group_tail_attribution.py" in source
    assert "train.py" not in source
    assert "Astrid" not in source
    assert "historical_EAGLE" not in source


def test_v84a1_runner_freezes_amended_program_and_new_output() -> None:
    source = AMENDED_RUNNER.read_text()
    assert "program_sha=6442d2dc41bd28b67a6efc86fa7129afc4a0ca0a1865cdee25fdb6b317a86621" in source
    assert "freeze_commit=a4ddf21f636efacd7df40b729221dd0596d21b6f" in source
    assert "tng100_simba_swift_v84a1_group_tail_attribution" in source
    assert source.count("taskset -c 64-79") == 2
    assert "taskset -c 0-" not in source
    assert "CUDA_VISIBLE_DEVICES=0" in source
    assert "train.py" not in source
