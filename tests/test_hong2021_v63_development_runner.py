import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v63_development_runner_requires_sealed_train_pass_before_sampling() -> None:
    text = (REPO / "scripts/hong2021_v63_development_lageunha.sh").read_text()
    authorization = text.index("complete_train_gate_pass_waiting_locked_development")
    sampling = text.index('printf "%s\\n" development_sampling')
    assert authorization < sampling
    assert "V63 development requires the sealed train-gate pass" in text


def test_v63_development_runner_keeps_independent_gate_locked() -> None:
    text = (REPO / "scripts/hong2021_v63_development_lageunha.sh").read_text()
    assert "complete_development_gate_pass_waiting_explicit_EAGLE_approval" in text
    assert "complete_development_gate_failure_independent_gate_locked" in text
    assert "eagle" not in text.lower().replace("eagle_approval", "")


def test_v63_train_record_authorizes_only_locked_development() -> None:
    record = json.loads(
        (REPO / "config/hong2021_v63_train_result_record.json").read_text()
    )
    assert record["status"] == (
        "complete_train_gate_pass_authorized_locked_development"
    )
    assert record["train_only_mechanism_decision"]["train_mechanism_pass"] is True
    authorization = record["authorization"]
    assert authorization["locked_V63_development_sampling_and_evaluation_allowed"]
    assert authorization["new_training_or_refit_allowed"] is False
    assert authorization["independent_EAGLE_gate_allowed"] is False
