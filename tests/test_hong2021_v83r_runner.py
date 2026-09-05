from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/hong2021_v83r_metadata_recovery_lageunha.sh"


def test_runner_orders_identity_proof_before_evaluation_and_gate() -> None:
    source = RUNNER.read_text()
    recovery = source.index("hong2021_v83r_metadata_recovery.py")
    evaluation = source.index("hong2021_v83_evaluate.py")
    gate = source.index("hong2021_v83r_development_gate.py")
    assert recovery < evaluation < gate
    assert "hong2021_v83_sample.py" not in source
    assert "resampling" not in source.lower()
    assert "taskset -c 64-79" in source
    assert "complete_V83R_consumed_gate_pass_waiting_user_approval" in source


def test_runner_binds_frozen_program_checkpoint_and_train_gate() -> None:
    source = RUNNER.read_text()
    assert "a4387f50d7d58de479f080b10bc2426bd1b97cde44ec83c5661591ed6f92ddf4" in source
    assert "fc06559221e3430f95dfd3de0131e3646d364d631cab462faabec55e1eb9572d" in source
    assert "d615f5b5d1a89ee8e007d3ccab1d35ec222907484814d984ce08b50c2438d032" in source
