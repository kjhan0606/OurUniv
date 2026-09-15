import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v69_program_freezes_two_prefix_streams_before_execution() -> None:
    program = json.loads(
        (REPO / "config/hong2021_v69_pair_estimator_rank_convergence_program.json").read_text()
    )
    assert program["status"] == "frozen_before_audit_implementation_or_execution"
    assert program["donor_streams"]["rank_levels"] == [8, 16, 32, 64]
    assert program["donor_streams"]["stream_A_seed"] != program["donor_streams"]["stream_B_seed"]
    assert "select_rank64_estimator" in program["convergence_rules"]


def test_v69_program_forbids_gradient_refit_and_independent_access() -> None:
    program = json.loads(
        (REPO / "config/hong2021_v69_pair_estimator_rank_convergence_program.json").read_text()
    )
    assert program["resource_gate"]["training_or_refit"] is False
    assert program["firewall"]["gradient_or_optimizer_step"] == "forbidden"
    assert program["firewall"]["validation_access"] == "forbidden"
    assert program["firewall"]["independent_gate_locked"] is True
