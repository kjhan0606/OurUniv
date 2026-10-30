import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v67_program_freezes_target_free_probe_before_execution() -> None:
    program = json.loads(
        (
            REPO
            / "config/hong2021_v67_nonlocal_context_predictability_audit_program.json"
        ).read_text()
    )
    assert program["status"] == "frozen_before_audit_implementation_or_execution"
    assert program["target_free_predictors"]["components"] == 33
    assert sum(
        row["components"]
        for row in program["target_free_predictors"]["ordered_blocks"]
    ) == 33
    assert program["fixed_probe"]["alpha"] == 10.0
    assert program["permutation_control"]["replicates"] == 256
    assert "select_nonlocal_context_head" in program["selection_rules"]


def test_v67_program_forbids_identity_truth_refit_and_independent_access() -> None:
    program = json.loads(
        (
            REPO
            / "config/hong2021_v67_nonlocal_context_predictability_audit_program.json"
        ).read_text()
    )
    assert program["target_free_predictors"]["simulation_identity"] == "forbidden"
    assert program["target_free_predictors"]["density_target_or_residual"] == "forbidden"
    assert program["fixed_probe"]["density_model_training_or_refit"] is False
    assert program["firewall"]["validation_access"] == "forbidden"
    assert program["firewall"]["historical_EAGLE_access"] == "forbidden"
    assert program["firewall"]["independent_gate_locked"] is True
