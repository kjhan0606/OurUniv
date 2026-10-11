import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v66_program_freezes_conditional_gradient_probe_before_execution() -> None:
    program = json.loads(
        (
            REPO
            / "config/hong2021_v66_conditional_gradient_routing_audit_program.json"
        ).read_text()
    )
    assert program["status"] == "frozen_before_audit_implementation_or_execution"
    assert program["gradient_targets"]["bias_control"]["components"] == 15
    assert program["gradient_targets"]["conditional_output_weight"][
        "components"
    ] == 480
    assert program["gradient_targets"]["joint_final_output_layer"][
        "components"
    ] == 495
    assert program["frozen_rank_and_pair_probe"]["rank_members_per_query"] == 4
    assert "select_final_output_layer_pair_model" in program["selection_rules"]


def test_v66_program_forbids_refit_and_keeps_independent_gate_locked() -> None:
    program = json.loads(
        (
            REPO
            / "config/hong2021_v66_conditional_gradient_routing_audit_program.json"
        ).read_text()
    )
    assert program["resource_gate"]["training_or_refit"] is False
    assert program["authorized_if_selected"]["authorization_is_not_training"] is True
    assert program["firewall"]["new_development_access"] == "forbidden"
    assert program["firewall"]["historical_EAGLE_access"] == "forbidden"
    assert program["firewall"]["independent_gate_locked"] is True
