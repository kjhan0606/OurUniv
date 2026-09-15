import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v65_program_freezes_multi_object_controls_before_execution() -> None:
    program = json.loads(
        (REPO / "config/hong2021_v65_structure_factorization_audit_program.json").read_text()
    )
    assert program["status"] == "frozen_before_audit_implementation_or_execution"
    assert program["immutable_train_queries"]["objects_per_domain"] == 16
    assert len(program["immutable_train_queries"]["TNG100"]) == 16
    assert program["frozen_rank_controls"]["members_per_query_and_control"] == 16
    assert program["frozen_rank_controls"]["spatially_permuted_rank_control"][
        "source"
    ] == "the source-balanced rank fields"
    assert "select_direct_pair_objective" in program["selection_rules"]


def test_v65_program_is_no_refit_and_keeps_independent_lock() -> None:
    program = json.loads(
        (REPO / "config/hong2021_v65_structure_factorization_audit_program.json").read_text()
    )
    assert program["resource_gate"]["training_or_refit"] is False
    assert program["firewall"]["new_development_access"] == "forbidden"
    assert program["firewall"]["historical_EAGLE_access"] == "forbidden"
    assert program["firewall"]["independent_gate_locked"] is True
