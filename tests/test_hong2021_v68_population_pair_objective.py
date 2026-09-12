import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v68_program_aggregates_objects_before_log_ratio() -> None:
    program = json.loads(
        (REPO / "config/hong2021_v68_population_pair_objective_audit_program.json").read_text()
    )
    assert program["status"] == "frozen_before_audit_implementation_or_execution"
    assert program["rank_and_pair_probe"]["rank_members_per_query"] == 8
    folds = program["immutable_queries_and_folds"]["fold_positions"]
    assert sorted(value for fold in folds.values() for value in fold) == list(range(16))
    assert "arithmetic-mean predicted pair moments" in program[
        "population_objective"
    ]["construction"]
    assert "select_population_pair_objective" in program["selection_rules"]


def test_v68_program_is_no_refit_and_independent_locked() -> None:
    program = json.loads(
        (REPO / "config/hong2021_v68_population_pair_objective_audit_program.json").read_text()
    )
    assert program["resource_gate"]["training_or_refit"] is False
    assert program["authorized_if_selected"]["authorization_is_not_training"] is True
    assert program["firewall"]["new_development_access"] == "forbidden"
    assert program["firewall"]["historical_EAGLE_access"] == "forbidden"
    assert program["firewall"]["independent_gate_locked"] is True
