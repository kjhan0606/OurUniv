import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_v64_program_is_no_refit_train_only_and_independent_locked() -> None:
    program = json.loads(
        (REPO / "config/hong2021_v64_sampler_alignment_audit_program.json").read_text()
    )
    assert program["status"] == "frozen_before_audit_implementation_or_execution"
    assert program["resource_gate"]["training_or_refit"] is False
    assert program["frozen_empirical_rank_ensemble"][
        "development_rank_or_selection_access"
    ] is False
    assert program["firewall"]["new_development_access"] == "forbidden"
    assert program["firewall"]["historical_EAGLE_access"] == "forbidden"
    assert program["firewall"]["independent_gate_locked"] is True


def test_v64_program_freezes_sampler_and_pair_diagnostics_before_execution() -> None:
    program = json.loads(
        (REPO / "config/hong2021_v64_sampler_alignment_audit_program.json").read_text()
    )
    assert program["frozen_empirical_rank_ensemble"]["members_per_query"] == 16
    assert program["fixed_train_batch"]["query_object_index"] == 0
    assert program["train_only_sub_mpc_compatibility_diagnostic"][
        "physical_separations_mpc_h"
    ] == [0.3125, 0.625, 0.9375]
    assert "select_sampler_aligned_tail_model" in program["selection_rules"]
