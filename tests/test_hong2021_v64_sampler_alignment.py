import json
from pathlib import Path

import torch

from hong2021_v64_sampler_alignment_audit import (
    _differentiable_bounded_inverse,
    classify,
)


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


def test_v64_implicit_empirical_quantile_has_finite_nonzero_gradient() -> None:
    parameters = torch.zeros(2, 15, 1, 1, 8, requires_grad=True)
    uniform = torch.linspace(0.05, 0.95, 16).reshape(2, 1, 1, 1, 8)
    value, error = _differentiable_bounded_inverse(parameters, uniform)
    value.sum().backward()
    assert float(error.max().detach()) <= 2.0e-6
    assert parameters.grad is not None
    assert torch.isfinite(parameters.grad).all()
    assert float(torch.linalg.vector_norm(parameters.grad)) > 0.0


def test_v64_pair_conflict_prevents_model_selection() -> None:
    classification, _, selected = classify(True, True, True, True, True, True)
    assert classification == (
        "sampler_aligned_tail_repair_is_not_a_sufficient_single_model_change"
    )
    assert selected is False


def test_v64_result_selects_structure_factorization_without_refit() -> None:
    record = json.loads((REPO / "config/hong2021_v64_result_record.json").read_text())
    assert record["status"] == (
        "complete_no_refit_audit_selected_train_only_structure_factorization"
    )
    assert record["tail_sampler_alignment"]["material_sampler_mismatch"] is False
    assert record["gradient_evidence"]["pair_gradient_scale_pass"] is True
    assert record["selected_next_step"]["action"] == (
        "freeze and run a no-refit multi-object train-only donor-dependence versus query-parameter structure audit"
    )
    assert record["firewall"]["independent_gate_locked"] is True
