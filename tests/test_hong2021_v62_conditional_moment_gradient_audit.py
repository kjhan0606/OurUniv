import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from hong2021_v62_conditional_moment_gradient_audit import (
    PROGRAM_SHA256,
    classify,
    conditional_log_moment_score,
    gradient_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/hong2021_v62_conditional_moment_gradient_audit_program.json"


def test_program_is_byte_bound_and_locked() -> None:
    assert hashlib.sha256(PROGRAM.read_bytes()).hexdigest() == PROGRAM_SHA256
    row = json.loads(PROGRAM.read_text())
    assert row["status"] == "frozen_before_audit_implementation_or_execution"
    assert row["candidate_objective"]["future_model_coefficient_if_selected"] == 0.1
    assert row["firewall"]["development_accessed"] is False
    assert row["firewall"]["independent_gate_locked"] is True


def test_conditional_log_moment_score_has_declared_identity() -> None:
    predicted = torch.tensor([2.0, 4.0], dtype=torch.float64, requires_grad=True)
    truth = torch.tensor([1.0, 8.0], dtype=torch.float64)
    score = conditional_log_moment_score(predicted, truth)
    assert float(score.detach()) == pytest.approx(0.5 * math.log(2.0) ** 2)
    score.backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[0] > 0.0
    assert predicted.grad[1] < 0.0


def test_conditional_log_moment_score_rejects_nonpositive_values() -> None:
    with pytest.raises(ValueError):
        conditional_log_moment_score(torch.tensor([0.0]), torch.tensor([1.0]))


def test_gradient_metrics_are_normalized_by_selected_voxels() -> None:
    row = gradient_metrics(torch.tensor([3.0, 4.0]), 5)
    assert row["L2"] == 5.0
    assert row["L2_per_selected_voxel"] == 1.0
    assert row["maximum_absolute"] == 4.0


def test_classification_follows_frozen_branches() -> None:
    assert classify(True, True, True, True) == (
        "direct_conditional_log_physical_moment_objective_has_gate_aligned_optimization_scale",
        "freeze_one_V63_model_that_retains_the_complete_V56_objective_and_adds_only_the_coefficient_0.1_conditional_log_physical_moment_term",
        True,
    )
    assert classify(False, True, True, True)[0].endswith("numerically_unresolved")
    assert classify(True, True, False, True)[0].endswith("not_optimization_feasible")
