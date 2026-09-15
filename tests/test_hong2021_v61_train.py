import json
from pathlib import Path

import pytest
import torch

from hong2021_v18_init import sha256_file
from hong2021_v50_network import INITIAL_BIASES
from hong2021_v61_preflight import (
    GRID_CELLS,
    PROGRAM_SHA256,
    reachable_survival_grid_score,
)
from hong2021_v61_train import (
    PREFLIGHT_IMPLEMENTATION_SHA256,
    PREFLIGHT_RECORD_SHA256,
    PREFLIGHT_SHA256,
    training_grid_score,
)


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/hong2021_v61_reachable_support_model_program.json"
RECORD = ROOT / "config/hong2021_v61_preflight_record.json"


def test_training_is_bound_to_program_preflight_record_and_approved_score():
    program = json.loads(PROGRAM.read_text())
    record = json.loads(RECORD.read_text())
    assert sha256_file(PROGRAM) == PROGRAM_SHA256
    assert sha256_file(RECORD) == PREFLIGHT_RECORD_SHA256
    assert sha256_file(ROOT / "src/hong2021_v61_preflight.py") == (
        PREFLIGHT_IMPLEMENTATION_SHA256
    )
    assert record["preflight"]["sha256"] == PREFLIGHT_SHA256
    assert record["authorization"]["training_allowed"] is True
    assert record["authorization"]["training_steps"] == 12_000
    assert program["firewall"]["development_access_before_train_gate_pass"] == (
        "forbidden"
    )
    assert program["firewall"]["independent_gate_locked"] is True


def test_training_grid_score_exactly_reuses_preflight_objective_and_gradient():
    initial = (
        torch.tensor(INITIAL_BIASES, dtype=torch.float32)
        .reshape(1, 15, 1, 1, 1)
        .expand(1, 15, 2, 2, 2)
        .clone()
    )
    train_parameters = initial.clone().requires_grad_(True)
    reference_parameters = initial.clone().requires_grad_(True)
    target = torch.linspace(-0.2, 0.3, 8).reshape(1, 1, 2, 2, 2)
    backbone = torch.linspace(0.1, -0.1, 8).reshape_as(target)
    thresholds = torch.linspace(-2.0, 3.0, GRID_CELLS)
    weights = torch.arange(1, GRID_CELLS + 1, dtype=torch.float64)
    weights /= weights.sum()
    score, components = training_grid_score(
        train_parameters,
        target,
        backbone,
        0.0,
        1.0,
        thresholds,
        weights,
        diagnostics=True,
    )
    reference, reference_components, _, _ = reachable_survival_grid_score(
        reference_parameters,
        target,
        backbone,
        0.0,
        1.0,
        thresholds,
        weights,
    )
    assert components is not None
    assert torch.allclose(score, reference, rtol=1e-13, atol=1e-13)
    assert torch.equal(components, reference_components)
    train_gradient = torch.autograd.grad(score, train_parameters)[0]
    reference_gradient = torch.autograd.grad(reference, reference_parameters)[0]
    assert torch.allclose(train_gradient, reference_gradient, rtol=1e-6, atol=1e-10)
    with torch.no_grad():
        no_grad_score, no_grad_components = training_grid_score(
            train_parameters.detach(),
            target,
            backbone,
            0.0,
            1.0,
            thresholds,
            weights,
            diagnostics=False,
        )
    assert no_grad_components is None
    assert torch.allclose(no_grad_score, reference.detach(), rtol=1e-13, atol=1e-13)


def test_training_grid_score_rejects_changed_grid_size():
    parameters = torch.zeros((1, 15, 1, 1, 1), requires_grad=True)
    target = torch.zeros((1, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="input differs"):
        training_grid_score(
            parameters,
            target,
            target,
            0.0,
            1.0,
            torch.zeros(GRID_CELLS - 1),
            torch.ones(GRID_CELLS - 1),
            diagnostics=False,
        )
