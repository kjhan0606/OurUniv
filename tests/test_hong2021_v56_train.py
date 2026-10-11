import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hong2021_v18_init import sha256_file
from hong2021_v50_network import INITIAL_BIASES
from hong2021_v56_train import (
    GRID_CELLS,
    GRID_COEFFICIENT,
    PROGRAM_SHA256,
    REFERENCE_PROBABILITY,
    composite_loss,
    grid_values,
    upper_survival_grid_score,
)


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/hong2021_v56_survival_grid_program.json"


def test_program_and_parent_record_are_hash_bound_and_locked():
    program = json.loads(PROGRAM.read_text())
    record_path = ROOT / program["parent_evidence"]["v55_record"]
    record = json.loads(record_path.read_text())
    assert sha256_file(PROGRAM) == PROGRAM_SHA256
    assert sha256_file(record_path) == program["parent_evidence"]["v55_record_sha256"]
    assert program["status"].startswith("frozen_before_grid_materialization")
    assert record["audit"]["classification"] == program["parent_evidence"][
        "required_classification"
    ]
    assert record["firewall"]["development_accessed"] is False
    assert program["firewall"]["independent_gate_locked"] is True


def test_grid_formula_is_strict_positive_normalized_and_fixed():
    program = json.loads(PROGRAM.read_text())
    rule = program["upper_survival_grid"]
    thresholds, weights = grid_values(rule["lower_edge_value"], rule["upper_edge_value"])
    assert len(thresholds) == len(weights) == GRID_CELLS
    assert np.all(np.diff(thresholds) > 0.0)
    assert np.all(weights > 0.0)
    assert weights.sum(dtype=np.float64) == pytest.approx(1.0, abs=1e-15)
    assert thresholds[-1] == rule["upper_edge_value"]
    assert thresholds[0] == pytest.approx(3.352849731221795, abs=1e-15)
    assert weights[-1] == pytest.approx(0.18251045630583962, rel=1e-14)


def test_grid_rejects_invalid_edges():
    with pytest.raises(ValueError, match="grid edges"):
        grid_values(1.0, 1.0)


def test_upper_grid_score_is_finite_proper_sum_with_gradient():
    lower, upper = 3.3113619089126587, 3.975167065858841
    thresholds, weights = grid_values(lower, upper)
    parameters = (
        torch.tensor(INITIAL_BIASES, dtype=torch.float32)
        .reshape(1, 15, 1, 1, 1)
        .repeat(1, 1, 1, 1, 2)
        .requires_grad_(True)
    )
    target = torch.tensor([0.8, 0.7], dtype=torch.float32).reshape(1, 1, 1, 1, 2)
    backbone = torch.zeros_like(target)
    threshold_tensor = torch.from_numpy(thresholds)
    weight_tensor = torch.from_numpy(weights)
    score, components = upper_survival_grid_score(
        parameters, target, backbone, 0.0, 1.0, threshold_tensor, weight_tensor
    )
    expected = torch.sum(weight_tensor * components) / (
        REFERENCE_PROBABILITY * (1.0 - REFERENCE_PROBABILITY)
    )
    assert torch.isfinite(score)
    assert float(score.detach()) == pytest.approx(float(expected.detach()), rel=1e-12)
    score.backward()
    assert parameters.grad is not None
    assert torch.isfinite(parameters.grad).all()
    assert torch.count_nonzero(parameters.grad)


def test_composite_adds_only_fixed_grid_term_to_v54_score():
    grid_thresholds, grid_weights = grid_values(
        3.3113619089126587, 3.975167065858841
    )
    parameters = (
        torch.tensor(INITIAL_BIASES, dtype=torch.float32)
        .reshape(1, 15, 1, 1, 1)
        .requires_grad_(True)
    )
    target = torch.tensor([0.75], dtype=torch.float32).reshape(1, 1, 1, 1, 1)
    backbone = torch.zeros_like(target)
    scores = composite_loss(
        parameters,
        target,
        backbone,
        0.0,
        1.0,
        torch.tensor([1.1158325, 2.0498113, 2.778472, 3.3113619]),
        torch.from_numpy(grid_thresholds),
        torch.from_numpy(grid_weights),
    )
    total, nll, tail, _, upper, _ = scores
    expected = nll + 0.1 * tail + GRID_COEFFICIENT * upper
    assert float(total.detach()) == pytest.approx(float(expected.detach()))
