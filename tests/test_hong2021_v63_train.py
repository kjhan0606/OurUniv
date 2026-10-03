import json
from pathlib import Path

import torch

from hong2021_v18_init import sha256_file
from hong2021_v50_network import INITIAL_BIASES
from hong2021_v62_conditional_moment_gradient_audit import (
    _quadrature_rule,
    conditional_log_moment_score,
    conditional_physical_moments,
)
from hong2021_v63_preflight import PROGRAM_SHA256
from hong2021_v63_train import (
    CANDIDATE_IMPLEMENTATION_SHA256,
    MOMENT_COEFFICIENT,
    PREFLIGHT_IMPLEMENTATION_SHA256,
    PREFLIGHT_RECORD_SHA256,
    PREFLIGHT_SHA256,
    composite_training_loss,
    conditional_moment_score,
)


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/hong2021_v63_conditional_moment_model_program.json"
RECORD = ROOT / "config/hong2021_v63_preflight_record.json"


def _small_batch() -> tuple[torch.Tensor, ...]:
    parameters = (
        torch.tensor(INITIAL_BIASES, dtype=torch.float32)
        .reshape(1, 15, 1, 1, 1)
        .expand(3, 15, 2, 2, 2)
        .clone()
        .requires_grad_(True)
    )
    target = torch.linspace(0.05, 0.25, 24).reshape(3, 1, 2, 2, 2)
    backbone = torch.linspace(0.1, 0.3, 24).reshape_as(target)
    boundaries = torch.full((3,), -1.0, dtype=torch.float64)
    nodes, weights = _quadrature_rule(32, torch.device("cpu"))
    return parameters, target, backbone, boundaries, nodes, weights


def test_training_is_bound_to_passed_preflight_and_implementations() -> None:
    program = json.loads(PROGRAM.read_text())
    record = json.loads(RECORD.read_text())
    assert sha256_file(PROGRAM) == PROGRAM_SHA256
    assert sha256_file(RECORD) == PREFLIGHT_RECORD_SHA256
    assert sha256_file(ROOT / "src/hong2021_v63_preflight.py") == (
        PREFLIGHT_IMPLEMENTATION_SHA256
    )
    assert sha256_file(
        ROOT / "src/hong2021_v62_conditional_moment_gradient_audit.py"
    ) == CANDIDATE_IMPLEMENTATION_SHA256
    assert record["preflight"]["sha256"] == PREFLIGHT_SHA256
    assert record["authorization"]["training_allowed"] is True
    assert record["authorization"]["development_access_allowed"] is False
    assert program["firewall"]["independent_gate_locked"] is True


def test_training_candidate_exactly_reuses_audited_functional() -> None:
    parameters, target, backbone, boundaries, nodes, weights = _small_batch()
    score, predicted, truth, counts = conditional_moment_score(
        parameters,
        target,
        backbone,
        0.0,
        0.1,
        boundaries,
        nodes,
        weights,
    )
    expected_predicted, expected_truth, expected_counts = conditional_physical_moments(
        parameters,
        target,
        backbone,
        0.0,
        0.1,
        boundaries,
        nodes,
        weights,
    )
    assert torch.equal(predicted, expected_predicted)
    assert torch.equal(truth, expected_truth)
    assert counts == expected_counts
    assert torch.equal(score, conditional_log_moment_score(predicted, truth))
    assert torch.isfinite(torch.autograd.grad(score, parameters)[0]).all()


def test_composite_adds_only_declared_candidate_term() -> None:
    parameters, target, backbone, boundaries, nodes, weights = _small_batch()
    v54_thresholds = torch.linspace(0.5, 2.0, 4)
    grid_thresholds = torch.linspace(2.1, 3.6, 16)
    grid_weights = torch.full((16,), 1.0 / 16.0, dtype=torch.float64)
    scores = composite_training_loss(
        parameters,
        target,
        backbone,
        0.0,
        0.1,
        v54_thresholds,
        grid_thresholds,
        grid_weights,
        boundaries,
        nodes,
        weights,
    )
    total, moment = scores[0], scores[6]
    base = total - MOMENT_COEFFICIENT * moment
    assert torch.allclose(total, base + 0.1 * moment, rtol=0.0, atol=1.0e-12)
