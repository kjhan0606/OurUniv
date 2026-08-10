import ast
import hashlib
import math
from pathlib import Path

import torch

from hong2021_v50_network import LocalMixtureUNet
from hong2021_v54_train import (
    PROGRAM_SHA256,
    QUANTILES,
    TAIL_COEFFICIENT,
    _learning_rate,
    composite_loss,
    physical_tail_brier_score,
)


REPO = Path(__file__).resolve().parents[1]


def test_program_hash_and_firewall() -> None:
    path = REPO / "config/hong2021_v54_physical_tail_brier_program.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == PROGRAM_SHA256
    text = path.read_text()
    assert '"coefficient": 0.1' in text
    assert '"development_access_before_train_mechanism_pass": "forbidden"' in text
    assert '"historical_EAGLE_access": "forbidden"' in text


def test_source_has_no_json_boolean_names() -> None:
    names = {
        node.id
        for node in ast.walk(ast.parse((REPO / "src/hong2021_v54_train.py").read_text()))
        if isinstance(node, ast.Name)
    }
    assert "false" not in names
    assert "true" not in names


def test_brier_score_boundary_probabilities_are_exact_and_finite() -> None:
    model = LocalMixtureUNet(base_channels=8)
    parameters = model(torch.zeros(1, 7, 16, 16, 16))
    target = torch.zeros(1, 1, 16, 16, 16)
    backbone = torch.zeros_like(target)
    thresholds = torch.tensor([-100.0, -1.0, 1.0, 100.0])
    tail, components = physical_tail_brier_score(
        parameters, target, backbone, 0.0, 1.0, thresholds
    )
    assert torch.isfinite(tail)
    assert torch.isfinite(components).all()
    assert components[0] == 0.0
    assert components[-1] == 0.0


def test_composite_is_exact_positive_sum_and_has_gradients() -> None:
    model = LocalMixtureUNet(base_channels=8)
    parameters = model(torch.randn(1, 7, 16, 16, 16))
    target = torch.zeros(1, 1, 16, 16, 16)
    backbone = torch.zeros_like(target)
    thresholds = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    total, nll, tail, components = composite_loss(
        parameters, target, backbone, 0.0, 1.0, thresholds
    )
    assert torch.allclose(total, nll + TAIL_COEFFICIENT * tail)
    assert components.shape == (len(QUANTILES),)
    total.backward()
    assert model.output.bias.grad is not None
    assert torch.isfinite(model.output.bias.grad).all()
    assert torch.count_nonzero(model.output.bias.grad)


def test_schedule_matches_v50() -> None:
    assert math.isclose(_learning_rate(12_000), 2.0e-5)
    assert 2.0e-5 < _learning_rate(6_000) < 2.0e-4
