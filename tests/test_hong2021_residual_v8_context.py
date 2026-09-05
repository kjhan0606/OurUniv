from __future__ import annotations

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_residual_diffusion import ConditionalResidualUNet
from hong2021_residual_v8_context import (
    FEATURE_NAMES,
    ObservableContextUNet,
    initialize_parent,
    observable_context_features,
)


def test_observable_context_features_are_cube_isometry_invariant() -> None:
    generator = torch.Generator().manual_seed(13)
    condition = torch.randn((2, 4, 8, 8, 8), generator=generator)
    condition[:, 0] = torch.poisson(condition[:, 0].abs(), generator=generator)
    condition[:, 1][condition[:, 0] == 0] = 0
    baseline = observable_context_features(condition)
    permutation, reflections = CUBE_ISOMETRIES[37]
    transformed = apply_cube_isometry(
        condition.numpy(), permutation, reflections
    )
    actual = observable_context_features(torch.from_numpy(transformed.copy()))
    torch.testing.assert_close(actual, baseline)
    assert actual.shape == (2, len(FEATURE_NAMES))


def test_zero_initialized_context_exactly_reproduces_parent() -> None:
    torch.manual_seed(7)
    parent = ConditionalResidualUNet(base_channels=8).eval()
    model = ObservableContextUNet(
        base_channels=8,
        context_mean=torch.arange(len(FEATURE_NAMES), dtype=torch.float32),
        context_std=torch.arange(1, len(FEATURE_NAMES) + 1, dtype=torch.float32),
    ).eval()
    initialize_parent(model, parent.state_dict())
    noisy = torch.randn(2, 1, 8, 8, 8)
    condition = torch.randn(2, 4, 8, 8, 8)
    condition[:, 0] = condition[:, 0].abs()
    time = torch.tensor([0.1, 0.7])

    with torch.inference_mode():
        expected = parent(noisy, condition, time)
        actual = model(noisy, condition, time)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_context_network_can_change_parent_output() -> None:
    torch.manual_seed(11)
    parent = ConditionalResidualUNet(base_channels=8).eval()
    model = ObservableContextUNet(base_channels=8).eval()
    initialize_parent(model, parent.state_dict())
    with torch.no_grad():
        model.context[-1].weight.fill_(0.01)
    noisy = torch.randn(1, 1, 8, 8, 8)
    condition = torch.randn(1, 4, 8, 8, 8)
    condition[:, 0] = condition[:, 0].abs()
    time = torch.tensor([0.3])
    with torch.inference_mode():
        expected = parent(noisy, condition, time)
        actual = model(noisy, condition, time)
    assert not torch.equal(actual, expected)
