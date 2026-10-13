from __future__ import annotations

import numpy as np
import torch
from torch import nn

from hong2021_residual_v9_tail import tail_balanced_edm_loss
from hong2021_v23_conditional_loss import (
    conditional_bin_indices,
    conditional_mean_penalty,
    conditional_mean_tail_edm_loss,
    fixed_conditional_validation,
)


class TinyDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.125))

    def forward(self, noisy, condition, noise_level):
        return self.scale * noisy + 0.01 * condition[:, :1]


def test_torch_bin_assignment_exactly_matches_numpy_right_clamped_rule() -> None:
    edges_np = np.asarray([-2.0, -0.5, 0.25, 3.0], dtype=np.float64)
    values_np = np.asarray(
        [-3.0, -2.0, -1.0, -0.5, 0.0, 0.25, 2.0, 3.0, 4.0],
        dtype=np.float32,
    ).reshape(1, 1, 1, 1, -1)
    expected = np.clip(
        np.searchsorted(edges_np, values_np.astype(np.float64), side="right") - 1,
        0,
        len(edges_np) - 2,
    )
    actual = conditional_bin_indices(
        torch.from_numpy(values_np), torch.from_numpy(edges_np)
    )
    assert np.array_equal(actual.numpy(), expected)


def test_conditional_penalty_is_zero_for_exact_denoising() -> None:
    residual = torch.linspace(-1, 1, 2 * 4**3).reshape(2, 1, 4, 4, 4)
    mean = torch.zeros_like(residual)
    edges = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    value = conditional_mean_penalty(
        residual, residual, mean, edges, torch.ones(2), minimum_count=64
    )
    assert torch.equal(value, torch.tensor(0.0))


def test_conditional_penalty_is_positive_and_differentiable() -> None:
    residual = torch.zeros(2, 1, 4, 4, 4)
    denoised = torch.full_like(residual, 0.25, requires_grad=True)
    mean = torch.zeros_like(residual)
    edges = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    value = conditional_mean_penalty(
        denoised, residual, mean, edges, torch.tensor([1.0, 2.0]),
        minimum_count=64,
    )
    assert value > 0
    value.backward()
    assert denoised.grad is not None
    assert torch.isfinite(denoised.grad).all()
    assert torch.count_nonzero(denoised.grad) == denoised.numel()


def test_v23_preserves_v22_noise_draws_and_base_loss_terms() -> None:
    model = TinyDenoiser()
    residual = torch.linspace(-1, 1, 2 * 4**3).reshape(2, 1, 4, 4, 4)
    condition = torch.zeros(2, 4, 4, 4, 4)
    truth = residual * 0.2
    tail_weights = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4])
    edges = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    parent_generator = torch.Generator().manual_seed(230011)
    v23_generator = torch.Generator().manual_seed(230011)
    parent = tail_balanced_edm_loss(
        model, residual, condition, truth, tail_weights, parent_generator,
        2.0, 0.18, 1.2,
    )
    v23 = conditional_mean_tail_edm_loss(
        model, residual, condition, truth, tail_weights, edges, v23_generator,
        2.0, 0.18, 1.2, 1.0, 64,
    )
    assert torch.equal(parent_generator.get_state(), v23_generator.get_state())
    assert torch.equal(parent[1], v23[1])
    assert torch.equal(parent[2], v23[2])
    assert torch.allclose(v23[0], 0.5 * v23[1] + 0.5 * v23[2] + v23[3])


def test_fixed_conditional_validation_uses_private_rng_stream() -> None:
    residual = torch.zeros(2, 1, 4, 4, 4)
    condition = torch.zeros(2, 4, 4, 4, 4)
    truth = torch.zeros_like(residual)
    loader = [(condition, residual, condition[:, 2:3], truth)]
    before = torch.random.get_rng_state().clone()
    value = fixed_conditional_validation(
        TinyDenoiser(), loader, torch.device("cpu"),
        torch.tensor([-1.0, 1.0], dtype=torch.float64), 99193,
        2.0, 0.18, 1.2, 64,
    )
    assert np.isfinite(value)
    assert torch.equal(before, torch.random.get_rng_state())
