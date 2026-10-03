from __future__ import annotations

import torch
from torch import nn

from hong2021_residual_v9_tail import tail_balanced_edm_loss
from hong2021_v17_loss import (
    LOSS_COEFFICIENTS,
    band_balanced_tail_edm_loss,
    fourier_band_error_per_sample,
    prepare_fourier_band_loss,
)


class ZeroNetwork(nn.Module):
    def forward(self, noisy, condition, time_fraction):
        return torch.zeros_like(noisy)


def test_production_fourier_masks_and_parseval_are_frozen() -> None:
    masks, specification = prepare_fourier_band_loss(
        64, 0.3125, torch.device("cpu")
    )
    assert masks.shape == (4, 64, 64, 64)
    assert specification["mode_counts"] == [146, 3596, 25296, 233105]
    assert specification["non_dc_modes"] == 64**3 - 1
    assert specification["fft_norm"] == "ortho"
    assert specification["parseval_self_check_relative_error"] <= 1.0e-4
    assert specification["mask_partition_self_check_relative_error"] <= 1.0e-4


def test_band_loss_excludes_dc_and_has_finite_gradient() -> None:
    masks, _ = prepare_fourier_band_loss(32, 0.5, torch.device("cpu"))
    constant = torch.ones(2, 1, 32, 32, 32, requires_grad=True)
    loss = fourier_band_error_per_sample(constant, masks)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1.0e-10)
    varying = torch.linspace(-1, 1, 32**3).reshape(1, 1, 32, 32, 32)
    varying.requires_grad_()
    value = fourier_band_error_per_sample(varying, masks).sum()
    value.backward()
    assert varying.grad is not None
    assert torch.isfinite(varying.grad).all()


def test_v17_adds_no_rng_draw_and_preserves_parent_u_t() -> None:
    device = torch.device("cpu")
    masks, _ = prepare_fourier_band_loss(32, 0.5, device)
    model = ZeroNetwork()
    residual = torch.linspace(-1, 1, 2 * 32**3).reshape(2, 1, 32, 32, 32)
    condition = torch.zeros(2, 4, 32, 32, 32)
    truth = torch.zeros_like(residual)
    bin_weights = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4])
    parent_generator = torch.Generator().manual_seed(144121)
    v17_generator = torch.Generator().manual_seed(144121)
    parent = tail_balanced_edm_loss(
        model, residual, condition, truth, bin_weights, parent_generator,
        2.0, 0.18, 1.2,
    )
    v17 = band_balanced_tail_edm_loss(
        model, residual, condition, truth, bin_weights, masks, v17_generator,
        2.0, 0.18, 1.2,
    )
    assert torch.equal(parent_generator.get_state(), v17_generator.get_state())
    assert torch.equal(parent[1], v17[1])
    assert torch.equal(parent[2], v17[2])
    expected = sum(
        coefficient * value
        for coefficient, value in zip(LOSS_COEFFICIENTS, v17[1:], strict=True)
    )
    assert torch.allclose(v17[0], expected)
