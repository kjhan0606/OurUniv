from __future__ import annotations

import math

import pytest
import torch

from hong2021_v83_network import (
    BINS,
    PARAMETERS,
    TAIL_BOUND,
    ConditionalMarginalSplineUNet,
    conditional_cdf,
    conditional_forward,
    conditional_icdf,
    conditional_inverse,
    conditional_log_probability,
    parameter_count,
    spline_parameters,
)


def test_identity_initialization_and_parameter_count() -> None:
    torch.manual_seed(83)
    model = ConditionalMarginalSplineUNet(base_channels=4).eval()
    condition = torch.randn(1, 7, 16, 16, 16)
    residual = torch.linspace(-13.5, 13.5, 16**3).reshape(1, 1, 16, 16, 16)
    with torch.no_grad():
        parameters = model(condition)
        latent, logdet = conditional_forward(parameters, residual)
    assert parameters.shape == (1, PARAMETERS, 16, 16, 16)
    assert parameter_count(model) > 0
    assert torch.max(torch.abs(latent - residual)).item() < 2.0e-5
    assert torch.max(torch.abs(logdet)).item() < 2.0e-5


def test_roundtrip_logdet_and_cdf_inverse() -> None:
    torch.manual_seed(8301)
    parameters = 0.02 * torch.randn(2, PARAMETERS, 4, 5, 6)
    residual = (2.0 * torch.randn(2, 1, 4, 5, 6)).clamp(-4.0, 4.0)
    latent, forward_logdet = conditional_forward(parameters, residual)
    recovered, inverse_logdet = conditional_inverse(parameters, latent)
    assert torch.max(torch.abs(recovered - residual)).item() < 2.0e-4
    assert torch.max(torch.abs(forward_logdet + inverse_logdet)).item() < 2.0e-4
    uniform = conditional_cdf(parameters, residual)
    inverse = conditional_icdf(parameters, uniform)
    assert torch.max(torch.abs(inverse - residual)).item() < 3.0e-3


def test_identity_likelihood_is_standard_normal() -> None:
    parameters = torch.zeros(1, PARAMETERS, 2, 3, 4)
    derivative = math.log(math.expm1(1.0 - 1.0e-3))
    parameters[:, 2 * BINS :] = derivative
    residual = torch.randn(1, 1, 2, 3, 4)
    observed = conditional_log_probability(parameters, residual)
    expected = -0.5 * residual.square() - 0.5 * math.log(2.0 * math.pi)
    assert torch.max(torch.abs(observed - expected)).item() < 2.0e-5


def test_linear_tails_and_parameter_validation() -> None:
    parameters = torch.randn(1, PARAMETERS, 1, 1, 2)
    residual = torch.tensor([[[[[-TAIL_BOUND - 1.0, TAIL_BOUND + 1.0]]]]])
    latent, logdet = conditional_forward(parameters, residual)
    assert torch.equal(latent, residual)
    assert torch.equal(logdet, torch.zeros_like(logdet))
    with pytest.raises(ValueError, match="parameter shape"):
        spline_parameters(parameters[:, :-1])
    with pytest.raises(ValueError, match="scalar field shape"):
        conditional_forward(parameters, residual[:, :, :, :, :1])


def test_proper_likelihood_has_finite_nonzero_gradient() -> None:
    torch.manual_seed(8302)
    parameters = (0.05 * torch.randn(1, PARAMETERS, 2, 2, 2)).requires_grad_()
    residual = torch.randn(1, 1, 2, 2, 2)
    loss = -conditional_log_probability(parameters, residual).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert parameters.grad is not None
    assert torch.isfinite(parameters.grad).all()
    assert torch.linalg.vector_norm(parameters.grad) > 0
