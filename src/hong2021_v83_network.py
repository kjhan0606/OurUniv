#!/usr/bin/env python
"""V83 conditional one-point spline flow with a frozen V72 spatial copula.

The model learns only the conditional marginal distribution of the
standardized residual.  It deliberately contains no spatial latent transform:
sampling supplies a correlated standard-normal latent from V72 SQT.
"""
from __future__ import annotations

import math

import torch
from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)
from torch import nn
from torch.nn import functional as F

from hong2021_v48_network import INPUT_CHANNELS, LocalMixtureUNet


BINS = 16
TAIL_BOUND = 14.0
MINIMUM_BIN_WIDTH = 1.0e-3
MINIMUM_BIN_HEIGHT = 1.0e-3
MINIMUM_DERIVATIVE = 1.0e-3
PARAMETERS = 3 * BINS - 1
RANK_EPSILON = 1.0e-7


def _identity_derivative_logit() -> float:
    return float(math.log(math.expm1(1.0 - MINIMUM_DERIVATIVE)))


class ConditionalMarginalSplineUNet(LocalMixtureUNet):
    """The V48 seven-channel U-Net with one conditional scalar spline head."""

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        base_channels: int = 32,
    ) -> None:
        super().__init__(input_channels=input_channels, base_channels=base_channels)
        feature_channels = self.output.in_channels
        self.output = nn.Conv3d(feature_channels, PARAMETERS, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[2 * BINS :].fill_(_identity_derivative_logit())


def spline_parameters(
    parameters: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if parameters.ndim != 5 or parameters.shape[1] != PARAMETERS:
        raise ValueError("V83 spline parameter shape differs")
    raw = parameters.float().permute(0, 2, 3, 4, 1)
    return raw[..., :BINS], raw[..., BINS : 2 * BINS], raw[..., 2 * BINS :]


def _scalar_field(value: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    expected = (len(parameters), 1, *parameters.shape[-3:])
    if value.shape != expected:
        raise ValueError("V83 scalar field shape differs")
    if not bool(torch.isfinite(value).all().detach().cpu()):
        raise ValueError("V83 scalar field is nonfinite")
    return value[:, 0].float()


def conditional_forward(
    parameters: torch.Tensor,
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a standardized physical residual to the standard-normal base."""
    value = _scalar_field(residual, parameters)
    widths, heights, derivatives = spline_parameters(parameters)
    latent, logabsdet = unconstrained_rational_quadratic_spline(
        value,
        widths,
        heights,
        derivatives,
        inverse=False,
        tails="linear",
        tail_bound=TAIL_BOUND,
        min_bin_width=MINIMUM_BIN_WIDTH,
        min_bin_height=MINIMUM_BIN_HEIGHT,
        min_derivative=MINIMUM_DERIVATIVE,
    )
    return latent[:, None], logabsdet[:, None]


def conditional_inverse(
    parameters: torch.Tensor,
    latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map a standard-normal latent to the standardized physical residual."""
    value = _scalar_field(latent, parameters)
    widths, heights, derivatives = spline_parameters(parameters)
    residual, inverse_logabsdet = unconstrained_rational_quadratic_spline(
        value,
        widths,
        heights,
        derivatives,
        inverse=True,
        tails="linear",
        tail_bound=TAIL_BOUND,
        min_bin_width=MINIMUM_BIN_WIDTH,
        min_bin_height=MINIMUM_BIN_HEIGHT,
        min_derivative=MINIMUM_DERIVATIVE,
    )
    return residual[:, None], inverse_logabsdet[:, None]


def conditional_log_probability(
    parameters: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    latent, logabsdet = conditional_forward(parameters, residual)
    return (
        -0.5 * torch.square(latent)
        - 0.5 * math.log(2.0 * math.pi)
        + logabsdet
    )


def conditional_cdf(parameters: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    latent, _ = conditional_forward(parameters, residual)
    return 0.5 * (1.0 + torch.erf(latent / math.sqrt(2.0)))


@torch.no_grad()
def conditional_icdf(
    parameters: torch.Tensor,
    uniform: torch.Tensor,
) -> torch.Tensor:
    probability = uniform.float().clamp(RANK_EPSILON, 1.0 - RANK_EPSILON)
    latent = math.sqrt(2.0) * torch.erfinv(2.0 * probability - 1.0)
    residual, _ = conditional_inverse(parameters, latent)
    return residual


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "BINS",
    "ConditionalMarginalSplineUNet",
    "MINIMUM_BIN_HEIGHT",
    "MINIMUM_BIN_WIDTH",
    "MINIMUM_DERIVATIVE",
    "PARAMETERS",
    "RANK_EPSILON",
    "TAIL_BOUND",
    "conditional_cdf",
    "conditional_forward",
    "conditional_icdf",
    "conditional_inverse",
    "conditional_log_probability",
    "parameter_count",
    "spline_parameters",
]
