#!/usr/bin/env python
"""V50 proper bounded-logit Gaussian-mixture likelihood."""
from __future__ import annotations

import math

import torch

from hong2021_v48_network import (
    BISECTION_STEPS,
    INPUT_CHANNELS,
    MIXTURES,
    RANK_EPSILON,
    LocalMixtureUNet as V48LocalMixtureUNet,
    gaussian_mixture_cdf,
    gaussian_mixture_inverse,
    gaussian_mixture_log_probability,
    mixture_parameters,
    parameter_count,
    standard_normal_cdf,
)


LOWER_SUPPORT = -13.78839653180272
UPPER_SUPPORT = 10.259036149654781
SUPPORT_RANGE = UPPER_SUPPORT - LOWER_SUPPORT
INITIAL_STANDARDIZED_LOCATIONS = (
    -1.2815515655446004,
    -0.5244005127080409,
    0.0,
    0.5244005127080407,
    1.2815515655446004,
)
INITIAL_LATENT_LOCATIONS = (
    0.0804059034222661,
    0.20704198374122115,
    0.2956685146367071,
    0.3854638253283423,
    0.5179828385810585,
)
INITIAL_LATENT_SCALES = (
    0.08043023276357575,
    0.08116400024760286,
    0.08206815944193456,
    0.08332029605961505,
    0.08580816082388276,
)
INITIAL_RAW_SCALES = (
    -2.6177108734287255,
    -2.606975198588936,
    -2.5938924693439107,
    -2.576033685703158,
    -2.5414058073807415,
)
INITIAL_BIASES = (0.0,) * MIXTURES + INITIAL_LATENT_LOCATIONS + INITIAL_RAW_SCALES
QUADRATURE_INITIAL_STANDARDIZED_MEAN = -0.0029472578587312003
QUADRATURE_INITIAL_STANDARDIZED_VARIANCE = 0.9966385826992292


class LocalMixtureUNet(V48LocalMixtureUNet):
    """Unchanged V48 U-Net with the frozen V50 transformed output biases."""

    def __init__(self, input_channels: int = INPUT_CHANNELS, base_channels: int = 32) -> None:
        super().__init__(input_channels=input_channels, base_channels=base_channels)
        with torch.no_grad():
            self.output.bias.copy_(
                torch.tensor(INITIAL_BIASES, dtype=self.output.bias.dtype)
            )


def _unit_interval(value: torch.Tensor) -> torch.Tensor:
    unit = (value.double() - LOWER_SUPPORT) / SUPPORT_RANGE
    if not bool(torch.all((unit > 0.0) & (unit < 1.0)).detach().cpu()):
        raise ValueError("V50 target lies outside the frozen open support")
    return unit


def bounded_to_latent(value: torch.Tensor) -> torch.Tensor:
    unit = _unit_interval(value)
    return torch.log(unit) - torch.log1p(-unit)


def latent_to_bounded(value: torch.Tensor) -> torch.Tensor:
    result = LOWER_SUPPORT + SUPPORT_RANGE * torch.sigmoid(value.double())
    if not bool(
        torch.all((result > LOWER_SUPPORT) & (result < UPPER_SUPPORT)).detach().cpu()
    ):
        raise RuntimeError("V50 inverse transform reached the support boundary")
    return result


def bounded_log_absolute_forward_jacobian(value: torch.Tensor) -> torch.Tensor:
    unit = _unit_interval(value)
    return (
        -math.log(SUPPORT_RANGE) - torch.log(unit) - torch.log1p(-unit)
    ).float()


def bounded_mixture_log_probability(
    parameters: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    latent = bounded_to_latent(target)
    return gaussian_mixture_log_probability(
        parameters, latent
    ) + bounded_log_absolute_forward_jacobian(target)


def bounded_mixture_cdf(parameters: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return gaussian_mixture_cdf(parameters, bounded_to_latent(value))


@torch.no_grad()
def bounded_mixture_inverse(
    parameters: torch.Tensor,
    uniform: torch.Tensor,
    *,
    steps: int = BISECTION_STEPS,
) -> torch.Tensor:
    latent = gaussian_mixture_inverse(parameters, uniform, steps=steps)
    return latent_to_bounded(latent)


def initial_standardized_quadrature(
    nodes: torch.Tensor, weights: torch.Tensor
) -> tuple[float, float]:
    """Gauss-Hermite mean and variance of the frozen equal-weight initial model."""
    if nodes.ndim != 1 or weights.shape != nodes.shape:
        raise ValueError("V50 quadrature shape differs")
    values = []
    for location, scale in zip(
        INITIAL_LATENT_LOCATIONS, INITIAL_LATENT_SCALES, strict=True
    ):
        latent = location + math.sqrt(2.0) * scale * nodes.double()
        values.append(latent_to_bounded(latent).double())
    stacked = torch.stack(values)
    normalized_weights = weights.double() / math.sqrt(math.pi)
    first = torch.mean(torch.sum(stacked * normalized_weights, dim=1))
    second = torch.mean(torch.sum(torch.square(stacked) * normalized_weights, dim=1))
    return float(first), float(second - first * first)


__all__ = [
    "BISECTION_STEPS",
    "INITIAL_BIASES",
    "INITIAL_LATENT_LOCATIONS",
    "INITIAL_LATENT_SCALES",
    "INITIAL_RAW_SCALES",
    "INITIAL_STANDARDIZED_LOCATIONS",
    "INPUT_CHANNELS",
    "LOWER_SUPPORT",
    "LocalMixtureUNet",
    "MIXTURES",
    "QUADRATURE_INITIAL_STANDARDIZED_MEAN",
    "QUADRATURE_INITIAL_STANDARDIZED_VARIANCE",
    "RANK_EPSILON",
    "SUPPORT_RANGE",
    "UPPER_SUPPORT",
    "bounded_log_absolute_forward_jacobian",
    "bounded_mixture_cdf",
    "bounded_mixture_inverse",
    "bounded_mixture_log_probability",
    "bounded_to_latent",
    "initial_standardized_quadrature",
    "latent_to_bounded",
    "mixture_parameters",
    "parameter_count",
    "standard_normal_cdf",
]
