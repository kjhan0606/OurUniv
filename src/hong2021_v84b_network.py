#!/usr/bin/env python
"""V84B conditional marginal with an RQS centre and exponential tail splices."""
from __future__ import annotations

import math

import torch
from torch import nn

from hong2021_v48_network import INPUT_CHANNELS
from hong2021_v83_network import (
    PARAMETERS as CORE_PARAMETERS,
    ConditionalMarginalSplineUNet,
    conditional_cdf as core_cdf,
    conditional_inverse as core_inverse,
    conditional_log_probability as core_log_probability,
)


LOWER_THRESHOLD = -2.35
UPPER_THRESHOLD = 3.10
INITIAL_TAIL_MASS = 0.005
MINIMUM_LOWER_SCALE = 0.05
MAXIMUM_LOWER_SCALE = 3.00
MINIMUM_UPPER_SCALE = 0.05
MAXIMUM_UPPER_SCALE = 1.50
INITIAL_LOWER_SCALE = 0.35
INITIAL_UPPER_SCALE = 0.54
TAIL_PARAMETERS = 4
PARAMETERS = CORE_PARAMETERS + TAIL_PARAMETERS
RANK_EPSILON = 1.0e-7


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _bounded_scale(raw: torch.Tensor, minimum: float, maximum: float) -> torch.Tensor:
    return minimum + (maximum - minimum) * torch.sigmoid(raw.float())


class ConditionalSplicedTailUNet(ConditionalMarginalSplineUNet):
    """V83 U-Net with two tail-mass logits and two bounded tail scales."""

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        base_channels: int = 32,
    ) -> None:
        super().__init__(input_channels=input_channels, base_channels=base_channels)
        old = self.output
        feature_channels = old.in_channels
        self.output = nn.Conv3d(feature_channels, PARAMETERS, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[:CORE_PARAMETERS].copy_(old.bias)
            tail_logit = math.log(INITIAL_TAIL_MASS / (1.0 - 2.0 * INITIAL_TAIL_MASS))
            self.output.bias[CORE_PARAMETERS : CORE_PARAMETERS + 2].fill_(tail_logit)
            lower_fraction = (INITIAL_LOWER_SCALE - MINIMUM_LOWER_SCALE) / (
                MAXIMUM_LOWER_SCALE - MINIMUM_LOWER_SCALE
            )
            upper_fraction = (INITIAL_UPPER_SCALE - MINIMUM_UPPER_SCALE) / (
                MAXIMUM_UPPER_SCALE - MINIMUM_UPPER_SCALE
            )
            self.output.bias[CORE_PARAMETERS + 2] = _logit(lower_fraction)
            self.output.bias[CORE_PARAMETERS + 3] = _logit(upper_fraction)


def spliced_parameters(
    parameters: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if parameters.ndim != 5 or parameters.shape[1] != PARAMETERS:
        raise ValueError("V84B parameter shape differs")
    core = parameters[:, :CORE_PARAMETERS].float()
    tail = parameters[:, CORE_PARAMETERS:].float()
    zero = torch.zeros_like(tail[:, :1])
    weights = torch.softmax(torch.cat((tail[:, :1], zero, tail[:, 1:2]), dim=1), dim=1)
    lower_scale = _bounded_scale(
        tail[:, 2:3], MINIMUM_LOWER_SCALE, MAXIMUM_LOWER_SCALE
    )
    upper_scale = _bounded_scale(
        tail[:, 3:4], MINIMUM_UPPER_SCALE, MAXIMUM_UPPER_SCALE
    )
    return core, weights, lower_scale, upper_scale


def _threshold_field(core: torch.Tensor, threshold: float) -> torch.Tensor:
    return torch.full(
        (len(core), 1, *core.shape[-3:]),
        threshold,
        dtype=torch.float32,
        device=core.device,
    )


def _core_bounds(core: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lower = core_cdf(core, _threshold_field(core, LOWER_THRESHOLD))
    upper = core_cdf(core, _threshold_field(core, UPPER_THRESHOLD))
    normalizer = upper - lower
    if torch.any(normalizer <= 0.0):
        raise RuntimeError("V84B central spline normalization is not positive")
    return lower, upper, normalizer


def _value_field(value: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    expected = (len(parameters), 1, *parameters.shape[-3:])
    if value.shape != expected or not bool(torch.isfinite(value).all().detach().cpu()):
        raise ValueError("V84B scalar field differs")
    return value.float()


def standard_normal_icdf(probability: torch.Tensor) -> torch.Tensor:
    """Acklam inverse-normal approximation using portable CUDA arithmetic only."""
    value = probability.float().clamp(RANK_EPSILON, 1.0 - RANK_EPSILON)
    a = (-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239)
    b = (-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572)
    c = (-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783)
    d = (0.007784695709041462, 0.3224671290700398,
         2.445134137142996, 3.754408661907416)

    def polynomial(coefficient: tuple[float, ...], argument: torch.Tensor) -> torch.Tensor:
        output = torch.full_like(argument, coefficient[0])
        for item in coefficient[1:]:
            output = output * argument + item
        return output

    lower_q = torch.sqrt(-2.0 * torch.log(value))
    lower = polynomial(c, lower_q) / (
        polynomial(d, lower_q) * lower_q + 1.0
    )
    upper_q = torch.sqrt(-2.0 * torch.log1p(-value))
    upper = -polynomial(c, upper_q) / (
        polynomial(d, upper_q) * upper_q + 1.0
    )
    central_q = value - 0.5
    central_r = torch.square(central_q)
    central = polynomial(a, central_r) * central_q / (
        polynomial(b, central_r) * central_r + 1.0
    )
    return torch.where(
        value < 0.02425,
        lower,
        torch.where(value > 1.0 - 0.02425, upper, central),
    )


def conditional_log_probability(
    parameters: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    value = _value_field(residual, parameters)
    core, weights, lower_scale, upper_scale = spliced_parameters(parameters)
    lower_cdf, _, central_normalizer = _core_bounds(core)
    del lower_cdf
    log_weights = torch.log(weights)
    lower_log = (
        log_weights[:, 0:1]
        - torch.log(lower_scale)
        - (LOWER_THRESHOLD - value) / lower_scale
    )
    central_log = (
        log_weights[:, 1:2]
        + core_log_probability(core, value)
        - torch.log(central_normalizer)
    )
    upper_log = (
        log_weights[:, 2:3]
        - torch.log(upper_scale)
        - (value - UPPER_THRESHOLD) / upper_scale
    )
    return torch.where(
        value < LOWER_THRESHOLD,
        lower_log,
        torch.where(value > UPPER_THRESHOLD, upper_log, central_log),
    )


def conditional_cdf(parameters: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    value = _value_field(residual, parameters)
    core, weights, lower_scale, upper_scale = spliced_parameters(parameters)
    lower_core, _, central_normalizer = _core_bounds(core)
    lower = weights[:, 0:1] * torch.exp(
        -(LOWER_THRESHOLD - value) / lower_scale
    )
    central_rank = (core_cdf(core, value) - lower_core) / central_normalizer
    central = weights[:, 0:1] + weights[:, 1:2] * central_rank
    upper = 1.0 - weights[:, 2:3] * torch.exp(
        -(value - UPPER_THRESHOLD) / upper_scale
    )
    return torch.where(
        value < LOWER_THRESHOLD,
        lower,
        torch.where(value > UPPER_THRESHOLD, upper, central),
    ).clamp(0.0, 1.0)


@torch.no_grad()
def conditional_icdf(parameters: torch.Tensor, uniform: torch.Tensor) -> torch.Tensor:
    probability = _value_field(uniform, parameters).clamp(
        RANK_EPSILON, 1.0 - RANK_EPSILON
    )
    core, weights, lower_scale, upper_scale = spliced_parameters(parameters)
    lower_core, _, central_normalizer = _core_bounds(core)
    lower = LOWER_THRESHOLD + lower_scale * torch.log(
        probability / weights[:, 0:1]
    )
    central_probability = (
        (probability - weights[:, 0:1]) / weights[:, 1:2]
    ).clamp(RANK_EPSILON, 1.0 - RANK_EPSILON)
    central_core_probability = lower_core + central_probability * central_normalizer
    central_latent = standard_normal_icdf(central_core_probability)
    central, _ = core_inverse(core, central_latent)
    upper = UPPER_THRESHOLD - upper_scale * torch.log(
        (1.0 - probability) / weights[:, 2:3]
    )
    return torch.where(
        probability < weights[:, 0:1],
        lower,
        torch.where(
            probability > 1.0 - weights[:, 2:3],
            upper,
            central,
        ),
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def upper_physical_second_moment_margin(target_std: float) -> float:
    """Positive means every allowed upper exponential tail has finite rho^2."""
    coefficient = 2.0 * math.log(10.0) * float(target_std)
    return 1.0 - coefficient * MAXIMUM_UPPER_SCALE


__all__ = [
    "ConditionalSplicedTailUNet",
    "INITIAL_LOWER_SCALE",
    "INITIAL_TAIL_MASS",
    "INITIAL_UPPER_SCALE",
    "LOWER_THRESHOLD",
    "MAXIMUM_LOWER_SCALE",
    "MAXIMUM_UPPER_SCALE",
    "MINIMUM_LOWER_SCALE",
    "MINIMUM_UPPER_SCALE",
    "PARAMETERS",
    "UPPER_THRESHOLD",
    "conditional_cdf",
    "conditional_icdf",
    "conditional_log_probability",
    "parameter_count",
    "spliced_parameters",
    "standard_normal_icdf",
    "upper_physical_second_moment_margin",
]
