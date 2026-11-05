#!/usr/bin/env python
"""V45 identifiable query-local logistic-mixture U-Net and likelihood."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


MIXTURES = 5
INPUT_CHANNELS = 7
OUTPUT_CHANNELS = 3 * MIXTURES
MINIMUM_SCALE = 0.01
BISECTION_STEPS = 28
RANK_EPSILON = 1.0e-7
INITIAL_LOCATIONS = (
    -1.2815515655446004,
    -0.5244005127080409,
    0.0,
    0.5244005127080407,
    1.2815515655446004,
)
INITIAL_SCALE = 0.26615648673140785
INITIAL_RAW_SCALE = -1.2311559889276684
INITIAL_BIASES = (
    (0.0,) * MIXTURES
    + INITIAL_LOCATIONS
    + (INITIAL_RAW_SCALE,) * MIXTURES
)


def _groups(channels: int) -> int:
    return max(group for group in range(1, 9) if channels % group == 0)


class ReflectConv3d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.padding = nn.ReflectionPad3d(kernel_size // 2)
        self.convolution = nn.Conv3d(
            input_channels,
            output_channels,
            kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.convolution(self.padding(value))


class ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(input_channels), input_channels)
        self.conv1 = ReflectConv3d(input_channels, output_channels)
        self.norm2 = nn.GroupNorm(_groups(output_channels), output_channels)
        self.conv2 = ReflectConv3d(output_channels, output_channels)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv3d(input_channels, output_channels, 1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return self.skip(value) + hidden


class Stage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.first = ResidualBlock(input_channels, output_channels)
        self.second = ResidualBlock(output_channels, output_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(value))


class Up(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = ReflectConv3d(input_channels, output_channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, scale_factor=2.0, mode="nearest")
        return self.conv(value)


class LocalMixtureUNet(nn.Module):
    """The architecture and identifiable initialization frozen for V45."""

    def __init__(self, input_channels: int = INPUT_CHANNELS, base_channels: int = 32) -> None:
        super().__init__()
        channels = (base_channels, 2 * base_channels, 4 * base_channels, 4 * base_channels)
        self.input = ReflectConv3d(input_channels, channels[0])
        self.encoder0 = Stage(channels[0], channels[0])
        self.down0 = ReflectConv3d(channels[0], channels[1], stride=2)
        self.encoder1 = Stage(channels[1], channels[1])
        self.down1 = ReflectConv3d(channels[1], channels[2], stride=2)
        self.encoder2 = Stage(channels[2], channels[2])
        self.down2 = ReflectConv3d(channels[2], channels[3], stride=2)
        self.bottleneck = Stage(channels[3], channels[3])
        self.up2 = Up(channels[3], channels[2])
        self.decoder2 = Stage(channels[2] + channels[2], channels[2])
        self.up1 = Up(channels[2], channels[1])
        self.decoder1 = Stage(channels[1] + channels[1], channels[1])
        self.up0 = Up(channels[1], channels[0])
        self.decoder0 = Stage(channels[0] + channels[0], channels[0])
        self.output_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
        self.output = nn.Conv3d(channels[0], OUTPUT_CHANNELS, 1)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.output.bias.copy_(
                torch.tensor(
                    INITIAL_BIASES,
                    dtype=self.output.bias.dtype,
                )
            )

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if condition.ndim != 5 or condition.shape[1] != INPUT_CHANNELS:
            raise ValueError("V45 condition shape differs")
        level0 = self.encoder0(self.input(condition))
        level1 = self.encoder1(self.down0(level0))
        level2 = self.encoder2(self.down1(level1))
        hidden = self.bottleneck(self.down2(level2))
        hidden = self.decoder2(torch.cat((self.up2(hidden), level2), dim=1))
        hidden = self.decoder1(torch.cat((self.up1(hidden), level1), dim=1))
        hidden = self.decoder0(torch.cat((self.up0(hidden), level0), dim=1))
        return self.output(F.silu(self.output_norm(hidden)))


def mixture_parameters(
    parameters: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if parameters.ndim != 5 or parameters.shape[1] != OUTPUT_CHANNELS:
        raise ValueError("V45 mixture parameter shape differs")
    logits, locations, raw_scales = parameters.chunk(3, dim=1)
    scales = F.softplus(raw_scales.float()) + MINIMUM_SCALE
    return logits.float(), locations.float(), scales


def logistic_mixture_log_probability(
    parameters: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    logits, locations, scales = mixture_parameters(parameters)
    value = target.float()
    if value.shape != (len(parameters), 1, *parameters.shape[-3:]):
        raise ValueError("V45 likelihood target shape differs")
    standardized = (value - locations) / scales
    component = -standardized - 2.0 * F.softplus(-standardized) - torch.log(scales)
    return torch.logsumexp(F.log_softmax(logits, dim=1) + component, dim=1, keepdim=True)


def logistic_mixture_cdf(parameters: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    logits, locations, scales = mixture_parameters(parameters)
    if value.shape != (len(parameters), 1, *parameters.shape[-3:]):
        raise ValueError("V45 CDF value shape differs")
    weights = F.softmax(logits, dim=1)
    return torch.sum(weights * torch.sigmoid((value.float() - locations) / scales), dim=1, keepdim=True)


@torch.no_grad()
def logistic_mixture_inverse(
    parameters: torch.Tensor,
    uniform: torch.Tensor,
    *,
    steps: int = BISECTION_STEPS,
) -> torch.Tensor:
    logits, locations, scales = mixture_parameters(parameters)
    if uniform.shape != (len(parameters), 1, *parameters.shape[-3:]):
        raise ValueError("V45 mixture rank shape differs")
    probability = uniform.float().clamp(RANK_EPSILON, 1.0 - RANK_EPSILON)
    tail = math.log((1.0 - RANK_EPSILON) / RANK_EPSILON)
    lower = torch.min(locations - tail * scales, dim=1, keepdim=True).values
    upper = torch.max(locations + tail * scales, dim=1, keepdim=True).values
    weights = F.softmax(logits, dim=1)
    for _ in range(steps):
        middle = 0.5 * (lower + upper)
        cdf = torch.sum(
            weights * torch.sigmoid((middle - locations) / scales),
            dim=1,
            keepdim=True,
        )
        lower = torch.where(cdf < probability, middle, lower)
        upper = torch.where(cdf < probability, upper, middle)
    return 0.5 * (lower + upper)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "BISECTION_STEPS",
    "INITIAL_BIASES",
    "INITIAL_LOCATIONS",
    "INITIAL_RAW_SCALE",
    "INITIAL_SCALE",
    "INPUT_CHANNELS",
    "LocalMixtureUNet",
    "MIXTURES",
    "OUTPUT_CHANNELS",
    "logistic_mixture_cdf",
    "logistic_mixture_inverse",
    "logistic_mixture_log_probability",
    "mixture_parameters",
    "parameter_count",
]
