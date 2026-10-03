#!/usr/bin/env python
"""V70 query-aligned joint latent-field EDM network and objective."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


CONDITION_CHANNELS = 7
BASE_CHANNELS = 32
TIME_CHANNELS = 128
ATTENTION_HEADS = 4


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


class NoiseEmbedding(nn.Module):
    def __init__(self, channels: int = TIME_CHANNELS) -> None:
        super().__init__()
        if channels % 2:
            raise ValueError("V70 noise embedding width must be even")
        self.channels = int(channels)
        self.network = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        if c_noise.ndim != 1:
            raise ValueError("V70 noise coordinate must be one-dimensional")
        half = self.channels // 2
        frequency = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=c_noise.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phase = c_noise.float()[:, None] * frequency[None] * 1000.0
        return self.network(torch.cat((phase.sin(), phase.cos()), dim=1))


class TimeResidualBlock(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, time_channels: int
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(input_channels), input_channels)
        self.conv1 = ReflectConv3d(input_channels, output_channels)
        self.time = nn.Linear(time_channels, output_channels)
        self.norm2 = nn.GroupNorm(_groups(output_channels), output_channels)
        self.conv2 = ReflectConv3d(output_channels, output_channels)
        self.skip = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv3d(input_channels, output_channels, 1)
        )

    def forward(self, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = hidden + self.time(time)[:, :, None, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return self.skip(value) + hidden


class TimeStage(nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, time_channels: int
    ) -> None:
        super().__init__()
        self.first = TimeResidualBlock(input_channels, output_channels, time_channels)
        self.second = TimeResidualBlock(output_channels, output_channels, time_channels)

    def forward(self, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(value, time), time)


class BottleneckAttention(nn.Module):
    """Position-free four-head attention over the complete 8^3 bottleneck."""

    def __init__(self, channels: int, heads: int = ATTENTION_HEADS) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("V70 attention channels must divide its head count")
        self.channels = int(channels)
        self.heads = int(heads)
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv3d(channels, 3 * channels, 1)
        self.output = nn.Conv3d(channels, channels, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, nz, ny, nx = value.shape
        spatial = nz * ny * nx
        head_channels = channels // self.heads
        qkv = self.qkv(self.norm(value)).reshape(
            batch, 3, self.heads, head_channels, spatial
        )
        query, key, content = qkv.unbind(dim=1)
        attended = F.scaled_dot_product_attention(
            query.transpose(-1, -2),
            key.transpose(-1, -2),
            content.transpose(-1, -2),
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(-1, -2).reshape(
            batch, channels, nz, ny, nx
        )
        return value + self.output(attended)


class Up(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = ReflectConv3d(input_channels, output_channels)

    def forward(self, value: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
        return self.conv(F.interpolate(value, size=shape, mode="nearest"))


class LatentSpatialUNet(nn.Module):
    """Three-scale conditional 3-D U-Net for a complete query-aligned z field."""

    def __init__(
        self,
        condition_channels: int = CONDITION_CHANNELS,
        base_channels: int = BASE_CHANNELS,
        time_channels: int = TIME_CHANNELS,
        attention_heads: int = ATTENTION_HEADS,
    ) -> None:
        super().__init__()
        if condition_channels != CONDITION_CHANNELS:
            raise ValueError("V70 requires all seven frozen condition channels")
        channels = (
            int(base_channels),
            2 * int(base_channels),
            4 * int(base_channels),
            4 * int(base_channels),
        )
        self.condition_channels = int(condition_channels)
        self.base_channels = int(base_channels)
        self.time_channels = int(time_channels)
        self.attention_heads = int(attention_heads)
        self.time = NoiseEmbedding(time_channels)
        self.input = ReflectConv3d(1 + condition_channels, channels[0])
        self.encoder0 = TimeStage(channels[0], channels[0], time_channels)
        self.down0 = ReflectConv3d(channels[0], channels[1], stride=2)
        self.encoder1 = TimeStage(channels[1], channels[1], time_channels)
        self.down1 = ReflectConv3d(channels[1], channels[2], stride=2)
        self.encoder2 = TimeStage(channels[2], channels[2], time_channels)
        self.down2 = ReflectConv3d(channels[2], channels[3], stride=2)
        self.bottleneck = TimeStage(channels[3], channels[3], time_channels)
        self.attention = BottleneckAttention(channels[3], attention_heads)
        self.up2 = Up(channels[3], channels[2])
        self.decoder2 = TimeStage(channels[2] + channels[2], channels[2], time_channels)
        self.up1 = Up(channels[2], channels[1])
        self.decoder1 = TimeStage(channels[1] + channels[1], channels[1], time_channels)
        self.up0 = Up(channels[1], channels[0])
        self.decoder0 = TimeStage(channels[0] + channels[0], channels[0], time_channels)
        self.output_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
        self.output = nn.Conv3d(channels[0], 1, 1)
        nn.init.normal_(self.output.weight, mean=0.0, std=1.0e-3)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        condition: torch.Tensor,
        c_noise: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_latent.ndim != 5 or noisy_latent.shape[1] != 1:
            raise ValueError("V70 noisy latent shape differs")
        if (
            condition.ndim != 5
            or condition.shape[1] != self.condition_channels
            or condition.shape[0] != noisy_latent.shape[0]
            or condition.shape[-3:] != noisy_latent.shape[-3:]
        ):
            raise ValueError("V70 condition shape differs")
        if any(size % 8 for size in noisy_latent.shape[-3:]):
            raise ValueError("V70 spatial dimensions must be divisible by eight")
        time = self.time(c_noise)
        level0 = self.encoder0(
            self.input(torch.cat((noisy_latent, condition), dim=1)), time
        )
        level1 = self.encoder1(self.down0(level0), time)
        level2 = self.encoder2(self.down1(level1), time)
        hidden = self.attention(self.bottleneck(self.down2(level2), time))
        hidden = self.decoder2(
            torch.cat((self.up2(hidden, level2.shape[-3:]), level2), dim=1), time
        )
        hidden = self.decoder1(
            torch.cat((self.up1(hidden, level1.shape[-3:]), level1), dim=1), time
        )
        hidden = self.decoder0(
            torch.cat((self.up0(hidden, level0.shape[-3:]), level0), dim=1), time
        )
        return self.output(F.silu(self.output_norm(hidden)))


def edm_coefficients(
    sigma: torch.Tensor, sigma_data: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma2 = sigma.float().square()
    data2 = float(sigma_data) ** 2
    denominator = sigma2 + data2
    return (
        data2 / denominator,
        sigma.float() * float(sigma_data) / denominator.sqrt(),
        denominator.rsqrt(),
        sigma.float().log() / 4.0,
    )


def edm_denoise(
    model: LatentSpatialUNet,
    noisy: torch.Tensor,
    condition: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 1.0,
) -> torch.Tensor:
    c_skip, c_out, c_in, c_noise = edm_coefficients(sigma, sigma_data)
    shape = (len(sigma),) + (1,) * (noisy.ndim - 1)
    network = model(c_in.reshape(shape) * noisy, condition, c_noise)
    return c_skip.reshape(shape) * noisy + c_out.reshape(shape) * network


def edm_loss(
    model: LatentSpatialUNet,
    latent: torch.Tensor,
    condition: torch.Tensor,
    sigma: torch.Tensor,
    noise: torch.Tensor,
    sigma_data: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if latent.shape != noise.shape:
        raise ValueError("V70 latent and noise shapes differ")
    noisy = latent + sigma[:, None, None, None, None] * noise
    denoised = edm_denoise(model, noisy, condition, sigma, sigma_data)
    weight = (sigma.square() + float(sigma_data) ** 2) / (
        sigma * float(sigma_data)
    ).square()
    per_object = (denoised.float() - latent.float()).square().mean(
        dim=(1, 2, 3, 4)
    )
    return (weight.float() * per_object).mean(), per_object


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "ATTENTION_HEADS",
    "BASE_CHANNELS",
    "CONDITION_CHANNELS",
    "TIME_CHANNELS",
    "LatentSpatialUNet",
    "edm_coefficients",
    "edm_denoise",
    "edm_loss",
    "parameter_count",
]
