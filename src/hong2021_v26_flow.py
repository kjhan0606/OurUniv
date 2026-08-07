#!/usr/bin/env python
"""Conditional multiscale Haar spline flow proposed for Hong-density V26.

The flow operates on all 64^3-1 Haar detail coefficients and fixes the sole
coarsest DC coefficient to zero.  Coarse-to-fine factorization makes every
detail scale conditional on observables and already reconstructed coarser
latent structure.  Sampling is direct; there is no diffusion trajectory.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from nflows.transforms.splines.rational_quadratic import (
    unconstrained_rational_quadratic_spline,
)
from torch import nn
from torch.nn import functional as F

from hong2021_residual_v8_context import FEATURE_NAMES, observable_context_features
from hong2021_v26_haar import haar_pyramid, haar_synthesis


DETAIL_CHANNELS = 7
DEFAULT_LEVELS = 6
DEFAULT_COUPLINGS = 4
DEFAULT_BINS = 8
DEFAULT_TAIL_BOUND = 6.0
SPLINE_PARAMETERS = 3 * DEFAULT_BINS - 1


def _identity_derivative_logit(minimum_derivative: float = 1.0e-3) -> float:
    return float(math.log(math.expm1(1.0 - minimum_derivative)))


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = ChannelNorm3D(channels)
        self.conv1 = nn.Conv3d(
            channels, channels, 3, padding=1, padding_mode="circular"
        )
        self.norm2 = ChannelNorm3D(channels)
        self.conv2 = nn.Conv3d(
            channels, channels, 3, padding=1, padding_mode="circular"
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = self.conv1(F.silu(self.norm1(value)))
        update = self.conv2(F.silu(self.norm2(update)))
        return value + update


class ChannelNorm3D(nn.Module):
    """Per-voxel channel normalization, including batch-one 1^3 tensors."""

    def __init__(self, channels: int, epsilon: float = 1.0e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.epsilon = float(epsilon)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mean = value.mean(dim=1, keepdim=True)
        variance = (value - mean).square().mean(dim=1, keepdim=True)
        normalized = (value - mean) * torch.rsqrt(variance + self.epsilon)
        return (
            normalized * self.weight[None, :, None, None, None]
            + self.bias[None, :, None, None, None]
        )


class SplineConditioner3D(nn.Module):
    """Local periodic conditioner with an exactly identity output at launch."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        *,
        blocks: int = 2,
        bins: int = DEFAULT_BINS,
    ) -> None:
        super().__init__()
        self.input = nn.Conv3d(
            input_channels,
            hidden_channels,
            3,
            padding=1,
            padding_mode="circular",
        )
        self.blocks = nn.Sequential(
            *(ResidualBlock3D(hidden_channels) for _ in range(blocks))
        )
        self.output = nn.Conv3d(hidden_channels, output_channels, 1)
        nn.init.zeros_(self.output.weight)
        bias = torch.zeros(output_channels)
        parameters = 3 * bins - 1
        derivative = _identity_derivative_logit()
        for channel in range(output_channels // parameters):
            start = channel * parameters + 2 * bins
            bias[start : start + bins - 1] = derivative
        with torch.no_grad():
            self.output.bias.copy_(bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(value)))


class ConditionalSplineCoupling3D(nn.Module):
    """One channel coupling transform over a seven-channel Haar detail field."""

    def __init__(
        self,
        passive_channels: Sequence[int],
        context_channels: int,
        *,
        hidden_channels: int = 32,
        bins: int = DEFAULT_BINS,
        tail_bound: float = DEFAULT_TAIL_BOUND,
        blocks: int = 2,
    ) -> None:
        super().__init__()
        passive = torch.zeros(DETAIL_CHANNELS, dtype=torch.bool)
        passive[list(passive_channels)] = True
        if not bool(passive.any()) or bool(passive.all()):
            raise ValueError("spline coupling requires passive and active channels")
        self.register_buffer("passive", passive)
        self.bins = int(bins)
        self.tail_bound = float(tail_bound)
        self.parameters_per_channel = 3 * self.bins - 1
        self.conditioner = SplineConditioner3D(
            DETAIL_CHANNELS + context_channels,
            hidden_channels,
            DETAIL_CHANNELS * self.parameters_per_channel,
            blocks=blocks,
            bins=self.bins,
        )

    def _spline_parameters(
        self, value: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        passive = self.passive.view(1, -1, 1, 1, 1)
        conditioned = torch.cat((value * passive, context), dim=1)
        raw = self.conditioner(conditioned)
        batch, _, depth, height, width = raw.shape
        raw = raw.reshape(
            batch,
            DETAIL_CHANNELS,
            self.parameters_per_channel,
            depth,
            height,
            width,
        ).permute(0, 1, 3, 4, 5, 2)
        active = ~self.passive
        raw = raw[:, active].float()
        widths = raw[..., : self.bins]
        heights = raw[..., self.bins : 2 * self.bins]
        derivatives = raw[..., 2 * self.bins :]
        return widths, heights, derivatives

    def transform(
        self,
        value: torch.Tensor,
        context: torch.Tensor,
        *,
        inverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if value.ndim != 5 or value.shape[1] != DETAIL_CHANNELS:
            raise ValueError("spline coupling expects seven detail channels")
        if context.ndim != 5 or context.shape[0] != value.shape[0]:
            raise ValueError("spline coupling context has the wrong shape")
        if context.shape[-3:] != value.shape[-3:]:
            raise ValueError("spline coupling context/detail grids differ")
        widths, heights, derivatives = self._spline_parameters(value, context)
        active = ~self.passive
        transformed, logabsdet = unconstrained_rational_quadratic_spline(
            value[:, active].float(),
            widths,
            heights,
            derivatives,
            inverse=inverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )
        output = value.clone()
        output[:, active] = transformed.to(dtype=value.dtype)
        return output, logabsdet.flatten(1).sum(dim=1)

    def forward(
        self, value: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(value, context, inverse=False)

    def inverse(
        self, value: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(value, context, inverse=True)


class ConditionalScaleFlow(nn.Module):
    def __init__(
        self,
        context_channels: int,
        *,
        hidden_channels: int = 32,
        couplings: int = DEFAULT_COUPLINGS,
        bins: int = DEFAULT_BINS,
        tail_bound: float = DEFAULT_TAIL_BOUND,
    ) -> None:
        super().__init__()
        if couplings <= 0 or couplings % 2:
            raise ValueError("scale flow requires a positive even coupling count")
        masks = ((0, 1, 2), (3, 4, 5, 6))
        self.layers = nn.ModuleList(
            [
                ConditionalSplineCoupling3D(
                    masks[index % 2],
                    context_channels,
                    hidden_channels=hidden_channels,
                    bins=bins,
                    tail_bound=tail_bound,
                )
                for index in range(couplings)
            ]
        )

    def forward(
        self, value: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total = value.new_zeros(len(value), dtype=torch.float32)
        for layer in self.layers:
            value, logabsdet = layer(value, context)
            total = total + logabsdet
        return value, total

    def inverse(
        self, value: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total = value.new_zeros(len(value), dtype=torch.float32)
        for layer in reversed(self.layers):
            value, logabsdet = layer.inverse(value, context)
            total = total + logabsdet
        return value, total


class ConditionalHaarSplineFlow(nn.Module):
    """Exact coarse-to-fine conditional density on the non-DC Haar subspace."""

    def __init__(
        self,
        *,
        detail_mean: Sequence[Sequence[float]],
        detail_std: Sequence[Sequence[float]],
        context_mean: Sequence[float],
        context_std: Sequence[float],
        condition_channels: int = 4,
        hidden_channels: int = 32,
        levels: int = DEFAULT_LEVELS,
        couplings: int = DEFAULT_COUPLINGS,
        bins: int = DEFAULT_BINS,
        tail_bound: float = DEFAULT_TAIL_BOUND,
    ) -> None:
        super().__init__()
        mean = torch.as_tensor(detail_mean, dtype=torch.float32)
        std = torch.as_tensor(detail_std, dtype=torch.float32)
        feature_mean = torch.as_tensor(context_mean, dtype=torch.float32)
        feature_std = torch.as_tensor(context_std, dtype=torch.float32)
        if mean.shape != (levels, DETAIL_CHANNELS) or std.shape != mean.shape:
            raise ValueError("flow detail standardization has the wrong shape")
        if feature_mean.shape != (len(FEATURE_NAMES),) or feature_std.shape != feature_mean.shape:
            raise ValueError("flow observable-context standardization has the wrong shape")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("flow detail standardization is invalid")
        if not torch.isfinite(feature_mean).all() or not torch.isfinite(feature_std).all() or torch.any(feature_std <= 0):
            raise ValueError("flow context standardization is invalid")
        self.register_buffer("detail_mean", mean)
        self.register_buffer("detail_std", std)
        self.register_buffer("context_mean", feature_mean)
        self.register_buffer("context_std", feature_std)
        self.levels = int(levels)
        self.condition_channels = int(condition_channels)
        flow_context_channels = condition_channels + 1 + len(FEATURE_NAMES)
        self.flows = nn.ModuleList(
            [
                ConditionalScaleFlow(
                    flow_context_channels,
                    hidden_channels=hidden_channels,
                    couplings=couplings,
                    bins=bins,
                    tail_bound=tail_bound,
                )
                for _ in range(levels)
            ]
        )

    def standardized_global_context(self, condition: torch.Tensor) -> torch.Tensor:
        features = observable_context_features(condition)
        return (features - self.context_mean) / self.context_std

    def scale_context(
        self,
        condition: torch.Tensor,
        coarse_lowpass: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        resolution = coarse_lowpass.shape[-3:]
        pooled = F.adaptive_avg_pool3d(condition, resolution)
        broadcast = global_context[:, :, None, None, None].expand(
            -1, -1, *resolution
        )
        return torch.cat((pooled, coarse_lowpass, broadcast), dim=1)

    def log_prob(
        self, latent: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if latent.shape[1] != 1 or condition.shape[1] != self.condition_channels:
            raise ValueError("flow latent or condition channels differ")
        if latent.shape[0] != condition.shape[0] or latent.shape[-3:] != condition.shape[-3:]:
            raise ValueError("flow latent and condition shapes differ")
        lowpass, details = haar_pyramid(latent, levels=self.levels)
        coarsest_dc = lowpass.flatten(1)[:, 0]
        global_context = self.standardized_global_context(condition)
        total = latent.new_zeros(len(latent), dtype=torch.float32)
        affine = latent.new_zeros(len(latent), dtype=torch.float32)
        base = latent.new_zeros(len(latent), dtype=torch.float32)
        scale_rows = []
        spline_rows = []
        for coarse_index, (flow, raw_detail) in enumerate(
            zip(self.flows, reversed(details), strict=True)
        ):
            level = self.levels - 1 - coarse_index
            mean = self.detail_mean[level][None, :, None, None, None]
            std = self.detail_std[level][None, :, None, None, None]
            detail = (raw_detail - mean) / std
            context = self.scale_context(condition, lowpass, global_context)
            transformed, logabsdet = flow(detail, context)
            base_log_prob = -0.5 * (
                transformed.float().square() + math.log(2.0 * math.pi)
            ).flatten(1).sum(dim=1)
            spatial = raw_detail.shape[-3] * raw_detail.shape[-2] * raw_detail.shape[-1]
            affine_logdet = -spatial * torch.log(self.detail_std[level]).sum()
            scale_log_prob = base_log_prob + logabsdet + affine_logdet
            total = total + scale_log_prob
            base = base + base_log_prob
            affine = affine + affine_logdet
            scale_rows.append(scale_log_prob)
            spline_rows.append(logabsdet)
            lowpass = haar_synthesis(lowpass, raw_detail)
        return total, {
            "base_log_prob": base,
            "affine_logabsdet": affine,
            "coarsest_dc": coarsest_dc,
            "scale_log_prob_coarse_to_fine": torch.stack(scale_rows, dim=1),
            "spline_logabsdet_coarse_to_fine": torch.stack(spline_rows, dim=1),
        }

    @torch.inference_mode()
    def sample(
        self,
        condition: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        sample, _ = self.sample_with_diagnostics(condition, generator=generator)
        return sample

    @torch.inference_mode()
    def sample_with_diagnostics(
        self,
        condition: torch.Tensor,
        *,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if condition.ndim != 5 or condition.shape[1] != self.condition_channels:
            raise ValueError("flow condition has the wrong shape")
        batch = len(condition)
        global_context = self.standardized_global_context(condition)
        lowpass = condition.new_zeros((batch, 1, 1, 1, 1))
        for coarse_index, flow in enumerate(self.flows):
            level = self.levels - 1 - coarse_index
            resolution = 2**coarse_index
            base = torch.randn(
                (batch, DETAIL_CHANNELS, resolution, resolution, resolution),
                device=condition.device,
                dtype=condition.dtype,
                generator=generator,
            )
            context = self.scale_context(condition, lowpass, global_context)
            detail, _ = flow.inverse(base, context)
            mean = self.detail_mean[level][None, :, None, None, None]
            std = self.detail_std[level][None, :, None, None, None]
            raw_detail = detail * std + mean
            lowpass = haar_synthesis(lowpass, raw_detail)
        pre_center_mean = lowpass.mean(dim=(-3, -2, -1), keepdim=True)
        lowpass -= pre_center_mean
        post_center_mean = lowpass.mean(dim=(-3, -2, -1), keepdim=True)
        return lowpass, {
            "pre_center_mean": pre_center_mean.flatten(),
            "post_center_mean": post_center_mean.flatten(),
        }


__all__ = [
    "ConditionalSplineCoupling3D",
    "ConditionalScaleFlow",
    "ConditionalHaarSplineFlow",
]
