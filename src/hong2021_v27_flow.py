#!/usr/bin/env python
"""V27 conditional Haar flow with parent-aligned observed child phases."""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from hong2021_residual_v8_context import FEATURE_NAMES
from hong2021_v26_flow import (
    DEFAULT_BINS,
    DEFAULT_COUPLINGS,
    DEFAULT_LEVELS,
    DEFAULT_TAIL_BOUND,
    DETAIL_CHANNELS,
    ConditionalHaarSplineFlow,
    ConditionalScaleFlow,
)


CHILD_PHASES = 8


def space_to_depth_3d(value: torch.Tensor) -> torch.Tensor:
    """Pack each 2x2x2 child block into z-major binary phase channels."""
    if value.ndim != 5 or any(size % 2 for size in value.shape[-3:]):
        raise ValueError("3-D space-to-depth requires even 5-D spatial input")
    batch, channels, depth, height, width = value.shape
    return (
        value.reshape(
            batch,
            channels,
            depth // 2,
            2,
            height // 2,
            2,
            width // 2,
            2,
        )
        .permute(0, 1, 3, 5, 7, 2, 4, 6)
        .reshape(
            batch,
            channels * CHILD_PHASES,
            depth // 2,
            height // 2,
            width // 2,
        )
    )


def depth_to_space_3d(value: torch.Tensor) -> torch.Tensor:
    """Invert :func:`space_to_depth_3d` exactly."""
    if value.ndim != 5 or value.shape[1] % CHILD_PHASES:
        raise ValueError("3-D depth-to-space requires channels divisible by eight")
    batch, packed, depth, height, width = value.shape
    channels = packed // CHILD_PHASES
    return (
        value.reshape(batch, channels, 2, 2, 2, depth, height, width)
        .permute(0, 1, 5, 2, 6, 3, 7, 4)
        .reshape(batch, channels, depth * 2, height * 2, width * 2)
    )


class ParentAlignedConditionalHaarSplineFlow(ConditionalHaarSplineFlow):
    """V26 likelihood with lossless observed child phases at every parent."""

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
        # Initialize nn.Module directly so no discarded V26 flows consume RNG.
        nn.Module.__init__(self)
        mean = torch.as_tensor(detail_mean, dtype=torch.float32)
        std = torch.as_tensor(detail_std, dtype=torch.float32)
        feature_mean = torch.as_tensor(context_mean, dtype=torch.float32)
        feature_std = torch.as_tensor(context_std, dtype=torch.float32)
        if mean.shape != (levels, DETAIL_CHANNELS) or std.shape != mean.shape:
            raise ValueError("flow detail standardization has the wrong shape")
        if (
            feature_mean.shape != (len(FEATURE_NAMES),)
            or feature_std.shape != feature_mean.shape
        ):
            raise ValueError("flow observable-context standardization has the wrong shape")
        if (
            not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or torch.any(std <= 0)
        ):
            raise ValueError("flow detail standardization is invalid")
        if (
            not torch.isfinite(feature_mean).all()
            or not torch.isfinite(feature_std).all()
            or torch.any(feature_std <= 0)
        ):
            raise ValueError("flow context standardization is invalid")
        self.register_buffer("detail_mean", mean)
        self.register_buffer("detail_std", std)
        self.register_buffer("context_mean", feature_mean)
        self.register_buffer("context_std", feature_std)
        self.levels = int(levels)
        self.condition_channels = int(condition_channels)
        flow_context_channels = (
            condition_channels * CHILD_PHASES + 1 + len(FEATURE_NAMES)
        )
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

    def scale_context(
        self,
        condition: torch.Tensor,
        coarse_lowpass: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        resolution = coarse_lowpass.shape[-3:]
        child_resolution = tuple(2 * size for size in resolution)
        if any(
            child > available
            for child, available in zip(
                child_resolution, condition.shape[-3:], strict=True
            )
        ):
            raise ValueError("condition grid is too small for parent-aligned phases")
        pooled_children = F.adaptive_avg_pool3d(condition, child_resolution)
        phases = space_to_depth_3d(pooled_children)
        broadcast = global_context[:, :, None, None, None].expand(
            -1, -1, *resolution
        )
        return torch.cat(
            (phases, coarse_lowpass.to(dtype=condition.dtype), broadcast), dim=1
        )


__all__ = [
    "CHILD_PHASES",
    "ParentAlignedConditionalHaarSplineFlow",
    "depth_to_space_3d",
    "space_to_depth_3d",
]
