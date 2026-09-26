#!/usr/bin/env python
"""Statistically proper unweighted EDM objective frozen for V25."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hong2021_residual_v6 import edm_denoise
from hong2021_residual_v9_tail import voxel_tail_weights


def proper_unweighted_edm_loss(
    model: nn.Module,
    residual: torch.Tensor,
    condition: torch.Tensor,
    truth: torch.Tensor,
    bin_weights: torch.Tensor,
    generator: torch.Generator,
    sigma_data: float,
    edm_p_mean: float,
    edm_p_std: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the optimized proper loss and a detached tail diagnostic.

    The random draw order and unweighted formula are exactly the inherited EDM
    path.  The target-dependent tail statistic is computed only from detached
    errors and cannot contribute to the optimized graph.
    """
    batch = len(residual)
    sigma = torch.exp(
        torch.randn(batch, device=residual.device, generator=generator)
        * edm_p_std
        + edm_p_mean
    )
    noise = torch.randn(
        residual.shape, device=residual.device, generator=generator
    )
    noisy = residual + sigma[:, None, None, None, None] * noise
    denoised = edm_denoise(model, noisy, condition, sigma, sigma_data)
    edm_weight = (sigma.square() + sigma_data**2) / (
        sigma * sigma_data
    ).square()
    error2 = (denoised - residual).square()
    unweighted = (edm_weight * error2.mean(dim=(1, 2, 3, 4))).mean()
    with torch.no_grad():
        tail_weight = voxel_tail_weights(truth, bin_weights)
        weighted_per_sample = (
            error2.detach() * tail_weight
        ).sum(dim=(1, 2, 3, 4)) / tail_weight.sum(
            dim=(1, 2, 3, 4)
        ).clamp_min(1.0)
        tail_diagnostic = (edm_weight.detach() * weighted_per_sample).mean()
    return unweighted, tail_diagnostic


@torch.inference_mode()
def fixed_proper_unweighted_validation(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    bin_weights: torch.Tensor,
    seed: int,
    sigma_data: float,
    edm_p_mean: float,
    edm_p_std: float,
) -> tuple[float, float]:
    """Measure fixed-draw proper and tail-diagnostic validation losses."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    unweighted_sum = 0.0
    tail_sum = 0.0
    samples = 0
    for condition, residual, _, truth in loader:
        condition = condition.to(device, non_blocking=True)
        residual = residual.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            unweighted, tail = proper_unweighted_edm_loss(
                model,
                residual,
                condition,
                truth,
                bin_weights,
                generator,
                sigma_data,
                edm_p_mean,
                edm_p_std,
            )
        unweighted_sum += float(unweighted) * len(residual)
        tail_sum += float(tail) * len(residual)
        samples += len(residual)
    return unweighted_sum / samples, tail_sum / samples


__all__ = ["fixed_proper_unweighted_validation", "proper_unweighted_edm_loss"]
