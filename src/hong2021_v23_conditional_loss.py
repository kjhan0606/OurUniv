#!/usr/bin/env python
"""Frozen conditional-mean objective and diagnostic for V23."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from hong2021_residual_v6 import edm_denoise
from hong2021_residual_v9_tail import voxel_tail_weights


def conditional_bin_indices(
    conditional_mean: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Match numpy.searchsorted(..., side='right') - 1 with clamped tails."""
    if conditional_mean.ndim != 5 or conditional_mean.shape[1] != 1:
        raise ValueError("conditional mean must have shape (batch, 1, z, y, x)")
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("conditional edges must be a one-dimensional boundary vector")
    if not bool(torch.all(edges[1:] > edges[:-1])):
        raise ValueError("conditional edges must be strictly increasing")
    return (
        torch.bucketize(
            conditional_mean.to(dtype=edges.dtype).contiguous(), edges, right=True
        ) - 1
    ).clamp_(0, len(edges) - 2)


def conditional_mean_statistics(
    denoised: torch.Tensor,
    residual: torch.Tensor,
    conditional_mean: torch.Tensor,
    edges: torch.Tensor,
    minimum_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-sample/bin mean errors and their occupancy mask."""
    if denoised.shape != residual.shape or denoised.ndim != 5:
        raise ValueError("denoised and residual tensors must have the same 5D shape")
    if denoised.shape[1] != 1 or conditional_mean.shape != residual.shape:
        raise ValueError("V23 requires one residual and one conditional-mean channel")
    if minimum_count < 1:
        raise ValueError("minimum_count must be positive")
    assignment = conditional_bin_indices(conditional_mean, edges)
    batch = len(residual)
    bins = len(edges) - 1
    flat_assignment = assignment.reshape(batch, -1)
    flat_error = (denoised - residual).reshape(batch, -1)
    sums = flat_error.new_zeros((batch, bins))
    sums.scatter_add_(1, flat_assignment, flat_error)
    counts = torch.zeros(
        (batch, bins), dtype=torch.int64, device=flat_error.device
    )
    counts.scatter_add_(1, flat_assignment, torch.ones_like(flat_assignment))
    valid = counts >= minimum_count
    if not bool(valid.any(dim=1).all()):
        raise ValueError("at least one conditional bin per sample must pass occupancy")
    means = sums / counts.clamp_min(1).to(dtype=sums.dtype)
    return means, valid


def conditional_mean_penalty(
    denoised: torch.Tensor,
    residual: torch.Tensor,
    conditional_mean: torch.Tensor,
    edges: torch.Tensor,
    edm_weight: torch.Tensor,
    minimum_count: int = 64,
) -> torch.Tensor:
    """EDM-weighted mean squared conditional-bin bias, balanced per sample."""
    means, valid = conditional_mean_statistics(
        denoised, residual, conditional_mean, edges, minimum_count
    )
    per_sample = (means.square() * valid).sum(dim=1) / valid.sum(dim=1)
    if edm_weight.shape != (len(residual),):
        raise ValueError("EDM weight must have one scalar per sample")
    return (edm_weight * per_sample).mean()


def conditional_mean_tail_edm_loss(
    model: nn.Module,
    residual: torch.Tensor,
    condition: torch.Tensor,
    truth: torch.Tensor,
    bin_weights: torch.Tensor,
    conditional_edges: torch.Tensor,
    generator: torch.Generator,
    sigma_data: float,
    edm_p_mean: float,
    edm_p_std: float,
    lambda_conditional_mean: float = 1.0,
    minimum_count: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """V22 draw order and losses plus the single frozen V23 objective term."""
    if lambda_conditional_mean != 1.0:
        raise ValueError("V23 freezes lambda_conditional_mean at 1.0")
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
    tail_weight = voxel_tail_weights(truth, bin_weights)
    weighted_per_sample = (error2 * tail_weight).sum(dim=(1, 2, 3, 4)) / (
        tail_weight.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
    )
    weighted = (edm_weight * weighted_per_sample).mean()
    conditional = conditional_mean_penalty(
        denoised, residual, condition[:, 2:3], conditional_edges,
        edm_weight, minimum_count,
    )
    combined = 0.5 * unweighted + 0.5 * weighted + conditional
    return combined, unweighted, weighted, conditional


@torch.inference_mode()
def fixed_conditional_validation(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    conditional_edges: torch.Tensor,
    seed: int,
    sigma_data: float,
    edm_p_mean: float,
    edm_p_std: float,
    minimum_count: int = 64,
) -> float:
    """Mean EDM-weighted maximum absolute conditional-bin denoising error."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    samples = 0
    for condition, residual, _, _ in loader:
        condition = condition.to(device, non_blocking=True)
        residual = residual.to(device, non_blocking=True)
        batch = len(residual)
        sigma = torch.exp(
            torch.randn(batch, device=device, generator=generator) * edm_p_std
            + edm_p_mean
        )
        noise = torch.randn(residual.shape, device=device, generator=generator)
        noisy = residual + sigma[:, None, None, None, None] * noise
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            denoised = edm_denoise(model, noisy, condition, sigma, sigma_data)
            means, valid = conditional_mean_statistics(
                denoised, residual, condition[:, 2:3], conditional_edges,
                minimum_count,
            )
            maximum = means.abs().masked_fill(~valid, 0.0).max(dim=1).values
            edm_weight = (sigma.square() + sigma_data**2) / (
                sigma * sigma_data
            ).square()
            metric = edm_weight * maximum
        total += float(metric.sum())
        samples += batch
    if samples == 0:
        raise ValueError("fixed conditional validation loader is empty")
    return total / samples
