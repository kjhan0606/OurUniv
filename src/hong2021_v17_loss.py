#!/usr/bin/env python
"""Parameter-free equal-Fourier-band EDM objective for V17-E5."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from hong2021_residual_v6 import edm_denoise
from hong2021_residual_v9_tail import voxel_tail_weights
from hong2021_v14_multiscale import BAND_EDGES_H_MPC, fourier_band_masks


LOSS_COEFFICIENTS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
FFT_NORM = "ortho"
PARSEVAL_RELATIVE_TOLERANCE = 1.0e-4


def prepare_fourier_band_loss(
    grid: int, voxel_mpc_h: float, device: torch.device
) -> tuple[torch.Tensor, dict[str, Any]]:
    numpy_masks = fourier_band_masks(grid, voxel_mpc_h)
    counts = numpy_masks.reshape(len(numpy_masks), -1).sum(axis=1).astype(np.int64)
    if np.any(counts <= 0) or int(counts.sum()) != grid**3 - 1:
        raise RuntimeError("Fourier loss masks do not partition every non-DC mode")
    masks = torch.from_numpy(numpy_masks).to(device=device, dtype=torch.bool)
    probe = torch.linspace(
        -1.0, 1.0, grid**3, device=device, dtype=torch.float32
    ).reshape(1, 1, grid, grid, grid)
    with torch.autocast(device_type=device.type, enabled=False):
        spectrum = torch.fft.fftn(
            probe, dim=(-3, -2, -1), norm=FFT_NORM
        )
        power = spectrum.real.square() + spectrum.imag.square()
        spatial_energy = probe.square().sum(dtype=torch.float64)
        spectral_energy = power.sum(dtype=torch.float64)
        masked_energy = sum(
            power[:, :, mask].sum(dtype=torch.float64) for mask in masks
        ) + power[..., 0, 0, 0].sum(dtype=torch.float64)
    scale = max(float(spatial_energy), 1.0)
    parseval_error = abs(float(spectral_energy - spatial_energy)) / scale
    partition_error = abs(float(masked_energy - spectral_energy)) / scale
    if max(parseval_error, partition_error) > PARSEVAL_RELATIVE_TOLERANCE:
        raise RuntimeError("Fourier loss failed its Parseval/partition self-check")
    specification = {
        "coefficients": {
            "unweighted": LOSS_COEFFICIENTS[0],
            "tail_weighted": LOSS_COEFFICIENTS[1],
            "band_balanced": LOSS_COEFFICIENTS[2],
        },
        "fft_transform": "full_complex_fftn",
        "fft_norm": FFT_NORM,
        "fft_input_dtype": "float32",
        "fft_autocast_enabled": False,
        "dc_in_band_loss": False,
        "band_edges_h_mpc": [0.0, 1.0, 3.0, 6.0, "infinity"],
        "mode_counts": counts.tolist(),
        "non_dc_modes": int(counts.sum()),
        "grid": int(grid),
        "voxel_mpc_h": float(voxel_mpc_h),
        "parseval_relative_tolerance": PARSEVAL_RELATIVE_TOLERANCE,
        "parseval_self_check_relative_error": parseval_error,
        "mask_partition_self_check_relative_error": partition_error,
        "additional_rng_draws": 0,
    }
    return masks, specification


def fourier_band_error_per_sample(
    error: torch.Tensor, masks: torch.Tensor
) -> torch.Tensor:
    if error.ndim != 5 or masks.ndim != 4:
        raise ValueError("expected error (B,C,N,N,N) and masks (bands,N,N,N)")
    if tuple(error.shape[-3:]) != tuple(masks.shape[-3:]):
        raise ValueError("Fourier loss mask grid differs from the error grid")
    if masks.shape[0] != 4:
        raise ValueError("V17 requires exactly four Fourier bands")
    with torch.autocast(device_type=error.device.type, enabled=False):
        spectrum = torch.fft.fftn(
            error.float(), dim=(-3, -2, -1), norm=FFT_NORM
        )
        power = spectrum.real.square() + spectrum.imag.square()
        per_band = torch.stack(
            [power[:, :, mask].mean(dim=(1, 2)) for mask in masks], dim=1
        )
        result = per_band.mean(dim=1)
    if result.shape != (len(error),) or not torch.isfinite(result).all():
        raise RuntimeError("invalid equal-band Fourier loss")
    return result


def band_balanced_tail_edm_loss(
    model: nn.Module,
    residual: torch.Tensor,
    condition: torch.Tensor,
    truth: torch.Tensor,
    bin_weights: torch.Tensor,
    band_masks: torch.Tensor,
    generator: torch.Generator,
    sigma_data: float,
    edm_p_mean: float,
    edm_p_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    error = denoised - residual
    error2 = error.square()
    unweighted = (edm_weight * error2.mean(dim=(1, 2, 3, 4))).mean()
    tail_weight = voxel_tail_weights(truth, bin_weights)
    weighted_per_sample = (error2 * tail_weight).sum(dim=(1, 2, 3, 4)) / (
        tail_weight.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
    )
    tail_weighted = (edm_weight * weighted_per_sample).mean()
    band_per_sample = fourier_band_error_per_sample(error, band_masks)
    band_balanced = (edm_weight * band_per_sample).mean()
    combined = sum(
        coefficient * value
        for coefficient, value in zip(
            LOSS_COEFFICIENTS,
            (unweighted, tail_weighted, band_balanced),
            strict=True,
        )
    )
    return combined, unweighted, tail_weighted, band_balanced


@torch.inference_mode()
def fixed_band_balanced_validation_loss(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    bin_weights: torch.Tensor,
    band_masks: torch.Tensor,
    seed: int,
    sigma_data: float,
    edm_p_mean: float,
    edm_p_std: float,
) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    samples = 0
    for condition, residual, _, truth in loader:
        condition = condition.to(device, non_blocking=True)
        residual = residual.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss, _, _, _ = band_balanced_tail_edm_loss(
                model, residual, condition, truth, bin_weights, band_masks,
                generator, sigma_data, edm_p_mean, edm_p_std,
            )
        total += float(loss) * len(residual)
        samples += len(residual)
    return total / samples
