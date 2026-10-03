#!/usr/bin/env python
"""Orthonormal 3-D Haar coordinates for the zero-DC V21 latent field."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def haar_kernels(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return eight fixed 2x2x2 orthonormal analysis kernels, DC first."""
    low = torch.tensor([1.0, 1.0], device=device, dtype=dtype) / math.sqrt(2.0)
    high = torch.tensor([1.0, -1.0], device=device, dtype=dtype) / math.sqrt(2.0)
    one_dimensional = (low, high)
    kernels = []
    for z_high in (0, 1):
        for y_high in (0, 1):
            for x_high in (0, 1):
                kernels.append(
                    torch.einsum(
                        "i,j,k->ijk",
                        one_dimensional[z_high],
                        one_dimensional[y_high],
                        one_dimensional[x_high],
                    )
                )
    return torch.stack(kernels)[:, None]


def haar_analysis(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split one-channel even 3-D fields into lowpass and seven details."""
    if value.ndim != 5 or value.shape[1] != 1:
        raise ValueError("Haar analysis expects shape (batch,1,z,y,x)")
    if any(size % 2 for size in value.shape[-3:]):
        raise ValueError("Haar analysis requires even spatial dimensions")
    kernels = haar_kernels(device=value.device, dtype=value.dtype)
    coefficients = F.conv3d(value, kernels, stride=2)
    return coefficients[:, :1], coefficients[:, 1:]


def haar_synthesis(lowpass: torch.Tensor, details: torch.Tensor) -> torch.Tensor:
    """Invert one orthonormal Haar level."""
    if lowpass.ndim != 5 or lowpass.shape[1] != 1:
        raise ValueError("Haar lowpass must have one channel")
    if details.ndim != 5 or details.shape[1] != 7:
        raise ValueError("Haar details must have seven channels")
    if lowpass.shape[0] != details.shape[0] or lowpass.shape[-3:] != details.shape[-3:]:
        raise ValueError("Haar lowpass/detail shapes differ")
    kernels = haar_kernels(device=lowpass.device, dtype=lowpass.dtype)
    return F.conv_transpose3d(torch.cat((lowpass, details), dim=1), kernels, stride=2)


def haar_pyramid(
    value: torch.Tensor, *, levels: int = 6
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return the coarsest lowpass and fine-to-coarse detail tensors."""
    if levels <= 0:
        raise ValueError("Haar pyramid levels must be positive")
    lowpass = value
    details = []
    for _ in range(levels):
        lowpass, detail = haar_analysis(lowpass)
        details.append(detail)
    return lowpass, details


def inverse_haar_pyramid(
    lowpass: torch.Tensor, details_fine_to_coarse: list[torch.Tensor]
) -> torch.Tensor:
    """Invert a list produced by :func:`haar_pyramid`."""
    if not details_fine_to_coarse:
        raise ValueError("Haar inverse requires at least one detail level")
    value = lowpass
    for detail in reversed(details_fine_to_coarse):
        value = haar_synthesis(value, detail)
    return value


def detail_dimensions(grid: int = 64, levels: int = 6) -> list[int]:
    """Return fine-to-coarse detail coefficient counts."""
    if grid <= 0 or grid % (2**levels):
        raise ValueError("grid must be divisible by every Haar level")
    return [7 * (grid // (2 ** (level + 1))) ** 3 for level in range(levels)]


__all__ = [
    "haar_kernels",
    "haar_analysis",
    "haar_synthesis",
    "haar_pyramid",
    "inverse_haar_pyramid",
    "detail_dimensions",
]
