#!/usr/bin/env python
"""Fourier-band location-scale transforms for the V14 residual likelihood.

The transform is not a post-hoc transfer correction.  A residual is split by
fixed physical Fourier bands, each band is divided by a positive scale that
must be predicted from target-free observables, and the inverse applies the
same predicted scales before summing the bands.  The masks exhaust every
non-DC Fourier mode, so the operation is exactly invertible apart from the
separately modelled scalar cube location.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


BAND_EDGES_H_MPC = (0.0, 1.0, 3.0, 6.0, float("inf"))


@lru_cache(maxsize=16)
def fourier_band_masks(
    grid: int,
    voxel_mpc_h: float,
    edges: tuple[float, ...] = BAND_EDGES_H_MPC,
) -> np.ndarray:
    if grid <= 0 or voxel_mpc_h <= 0:
        raise ValueError("grid and voxel size must be positive")
    if len(edges) < 2 or edges[0] != 0 or not all(
        first < second for first, second in zip(edges[:-1], edges[1:], strict=True)
    ):
        raise ValueError("band edges must increase from zero")
    frequency = 2.0 * np.pi * np.fft.fftfreq(grid, d=voxel_mpc_h)
    radius = np.sqrt(
        frequency[:, None, None] ** 2
        + frequency[None, :, None] ** 2
        + frequency[None, None, :] ** 2
    )
    masks = np.stack(
        [
            (radius >= lower) & (radius < upper) & (radius > 0)
            for lower, upper in zip(edges[:-1], edges[1:], strict=True)
        ]
    )
    membership = masks.sum(axis=0)
    if membership[0, 0, 0] != 0:
        raise RuntimeError("DC mode entered a stochastic Fourier band")
    membership = membership.copy()
    membership[0, 0, 0] = 1
    if not np.all(membership == 1):
        raise RuntimeError("Fourier bands overlap or leave uncovered modes")
    return masks


def decompose_residual(
    residual: np.ndarray,
    *,
    voxel_mpc_h: float,
    edges: tuple[float, ...] = BAND_EDGES_H_MPC,
) -> tuple[float, np.ndarray]:
    value = np.asarray(residual, dtype=np.float64)
    if value.ndim != 3 or len(set(value.shape)) != 1 or not np.isfinite(value).all():
        raise ValueError("residual must be a finite cubic 3-D field")
    location = float(value.mean(dtype=np.float64))
    spectrum = np.fft.fftn(value - location)
    masks = fourier_band_masks(value.shape[0], voxel_mpc_h, edges)
    bands = np.stack(
        [np.fft.ifftn(spectrum * mask).real for mask in masks]
    )
    return location, bands


def compose_residual(location: float, bands: np.ndarray) -> np.ndarray:
    value = np.asarray(bands, dtype=np.float64)
    if value.ndim != 4 or len(set(value.shape[1:])) != 1:
        raise ValueError("bands must have shape (B,N,N,N)")
    if not np.isfinite(value).all() or not np.isfinite(location):
        raise ValueError("location and bands must be finite")
    return value.sum(axis=0) + float(location)


def standardize_residual(
    residual: np.ndarray,
    *,
    predicted_scales: np.ndarray,
    voxel_mpc_h: float,
    edges: tuple[float, ...] = BAND_EDGES_H_MPC,
) -> tuple[float, np.ndarray]:
    location, bands = decompose_residual(
        residual, voxel_mpc_h=voxel_mpc_h, edges=edges
    )
    scales = np.asarray(predicted_scales, dtype=np.float64)
    if scales.shape != (len(bands),) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("predicted scales must be one finite positive value per band")
    return location, compose_residual(0.0, bands / scales[:, None, None, None])


def inverse_standardized_residual(
    standardized: np.ndarray,
    *,
    predicted_location: float,
    predicted_scales: np.ndarray,
    voxel_mpc_h: float,
    edges: tuple[float, ...] = BAND_EDGES_H_MPC,
) -> np.ndarray:
    _, bands = decompose_residual(
        standardized, voxel_mpc_h=voxel_mpc_h, edges=edges
    )
    scales = np.asarray(predicted_scales, dtype=np.float64)
    if scales.shape != (len(bands),) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("predicted scales must be one finite positive value per band")
    return compose_residual(
        predicted_location, bands * scales[:, None, None, None]
    )
