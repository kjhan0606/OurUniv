#!/usr/bin/env python3
"""Rejected stationary covariance diagnostic for peak evidence.

The exact projection null space differs from a diagonal Fourier mask only on
the coarse/output-Nyquist equivalence classes.  This module removes every
strictly interior Ncoarse mode exactly and leaves those boundary classes free,
yielding a translation-invariant covariance grid.  Exact small-grid controls
show that this approximation is not accurate enough: conditional variance
depends materially on position relative to the coarse grid.  The routines are
retained only to reproduce that negative engineering result.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from cf4_peak_evidence import normalized_gaussian_logpdf


def stationary_null_mask_full(fine_n: int, coarse_n: int) -> np.ndarray:
    if fine_n <= 0 or coarse_n <= 0 or fine_n % 2 or coarse_n % 2:
        raise ValueError("fine_n and coarse_n must be positive and even")
    if coarse_n >= fine_n:
        raise ValueError("coarse_n must be smaller than fine_n")
    half = coarse_n // 2
    # Strict interior excludes +/- coarse/output Nyquist.  This is the exact
    # diagonal part of ker(R); only the folded boundary classes are omitted.
    signed = np.arange(-half + 1, half, dtype=np.int64)
    mapped = np.mod(signed, fine_n)
    free = np.ones((fine_n, fine_n, fine_n), dtype=bool)
    free[np.ix_(mapped, mapped, mapped)] = False
    return free


def stationary_signal_covariance_grid(
    filter_full: np.ndarray,
    coarse_n: int,
) -> np.ndarray:
    filter_full = np.asarray(filter_full)
    if filter_full.ndim != 3 or not (
        filter_full.shape[0] == filter_full.shape[1] == filter_full.shape[2]
    ):
        raise ValueError("filter_full must be cubic")
    fine_n = filter_full.shape[0]
    free = stationary_null_mask_full(fine_n, coarse_n)
    spectrum = np.abs(filter_full) ** 2 * free
    covariance = np.fft.ifftn(spectrum)
    imaginary_relative_rms = float(
        np.sqrt(np.mean(covariance.imag ** 2))
        / max(np.sqrt(np.mean(covariance.real ** 2)), np.finfo(float).tiny)
    )
    if imaginary_relative_rms > 1e-12:
        raise RuntimeError("stationary covariance is not real")
    return covariance.real


def covariance_at_points(
    covariance_grid: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    covariance_grid = np.asarray(covariance_grid, dtype=np.float64)
    if covariance_grid.ndim != 3 or not (
        covariance_grid.shape[0]
        == covariance_grid.shape[1]
        == covariance_grid.shape[2]
    ):
        raise ValueError("covariance_grid must be cubic")
    n = covariance_grid.shape[0]
    points = np.mod(np.asarray(points, dtype=np.int64), n)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (m,3)")
    covariance = np.empty((len(points), len(points)), dtype=np.float64)
    for row in range(len(points)):
        for column in range(len(points)):
            displacement = tuple(np.mod(points[row] - points[column], n))
            covariance[row, column] = covariance_grid[displacement]
    return 0.5 * (covariance + covariance.T)


def stationary_peak_log_evidence(
    predicted_mean: np.ndarray,
    targets: np.ndarray,
    signal_covariance: np.ndarray,
    sigma: float | np.ndarray,
) -> tuple[float, dict[str, Any]]:
    predicted_mean = np.asarray(predicted_mean, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    sig = np.broadcast_to(np.asarray(sigma, dtype=np.float64), targets.shape)
    observation_covariance = (
        np.asarray(signal_covariance, dtype=np.float64) + np.diag(sig ** 2)
    )
    logpdf, terms = normalized_gaussian_logpdf(
        targets, predicted_mean, observation_covariance
    )
    return logpdf, {
        "predicted_mean": predicted_mean,
        "signal_covariance": signal_covariance,
        "observation_covariance": observation_covariance,
        **terms,
    }


def approximation_metadata() -> dict[str, Any]:
    return {
        "role": "rejected translation-invariant covariance diagnostic",
        "exact_modes": "all strict-interior Ncoarse Fourier modes",
        "approximation": "coarse/output-Nyquist classes remain free",
        "rejection_reason": (
            "exact controls show material coarse-grid phase dependence and "
            "order-unity relative covariance error despite Gaussian smoothing"
        ),
        "all_parent_evidence_authorized": False,
    }
