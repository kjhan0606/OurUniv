#!/usr/bin/env python3
"""Exact coarse-translation phase cache for scalable peak evidence.

For Nfine/Ncoarse = r, the exact projector Q=I-R*R commutes with translations
by r fine cells.  A point-template response is therefore determined by its
absolute fine-cell phase modulo r and a periodic displacement.  At the
production ratio 576/192=3, only 27 response grids are required for any number
of Local-Group geometries.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from cf4_projection_contract import (
    add_restriction_adjoint,
    restriction_adjoint_spectrum,
    restrict_spectrum,
)


def full_spectrum_from_rfft(rfft: np.ndarray) -> np.ndarray:
    """Expand a cubic real-input rFFT using exact Hermitian indexing."""
    rfft = np.asarray(rfft)
    if rfft.ndim != 3 or rfft.shape[0] != rfft.shape[1]:
        raise ValueError("rfft must have shape (n,n,n//2+1)")
    n = rfft.shape[0]
    if rfft.shape[2] != n // 2 + 1:
        raise ValueError("rfft final-axis length mismatch")
    dtype = np.result_type(rfft.dtype, np.complex64)
    full = np.empty((n, n, n), dtype=dtype)
    half = n // 2
    full[..., :half + 1] = rfft
    if half > 1:
        negative_xy = np.mod(-np.arange(n), n)
        positive_z = np.arange(1, half)
        full[..., n - positive_z] = np.conjugate(
            rfft[np.ix_(negative_xy, negative_xy, positive_z)]
        )
    return full


def _validate_ratio(fine_n: int, coarse_n: int) -> int:
    if fine_n <= 0 or coarse_n <= 0 or fine_n % 2 or coarse_n % 2:
        raise ValueError("fine_n and coarse_n must be positive and even")
    if fine_n % coarse_n:
        raise ValueError("fine_n must be an integer multiple of coarse_n")
    ratio = fine_n // coarse_n
    if ratio <= 1:
        raise ValueError("fine_n/coarse_n must exceed one")
    return ratio


def impulse_spectrum(n: int, point: np.ndarray) -> np.ndarray:
    """Unitary full FFT of a unit spatial impulse without a forward FFT."""
    point = np.mod(np.asarray(point, dtype=np.int64), n)
    if point.shape != (3,):
        raise ValueError("point must have three coordinates")
    index = np.arange(n, dtype=np.float64)
    phases = [
        np.exp(-2j * np.pi * index * float(coordinate) / n)
        for coordinate in point
    ]
    return (
        phases[0][:, None, None]
        * phases[1][None, :, None]
        * phases[2][None, None, :]
        / np.sqrt(float(n ** 3))
    )


def phase_response_grid(
    filter_full: np.ndarray,
    coarse_n: int,
    phase: np.ndarray,
) -> np.ndarray:
    """Return A Q A* response to an impulse at one refinement phase."""
    filter_full = np.asarray(filter_full)
    if filter_full.ndim != 3 or not (
        filter_full.shape[0] == filter_full.shape[1] == filter_full.shape[2]
    ):
        raise ValueError("filter_full must be cubic")
    fine_n = filter_full.shape[0]
    ratio = _validate_ratio(fine_n, coarse_n)
    phase = np.asarray(phase, dtype=np.int64)
    if phase.shape != (3,) or np.any(phase < 0) or np.any(phase >= ratio):
        raise ValueError("phase must lie in [0, fine_n/coarse_n) on each axis")
    adjoint_template = (
        impulse_spectrum(fine_n, phase) * np.conjugate(filter_full)
    )
    low = restrict_spectrum(adjoint_template, coarse_n)
    null_template = add_restriction_adjoint(adjoint_template, -low)
    response_spectrum = filter_full * null_template
    response = np.fft.ifftn(response_spectrum, norm="ortho")
    real_rms = float(np.sqrt(np.mean(response.real ** 2)))
    imaginary_rms = float(np.sqrt(np.mean(response.imag ** 2)))
    if imaginary_rms / max(real_rms, np.finfo(float).tiny) > 1e-12:
        raise RuntimeError("phase response broke Hermitian symmetry")
    return response.real


def covariance_for_point_sets(
    filter_full: np.ndarray,
    coarse_n: int,
    point_sets: list[np.ndarray],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Evaluate exact AQA* matrices while holding one phase grid at a time."""
    fine_n = np.asarray(filter_full).shape[0]
    ratio = _validate_ratio(fine_n, coarse_n)
    normalized_sets = []
    tasks: dict[tuple[int, int, int], list[tuple[int, int, int]]] = defaultdict(list)
    covariance = []
    for set_index, points in enumerate(point_sets):
        points = np.mod(np.asarray(points, dtype=np.int64), fine_n)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("every point set must have shape (m,3)")
        normalized_sets.append(points)
        covariance.append(np.empty((len(points), len(points)), dtype=np.float64))
        for row in range(len(points)):
            for column in range(len(points)):
                phase = tuple(int(value) for value in np.mod(points[column], ratio))
                tasks[phase].append((set_index, row, column))

    for phase in sorted(tasks):
        grid = phase_response_grid(filter_full, coarse_n, np.asarray(phase))
        for set_index, row, column in tasks[phase]:
            points = normalized_sets[set_index]
            source_translation = points[column] - np.asarray(phase)
            location = tuple(np.mod(points[row] - source_translation, fine_n))
            covariance[set_index][row, column] = grid[location]
        del grid
    maximum_asymmetry = 0.0
    for index, matrix in enumerate(covariance):
        maximum_asymmetry = max(
            maximum_asymmetry, float(np.max(np.abs(matrix - matrix.T)))
        )
        covariance[index] = 0.5 * (matrix + matrix.T)
    return covariance, {
        "refinement_ratio": ratio,
        "phase_count_used": len(tasks),
        "maximum_possible_phase_count": ratio ** 3,
        "response_grids_held_simultaneously": 1,
        "maximum_pre_symmetrization_asymmetry": maximum_asymmetry,
    }


def parent_mean_at_point_sets(
    coarse: np.ndarray,
    filter_full: np.ndarray,
    point_sets: list[np.ndarray],
) -> list[np.ndarray]:
    """Evaluate exact A R* y means for all point sets with one fine inverse FFT."""
    coarse = np.asarray(coarse, dtype=np.float64)
    filter_full = np.asarray(filter_full)
    if coarse.ndim != 3 or not (
        coarse.shape[0] == coarse.shape[1] == coarse.shape[2]
    ):
        raise ValueError("coarse must be cubic")
    fine_n = filter_full.shape[0]
    _validate_ratio(fine_n, coarse.shape[0])
    coarse_fft = np.fft.fftn(coarse, norm="ortho")
    fine_mean_fft = restriction_adjoint_spectrum(coarse_fft, fine_n)
    response = np.fft.ifftn(filter_full * fine_mean_fft, norm="ortho")
    real_rms = float(np.sqrt(np.mean(response.real ** 2)))
    imaginary_rms = float(np.sqrt(np.mean(response.imag ** 2)))
    if imaginary_rms / max(real_rms, np.finfo(float).tiny) > 1e-12:
        raise RuntimeError("parent mean response broke Hermitian symmetry")
    result = []
    for points in point_sets:
        points = np.mod(np.asarray(points, dtype=np.int64), fine_n)
        result.append(response.real[tuple(points.T)])
    return result


def phase_cache_metadata() -> dict[str, Any]:
    return {
        "covariance": "exact AQA*; no stationary approximation",
        "production_ratio": 3,
        "production_phase_count": 27,
        "memory_policy": "one Nfine response grid at a time",
        "parent_mean": "one exact fine inverse FFT per parent for all point sets",
        "FFT_workers": "NumPy default; no workers=-1",
        "all_parent_evidence_authorized": False,
    }
