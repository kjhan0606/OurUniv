#!/usr/bin/env python3
"""Exact capped-worker Matheron completion of an N192 parent on N576.

The implementation keeps the variance-preserving restriction contract,
including output-Nyquist folds, while permitting complex64 production FFTs.
It consumes one already-promoted geometry/parent state and therefore does not
apply the LG peak evidence a second time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import fft as spfft

from cf4_peak_evidence_phase_cache import covariance_for_point_sets


def _mesh_indices(source_n: int, output_n: int) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    if source_n <= 0 or output_n <= 0 or source_n % 2 or output_n % 2:
        raise ValueError("mesh sizes must be positive and even")
    if output_n > source_n:
        raise ValueError("output mesh cannot exceed source mesh")
    half = output_n // 2
    signed = np.r_[np.arange(half), np.arange(-half, 0)]
    base = np.mod(signed, source_n)
    alternate_signed = signed.copy()
    alternate_signed[half] = half
    alternate = np.mod(alternate_signed, source_n)
    boundary = np.zeros(output_n, dtype=np.int8)
    boundary[half] = 1
    multiplicity_power = (
        boundary[:, None, None]
        + boundary[None, :, None]
        + boundary[None, None, :]
    )
    denominator = np.power(2.0, 3.0 - 0.5 * multiplicity_power)
    return (base, alternate), denominator


def restrict_spectrum_preserve_dtype(
    source_fft: np.ndarray, output_n: int
) -> np.ndarray:
    """Apply R without promoting a production complex64 spectrum."""
    source = np.asarray(source_fft)
    if source.ndim != 3 or not (source.shape[0] == source.shape[1] == source.shape[2]):
        raise ValueError("source spectrum must be cubic")
    choices, denominator = _mesh_indices(source.shape[0], int(output_n))
    result = np.zeros((output_n,) * 3, dtype=source.dtype)
    for choose_x in range(2):
        for choose_y in range(2):
            for choose_z in range(2):
                result += source[np.ix_(
                    choices[choose_x], choices[choose_y], choices[choose_z]
                )]
    result /= denominator.astype(result.real.dtype, copy=False)
    return result


def add_restriction_adjoint_preserve_dtype(
    source_fft: np.ndarray, target_fft: np.ndarray
) -> np.ndarray:
    """Return source + R*target without implicit complex128 promotion."""
    source = np.asarray(source_fft)
    target = np.asarray(target_fft)
    if source.ndim != 3 or not (source.shape[0] == source.shape[1] == source.shape[2]):
        raise ValueError("source spectrum must be cubic")
    if target.ndim != 3 or not (target.shape[0] == target.shape[1] == target.shape[2]):
        raise ValueError("target spectrum must be cubic")
    choices, denominator = _mesh_indices(source.shape[0], target.shape[0])
    result = source.copy()
    contribution = target.astype(source.dtype, copy=False) / denominator.astype(
        source.real.dtype, copy=False
    )
    for choose_x in range(2):
        for choose_y in range(2):
            for choose_z in range(2):
                result[np.ix_(
                    choices[choose_x], choices[choose_y], choices[choose_z]
                )] += contribution
    return result


def null_project_spectrum_preserve_dtype(
    source_fft: np.ndarray, coarse_n: int
) -> np.ndarray:
    low = restrict_spectrum_preserve_dtype(source_fft, coarse_n)
    return add_restriction_adjoint_preserve_dtype(source_fft, -low)


def prolong_white_spectrum(
    coarse: np.ndarray,
    fine_n: int,
    seed: int,
    *,
    float_dtype: np.dtype = np.dtype(np.float32),
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw z+R*(y-Rz), returning its spectrum and the coarse spectrum."""
    if workers <= 0:
        raise ValueError("FFT workers must be a positive explicit cap")
    dtype = np.dtype(float_dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("float_dtype must be float32 or float64")
    coarse = np.asarray(coarse, dtype=dtype)
    if coarse.ndim != 3 or not (coarse.shape[0] == coarse.shape[1] == coarse.shape[2]):
        raise ValueError("coarse field must be cubic")
    if fine_n <= coarse.shape[0] or fine_n % coarse.shape[0]:
        raise ValueError("fine mesh must be a larger integer multiple")
    rng = np.random.Generator(np.random.PCG64DXSM(int(seed)))
    if dtype == np.dtype(np.float32):
        base = rng.standard_normal((fine_n,) * 3, dtype=np.float32)
    else:
        base = rng.standard_normal((fine_n,) * 3)
    base_fft = spfft.fftn(base, norm="ortho", workers=workers)
    del base
    coarse_fft = spfft.fftn(coarse, norm="ortho", workers=workers)
    residual = coarse_fft - restrict_spectrum_preserve_dtype(
        base_fft, coarse.shape[0]
    )
    conditioned_fft = add_restriction_adjoint_preserve_dtype(base_fft, residual)
    return conditioned_fft, coarse_fft


def _response_at_points(
    spectrum: np.ndarray,
    filter_full: np.ndarray,
    points: np.ndarray,
    *,
    workers: int,
) -> np.ndarray:
    response = spfft.ifftn(
        spectrum * filter_full, norm="ortho", workers=workers
    )
    real_rms = float(np.sqrt(np.mean(response.real**2)))
    imaginary_rms = float(np.sqrt(np.mean(response.imag**2)))
    tolerance = 2.0e-5 if spectrum.dtype == np.dtype(np.complex64) else 1.0e-12
    if imaginary_rms / max(real_rms, np.finfo(float).tiny) > tolerance:
        raise RuntimeError("filtered response broke Hermitian symmetry")
    return np.asarray(response.real[tuple(points.T)], dtype=np.float64)


def _spectrum_relative_error(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left) - np.asarray(right)
    scale = float(np.sqrt(np.mean(np.abs(right) ** 2)))
    return float(np.sqrt(np.mean(np.abs(difference) ** 2)) / max(scale, 1.0e-30))


def conditional_field(
    coarse: np.ndarray,
    filter_full: np.ndarray,
    points: np.ndarray,
    targets: np.ndarray,
    sigma: float | np.ndarray,
    *,
    fine_seed: int,
    noise_seed: int,
    signal_covariance: np.ndarray | None = None,
    float_dtype: np.dtype = np.dtype(np.float32),
    workers: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw one exact conditional fine field for an already weighted joint row."""
    if workers <= 0:
        raise ValueError("FFT workers must be a positive explicit cap")
    coarse = np.asarray(coarse, dtype=float_dtype)
    filter_full = np.asarray(filter_full)
    fine_n = filter_full.shape[0]
    if filter_full.shape != (fine_n,) * 3:
        raise ValueError("filter_full must be cubic")
    points = np.mod(np.asarray(points, dtype=np.int64), fine_n)
    targets = np.asarray(targets, dtype=np.float64)
    if points.shape != (len(targets), 3):
        raise ValueError("points and targets do not align")
    sig = np.broadcast_to(np.asarray(sigma, dtype=np.float64), targets.shape)
    if np.any(~np.isfinite(sig)) or np.any(sig <= 0.0):
        raise ValueError("likelihood sigma must be finite and positive")
    if signal_covariance is None:
        covariance, _ = covariance_for_point_sets(
            filter_full, coarse.shape[0], [points]
        )
        signal = covariance[0]
    else:
        signal = np.asarray(signal_covariance, dtype=np.float64)
    if signal.shape != (len(targets), len(targets)):
        raise ValueError("signal covariance shape mismatch")
    signal = 0.5 * (signal + signal.T)
    observation = signal + np.diag(sig**2)
    np.linalg.cholesky(observation)

    base_fft, coarse_fft = prolong_white_spectrum(
        coarse, fine_n, fine_seed, float_dtype=float_dtype, workers=workers
    )
    predicted = _response_at_points(
        base_fft, filter_full, points, workers=workers
    )
    rng = np.random.Generator(np.random.PCG64DXSM(int(noise_seed)))
    mock_noise = rng.normal(0.0, sig)
    weights = np.linalg.solve(observation, targets - predicted - mock_noise)

    impulse = np.zeros((fine_n,) * 3, dtype=float_dtype)
    np.add.at(impulse, tuple(points.T), weights.astype(impulse.dtype))
    impulse_fft = spfft.fftn(impulse, norm="ortho", workers=workers)
    del impulse
    adjoint = impulse_fft * np.conjugate(filter_full)
    del impulse_fft
    correction_fft = null_project_spectrum_preserve_dtype(
        adjoint, coarse.shape[0]
    )
    conditioned_fft = base_fft + correction_fft
    achieved = _response_at_points(
        conditioned_fft, filter_full, points, workers=workers
    )
    coarse_roundtrip = restrict_spectrum_preserve_dtype(
        conditioned_fft, coarse.shape[0]
    )
    correction_low = restrict_spectrum_preserve_dtype(
        correction_fft, coarse.shape[0]
    )
    roundtrip_error = _spectrum_relative_error(coarse_roundtrip, coarse_fft)
    correction_null_error = float(np.sqrt(np.mean(np.abs(correction_low) ** 2)))
    conditioned = spfft.ifftn(
        conditioned_fft, norm="ortho", workers=workers
    )
    real_rms = float(np.sqrt(np.mean(conditioned.real**2)))
    imaginary_relative_rms = float(
        np.sqrt(np.mean(conditioned.imag**2)) / max(real_rms, 1.0e-30)
    )
    tolerance = 2.0e-5 if np.dtype(float_dtype) == np.dtype(np.float32) else 1.0e-12
    if imaginary_relative_rms > tolerance:
        raise RuntimeError("conditional field broke Hermitian symmetry")
    field = np.asarray(conditioned.real, dtype=float_dtype)
    metadata = {
        "fine_seed": int(fine_seed),
        "noise_seed": int(noise_seed),
        "predicted_before": predicted.tolist(),
        "achieved_after": achieved.tolist(),
        "targets": targets.tolist(),
        "sigma": sig.tolist(),
        "mock_noise": mock_noise.tolist(),
        "weights": weights.tolist(),
        "coarse_roundtrip_relative_RMS": roundtrip_error,
        "correction_restriction_absolute_RMS": correction_null_error,
        "field_RMS": real_rms,
        "field_imaginary_relative_RMS": imaginary_relative_rms,
        "FFT_workers": int(workers),
        "peak_evidence_reapplied": False,
    }
    return field, metadata
