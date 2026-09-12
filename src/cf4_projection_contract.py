#!/usr/bin/env python3
"""Future-only variance-preserving Fourier restriction/prolongation pair.

For an even output mesh, each output-Nyquist component identifies the two
source frequencies +/-Nout/2.  Restriction therefore folds a class of
``2**r`` source modes with ``1/sqrt(2**r)`` normalization, where ``r`` is the
number of output-Nyquist components.  Its adjoint distributes a target
coefficient over the same class with the same normalization.

The prolongation is the Gaussian conditional draw

    x = z + R* (y - R z),

where ``z`` is a fresh fine white field, ``y`` is the supplied coarse field,
and ``R R* = I``.  Thus ``R x = y`` including every even-grid Nyquist mode,
while the null-space modes retain their independent standard-normal prior.
This module is not used by the immutable V8 products.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _validate_mesh(source_n: int, output_n: int) -> None:
    if source_n <= 0 or output_n <= 0 or source_n % 2 or output_n % 2:
        raise ValueError("source_n and output_n must be positive even integers")
    if output_n > source_n:
        raise ValueError("output_n cannot exceed source_n")


def _equivalence_indices(
    source_n: int,
    output_n: int,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Return the two separable representatives and the adjoint denominator."""
    _validate_mesh(source_n, output_n)
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
    # Eight separable selections repeat each distinct class member 2**(3-r)
    # times.  This denominator leaves each of the 2**r distinct members with
    # coefficient 1/sqrt(2**r).
    denominator = np.power(2.0, 3.0 - 0.5 * multiplicity_power)
    return (base, alternate), denominator


def restrict_spectrum(source_fft: np.ndarray, output_n: int) -> np.ndarray:
    """Apply the unitary full-spectrum restriction R."""
    source_fft = np.asarray(source_fft)
    if source_fft.ndim != 3 or not (
        source_fft.shape[0] == source_fft.shape[1] == source_fft.shape[2]
    ):
        raise ValueError("source_fft must be a cubic full complex spectrum")
    source_n = source_fft.shape[0]
    choices, denominator = _equivalence_indices(source_n, output_n)
    restricted = np.zeros((output_n, output_n, output_n), dtype=np.complex128)
    for choose_x in range(2):
        for choose_y in range(2):
            for choose_z in range(2):
                restricted += source_fft[np.ix_(
                    choices[choose_x], choices[choose_y], choices[choose_z]
                )]
    restricted /= denominator
    return restricted


def add_restriction_adjoint(
    source_fft: np.ndarray,
    target_fft: np.ndarray,
) -> np.ndarray:
    """Return ``source_fft + R* target_fft`` in unitary full-FFT coordinates."""
    source_fft = np.asarray(source_fft)
    target_fft = np.asarray(target_fft)
    if source_fft.ndim != 3 or not (
        source_fft.shape[0] == source_fft.shape[1] == source_fft.shape[2]
    ):
        raise ValueError("source_fft must be a cubic full complex spectrum")
    if target_fft.ndim != 3 or not (
        target_fft.shape[0] == target_fft.shape[1] == target_fft.shape[2]
    ):
        raise ValueError("target_fft must be a cubic full complex spectrum")
    source_n = source_fft.shape[0]
    output_n = target_fft.shape[0]
    choices, denominator = _equivalence_indices(source_n, output_n)
    result = source_fft.astype(np.complex128, copy=True)
    contribution = target_fft / denominator
    for choose_x in range(2):
        for choose_y in range(2):
            for choose_z in range(2):
                result[np.ix_(
                    choices[choose_x], choices[choose_y], choices[choose_z]
                )] += contribution
    return result


def restriction_adjoint_spectrum(target_fft: np.ndarray, source_n: int) -> np.ndarray:
    """Apply R* to a target full spectrum."""
    target_fft = np.asarray(target_fft)
    zero = np.zeros((source_n, source_n, source_n), dtype=np.complex128)
    return add_restriction_adjoint(zero, target_fft)


def restrict_white_field(source: np.ndarray, output_n: int) -> np.ndarray:
    """Restrict a real field with variance-preserving output-Nyquist folds."""
    source = np.asarray(source)
    if source.ndim != 3 or not (
        source.shape[0] == source.shape[1] == source.shape[2]
    ):
        raise ValueError("source must be a cubic real field")
    source_fft = np.fft.fftn(source.astype(np.float64, copy=False), norm="ortho")
    output_fft = restrict_spectrum(source_fft, output_n)
    output = np.fft.ifftn(output_fft, norm="ortho")
    imaginary_relative_rms = float(
        np.sqrt(np.mean(output.imag ** 2))
        / max(np.sqrt(np.mean(output.real ** 2)), np.finfo(np.float64).tiny)
    )
    if imaginary_relative_rms > 1e-12:
        raise RuntimeError(
            f"restriction broke Hermitian symmetry: {imaginary_relative_rms:g}"
        )
    return output.real.astype(np.float32)


def prolong_white_field(
    coarse: np.ndarray,
    fine_n: int,
    seed: int,
) -> np.ndarray:
    """Draw a fine white field conditional on exact restriction to ``coarse``."""
    coarse = np.asarray(coarse)
    if coarse.ndim != 3 or not (
        coarse.shape[0] == coarse.shape[1] == coarse.shape[2]
    ):
        raise ValueError("coarse must be a cubic real field")
    coarse_n = coarse.shape[0]
    _validate_mesh(fine_n, coarse_n)
    rng = np.random.Generator(np.random.PCG64DXSM(int(seed)))
    base = rng.standard_normal((fine_n, fine_n, fine_n))
    base_fft = np.fft.fftn(base, norm="ortho")
    coarse_fft = np.fft.fftn(
        coarse.astype(np.float64, copy=False), norm="ortho"
    )
    residual = coarse_fft - restrict_spectrum(base_fft, coarse_n)
    conditioned_fft = add_restriction_adjoint(base_fft, residual)
    conditioned = np.fft.ifftn(conditioned_fft, norm="ortho")
    imaginary_relative_rms = float(
        np.sqrt(np.mean(conditioned.imag ** 2))
        / max(
            np.sqrt(np.mean(conditioned.real ** 2)),
            np.finfo(np.float64).tiny,
        )
    )
    if imaginary_relative_rms > 1e-12:
        raise RuntimeError(
            f"prolongation broke Hermitian symmetry: {imaginary_relative_rms:g}"
        )
    return conditioned.real.astype(np.float32)


def normalized_errors(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    difference = np.asarray(left) - np.asarray(right)
    scale = float(np.sqrt(np.mean(np.abs(np.asarray(right)) ** 2)))
    return {
        "relative_RMS": float(np.sqrt(np.mean(np.abs(difference) ** 2)) / scale),
        "maximum_normalized_error": float(np.max(np.abs(difference)) / scale),
    }


def white_moments(field: np.ndarray) -> dict[str, float]:
    value = np.asarray(field, dtype=np.float64)
    centered = value - np.mean(value)
    variance = float(np.mean(centered ** 2))
    standard_deviation = math.sqrt(variance)
    return {
        "mean": float(np.mean(value)),
        "std": standard_deviation,
        "skew": float(np.mean(centered ** 3) / standard_deviation ** 3),
        "excess_kurtosis": float(
            np.mean(centered ** 4) / standard_deviation ** 4 - 3.0
        ),
    }


def contract_metadata() -> dict[str, Any]:
    return {
        "FFT_normalization": "ortho",
        "output_Nyquist_class_normalization": "1/sqrt(2**r)",
        "prolongation": "z + R* (y - R z)",
        "identities": ["R R* = I", "R prolong(y,z) = y"],
        "rng": "NumPy Generator PCG64DXSM",
        "legacy_V8_products_modified": False,
    }
