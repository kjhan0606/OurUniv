"""Small exact-correction helpers for the R2 delayed-acceptance screen."""

from __future__ import annotations

import numpy as np
from scipy import fft as spfft

from cf4_lg_highk_conditional_field import (
    add_restriction_adjoint_preserve_dtype,
    restrict_spectrum_preserve_dtype,
)


def _real_cubic(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (array.ndim != 3 or len(set(array.shape)) != 1
            or not np.isfinite(array).all()):
        raise ValueError(f"{name} must be a finite real cubic field")
    return array


def restrict_white_field(field: np.ndarray, output_n: int) -> np.ndarray:
    """Apply the project's unitary real-field Fourier restriction ``R``."""
    source = _real_cubic(field, name="fine field")
    if not isinstance(output_n, (int, np.integer)) or output_n <= 0 or output_n % 2:
        raise ValueError("output mesh must be a positive even integer")
    spectrum = spfft.fftn(source, norm="ortho", workers=1)
    restricted = restrict_spectrum_preserve_dtype(spectrum, int(output_n))
    result = spfft.ifftn(restricted, norm="ortho", workers=1)
    real_scale = max(float(np.sqrt(np.mean(result.real * result.real))),
                     np.finfo(np.float64).tiny)
    imag_relative = float(np.sqrt(np.mean(result.imag * result.imag)) / real_scale)
    if imag_relative > 2e-12:
        raise FloatingPointError("Fourier restriction did not preserve a real field")
    return np.asarray(result.real, dtype=np.float64)


def adjoint_restriction_white_field(coarse_field: np.ndarray, fine_n: int) -> np.ndarray:
    """Apply the real-field adjoint ``R*`` from a coarse white field."""
    coarse = _real_cubic(coarse_field, name="coarse field")
    if not isinstance(fine_n, (int, np.integer)) or fine_n <= coarse.shape[0]:
        raise ValueError("fine mesh must be larger than the coarse mesh")
    coarse_spectrum = spfft.fftn(coarse, norm="ortho", workers=1)
    fine_spectrum = add_restriction_adjoint_preserve_dtype(
        np.zeros((int(fine_n),) * 3, dtype=np.complex128), coarse_spectrum
    )
    result = spfft.ifftn(fine_spectrum, norm="ortho", workers=1)
    scale = max(float(np.sqrt(np.mean(result.real * result.real))),
                np.finfo(np.float64).tiny)
    imag_relative = float(np.sqrt(np.mean(result.imag * result.imag)) / scale)
    if imag_relative > 2e-12:
        raise FloatingPointError("Fourier adjoint did not preserve a real field")
    return np.asarray(result.real, dtype=np.float64)


def lift_restricted_white_field(
    fine_field: np.ndarray,
    coarse_field_new: np.ndarray,
    *,
    tolerance: float = 2e-11,
) -> tuple[np.ndarray, dict[str, float]]:
    """Change only ``R q`` using ``q' = q + R*(z' - Rq)``.

    The orthogonal complement is preserved. The returned errors check the
    coisometry and complement identities in real canonical coordinates.
    """
    fine = _real_cubic(fine_field, name="fine field")
    coarse = _real_cubic(coarse_field_new, name="new coarse field")
    if coarse.shape[0] >= fine.shape[0]:
        raise ValueError("coarse mesh must be smaller than fine mesh")
    current_coarse = restrict_white_field(fine, coarse.shape[0])
    delta = coarse - current_coarse
    fine_spectrum = spfft.fftn(fine, norm="ortho", workers=1)
    delta_spectrum = spfft.fftn(delta, norm="ortho", workers=1)
    lifted_spectrum = add_restriction_adjoint_preserve_dtype(
        fine_spectrum, delta_spectrum
    )
    lifted_complex = spfft.ifftn(lifted_spectrum, norm="ortho", workers=1)
    fine_scale = max(float(np.sqrt(np.mean(lifted_complex.real ** 2))),
                     np.finfo(np.float64).tiny)
    imag_relative = float(np.sqrt(np.mean(lifted_complex.imag ** 2)) / fine_scale)
    if imag_relative > tolerance:
        raise FloatingPointError("adjoint lift did not preserve a real field")
    lifted = np.asarray(lifted_complex.real, dtype=np.float64)
    reproduced = restrict_white_field(lifted, coarse.shape[0])
    low_error = float(np.sqrt(np.mean((reproduced - coarse) ** 2)))
    low_scale = max(float(np.sqrt(np.mean(coarse * coarse))), 1.0)
    complement_before = fine - adjoint_restriction_white_field(current_coarse, fine.shape[0])
    complement_after = lifted - adjoint_restriction_white_field(reproduced, fine.shape[0])
    complement_error = float(np.sqrt(np.mean((complement_after - complement_before) ** 2)))
    fine_norm_change = float(np.sum(lifted * lifted) - np.sum(fine * fine))
    coarse_norm_change = float(np.sum(coarse * coarse) - np.sum(current_coarse * current_coarse))
    norm_error = abs(fine_norm_change - coarse_norm_change) / max(
        abs(fine_norm_change), abs(coarse_norm_change), 1.0
    )
    diagnostics = dict(
        restricted_reconstruction_relative_rms=low_error / low_scale,
        complement_preservation_rms=complement_error,
        prior_norm_identity_relative_error=float(norm_error),
        imaginary_relative_rms=imag_relative,
    )
    if (diagnostics["restricted_reconstruction_relative_rms"] > tolerance
            or complement_error > tolerance * fine_scale
            or norm_error > 5e-10):
        raise FloatingPointError("restricted-mode lift failed its canonical-coordinate identities")
    return lifted, diagnostics


def delayed_acceptance_stage2_log_alpha(
    exact_delta: float,
    surrogate_delta: float,
) -> float:
    """Log stage-two MH probability for an already stage-one-accepted move."""
    exact_delta = float(exact_delta)
    surrogate_delta = float(surrogate_delta)
    if np.isnan(exact_delta) or not np.isfinite(surrogate_delta):
        raise FloatingPointError("invalid exact or surrogate potential difference")
    if exact_delta == np.inf:
        return -np.inf
    if exact_delta == -np.inf:
        return 0.0
    if not np.isfinite(exact_delta):
        raise FloatingPointError("unsupported exact potential difference")
    return float(min(0.0, -(exact_delta - surrogate_delta)))


def delayed_acceptance_stage2_accept(
    exact_delta: float,
    surrogate_delta: float,
    log_uniform: float,
) -> tuple[bool, float]:
    """Apply stage two and return ``(accepted, log_alpha)``."""
    log_uniform = float(log_uniform)
    if np.isnan(log_uniform) or log_uniform > 0.0:
        raise ValueError("log uniform variate must lie in [-inf, 0]")
    log_alpha = delayed_acceptance_stage2_log_alpha(exact_delta, surrogate_delta)
    return bool(log_uniform < log_alpha), log_alpha
