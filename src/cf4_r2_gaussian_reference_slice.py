"""Fixed-Gaussian-reference ESS mechanics; not an R2 production sampler.

The reference transform must be one fixed, full-rank linear map for the
entire chain: ``color(x) = L @ x`` and ``whiten(color(x)) = x``.  In whitened
coordinates the reference is standard normal, so the exact slice score is
``log_target(q) + 0.5 * dot(x, x)``.  The omitted Gaussian normalization and
constant Jacobian cancel from every slice comparison.  The implementation
centers that score at the current state and evaluates the norm difference as
``(x_new-x) dot (x_new+x) / 2``.

This module provides only the target-preserving transition primitive.  It
does not construct or validate an R2 covariance, basis, likelihood, or cost
budget, and it must not be read as approval for an R2 inference run.
"""

from dataclasses import dataclass

import numpy as np

from cf4_r2_block_elliptical_slice import (
    SliceEvaluationLimit,
    elliptical_slice_step,
)


@dataclass(frozen=True)
class GaussianReferenceSliceResult:
    position: np.ndarray
    log_target: float
    log_residual_change: float
    angle: float
    evaluations: int
    bracket_contractions: int
    whitened_rms_displacement: float


def gaussian_reference_elliptical_slice_step(
    position,
    log_target,
    reference_mean,
    whiten,
    color,
    rng,
    *,
    max_evaluations,
):
    """Perform one full-space ESS update under a fixed Gaussian reference.

    ``whiten`` applies the fixed inverse square root ``L^-1`` to a centered
    physical-coordinate vector, and ``color`` applies its fixed inverse ``L``.
    Both must be linear, mutually inverse, deterministic, full-rank operators
    that do not adapt to the current state.  ``log_target`` must evaluate the
    complete unchanged target, never a proposal approximation or partial
    likelihood.

    The full target is preserved by slicing on ``log_target(q) - log g(q)``.
    The implementation omits only constants in ``log g``; its state-varying
    term is exactly ``-0.5 * ||whiten(q - mean)||^2``. Slice heights are
    centered on the current state to avoid subtracting large reference norms.
    A numerical approximation to the target remains the target being
    preserved; this function does not make it scientifically calibrated.
    """
    q = np.asarray(position, dtype=np.float64)
    mean = np.asarray(reference_mean, dtype=np.float64)
    if (q.ndim != 1 or mean.shape != q.shape
            or not np.isfinite(q).all() or not np.isfinite(mean).all()
            or not callable(log_target) or not callable(whiten)
            or not callable(color)):
        raise ValueError("finite position/mean and fixed transform callables required")

    x = np.asarray(whiten(q - mean), dtype=np.float64)
    if x.shape != q.shape or not np.isfinite(x).all():
        raise ValueError("whitened position must be finite and match the target dimension")
    current_target = float(log_target(q.copy()))
    if not np.isfinite(current_target):
        raise ValueError("current state must have finite complete-target log density")
    current_white_norm2 = float(x @ x)
    if not np.isfinite(current_white_norm2):
        raise FloatingPointError("current whitened norm is nonfinite")

    last_target_value = None
    last_physical = None
    last_white = None

    def centered_residual_at(white):
        nonlocal last_target_value, last_physical, last_white
        white = np.asarray(white, dtype=np.float64)
        if white.shape != q.shape or not np.isfinite(white).all():
            raise ValueError("slice proposal is not a finite canonical vector")
        physical = mean + np.asarray(color(white), dtype=np.float64)
        if physical.shape != q.shape or not np.isfinite(physical).all():
            raise ValueError("colored proposal must be finite and match the target dimension")
        target_value = float(log_target(physical))
        if np.isnan(target_value) or target_value == np.inf:
            raise FloatingPointError("invalid complete-target log density in slice bracket")
        # Center at the current state and use a difference of squares.  This
        # avoids adding/subtracting the large O(d) Gaussian-reference terms
        # merely to compare two slice heights.
        white_delta = white - x
        white_sum = white + x
        residual_change = ((target_value - current_target)
                           + 0.5 * float(white_delta @ white_sum))
        if np.isnan(residual_change) or residual_change == np.inf:
            raise FloatingPointError("invalid Gaussian-reference residual in slice bracket")
        last_target_value = target_value
        last_physical = physical
        last_white = white
        return residual_change

    def identity_projector(vector):
        return np.asarray(vector, dtype=np.float64).copy()

    identity_projector.dimension = len(q)
    identity_projector.draw_auxiliary = lambda generator: generator.standard_normal(q.shape)

    update = elliptical_slice_step(
        x, 0.0, centered_residual_at, identity_projector, rng,
        max_evaluations=max_evaluations,
    )
    white_new = update.position
    if (last_target_value is None or last_physical is None or last_white is None
            or not np.array_equal(last_white, white_new)):
        raise AssertionError("accepted state does not match the last full-target evaluation")
    q_new = last_physical
    if q_new.shape != q.shape or not np.isfinite(q_new).all():
        raise FloatingPointError("accepted Gaussian-reference state is invalid")
    log_target_new = float(last_target_value)
    if not np.isfinite(log_target_new):
        raise FloatingPointError("accepted complete-target log density is nonfinite")

    return GaussianReferenceSliceResult(
        position=q_new,
        log_target=float(log_target_new),
        log_residual_change=float(update.log_likelihood),
        angle=update.angle,
        evaluations=update.evaluations,
        bracket_contractions=update.bracket_contractions,
        whitened_rms_displacement=update.block_rms_displacement,
    )


__all__ = [
    "GaussianReferenceSliceResult",
    "SliceEvaluationLimit",
    "gaussian_reference_elliptical_slice_step",
]
