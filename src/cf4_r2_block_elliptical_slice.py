"""Exact prior-preserving block elliptical-slice mechanics for R2.

The transition only needs a scalar log likelihood; it does not use a gradient.
Projectors must be fixed, linear, orthogonal, and idempotent under the
standard-normal prior coordinates. A full target likelihood must be evaluated
at every proposed point.
"""

from dataclasses import dataclass

import numpy as np
from scipy import fft


class SliceEvaluationLimit(RuntimeError):
    """A bracket did not close within its predeclared evaluation budget."""

    def __init__(self, maximum_evaluations, attempts, bracket_contractions):
        super().__init__("elliptical-slice bracket evaluation budget exhausted")
        self.maximum_evaluations = int(maximum_evaluations)
        self.attempts = int(attempts)
        self.bracket_contractions = int(bracket_contractions)


@dataclass(frozen=True)
class SliceResult:
    position: np.ndarray
    log_likelihood: float
    angle: float
    evaluations: int
    bracket_contractions: int
    block_rms_displacement: float


def elliptical_slice_step(position, current_log_likelihood, log_likelihood,
                          project_block, rng, *, max_evaluations):
    """One exact ESS update in a fixed orthogonal Gaussian-prior subspace.

    ``project_block`` is an orthogonal projector in canonical N(0,I)
    coordinates. The likelihood callable must return the complete, unchanged
    R2 likelihood score (not posterior log density and not a partial factor).
    No input array is mutated. Budget exhaustion raises rather than returning
    a falsely completed transition.
    """
    q = np.asarray(position, dtype=np.float64)
    if (q.ndim != 1 or not np.isfinite(q).all()
            or not isinstance(max_evaluations, int) or max_evaluations < 1
            or not np.isfinite(current_log_likelihood)):
        raise ValueError("finite position/current likelihood and positive budget required")
    q0 = q.copy()
    current = float(current_log_likelihood)
    block = np.asarray(project_block(q0), dtype=np.float64)
    if block.shape != q0.shape or not np.isfinite(block).all():
        raise ValueError("block projector returned an invalid vector")
    block_norm = float(np.linalg.norm(block))
    if not np.isfinite(block_norm):
        raise ValueError("the selected Gaussian-prior block is nonfinite")
    block_dimension = getattr(project_block, "dimension", None)
    if block_dimension is None:
        block_dimension = int(np.count_nonzero(block))
    if not isinstance(block_dimension, (int, np.integer)) or block_dimension < 1:
        raise ValueError("the selected block must declare a positive dimension")

    draw_auxiliary = getattr(project_block, "draw_auxiliary", None)
    if draw_auxiliary is None:
        auxiliary = project_block(rng.standard_normal(q0.shape))
    else:
        auxiliary = draw_auxiliary(rng)
    auxiliary = np.asarray(auxiliary, dtype=np.float64)
    if auxiliary.shape != q0.shape or not np.isfinite(auxiliary).all():
        raise ValueError("projected Gaussian auxiliary is invalid")
    if float(np.linalg.norm(auxiliary)) <= 0.0:
        raise ValueError("the projected Gaussian auxiliary is empty")

    # Protect against the extremely rare exact zero from Generator.random().
    slice_threshold = current + np.log(max(float(rng.random()), np.finfo(float).tiny))
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    lower, upper = theta - 2.0 * np.pi, theta
    contractions = 0

    for attempt in range(1, max_evaluations + 1):
        proposal = q0 - block + block * np.cos(theta) + auxiliary * np.sin(theta)
        proposed_log_likelihood = float(log_likelihood(proposal))
        if np.isnan(proposed_log_likelihood) or proposed_log_likelihood == np.inf:
            raise FloatingPointError("invalid full-target likelihood in slice bracket")
        if proposed_log_likelihood >= slice_threshold:
            displacement = proposal - q0
            return SliceResult(
                position=proposal,
                log_likelihood=proposed_log_likelihood,
                angle=theta,
                evaluations=attempt,
                bracket_contractions=contractions,
                block_rms_displacement=float(
                    np.linalg.norm(displacement) / np.sqrt(block_dimension)),
            )

        # A true zero-likelihood point (-inf) is rejected normally. Exceptions,
        # NaNs, capacity failures, and unsupported states are never converted
        # into a zero likelihood.
        if theta < 0.0:
            lower = theta
        else:
            upper = theta
        theta = float(rng.uniform(lower, upper))
        contractions += 1

    raise SliceEvaluationLimit(max_evaluations, max_evaluations, contractions)


def real_fourier_band_projectors(n, cutoff, *, nuisance_count=24):
    """Return fixed radial low/high projectors for a real cubic IC field.

    Integer Fourier modes with |k| <= cutoff form the low block; all remaining
    modes form its orthogonal complement. The nuisance tail is zeroed by both
    projectors. ``cutoff`` is an engineering partition only, not a claim that
    observations contain no information above that scale.
    """
    if (not isinstance(n, int) or n < 2 or not np.isfinite(cutoff)
            or cutoff < 0.0 or not isinstance(nuisance_count, int)
            or nuisance_count < 0):
        raise ValueError("valid cubic grid, nonnegative cutoff and nuisance size required")

    kx = np.fft.fftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kz = np.fft.rfftfreq(n) * n
    radius2 = (kx[:, None, None] ** 2 + ky[None, :, None] ** 2
               + kz[None, None, :] ** 2)
    low_mask = radius2 <= float(cutoff) ** 2
    high_mask = ~low_mask
    n_ic = n ** 3
    full_size = n_ic + nuisance_count
    full_kx = np.fft.fftfreq(n) * n
    full_ky = np.fft.fftfreq(n) * n
    full_kz = np.fft.fftfreq(n) * n
    full_radius2 = (full_kx[:, None, None] ** 2 + full_ky[None, :, None] ** 2
                    + full_kz[None, None, :] ** 2)
    low_dimension = int(np.count_nonzero(full_radius2 <= float(cutoff) ** 2))
    if not 0 < low_dimension < n_ic:
        raise ValueError("cutoff must leave both Fourier blocks nonempty")

    def projector(mask, dimension, name):
        def apply(vector):
            q = np.asarray(vector, dtype=np.float64)
            if q.shape != (full_size,) or not np.isfinite(q).all():
                raise ValueError("canonical vector does not match the declared IC/nuisance shape")
            field = q[:n_ic].reshape((n, n, n))
            spectrum = fft.rfftn(field, norm="ortho", workers=1)
            projected = fft.irfftn(spectrum * mask, s=(n, n, n),
                                   norm="ortho", workers=1)
            result = np.zeros_like(q)
            result[:n_ic] = projected.reshape(-1)
            return result

        def draw_auxiliary(rng):
            white = rng.standard_normal((n, n, n))
            spectrum = fft.rfftn(white, norm="ortho", workers=1)
            projected = fft.irfftn(spectrum * mask, s=(n, n, n),
                                   norm="ortho", workers=1)
            result = np.zeros(full_size, dtype=np.float64)
            result[:n_ic] = projected.reshape(-1)
            return result

        apply.dimension = dimension
        apply.block_name = name
        apply.draw_auxiliary = draw_auxiliary
        return apply

    def nuisance_projector(vector):
        q = np.asarray(vector, dtype=np.float64)
        if q.shape != (full_size,) or not np.isfinite(q).all():
            raise ValueError("canonical vector does not match the declared IC/nuisance shape")
        result = np.zeros_like(q)
        if nuisance_count:
            result[n_ic:] = q[n_ic:]
        return result

    def draw_nuisance_auxiliary(rng):
        result = np.zeros(full_size, dtype=np.float64)
        if nuisance_count:
            result[n_ic:] = rng.standard_normal(nuisance_count)
        return result

    nuisance_projector.dimension = nuisance_count
    nuisance_projector.block_name = "nuisance"
    nuisance_projector.draw_auxiliary = draw_nuisance_auxiliary

    return (projector(low_mask, low_dimension, "ic_low_fourier"),
            projector(high_mask, n_ic - low_dimension, "ic_high_fourier"),
            nuisance_projector)
