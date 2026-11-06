"""Candidate FP response conditional on observed magnitude/dispersion coordinates.

X=(r_z,s,i), U=(2*r_z+i, .04*r_z+s), response T=r_z. At fixed
redshift/aperture metadata U is affine in observed magnitude/raw dispersion;
the transformation (U,T) has unit absolute determinant. This is a Gaussian
component primitive, NOT a source-selection model or calibrated field target.
Condition on observed (not error-free latent) coordinates with their full
covariance. Additional response/latent-dependent selection does not cancel.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


CONDITION_MATRIX = np.array([[2., 0., 1.], [.04, 1., 0.]])
_TRANSFORM = np.vstack((CONDITION_MATRIX, [1., 0., 0.]))


def component_terms(mean, covariance, observed_u, *, mean_direction=(1., 0., 0.)):
    """Gaussian T|U and U density, allowing broadcast batch/component axes.

mean_direction is d(mean_X)/d(eta) holding covariance fixed. The default is
the radius-only shift at fixed population/luminosity, not the derivative of
a complete field/distance mixture. A shared zero point in this same direction
is exactly confounded with eta here. The Fisher terms hold population
parameters fixed; they are not marginal distance constraints.
"""
    mean, cov, u, direction = map(lambda a: np.asarray(a, dtype=float),
                                 (mean, covariance, observed_u, mean_direction))
    if (mean.shape[-1:] != (3,) or cov.shape[-2:] != (3, 3)
            or u.shape[-1:] != (2,) or direction.shape[-1:] != (3,)):
        raise ValueError("expected mean(3), covariance(3,3), U(2), direction(3)")
    if not all(np.isfinite(a).all() for a in (mean, cov, u, direction)):
        raise ValueError("finite component inputs required")
    if not np.allclose(cov, np.swapaxes(cov, -1, -2), atol=1e-14, rtol=1e-12):
        raise ValueError("covariance must be symmetric")
    shape = np.broadcast_shapes(mean.shape[:-1], cov.shape[:-2],
                                u.shape[:-1], direction.shape[:-1])
    mean, cov, u, direction = (np.broadcast_to(a, shape + tail) for a, tail in
                               ((mean, (3,)), (cov, (3, 3)), (u, (2,)),
                                (direction, (3,))))
    try:
        chol_x = np.linalg.cholesky(cov)
        transformed = _TRANSFORM @ cov @ _TRANSFORM.T
        chol = np.linalg.cholesky(transformed)
    except np.linalg.LinAlgError as exc:
        raise ValueError("positive-definite covariance required; no floor") from exc
    cross = transformed[..., :2, 2]
    regression = np.linalg.solve(transformed[..., :2, :2], cross[..., None])[..., 0]
    displacement = u - mean @ CONDITION_MATRIX.T
    conditional_mean = mean[..., 0] + np.sum(regression * displacement, axis=-1)
    conditional_variance = chol[..., 2, 2] ** 2
    standard_u = np.linalg.solve(chol[..., :2, :2], displacement[..., None])[..., 0]
    log_u = (-np.log(2 * np.pi) - np.log(chol[..., 0, 0]) - np.log(chol[..., 1, 1])
             - .5 * np.sum(standard_u ** 2, axis=-1))
    response = direction[..., 0] - np.sum(
        regression * (direction @ CONDITION_MATRIX.T), axis=-1)
    white_direction = np.linalg.solve(chol_x, direction[..., None])[..., 0]
    return dict(mean=conditional_mean, variance=conditional_variance,
                log_condition_density=log_u, mean_response=response,
                conditional_fisher=response ** 2 / conditional_variance,
                joint_fisher=np.sum(white_direction ** 2, axis=-1))


def conditioned_log_weights(log_weights, log_condition_density):
    """Reweight latent nodes by p(U|node), not just their previous weights.

Any other fixed conditioning datum (e.g. K) must already have its correctly
owned factor in the supplied weights. These are not automatically 2M++
weights for a source-native FP-only object.
"""
    lw, lu = map(lambda a: np.asarray(a, dtype=float),
                 (log_weights, log_condition_density))
    if lw.ndim != 1 or lw.size == 0 or lw.shape != lu.shape:
        raise ValueError("aligned nonempty one-dimensional component arrays required")
    if np.isnan(lw).any() or np.isposinf(lw).any() or not np.isfinite(lu).all():
        raise ValueError("invalid component weights or covariate density")
    joint = lw + lu
    normalization = logsumexp(joint)
    if not np.isfinite(normalization):
        raise ValueError("at least one supported component required")
    return joint - normalization


def mixture_logpdf(response, terms, log_weights):
    """Evaluate a one-object conditional mixture; no unknown selection correction."""
    mean, variance = np.asarray(terms['mean']), np.asarray(terms['variance'])
    if (mean.ndim != 1 or variance.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(variance).all()
            or np.any(variance <= 0)):
        raise ValueError("finite one-dimensional component means/positive variances required")
    y = np.asarray(response, dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("finite response required")
    weights = conditioned_log_weights(log_weights, terms['log_condition_density'])
    if weights.shape != mean.shape:
        raise ValueError("weights must match component moments")
    logpdf = -.5 * (np.log(2 * np.pi * variance)
                    + (y[..., None] - mean) ** 2 / variance)
    return logsumexp(weights + logpdf, axis=-1)
