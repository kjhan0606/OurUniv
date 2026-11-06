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


def grouped_log_integrals(log_response, log_conditions, row_group, log_source_weights):
    """Integrate member PRODUCTS over shared group-local latent nodes.

    Row densities have shape (shared-nuisance, row, local-node); source weights
    have shape (shared-nuisance, group, local-node). All members of a group
    share a local node (e.g. distance AND group offset). Conditional on those
    nodes, row measurement errors must be independent. Correlations remaining
    at fixed nodes require a joint row density, not this product interface.

    Source weights own the SAME state's source/redshift/other already observed
    data exactly once. Normalize them within each group and shared-nuisance
    value before any global integration: their arbitrary amplitudes must not
    change the shared-nuisance prior. FP-only weights are NOT count weights.
    Returns log p(T_g,U_g|b) and log p(U_g|b), not normalized mark scores.
    No availability, morphology, outlier or association law is supplied here.
    """
    lt, lu, weights = (np.asarray(x, dtype=float) for x in
                       (log_response, log_conditions, log_source_weights))
    groups = np.asarray(row_group)
    if (lt.ndim != 3 or lt.shape != lu.shape or weights.ndim != 3
            or any(n == 0 for n in lt.shape + weights.shape)
            or weights.shape[0] != lt.shape[0] or weights.shape[2] != lt.shape[2]
            or groups.shape != (lt.shape[1],)
            or not np.issubdtype(groups.dtype, np.integer)):
        raise ValueError('aligned nonempty shared/row/group/node arrays required')
    ngroup = weights.shape[1]
    if not np.array_equal(np.unique(groups), np.arange(ngroup)):
        raise ValueError('every group must have members, indexed 0 through G-1')
    if any(np.isnan(x).any() or np.isposinf(x).any() for x in (lt, lu, weights)):
        raise ValueError('log densities/weights may be finite or -inf only')
    norm = logsumexp(weights, axis=-1, keepdims=True)
    if not np.isfinite(norm).all():
        raise ValueError('source weights need support for every group/nuisance node')
    base = weights - norm
    sums_u = np.zeros_like(base)
    sums_joint = np.zeros_like(base)
    # Segment accumulation handles non-contiguous member order without padding
    # fictitious galaxies, or integrating each galaxy's group distance alone.
    for shared in range(lt.shape[0]):
        np.add.at(sums_u[shared], groups, lu[shared])
        np.add.at(sums_joint[shared], groups, lu[shared] + lt[shared])
    return dict(log_joint=logsumexp(base + sums_joint, axis=-1),
                log_conditions=logsumexp(base + sums_u, axis=-1))


def shared_conditional_logpdf(group_integrals, log_shared_weights):
    """Condition AFTER integrating one common nuisance across all groups.

    p(T|U) = sum_b q_b prod_g p(T_g,U_g|b) /
             sum_b q_b prod_g p(U_g|b).
    Averaging prod_g p(T_g|U_g,b) with the OLD q_b is generally wrong. The
    correct weights include ALL groups' covariate evidence. Return that
    evidence separately so a caller cannot silently lose or count it twice.
    A field posterior also needs an explicit decision about p(F|U) versus a
    fixed-design conditional model; this kernel does not make them equivalent.
    q_b must be calibrated externally or jointly, not fitted-data reused as a
    prior. A global shared nuisance cannot be marginalized independently in
    separate chunks/groups. This is a numerical component, not calibration.
    """
    lj, lu, lw = (np.asarray(x, dtype=float) for x in
                  (group_integrals['log_joint'], group_integrals['log_conditions'],
                   log_shared_weights))
    if (lj.ndim != 2 or lj.shape != lu.shape or not all(lj.shape)
            or lw.shape != (lj.shape[0],)):
        raise ValueError('aligned nonempty shared/group integrals and weights required')
    if any(np.isnan(x).any() or np.isposinf(x).any() for x in (lj, lu, lw)):
        raise ValueError('log integrals/weights may be finite or -inf only')
    if np.any(np.isneginf(lu) & np.isfinite(lj)):
        raise ValueError('joint support cannot exceed condition support')
    norm = logsumexp(lw)
    if not np.isfinite(norm):
        raise ValueError('at least one supported shared node required')
    lw = lw - norm
    conditioned = lw + lu.sum(axis=1)
    denominator = logsumexp(conditioned)
    if not np.isfinite(denominator):
        raise ValueError('observed conditions have no joint support')
    numerator = logsumexp(lw + lj.sum(axis=1))
    return dict(logpdf=numerator-denominator, log_joint=numerator,
                log_conditions=denominator,
                log_shared_given_conditions=conditioned-denominator)
