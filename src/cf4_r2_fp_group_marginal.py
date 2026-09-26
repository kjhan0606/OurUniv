"""Conditional distance marks with shared group distance and global zero point.

The caller owns the joint redshift law and the selected-group distance prior.
This module conditions on that redshift vector; it never exports its marginal
as another independent redshift likelihood. Covariance inputs need physical
calibration before production. Catalogue identities label observations only.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np

from cf4_r2_fp_distance import fp_log_likelihood_ratio


def redshift_sufficient(observed_cz, covariance_v, redshift_ids):
    """Compress a supplied correlated Gaussian with common model mean.

    covariance_v is in peculiar-(km/s)^2. The model's observed-cz covariance
    is (1+zcos)^2 covariance_v. Duplicate datum IDs are forbidden. A catalogue
    group mean and member measurements may be distinct but correlated rows;
    the supplied covariance must describe that correlation, not ignore it.
    """
    y, c = np.asarray(observed_cz, float), np.asarray(covariance_v, float)
    if y.ndim != 1 or not y.size or c.shape != (y.size, y.size):
        raise ValueError('invalid redshift vector/covariance shape')
    if len(redshift_ids) != y.size or len(set(redshift_ids)) != y.size:
        raise ValueError('duplicate or mismatched redshift datum IDs')
    if not np.isfinite(y).all() or not np.isfinite(c).all() or not np.allclose(c, c.T):
        raise ValueError('nonfinite or asymmetric covariance')
    chol = np.linalg.cholesky(c)
    one_white = np.linalg.solve(chol, np.ones_like(y))
    y_white = np.linalg.solve(chol, y)
    precision = one_white @ one_white
    centre = (one_white @ y_white) / precision
    residual = y_white - centre * one_white
    return np.array([precision, centre, residual @ residual,
                     2*np.log(np.diag(chol)).sum(), y.size])


def joint_redshift_logkernel(predicted_cz, zcos, sufficient):
    """Full joint-redshift kernel at each distance, including its Jacobian."""
    p, centre, q, logdet, n = [sufficient[..., k, None] for k in range(5)]
    scale = 1 + zcos
    return -.5*((q+p*(centre-predicted_cz)**2)/scale**2 + logdet
                + 2*n*jnp.log(scale) + n*jnp.log(2*jnp.pi))


def conditional_group_scores(distance, log_distance_weight, redshift_logkernel,
                             row_group, dz_row, mean, std, alpha, zero_nodes):
    """Return log conditional mark factor per zero-point node and group.

    distance/weights/kernel are (groups, quadrature). weights include dd and
    the explicit selected-group prior, not another FP measurement correction.
    Product of member distance likelihoods is INSIDE the group integral.
    The fixed reference eta=0 only cancels source-data constants.
    """
    groups = distance.shape[0]
    base = log_distance_weight + redshift_logkernel
    denominator = logsumexp(base, axis=-1)
    eta = jnp.log10(dz_row[:, None] / distance[row_group])
    def at_zero(b):
        member = fp_log_likelihood_ratio(eta+b, 0., mean[:, None],
                                        std[:, None], alpha[:, None])
        marks = jax.ops.segment_sum(member, row_group, num_segments=groups)
        return logsumexp(base + marks, axis=-1) - denominator
    return jax.lax.map(at_zero, zero_nodes)


def shared_zero_logfactor(scores, log_zero_weights, group_mask):
    """Integrate ONE global zero point after multiplying all selected groups."""
    logw = log_zero_weights - logsumexp(log_zero_weights)
    return logsumexp(logw + jnp.sum(jnp.where(group_mask[None, :], scores, 0.), axis=1))
