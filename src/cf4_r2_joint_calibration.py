"""Shared calibration and selected radial shape, conditional on a supplied state.

No fitted-data prior is constructed here. The caller supplies a proper prior,
possibly correlated, and owns its scientific calibration. A fixed geometry
cache is useful for nuisance-only controls but cannot replace field dependence
in IC inference. Kappa describes selected intensity, NOT inclusion probability.
"""
import jax.numpy as jnp
from cf4_r2_fp_group_marginal import (conditional_group_scores,
                                     nonfp_modulus_logmarks)


def group_scores(parameters, data):
    """Parameters: FP eta zero [dex], method zeros [mag], radial tilt kappa.

    Method offsets have sign mu_observed ~ mu_predicted + offset; FP uses
    eta_predicted + zero. All groups see the SAME parameter vector. The
    distance-independent intensity amplitude is deliberately not a parameter.
    """
    d = data['distance']
    logw = data['log_distance_weight'] + parameters[-1]*jnp.log(d/100.)
    marks = nonfp_modulus_logmarks(data['predicted_modulus'], data['anchor_group'],
        data['anchor_modulus'], data['anchor_error'], data['anchor_method'],
        parameters[1:-1])
    return conditional_group_scores(d, logw, data['redshift_logkernel'],
        data['row_group'], data['dz_row'], data['eta_mean'], data['eta_std'],
        data['eta_alpha'], parameters[:1], extra_log_marks=marks)[0]


def white_logdensity(white, data, prior_mean, prior_cholesky):
    """Training-only target in white prior coordinates; one joint prior.

    Constant Jacobian/normalizers cancel in sampling. Cholesky must be a
    nonsingular lower-triangular factor, checked by the caller. This interface
    represents zero-point covariance, not unmodelled source shape covariance.
    """
    parameters = prior_mean + prior_cholesky @ white
    scores = group_scores(parameters, data)
    return jnp.where(data['group_holdout'], 0., scores).sum() - .5*jnp.dot(white, white)
