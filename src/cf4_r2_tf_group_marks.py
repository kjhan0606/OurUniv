"""Conditional corrected-CF4 TF group-modulus development factor.

The observed group redshift is conditioned in numerator and denominator.
This does not model TF group inclusion or source-calibration covariance.
"""

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from cf4_r2_fp_group_marginal import selected_group_logweights
from cf4_z0_physical_field import read_centred


def tf_group_logratios(rho, velocity, directions, observed_cz, modulus, modulus_error,
                       distance, quadrature_weight, zcos, *, box=384., h=.746,
                       selected_bias=1., sigma_v=150., modulus_zero=0.,
                       reference_modulus=35.):
    """One conditional TF distance-mark likelihood ratio per disjoint group.

    `distance`, `quadrature_weight`, and `zcos` are shared 1D integration
    nodes. The corrected CF4 group `DMtf/e_DMtf` is approximated as ONE
    Gaussian group summary; neither its raw member distances nor FP marks are
    multiplied again. `sigma_v` and the selected-density exponent are
    explicitly provisional, not source calibrations.
    """
    groups = directions.shape[0]
    d = jnp.broadcast_to(distance[None, :], (groups, distance.shape[0]))
    z = jnp.broadcast_to(zcos[None, :], d.shape)
    q = jnp.broadcast_to(quadrature_weight[None, :], d.shape)
    position = box/2 + directions[:, None, :] * d[:, :, None]
    density = read_centred(rho, position, box, 0.)
    radial = sum(read_centred(velocity[k], position, box, 0.) * directions[:, k, None]
                 for k in range(3))
    selected_bias = jnp.asarray(selected_bias)
    sigma_v = jnp.asarray(sigma_v)
    if selected_bias.ndim == 1:
        selected_bias = selected_bias[:, None]
    if sigma_v.ndim == 1:
        sigma_v = sigma_v[:, None]
    log_measure = selected_group_logweights(d, q, density, selected_bias,
                                            jnp.zeros_like(d))
    sigma_cz = sigma_v * (1. + z)
    predicted_cz = 299792.458*z + (1.+z)*radial
    redshift = (-.5*((observed_cz[:, None]-predicted_cz)/sigma_cz)**2
                - jnp.log(sigma_cz) - .5*jnp.log(2*jnp.pi))
    base = log_measure + redshift
    predicted_modulus = 5*jnp.log10((1.+z)*d/h)+25.
    sigma_dm = modulus_error[:, None]
    mark = -.5*((modulus[:, None]-predicted_modulus-modulus_zero)/sigma_dm)**2
    reference = -.5*((modulus-reference_modulus)/modulus_error)**2
    sigma_valid = sigma_v[:, 0] if sigma_v.ndim == 2 else sigma_v
    valid = ((modulus_error > 0) & jnp.isfinite(modulus_error)
             & jnp.isfinite(modulus) & (sigma_valid > 0) & jnp.isfinite(sigma_valid))
    result = logsumexp(base+mark, axis=1)-logsumexp(base, axis=1)-reference
    return jnp.where(valid, result, jnp.nan)
