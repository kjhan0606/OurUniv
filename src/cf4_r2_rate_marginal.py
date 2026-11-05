"""Exact six-population Poisson likelihood with uncertain mean galaxy rates.

Each unselected rate per cell has a Gamma(shape, shape/prior_mean) prior.
The input ``unit_intensity`` is the selected expected cell count for rate one,
including the predeclared training fraction. No observed-count normalization
is applied to it. This is a development nuisance law; its Gamma shape is not
claimed to be calibrated for the CF4-disjoint 2M++ sample.
"""

import jax.numpy as jnp
from jax.ops import segment_sum
from jax.scipy.special import gammaln


def gamma_poisson_log_marginal_sparse(
    unit_intensity, observed_keys, observed_counts, prior_mean, shape,
):
    """Marginal likelihood for six independent rates and sparse positive counts.

    For population p, I_p=sum_i unit_intensity[p,i], N_p=sum_i n[p,i],
    beta_p=shape[p]/prior_mean[p]. Integrating rate r_p gives

      beta^a Gamma(a+N)/(Gamma(a)(beta+I)^(a+N))
      * product_i unit_intensity_i^n_i / n_i!.

    Any positive count at exact zero support yields -inf. The caller must
    preflight support before differentiating; no numerical intensity floor is
    inserted here.
    """
    if unit_intensity.shape[0] != 6 or prior_mean.shape != (6,) or shape.shape != (6,):
        raise ValueError('six-population geometry required')
    if observed_keys.shape != observed_counts.shape:
        raise ValueError('sparse key/count geometry mismatch')
    cells = unit_intensity.size // 6
    pop = observed_keys // cells
    unit = unit_intensity.reshape(-1)
    occupied = jnp.take(unit, observed_keys)
    # gammaln(integer) is evaluated in float32 by JAX even when x64 is
    # enabled. Cast counts explicitly before both the factorial and totals.
    counts = observed_counts.astype(unit_intensity.dtype)
    N = segment_sum(counts, pop, num_segments=6)
    weighted_log = segment_sum(counts*jnp.log(occupied), pop,
                               num_segments=6)
    exposure = jnp.sum(unit_intensity.reshape(6,-1), axis=1)
    beta = shape/prior_mean
    per_population = (shape*jnp.log(beta) - gammaln(shape)
        + gammaln(shape+N) - (shape+N)*jnp.log(beta+exposure)
        + weighted_log - segment_sum(gammaln(counts+1),pop,
                                     num_segments=6))
    return jnp.sum(per_population)
