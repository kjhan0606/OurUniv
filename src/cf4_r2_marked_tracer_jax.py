"""Source-selected six-population 2M++ count operator (development model).

Intrinsic K-bin count mass is assigned at the true source cell.  Each coherent
and stochastic LOS displacement changes the catalogue's corrected apparent K,
observed-redshift absolute K, and radial support *before* deposition.  The
angular completeness is supplied at the fixed source sightline.  Unlike the
older operator, no observed-voxel exposure is multiplied after deposition.

This does not calibrate intrinsic luminosity rates/bias, source inclusion,
group overlap or the Schechter within-bin shape.  It is not a production
likelihood without those ingredients and a held-out/mock assessment.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import gammainc

from cf4_2mpp_joint_likelihood_jax import (
    _gaussian_hermite_rule, observer_centred_spherical_rsd_jax,
    tsc_deposit_jax,
)


TRUE_EDGES = (-jnp.inf, -25., -23.6666666666667,
              -22.3333333333333, -21., jnp.inf)
OBS_EDGES = (-25., -23.6666666666667, -22.3333333333333, -21.)


def _lf_interval(lower, upper, *, mstar, alpha):
    shape = alpha + 1.
    x_lower = jnp.power(10., .4*(mstar-lower))
    x_upper = jnp.power(10., .4*(mstar-upper))
    return jnp.where(upper > lower,
                     jnp.maximum(gammainc(shape, x_lower)
                                 - gammainc(shape, x_upper), 0.), 0.)


def source_mark_transfer(true_modulus_h, observed_modulus_h,
                         true_redshift, observed_redshift, *,
                         mstar=-23.28, alpha=-.94):
    """Return six-by-five-by-source intrinsic-to-observed K probabilities.

    The within-intrinsic-bin LF is Schechter.  This is a conditional transfer,
    not an intrinsic rate or an actual-survey selection probability.
    """
    correction = (1.16*2.9*(observed_redshift-true_redshift)
                  - 1.6*jnp.log10((1.+observed_redshift)
                                    /(1.+true_redshift)))
    shift = observed_modulus_h - true_modulus_h - correction
    rows = []
    for app_lo, app_hi in ((-jnp.inf, 11.5), (11.5, 12.5)):
        apparent_lo = app_lo - true_modulus_h - correction
        apparent_hi = app_hi - true_modulus_h - correction
        for observed_bin in range(3):
            columns = []
            for intrinsic_bin in range(5):
                true_lo, true_hi = TRUE_EDGES[intrinsic_bin:intrinsic_bin+2]
                lower = jnp.maximum(jnp.maximum(true_lo, apparent_lo),
                                    OBS_EDGES[observed_bin]+shift)
                upper = jnp.minimum(jnp.minimum(true_hi, apparent_hi),
                                    OBS_EDGES[observed_bin+1]+shift)
                columns.append(_lf_interval(lower, upper, mstar=mstar,
                                            alpha=alpha)
                               / _lf_interval(true_lo, true_hi,
                                              mstar=mstar, alpha=alpha))
            rows.append(jnp.stack(columns))
    return jnp.stack(rows)


def predict_source_marked_intensity(
    source_positions, source_velocities_km_s, intrinsic_bin_masses,
    angular_completeness, *, observer, box_size_cMpc_h,
    hubble_km_s_Mpc, little_h, radius_table_cMpc_h,
    modulus_table_h, redshift_table, grid_size,
    sigma_los_km_s=0., radial_min_cMpc_h=5.,
    radial_max_cMpc_h=180., quadrature_order=3,
):
    """Deposit source-selected K counts after spherical RSD/LOS convolution.

    ``intrinsic_bin_masses`` has shape (5, nsource); the five true-M bins span
    all luminosities and are free *intrinsic* count masses.  ``angular_completeness``
    has shape (2, nsource), one fixed-sightline value per apparent-K sample.
    The radial lookup tables must be monotone and include the selected volume.
    The supplied count masses/rates and LF shape are not calibrated here.
    """
    positions = jnp.asarray(source_positions)
    velocity = jnp.asarray(source_velocities_km_s)
    intrinsic = jnp.asarray(intrinsic_bin_masses)
    angular = jnp.asarray(angular_completeness)
    radius_table = jnp.asarray(radius_table_cMpc_h)
    modulus_table = jnp.asarray(modulus_table_h)
    redshift_values = jnp.asarray(redshift_table)
    count = positions.shape[0]
    if (positions.shape != (count, 3) or velocity.shape != positions.shape
            or intrinsic.shape != (5, count) or angular.shape != (2, count)):
        raise ValueError('source, intrinsic-bin or angular geometry mismatch')
    if (radius_table.ndim != 1 or modulus_table.shape != radius_table.shape
            or redshift_values.shape != radius_table.shape
            or radius_table.size < 2):
        raise ValueError('radial lookup geometry mismatch')
    if grid_size < 1 or radial_min_cMpc_h >= radial_max_cMpc_h:
        raise ValueError('invalid count grid or selected radial interval')
    shifted, _, rhat = observer_centred_spherical_rsd_jax(
        positions, velocity, observer, box_size_cMpc_h,
        hubble_km_s_Mpc, little_h=little_h, scale_factor=1.)
    true_relative = (positions-observer+box_size_cMpc_h/2.) % box_size_cMpc_h
    true_relative -= box_size_cMpc_h/2.
    true_radius = jnp.linalg.norm(true_relative, axis=1)
    true_modulus = jnp.interp(true_radius, radius_table, modulus_table)
    true_redshift = jnp.interp(true_radius, radius_table, redshift_values)
    nodes, weights = _gaussian_hermite_rule(quadrature_order)
    outputs = [jnp.zeros((grid_size,)*3, dtype=intrinsic.dtype) for _ in range(6)]
    for node, weight in zip(nodes, weights):
        extra = node*little_h*sigma_los_km_s/hubble_km_s_Mpc
        observed_positions = (shifted + extra*rhat) % box_size_cMpc_h
        observed_relative = ((observed_positions-observer
                              +box_size_cMpc_h/2.) % box_size_cMpc_h
                             -box_size_cMpc_h/2.)
        observed_radius = jnp.linalg.norm(observed_relative, axis=1)
        observed_modulus = jnp.interp(observed_radius, radius_table,
                                      modulus_table)
        observed_redshift = jnp.interp(observed_radius, radius_table,
                                       redshift_values)
        transfer = source_mark_transfer(true_modulus, observed_modulus,
                                        true_redshift, observed_redshift)
        selected = (observed_radius >= radial_min_cMpc_h) & (
            observed_radius <= radial_max_cMpc_h)
        for population in range(6):
            mass = (weight * selected * angular[population//3]
                    * jnp.sum(transfer[population]*intrinsic, axis=0))
            outputs[population] = outputs[population] + tsc_deposit_jax(
                observed_positions, mass, grid_size, box_size_cMpc_h)
    return jnp.stack(outputs)
