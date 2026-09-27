"""One-state R2 count + FP + disjoint TF partial-target factorization."""

import jax.numpy as jnp

from cf4_r2_coarsened_live_joint import count_and_mark_parts
from cf4_r2_hierarchical_marks import decode_hyper
from cf4_r2_tf_group_marks import tf_group_logratios


def count_fp_tf_parts(rho, velocity, white_ic, white_hyper, white_group,
                      fp_geometry, tf_geometry, calibration_sd,
                      exposure, keys, counts, rate_mean, rate_shape,
                      count_bias, count_fog, count_redshift,
                      distance_nodes, quadrature_weights, zcos_nodes,
                      *, box=384., hubble=74.6, h=.746,
                      tf_group_sigma_v=150., tf_selected_bias=1.):
    """Return count, FP, TF, prior with distinct datum ownership.

    TF uses the SAME CF4 relative-TF zero parameter as the existing FP-group
    non-FP anchors. Its group redshift is only a conditioned covariate inside
    numerator and denominator, never an independent velocity likelihood.
    The TF selected-density exponent and group velocity width remain
    uncalibrated development inputs.
    """
    base, unit = count_and_mark_parts(rho, velocity, white_ic, white_hyper,
        white_group, fp_geometry, calibration_sd, exposure, keys, counts,
        rate_mean, rate_shape, count_bias, count_fog, count_redshift,
        box=box, hubble=hubble, h=h)
    parameters, _fp_selected_bias, _tau = decode_hyper(white_hyper, calibration_sd)
    tf_scores = tf_group_logratios(rho, velocity, tf_geometry['directions'],
        tf_geometry['observed_cz'], tf_geometry['modulus'],
        tf_geometry['modulus_error'], distance_nodes, quadrature_weights,
        zcos_nodes, box=box, h=h, selected_bias=tf_selected_bias,
        sigma_v=tf_group_sigma_v, modulus_zero=parameters[4])
    tf = jnp.where(tf_geometry['holdout'], 0., tf_scores).sum()
    return jnp.stack((base[0], base[1], tf, base[2])), unit
