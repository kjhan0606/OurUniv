"""Source-selected 2M++ sky counts plus graph-closed FP marks on one IC.

This is an explicit partial R2 target. Selection, LF and group covariance
remain development assumptions, not a scientifically calibrated posterior.
"""

import jax.numpy as jnp

from cf4_r2_hierarchical_marks import hierarchical_group_scores
from cf4_r2_marked_tracer_jax import (
    intrinsic_biased_source_masses, intrinsic_lf_bin_fractions,
    predict_source_marked_intensity,
    sparse_marked_poisson_log_likelihood,
)
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells


def source_sky_count_fp_parts(rho_node, velocity_node, white_ic, white_hyper,
                              white_group, white_tracer, fp_geometry,
                              source_geometry, calibration_sd,
                              train_keys, train_counts, heldout_keys,
                              heldout_counts, heldout_voxel_mask, *,
                              box=384., hubble=74.6, h=.746, n=128,
                              rate_parameterization='reference'):
    """Return train count, train FP, white prior, heldout count, and intensity.

    The nine tracer coordinates have a caller-applied standard-normal
    regularizer: one log rate, five positive intrinsic-K bias responses, one
    FoG width, and the shared LF alpha/Mstar. The log-rate scale is 2 and its
    reference centre includes the finite-window LF fraction at the default
    same-catalogue LF shape. Only the five bias coordinates
    are unit-centred broad bias regularizers. The LF coordinates are centred
    on the Lavaux-Hudson (2011) 2M++ K<11.5 LF row; they are same-survey
    empirical regularization, not independent calibration. In particular the
    alpha transform is support-restricted at -1 and must be redesigned with
    the finite faint-end/selection law before future sampling. No separate
    conditional-K likelihood is multiplied in. The heldout count is a
    *readout*, never part of target.
    """
    if white_tracer.shape != (9,):
        raise ValueError('source tracer requires nine white coordinates')
    rho, velocity = native_mass_momentum_to_count_cells(
        rho_node, velocity_node, box)
    # Match the old zero-coordinate prediction at the development LF centre,
    # while making the rate coordinate refer to the source's finite bright
    # interval. The broad white prior remains development regularization.
    if rate_parameterization == 'reference':
        centre_reference_fraction = jnp.sum(
            intrinsic_lf_bin_fractions(mstar=-23.28, alpha=-.94)[1:4])
        log_rate = jnp.log(centre_reference_fraction) + 2.*white_tracer[0]
        reference_interval = (-25., -21.)
    elif rate_parameterization == 'all_faint_historical':
        log_rate = 2.*white_tracer[0]
        reference_interval = None
    else:
        raise ValueError('unknown source-rate parameterization')
    bias = jnp.exp(.5*white_tracer[1:6])
    sigma_los = 100.*jnp.exp(.5*white_tracer[6])
    alpha = -1. + .06*jnp.exp(.5*white_tracer[7])
    mstar = -23.28 + .2*white_tracer[8]
    intrinsic = intrinsic_biased_source_masses(
        rho, log_rate, bias, mstar=mstar, alpha=alpha,
        reference_interval=reference_interval)
    intensity = predict_source_marked_intensity(
        source_geometry['positions'],
        jnp.moveaxis(velocity, 0, -1).reshape(-1, 3),
        intrinsic, source_geometry['angular'],
        observer=jnp.full(3, box/2), box_size_cMpc_h=box,
        hubble_km_s_Mpc=hubble, little_h=h,
        radius_table_cMpc_h=source_geometry['radial_table'],
        modulus_table_h=source_geometry['modulus_table'],
        redshift_table=source_geometry['redshift_table'], grid_size=n,
        sigma_los_km_s=sigma_los, radial_min_cMpc_h=5.,
        radial_max_cMpc_h=180., quadrature_order=3,
        mstar=mstar, alpha=alpha)
    train = sparse_marked_poisson_log_likelihood(
        intensity, train_keys, train_counts,
        selected_voxel_mask=~heldout_voxel_mask)
    heldout = sparse_marked_poisson_log_likelihood(
        intensity, heldout_keys, heldout_counts,
        selected_voxel_mask=heldout_voxel_mask)
    marks = hierarchical_group_scores(
        rho_node, velocity_node, white_hyper, white_group,
        fp_geometry, calibration_sd, box=box)
    fp_train = jnp.where(fp_geometry['group_holdout'], 0., marks).sum()
    prior = -.5*(jnp.vdot(white_ic, white_ic)
                 + jnp.vdot(white_hyper, white_hyper)
                 + jnp.vdot(white_group, white_group)
                 + jnp.vdot(white_tracer, white_tracer))
    return jnp.stack((train, fp_train, prior, heldout)), intensity
