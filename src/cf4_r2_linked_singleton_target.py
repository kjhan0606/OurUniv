"""Partial v6 count + linked-singleton target on one supplied PM field."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from cf4_r2_marked_tracer_jax import (
    intrinsic_biased_source_masses,
    intrinsic_lf_bin_fractions,
    predict_source_marked_intensity_los_node,
    sparse_marked_poisson_log_likelihood,
)
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_linked_singleton_jax import linked_singleton_logfactors_for_population
from cf4_2mpp_joint_likelihood_jax import _gaussian_hermite_rule
from cf4_r2_shell_cdf_count import predict_shell_cdf_intensity


def partial_v6_count_singleton_parts(
    rho_node,
    velocity_node,
    white_ic,
    white_tracer,
    source_geometry,
    links_by_population,
    train_keys,
    train_counts,
    train_exposure_mask,
    heldout_keys=None,
    heldout_counts=None,
    heldout_exposure_mask=None,
    *,
    box=384.,
    hubble=74.6,
    h=.746,
    rate_parameterization='reference',
    radial_min_cMpc_h=5.,
    radial_max_cMpc_h=180.,
    quadrature_order=15,
    white_fp_zero=0.,
    fp_zero_sd_dex=.004,
    count_integration='gh',
    count_cdf_order=4,
    count_cdf_segments=32,
):
    """Score v6 counts and strict training singleton marks on one live field.

    ``rho_node`` and ``velocity_node`` must be the density/mean velocity from
    the same evolved IC. ``links_by_population`` contains only strict training
    one-point/one-FP-row links with a caller-provided live-state candidate
    support. That support must be refreshed or bounded for optimizer trials.
    The 985 Tempel-grouped rows are deliberately not accepted here: their
    shared group-distance and FP covariance require a separate group factor.

    Return order is training-count, linked-singleton-mark, Gaussian-white
    prior, and untouched-heldout-count readout, followed by predicted count
    intensity. Only the first three terms belong in a MAP objective; heldout
    data are diagnostic output and are not included in the target. During
    fitting omit ALL heldout arguments: no heldout data are needed or scored.
    The shared FP zero point is a separate standard-normal coordinate;
    .004 dex is the existing development prior, not survey calibration.
    Association log probabilities are explicit per link; a zero array is a
    field-independent association sensitivity assumption, not calibration.
    """
    rho = jnp.asarray(rho_node)
    velocity = jnp.asarray(velocity_node)
    white = jnp.asarray(white_ic)
    tracer = jnp.asarray(white_tracer)
    n = rho.shape[0]
    if (rho.shape != (n, n, n) or velocity.shape != (3, n, n, n)
            or tracer.shape != (9,)):
        raise ValueError('partial v6 target requires rho[N,N,N], velocity[3,N,N,N], tracer[9]')
    density, count_velocity = native_mass_momentum_to_count_cells(rho, velocity, box)
    reference_fraction = jnp.sum(
        intrinsic_lf_bin_fractions(mstar=-23.28, alpha=-.94)[1:4])
    if rate_parameterization == 'reference':
        log_rate = jnp.log(reference_fraction) + 2.*tracer[0]
        reference_interval = (-25., -21.)
    elif rate_parameterization == 'all_faint_historical':
        log_rate = 2.*tracer[0]
        reference_interval = None
    else:
        raise ValueError('unknown source-rate parameterization')
    bias = jnp.exp(.5*tracer[1:6])
    sigma_los = 100.*jnp.exp(.5*tracer[6])
    alpha = -1. + .06*jnp.exp(.5*tracer[7])
    mstar = -23.28 + .2*tracer[8]
    intrinsic = intrinsic_biased_source_masses(
        density, log_rate, bias, mstar=mstar, alpha=alpha,
        reference_interval=reference_interval)
    source_velocity = jnp.moveaxis(count_velocity, 0, -1).reshape(-1, 3)
    observer = jnp.full(3, box/2.)
    radial_geometry = dict(observer=observer,
        box_size_cMpc_h=box, hubble_km_s_Mpc=hubble, little_h=h,
        radius_table_cMpc_h=source_geometry['radial_table'],
        modulus_table_h=source_geometry['modulus_table'],
        redshift_table=source_geometry['redshift_table'], grid_size=n,
        radial_min_cMpc_h=radial_min_cMpc_h,
        radial_max_cMpc_h=radial_max_cMpc_h,
        mstar=mstar, alpha=alpha)
    nodes, weights = _gaussian_hermite_rule(quadrature_order)

    @jax.checkpoint
    def add_node(total, node_weight):
        node, weight = node_weight
        contribution = predict_source_marked_intensity_los_node(
            source_geometry['positions'], source_velocity, intrinsic,
            source_geometry['angular'], node, weight,
            sigma_los_km_s=sigma_los, **radial_geometry)
        return total + contribution, None

    # One compiled body, rematerialized on reverse mode; no GH15 graph unroll.
    if count_integration=='gh':
        intensity, _ = jax.lax.scan(add_node, jnp.zeros((6, n, n, n), dtype=rho.dtype),
                                    (jnp.asarray(nodes), jnp.asarray(weights)))
    elif count_integration=='shell_cdf':
        intensity=predict_shell_cdf_intensity(source_geometry['positions'],
            source_velocity,intrinsic,source_geometry['angular'],sigma_los_km_s=sigma_los,
            order=count_cdf_order,segments=count_cdf_segments,**radial_geometry)
    else:
        raise ValueError('unknown count integration rule')
    count_train = sparse_marked_poisson_log_likelihood(
        intensity, jnp.asarray(train_keys), jnp.asarray(train_counts),
        selected_voxel_mask=jnp.asarray(train_exposure_mask))
    if heldout_keys is None and heldout_counts is None and heldout_exposure_mask is None:
        count_heldout = jnp.asarray(0., dtype=rho.dtype)
    elif any(x is None for x in (heldout_keys, heldout_counts, heldout_exposure_mask)):
        raise ValueError('supply all heldout arguments together, or none during fitting')
    else:
        count_heldout = sparse_marked_poisson_log_likelihood(
            intensity, jnp.asarray(heldout_keys), jnp.asarray(heldout_counts),
            selected_voxel_mask=jnp.asarray(heldout_exposure_mask))
    mark_train = jnp.asarray(0., dtype=intrinsic.dtype)
    for population in range(6):
        links = links_by_population[population]
        ids = jnp.asarray(links['candidate_source_ids'])
        if ids.shape[0] == 0:
            continue
        factors, _ = linked_singleton_logfactors_for_population(
            source_geometry['positions'], source_velocity, intrinsic,
            source_geometry['angular'], ids, links['candidate_mask'],
            links['association_logprob'], links['voxel_ijk'],
            links['observed_radius_cMpc_h'], links['dz_row'],
            links['eta_mean'], links['eta_std'], links['eta_alpha'],
            population=population, sigma_los_km_s=sigma_los,
            radial_geometry=radial_geometry,
            fp_zero_dex=fp_zero_sd_dex*white_fp_zero)
        mark_train = mark_train + jnp.sum(factors)
    prior = -.5*(jnp.vdot(white, white)+jnp.vdot(tracer, tracer)+white_fp_zero**2)
    return jnp.stack((count_train, mark_train, prior, count_heldout)), intensity
