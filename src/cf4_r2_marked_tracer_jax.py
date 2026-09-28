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

import math
import numbers

import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaln, logsumexp

from cf4_2mpp_joint_likelihood_jax import (
    _gaussian_hermite_rule, observer_centred_spherical_rsd_jax,
    tsc_deposit_jax,
)


TRUE_EDGES = (-jnp.inf, -25., -23.6666666666667,
              -22.3333333333333, -21., jnp.inf)
OBS_EDGES = (-25., -23.6666666666667, -22.3333333333333, -21.)


def _lf_interval(lower, upper, *, mstar, alpha):
    shape = alpha + 1.
    # Keep the argument to gammainc finite even at the unbounded intrinsic
    # bin edges.  Masking its *output* is insufficient: reverse-mode AD can
    # still encounter 0*inf at those inactive branches.
    lower_infinite = jnp.isneginf(lower)
    upper_infinite = jnp.isposinf(upper)
    x_lower = jnp.power(10., .4*(mstar-jnp.where(lower_infinite,mstar,lower)))
    x_upper = jnp.power(10., .4*(mstar-jnp.where(upper_infinite,mstar,upper)))
    cdf_lower = jnp.where(lower_infinite,1.,gammainc(shape,x_lower))
    cdf_upper = jnp.where(upper_infinite,0.,gammainc(shape,x_upper))
    return jnp.where(upper > lower,
                     jnp.maximum(cdf_lower-cdf_upper, 0.), 0.)


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


def intrinsic_lf_bin_fractions(*, mstar=-23.28, alpha=-.94):
    """Five true-K LF fractions for the same shape used by the transfer.

    The full faint tail is integrable only for ``alpha > -1``.  This helper
    does not supply the LF normalization or a prior on its shape. Both LF
    coordinates can be differentiated by the installed JAX incomplete-gamma
    rule within the physically allowed ``alpha > -1`` domain.
    """
    if isinstance(alpha, (int, float)) and alpha <= -1.:
        raise ValueError('unbounded intrinsic LF requires alpha > -1')
    total = _lf_interval(-jnp.inf, jnp.inf, mstar=mstar, alpha=alpha)
    return jnp.stack([_lf_interval(TRUE_EDGES[i], TRUE_EDGES[i+1],
                                   mstar=mstar, alpha=alpha)/total
                      for i in range(5)])


def intrinsic_lf_reference_weights(*, mstar=-23.28, alpha=-.94,
                                   reference_interval=(-25., -21.)):
    """True-K bin weights per galaxy in a finite reference-M interval.

    These weights need not sum to one: the outer bins retain possible
    migration into the observed sample.  The reference rate avoids defining
    its amplitude by the unobserved, extrapolated all-faint Schechter tail.
    The current unbounded faint bin still requires alpha > -1; this is not
    a validated faint-end law or a relaxation of that domain restriction.
    """
    if isinstance(alpha, (int, float)) and alpha <= -1.:
        raise ValueError('unbounded intrinsic LF requires alpha > -1')
    lower, upper = reference_interval
    if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
        raise ValueError('finite ordered LF reference interval required')
    reference = _lf_interval(lower, upper, mstar=mstar, alpha=alpha)
    return jnp.stack([_lf_interval(TRUE_EDGES[i], TRUE_EDGES[i+1],
                                   mstar=mstar, alpha=alpha)/reference
                      for i in range(5)])


def intrinsic_biased_source_masses(density, log_mean_rate_per_cell,
                                   intrinsic_bias, *, mstar=-23.28,
                                   alpha=-.94, reference_interval=None):
    """Five intrinsic true-K masses from one matter field and LF shape.

    Each luminosity response is normalized to unit spatial mean over the
    full periodic box.  With ``reference_interval``, the log rate denotes
    galaxies in that finite intrinsic magnitude interval, not all galaxies
    down to an unobserved infinitely faint limit.  This keeps the single
    intrinsic rate distinct from clustering bias. It does not specify priors,
    unresolved/empty-cell galaxies, or source calibration.
    The caller must keep intrinsic biases positive and alpha above -1.
    """
    flat = jnp.asarray(density).reshape(-1)
    bias = jnp.asarray(intrinsic_bias)
    if bias.shape != (5,) or flat.size == 0:
        raise ValueError('five true-K biases and a nonempty density required')
    occupied = flat[None, :] > 0
    safe_density = jnp.where(occupied, flat[None, :], 1.)
    response = jnp.where(occupied,
                         jnp.exp(bias[:, None]*jnp.log(safe_density)), 0.)
    response /= jnp.mean(response, axis=1, keepdims=True)
    weights = (intrinsic_lf_bin_fractions(mstar=mstar, alpha=alpha)
               if reference_interval is None else
               intrinsic_lf_reference_weights(
                   mstar=mstar, alpha=alpha,
                   reference_interval=reference_interval))
    return (jnp.exp(log_mean_rate_per_cell)*weights[:, None]*response)


def tsc_weight_at_voxel(positions, voxel_ijk, grid_size, box_size_cMpc_h):
    """Weight of each source in ONE observed voxel of the deposit kernel.

    This is the transpose of the *same* periodic, cell-centred TSC stencil
    used by ``tsc_deposit_jax``. It evaluates a specified voxel without
    allocating a source-by-observed-voxel matrix. ``grid_size >= 3`` avoids
    aliased TSC neighbours on smaller periodic grids.
    """
    if grid_size < 3 or jnp.asarray(voxel_ijk).shape != (3,):
        raise ValueError('TSC target requires three voxel indices and grid_size >= 3')
    cell = (jnp.asarray(positions) % box_size_cMpc_h) / (
        box_size_cMpc_h/grid_size) - .5
    target = jnp.asarray(voxel_ijk)
    delta = (target[None, :] - cell + grid_size/2.) % grid_size - grid_size/2.
    distance = jnp.abs(delta)
    axis_weight = jnp.where(distance < .5, .75-distance**2,
                            jnp.where(distance < 1.5,
                                      .5*(1.5-distance)**2, 0.))
    return jnp.prod(axis_weight, axis=1)


def _source_marked_nodes(positions, velocity, observer, box_size_cMpc_h,
                         hubble_km_s_Mpc, little_h, radius_table,
                         modulus_table, redshift_values, sigma_los_km_s,
                         radial_min_cMpc_h, radial_max_cMpc_h,
                         quadrature_order, mstar, alpha):
    """Yield the shared source selection/K/RSD transfer for count and link."""
    shifted, _, rhat = observer_centred_spherical_rsd_jax(
        positions, velocity, observer, box_size_cMpc_h,
        hubble_km_s_Mpc, little_h=little_h, scale_factor=1.)
    true_relative = (positions-observer+box_size_cMpc_h/2.) % box_size_cMpc_h
    true_relative -= box_size_cMpc_h/2.
    true_radius = jnp.linalg.norm(true_relative, axis=1)
    true_modulus = jnp.interp(true_radius, radius_table, modulus_table)
    true_redshift = jnp.interp(true_radius, radius_table, redshift_values)
    nodes, weights = _gaussian_hermite_rule(quadrature_order)
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
                                        true_redshift, observed_redshift,
                                        mstar=mstar, alpha=alpha)
        selected = (observed_radius >= radial_min_cMpc_h) & (
            observed_radius <= radial_max_cMpc_h)
        yield observed_positions, selected, transfer, weight


def sparse_marked_poisson_log_likelihood(intensity, observed_keys,
                                          observed_counts, *,
                                          selected_voxel_mask=None):
    """One Poisson factor for binned selected counts, with no support floor.

    ``intensity`` must have the same mark definition as ``observed_counts``.
    The boolean exposure mask may be over voxels shared by every population,
    or flattened over population-by-voxel keys for population-specific
    exposure. Its exact integral includes empty selected cells only.
    It is not a globally thinned catalogue, so no fraction rescales rates.
    This factor must not be multiplied by another likelihood for the same
    observed absolute-K bin frequencies.
    """
    if observed_keys.shape != observed_counts.shape:
        raise ValueError('sparse observed count geometry mismatch')
    flat = jnp.asarray(intensity).reshape(-1)
    occupied = jnp.take(flat, observed_keys)
    counts = jnp.asarray(observed_counts, dtype=flat.dtype)
    integral = jnp.sum(flat)
    valid = jnp.array(True)
    if selected_voxel_mask is not None:
        mask = jnp.asarray(selected_voxel_mask, dtype=bool).reshape(-1)
        if mask.size == 0 or flat.size % mask.size:
            raise ValueError('spatial count mask does not divide intensity geometry')
        nvoxel = mask.size
        keys = jnp.asarray(observed_keys)
        valid_keys = (keys >= 0) & (keys < flat.size)
        safe_voxels = jnp.clip(keys % nvoxel, 0, nvoxel-1)
        valid = jnp.all(valid_keys & jnp.take(mask, safe_voxels))
        integral = jnp.sum(jnp.reshape(flat, (-1, nvoxel))*mask[None, :])
    score = jnp.sum(counts*jnp.log(occupied)-gammaln(counts+1.))-integral
    return jnp.where(valid, score, -jnp.inf)


def predict_source_marked_intensity(
    source_positions, source_velocities_km_s, intrinsic_bin_masses,
    angular_completeness, *, observer, box_size_cMpc_h,
    hubble_km_s_Mpc, little_h, radius_table_cMpc_h,
    modulus_table_h, redshift_table, grid_size,
    sigma_los_km_s=0., radial_min_cMpc_h=5.,
    radial_max_cMpc_h=180., quadrature_order=3,
    mstar=-23.28, alpha=-.94,
):
    """Deposit source-selected K counts after spherical RSD/LOS convolution.

    ``intrinsic_bin_masses`` has shape (5, nsource); the five true-M bins span
    all luminosities and are free *intrinsic* count masses.  ``angular_completeness``
    has shape (2, nsource), one fixed-sightline value per apparent-K sample.
    The radial lookup tables must be monotone and include the selected volume.
    The supplied count masses/rates and LF shape are not calibrated here.
    If LF parameters vary, the caller must rebuild the five intrinsic bin
    masses using the same parameters (e.g. intrinsic_lf_bin_fractions).
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
    outputs = [jnp.zeros((grid_size,)*3, dtype=intrinsic.dtype) for _ in range(6)]
    for observed_positions, selected, transfer, weight in _source_marked_nodes(
            positions, velocity, observer, box_size_cMpc_h,
            hubble_km_s_Mpc, little_h, radius_table, modulus_table,
            redshift_values, sigma_los_km_s, radial_min_cMpc_h,
            radial_max_cMpc_h, quadrature_order, mstar, alpha):
        for population in range(6):
            mass = (weight * selected * angular[population//3]
                    * jnp.sum(transfer[population]*intrinsic, axis=0))
            outputs[population] = outputs[population] + tsc_deposit_jax(
                observed_positions, mass, grid_size, box_size_cMpc_h)
    return jnp.stack(outputs)


def predict_source_marked_intensity_los_node(
    source_positions, source_velocities_km_s, intrinsic_bin_masses,
    angular_completeness, los_node, quadrature_weight, *, observer,
    box_size_cMpc_h, hubble_km_s_Mpc, little_h,
    radius_table_cMpc_h, modulus_table_h, redshift_table, grid_size,
    sigma_los_km_s=0., radial_min_cMpc_h=5.,
    radial_max_cMpc_h=180., mstar=-23.28, alpha=-.94,
):
    """One Gaussian-Hermite LOS node of the selected count intensity.

    This exposes a single node so callers can accumulate high-order rules
    without statically unrolling every node into one large XLA graph. Summing
    this result over the nodes and weights from ``_gaussian_hermite_rule``
    reproduces ``predict_source_marked_intensity``.
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
            or intrinsic.shape != (5, count) or angular.shape != (2, count)
            or radius_table.ndim != 1 or modulus_table.shape != radius_table.shape
            or redshift_values.shape != radius_table.shape or radius_table.size < 2
            or grid_size < 1 or radial_min_cMpc_h >= radial_max_cMpc_h):
        raise ValueError('source, intrinsic-bin or angular geometry mismatch')
    shifted, _, rhat = observer_centred_spherical_rsd_jax(
        positions, velocity, observer, box_size_cMpc_h,
        hubble_km_s_Mpc, little_h=little_h, scale_factor=1.)
    relative = (positions-observer+box_size_cMpc_h/2.) % box_size_cMpc_h
    relative -= box_size_cMpc_h/2.
    true_radius = jnp.linalg.norm(relative, axis=1)
    true_modulus = jnp.interp(true_radius, radius_table, modulus_table)
    true_redshift = jnp.interp(true_radius, radius_table, redshift_values)
    extra = (los_node*little_h*sigma_los_km_s/hubble_km_s_Mpc)
    observed_positions = (shifted+extra*rhat) % box_size_cMpc_h
    observed_relative = ((observed_positions-observer+box_size_cMpc_h/2.)
                         % box_size_cMpc_h-box_size_cMpc_h/2.)
    observed_radius = jnp.linalg.norm(observed_relative, axis=1)
    observed_modulus = jnp.interp(observed_radius, radius_table, modulus_table)
    observed_redshift = jnp.interp(observed_radius, radius_table, redshift_values)
    transfer = source_mark_transfer(true_modulus, observed_modulus,
                                    true_redshift, observed_redshift,
                                    mstar=mstar, alpha=alpha)
    selected = ((observed_radius >= radial_min_cMpc_h)
                & (observed_radius <= radial_max_cMpc_h))
    outputs = []
    for population in range(6):
        mass = (quadrature_weight*selected*angular[population//3]
                *jnp.sum(transfer[population]*intrinsic, axis=0))
        outputs.append(tsc_deposit_jax(
            observed_positions, mass, grid_size, box_size_cMpc_h))
    return jnp.stack(outputs)


def predict_source_marked_key_contributions(
    source_positions, source_velocities_km_s, intrinsic_bin_masses,
    angular_completeness, population, voxel_ijk, *, observer,
    box_size_cMpc_h, hubble_km_s_Mpc, little_h,
    radius_table_cMpc_h, modulus_table_h, redshift_table, grid_size,
    sigma_los_km_s=0., radial_min_cMpc_h=5.,
    radial_max_cMpc_h=180., quadrature_order=3,
    mstar=-23.28, alpha=-.94,
):
    """Five-true-K-by-source contributions to ONE count population/voxel.

    Summing both axes must reproduce the corresponding element of
    ``predict_source_marked_intensity`` with identical inputs. The entries
    provide an exact **coarsened count-key** source mixture; they are not yet
    conditioned on an individual's measured redshift, FP inclusion, or group
    association. Never multiply their marginal intensity into a second count
    likelihood for the same point.
    """
    positions = jnp.asarray(source_positions)
    velocity = jnp.asarray(source_velocities_km_s)
    intrinsic = jnp.asarray(intrinsic_bin_masses)
    angular = jnp.asarray(angular_completeness)
    radius_table = jnp.asarray(radius_table_cMpc_h)
    modulus_table = jnp.asarray(modulus_table_h)
    redshift_values = jnp.asarray(redshift_table)
    count = positions.shape[0]
    if (not isinstance(population, int) or population < 0 or population >= 6
            or positions.shape != (count, 3) or velocity.shape != positions.shape
            or intrinsic.shape != (5, count) or angular.shape != (2, count)
            or radius_table.ndim != 1 or modulus_table.shape != radius_table.shape
            or redshift_values.shape != radius_table.shape or radius_table.size < 2
            or radial_min_cMpc_h >= radial_max_cMpc_h):
        raise ValueError('invalid source-marked key geometry')
    # Also checks the target shape. The array value is allowed to be traced.
    tsc_weight_at_voxel(positions[:1], voxel_ijk, grid_size,
                        box_size_cMpc_h)
    result = jnp.zeros_like(intrinsic)
    for observed_positions, selected, transfer, weight in _source_marked_nodes(
            positions, velocity, observer, box_size_cMpc_h,
            hubble_km_s_Mpc, little_h, radius_table, modulus_table,
            redshift_values, sigma_los_km_s, radial_min_cMpc_h,
            radial_max_cMpc_h, quadrature_order, mstar, alpha):
        spatial = tsc_weight_at_voxel(observed_positions, voxel_ijk, grid_size,
                                      box_size_cMpc_h)
        result = result + (weight*selected*angular[population//3]*spatial)[None, :] * (
            transfer[population]*intrinsic)
    return result


def predict_source_marked_radial_key_density(
    source_positions, source_velocities_km_s, intrinsic_bin_masses,
    angular_completeness, population, voxel_ijk, observed_radius_cMpc_h,
    *, observer, box_size_cMpc_h, hubble_km_s_Mpc, little_h,
    radius_table_cMpc_h, modulus_table_h, redshift_table, grid_size,
    sigma_los_km_s, radial_min_cMpc_h=5., radial_max_cMpc_h=180.,
    mstar=-23.28, alpha=-.94,
):
    """Source contribution per unit *observed comoving radius* at one key.

    Both signed LOS branches through the observer are included. Integrating
    over observed radius approaches the count-key source contribution when
    periodic-image aliases are negligible and the quadrature is converged.
    The current count operator uses finite Gaussian-Hermite quadrature, so
    equality is **not** automatic at finite order or near K/radial cuts.
    A measured individual redshift can supply ``observed_radius``;
    multiplying by dr/dz converts to density per unit z, but that Jacobian
    cancels from a mark factor conditioned on the same observed redshift.
    ``sigma_los_km_s`` must be positive; a traced JAX scalar is supported so
    the LOS-width nuisance can be differentiated inside a live-field target.
    This does not model FP-group inclusion or association probability.
    """
    positions = jnp.asarray(source_positions)
    velocity = jnp.asarray(source_velocities_km_s)
    intrinsic = jnp.asarray(intrinsic_bin_masses)
    angular = jnp.asarray(angular_completeness)
    radius_table = jnp.asarray(radius_table_cMpc_h)
    modulus_table = jnp.asarray(modulus_table_h)
    redshift_values = jnp.asarray(redshift_table)
    count = positions.shape[0]
    if (not isinstance(population, int) or population < 0 or population >= 6
            or positions.shape != (count, 3) or velocity.shape != positions.shape
            or intrinsic.shape != (5, count) or angular.shape != (2, count)
            or radius_table.ndim != 1 or modulus_table.shape != radius_table.shape
            or redshift_values.shape != radius_table.shape or radius_table.size < 2
            or (isinstance(sigma_los_km_s, numbers.Real)
                and sigma_los_km_s <= 0)
            or radial_min_cMpc_h >= radial_max_cMpc_h
            or radial_max_cMpc_h > box_size_cMpc_h/2.):
        raise ValueError('invalid continuous source-marked radial geometry')
    tsc_weight_at_voxel(positions[:1], voxel_ijk, grid_size,
                        box_size_cMpc_h)
    shifted, _, rhat = observer_centred_spherical_rsd_jax(
        positions, velocity, observer, box_size_cMpc_h,
        hubble_km_s_Mpc, little_h=little_h, scale_factor=1.)
    relative = (positions-observer+box_size_cMpc_h/2.) % box_size_cMpc_h
    relative -= box_size_cMpc_h/2.
    true_radius = jnp.linalg.norm(relative, axis=1)
    shifted_relative = (shifted-observer+box_size_cMpc_h/2.) % box_size_cMpc_h
    shifted_relative -= box_size_cMpc_h/2.
    shifted_radius = jnp.linalg.norm(shifted_relative, axis=1)
    r_observed = jnp.asarray(observed_radius_cMpc_h)
    sigma_radius = little_h*sigma_los_km_s/hubble_km_s_Mpc
    log_normalizer = -jnp.log(sigma_radius)-.5*jnp.log(2*jnp.pi)
    radial_pdf_plus = jnp.exp(
        -.5*((r_observed-shifted_radius)/sigma_radius)**2+log_normalizer)
    radial_pdf_minus = jnp.exp(
        -.5*((-r_observed-shifted_radius)/sigma_radius)**2+log_normalizer)
    observed_plus = (observer + r_observed*rhat) % box_size_cMpc_h
    observed_minus = (observer - r_observed*rhat) % box_size_cMpc_h
    spatial = (radial_pdf_plus*tsc_weight_at_voxel(
        observed_plus, voxel_ijk, grid_size, box_size_cMpc_h)
        + radial_pdf_minus*tsc_weight_at_voxel(
            observed_minus, voxel_ijk, grid_size, box_size_cMpc_h))
    transfer = source_mark_transfer(
        jnp.interp(true_radius, radius_table, modulus_table),
        jnp.interp(r_observed, radius_table, modulus_table),
        jnp.interp(true_radius, radius_table, redshift_values),
        jnp.interp(r_observed, radius_table, redshift_values),
        mstar=mstar, alpha=alpha)[population]
    selected = ((r_observed >= radial_min_cMpc_h)
                & (r_observed <= radial_max_cMpc_h))
    return (selected*spatial*angular[population//3])[None, :] * transfer*intrinsic


def conditional_single_link_logfactor(source_key_radial_density,
                                      log_association, log_fp_mark):
    """Normalized FP-mark factor given a counted point's key and redshift.

    All arguments are (five true-K bins, source cells). The caller supplies
    the *same* selected source response as the count factor, a calibrated or
    explicitly conditional association law, and a source-corrected FP mark
    log likelihood ratio. Count occurrence and observed redshift are not
    multiplied again. Group-shared offsets/anchors require an outer group
    calculation; this function alone is not the all-group R2 likelihood.
    """
    mass = jnp.asarray(source_key_radial_density)
    association = jnp.asarray(log_association)
    mark = jnp.asarray(log_fp_mark)
    if (mass.ndim != 2 or mass.shape[0] != 5
            or association.shape != mass.shape or mark.shape != mass.shape):
        raise ValueError('linked mark requires aligned five-bin source arrays')
    safe = jnp.where(mass > 0., mass, 1.)
    base = jnp.where(mass > 0., jnp.log(safe), -jnp.inf) + association
    normalizer = logsumexp(base)
    numerator = logsumexp(base+mark)
    return jnp.where((mass >= 0.).all() & jnp.isfinite(normalizer),
                     numerator-normalizer, -jnp.inf)
