"""Batched conditional FP factors for one-point/one-row training links."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from cf4_r2_fp_distance import fp_log_likelihood_ratio
from cf4_r2_marked_tracer_jax import (
    conditional_single_link_logfactor,
    predict_source_marked_radial_key_density,
)


def linked_singleton_logfactors_for_population(
    source_positions,
    source_velocities_km_s,
    intrinsic_bin_masses,
    angular_completeness,
    candidate_source_ids,
    candidate_mask,
    association_logprob,
    voxel_ijk,
    observed_radius_cMpc_h,
    dz_row,
    eta_mean,
    eta_std,
    eta_alpha,
    *,
    population,
    sigma_los_km_s,
    radial_geometry,
    fp_zero_dex=0.,
    return_eta_moments=False,
    return_eta_mixture=False,
):
    """Evaluate batched linked singleton marks for one observed population.

    Candidate IDs/support are a caller-owned state-local spatial index. The
    caller must rebuild or conservatively bound it whenever the live RSD
    source state changes. No physical association calibration is implied;
    ``association_logprob`` is explicit and may be zero for the declared
    field-independent sensitivity model. MULTIPLE scored FP members of one
    source group need a shared group-distance/covariance factor. A physically
    grouped galaxy with exactly ONE scored FP mark and one linked point may
    use this conditional working marginal: dz_row is the fixed source FP
    reference, not an independently scored group redshift. This does not
    calibrate its LOS kernel or missing common fitted-FP errors.

    Arrays are group-major: candidate IDs/mask ``(group, padded-source)``,
    association log probabilities ``(group, true-K-bin, padded-source)``,
    voxels ``(group,3)``, and the remaining observations ``(group,)``.
    The returned per-group factors are conditional FP mark log-likelihood
    ratios; count occurrence and observed redshift are not scored again.
    The static readout option ``return_eta_moments`` additionally returns
    count/association-conditioned eta mean and variance BEFORE the FP mark
    or zero point is applied. These are NOT posterior field uncertainty.
    ``return_eta_mixture`` requires moments and appends candidate eta and
    normalized count/association log weights, collapsing only true-K bins.
    It permits exact fixed-field zero-point comparisons without rerunning
    the source-selection kernel; it does not change the live target.
    """
    positions = jnp.asarray(source_positions)
    velocities = jnp.asarray(source_velocities_km_s)
    intrinsic = jnp.asarray(intrinsic_bin_masses)
    angular = jnp.asarray(angular_completeness)
    ids = jnp.asarray(candidate_source_ids, dtype=jnp.int32)
    active = jnp.asarray(candidate_mask, dtype=bool)
    association = jnp.asarray(association_logprob)
    voxels = jnp.asarray(voxel_ijk, dtype=jnp.int32)
    radius = jnp.asarray(observed_radius_cMpc_h)
    dz = jnp.asarray(dz_row)
    mean = jnp.asarray(eta_mean)
    std = jnp.asarray(eta_std)
    alpha = jnp.asarray(eta_alpha)
    groups = ids.shape[0] if ids.ndim == 2 else -1
    if return_eta_mixture and not return_eta_moments:
        raise ValueError('eta mixture readout requires eta moments')
    if (population not in range(6) or positions.ndim != 2
            or positions.shape[1] != 3 or velocities.shape != positions.shape
            or intrinsic.shape != (5, positions.shape[0])
            or angular.shape != (2, positions.shape[0])
            or ids.ndim != 2 or active.shape != ids.shape
            or ids.shape[1] < 1 or association.shape != (groups, 5, ids.shape[1])
            or voxels.shape != (groups, 3)
            or any(x.shape != (groups,) for x in (radius, dz, mean, std, alpha))):
        raise ValueError('linked singleton batch geometry mismatch')

    def one_group(group_ids, group_active, group_association, voxel, r_obs,
                  group_dz, fp_mean, fp_std, fp_alpha):
        pos = positions[group_ids]
        vel = velocities[group_ids]
        mass = intrinsic[:, group_ids] * group_active[None, :]
        sky = angular[:, group_ids]
        density = predict_source_marked_radial_key_density(
            pos, vel, mass, sky, population, voxel, r_obs,
            sigma_los_km_s=sigma_los_km_s, **radial_geometry)
        relative = (pos-radial_geometry['observer']
                    +radial_geometry['box_size_cMpc_h']/2.) % radial_geometry[
                        'box_size_cMpc_h']
        relative -= radial_geometry['box_size_cMpc_h']/2.
        true_radius = jnp.linalg.norm(relative, axis=1)
        eta = jnp.log10(group_dz/true_radius)
        log_mark = fp_log_likelihood_ratio(
            eta + fp_zero_dex, 0., fp_mean, fp_std, fp_alpha)
        mark_matrix = jnp.broadcast_to(log_mark[None, :], density.shape)
        factor = conditional_single_link_logfactor(
            density, group_association, mark_matrix)
        if return_eta_moments:
            safe = jnp.where(density > 0., density, 1.)
            base = jnp.where(density > 0., jnp.log(safe), -jnp.inf) + group_association
            probability = jnp.exp(base-logsumexp(base))
            eta_bar = jnp.sum(probability*eta[None, :])
            eta_variance = jnp.sum(probability*(eta[None, :]-eta_bar)**2)
            if return_eta_mixture:
                log_weight = logsumexp(base, axis=0)-logsumexp(base)
                return factor, jnp.sum(density, axis=1), eta_bar, eta_variance, eta, log_weight
            return factor, jnp.sum(density, axis=1), eta_bar, eta_variance
        return factor, jnp.sum(density, axis=1)

    return jax.vmap(one_group)(
        ids, active, association, voxels, radius, dz, mean, std, alpha)


def cached_eta_mixture_logfactors(eta, log_weight, mean, std, alpha, zero_dex):
    """Exact fixed-field FP factors; ONE caller-owned shared zero coordinate.

    Inputs eta/log_weight are (rows,candidates), already conditioned on count
    key/redshift/association but NOT this FP measurement. A row-specific zero
    or re-estimated association is not implied. No Gaussian approximation to
    the source PDF or to the candidate mixture is made.
    """
    marks = fp_log_likelihood_ratio(eta+zero_dex, 0., mean[:, None],
                                    std[:, None], alpha[:, None])
    return logsumexp(log_weight+marks, axis=-1)
