"""Differentiable counterpart of :mod:`cf4_2mpp_joint_likelihood_local`.

The NumPy implementation is the transparent reference oracle.  This module
contains only JAX-traceable array operations for use by a future HMC/SMC
driver; it performs no I/O and deliberately leaves host-side schema checks to
the caller.
"""

from __future__ import annotations

import math

try:  # JAX is optional in the lightweight local test environment.
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln
except ModuleNotFoundError:  # pragma: no cover - exercised by import smoke
    jax = None
    jnp = None
    gammaln = None


POPULATIONS = 6
VELOCITY_CONVENTION = "physical_peculiar_km_s_observer_subtracted"


def _gaussian_hermite_rule(order: int):
    if not isinstance(order, int) or order < 3 or order % 2 == 0 or order > 15:
        raise ValueError("quadrature order must be an odd integer in [3, 15]")
    import numpy as np
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return tuple(float(value) for value in (2.0**0.5 * nodes)), tuple(float(value) for value in (weights / math.sqrt(math.pi)))


class JaxUnavailable(RuntimeError):
    """JAX is required for the differentiable likelihood path."""


def _require_jax() -> None:
    if jax is None or jnp is None:
        raise JaxUnavailable("JAX is not installed; use the NumPy reference oracle")


def _require_jax_x64() -> None:
    """Require the precision mode needed for NumPy-oracle comparisons."""

    _require_jax()
    if not bool(getattr(jax.config, "x64_enabled", False)):
        raise JaxUnavailable(
            "JAX x64 is disabled; enable jax_enable_x64 before oracle comparison"
        )


def population_log_masses_from_eta_jax(eta_at_sources, alpha, log_bias):
    """Differentiable population log masses; callers choose stable exponentiation."""

    _require_jax()
    bias = jnp.exp(log_bias)
    return alpha[:, None] + bias[:, None] * eta_at_sources[None, :]


def population_masses_from_eta_jax(eta_at_sources, alpha, log_bias):
    """Differentiable positive population source masses."""

    return jnp.exp(population_log_masses_from_eta_jax(eta_at_sources, alpha, log_bias))


def observer_centred_spherical_rsd_jax(
    positions,
    velocities_km_s,
    observer,
    box_size_cMpc_h,
    hubble_km_s_Mpc,
    *,
    little_h,
    scale_factor,
    velocity_convention=VELOCITY_CONVENTION,
):
    """Traceable spherical RSD map with explicit cMpc/h conversion."""

    _require_jax()
    if velocity_convention != VELOCITY_CONVENTION:
        raise ValueError("velocity_convention must be observer-subtracted physical peculiar km/s")
    relative = (positions - observer + box_size_cMpc_h / 2.0) % box_size_cMpc_h
    relative -= box_size_cMpc_h / 2.0
    radius = jnp.linalg.norm(relative, axis=1)
    # Keep the traceable kernel finite at an observer-coincident source;
    # host-side input validation should reject that degenerate geometry.
    safe_radius = jnp.maximum(radius, jnp.finfo(relative.dtype).tiny)
    rhat = jnp.where((radius > 0.0)[:, None], relative / safe_radius[:, None], 0.0)
    radial_velocity = jnp.sum(velocities_km_s * rhat, axis=1)
    displacement = little_h * radial_velocity / (scale_factor * hubble_km_s_Mpc)
    shifted = (positions + displacement[:, None] * rhat) % box_size_cMpc_h
    return shifted, displacement, rhat


def tsc_deposit_jax(positions, masses, grid_size, box_size_cMpc_h):
    """Periodic TSC deposit implemented with differentiable indexed updates."""

    _require_jax()
    spacing = box_size_cMpc_h / grid_size
    cell = (positions % box_size_cMpc_h) / spacing - 0.5
    nearest = jnp.floor(cell + 0.5).astype(jnp.int32)
    offset = cell - nearest

    def weights(component):
        return (
            0.5 * (0.5 - component) ** 2,
            0.75 - component**2,
            0.5 * (0.5 + component) ** 2,
        )

    wx, wy, wz = (weights(offset[:, axis]) for axis in range(3))
    result = jnp.zeros((grid_size, grid_size, grid_size), dtype=masses.dtype)
    for ix, dx in enumerate((-1, 0, 1)):
        for iy, dy in enumerate((-1, 0, 1)):
            for iz, dz in enumerate((-1, 0, 1)):
                weight = wx[ix] * wy[iy] * wz[iz]
                result = result.at[
                    (nearest[:, 0] + dx) % grid_size,
                    (nearest[:, 1] + dy) % grid_size,
                    (nearest[:, 2] + dz) % grid_size,
                ].add(masses * weight)
    return result


def predict_selected_intensity_jax(
    source_positions,
    source_velocities_km_s,
    population_masses,
    selection_exposure,
    *,
    observer,
    box_size_cMpc_h,
    hubble_km_s_Mpc,
    little_h,
    scale_factor,
    sigma_fog_km_s,
    sigma_redshift_km_s,
    velocity_convention=VELOCITY_CONVENTION,
    quadrature_order=3,
):
    """Differentiable selected intensity using the same GH3/TSC oracle contract."""

    _require_jax()
    shifted, _displacement, rhat = observer_centred_spherical_rsd_jax(
        source_positions,
        source_velocities_km_s,
        observer,
        box_size_cMpc_h,
        hubble_km_s_Mpc,
        little_h=little_h,
        scale_factor=scale_factor,
        velocity_convention=velocity_convention,
    )
    grid_size = selection_exposure.shape[1]
    total_sigma = jnp.hypot(sigma_fog_km_s, sigma_redshift_km_s)
    quadrature_nodes, quadrature_weights = _gaussian_hermite_rule(quadrature_order)
    rows = []
    for population in range(POPULATIONS):
        deposited = jnp.zeros((grid_size, grid_size, grid_size), dtype=population_masses.dtype)
        for node, weight in zip(quadrature_nodes, quadrature_weights):
            extra = node * little_h * total_sigma[population] / (scale_factor * hubble_km_s_Mpc)
            positions_node = (shifted + extra * rhat) % box_size_cMpc_h
            deposited = deposited + weight * tsc_deposit_jax(
                positions_node,
                population_masses[population],
                grid_size,
                box_size_cMpc_h,
            )
        rows.append(selection_exposure[population] * deposited)
    return jnp.stack(rows)


def poisson_log_likelihood_jax(counts, intensity):
    """Differentiable exact Poisson log likelihood (caller validates support)."""

    _require_jax()
    # Avoid differentiating through log(0): positive counts at zero intensity
    # still evaluate to a very negative finite value, with finite gradients.
    log_intensity = jnp.log(jnp.maximum(intensity, jnp.finfo(intensity.dtype).tiny))
    return poisson_log_likelihood_from_log_intensity_jax(counts, log_intensity)


def poisson_log_likelihood_from_log_intensity_jax(counts, log_intensity):
    """Differentiable Poisson likelihood in log space."""

    expected = jnp.where(jnp.isneginf(log_intensity), 0.0, jnp.exp(log_intensity))
    count_term = jnp.where(counts > 0, counts * log_intensity, 0.0)
    return jnp.sum(count_term - expected - gammaln(counts + 1.0))


def shared_redshift_log_likelihood_jax(
    observed_km_s,
    predicted_km_s,
    measurement_sigma_km_s,
    group_ids,
    shared_sigma_km_s,
):
    """Differentiable Sherman–Morrison marginalization for shared group latents."""

    _require_jax()
    residual = observed_km_s - predicted_km_s
    result = jnp.asarray(0.0, dtype=observed_km_s.dtype)
    for group in range(shared_sigma_km_s.shape[0]):
        mask = group_ids == group
        variance = jnp.where(mask, measurement_sigma_km_s**2, 1.0)
        inv_diag = jnp.where(mask, 1.0 / variance, 0.0)
        contraction = jnp.sum(inv_diag)
        weighted = jnp.sum(jnp.where(mask, residual * inv_diag, 0.0))
        quadratic = jnp.sum(jnp.where(mask, residual**2 * inv_diag, 0.0))
        shared_variance = shared_sigma_km_s[group] ** 2
        denominator = 1.0 + shared_variance * contraction
        quadratic = quadratic - shared_variance * weighted**2 / denominator
        logdet = jnp.sum(jnp.where(mask, jnp.log(variance), 0.0)) + jnp.log(denominator)
        count = jnp.sum(mask)
        result = result - 0.5 * (quadratic + logdet + count * math.log(2.0 * math.pi))
    return result


def joint_log_likelihood_jax(
    counts,
    intensity,
    observed_km_s,
    predicted_km_s,
    measurement_sigma_km_s,
    group_ids,
    shared_sigma_km_s,
):
    """Differentiable sum of the count and single shared-redshift factors."""

    return poisson_log_likelihood_jax(counts, intensity) + shared_redshift_log_likelihood_jax(
        observed_km_s,
        predicted_km_s,
        measurement_sigma_km_s,
        group_ids,
        shared_sigma_km_s,
    )


def joint_log_likelihood_jax_checked(
    counts,
    intensity,
    observed_km_s,
    predicted_km_s,
    measurement_sigma_km_s,
    group_ids,
    shared_sigma_km_s,
    *,
    secure_object_ids,
    twompp_object_ids=None,
    crossmatch_manifest=None,
    crossmatch_mapping_path=None,
    crossmatch_summary_path=None,
    independent_twompp_redshift_ids=(),
):
    """Call the JAX kernel only after the host-side ownership contract passes.

    JAX kernels intentionally contain only traceable array operations and cannot
    safely enforce object identity or factor ownership inside ``jit``.  This
    wrapper is the single public bridge for inference code: it validates the
    count/redshift shapes and the no-double-counting rule with the NumPy oracle,
    then converts inputs to JAX arrays and calls the differentiable kernel.  A
    caller cannot satisfy this bridge with arbitrary unique labels: the labels,
    2M++ IDs, and integer group indices must reproduce the source-bound
    canonical manifest exactly.
    """

    import numpy as np

    from cf4_2mpp_joint_likelihood_local import (
        LikelihoodInputError,
        validate_factor_ownership,
    )

    secure_ids = tuple(secure_object_ids)
    independent_ids = tuple(independent_twompp_redshift_ids)
    observed = np.asarray(counts)
    expected = np.asarray(intensity, dtype=np.float64)
    redshift_observed = np.asarray(observed_km_s, dtype=np.float64)
    redshift_predicted = np.asarray(predicted_km_s, dtype=np.float64)
    redshift_sigma = np.asarray(measurement_sigma_km_s, dtype=np.float64)
    groups = np.asarray(group_ids)
    shared_sigma = np.asarray(shared_sigma_km_s, dtype=np.float64)
    if observed.ndim != 4 or observed.shape[0] != POPULATIONS:
        raise LikelihoodInputError("counts must have shape (6, N, N, N)")
    if observed.dtype != np.dtype(np.int64):
        raise LikelihoodInputError("counts must have exact int64 dtype")
    if expected.shape != observed.shape:
        raise LikelihoodInputError("intensity shape must match counts")
    if redshift_observed.ndim != 1 or redshift_observed.size == 0:
        raise LikelihoodInputError("observed_km_s must be a non-empty vector")
    if (
        redshift_predicted.shape != redshift_observed.shape
        or redshift_sigma.shape != redshift_observed.shape
        or groups.shape != redshift_observed.shape
        or groups.dtype.kind not in "iu"
        or shared_sigma.ndim != 1
    ):
        raise LikelihoodInputError(
            "redshift arrays and integer group_ids must have aligned one-dimensional shapes"
        )
    validate_factor_ownership(
        secure_ids,
        groups,
        independent_twompp_redshift_ids=independent_ids,
    )
    if len(secure_ids) != redshift_observed.size:
        raise LikelihoodInputError("secure_object_ids must align with redshift observations")
    if (
        np.any(observed < 0)
        or not np.all(np.isfinite(expected))
        or np.any(expected < 0.0)
        or np.any((observed > 0) & (expected <= 0.0))
        or not np.all(np.isfinite(redshift_observed))
        or not np.all(np.isfinite(redshift_predicted))
        or not np.all(np.isfinite(redshift_sigma))
        or not np.all(np.isfinite(shared_sigma))
        or np.any(redshift_sigma <= 0.0)
        or np.any(shared_sigma < 0.0)
        or np.any(groups < 0)
        or np.any(groups >= shared_sigma.size)
    ):
        raise LikelihoodInputError("JAX joint inputs violate finite, support, or group-index contract")
    if crossmatch_manifest is None or crossmatch_mapping_path is None or crossmatch_summary_path is None:
        raise LikelihoodInputError(
            "a source-bound crossmatch manifest and its canonical source paths are required"
        )
    if twompp_object_ids is None:
        raise LikelihoodInputError("twompp_object_ids aligned with secure_object_ids are required")
    from cf4_2mpp_crossmatch_manifest import validate_secure_crossmatch_manifest

    validated_manifest = validate_secure_crossmatch_manifest(
        crossmatch_manifest,
        mapping_path=crossmatch_mapping_path,
        summary_path=crossmatch_summary_path,
    )
    entries = validated_manifest["entries"]
    manifest_secure_ids = tuple(str(entry["secure_object_id"]) for entry in entries)
    manifest_twompp_ids = tuple(str(entry["twompp_object_id"]) for entry in entries)
    manifest_group_indices = np.asarray(
        [int(entry["group_index"]) for entry in entries], dtype=groups.dtype
    )
    supplied_twompp_ids = tuple(str(value).strip() for value in twompp_object_ids)
    if secure_ids != manifest_secure_ids:
        raise LikelihoodInputError("secure_object_ids are not the canonical manifest order")
    if supplied_twompp_ids != manifest_twompp_ids:
        raise LikelihoodInputError("twompp_object_ids are not the canonical manifest order")
    if groups.shape != manifest_group_indices.shape or not np.array_equal(groups, manifest_group_indices):
        raise LikelihoodInputError("group_ids are not the canonical manifest group indices")
    expected_group_count = int(validated_manifest["counts"]["secure_cf4_groups"])
    if shared_sigma.size != expected_group_count:
        raise LikelihoodInputError("shared_sigma size must equal the manifest group count")
    if not np.array_equal(
        np.unique(groups), np.arange(expected_group_count, dtype=groups.dtype)
    ):
        raise LikelihoodInputError("all manifest group indices must be used exactly")
    _require_jax_x64()
    return joint_log_likelihood_jax(
        jnp.asarray(observed),
        jnp.asarray(expected),
        jnp.asarray(redshift_observed),
        jnp.asarray(redshift_predicted),
        jnp.asarray(redshift_sigma),
        jnp.asarray(groups),
        jnp.asarray(shared_sigma),
    )
