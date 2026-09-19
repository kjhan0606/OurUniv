"""Differentiable state-dependent Q1 LOS response candidate.

This is a bounded development candidate, not a production operator.  It
integrates the periodic TSC response with fixed Gauss--Legendre nodes in the
standard-normal LOS coordinate, so positions, velocities and sigma remain
JAX-traceable.  The sealed NumPy interval-integrated oracle remains the value
authority; callers must apply the value and gradient gates before promotion.
"""

from __future__ import annotations

import math
import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jax = None
    jnp = None

from cf4_2mpp_joint_likelihood_jax import (
    POPULATIONS,
    VELOCITY_CONVENTION,
    _require_jax,
    observer_centred_spherical_rsd_jax,
)


class StateOperatorInputError(ValueError):
    """Invalid state-dependent Q1 candidate input."""


def _nodes_weights(order: int, tail_cutoff: float):
    if order < 16 or order % 2:
        raise StateOperatorInputError("quadrature_order must be an even integer >= 16")
    if not math.isfinite(tail_cutoff) or tail_cutoff <= 0.0:
        raise StateOperatorInputError("tail_cutoff must be positive and finite")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    epsilon = tail_cutoff * nodes
    normal = np.exp(-0.5 * epsilon * epsilon) / math.sqrt(2.0 * math.pi)
    weight = tail_cutoff * weights * normal
    weight /= weight.sum()
    return epsilon.astype(np.float64), weight.astype(np.float64)


def _single_tsc(position, mass, grid_size: int, box_size):
    spacing = box_size / grid_size
    cell = (jnp.mod(position, box_size) / spacing) - 0.5
    nearest = jnp.floor(cell + 0.5).astype(jnp.int32)
    offset = cell - nearest

    def weights(component):
        return (0.5 * (0.5 - component) ** 2, 0.75 - component**2,
                0.5 * (0.5 + component) ** 2)

    wx, wy, wz = (weights(offset[axis]) for axis in range(3))
    field = jnp.zeros((grid_size, grid_size, grid_size), dtype=jnp.asarray(mass).dtype)
    for ix, dx in enumerate((-1, 0, 1)):
        for iy, dy in enumerate((-1, 0, 1)):
            for iz, dz in enumerate((-1, 0, 1)):
                field = field.at[
                    (nearest[0] + dx) % grid_size,
                    (nearest[1] + dy) % grid_size,
                    (nearest[2] + dz) % grid_size,
                ].add(mass * wx[ix] * wy[iy] * wz[iz])
    return field


def _particle_expectation(position, mass, rhat, displacement_scale,
                          grid_size, box_size, epsilon, weights):
    field = jnp.zeros((grid_size, grid_size, grid_size), dtype=jnp.asarray(mass).dtype)

    def body(index, carry):
        shifted = position + epsilon[index] * displacement_scale * rhat
        return carry + weights[index] * _single_tsc(shifted, mass, grid_size, box_size)

    return jax.lax.fori_loop(0, epsilon.shape[0], body, field)


def predict_selected_intensity_state_jax(
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
    quadrature_order=64,
    tail_cutoff=8.0,
):
    """Return a differentiable state-dependent Q1 intensity candidate."""
    _require_jax()
    positions = jnp.asarray(source_positions, dtype=jnp.float64)
    velocities = jnp.asarray(source_velocities_km_s, dtype=jnp.float64)
    masses = jnp.asarray(population_masses, dtype=jnp.float64)
    exposure = jnp.asarray(selection_exposure, dtype=jnp.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise StateOperatorInputError("source_positions must have shape (M,3)")
    if velocities.shape != positions.shape or masses.shape != (POPULATIONS, positions.shape[0]):
        raise StateOperatorInputError("state arrays have inconsistent shapes")
    if exposure.ndim != 4 or exposure.shape[0] != POPULATIONS:
        raise StateOperatorInputError("selection_exposure must have shape (6,N,N,N)")
    epsilon, weights = _nodes_weights(quadrature_order, tail_cutoff)
    epsilon = jnp.asarray(epsilon)
    weights = jnp.asarray(weights)
    shifted, _displacement, rhat = observer_centred_spherical_rsd_jax(
        positions, velocities, observer, box_size_cMpc_h, hubble_km_s_Mpc,
        little_h=little_h, scale_factor=scale_factor,
        velocity_convention=velocity_convention,
    )
    total_sigma = jnp.hypot(jnp.asarray(sigma_fog_km_s), jnp.asarray(sigma_redshift_km_s))
    scale = little_h * total_sigma / (scale_factor * hubble_km_s_Mpc)
    rows = []
    for population in range(POPULATIONS):
        field = jnp.zeros(exposure.shape[1:], dtype=jnp.float64)
        for source in range(positions.shape[0]):
            field = field + _particle_expectation(
                shifted[source], masses[population, source], rhat[source], scale[population],
                exposure.shape[1], box_size_cMpc_h, epsilon, weights,
            )
        rows.append(exposure[population] * field)
    return jnp.stack(rows)

