"""SDSS FP source-likelihood shape and same-state distance prediction.

Howlett+2022 sec.5 uses a flat eta prior on [-1.5,1.5]. Thus its
selection-corrected eta PDF is proportional to the individual likelihood
on that interval, conditional on the published FP fit and selection.
This is NOT a likelihood for CF4 group inclusion or independent redshifts.
Shared FP/zero-point uncertainty and group FoG remain caller responsibilities.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr

from cf4_z0_physical_field import read_centred

C_KM_S = 299792.458


def skew_parameters(mean, std, alpha):
    """Published columns are moments, not skew-normal location and scale."""
    delta = alpha / jnp.sqrt(1.0 + alpha**2)
    scale = std / jnp.sqrt(1.0 - 2.0 * delta**2 / jnp.pi)
    location = mean - scale * delta * jnp.sqrt(2.0 / jnp.pi)
    return location, scale


def fp_log_likelihood_ratio(eta, reference_eta, mean, std, alpha):
    """Source likelihood ratio; source posterior's eta normalizer cancels.

    No eta-to-distance Jacobian: we evaluate a likelihood, not transform
    a probability density over parameters. Both evaluation points must
    remain within the source prior support.
    Do not multiply by another Sn or fn selection correction.
    """
    loc, scale = skew_parameters(mean, std, alpha)
    def shape(x):
        t = (x - loc) / scale
        return -0.5 * t**2 + log_ndtr(alpha * t)
    valid = ((jnp.abs(eta) <= 1.5) & (jnp.abs(reference_eta) <= 1.5)
             & (std > 0) & jnp.isfinite(mean) & jnp.isfinite(alpha))
    return jnp.where(valid, shape(eta) - shape(reference_eta), -jnp.inf)


def predicted_eta(velocity, directions, z_group, z_table, distance_table,
                  amplitude=1.0, box=384.0, iterations=128):
    """Cold single-stream conditional root on the SAME PM velocity grid.

    Positions are observer-centred supergalactic, in cMpc/h; PM origin box/2.
    1+z_group=(1+z_cos)(1+v_rad/c), not additive redshifts.
    Damped iteration plus local Newton refinement is differentiable where
    converged and single-stream. It is not a multi-stream marginalization.
    Returns eta, distance and redshift residual; never clips out-of-box roots.
    This cold control is not a calibrated group velocity/FoG marginalization.
    """
    dz = jnp.interp(z_group, z_table, distance_table)
    def radial(distance):
        x = directions * distance[:, None] + box / 2
        vector = jnp.stack([read_centred(velocity[k], x, box) for k in range(3)], axis=-1)
        return amplitude * jnp.sum(vector * directions, axis=-1)
    def step(_, distance):
        zcos = (1 + z_group) / (1 + radial(distance) / C_KM_S) - 1
        proposed = jnp.interp(zcos, z_table, distance_table, left=jnp.nan, right=jnp.nan)
        # Under-relax the redshift inversion; an undamped iteration can
        # oscillate across adjacent velocity-grid cells even at a valid root.
        return 0.5 * (distance + proposed)
    distance = jax.lax.fori_loop(0, iterations, step, dz)
    def equation(distance):
        zcos = jnp.interp(distance, distance_table, z_table)
        return (1 + zcos) * (1 + radial(distance) / C_KM_S) - 1 - z_group
    def refine(_, distance):
        # Rows are independent: a ones JVP gives the diagonal derivative.
        value, slope = jax.jvp(equation, (distance,), (jnp.ones_like(distance),))
        return distance - value / slope
    distance = jax.lax.fori_loop(0, 4, refine, distance)
    residual = equation(distance)
    return jnp.log10(dz / distance), distance, residual
