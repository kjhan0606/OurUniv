"""Development continuous-grid tracer response for one evolved PM state.

The diffuse fraction is an explicit, *uncalibrated* unresolved/contaminant
population, not a hidden numerical likelihood floor. It must be constrained
with suitable mocks/observations before an actual-data posterior fit.
"""

import jax.numpy as jnp

from cf4_2mpp_joint_likelihood_jax import predict_selected_intensity_jax


def cell_centres(n, box, origin_fraction=0.5):
    axis = (jnp.arange(n) + origin_fraction) * (box / n)
    return jnp.stack(jnp.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def predict_continuous_intensity(
    density, velocity, exposure, nbar, bias, diffuse_fraction, *,
    box, observer, hubble, little_h, origin_fraction=0.5,
    sigma_fog, sigma_redshift, quadrature_order=3,
):
    """Six selected count intensities with spherical RSD and Gaussian LOS scatter.

    density is nonnegative cell-mean matter density, velocity is (3,N,N,N)
    observer-subtracted physical peculiar km/s, and nbar is *unselected*
    expected count per output cell at unit density. A population's diffuse
    component is spatially uniform before survey selection. Its normalization
    follows the model, never the observed count total.
    """
    n = density.shape[0]
    if density.shape != (n, n, n) or velocity.shape != (3, n, n, n):
        raise ValueError("density/velocity geometry mismatch")
    if exposure.shape != (6, n, n, n):
        raise ValueError("six-population exposure geometry mismatch")
    if nbar.shape != (6,) or bias.shape != (6,) or diffuse_fraction.shape != (6,):
        raise ValueError("six-population parameter geometry mismatch")
    positions = cell_centres(n, box, origin_fraction)
    # CIC-deposited PM cells can be exactly empty. At the fixed published
    # biases (all >=1), rho**b is well-defined without an arbitrary floor.
    # Differentiation with respect to a free b at rho=0 needs a separate
    # smooth occupancy model; this development operator does not assert it.
    response = density.reshape(1, -1) ** bias[:, None]
    response = response / jnp.mean(response, axis=1, keepdims=True)
    clustered_mass = nbar[:, None] * (1.0 - diffuse_fraction[:, None]) * response
    clustered = predict_selected_intensity_jax(
        positions, jnp.moveaxis(velocity, 0, -1).reshape(-1, 3),
        clustered_mass, jnp.ones_like(exposure), observer=observer,
        box_size_cMpc_h=box, hubble_km_s_Mpc=hubble, little_h=little_h,
        scale_factor=1.0, sigma_fog_km_s=sigma_fog,
        sigma_redshift_km_s=sigma_redshift, quadrature_order=quadrature_order,
    )
    # Positive intensity wherever survey exposure is positive, even when
    # finite GH/TSC quadrature leaves a particular target cell unoccupied.
    return exposure * (clustered + nbar[:, None, None, None] * diffuse_fraction[:, None, None, None])
