"""Shared diagonal-moment Gaussian-mixture closure; not a calibrated prior.

Widths are km/s, variances km²/s². Mathematical components are not identified
centrals/satellites. Caller validates variance positivity and periodic bounds.
"""
import jax.numpy as jnp
import jax


def conditional_los_sigma(direction, base_sigma, variances, scale):
    if jnp.asarray(variances).shape != direction.shape:
        raise ValueError('one diagonal velocity variance vector per source required')
    return jnp.sqrt(base_sigma**2+scale**2*jnp.sum(direction**2*variances,axis=1))


def mixture_components(closure):
    """Same normalized weights/widths for counts, raw marks and support."""
    fraction=closure['broad_fraction']
    core=closure['core_sigma_km_s']
    return ((1-fraction,core,0.),(fraction,core,closure['dispersion_scale']))


def closure_from_coordinates(coordinates):
    """Physical log-core/log-scale/logit-fraction; caller owns proper priors.

    No native fitted values or invisible empirical priors are injected.
    """
    if jnp.asarray(coordinates).shape!=(3,):
        raise ValueError('three physical closure coordinates required')
    return dict(core_sigma_km_s=jnp.exp(coordinates[0]),
        dispersion_scale=jnp.exp(coordinates[1]),broad_fraction=jax.nn.sigmoid(coordinates[2]))


def active_tracer_coordinates(coordinates):
    """Eight rate/bias/LF coordinates; remove the obsolete global LOS width.

    Restoring a dummy slot is only an adapter to existing LF functions, not
    an extra sampled/prior-penalized coordinate.
    """
    if jnp.asarray(coordinates).shape!=(8,):
        raise ValueError('eight active rate/bias/LF coordinates required')
    return jnp.concatenate((coordinates[:6],jnp.zeros(1),coordinates[6:]))
