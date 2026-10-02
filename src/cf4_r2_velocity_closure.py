"""Shared diagonal-moment Gaussian-mixture closure; not a calibrated prior.

Widths are km/s, variances km²/s². Mathematical components are not identified
centrals/satellites. Caller validates variance positivity and periodic bounds.
"""
import jax.numpy as jnp


def conditional_los_sigma(direction, base_sigma, variances, scale):
    if jnp.asarray(variances).shape != direction.shape:
        raise ValueError('one diagonal velocity variance vector per source required')
    return jnp.sqrt(base_sigma**2+scale**2*jnp.sum(direction**2*variances,axis=1))


def mixture_components(closure):
    """Same normalized weights/widths for counts, raw marks and support."""
    fraction=closure['broad_fraction']
    core=closure['core_sigma_km_s']
    return ((1-fraction,core,0.),(fraction,core,closure['dispersion_scale']))
