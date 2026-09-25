import numpy as np
import unittest

import jax
import jax.numpy as jnp

from cf4_r2_continuous_tracer import predict_continuous_intensity


def test_continuous_tracer_support_conservation_and_state_gradient():
    jax.config.update("jax_enable_x64", True)
    n = 4
    rho = jnp.ones((n, n, n), dtype=jnp.float64)
    velocity = jnp.zeros((3, n, n, n), dtype=jnp.float64)
    exposure = jnp.ones((6, n, n, n), dtype=jnp.float64)
    nbar = jnp.full((6,), 0.7)
    bias = jnp.ones((6,))
    fraction = jnp.full((6,), 0.03)
    options = dict(box=12., observer=jnp.array([6., 6., 6.]), hubble=75.,
                   little_h=.75, sigma_fog=jnp.zeros(6),
                   sigma_redshift=jnp.zeros(6))

    def model(scale):
        perturbed = rho.at[1, 1, 1].set(1.0 + scale)
        return predict_continuous_intensity(
            perturbed, velocity, exposure, nbar, bias, fraction, **options)

    base = np.asarray(model(0.))
    np.testing.assert_allclose(base, 0.7, atol=1e-12)
    np.testing.assert_allclose(base.sum(axis=(1, 2, 3)), 64 * 0.7, atol=1e-10)
    assert np.all(base >= 0.7 * 0.03)
    value, derivative = jax.value_and_grad(lambda s: model(s)[0, 1, 1, 1])(0.1)
    assert np.isfinite(float(value)) and np.isfinite(float(derivative))
    assert abs(float(derivative)) > 1e-6


def test_geometry_rejects_wrong_population_count():
    with np.testing.assert_raises_regex(ValueError, "six-population"):
        predict_continuous_intensity(
            jnp.ones((2, 2, 2)), jnp.zeros((3, 2, 2, 2)),
            jnp.ones((5, 2, 2, 2)), jnp.ones(6), jnp.ones(6), jnp.ones(6),
            box=12., observer=jnp.ones(3), hubble=75., little_h=.75,
            sigma_fog=jnp.ones(6), sigma_redshift=jnp.ones(6),
        )


class TestContinuousTracer(unittest.TestCase):
    def test_support_and_gradient(self):
        test_continuous_tracer_support_conservation_and_state_gradient()

    def test_geometry(self):
        test_geometry_rejects_wrong_population_count()
