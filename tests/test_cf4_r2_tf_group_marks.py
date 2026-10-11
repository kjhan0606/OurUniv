import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r2_tf_group_marks import tf_group_logratios


class TFGroupMarksTest(unittest.TestCase):
    def test_constant_field_and_velocity_response(self):
        rho = jnp.ones((8, 8, 8))
        vel = jnp.zeros((3, 8, 8, 8)).at[0].set(120.)
        direction = jnp.array([[1., 0., 0.]])
        distance = jnp.array([1., 2., 3.])
        weights = jnp.array([.5, 1., .5])
        zcos = jnp.array([.001, .002, .003])
        modulus = jnp.array([5*np.log10((1.002)*2/.746)+25])

        def score(amplitude):
            return tf_group_logratios(rho, vel*amplitude, direction,
                jnp.array([299792.458*.002+120.*1.002]), modulus,
                jnp.array([.25]), distance, weights, zcos, box=8.)[0]

        value, gradient = jax.value_and_grad(score)(1.)
        difference = (score(1.0001)-score(.9999))/.0002
        self.assertTrue(np.isfinite(float(value)))
        self.assertAlmostEqual(float(gradient), float(difference), delta=1e-4)
        self.assertNotAlmostEqual(float(score(0.)), float(value), delta=1e-4)

    def test_nonpositive_reported_error_rejected(self):
        result = tf_group_logratios(jnp.ones((8, 8, 8)), jnp.zeros((3, 8, 8, 8)),
            jnp.array([[1., 0., 0.]]), jnp.array([300.]), jnp.array([30.]),
            jnp.array([0.]), jnp.array([1., 2.]), jnp.array([1., 1.]),
            jnp.array([.001, .002]), box=8.)
        self.assertTrue(np.isnan(float(result[0])))

    def test_groupwise_bias_and_velocity_width_match_scalar_evaluations(self):
        rho = jnp.ones((8, 8, 8)).at[4, :, :].set(2.)
        velocity = jnp.zeros((3, 8, 8, 8))
        direction = jnp.array([[1., 0., 0.], [1., 0., 0.]])
        cz = jnp.array([600., 820.])
        modulus = jnp.array([28., 29.])
        error = jnp.array([.25, .4])
        distance = jnp.array([1., 2., 3.])
        weight = jnp.array([.5, 1., .5])
        zcos = jnp.array([.001, .002, .003])
        bias = jnp.array([.8, 1.3])
        width = jnp.array([100., 250.])
        vector = tf_group_logratios(rho, velocity, direction, cz, modulus,
            error, distance, weight, zcos, box=8., selected_bias=bias,
            sigma_v=width)
        scalar = [tf_group_logratios(rho, velocity, direction[i:i+1],
            cz[i:i+1], modulus[i:i+1], error[i:i+1], distance, weight,
            zcos, box=8., selected_bias=bias[i], sigma_v=width[i])[0]
            for i in range(2)]
        np.testing.assert_allclose(np.asarray(vector), np.asarray(scalar), rtol=1e-6)

    def test_redshift_error_convolution_matches_gaussian_quadrature_width(self):
        rho = jnp.ones((8, 8, 8))
        velocity = jnp.zeros((3, 8, 8, 8))
        direction = jnp.array([[1., 0., 0.], [1., 0., 0.]])
        cz = jnp.array([600., 820.])
        modulus = jnp.array([28., 29.])
        error = jnp.array([.25, .4])
        distance = jnp.array([1., 2., 3.])
        weight = jnp.array([.5, 1., .5])
        zcos = jnp.array([.001, .002, .003])
        fog = jnp.array([100., 250.])
        redshift = jnp.array([35., 50.])
        convolved = tf_group_logratios(rho, velocity, direction, cz, modulus,
            error, distance, weight, zcos, box=8., sigma_v=fog,
            sigma_redshift=redshift)
        width = jnp.hypot(fog, redshift)
        combined = tf_group_logratios(rho, velocity, direction, cz, modulus,
            error, distance, weight, zcos, box=8., sigma_v=width)
        np.testing.assert_allclose(np.asarray(convolved), np.asarray(combined), rtol=1e-12)
        invalid = tf_group_logratios(rho, velocity, direction, cz, modulus,
            error, distance, weight, zcos, box=8., sigma_v=fog,
            sigma_redshift=jnp.array([35., -1.]))
        self.assertTrue(np.isfinite(float(invalid[0])))
        self.assertTrue(np.isnan(float(invalid[1])))


if __name__ == "__main__":
    unittest.main()
