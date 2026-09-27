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


if __name__ == "__main__":
    unittest.main()
