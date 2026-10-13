import unittest

import jax.numpy as jnp
import numpy as np

from cf4_r1_resolution_mixing import diagnostics, observation_vector


class ResolutionMixingTest(unittest.TestCase):
    def test_observable_order_units(self):
        m = dict(mass_Msun_h=jnp.array([10., 20., 30.]),
                 offset_cMpc_h=jnp.arange(9.).reshape(3, 3),
                 mean_velocity_km_s=100+jnp.arange(9.).reshape(3, 3))
        v = np.asarray(observation_vector(m, 10.)).reshape(3, 7)
        np.testing.assert_allclose(v[:, 0], np.log([1., 2., 3.]), atol=1e-12)
        np.testing.assert_array_equal(v[:, 1:4], m['offset_cMpc_h'])
        np.testing.assert_array_equal(v[:, 4:7], m['mean_velocity_km_s'])

    def test_projection_diagnostics_detect_shifted_chains(self):
        rng = np.random.default_rng(592)
        x = rng.normal(size=(4, 1024, 2))
        good = diagnostics(x, ['a', 'b'])
        self.assertLess(good['max_Rhat'], 1.02)
        self.assertGreater(good['min_bulk_ESS'], 1000)
        x[0, :, 0] += 3
        self.assertGreater(diagnostics(x, ['a', 'b'])['max_Rhat'], 1.1)


if __name__ == '__main__':
    unittest.main()
