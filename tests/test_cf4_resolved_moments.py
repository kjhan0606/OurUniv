import unittest
import numpy as np
from cf4_resolved_moments import particle_moments, aggregate, derived, resolved_catalog


class ResolvedTests(unittest.TestCase):
    def test_mass_momentum_and_total_variance(self):
        x = np.array([[.1, .1, .1], [.2, .1, .1], [1.1, .1, .1]])
        v = np.array([[1., 2., 3.], [3., 4., 5.], [9., 8., 7.]])
        m = np.array([1., 3., 2.])
        fine = particle_moments(x, v, m, lower=[0, 0, 0], dx=1, n=2)
        coarse = aggregate(fine, 2)
        np.testing.assert_allclose(coarse['mass'].sum(), m.sum())
        np.testing.assert_allclose(coarse['momentum'].sum(axis=(1, 2, 3)), (m[:, None] * v).sum(axis=0))
        result = derived(coarse, 2)
        mean = np.average(v, weights=m, axis=0)
        np.testing.assert_allclose(result['mean_velocity_km_s'][:, 0, 0, 0], mean)
        np.testing.assert_allclose(result['sigma_velocity_km_s'][:, 0, 0, 0]**2,
                                   np.average((v - mean)**2, weights=m, axis=0))
        self.assertTrue(np.isnan(derived(fine, 1)['mean_velocity_km_s'][:, 1, 1, 1]).all())

    def test_bound_mass_is_not_host_m200_and_periodic_units(self):
        cat = resolved_catalog(np.array([[99., 0, 0], [1., 0, 0], [2., 0, 0]]),
            np.zeros((3, 3)), np.array([10., 2., 1.]), np.zeros(3, dtype=int),
            np.array([0]), np.array([20.]), dict(MW=0, M31=1, M33=2),
            h=.5, a=1, box_ckpc_h=100, rotation=np.eye(3), frame='test')
        np.testing.assert_allclose(cat['M31']['position_kpc'], [4, 0, 0])
        self.assertEqual(cat['MW']['host_M200c_Msun'], 4e11)
        self.assertIsNone(cat['M33']['host_M200c_Msun'])
        self.assertEqual(cat['M33']['bound_mass_Msun'], 2e10)


if __name__ == '__main__':
    unittest.main()
