import unittest
import numpy as np
from cf4_empirical_field_prior import conditional_weights, mixture_moments
from cf4_bundle_c_total_matter import deposit
from cf4_resolved_moments import particle_moments


class EmpiricalPriorTests(unittest.TestCase):
    def test_total_matter_deposition_keeps_diffuse_and_halo_particles(self):
        rng = np.random.default_rng(72)
        x = np.concatenate([rng.uniform(0, 4, (1000, 3)), rng.normal(0, .08, (100, 3))])
        v = rng.normal(0, 20, x.shape)
        m = rng.uniform(1, 3, len(x))
        acc = np.zeros((7, 16, 16, 16))
        deposit(acc, x[:1000], v[:1000], m[:1000], 4.)
        deposit(acc, x[1000:], v[1000:], m[1000:], 4.)
        direct = particle_moments(x % 4., v, m, lower=[0, 0, 0], dx=.25, n=16)
        np.testing.assert_allclose(acc[0], direct['mass'])
        np.testing.assert_allclose(acc[1:4], direct['momentum'], atol=1e-10)
        np.testing.assert_allclose(acc[4:7], direct['second_moment'], atol=1e-10)
        np.testing.assert_allclose(acc[0].sum(), m.sum())

    def test_conditioning_and_no_support(self):
        features = np.arange(6.)[:, None]
        off = conditional_weights(features, [2.5], [2.])
        on = conditional_weights(features, [2.5], [2.], log_data=[-20, -20, -20, 0, -20, -20])
        np.testing.assert_allclose(off['weights'], off['prior_weights'])
        self.assertGreater(on['weights'][3], .99)
        self.assertEqual(on['status'], 'LIMITED_FINITE_SUPPORT')
        fail = conditional_weights(features, [2.5], [2.], log_data=np.full(6, -np.inf))
        self.assertEqual(fail['status'], 'NO_SUPPORT')
        with self.assertRaises(ValueError):
            conditional_weights(features[:1], [0.], [1.])

    def test_physical_dispersion_is_not_posterior_uncertainty(self):
        mass = np.ones((2, 1))
        v = np.array([[[0.], [0.], [0.]], [[10.], [0.], [0.]]])
        out = mixture_moments(mass, v, v**2 + 4., [.5, .5])
        np.testing.assert_allclose(out['variance_mean_velocity'][:, 0], [25., 0., 0.])
        np.testing.assert_allclose(out['mean_physical_velocity_variance'], 4.)


if __name__ == '__main__':
    unittest.main()
