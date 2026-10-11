import unittest
import numpy as np
from cf4_continuous_matter import restrict, check_realizable
from cf4_spatial_copula import conditional_moments, Marginals, spectral_draw
from cf4_bundle_c_spatial_model import overlaps


class SpatialCopulaTests(unittest.TestCase):
    def test_conditional_moments_and_periodic_overlap(self):
        rng = np.random.default_rng(91)
        mass = rng.uniform(1, 3, (2,)*3)
        mean = rng.normal(size=(3, 2, 2, 2))
        variance = rng.uniform(.5, 2, (3, 2, 2, 2))
        coarse = np.concatenate([mass[None], mass*mean, mass*(mean**2+variance)])
        channels = rng.normal(size=(5, 4, 4, 4))
        channels[0] = np.exp(channels[0])
        channels[4] = np.exp(channels[4])
        value = conditional_moments(channels, coarse, ratio=2)
        check_realizable(value)
        np.testing.assert_allclose(restrict(value, 2), coarse, atol=1e-12)
        self.assertTrue(overlaps([70, 0, 0], [0, 0, 0]))
        self.assertFalse(overlaps([0, 0, 0], [24, 0, 0]))
        with self.assertRaises(ValueError):
            conditional_moments(np.zeros_like(channels), coarse, ratio=2)

    def test_empirical_atoms_and_spectral_cross_channel(self):
        rng = np.random.default_rng(25)
        knots = np.tile(np.r_[np.zeros(5), np.linspace(.1, 2, 12)], (5, 1))
        marginal = Marginals(knots)
        z = marginal.gaussianize(np.zeros((5, 4, 4, 4)), rng)
        self.assertTrue(np.isfinite(z).all())
        self.assertGreater(np.std(z[0]), 0)
        self.assertTrue(np.all(marginal.invert(z) >= 0))
        f = np.fft.rfftn(rng.normal(size=(4, 4, 4)), norm='ortho')
        factor = np.tile(f, (5, 1, 1, 1))
        out = spectral_draw([factor], abs(factor)**2, rng, (4,)*3, shrink=0)
        np.testing.assert_allclose(out[0], out[4], atol=1e-12)
        np.testing.assert_allclose(out.mean(axis=(1, 2, 3)), 0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
