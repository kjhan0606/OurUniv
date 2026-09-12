import unittest
import healpy as hp
import numpy as np
from scipy.stats import qmc
from cf4_bundle_c_selection_support import NSIDE, footprint_possible, integrate_cells


class SelectionSupportTests(unittest.TestCase):
    def test_constant_volume_and_shell_partition(self):
        maps = [np.ones(hp.nside2npix(NSIDE)) for _ in range(2)]
        radial = np.ones((6, 2))
        centers = np.array([[20., 0., 0.], [30., 0., 0.]])
        points = qmc.Sobol(3, scramble=True, seed=4).random_base2(11) - .5
        result = integrate_cells(centers, 1.5, points, np.eye(3), maps, [5., 180.], radial)
        np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-14)
        self.assertTrue(np.all(result[:, 0, 1] > 0))
        self.assertTrue(np.all(result[:, 1, 1] > 0))

    def test_thin_footprint_without_any_catalog(self):
        # A thin HEALPix strip near a cube face is missed by the old tensor rule.
        center = np.array([[30.75, 2.25, .75]])
        mask = np.zeros(hp.nside2npix(NSIDE))
        ids = hp.query_disc(NSIDE, center[0] / np.linalg.norm(center[0]), .1)
        vec = np.array(hp.pix2vec(NSIDE, ids))
        ratio = vec[1] / vec[0]
        # Leave a full pixel-width gap from the outermost order4 node;
        # selecting pixel centres too close to that node includes its pixel.
        mask[ids[(ratio > .098) & (ratio < .0995)]] = .5
        maps = [mask, mask]
        nodes, _ = np.polynomial.legendre.leggauss(4)
        old = np.array(np.meshgrid(nodes, nodes, nodes, indexing="ij")).reshape(3, -1).T / 2
        args = (np.eye(3), maps, np.array([5., 180.]), np.ones((6, 2)))
        before = integrate_cells(center, 1.5, old, *args)
        points = qmc.Sobol(3, scramble=True, seed=2026090801).random_base2(11) - .5
        after = integrate_cells(center, 1.5, points, *args)
        self.assertEqual(float(before.sum()), 0.)
        self.assertGreater(float(after.sum()), 0.)

    def test_full_and_empty_footprints(self):
        centers = np.array([[30., 1., 1.]])
        for value in (0, 1):
            mask = np.full(hp.nside2npix(NSIDE), value)
            result = footprint_possible(centers, 1.5, np.eye(3), mask, None)
            self.assertEqual(bool(result[0]), bool(value))


if __name__ == "__main__":
    unittest.main()
