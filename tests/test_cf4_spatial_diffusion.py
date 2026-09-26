import unittest

import numpy as np
import torch

from cf4_continuous_matter import restrict
from cf4_split_moments import encode_tree, state
from cf4_spatial_diffusion import (augment, canonical, pack, unpack,
    categorical_reverse_probs, gaussian_schedule, SpatialDenoiser, diffusion_loss, sample)


def fixture(n=8):
    rng = np.random.default_rng(32)
    mass = rng.uniform(1, 5, (n,)*3)
    velocity = rng.normal(0, 100, (3, n, n, n))
    var = rng.uniform(1, 100, velocity.shape)
    mass[:2, :2, :2] = 0
    mass[0, 0, 0] = 2
    var[:, 4:, 4:, 4:] = 0
    return np.concatenate([mass[None], mass*velocity, mass*(velocity**2+var)])


class SpatialDiffusionTests(unittest.TestCase):
    def test_native_identity_and_legal_branches(self):
        fine = fixture()
        for z, raw, parent in encode_tree(fine):
            np.testing.assert_array_equal(canonical(raw, parent), raw)
        z, raw, root = pack(fine)
        restored, counts = unpack(z, raw, root)
        np.testing.assert_allclose(restored, fine, rtol=2e-10, atol=1e-8)
        self.assertEqual(np.sum(counts['corrected_per_node_channel']), 0)
        for raw_code in range(4):
            legal = canonical(np.full(root.shape, raw_code, np.uint8), root)
            both = legal[0] == 2
            variance = state(root)[1]
            for a in range(3):
                self.assertTrue(np.all(legal[a+1][~both | (variance[a] == 0)] == 0))
                self.assertTrue(np.all(legal[a+4][legal[a+1] != 2] == 0))
        with self.assertRaises(ValueError):
            canonical(np.full_like(root, 4), root)
        with self.assertRaises(ValueError):
            unpack(np.full_like(z, np.nan), raw, root)

    def test_signed_symmetries_and_coarse_restriction(self):
        fine = fixture()
        from cf4_spatial_diffusion import SYMMETRIES
        self.assertEqual(len(set(SYMMETRIES)), 48)
        for i, (axes, signs) in enumerate(SYMMETRIES):
            transformed = augment(fine, i)
            np.testing.assert_allclose(restrict(transformed, 2), augment(restrict(fine, 2), i), rtol=1e-12, atol=1e-9)
            sums = fine.sum(axis=(1, 2, 3))
            expected = np.r_[sums[0], sums[1:4][list(axes)]*signs, sums[4:][list(axes)]]
            np.testing.assert_allclose(transformed.sum(axis=(1, 2, 3)), expected, rtol=1e-12, atol=1e-9)

    def test_absorbing_kernel_against_bayes(self):
        clean = 2
        for t in (1, 2, 37, 100):
            # Enumerate five states using q_t=I with absorption beta=1/(T-t+1).
            beta = 1/(101-t)
            qt = np.eye(5)*(1-beta)
            qt[:, 4] += beta
            prior = np.zeros(5)
            prior[clean], prior[4] = 1-(t-1)/100, (t-1)/100
            posterior = prior*qt[:, 4]
            posterior /= posterior.sum()
            expected = np.zeros(5)
            expected[clean], expected[4] = 1/t, 1-1/t
            np.testing.assert_allclose(posterior, expected, atol=1e-14)
            logits = torch.zeros(1, 1, 4, 1, 1, 1)
            current = torch.full((1, 1, 1, 1, 1), 4)
            probs = categorical_reverse_probs(logits, current, t)
            torch.testing.assert_close(probs.sum(2), torch.ones_like(current).float())
            self.assertAlmostEqual(float(probs[0, 0, 0, 0, 0, 0]), .25/t, places=7)
            current.zero_()
            self.assertEqual(float(categorical_reverse_probs(logits, current, t)[0, 0, 0, 0, 0, 0]), 1)
        beta, abar = gaussian_schedule(100)
        self.assertTrue(bool(((beta > 0) & (beta < 1)).all()))
        self.assertTrue(bool((abar[1:] < abar[:-1]).all()))

    def test_actual_network_backward_and_joint_sampling(self):
        torch.manual_seed(924)
        model = SpatialDenoiser((8, 16, 32, 64))
        clean = torch.randn(1, 49, 8, 8, 8)
        code = torch.randint(0, 4, clean.shape)
        coarse = torch.randn(1, 7, 8, 8, 8)
        loss, report = diffusion_loss(model, clean, code, coarse, torch.zeros(1), 4)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        z, m = sample(model, coarse, .1875, np.zeros(49), np.ones(49), steps=4)
        self.assertEqual(z.shape, (49, 8, 8, 8))
        self.assertTrue(np.isfinite(z).all())
        self.assertTrue(np.isin(m, range(4)).all())
        self.assertGreater(report['categorical'], 0)


if __name__ == '__main__':
    unittest.main()
