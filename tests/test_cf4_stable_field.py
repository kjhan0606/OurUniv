import unittest
import numpy as np
import torch

from cf4_stable_field import StableField, noisy_record, reverse_mean, conditioning
from cf4_spatial_diffusion import gaussian_schedule, pack, unpack, augment
from cf4_role_locations import transform_positions
from test_cf4_spatial_diffusion import fixture


class StableFieldTests(unittest.TestCase):
    def test_v_identities_and_nonamplifying_reference(self):
        torch.manual_seed(912)
        clean, noise = torch.randn(1, 49, 4, 4, 4), torch.randn(1, 49, 4, 4, 4)
        codes = torch.full(clean.shape, 2)
        beta, abar = gaussian_schedule(100)
        self.assertEqual(float(abar[0]), 1.)
        self.assertEqual(float(beta[0]*(1-abar[0])/(1-abar[1])), 0.)
        for timestep in (1, 10, 50, 100):
            t = torch.tensor([timestep])
            z, _, v, _ = noisy_record(clean, codes, t, noise=noise)
            a = abar[timestep]
            torch.testing.assert_close(a.sqrt()*z-(1-a).sqrt()*v, clean, rtol=2e-5, atol=1e-6)
            eps = (1-a).sqrt()*z+a.sqrt()*v
            torch.testing.assert_close(eps, noise, rtol=2e-5, atol=1e-6)
            old = (z-beta[timestep-1]/(1-a).sqrt()*eps)/(1-beta[timestep-1]).sqrt()
            stable = reverse_mean(z, v, beta[timestep-1], a, abar[timestep-1])
            torch.testing.assert_close(stable, old, rtol=1e-3, atol=1e-5)
        variance = 1.
        for t in range(100, 0, -1):
            variance = (1-float(beta[t-1]))*variance+(float(beta[t-1]*(1-abar[t-1])/(1-abar[t])) if t > 1 else 0.)
            self.assertLessEqual(variance, 1.+1e-7)
            self.assertGreaterEqual(variance, 0.)

    def test_exact_size_zero_start_and_separate_gradients(self):
        model = StableField()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1490406)
        noisy = torch.randn(1, 49, 4, 4, 4)
        codes = torch.full(noisy.shape, 4)
        inputs = (noisy, codes, torch.randn(1, 10, 4, 4, 4), torch.tensor([50]), torch.tensor([0.]))
        v, logits = model(*inputs)
        self.assertEqual(float(v.detach().abs().max()), 0.)
        self.assertEqual(float(logits.detach().abs().max()), 0.)
        (v-torch.ones_like(v)).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.categorical.parameters()))
        self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any()) for p in model.continuous.parameters()))
        model.zero_grad(set_to_none=True)
        logits = model.categorical(*inputs)
        (-logits.log_softmax(2)[:, :, 0].mean()).backward()
        self.assertTrue(all(p.grad is None for p in model.continuous.parameters()))

    def test_native_chart_and_observer_augmentation(self):
        native = fixture()
        observer = torch.tensor([5., 5., 5.])
        for symmetry in range(48):
            changed = augment(native, symmetry)
            z, codes, parent = pack(changed)
            restored, _ = unpack(z, codes, parent)
            np.testing.assert_allclose(restored, changed, rtol=1e-9, atol=1e-7)
            point = transform_positions(observer, symmetry, 8).numpy()*.1875
            value = conditioning(parent, .1875, point)
            self.assertEqual(value.shape, (10, 4, 4, 4))
            self.assertTrue(np.isfinite(value).all())
            expected = (np.array([.5, .5, .5])*.375-point)/12
            np.testing.assert_allclose(value[-3:, 0, 0, 0], expected, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
