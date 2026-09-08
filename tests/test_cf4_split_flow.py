import unittest
import numpy as np
import torch
from cf4_split_moments import roundtrip, encode_tree, decode_tree
from cf4_conditional_split_flow import condition, ConditionalSplitFlow


class SplitFlowTests(unittest.TestCase):
    def test_full_context_preserves_old_sampler_rng(self):
        from cf4_bundle_c_flow_pilot import tensors
        from unittest.mock import patch
        record = (np.zeros((7, 32, 32, 32), np.float32),
                  np.zeros((7, 32, 32, 32), np.uint8),
                  np.zeros((22, 32, 32, 32), np.float32),
                  np.zeros((4, 32, 32, 32), bool))
        a, b = np.random.default_rng(91), np.random.default_rng(91)
        # Test only slicing/RNG here; numerical GPU checks run in the same job.
        with patch.object(torch.Tensor, 'to', lambda self, *args, **kwargs: self):
            cropped = tensors(record, a)
            full = tensors(record, b, full_context=True)
        self.assertEqual(cropped[0].shape[-1], 24)
        self.assertEqual(full[0].shape[-1], 32)
        np.testing.assert_array_equal(a.integers(10000, size=20), b.integers(10000, size=20))

    def test_roundtrip_including_empty_cold_and_single_child(self):
        rng = np.random.default_rng(18)
        mass = rng.uniform(1, 4, (4,)*3)
        v = rng.normal(size=(3, 4, 4, 4))*100
        variance = rng.uniform(10, 100, v.shape)
        mass[:2, :2, :2] = 0
        mass[0, 0, 0] = 1
        variance[:, 2:, 2:, 2:] = 0
        value = np.concatenate([mass[None], mass*v, mass*(v*v+variance)])
        self.assertLess(roundtrip(value).max(), 1e-10)
        self.assertLess(roundtrip(np.zeros_like(value)).max(), 1e-10)

    def test_masked_flow_inverse_and_finite_likelihood(self):
        torch.manual_seed(90)
        model = ConditionalSplitFlow(np.zeros(7), np.ones(7))
        context = torch.randn(1, 22, 3, 3, 3)
        valid = torch.ones(1, 4, 3, 3, 3, dtype=torch.bool)
        z, mask = model.sample(context, valid)
        logp = model.log_prob(z, mask, context, valid)
        self.assertTrue(torch.isfinite(logp).all())
        (-logp.mean()).backward()
        x = torch.randn_like(z)*(mask == 2)
        original = x.clone()
        total = torch.zeros_like(x[:, 0])
        for layer in model.layers:
            # Exercise nonidentity transforms too.
            torch.nn.init.normal_(layer.net[-1].weight, std=.005)
            x, jac = layer(x, context, mask)
            total = total+jac
        for layer in reversed(model.layers):
            x, jac = layer(x, context, mask, inverse=True)
            total = total+jac
        torch.testing.assert_close(x, original, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(total, torch.zeros_like(total), atol=2e-6, rtol=0)


if __name__ == '__main__':
    unittest.main()
