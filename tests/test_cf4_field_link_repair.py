import unittest
import numpy as np
import torch
from cf4_field_link_repair import OctetLift, FinePath, LinkedContinuous, budget_weights
from cf4_stable_field import Branch
from test_cf4_spatial_diffusion import fixture
from cf4_continuous_matter import restrict


class FieldLinkTests(unittest.TestCase):
    def test_orthogonal_spatial_lift_and_gradient(self):
        lift = OctetLift()
        z = torch.randn(1, 49, 3, 4, 5, dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(lift.project(lift(z)), z)
        torch.testing.assert_close(lift(z).square().sum(), z.square().sum())
        lift(z).square().sum().backward()
        torch.testing.assert_close(z.grad, 2*z)

    def test_zero_change_initialization_and_cross_parent_path(self):
        torch.manual_seed(91313)
        base = Branch(True)
        linked = LinkedContinuous(base)
        self.assertEqual(sum(p.numel() for p in linked.fine.parameters()), 37991)
        z = torch.randn(1, 49, 4, 4, 4, requires_grad=True)
        args = (z, torch.full_like(z, 2, dtype=torch.long), torch.randn(1, 10, 4, 4, 4), torch.tensor([50]), torch.tensor([0.]))
        torch.testing.assert_close(linked(*args), base(*args), rtol=0, atol=0)
        with torch.no_grad():
            linked.fine.head.weight.normal_(0, .1)
        output = linked.fine(z, *args[2:])
        output[0, 0, 1, 1, 1].backward()
        self.assertGreater(float(z.grad[0, :, 2, 1, 1].abs().sum()), 0.)

    def test_positive_parent_weights_unit_mean_and_cold_limit(self):
        parent = restrict(fixture(), 2)
        weights = budget_weights(parent)
        self.assertEqual(weights.shape, (49,)+parent.shape[1:])
        self.assertTrue(np.isfinite(weights).all() and (weights > 0).all())
        np.testing.assert_allclose(weights.mean(axis=(1, 2, 3)), 1., rtol=1e-6)
        parent[1:] = 0
        cold = budget_weights(parent).reshape(7, 7, *parent.shape[1:])
        np.testing.assert_array_equal(cold[:, 1:], 1.)


if __name__ == '__main__':
    unittest.main()
