import unittest
import numpy as np
import torch
from cf4_member_mass_readout import MemberMassNet, features, transform, map_loss, metrics
from cf4_spatial_diffusion import augment, SYMMETRIES
from test_cf4_spatial_diffusion import fixture


class MemberMassTests(unittest.TestCase):
    def test_allocation_loss_and_backward(self):
        torch.manual_seed(17)
        model = MemberMassNet()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 530804)
        mass = torch.ones(1, 1, 8, 8, 8)
        target = mass*torch.tensor([.6, .3, .01, .09])[None, :, None, None, None]
        prediction = model(torch.randn(1, 10, 8, 8, 8))*mass
        torch.testing.assert_close(prediction.sum(1), mass[:, 0])
        loss, errors = map_loss(prediction, target)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        zero, _ = map_loss(target, target)
        self.assertEqual(float(zero), 0.)
        m = metrics(target[0].numpy(), target[0].numpy(), mass[0, 0].numpy())
        self.assertTrue(all(v == 0 for v in m['map_L1']+m['centroid_error_cMpc_h']))
        with self.assertRaises(ValueError):
            map_loss(prediction, torch.zeros_like(target))

    def test_features_and_observer_symmetries(self):
        native = fixture()
        observer = np.array([2.5, 4., 5.5])
        encoded = torch.from_numpy(features(native, observer))
        for i, (axes, signs) in enumerate(SYMMETRIES):
            new_observer = observer[list(axes)].copy()
            new_observer = np.where(np.asarray(signs) < 0, 8-new_observer, new_observer)
            expected = features(augment(native, i), new_observer)
            np.testing.assert_allclose(transform(encoded, i, True).numpy(), expected, atol=1e-6)
            weights = torch.full((4, 8, 8, 8), .25)
            torch.testing.assert_close(transform(weights, i).sum(0), torch.ones(8, 8, 8))


if __name__ == '__main__':
    unittest.main()
