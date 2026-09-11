import unittest
import numpy as np
import torch
from cf4_member_mass_readout import MemberMassNet, features, transform, map_loss, mass_shape_loss, metrics
from cf4_spatial_diffusion import augment, SYMMETRIES
from test_cf4_spatial_diffusion import fixture


class MemberMassTests(unittest.TestCase):
    def test_training_fraction_start_and_backbone_learning(self):
        torch.manual_seed(19)
        model = MemberMassNet()
        pi = torch.tensor([.002, .003, .00001, .99499])
        model.initialize_mass_fractions(pi)
        x = torch.randn(1, 10, 8, 8, 8)
        torch.testing.assert_close(model(x), pi[None, :, None, None, None].expand(1, 4, 8, 8, 8))
        mass = torch.ones(1, 1, 8, 8, 8)
        truth = mass*torch.tensor([.004, .006, .00002, .98998])[None, :, None, None, None]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = mass_shape_loss(model.logits(x), mass, truth)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            backbone_grad = float(model.enc[0][0].weight.grad.abs().sum())
            if step == 0:
                self.assertEqual(backbone_grad, 0.)
                self.assertGreater(float(model.head.weight.grad.abs().sum()), 0.)
            else:
                self.assertGreater(backbone_grad, 0.)
            optimizer.step()
        with self.assertRaises(ValueError):
            model.initialize_mass_fractions([0., .1, .1, .8])

    def test_mass_shape_sparse_support_and_exact_target(self):
        # Empty total cells and sparse roles; no epsilon mass is added.
        mass = torch.ones(1, 1, 2, 2, 2, dtype=torch.float64)
        mass[..., 0, 0, 0] = 0
        logits = torch.randn(1, 4, 2, 2, 2, dtype=torch.float64)
        truth = logits.softmax(1)*mass
        exact, mt, st = mass_shape_loss(logits, mass, truth)
        self.assertLess(abs(float(exact)), 1e-12)
        self.assertLess(float(mt.abs().max()), 1e-12)
        self.assertLess(float(st.abs().max()), 1e-12)
        sparse = torch.zeros_like(truth)
        sparse[:, 3] = mass[:, 0]
        for role, cell in enumerate([(0, 0, 1), (0, 1, 0), (1, 0, 0)]):
            sparse[(0, role)+cell] = .25
            sparse[(0, 3)+cell] -= .25
        # Very rare finite predicted mass must still receive a finite gradient.
        logits[:, 2] -= 100
        logits.requires_grad_()
        loss, _, _ = mass_shape_loss(logits, mass, sparse)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad[:, 2].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[..., 0, 0, 0].abs().sum()), 0.)
        with self.assertRaises(ValueError):
            mass_shape_loss(logits, mass, torch.zeros_like(sparse))
        invalid = sparse.clone()
        invalid[0, 0, 0, 0, 0] = .1
        with self.assertRaises(ValueError):
            mass_shape_loss(logits, mass, invalid)

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
