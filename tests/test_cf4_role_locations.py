import unittest
import numpy as np
import torch

from cf4_role_locations import (RoleLocationNet, native_state, native_center_cells, transform_state,
    transform_positions, location_features, reference_log_probs, sample_triples)
from cf4_member_mass_readout import transform, inverse_symmetry
from cf4_spatial_diffusion import augment
from test_cf4_spatial_diffusion import fixture


class RoleLocationTests(unittest.TestCase):
    def test_native_periodic_center_labels(self):
        centers = np.array([[74900., 100., 375.], [100., 100., 375.], [150., 100., 375.]])
        cells = native_center_cells(centers, [73.5, 0., 0.])
        np.testing.assert_array_equal(cells, [[7, 0, 2], [8, 0, 2], [8, 0, 2]])
        with self.assertRaises(ValueError):
            native_center_cells(centers, [30., 0., 0.])

    def test_scalar_features_and_whole_network_signed_equivariance(self):
        torch.manual_seed(123)
        native = fixture()
        raw = native_state(native)
        observer = torch.tensor([2.5, 4., 5.5])
        labels = torch.tensor([[2, 3, 5], [5, 4, 3], [5, 4, 3]])
        model = RoleLocationNet().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 72417)
        with torch.no_grad():
            references = [model.log_prob(raw, observer, labels[:r], r) for r in range(3)]
            for symmetry in range(48):
                transformed = transform_state(raw, symmetry)
                torch.testing.assert_close(transformed, native_state(augment(native, symmetry)), rtol=2e-6, atol=1e-4)
                new_o = transform_positions(observer, symmetry, 8)
                new_c = transform_positions(labels, symmetry, 8, cell_indices=True)
                for role in range(3):
                    encoded, geom = location_features(raw, observer, labels[:role], role)
                    changed, changed_geom = location_features(transformed, new_o, new_c[:role], role)
                    torch.testing.assert_close(changed, transform(encoded, symmetry), rtol=2e-6, atol=1e-5)
                    torch.testing.assert_close(changed_geom, transform(geom[None], symmetry)[0], rtol=2e-6, atol=1e-5)
                    logp = model.log_prob(transformed, new_o, new_c[:role], role)
                    restored = transform(logp.exp()[None], inverse_symmetry(symmetry))[0]
                    self.assertLess(float(abs(restored-references[role].exp()).sum()), 1e-4)

    def test_probability_training_references_and_nonoracle_rollout(self):
        torch.manual_seed(321)
        raw = native_state(fixture())
        observer = torch.tensor([4., 4., 4.])
        targets = torch.tensor([[3, 4, 4], [5, 4, 3], [5, 4, 3]])
        model = RoleLocationNet().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
        optimizer.zero_grad(set_to_none=True)
        initial = model.enc[0][0].weight.detach().clone()
        for role in range(3):
            logp = model.log_prob(raw, observer, targets[:role], role)
            self.assertLess(abs(float(logp.detach().exp().sum())-1), 2e-6)
            (-logp[tuple(targets[role])]/3).backward()
            _, geom = location_features(raw, observer, targets[:role], role)
            for reference in reference_log_probs(raw, geom, role, targets[:role], .4):
                self.assertTrue(torch.isfinite(reference).all())
                self.assertLess(abs(float(reference.exp().sum())-1), 2e-6)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.enc[0][0].weight.grad.abs().sum()), 0)
        optimizer.step()
        self.assertFalse(torch.equal(initial, model.enc[0][0].weight))
        with self.assertRaises(ValueError):
            model.log_prob(raw, observer, targets, 0)

        class RecordingModel:
            training = False
            def __init__(self):
                self.parents = []
            def log_prob(self, value, origin, parents, role):
                self.parents.append((role, np.asarray(parents).reshape(-1, 3).tolist()))
                # Deterministic test distribution: all roles may share one cell.
                answer = torch.full(value.shape[1:], -1000.)
                answer[1, 2, 3] = 0.
                return answer

        recorder = RecordingModel()
        samples, _ = sample_triples(recorder, raw, observer, 2, torch.Generator().manual_seed(2))
        np.testing.assert_array_equal(samples, np.tile([1, 2, 3], (2, 3, 1)))
        for role, parents in recorder.parents:
            self.assertEqual(parents, [[1, 2, 3]]*role)


if __name__ == '__main__':
    unittest.main()
