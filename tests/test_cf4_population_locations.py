import unittest
import numpy as np
import torch

from cf4_population_locations import (native_features, PopulationRoleModel, tuple_targets,
    training_shared_probability, selected, calibrate_alpha, footprint_overlap, population_cases,
    field_patch, cells)
from cf4_member_mass_readout import transform, inverse_symmetry
from cf4_role_locations import transform_positions, sample_triples
from cf4_spatial_diffusion import augment
from test_cf4_spatial_diffusion import fixture


class PopulationLocationTests(unittest.TestCase):
    def test_features_symmetry_and_boost(self):
        native = fixture()
        raw = native_features(native)
        changed = native.copy()
        boost = np.array([120., -80., 40.])[:, None, None, None]
        changed[1:4] += native[0]*boost
        changed[4:] += 2*boost*native[1:4]+native[0]*boost**2
        np.testing.assert_allclose(native_features(changed), raw, rtol=1e-5, atol=1e-4)
        model = PopulationRoleModel().eval()
        with torch.no_grad():
            for theta in model.theta:
                theta.copy_(torch.linspace(-.3, .3, len(theta)))
            tensor = torch.from_numpy(raw)
            observer = torch.tensor([3., 4., 5.])
            labels = torch.tensor([[2, 3, 4], [5, 4, 3], [5, 4, 3]])
            baseline = [model.log_prob(tensor, observer, labels[:r], r).exp() for r in range(3)]
            for symmetry in range(48):
                mapped = torch.from_numpy(native_features(augment(native, symmetry)))
                torch.testing.assert_close(mapped, transform(tensor, symmetry), rtol=1e-5, atol=1e-4)
                new_o = transform_positions(observer, symmetry, 8)
                new_c = transform_positions(labels, symmetry, 8, cell_indices=True)
                for role in range(3):
                    logp = model.log_prob(mapped, new_o, new_c[:role], role)
                    restored = transform(logp.exp()[None], inverse_symmetry(symmetry))[0]
                    self.assertLess(float(abs(restored-baseline[role]).sum()), 1e-4)

    def test_hierarchical_weights_gradient_and_references(self):
        case = dict(center=[2, 3, 4], pairs=[dict(center=[4, 3, 4], centers=[[4, 3, 4]]),
            dict(center=[5, 4, 3], centers=[[5, 4, 3], [5, 4, 3], [6, 4, 3]])])
        targets, weights = tuple_targets(case)
        np.testing.assert_allclose(weights, [.5, 1/6, 1/6, 1/6])
        self.assertAlmostEqual(training_shared_probability([case]), ((.5+2/6)+1)/3)
        raw = torch.from_numpy(native_features(fixture()))
        model = PopulationRoleModel()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 21)
        for role in range(3):
            parents = targets[0, :role]
            outputs = model.distributions(raw, [4., 4., 4.], parents, role)
            torch.testing.assert_close(outputs['raw'], outputs['density'], rtol=0, atol=0)
            for q in outputs.values():
                self.assertLess(abs(float(q.detach().exp().sum())-1), 1e-5)
            features, _, _ = model.design(raw, [4., 4., 4.], parents, role)
            labels = np.array([[2, 3, 4], [5, 3, 4], [5, 3, 4]])
            loss = -selected(outputs['raw'], labels).mean()
            loss.backward()
            expected = (outputs['raw'].detach().exp()*features).sum((1, 2, 3))-features[:, labels[:, 0], labels[:, 1], labels[:, 2]].mean(1)
            torch.testing.assert_close(model.theta[role].grad, expected, atol=2e-6, rtol=1e-4)
        with self.assertRaises(ValueError):
            model.log_prob(raw, [4., 4., 4.], targets[0], 0)
        # Reused sampler accepts only fields/O and obtains ALL parents from its draws.
        class Recorder:
            training = False
            def log_prob(self, field, observer, parents, role):
                if parents != [[1, 2, 3]]*role:
                    raise AssertionError('oracle/incorrect earlier parents')
                q = torch.full(field.shape[1:], -1000.)
                q[1, 2, 3] = 0
                return q
        draws, _ = sample_triples(Recorder(), raw, [4., 4., 4.], 2, torch.Generator().manual_seed(41))
        np.testing.assert_array_equal(draws, np.tile([1, 2, 3], (2, 3, 1)))

    def test_fixed_spatial_split_periodic_cells_and_footprints(self):
        positions = {1:[15., 30., 30.], 2:[15.5,30.,30.], 3:[15.6,30.,30.],
                     4:[43.,30.,30.], 5:[43.5,30.,30.], 6:[43.6,30.,30.],
                     7:[63.,30.,60.], 8:[63.5,30.,60.], 9:[63.6,30.,60.]}
        ids = np.array([[1,2,3],[1,2,3],[4,5,6],[7,8,9]])
        cases, report = population_cases(ids, np.zeros(4), positions, [])
        self.assertEqual(report['observer_counts'], dict(train=1, calibration=1, test=1))
        self.assertEqual(report['archived_satellite_rows'], 3)
        self.assertTrue(footprint_overlap([392,0,0], [0,0,0]))
        self.assertFalse(footprint_overlap([80,160,240], [40,10,10]))
        self.assertFalse(footprint_overlap([0,0,0], [12.75,0,0]))
        self.assertTrue(footprint_overlap([0,0,0], [12.5625,0,0]))
        np.testing.assert_array_equal(cells([[74.9,.1,.2]], np.array([392,0,0])), [[7,0,1]])
        field = torch.arange(5*8**3).reshape(5,8,8,8)
        cropped = field_patch(field, [7,7,7], 2)
        self.assertEqual(int(cropped[0,1,1,1]), 0)

    def test_calibration_scalar_and_endpoint(self):
        first = np.log([.2,.8])
        second = np.log([.8,.2])
        fit = calibrate_alpha(first, second, [.38,.62])
        self.assertAlmostEqual(fit['alpha'], .3, places=5)
        self.assertEqual(calibrate_alpha(first, second, [.2,.8])['alpha'], 0.)
        with self.assertRaises(ValueError):
            calibrate_alpha(first, second, [.3,.3])


if __name__ == '__main__':
    unittest.main()
