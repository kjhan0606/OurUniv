import math
import unittest

import numpy as np
import torch

from cf4_population_locations import native_features, PopulationRoleModel, DX
from cf4_position_link import (features_torch, rectangle_probability, observation_kernel,
    log_likelihood, conservative_permutation)
from cf4_continuous_matter import restrict
from test_cf4_spatial_diffusion import fixture


class PositionLinkTests(unittest.TestCase):
    def test_native_features_and_parent_conservation(self):
        native = fixture(16)
        halo = np.pad(native, ((0, 0), (4, 4), (4, 4), (4, 4)), mode='wrap')
        value = torch.from_numpy(halo)
        np.testing.assert_allclose(features_torch(value).numpy(), native_features(native), rtol=1e-5, atol=1e-4)
        for axis in range(3):
            changed = conservative_permutation(value, axis).numpy()[:, 4:-4, 4:-4, 4:-4]
            np.testing.assert_allclose(restrict(changed, 8), restrict(native, 8), rtol=1e-12, atol=1e-8)

    def test_measurement_integral_jacobian_correlations_shared_cells(self):
        rho = .7
        probability, error = rectangle_probability([-12, 0], [-12, 0], np.zeros(2),
            np.array([[1., rho], [rho, 1.]]), 1e-12)
        self.assertAlmostEqual(probability, .25+math.asin(rho)/(2*math.pi), places=11)
        self.assertLess(error, 1e-10)
        kernel = observation_kernel([1.49, 1.5, 1.5], [1.5]*3,
            [[1., 0, 0], [1., 0, 0]], [24.4, 24.4], np.array([[.001, .0007], [.0007, .001]]), n=16)
        self.assertTrue(any(a == t for a, t, _ in kernel['pairs']))
        self.assertAlmostEqual(kernel['tilted_probability_in_field'], 1., places=10)
        class Uniform:
            def log_prob(self, raw, observer, parents, role):
                return torch.full(raw.shape[1:], -3*math.log(raw.shape[-1]), dtype=raw.dtype)
        features = torch.ones((5, 16, 16, 16), dtype=torch.float64)
        value, factors = log_likelihood(Uniform(), features, kernel)
        self.assertAlmostEqual(float(value), -9*math.log(16*DX)+kernel['log_jacobian'], places=9)
        self.assertAlmostEqual(float(factors['MW']+factors['M31_given_MW']+factors['M33_given_MW_M31_data']), float(value), places=12)
        model = PopulationRoleModel().double().eval()
        for role in range(3):
            logp = model.log_prob(features, [8.]*3, [[8, 8, 8]]*role, role)
            self.assertTrue(bool(torch.isfinite(logp).all()))
            self.assertLess(abs(float(logp.exp().sum())-1), 1e-10)

    def test_interior_physical_direction_gradient(self):
        native = fixture()
        value = torch.from_numpy(np.pad(native, ((0, 0), (4, 4), (4, 4), (4, 4)), mode='wrap'))
        kernel = observation_kernel([.74, .75, .75], [.75]*3,
            [[1., 0, 0], [0, 1., 0]], [23.4, 23.4], np.array([[.001, .0007], [.0007, .001]]), n=8)
        model = PopulationRoleModel().double().eval()
        with torch.no_grad():
            for theta in model.theta:
                theta.copy_(torch.linspace(-.3, .3, len(theta)))
        model.requires_grad_(False)
        for axis in range(3):
            direction = conservative_permutation(value, axis)-value
            middle = (value+.05*direction).requires_grad_(True)
            loss, _ = log_likelihood(model, features_torch(middle), kernel)
            gradient, = torch.autograd.grad(loss, middle)
            analytic = float((gradient*direction).sum())
            for step in (1e-4, 5e-5):
                with torch.no_grad():
                    plus, _ = log_likelihood(model, features_torch(middle+step*direction), kernel)
                    minus, _ = log_likelihood(model, features_torch(middle-step*direction), kernel)
                self.assertAlmostEqual(analytic, float((plus-minus)/(2*step)), delta=1e-4+.01*abs(analytic))


if __name__ == '__main__':
    unittest.main()
