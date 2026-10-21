import json
from pathlib import Path
import unittest
import numpy as np
from cf4_lg_population import invariant_kinematics, sky_observable_log_jacobian, conditional_gaussian
from cf4_bundle_c_lg_population import observables_to_kinematics


class PopulationTests(unittest.TestCase):
    def test_rotation_and_cartesian_volume_jacobian(self):
        r, s = np.array([800., 0, 0]), np.array([60., 180, 0])
        v, w = np.array([-100., 40, 20]), np.array([20., 15, -30])
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        original = invariant_kinematics(r, s, v, w)
        np.testing.assert_allclose(original, invariant_kinematics(R@r, R@s, R@v, R@w))
        # Independent product of scalar coordinate derivatives; constants
        # of units cancel against this convention's omitted constant factors.
        cosine = np.dot(r, s)/np.linalg.norm(r)/np.linalg.norm(s)
        expected = (800.**5*850.**5)/(np.linalg.norm(r)**3*np.linalg.norm(s)**3*(1-cosine*cosine))
        expected /= np.prod(np.sqrt(1+np.r_[v, w]**2/100**2))
        self.assertAlmostEqual(float(sky_observable_log_jacobian(original, np.array([800, 850]))), np.log(expected))

    def test_conditional_and_observed_frame(self):
        other, mean, cov = conditional_gaussian(np.zeros(2), np.array([[4., 1.], [1., 9.]]), [0], np.array([[2.]]))
        np.testing.assert_array_equal(other, [1])
        np.testing.assert_allclose(mean, [[.5]])
        np.testing.assert_allclose(cov, [[8.75]])
        contract = json.loads((Path(__file__).resolve().parents[1]/'config/cf4_lg_observation_contract_v1.json').read_text())
        values = np.array([np.concatenate([contract['measurements'][g]['value'] for g in contract['galaxy_order']])])
        kin, distance = observables_to_kinematics(values, contract, .6774)
        self.assertTrue(np.isfinite(kin).all())
        np.testing.assert_allclose(5*np.log10(distance)+10, values[:, [0, 4]])
        # Common offsets of all three objects cancel exactly, including MW.
        common = np.tile(np.array([1., 2., -1., 5., 6., 7.]), (1, 3, 1))
        shifted, _ = observables_to_kinematics(values, contract, .6774, common)
        np.testing.assert_allclose(shifted, kin, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
