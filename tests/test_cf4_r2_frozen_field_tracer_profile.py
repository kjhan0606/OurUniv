import unittest

import numpy as np

from cf4_r2_frozen_field_tracer_profile import (
    observed_radial_population, radial_population_bins,
    radial_population_l1, radial_population_table, unpack_profile_result)


class FrozenFieldTracerProfileTest(unittest.TestCase):
    def test_value_and_grad_aux_nested_return_contract(self):
        gradient = np.array([1., 2.])
        means = np.array([[3., 4.]])
        objective, score, radial_means, unpacked_gradient = unpack_profile_result(
            ((5., (6., means)), gradient))
        self.assertEqual(objective, 5.)
        self.assertEqual(score, 6.)
        np.testing.assert_array_equal(radial_means, means)
        np.testing.assert_array_equal(unpacked_gradient, gradient)

    def test_known_training_key_radius_population_and_octant_geometry(self):
        n = 4
        edges = np.array([0., 2., 3.])
        radial = radial_population_bins(n, 4., edges)
        key = 2*n**3 + (n**3-1)  # pop2, cell(3,3,3), radius sqrt(6.75)
        observed = observed_radial_population(np.array([key]), np.array([3]), n, radial)
        self.assertEqual(int(observed[2, 1]), 3)
        self.assertEqual(int(observed.sum()), 3)

    def test_radial_table_keeps_zeros_and_computes_descriptive_l1(self):
        observed = np.zeros((6, 2), dtype=np.int64)
        expected = np.zeros((6, 2), dtype=np.float64)
        observed[0, 1] = 2
        expected[0, 1] = 1
        rows = radial_population_table(observed, expected, np.array([0., 1., 2.]))
        self.assertEqual(len(rows), 12)
        self.assertIsNone(rows[0]['observed_to_expected'])
        self.assertEqual(rows[1]['observed_to_expected'], 2.)
        self.assertEqual(radial_population_l1(rows), .5)


if __name__ == '__main__':
    unittest.main()
