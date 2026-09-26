import math
import unittest

import numpy as np

from cf4_r2_coarsened_observation import (
    CountSupportError, sparse_poisson_count_logpmf,
)


class CoarsenedObservationTests(unittest.TestCase):
    def test_exact_count_pmf_includes_empty_cells(self):
        # Occupied cells 0 and 2; the empty cell contributes -mu_1.
        value = sparse_poisson_count_logpmf(
            np.array([0, 2]), np.array([2, 1]), np.array([1.5, 0.7]), 2.6)
        expected = -2.6 + 2 * math.log(1.5) - math.log(2) + math.log(0.7)
        self.assertAlmostEqual(value, expected)

    def test_map_zero_point_does_not_force_zero_cell_mean(self):
        # Exact point selection may be zero while the integrated cell
        # selection is positive. The latter, not the former, defines this PMF.
        self.assertTrue(math.isfinite(sparse_poisson_count_logpmf(
            np.array([4]), np.array([1]), np.array([0.5]), 0.5)))
        with self.assertRaises(CountSupportError):
            sparse_poisson_count_logpmf(
                np.array([4]), np.array([1]), np.array([0.0]), 0.5)

    def test_full_integral_cannot_be_replaced_by_occupied_sum(self):
        with self.assertRaises(ValueError):
            sparse_poisson_count_logpmf(
                np.array([1]), np.array([1]), np.array([1.0]), 0.9)


if __name__ == "__main__":
    unittest.main()
