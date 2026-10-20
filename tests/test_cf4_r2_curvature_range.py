import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cf4_r2_curvature_range import (
    likelihood_hessian_action,
    local_probe_schedule,
    orthonormal_random_directions,
    outside_range_energy_fraction,
    relative_action_change,
    response_range,
    support_summary_difference,
)


class CurvatureRangeTests(unittest.TestCase):
    def test_local_probe_schedule_has_exact_budget_and_validation_roles(self):
        directions = tuple(np.eye(6)[i] for i in range(6))
        schedule = local_probe_schedule(directions, step=0.1)
        self.assertEqual(len(schedule), 8)
        self.assertEqual([row[1] for row in schedule[:4]], ["range_sketch"] * 4)
        self.assertEqual([row[1] for row in schedule[4:6]],
                         ["independent_random_direction_validation"] * 2)
        self.assertEqual(schedule[6][0], "sketch_1_half_step")
        np.testing.assert_array_equal(schedule[6][2], directions[0])
        self.assertAlmostEqual(schedule[6][3], 0.05)
        self.assertEqual(schedule[7][0], "validation_1_half_step")
        np.testing.assert_array_equal(schedule[7][2], directions[4])
        self.assertAlmostEqual(schedule[7][3], 0.05)

    def test_deterministic_orthonormal_directions(self):
        first = orthonormal_random_directions(64, 6, seed=20261021)
        second = orthonormal_random_directions(64, 6, seed=20261021)
        self.assertEqual(len(first), 6)
        np.testing.assert_array_equal(np.stack(first), np.stack(second))
        gram = np.stack(first) @ np.stack(first).T
        np.testing.assert_allclose(gram, np.eye(6), rtol=0, atol=2e-14)

    def test_likelihood_action_subtracts_unit_gaussian_prior(self):
        n_ic = 5
        matrix = np.diag([2., -1., 0.5, 3., -2.] + [4.] * 24)
        direction = np.array([1., 0., 0., 0., 0.])
        full_direction = np.r_[direction, np.zeros(24)]
        q = np.arange(1., 30.)
        step = 1e-3
        gradient_plus = (np.eye(n_ic + 24) + matrix) @ (q + step * full_direction)
        gradient_minus = (np.eye(n_ic + 24) + matrix) @ (q - step * full_direction)
        action = likelihood_hessian_action(
            gradient_plus, gradient_minus, direction, step=step, n_ic=n_ic
        )
        np.testing.assert_allclose(action, matrix @ full_direction, atol=1e-12)
        # Off-diagonal nuisance gradient response is retained rather than dropped.
        matrix[5, 0] = 0.75
        gradient_plus = (np.eye(n_ic + 24) + matrix) @ (q + step * full_direction)
        gradient_minus = (np.eye(n_ic + 24) + matrix) @ (q - step * full_direction)
        action = likelihood_hessian_action(
            gradient_plus, gradient_minus, direction, step=step, n_ic=n_ic
        )
        self.assertAlmostEqual(action[n_ic], 0.75, places=12)

    def test_range_captures_span_and_reports_singular_values(self):
        first = np.array([1., 0., 0., 0.])
        second = np.array([0., 2., 0., 0.])
        dependent = first + second
        basis, singular_values = response_range([first, second, dependent])
        self.assertEqual(len(basis), 2)
        self.assertEqual(singular_values.shape, (3,))
        self.assertAlmostEqual(outside_range_energy_fraction(dependent, basis), 0.0, places=14)
        outside = np.array([0., 0., 0., 1.])
        self.assertAlmostEqual(outside_range_energy_fraction(outside, basis), 1.0, places=14)

    def test_relative_action_change_is_scale_invariant(self):
        first = np.array([1., 2., -3.])
        self.assertAlmostEqual(relative_action_change(first, 3. * first), 2. / 3.)
        self.assertEqual(relative_action_change(first, first), 0.0)

    def test_support_summary_difference_ignores_volatile_timing(self):
        plus = dict(components=101, largest_candidate_cells=10, radius=42.1,
                    packing_seconds=20.0)
        minus = dict(components=100, largest_candidate_cells=9, radius=42.0,
                     packing_seconds=1.0)
        difference = support_summary_difference(plus, minus)
        self.assertEqual(difference["component_count_delta"], 1.0)
        self.assertEqual(difference["largest_candidate_cells_delta"], 1.0)
        self.assertAlmostEqual(difference["radius_delta_cMpc_h"], 0.1)

    def test_support_summary_difference_rejects_incomplete_metadata(self):
        with self.assertRaises(ValueError):
            support_summary_difference(
                dict(components=10, largest_candidate_cells=4),
                dict(components=10, largest_candidate_cells=4),
            )

    def test_invalid_shapes_and_step_are_rejected(self):
        with self.assertRaises(ValueError):
            orthonormal_random_directions(0, 1, seed=1)
        with self.assertRaises(ValueError):
            likelihood_hessian_action(np.zeros(8), np.zeros(8), np.ones(5), step=0., n_ic=5)
        with self.assertRaises(ValueError):
            outside_range_energy_fraction(np.zeros(2), [np.ones(3)])


if __name__ == "__main__":
    unittest.main()
