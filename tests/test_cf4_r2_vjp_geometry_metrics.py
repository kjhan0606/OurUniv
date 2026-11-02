"""Synthetic algebra/uncertainty tests for the bounded R2 geometry sketch."""

import json
import math
import unittest

import numpy as np

from cf4_r2_vjp_geometry_metrics import (
    chi_square_second_moment_interval,
    chi_square_second_moment_relative_half_width,
    leading_left_basis,
    project_probe,
    subspace_canonical_correlations,
    summarize_validation,
)


class VJPGeometryMetricsTests(unittest.TestCase):
    def test_tall_skinny_basis_is_orthonormal_and_spans_leading_columns(self):
        rng = np.random.default_rng(701)
        left, _ = np.linalg.qr(rng.normal(size=(60, 4)))
        right, _ = np.linalg.qr(rng.normal(size=(8, 8)))
        singular = np.array([10.0, 7.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0])
        sketch = left @ np.diag(singular[:4]) @ right[:4, :]
        basis, values, error = leading_left_basis(sketch, 4)
        np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(subspace_canonical_correlations(basis, left),
                                   np.ones(4), atol=1e-12)
        self.assertLess(error, 1e-12)
        np.testing.assert_allclose(values[:4], singular[:4], atol=1e-12)

    def test_rank_deficiency_fails_closed(self):
        matrix = np.zeros((20, 4))
        matrix[:, 0] = np.arange(20)
        with self.assertRaises(np.linalg.LinAlgError):
            leading_left_basis(matrix, 2)

    def test_subspace_overlap_compares_spans_not_column_labels(self):
        first = np.eye(4)[:, :2]
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        same_span = first @ rotation
        orthogonal = np.eye(4)[:, 2:]
        np.testing.assert_allclose(
            subspace_canonical_correlations(first, same_span), [1.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(
            subspace_canonical_correlations(first, orthogonal), [0.0, 0.0], atol=1e-12)

    def test_probe_projection_reports_complement_and_total_norm(self):
        basis = np.array([[1.0], [0.0]])
        coordinates, residual_sq, full_sq = project_probe(basis, np.array([3.0, 4.0]))
        np.testing.assert_array_equal(coordinates, [3.0])
        self.assertEqual(residual_sq, 16.0)
        self.assertEqual(full_sq, 25.0)

    def test_exact_directional_interval_uses_zero_mean_chi_square(self):
        values = np.tile(np.array([1.0, -1.0]), 32)
        result = chi_square_second_moment_interval(values, family_size=4)
        self.assertEqual(result["estimate"], 1.0)
        self.assertEqual(result["degrees_of_freedom"], 64)
        self.assertGreater(result["lower"], 0.0)
        self.assertGreater(result["upper"], result["lower"])
        self.assertAlmostEqual(result["per_interval_confidence"], 0.9875)

    def test_precision_gate_feasibility_is_checked_before_probe_design(self):
        # With four Bonferroni-adjusted two-sided intervals, n=64 cannot meet
        # a 0.50 relative half-width for any positive second-moment estimate.
        width64 = chi_square_second_moment_relative_half_width(64, family_size=4)
        width90 = chi_square_second_moment_relative_half_width(90, family_size=4)
        width91 = chi_square_second_moment_relative_half_width(91, family_size=4)
        self.assertAlmostEqual(width64, 0.6309723942, places=9)
        self.assertGreater(width90, 0.50)
        self.assertLessEqual(width91, 0.50)
        values = np.random.default_rng(81).normal(size=64)
        interval = chi_square_second_moment_interval(values, family_size=4)
        self.assertAlmostEqual(interval["relative_half_width"], width64, places=14)

    def test_precision_width_rejects_invalid_sample_or_family_settings(self):
        with self.assertRaises(ValueError):
            chi_square_second_moment_relative_half_width(0)
        with self.assertRaises(ValueError):
            chi_square_second_moment_relative_half_width(64, family_size=0)

    def test_residual_ucl_and_validation_summary_are_finite(self):
        rng = np.random.default_rng(992)
        projections = rng.normal(size=(64, 4))
        residual_sq = np.ones(64)
        full_sq = np.full(64, 100.0)
        result = summarize_validation(
            projections, residual_sq, full_sq,
            matrix_disagreement_limit=100.0,
            directional_relative_half_width_limit=100.0,
            bootstrap_replicates=100, bootstrap_seed=55)
        eps = 2.0 * math.sqrt(math.log(20.0) / 64.0)
        expected_ucl = 1.0 / (1.0 - eps)
        self.assertAlmostEqual(result["residual_trace_one_sided_95_ucl"], expected_ucl)
        self.assertAlmostEqual(result["residual_quadratic_cost_nats"], 0.005 * expected_ucl)
        self.assertEqual(len(result["directional_second_moment_intervals"]), 4)
        self.assertEqual(len(result["projected_uncertainty"]["sorted_eigenvalue_lower"]), 4)
        json.dumps(result, allow_nan=False)

    def test_zero_direction_is_a_json_safe_failed_gate(self):
        rng = np.random.default_rng(123)
        projections = rng.normal(size=(64, 4))
        projections[:, 0] = 0.0
        result = summarize_validation(
            projections, np.ones(64), np.full(64, 100.0),
            matrix_disagreement_limit=100.0,
            directional_relative_half_width_limit=100.0,
            bootstrap_replicates=50)
        self.assertFalse(result["directional_second_moment_intervals"][0]["passed"])
        self.assertIsNone(
            result["directional_second_moment_intervals"][0]["relative_half_width"])
        self.assertFalse(result["passed"])
        json.dumps(result, allow_nan=False)

    def test_rejects_misaligned_or_nonfinite_validation_inputs(self):
        with self.assertRaises(ValueError):
            summarize_validation(np.ones((64, 4)), np.ones(63), np.ones(64))
        with self.assertRaises(ValueError):
            project_probe(np.eye(2), np.array([1.0, np.nan]))


if __name__ == "__main__":
    unittest.main()
