"""Deterministic algebra tests for the bounded VJP tail screen."""

import unittest

import numpy as np

from cf4_r2_vjp_tail_metrics import (
    normalized_repeat_error,
    split_low_intensity_tail,
    summarize_vjp_split,
)


class VJPTailMetricsTests(unittest.TestCase):
    def test_tail_uses_fixed_fraction_and_row_id_tie_break(self):
        rows = np.array([9, 3, 7, 1], dtype=np.int32)
        values = np.array([1e-9, 1e-12, 1e-12, 1.0])
        tail, rest = split_low_intensity_tail(rows, values, fraction=0.5)
        np.testing.assert_array_equal(rows[tail], [3, 7])
        np.testing.assert_array_equal(rows[rest], [9, 1])

    def test_tail_share_is_reported_not_used_as_a_failure_gate(self):
        tail = np.array([4.0, 0.0, 0.0])
        rest = np.array([0.0, 3.0, 0.0])
        full = tail + rest
        result = summarize_vjp_split(full, tail, rest)
        self.assertAlmostEqual(result["tail_squared_norm_share"], 16.0 / 25.0)
        self.assertEqual(result["doubled_tail_rest_cross_term"], 0.0)
        self.assertTrue(result["decomposition_passed"])

    def test_decomposition_error_uses_tail_plus_rest_norm(self):
        tail = np.array([1.0, 2.0])
        rest = np.array([-1.0, 1.0])
        full = tail + rest + np.array([1e-8, 0.0])
        result = summarize_vjp_split(full, tail, rest, tolerance=1e-9)
        self.assertFalse(result["decomposition_passed"])
        self.assertGreater(result["decomposition_relative_error"], 1e-9)

    def test_repeat_error_is_scale_relative_and_detects_change(self):
        result = normalized_repeat_error(np.array([1e-12, 2e-12]),
                                         np.array([1e-12, 2.1e-12]))
        self.assertGreater(result["relative_error"], 0.0)
        self.assertLess(result["relative_error"], 0.1)

    def test_invalid_rows_and_nonfinite_gradients_fail_closed(self):
        with self.assertRaises(ValueError):
            split_low_intensity_tail(np.array([1, 1]), np.array([1e-3, 2e-3]))
        with self.assertRaises(ValueError):
            summarize_vjp_split(np.array([1.0]), np.array([np.nan]), np.array([0.0]))


if __name__ == "__main__":
    unittest.main()
