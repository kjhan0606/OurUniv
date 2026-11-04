import unittest

import numpy as np

from cf4_r2_cross_chain_field_diagnostics import compare_chain_field_maps


class CrossChainFieldDiagnosticsTests(unittest.TestCase):
    def test_unequal_chain_lengths_are_weighted_but_not_silently_trimmed(self):
        means = np.asarray([[[0.]], [[2.]], [[4.]]])
        sds = np.asarray([[[1.]], [[2.]], [[3.]]])
        result = compare_chain_field_maps(means, sds, np.asarray([2, 4, 8]))
        self.assertEqual(result["draw_counts"].tolist(), [2, 4, 8])
        self.assertAlmostEqual(float(result["pooled_mean"][0, 0]), 20. / 7.)
        self.assertAlmostEqual(float(result["between_chain_mean_sd"][0, 0]), 2.)
        self.assertAlmostEqual(float(result["within_chain_posterior_sd"][0, 0]),
            np.sqrt(45. / 7.))
        self.assertAlmostEqual(float(result["mean_chain_posterior_sd"][0, 0]),
            17. / 7.)
        self.assertAlmostEqual(float(result["between_chain_sd_map_sd"][0, 0]), 1.)
        self.assertAlmostEqual(float(
            result["posterior_sd_map_disagreement_over_mean_sd"][0, 0]),
            7. / 17.)
        self.assertIn("NOT_CONVERGENCE_EVIDENCE", result["status"])

    def test_a_missing_chain_cell_is_reported_as_undefined_not_dropped(self):
        means = np.asarray([
            [[1., 1.]], [[1., 2.]], [[1., np.nan]], [[1., 4.]]])
        sds = np.ones_like(means)
        result = compare_chain_field_maps(means, sds, np.asarray([8, 8, 8, 8]))
        np.testing.assert_array_equal(result["defined_mask"], [[True, False]])
        np.testing.assert_array_equal(result["valid_chain_count"], [[4, 3]])
        summary = result["regional_summary"][
            "mean_map_disagreement_over_within_sd"]["full_box"]
        self.assertEqual(summary["region_cells"], 2)
        self.assertEqual(summary["defined_cells"], 1)
        self.assertEqual(summary["undefined_cells"], 1)

    def test_zero_within_sd_is_counted_and_ratio_is_explicit(self):
        means = np.asarray([[[1., 0.]], [[1., 2.]], [[1., 4.]], [[1., 6.]]])
        sds = np.zeros_like(means)
        result = compare_chain_field_maps(means, sds, np.asarray([4, 4, 4, 4]))
        ratio = result["mean_map_disagreement_over_within_sd"]
        self.assertEqual(float(ratio[0, 0]), 0.)
        self.assertTrue(np.isinf(ratio[0, 1]))
        summary = result["regional_summary"][
            "mean_map_disagreement_over_within_sd"]["full_box"]
        self.assertEqual(summary["zero_denominator_cells"], 2)
        self.assertEqual(summary["nonfinite_ratio_cells"], 1)

    def test_valid_flag_cannot_hide_nonfinite_chain_map(self):
        means = np.asarray([[[1.]], [[np.nan]]])
        sds = np.ones_like(means)
        valid = np.ones_like(means, dtype=bool)
        with self.assertRaisesRegex(ValueError, "valid chain cells"):
            compare_chain_field_maps(means, sds, np.asarray([2, 2]), valid=valid)

    def test_invalid_shapes_and_draw_counts_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "at least two chains"):
            compare_chain_field_maps(np.ones((1, 1)), np.ones((1, 1)), [2])
        with self.assertRaisesRegex(ValueError, "integer count"):
            compare_chain_field_maps(np.ones((2, 1)), np.ones((2, 1)), [2, 1.5])
        with self.assertRaisesRegex(ValueError, "aligned boolean map"):
            compare_chain_field_maps(np.ones((2, 1, 1)), np.ones((2, 1, 1)), [2, 2],
                regions={"wrong": np.ones((2, 2), dtype=bool)})


if __name__ == "__main__":
    unittest.main()
