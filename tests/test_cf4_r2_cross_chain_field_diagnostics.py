import unittest

import numpy as np

from cf4_r2_cross_chain_field_diagnostics import (
    compare_chain_field_maps,
    compare_chain_field_maps_streaming,
)


class CrossChainFieldDiagnosticsTests(unittest.TestCase):
    def assert_summary_close(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_summary_close(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for actual_value, expected_value in zip(actual, expected):
                self.assert_summary_close(actual_value, expected_value)
        elif isinstance(expected, float):
            self.assertAlmostEqual(actual, expected, delta=1e-12)
        else:
            self.assertEqual(actual, expected)

    def test_streamed_scalar_summaries_match_full_chain_arrays(self):
        rng = np.random.default_rng(20261104)
        means = rng.normal(size=(4, 4, 4, 4))
        sds = rng.uniform(0.5, 1.5, size=means.shape)
        valid = np.ones(means.shape, dtype=bool)
        valid[2, 0, 0, 0] = False
        means[2, 0, 0, 0] = np.nan
        region = np.zeros((4, 4, 4), dtype=bool)
        region[:2] = True
        regions = {"full_box": np.ones_like(region), "training": region}
        counts = np.asarray([8, 12, 16, 20], dtype=np.int64)
        expected = compare_chain_field_maps(means, sds, counts,
            valid=valid, regions=regions)
        actual = compare_chain_field_maps_streaming(
            ((means[index], sds[index], valid[index]) for index in range(4)),
            counts, regions=regions)
        self.assertEqual(actual["draw_counts"].tolist(), counts.tolist())
        self.assert_summary_close(actual["regional_summary"],
            expected["regional_summary"])
        default_regions = compare_chain_field_maps_streaming(
            ((means[index], sds[index], valid[index]) for index in range(4)),
            counts)
        self.assert_summary_close(default_regions["regional_summary"],
            {"mean_map_disagreement_over_within_sd": {
                "full_box": expected["regional_summary"]
                    ["mean_map_disagreement_over_within_sd"]["full_box"]},
             "posterior_sd_map_disagreement_over_mean_sd": {
                "full_box": expected["regional_summary"]
                    ["posterior_sd_map_disagreement_over_mean_sd"]["full_box"]}})

    def test_streamed_vector_summaries_match_componentwise_reference(self):
        rng = np.random.default_rng(301)
        means = rng.normal(size=(4, 3, 4, 4, 4))
        sds = rng.uniform(0.25, 2., size=means.shape)
        valid = rng.random(size=means.shape) > 0.05
        region = np.zeros((4, 4, 4), dtype=bool)
        region[:, :, :2] = True
        regions = {"full_box": np.ones_like(region), "training": region}
        counts = np.asarray([10, 10, 10, 10], dtype=np.int64)
        expected_by_axis = [compare_chain_field_maps(
            means[:, axis], sds[:, axis], counts,
            valid=valid[:, axis], regions=regions)["regional_summary"]
            for axis in range(3)]
        expected = dict(
            mean_map_disagreement_over_within_sd=[item[
                "mean_map_disagreement_over_within_sd"]
                for item in expected_by_axis],
            posterior_sd_map_disagreement_over_mean_sd=[item[
                "posterior_sd_map_disagreement_over_mean_sd"]
                for item in expected_by_axis])
        actual = compare_chain_field_maps_streaming(
            ((means[index], sds[index], valid[index]) for index in range(4)),
            counts, regions=regions)
        self.assert_summary_close(actual["regional_summary"], expected)

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
