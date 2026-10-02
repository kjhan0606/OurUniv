import unittest

import numpy as np

from cf4_r2_fixed_field_lf_prior_sensitivity import (
    aggregate_score_delta, pearson, rate_scaled_score,
)


class FixedFieldLfPriorSensitivityTest(unittest.TestCase):
    def test_pearson_reports_only_bins_at_the_fixed_threshold(self):
        result = pearson(np.array([[12, 8, 0]]), np.array([[10., 10., 4.]]))
        self.assertEqual(result['bins'], 2)
        self.assertAlmostEqual(result['statistic'], .8)
        self.assertEqual(result['expected_count_threshold'], 5.)

    def test_pearson_rejects_empty_threshold_selection(self):
        with self.assertRaises(ValueError):
            pearson(np.array([[1, 2]]), np.array([[1., 2.]]), minimum_expected=5.)

    def test_total_rate_poisson_score_identity(self):
        observed = np.array([2, 4])
        expected = np.array([1., 3.])
        scale = 1.5
        score = float(np.sum(observed*np.log(expected)-expected))
        direct = float(np.sum(observed*np.log(expected*scale)-expected*scale))
        self.assertAlmostEqual(rate_scaled_score(score, observed.sum(),
            expected.sum(), scale), direct)

    def test_aggregate_score_delta(self):
        result = aggregate_score_delta(np.array([[3, 1]]),
            np.array([[2., 2.]]), np.array([[3., 1.]]))
        expected = 3*np.log(3/2)+np.log(1/2)
        self.assertAlmostEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
