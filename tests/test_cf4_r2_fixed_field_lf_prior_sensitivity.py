import unittest

import numpy as np

from cf4_r2_fixed_field_lf_prior_sensitivity import pearson


class FixedFieldLfPriorSensitivityTest(unittest.TestCase):
    def test_pearson_reports_only_bins_at_the_fixed_threshold(self):
        result = pearson(np.array([[12, 8, 0]]), np.array([[10., 10., 4.]]))
        self.assertEqual(result['bins'], 2)
        self.assertAlmostEqual(result['statistic'], .8)
        self.assertEqual(result['expected_count_threshold'], 5.)

    def test_pearson_rejects_empty_threshold_selection(self):
        with self.assertRaises(ValueError):
            pearson(np.array([[1, 2]]), np.array([[1., 2.]]), minimum_expected=5.)


if __name__ == '__main__':
    unittest.main()
