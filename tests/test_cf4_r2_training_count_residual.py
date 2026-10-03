import unittest

import numpy as np

from cf4_r2_training_count_residual import summarize_training_counts


class TrainingCountResidualTest(unittest.TestCase):
    def test_known_corner_lands_in_expected_radius_octant_and_high_intensity_bin(self):
        n = 4
        expected = np.zeros((6, n, n, n), dtype=float)
        expected[0] = np.arange(1, n**3+1, dtype=float).reshape(n, n, n)
        exposure = np.ones((6, n**3), dtype=bool)
        keys = np.array([n**3-1], dtype=np.int64)  # population 0, cell (3,3,3)
        counts = np.array([2], dtype=np.int64)
        result = summarize_training_counts(expected, keys, counts, exposure.reshape(-1),
            grid_size=n, box_size=4., radial_edges=np.array([0., 2., 3.]),
            intensity_quantiles=5)

        radial = next(r for r in result['radial_by_population']
            if r['population'] == 0 and r['radius_lower_cMpc_h'] == 2.)
        sector = next(r for r in result['radial_population_by_exposed_octant']
            if r['population'] == 0 and r['radius_lower_cMpc_h'] == 2.
            and r['sector'] == '+++')
        high_bin = next(r for r in result['radial_population_by_model_intensity_quantile']
            if r['population'] == 0 and r['radius_lower_cMpc_h'] == 2.
            and r['model_intensity_quantile'] == 4)
        self.assertEqual(radial['observed_count'], 2)
        self.assertEqual(sector['observed_count'], 2)
        self.assertEqual(sector['exposed_cells'], 4)
        self.assertEqual(high_bin['observed_count'], 2)

    def test_radial_and_octant_bins_use_only_exposed_training_cells(self):
        n = 4
        expected = np.ones((6, n, n, n), dtype=float)
        exposure = np.ones((6, n**3), dtype=bool)
        # Exclude one cell without letting its model expectation enter totals.
        excluded = 0
        exposure[0, excluded] = False
        flat = n**3
        keys = np.array([1, flat+2, 2*flat+3], dtype=np.int64)
        counts = np.array([2, 1, 4], dtype=np.int64)
        result = summarize_training_counts(expected, keys, counts, exposure.reshape(-1),
            grid_size=n, box_size=4., radial_edges=np.array([0., 1.5, 4.]),
            intensity_quantiles=4)

        self.assertEqual(result['total']['observed_training_count'], 7)
        self.assertEqual(result['total']['expected_training_count'], 6*n**3-1)
        self.assertEqual(sum(r['observed_count'] for r in result['radial_by_population']), 7)
        self.assertEqual(sum(r['observed_count'] for r in result['radial_population_by_exposed_octant']), 7)
        self.assertEqual(sum(r['observed_count'] for r in result['radial_population_by_model_intensity_quantile']), 7)
        self.assertTrue(all(r['exposed_cells'] >= 0 for r in result['radial_population_by_exposed_octant']))

    def test_duplicate_intensity_quantile_cuts_are_merged(self):
        n = 3
        expected = np.zeros((6, n, n, n), dtype=float)
        exposure = np.ones((6, n**3), dtype=bool)
        expected[0].fill(2.)
        flat = n**3
        keys = np.array([flat+0], dtype=np.int64)
        counts = np.array([1], dtype=np.int64)
        result = summarize_training_counts(expected, keys, counts, exposure.reshape(-1),
            grid_size=n, box_size=3., radial_edges=np.array([0., 4.]),
            intensity_quantiles=5)
        rows = result['radial_population_by_model_intensity_quantile']
        pop1 = [r for r in rows if r['population'] == 1]
        self.assertEqual(len(pop1), 1)
        self.assertEqual(pop1[0]['exposed_cells'], n**3)
        self.assertEqual(pop1[0]['observed_count'], 1)

    def test_rejects_observed_key_outside_training_exposure(self):
        n = 3
        expected = np.ones((6, n, n, n), dtype=float)
        exposure = np.ones((6, n**3), dtype=bool)
        exposure[0, 0] = False
        with self.assertRaisesRegex(ValueError, 'outside training exposure'):
            summarize_training_counts(expected, np.array([0]), np.array([1]),
                exposure.reshape(-1), grid_size=n, box_size=3.,
                radial_edges=np.array([0., 4.]))


if __name__ == '__main__':
    unittest.main()
