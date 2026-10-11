import unittest

import numpy as np

from cf4_r2_native_128_observations import coarsen_sparse


class TestNative128Aggregation(unittest.TestCase):
    def test_sparse_counts_preserve_population_and_total(self):
        n = 256
        cells = np.ravel_multi_index(([0, 1, 2], [0, 1, 0], [0, 1, 0]), (n,)*3)
        keys = np.array([cells[0], cells[1], n**3 + cells[2]])
        parent, value = coarsen_sparse(keys, np.array([2, 3, 5]))
        self.assertEqual(value.sum(), 10)
        np.testing.assert_array_equal(value, [5, 5])
        self.assertEqual(parent[0], 0)
        self.assertEqual(parent[1], 128**3 + 128**2)

    def test_invalid_count_rejected(self):
        with self.assertRaisesRegex(ValueError, 'invalid sparse'):
            coarsen_sparse(np.array([0]), np.array([0]))
