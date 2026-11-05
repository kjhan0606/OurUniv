import unittest

import numpy as np

from cf4_r2_field_reporting_regions import training_count_cell_region


class TrainingCountCellRegionTests(unittest.TestCase):
    def test_population_union_lifts_coarse_cells_to_aligned_node_blocks(self):
        n = 2
        nvoxel = n ** 3
        keys = np.asarray([
            np.ravel_multi_index((0, 1, 0), (n,) * 3),
            nvoxel + np.ravel_multi_index((1, 0, 1), (n,) * 3),
        ], dtype=np.int64)

        region = training_count_cell_region(
            keys, count_grid_size=2, field_grid_size=4, population_count=2)

        expected = np.zeros((4, 4, 4), dtype=bool)
        expected[0:2, 2:4, 0:2] = True
        expected[2:4, 0:2, 2:4] = True
        np.testing.assert_array_equal(region, expected)
        self.assertEqual(int(region.sum()), 16)

    def test_empty_training_keys_return_empty_region(self):
        region = training_count_cell_region(
            np.asarray([], dtype=np.int64), count_grid_size=2,
            field_grid_size=4, population_count=2)
        self.assertFalse(region.any())

    def test_rejects_duplicate_or_out_of_range_keys(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            training_count_cell_region(
                np.asarray([0, 0], dtype=np.int64), count_grid_size=2,
                field_grid_size=4, population_count=2)
        with self.assertRaisesRegex(ValueError, "valid population-major"):
            training_count_cell_region(
                np.asarray([16], dtype=np.int64), count_grid_size=2,
                field_grid_size=4, population_count=2)
        with self.assertRaisesRegex(ValueError, "valid population-major"):
            training_count_cell_region(
                np.asarray([0.0]), count_grid_size=2,
                field_grid_size=4, population_count=2)

    def test_rejects_noninteger_refinement(self):
        with self.assertRaisesRegex(ValueError, "integer refinement"):
            training_count_cell_region(
                np.asarray([0], dtype=np.int64), count_grid_size=3,
                field_grid_size=4, population_count=2)

    def test_rejects_noninteger_dimensions(self):
        with self.assertRaisesRegex(ValueError, "count_grid_size must be an integer"):
            training_count_cell_region(
                np.asarray([0], dtype=np.int64), count_grid_size=2.5,
                field_grid_size=4, population_count=2)


if __name__ == "__main__":
    unittest.main()
