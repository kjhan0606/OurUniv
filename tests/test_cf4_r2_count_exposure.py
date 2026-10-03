import unittest

import numpy as np

from cf4_r2_count_exposure import build_population_exposure_masks


class PopulationExposureMaskTests(unittest.TestCase):
    def test_graph_buffer_exclusions_are_population_specific_and_disjoint(self):
        # N=2 gives eight voxels per population. The two heldout voxels are
        # replicated across populations, then role-specific buffer keys are
        # removed from their respective exposure masks.
        train, heldout = build_population_exposure_masks(
            2, np.array([1, 4], dtype=np.int32),
            train_window_excluded_keys=np.array([10], dtype=np.int32),
            heldout_window_excluded_keys=np.array([4], dtype=np.int32),
            population_count=2)
        train = train.reshape(2, 8)
        heldout = heldout.reshape(2, 8)
        self.assertEqual(int(train.sum()), 11)
        self.assertEqual(int(heldout.sum()), 3)
        self.assertFalse(train[:, [1, 4]].any())
        self.assertTrue(heldout[:, 1].all())
        self.assertTrue(heldout[1, 4])
        self.assertFalse(train[1, 2])  # population 1, voxel 2: buffered
        self.assertFalse(heldout[0, 4])  # population 0, voxel 4: buffered
        self.assertFalse(np.any(train & heldout))

    def test_rejects_role_mismatched_and_out_of_range_exclusions(self):
        heldout_voxels = np.array([1, 4], dtype=np.int32)
        with self.assertRaisesRegex(ValueError, 'training exclusion'):
            build_population_exposure_masks(
                2, heldout_voxels,
                train_window_excluded_keys=np.array([1], dtype=np.int32),
                population_count=2)
        with self.assertRaisesRegex(ValueError, 'heldout exclusion'):
            build_population_exposure_masks(
                2, heldout_voxels,
                heldout_window_excluded_keys=np.array([0], dtype=np.int32),
                population_count=2)
        with self.assertRaisesRegex(ValueError, 'outside the key geometry'):
            build_population_exposure_masks(
                2, heldout_voxels,
                train_window_excluded_keys=np.array([16], dtype=np.int32),
                population_count=2)

    def test_rejects_duplicate_heldout_voxel_indices(self):
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            build_population_exposure_masks(
                2, np.array([1, 1], dtype=np.int32), population_count=2)


if __name__ == '__main__':
    unittest.main()
