import inspect
import unittest

import numpy as np
from cf4_lg_field_candidates import identify, match_for_evaluation, aperture_offsets


class FieldCandidateTests(unittest.TestCase):
    def field(self):
        field = np.zeros((7, 16, 16, 16))
        for x, mass in [(3, 10.), (6, 8.), (9, 2.)]:
            field[:, x, 6, 6] = mass*np.array([1., 2., -3., 4., 8., 13., 20.])
        return field

    def test_field_only_interface_and_aperture_moments(self):
        self.assertEqual(set(inspect.signature(identify).parameters), {'field', 'dx', 'observer_cell', 'h'})
        field = self.field()
        result = identify(field, 1., [[3, 6, 6], [4, 7, 7]], h=1.)
        self.assertEqual(len(result['mw_candidates']), 1)
        self.assertEqual(len(result['cells']), 3)
        self.assertGreater(result['triplet_count'], 0)
        for radius in (1, 2):
            for i, cell in enumerate(result['cells']):
                xyz = cell+aperture_offsets(radius)
                native = field[:, xyz[:, 0], xyz[:, 1], xyz[:, 2]].sum(1)
                np.testing.assert_allclose(result['moments'][radius-1, :, i], native)
        np.testing.assert_allclose(result['mean_velocity'], np.broadcast_to(np.array([2., -3., 4.])[None, :, None], (2, 3, 3)))
        np.testing.assert_allclose(result['physical_sigma'], 2.)

    def test_shared_peak_unmatched_and_empty_observer(self):
        match = match_for_evaluation([[3.5, 6.5, 6.5], [6.5, 6.5, 6.5]],
            [[3.5, 6.5, 6.5], [6.5, 6.5, 6.5], [6.6, 6.5, 6.5]], 1.)
        self.assertEqual(sum(i >= 0 for i in match['matched']), 2)
        self.assertIn([1, 2], match['shared_nearest_pairs'])
        result = identify(self.field(), 1., [[11, 11, 11], [12, 12, 12]], h=1.)
        self.assertEqual(len(result['cells']), 0)
        self.assertEqual(result['triplet_count'], 0)
        self.assertEqual(match_for_evaluation([], [[1, 2, 3]], 1.)['matched'], [-1])


if __name__ == '__main__':
    unittest.main()
