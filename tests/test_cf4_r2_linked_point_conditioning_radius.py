import unittest

import numpy as np

from cf4_r2_linked_fp_sparse_train import linked_point_conditioning_radii


class LinkedPointConditioningRadiusTests(unittest.TestCase):
    def setUp(self):
        self.chosen = [('G1', 0, 2, 5, 0), ('G2', 1, 0, 8, 1)]
        self.point = {'radius_cMpc_h': np.asarray([12., 24., 37., 49.])}

    def test_radius_is_taken_from_the_aligned_secure_count_point(self):
        result = linked_point_conditioning_radii(
            self.chosen, self.point, [501, 502], [501, 502])
        np.testing.assert_array_equal(result, [37., 12.])

    def test_pgc_order_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'order/identity mismatch'):
            linked_point_conditioning_radii(
                self.chosen, self.point, [501, 502], [502, 501])

    def test_invalid_linked_radius_fails_closed(self):
        point = {'radius_cMpc_h': np.asarray([12., 24., 0., 49.])}
        with self.assertRaisesRegex(ValueError, 'invalid linked-point'):
            linked_point_conditioning_radii(
                self.chosen, point, [501, 502], [501, 502])


if __name__ == '__main__':
    unittest.main()
