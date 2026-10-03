import unittest
import numpy as np
from cf4_patch_symmetry import cube_rotations, rotate_moments, rotate_features, source_groups, grouped_support


class PatchSymmetryTests(unittest.TestCase):
    def test_rotations_preserve_physics_and_match_features(self):
        rng = np.random.default_rng(23)
        mass = rng.uniform(1, 2, (2,) * 3)
        velocity = rng.normal(size=(3, 2, 2, 2))
        value = np.concatenate([mass[None], mass * velocity, mass * (velocity**2 + 3)], axis=0)
        feature = np.r_[np.log(mass).ravel(), velocity.ravel()][None]
        self.assertEqual(len(cube_rotations()), 24)
        for perm, signs in cube_rotations():
            out = rotate_moments(value, perm, signs)
            got = np.r_[np.log(out[0]).ravel(), (out[1:4] / out[0]).ravel()][None]
            np.testing.assert_allclose(got, rotate_features(feature, perm, signs))
            np.testing.assert_allclose(out[0].sum(), mass.sum())
            np.testing.assert_allclose(out[4:7] / out[0] - (out[1:4] / out[0])**2, 3)

    def test_duplicates_do_not_manufacture_group_support(self):
        result = grouped_support(np.full(24, 1 / 24), np.zeros(24, dtype=int))
        self.assertAlmostEqual(result['ess'], 1.)
        self.assertAlmostEqual(result['max_weight'], 1.)
        np.testing.assert_array_equal(source_groups([[0, 0, 0], [1, 1, 1]]), [0, 0])


if __name__ == '__main__':
    unittest.main()
