import unittest
import numpy as np
from cf4_continuous_matter import ContinuousMatter, boost_scale, transport, check_realizable, restrict


class ContinuousTests(unittest.TestCase):
    def fixture(self):
        n = 8
        total = np.zeros((7, n, n, n))
        total[0] = 2
        total[4:] = 6
        value = np.array([[4.], [8.], [-4.], [0.], [20.], [8.], [4.]])
        comp = dict(cell=np.array([[3, 3, 3]]), moments=value,
                    position_cMpc_h=np.array([3.5]*3), subhalo_id=1)
        total[:, 3, 3, 3] += value[:, 0]
        return ContinuousMatter(total, [comp], dx=1), total

    def test_overlap_and_boost_independent_expected_values(self):
        value = np.array([[4.], [8.], [-4.], [0.], [20.], [8.], [4.]])
        cell, moved = transport(np.array([[3, 3, 3]]), value, [.25, 0, 0], 8)
        np.testing.assert_array_equal(cell, [[3, 3, 3], [4, 3, 3]])
        np.testing.assert_allclose(moved, value * [.75, .25])
        out = boost_scale(value, 2, [1, 2, -1])
        np.testing.assert_allclose(out[:, 0], [8, 24, 8, -8, 80, 16, 16])
        check_realizable(out)
        with self.assertRaises(ValueError):
            transport(np.array([[0, 3, 3]]), value, [-.01, 0, 0], 8)

    def test_joint_change_conservation_selection_and_restriction(self):
        model, native = self.fixture()
        zero, _, _ = model.evaluate(np.zeros((1, 7)))
        np.testing.assert_allclose(zero, native, atol=1e-10)
        theta = np.array([[.3, -.2, .4, .2, 3, 4, 5]])
        value, marks, _ = model.evaluate(theta)
        check_realizable(value)
        np.testing.assert_allclose(value[:4].sum(axis=(1, 2, 3)), native[:4].sum(axis=(1, 2, 3)), atol=1e-10)
        np.testing.assert_allclose(marks[0]['position_cMpc_h'], [3.8, 3.3, 3.9])
        self.assertFalse(marks[0]['resolved_halos'])
        coarse, _, _ = model.evaluate(theta, ratio=2)
        np.testing.assert_allclose(coarse, restrict(value, 2), atol=1e-10)
        keys = np.arange(0, 8**3, 2)
        np.testing.assert_allclose(model.selected(theta, keys), value.reshape(7, -1)[:, keys])
        # Continuity across integer-cell offsets, not just positive shifts.
        left, _, _ = model.evaluate(np.array([[1-1e-8, 0, 0, 0, 0, 0, 0]]))
        right, _, _ = model.evaluate(np.array([[1+1e-8, 0, 0, 0, 0, 0, 0]]))
        self.assertLess(np.max(abs(left-right)), 1e-6)
        with self.assertRaises(ValueError):
            model.evaluate(np.array([[0, 0, 0, 10, 0, 0, 0]]))


if __name__ == '__main__':
    unittest.main()
