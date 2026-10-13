"""Focused algebra/support checks; no actual-data likelihood claim."""

import math
import unittest

import numpy as np

from cf4_r2_point_process import PointProcessSupportError, log_point_process


class PointProcessTests(unittest.TestCase):
    def test_cell_count_decomposition(self):
        u = np.ones((6, 2))
        u[0] = [3.0, 4.0]
        exposure = np.full((6, 2), 0.5)
        population = np.array([0, 0, 0], dtype=int)
        cell = np.array([0, 0, 1], dtype=int)
        selection = np.array([0.2, 0.6, 0.4])
        volume = 8.0
        point = log_point_process(u, exposure, population, cell, selection, volume)
        mean = u * exposure
        count = -mean.sum() + 2 * math.log(mean[0, 0]) + math.log(mean[0, 1]) - math.lgamma(3)
        correction = sum(math.log(s / (volume * exposure[0, c]))
                         for s, c in zip(selection, cell)) + math.lgamma(3)
        self.assertAlmostEqual(point, count + correction)

    def test_zero_support_fails_closed(self):
        u = np.ones((6, 1))
        e = np.ones((6, 1))
        with self.assertRaises(PointProcessSupportError):
            log_point_process(u, e, np.array([0]), np.array([0]), np.array([0.0]), 1.0)
        e[0, 0] = 0.0
        with self.assertRaises(PointProcessSupportError):
            log_point_process(u, e, np.array([0]), np.array([0]), np.array([0.2]), 1.0)


if __name__ == "__main__":
    unittest.main()
