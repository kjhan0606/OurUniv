"""Toy exact-target tests; these do not establish N256 sampler performance."""

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cf4_r2_gaussian_reference_slice import (  # noqa: E402
    gaussian_reference_elliptical_slice_step,
)


class GaussianReferenceSliceTests(unittest.TestCase):
    def test_fixed_reference_residual_recovers_correlated_gaussian_target(self):
        target_mean = np.array([0.35, -0.2, 0.15])
        target_covariance = np.array([
            [1.1, 0.48, -0.16],
            [0.48, 0.9, 0.22],
            [-0.16, 0.22, 0.72],
        ])
        target_precision = np.linalg.inv(target_covariance)

        # Deliberately differs in both mean and covariance from the target.
        reference_mean = np.array([-0.1, 0.25, -0.3])
        reference_covariance = np.array([
            [0.85, -0.12, 0.08],
            [-0.12, 1.25, 0.28],
            [0.08, 0.28, 0.95],
        ])
        reference_cholesky = np.linalg.cholesky(reference_covariance)

        def whiten(vector):
            return np.linalg.solve(reference_cholesky, vector)

        def color(vector):
            return reference_cholesky @ vector

        def log_target(q):
            delta = q - target_mean
            return -0.5 * float(delta @ target_precision @ delta)

        rng = np.random.default_rng(20261029)
        q = target_mean.copy()
        draws = []
        for iteration in range(30000):
            update = gaussian_reference_elliptical_slice_step(
                q, log_target, reference_mean, whiten, color, rng,
                max_evaluations=96,
            )
            q = update.position
            self.assertGreaterEqual(update.evaluations, 1)
            self.assertTrue(np.isfinite(update.log_target))
            if iteration >= 3000:
                draws.append(q.copy())

        draws = np.asarray(draws)
        np.testing.assert_allclose(draws.mean(axis=0), target_mean, rtol=0, atol=0.04)
        np.testing.assert_allclose(
            np.cov(draws, rowvar=False, bias=True),
            target_covariance,
            rtol=0,
            atol=0.08,
        )

    def test_matching_reference_has_constant_residual_and_one_evaluation(self):
        covariance = np.array([[1.3, 0.35], [0.35, 0.8]])
        cholesky = np.linalg.cholesky(covariance)
        mean = np.array([0.2, -0.15])

        def whiten(vector):
            return np.linalg.solve(cholesky, vector)

        def color(vector):
            color_calls.append(None)
            return cholesky @ vector

        precision = np.linalg.inv(covariance)

        target_calls = []
        color_calls = []

        def log_target(q):
            target_calls.append(None)
            delta = q - mean
            return -0.5 * float(delta @ precision @ delta)

        q = mean + cholesky @ np.array([0.4, -0.3])
        update = gaussian_reference_elliptical_slice_step(
            q, log_target, mean, whiten, color, np.random.default_rng(7),
            max_evaluations=1,
        )
        self.assertEqual(update.evaluations, 1)
        self.assertEqual(len(color_calls), 1)
        self.assertEqual(len(target_calls), 2)
        self.assertAlmostEqual(update.log_residual_change, 0.0, places=12)
        white_new = whiten(update.position - mean)
        self.assertAlmostEqual(update.log_target,
                               -0.5 * float(white_new @ white_new), places=12)

    def test_invalid_current_target_fails_closed(self):
        with self.assertRaises(ValueError):
            gaussian_reference_elliptical_slice_step(
                np.array([0.0]), lambda _q: -np.inf, np.array([0.0]),
                lambda x: x, lambda x: x, np.random.default_rng(8),
                max_evaluations=1,
            )


if __name__ == "__main__":
    unittest.main()
