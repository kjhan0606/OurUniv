"""Small exact-target mechanics tests; not evidence of N256 mixing."""

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cf4_r2_block_elliptical_slice import (  # noqa: E402
    SliceEvaluationLimit,
    elliptical_slice_step,
    real_fourier_band_projectors,
)


def coordinate_projector(size, indices):
    indices = np.asarray(indices, dtype=int)

    def project(vector):
        result = np.zeros_like(vector)
        result[indices] = vector[indices]
        return result

    project.dimension = len(indices)
    return project


class BlockEllipticalSliceTests(unittest.TestCase):
    def test_fourier_blocks_are_orthogonal_idempotent_and_close(self):
        n = 8
        low, high, nuisance = real_fourier_band_projectors(n, 2, nuisance_count=3)
        self.assertEqual(low.dimension + high.dimension, n ** 3)
        self.assertEqual(nuisance.dimension, 3)
        q = np.random.default_rng(15).standard_normal(n ** 3 + 3)
        q_low, q_high, q_nuisance = low(q), high(q), nuisance(q)
        np.testing.assert_allclose(q_low + q_high + q_nuisance, q, rtol=0, atol=2e-14)
        np.testing.assert_allclose(low(q_low), q_low, rtol=0, atol=2e-14)
        np.testing.assert_allclose(high(q_high), q_high, rtol=0, atol=2e-14)
        np.testing.assert_allclose(nuisance(q_nuisance), q_nuisance, rtol=0, atol=0)
        self.assertAlmostEqual(float(q_low @ q_high), 0.0, places=12)
        self.assertAlmostEqual(float(q_low @ q_nuisance), 0.0, places=12)
        self.assertAlmostEqual(float(q_high @ q_nuisance), 0.0, places=12)

    def test_sequential_blocks_recover_correlated_gaussian_target(self):
        precision_likelihood = np.array([[2.0, 0.8], [0.8, 1.4]])
        posterior_covariance = np.linalg.inv(np.eye(2) + precision_likelihood)

        def log_likelihood(q):
            return -0.5 * float(q @ precision_likelihood @ q)

        blocks = (coordinate_projector(2, [0]), coordinate_projector(2, [1]))
        rng = np.random.default_rng(7321)
        q = np.array([1.2, -0.7])
        loglike = log_likelihood(q)
        draws = []
        for iteration in range(5200):
            for project in blocks:
                update = elliptical_slice_step(
                    q, loglike, log_likelihood, project, rng, max_evaluations=64)
                q, loglike = update.position, update.log_likelihood
            if iteration >= 400:
                draws.append(q.copy())

        draws = np.asarray(draws)
        np.testing.assert_allclose(draws.mean(axis=0), 0.0, rtol=0, atol=0.08)
        np.testing.assert_allclose(draws.T @ draws / len(draws),
                                   posterior_covariance, rtol=0, atol=0.07)

    def test_budget_exhaustion_is_not_a_completed_update_and_does_not_mutate(self):
        q = np.array([0.2, -0.1])
        original = q.copy()
        with self.assertRaises(SliceEvaluationLimit) as raised:
            elliptical_slice_step(
                q, 0.0, lambda _: -np.inf,
                coordinate_projector(2, [0]), np.random.default_rng(11),
                max_evaluations=2)
        self.assertEqual(raised.exception.attempts, 2)
        np.testing.assert_array_equal(q, original)

    def test_nan_likelihood_fails_closed(self):
        with self.assertRaises(FloatingPointError):
            elliptical_slice_step(
                np.array([0.3]), 0.0, lambda _: np.nan,
                coordinate_projector(1, [0]), np.random.default_rng(17),
                max_evaluations=1)

    def test_true_zero_likelihood_shrinks_slice_bracket_normally(self):
        calls = []

        def likelihood(_):
            calls.append(None)
            return -np.inf if len(calls) == 1 else 0.0

        result = elliptical_slice_step(
            np.array([0.3]), 0.0, likelihood, coordinate_projector(1, [0]),
            np.random.default_rng(23), max_evaluations=2)
        self.assertEqual(result.evaluations, 2)
        self.assertEqual(result.bracket_contractions, 1)
        self.assertEqual(len(calls), 2)

    def test_zero_current_block_can_move_under_its_prior(self):
        q = np.zeros(2)
        update = elliptical_slice_step(
            q, 0.0, lambda _: 0.0, coordinate_projector(2, [0]),
            np.random.default_rng(19), max_evaluations=1)
        self.assertNotEqual(float(update.position[0]), 0.0)
        self.assertEqual(float(update.position[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
