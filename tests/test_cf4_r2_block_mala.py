"""Deterministic block-MALA math tests; not sampler/mixing evidence."""

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cf4_r2_block_mala import (  # noqa: E402
    block_gaussian_logpdf,
    block_mala_log_acceptance,
    block_mala_step,
)


def coordinate_projector(size, indices):
    indices = np.asarray(indices, dtype=int)

    def project(vector):
        value = np.asarray(vector, dtype=np.float64)
        result = np.zeros_like(value)
        result[indices] = value[indices]
        return result

    project.dimension = int(len(indices))
    project.block_name = "test_coordinates"
    return project


class FixedRNG:
    def __init__(self, noise, uniform):
        self.noise = np.asarray(noise, dtype=np.float64)
        self.uniform_value = float(uniform)

    def standard_normal(self, shape):
        if tuple(shape) != self.noise.shape:
            raise AssertionError("unexpected Gaussian-noise shape")
        return self.noise.copy()

    def random(self):
        return self.uniform_value


class BlockMALATests(unittest.TestCase):
    @staticmethod
    def gaussian(value):
        precision = np.array([[2.4, 0.7], [0.7, 1.6]])
        mean = np.array([0.6, -0.2])
        delta = value - mean
        return 0.5 * float(delta @ precision @ delta), precision @ delta

    def test_nonzero_drift_proposal_and_block_isolation(self):
        project = coordinate_projector(2, [0])
        current = np.array([0.15, 0.8])
        value, gradient = self.gaussian(current)
        epsilon = 0.18
        noise = np.array([0.75, -99.0])
        block_noise = np.array([0.75, 0.0])
        projected_gradient = np.array([gradient[0], 0.0])
        expected_delta = -0.5 * epsilon ** 2 * projected_gradient + epsilon * block_noise
        expected_proposal = current + expected_delta
        calls = []

        def evaluate(candidate):
            calls.append(candidate.copy())
            return self.gaussian(candidate)

        result = block_mala_step(
            current, value, gradient, evaluate, project,
            FixedRNG(noise, 1e-100), step_size=epsilon)

        np.testing.assert_allclose(result.proposed_position, expected_proposal,
                                   rtol=0, atol=1e-15)
        self.assertEqual(result.proposed_position[1], current[1])
        self.assertEqual(len(calls), 1)
        self.assertTrue(result.accepted)
        np.testing.assert_array_equal(result.position, result.proposed_position)
        np.testing.assert_array_equal(result.gradient, result.proposed_gradient)
        self.assertGreater(result.proposal_displacement_rms, 0.0)
        self.assertGreater(result.accepted_displacement_rms, 0.0)
        self.assertGreater(result.retained_displacement_rms, 0.0)

    def test_forward_reverse_density_is_asymmetric_and_ratio_is_pairwise_balanced(self):
        project = coordinate_projector(2, [0])
        x = np.array([-0.3, 0.25])
        y = np.array([0.17, 0.25])
        ux, gx = self.gaussian(x)
        uy, gy = self.gaussian(y)
        epsilon = 0.22
        log_q_y_x = block_gaussian_logpdf(y, x, gx, project, epsilon)
        log_q_x_y = block_gaussian_logpdf(x, y, gy, project, epsilon)
        self.assertGreater(abs(log_q_y_x - log_q_x_y), 1e-5)

        log_alpha_xy, forward_xy, reverse_xy = block_mala_log_acceptance(
            x, ux, gx, y, uy, gy, project, epsilon)
        log_alpha_yx, forward_yx, reverse_yx = block_mala_log_acceptance(
            y, uy, gy, x, ux, gx, project, epsilon)
        self.assertAlmostEqual(forward_xy, log_q_y_x, places=14)
        self.assertAlmostEqual(reverse_xy, log_q_x_y, places=14)
        self.assertAlmostEqual(forward_yx, log_q_x_y, places=14)
        self.assertAlmostEqual(reverse_yx, log_q_y_x, places=14)
        self.assertAlmostEqual(log_alpha_xy, -log_alpha_yx, places=13)
        lhs = -ux + log_q_y_x + min(0.0, log_alpha_xy)
        rhs = -uy + log_q_x_y + min(0.0, log_alpha_yx)
        # Pairwise detailed balance includes both the target and MH acceptance.
        self.assertAlmostEqual(lhs, rhs, places=13)

    def test_rejection_is_retained_self_transition_and_preserves_gradient_cache(self):
        project = coordinate_projector(2, [0])
        current = np.array([0.3, -0.6])
        gradient = np.array([0.4, -1.2])
        calls = []

        def expensive_evaluate(candidate):
            calls.append(candidate.copy())
            return 1e6, np.array([0.0, 0.0])

        result = block_mala_step(
            current, 0.0, gradient, expensive_evaluate, project,
            FixedRNG([1.0, 0.0], 0.5), step_size=0.2)

        self.assertEqual(len(calls), 1)
        self.assertFalse(result.accepted)
        np.testing.assert_array_equal(result.position, current)
        np.testing.assert_array_equal(result.gradient, gradient)
        self.assertEqual(result.potential, 0.0)
        self.assertGreater(result.proposal_displacement_rms, 0.0)
        self.assertEqual(result.accepted_displacement_rms, 0.0)
        self.assertEqual(result.retained_displacement_rms, 0.0)

    def test_rejects_displacement_outside_projected_block(self):
        project = coordinate_projector(2, [0])
        with self.assertRaises(ValueError):
            block_gaussian_logpdf(
                [0.1, 0.1], [0.0, 0.0], [0.2, 0.4], project, 0.1)

    def test_nonfinite_gradient_fails_before_target_evaluation(self):
        project = coordinate_projector(2, [0])
        calls = []
        with self.assertRaises(ValueError):
            block_mala_step(
                np.zeros(2), 0.0, np.array([np.nan, 0.0]),
                lambda candidate: calls.append(candidate), project,
                FixedRNG([1.0, 1.0], 0.5), step_size=0.1)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
