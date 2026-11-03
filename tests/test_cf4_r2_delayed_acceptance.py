"""Synthetic invariance and Fourier-map tests; not CF4 target validation."""

import numpy as np
import unittest
import jax
import jax.numpy as jnp

from cf4_r2_delayed_acceptance import (
    adjoint_restriction_white_field,
    delayed_acceptance_stage2_accept,
    delayed_acceptance_stage2_log_alpha,
    lift_restricted_white_field,
    restrict_white_field,
)
from cf4_r2_prior_split_hmc import inverse_laplacian_metric_symbol
from cf4_lg_highk_conditional_field import (
    add_restriction_adjoint_preserve_dtype,
    restrict_spectrum_preserve_dtype,
)


class DelayedAcceptanceTests(unittest.TestCase):
    def test_restriction_adjoint_is_coisometric_including_nyquist_modes(self):
        rng = np.random.default_rng(20261103)
        fine_probe = rng.normal(size=(8, 8, 8))
        coarse = rng.normal(size=(4, 4, 4))
        coarse_spectrum = np.fft.fftn(coarse, norm="ortho")
        fine_spectrum = add_restriction_adjoint_preserve_dtype(
            np.zeros((8, 8, 8), dtype=np.complex128), coarse_spectrum
        )
        fine = np.fft.ifftn(fine_spectrum, norm="ortho")
        self.assertLess(np.sqrt(np.mean(fine.imag ** 2)), 1e-13)
        restricted_spectrum = restrict_spectrum_preserve_dtype(fine_spectrum, 4)
        np.testing.assert_allclose(restricted_spectrum, coarse_spectrum, rtol=0, atol=2e-13)
        np.testing.assert_allclose(np.sum(np.abs(fine_spectrum) ** 2),
                                   np.sum(np.abs(coarse_spectrum) ** 2), rtol=2e-13)
        lhs = np.vdot(restrict_white_field(fine_probe, 4), coarse)
        rhs = np.vdot(fine_probe, adjoint_restriction_white_field(coarse, 8))
        np.testing.assert_allclose(lhs, rhs, rtol=0, atol=2e-13)


    def test_lift_changes_only_restricted_coordinates_and_obeys_prior_norm_identity(self):
        rng = np.random.default_rng(20261104)
        fine = rng.normal(size=(8, 8, 8))
        current = restrict_white_field(fine, 4)
        coarse_new = current + 0.2 * rng.normal(size=(4, 4, 4))
        lifted, info = lift_restricted_white_field(fine, coarse_new)
        np.testing.assert_allclose(restrict_white_field(lifted, 4), coarse_new,
                                   rtol=0, atol=2e-11)
        complement_before = fine - adjoint_restriction_white_field(current, 8)
        complement_after = lifted - adjoint_restriction_white_field(
            restrict_white_field(lifted, 4), 8
        )
        np.testing.assert_allclose(complement_after, complement_before, rtol=0, atol=2e-11)
        expected_norm_change = np.sum(coarse_new ** 2) - np.sum(current ** 2)
        np.testing.assert_allclose(np.sum(lifted ** 2) - np.sum(fine ** 2),
                                   expected_norm_change, rtol=2e-10, atol=2e-10)
        self.assertLess(info["restricted_reconstruction_relative_rms"], 2e-11)
        self.assertLess(info["prior_norm_identity_relative_error"], 5e-10)


    def test_n128_metric_is_exact_restriction_of_n256_metric(self):
        rng = np.random.default_rng(20261105)
        coarse = rng.normal(size=(4, 4, 4))
        coarse_spectrum = np.fft.fftn(coarse, norm="ortho")
        lifted_spectrum = add_restriction_adjoint_preserve_dtype(
            np.zeros((8, 8, 8), dtype=np.complex128), coarse_spectrum
        )
        fine_metric = inverse_laplacian_metric_symbol(8, fundamental_mass=6000.0)
        coarse_metric = inverse_laplacian_metric_symbol(4, fundamental_mass=6000.0)
        fine_applied = np.fft.ifftn(fine_metric * lifted_spectrum, norm="ortho")
        restricted = restrict_spectrum_preserve_dtype(np.fft.fftn(fine_applied, norm="ortho"), 4)
        np.testing.assert_allclose(restricted, coarse_metric * coarse_spectrum,
                                   rtol=3e-13, atol=3e-13)


    def test_two_stage_transition_obeys_fine_detailed_balance_and_rejection_paths(self):
        # Three discrete states make both stage-one and stage-two rejection occur.
        log_surrogate = np.log(np.array([0.55, 0.30, 0.15]))
        log_fine = np.log(np.array([0.18, 0.52, 0.30]))
        proposal = np.array([[0.0, 0.45, 0.15],
                             [0.45, 0.0, 0.40],
                             [0.15, 0.40, 0.0]])
        proposal[np.diag_indices_from(proposal)] = 1.0 - proposal.sum(axis=1)
        stage_one = _stage_one_kernel(log_surrogate, proposal)
        np.testing.assert_allclose(np.exp(log_surrogate)[:, None] * stage_one,
                                   (np.exp(log_surrogate)[:, None] * stage_one).T,
                                   rtol=0, atol=2e-16)

        delayed = np.zeros_like(stage_one)
        for i in range(len(log_fine)):
            for j in range(len(log_fine)):
                if i == j:
                    continue
                delta_s = log_surrogate[i] - log_surrogate[j]
                delta_f = log_fine[i] - log_fine[j]
                log_alpha = delayed_acceptance_stage2_log_alpha(delta_f, delta_s)
                delayed[i, j] = stage_one[i, j] * np.exp(log_alpha)
            delayed[i, i] = 1.0 - delayed[i].sum()
        fine_mass = np.exp(log_fine)
        np.testing.assert_allclose(fine_mass[:, None] * delayed,
                                   (fine_mass[:, None] * delayed).T,
                                   rtol=0, atol=2e-16)
        np.testing.assert_allclose(delayed.sum(axis=1), 1.0, rtol=0, atol=2e-16)
        self.assertTrue(np.any(stage_one.diagonal() > 0.0))
        self.assertTrue(np.any(delayed.diagonal() > stage_one.diagonal()))

        accepted, log_alpha = delayed_acceptance_stage2_accept(2.0, 0.0, -0.1)
        self.assertFalse(accepted)
        self.assertEqual(log_alpha, -2.0)
        accepted, log_alpha = delayed_acceptance_stage2_accept(-2.0, 0.0, -0.1)
        self.assertTrue(accepted)
        self.assertEqual(log_alpha, 0.0)
        self.assertEqual(delayed_acceptance_stage2_log_alpha(np.inf, 0.0), -np.inf)


    def test_training_count_surrogate_floor_is_fixed_and_differentiable(self):
        from cf4_r2_da_count_surrogate_screen import EPSILON, _count_score_with_floor

        keys = jnp.asarray([1], dtype=jnp.int32)
        counts = jnp.asarray([2.0])
        exposure = jnp.asarray([True, True, True])

        def score(values):
            return _count_score_with_floor(values.reshape(1, 3), keys, counts, exposure)[0]

        intensity = jnp.asarray([0.0, EPSILON / 100.0, 2.0])
        value = score(intensity)
        gradient = jax.grad(score)(intensity)
        floored, minimum, maximum = _count_score_with_floor(
            intensity.reshape(1, 3), keys, counts, exposure
        )[1:]
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())
        self.assertEqual(int(floored), 1)
        self.assertAlmostEqual(float(minimum), EPSILON / 100.0, delta=EPSILON / 1e6)
        self.assertAlmostEqual(float(maximum), EPSILON / 100.0, delta=EPSILON / 1e6)


def _stage_one_kernel(log_surrogate, symmetric_proposal):
    """Build an explicit reversible MH kernel for a finite-state test."""
    size = len(log_surrogate)
    kernel = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            ratio = log_surrogate[j] - log_surrogate[i]
            kernel[i, j] = symmetric_proposal[i, j] * min(1.0, np.exp(ratio))
        kernel[i, i] = 1.0 - kernel[i].sum()
    return kernel


if __name__ == "__main__":
    unittest.main()
