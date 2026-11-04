import unittest

import numpy as np

from cf4_r2_field_batch_mcse import (
    FieldBatchMCSEAccumulator, MultiScaleFieldMCSEAccumulator)


def field(rho, velocity, valid=None):
    rho = np.asarray(rho, dtype=np.float64).reshape(1, 1, 1)
    velocity = np.asarray(velocity, dtype=np.float64).reshape(3, 1, 1, 1)
    if valid is None:
        valid = rho > 0
    return dict(rho=rho, mean_velocity_km_s=velocity,
        velocity_valid=np.asarray(valid, dtype=bool).reshape(1, 1, 1))


def explicit_sample_sd_mcse(batch_p, batch_y, batch_x, n_states):
    """Delta-method reference from batch means of occupancy, v, and v^2."""
    p = np.asarray(batch_p, dtype=np.float64)
    y = np.asarray(batch_y, dtype=np.float64)
    x = np.asarray(batch_x, dtype=np.float64)
    covariance = np.cov(np.column_stack((p, y, x)), rowvar=False, ddof=1)
    mean_p, mean_y, mean_x = p.mean(), y.mean(), x.mean()
    mean = mean_y / mean_p
    raw_variance = mean_x / mean_p - mean * mean
    count = n_states * mean_p
    correction = count / (count - 1.)
    sd = np.sqrt(raw_variance * correction)
    gradient = np.array([
        correction * (mean * mean - raw_variance) / mean_p
        - n_states * raw_variance / (count - 1.) ** 2,
        -2. * correction * mean / mean_p,
        correction / mean_p,
    ])
    variance_mcse = np.sqrt(gradient @ covariance @ gradient / p.size)
    return sd, variance_mcse / (2. * sd)


class FieldBatchMCSETests(unittest.TestCase):
    def test_density_moments_and_mcse_match_explicit_batch_means(self):
        rng = np.random.default_rng(3107)
        rho = np.maximum(.05, 1. + rng.normal(size=80))
        velocity = rng.normal(size=(3, 80))
        acc = FieldBatchMCSEAccumulator((1, 1, 1), batch_size=8,
            minimum_batches=10)
        for i in range(80):
            acc.update(field(rho[i], velocity[:, i]))
        result = acc.finalize()
        batch_means = rho.reshape(10, 8).mean(axis=1)
        expected_mcse = batch_means.std(ddof=1) / np.sqrt(10)
        batch_second = (rho * rho).reshape(10, 8).mean(axis=1)
        cov = np.cov(np.column_stack((batch_means, batch_second)),
            rowvar=False, ddof=1)
        mean = rho.mean()
        sample_sd = rho.std(ddof=1)
        variance_mcse = (80. / 79.) * np.sqrt(
            (cov[1, 1] + 4 * mean * mean * cov[0, 0]
             - 4 * mean * cov[0, 1]) / 10.)
        self.assertEqual(result["retained_states"], 80)
        self.assertAlmostEqual(float(result["density_mean"][0, 0, 0]), rho.mean(), places=12)
        self.assertAlmostEqual(float(result["density_contrast_mean"][0, 0, 0]),
            rho.mean() - 1., places=12)
        self.assertAlmostEqual(float(result["density_posterior_sd"][0, 0, 0]),
            rho.std(ddof=1), places=12)
        self.assertAlmostEqual(float(result["density_mean_mcse"][0, 0, 0]),
            expected_mcse, places=12)
        self.assertAlmostEqual(float(result["density_sd_mcse"][0, 0, 0]),
            variance_mcse / (2 * sample_sd), places=12)

    def test_velocity_is_conditional_on_occupancy_and_occupancy_is_reported(self):
        rho = [1., 2., 0., 3., 0., 0., 0., 4.]
        vx = [1., 3., 0., 5., 0., 0., 0., 7.]
        acc = FieldBatchMCSEAccumulator((1, 1, 1), batch_size=4,
            minimum_batches=2)
        for r, v in zip(rho, vx):
            acc.update(field(r, [v, -v, 2 * v]))
        result = acc.finalize()
        self.assertEqual(float(result["velocity_occupancy_fraction"][0, 0, 0]), .5)
        self.assertAlmostEqual(float(result["velocity_conditional_mean"][0, 0, 0, 0]), 4.)
        self.assertAlmostEqual(float(result["velocity_conditional_posterior_sd"][0, 0, 0, 0]),
            np.std([1., 3., 5., 7.], ddof=1))
        expected_sd, expected_sd_mcse = explicit_sample_sd_mcse(
            [.75, .25], [2.25, 1.75], [8.75, 12.25], n_states=8)
        covariance = np.cov(np.column_stack((
            [.75, .25], [2.25, 1.75], [8.75, 12.25])),
            rowvar=False, ddof=1)
        mean_gradient = np.array([-4. / .5, 1. / .5, 0.])
        expected_mean_mcse = np.sqrt(
            mean_gradient @ covariance @ mean_gradient / 2.)
        self.assertAlmostEqual(float(result["velocity_conditional_posterior_sd"][0, 0, 0, 0]),
            expected_sd, places=12)
        self.assertAlmostEqual(float(result["velocity_sd_mcse"][0, 0, 0, 0]),
            expected_sd_mcse, places=12)
        self.assertAlmostEqual(float(result["velocity_mean_mcse"][0, 0, 0, 0]),
            expected_mean_mcse, places=12)
        self.assertGreater(float(result["velocity_occupancy_mcse"][0, 0, 0]), 0.)
        self.assertGreater(float(result["velocity_occupancy_posterior_sd"][0, 0, 0]), 0.)
        summary = result["regional_summary"]["velocity_mean_mcse_over_sd"][0]["full_box"]
        self.assertEqual(summary["region_cells"], 1)
        self.assertEqual(summary["defined_cells"], 1)

    def test_repeated_rejected_states_count_as_retained_draws(self):
        acc = FieldBatchMCSEAccumulator((1, 1, 1), batch_size=4,
            minimum_batches=2)
        for _ in range(8):
            acc.update(field(1., [2., 3., 4.]))
        result = acc.finalize()
        self.assertEqual(result["retained_states"], 8)
        self.assertEqual(float(result["density_posterior_sd"][0, 0, 0]), 0.)
        self.assertEqual(float(result["density_mean_mcse_over_sd"][0, 0, 0]), 0.)
        summary = result["regional_summary"]["density_mean_mcse_over_sd"]["full_box"]
        self.assertEqual(summary["zero_posterior_sd_cells"], 1)
        self.assertTrue(np.all(result["velocity_conditional_posterior_sd"] == 0.))
        self.assertTrue(np.all(result["velocity_sd_mcse"] == 0.))

    def test_undefined_velocity_cells_are_counted_not_dropped(self):
        acc = FieldBatchMCSEAccumulator((1, 1, 1), batch_size=2,
            minimum_batches=2)
        for _ in range(4):
            acc.update(field(0., [0., 0., 0.]))
        result = acc.finalize()
        summary = result["regional_summary"]["velocity_mean_mcse_over_sd"][0]["full_box"]
        self.assertEqual(summary["region_cells"], 1)
        self.assertEqual(summary["defined_cells"], 0)
        self.assertEqual(summary["undefined_cells"], 1)
        self.assertTrue(np.isnan(result["velocity_conditional_mean"][0, 0, 0, 0]))

    def test_single_occupied_state_has_no_velocity_sd_or_relative_precision(self):
        acc = FieldBatchMCSEAccumulator((1, 1, 1), batch_size=2,
            minimum_batches=2)
        for rho, v in ((1., 9.), (0., 0.), (0., 0.), (0., 0.)):
            acc.update(field(rho, [v, v, v]))
        result = acc.finalize()
        self.assertTrue(np.isfinite(result["velocity_conditional_mean"][0, 0, 0, 0]))
        self.assertTrue(np.isnan(result["velocity_conditional_posterior_sd"][0, 0, 0, 0]))
        self.assertTrue(np.isnan(result["velocity_mean_mcse_over_sd"][0, 0, 0, 0]))
        summary = result["regional_summary"]["velocity_mean_mcse_over_sd"][0]["full_box"]
        self.assertEqual(summary["undefined_cells"], 1)

    def test_incomplete_batch_and_invalid_field_fail_closed(self):
        acc = FieldBatchMCSEAccumulator((1, 1, 1), batch_size=4,
            minimum_batches=2)
        acc.update(field(1., [0., 0., 0.]))
        with self.assertRaisesRegex(ValueError, "incomplete fixed batch"):
            acc.finalize()
        with self.assertRaisesRegex(ValueError, "finite, nonnegative"):
            acc.update(field(np.nan, [0., 0., 0.]))
        self.assertEqual(acc.states, 1)

    def test_squared_moment_overflow_fails_before_state_is_counted(self):
        acc = FieldBatchMCSEAccumulator((1, 1, 1), batch_size=2,
            minimum_batches=2)
        with self.assertRaisesRegex(ValueError, "moments overflow"):
            acc.update(field(np.finfo(np.float64).max, [0., 0., 0.]))
        self.assertEqual(acc.states, 0)

    def test_memory_contract_matches_native_n256_accumulator(self):
        acc = FieldBatchMCSEAccumulator((256, 256, 256), batch_size=16)
        self.assertEqual(acc.estimated_accumulator_bytes,
            37 * 256 ** 3 * np.dtype(np.float64).itemsize)

    def test_checkpoint_resume_preserves_partial_batch_and_mcse(self):
        states = [
            field(1. + (i % 5), [float(i), -float(i), 2. * float(i)])
            for i in range(24)]
        uninterrupted = FieldBatchMCSEAccumulator((1, 1, 1), batch_size=4,
            minimum_batches=4)
        for item in states:
            uninterrupted.update(item)
        partial = FieldBatchMCSEAccumulator((1, 1, 1), batch_size=4,
            minimum_batches=4)
        for item in states[:7]:
            partial.update(item)
        snapshot = {key: np.array(value, copy=True)
            for key, value in partial.checkpoint_arrays().items()}
        resumed = FieldBatchMCSEAccumulator.from_checkpoint(snapshot)
        for item in states[7:]:
            resumed.update(item)
        expected, actual = uninterrupted.finalize(), resumed.finalize()
        for key in ("density_mean", "density_posterior_sd", "density_mean_mcse",
                    "density_contrast_mean", "density_sd_mcse", "velocity_conditional_mean",
                    "velocity_conditional_posterior_sd", "velocity_mean_mcse",
                    "velocity_sd_mcse", "velocity_occupancy_fraction",
                    "velocity_occupancy_mcse"):
            np.testing.assert_allclose(actual[key], expected[key], rtol=0., atol=0.,
                equal_nan=True, err_msg=key)
        self.assertEqual(actual["retained_states"], 24)

    def test_inconsistent_checkpoint_counters_fail_closed(self):
        acc = FieldBatchMCSEAccumulator((1, 1, 1), batch_size=4,
            minimum_batches=2)
        acc.update(field(1., [1., 2., 3.]))
        state = {key: np.array(value, copy=True)
            for key, value in acc.checkpoint_arrays().items()}
        state["states"] = np.array(9, dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "inconsistent .* counters"):
            FieldBatchMCSEAccumulator.from_checkpoint(state)

    def test_multiscale_requires_complete_batches_at_each_scale(self):
        acc = MultiScaleFieldMCSEAccumulator((1, 1, 1), batch_sizes=(2, 4),
            minimum_batches=2)
        for i in range(10):
            acc.update(field(1. + i % 3, [float(i), 2. * i, -float(i)]))
        result = acc.finalize()
        self.assertEqual(result["retained_states"], 10)
        self.assertEqual(result["scales"]["2"]["status"],
            "FIELD_MCSE_AVAILABLE_NOT_CONVERGENCE_EVIDENCE")
        self.assertEqual(result["scales"]["4"]["status"],
            "INSUFFICIENT_COMPLETE_BATCHES")
        self.assertEqual(result["scales"]["4"]["complete_batches"], 2)
        self.assertEqual(result["scales"]["4"]["pending_states"], 2)
        self.assertEqual(result["estimated_accumulator_bytes"],
            2 * 37 * 8)

    def test_multiscale_checkpoint_resume_matches_uninterrupted_chain(self):
        states = [
            field(1. + (i % 4), [float(i), -float(i), 3. * float(i)])
            for i in range(16)]
        uninterrupted = MultiScaleFieldMCSEAccumulator((1, 1, 1),
            batch_sizes=(2, 4), minimum_batches=2)
        for item in states:
            uninterrupted.update(item)
        partial = MultiScaleFieldMCSEAccumulator((1, 1, 1),
            batch_sizes=(2, 4), minimum_batches=2)
        for item in states[:7]:
            partial.update(item)
        snapshot = {key: np.array(value, copy=True)
            for key, value in partial.checkpoint_arrays().items()}
        resumed = MultiScaleFieldMCSEAccumulator.from_checkpoint(snapshot)
        for item in states[7:]:
            resumed.update(item)
        expected, actual = uninterrupted.finalize(), resumed.finalize()
        for size in ("2", "4"):
            for key in ("density_contrast_mean", "density_posterior_sd",
                        "density_mean_mcse", "density_sd_mcse",
                        "velocity_conditional_mean", "velocity_mean_mcse",
                        "velocity_sd_mcse", "velocity_occupancy_fraction"):
                np.testing.assert_allclose(actual["scales"][size][key],
                    expected["scales"][size][key], rtol=0., atol=0.,
                    equal_nan=True, err_msg=f"scale={size}, key={key}")

    def test_region_mask_must_match_the_field(self):
        acc = FieldBatchMCSEAccumulator((1, 1, 1), batch_size=2,
            minimum_batches=2)
        for _ in range(4):
            acc.update(field(1., [1., 1., 1.]))
        with self.assertRaisesRegex(ValueError, "aligned boolean map"):
            acc.finalize(regions={"bad": np.ones((2, 2, 2), dtype=bool)})


if __name__ == "__main__":
    unittest.main()
