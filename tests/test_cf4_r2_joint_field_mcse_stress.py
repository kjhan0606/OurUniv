import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cf4_r2_joint_field_mcse_stress import (
    _occupancy_step,
    _one_replicate,
    _one_sided_binomial_upper,
    _regional_gate_summary,
    _stress_status,
    conditional_velocity_mean_variance,
    estimate_velocity_mean_oracle,
    reporting_regions,
    stationary_gaussian_path,
)
from cf4_r2_mcse_spatial_calibration import correlated_gaussian_sequence


class JointFieldMCSEStressTests(unittest.TestCase):
    def test_fixed_central_reporting_region_has_147_cells(self):
        regions = reporting_regions(16)
        self.assertEqual(int(regions["training_region"].sum()), 147)
        self.assertEqual(int(regions["full_box"].sum()), 16**3)

    def test_occupancy_process_has_stationary_marginal_and_lag_one_persistence(self):
        rng = np.random.default_rng(20261105)
        probability = 0.2
        persistence = 0.2
        previous = rng.random(100_000) < probability
        current = _occupancy_step(rng, previous, probability, persistence)
        self.assertAlmostEqual(float(previous.mean()), probability, delta=0.006)
        self.assertAlmostEqual(float(current.mean()), probability, delta=0.006)
        centered_previous = previous.astype(float) - previous.mean()
        centered_current = current.astype(float) - current.mean()
        empirical_persistence = float(np.mean(centered_previous * centered_current)
            / np.mean(centered_previous**2))
        self.assertAlmostEqual(empirical_persistence, persistence, delta=0.015)

    def test_gaussian_path_starts_from_the_same_filtered_law_as_innovations(self):
        seed = 9041
        draws = 5
        size = 8
        persistence = 0.9
        correlation_length = 1.5
        path = stationary_gaussian_path(np.random.default_rng(seed),
            draws=draws, size=size, persistence=persistence,
            correlation_length=correlation_length)
        innovations = correlated_gaussian_sequence(
            np.random.default_rng(seed), draws, size, correlation_length)
        np.testing.assert_allclose(path[0], innovations[0], rtol=0., atol=0.)
        expected_next = (persistence * innovations[0]
            + np.sqrt(1. - persistence**2) * innovations[1])
        np.testing.assert_allclose(path[1], expected_next, rtol=1e-14, atol=1e-14)

    def test_conditional_velocity_variance_recurrence_matches_direct_quadratic(self):
        occupancy = np.asarray([
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
        ], dtype=bool)
        phi = 0.73
        count = int(occupancy.sum())
        direct_q = 0.0
        for chain in range(occupancy.shape[0]):
            for left in range(occupancy.shape[1]):
                for right in range(occupancy.shape[1]):
                    direct_q += (occupancy[chain, left]
                        * occupancy[chain, right]
                        * phi ** abs(left - right))
        expected = direct_q / count**2
        self.assertAlmostEqual(
            conditional_velocity_mean_variance(occupancy, phi),
            expected, places=14)

    def test_conditional_velocity_variance_rejects_zero_occupancy(self):
        with self.assertRaisesRegex(ValueError, "M=0"):
            conditional_velocity_mean_variance(np.zeros((4, 16), dtype=bool), 0.9)

    def test_streamed_oracle_matches_enumerated_two_state_histories(self):
        probability = 0.2
        occupancy_persistence = 0.2
        phi = (0.0, 0.5, 0.9)
        report = estimate_velocity_mean_oracle(histories=50_000,
            chunk_size=4096, chains=1, draws=2, probability=probability,
            occupancy_persistence=occupancy_persistence,
            velocity_persistence=phi, seed=813)
        self.assertEqual(report["histories_processed"], 50_000)
        self.assertGreater(report["components"]["x"][
            "zero_occupancy_history_count"], 0)
        p00 = (1. - probability) * (1. - probability
            * (1. - occupancy_persistence))
        p01 = (1. - probability) * probability * (1. - occupancy_persistence)
        p10 = probability * (1. - (probability
            + occupancy_persistence * (1. - probability)))
        p11 = probability * (probability
            + occupancy_persistence * (1. - probability))
        nonzero_probability = 1. - p00
        for label, persistence in zip(("x", "y", "z"), phi):
            exact = (p01 + p10 + p11 * 0.5 * (1. + persistence))
            exact /= nonzero_probability
            component = report["components"][label]
            self.assertAlmostEqual(component["zero_occupancy_probability"],
                p00, places=14)
            self.assertLessEqual(abs(component["variance_ratio_mean"] - exact),
                6. * component["variance_ratio_mcse"])

    def test_regional_gate_fails_closed_for_undefined_or_nonfinite_cells(self):
        region = np.ones((2, 2, 2), dtype=bool)
        values = np.full((2, 2, 2), 0.05)
        defined = np.ones_like(region)
        self.assertTrue(_regional_gate_summary(values, defined, region, 0.1)["passed"])
        defined[0, 0, 0] = False
        self.assertFalse(_regional_gate_summary(values, defined, region, 0.1)["passed"])
        defined[0, 0, 0] = True
        values[0, 0, 0] = np.inf
        self.assertFalse(_regional_gate_summary(values, defined, region, 0.1)["passed"])

    def test_zero_false_acceptances_give_exact_one_sided_binomial_bound(self):
        upper = _one_sided_binomial_upper(0, 256, 0.05)
        self.assertAlmostEqual(upper, 1. - 0.05 ** (1. / 256), places=14)
        self.assertLess(upper, 0.05)
        self.assertGreater(_one_sided_binomial_upper(4, 256, 0.05), upper)

    def test_zero_passes_do_not_bypass_unmet_risk_bound(self):
        upper = _one_sided_binomial_upper(0, 1, 0.05)
        self.assertAlmostEqual(upper, 0.95, places=14)
        self.assertGreater(upper, 0.05)
        self.assertEqual(_stress_status(risk_passed=upper <= 0.05,
            g_field_passes=0),
            "STRESS_CASE_RISK_BOUND_NOT_MET_BUT_GATE_REJECTED_ALL_REPLICATES")
        self.assertEqual(_stress_status(risk_passed=True, g_field_passes=0),
            "STRESS_CASE_RISK_BOUND_MET_BUT_GATE_REJECTED_ALL_REPLICATES")
        self.assertEqual(_stress_status(risk_passed=True, g_field_passes=1),
            "STRESS_CASE_RISK_BOUND_MET")

    def test_one_replicate_preserves_paired_gates_and_checks_cross_chain_once(self):
        size = 4
        regions = reporting_regions(size)
        outcome = _one_replicate(0, seed=20261105, size=size, draws=16,
            chains=2, batch_sizes=(2, 4), minimum_batches=2,
            correlation_length=0.0, lognormal_sigma=0.02,
            regions=regions, threshold=0.1)
        self.assertEqual(len(outcome["mcse_components"]), 2 * 2 * 9)
        self.assertEqual(set(outcome["cross_chain"]["quantities"]), {
            "density", "occupancy", "velocity_x", "velocity_y", "velocity_z"})
        self.assertIsInstance(outcome["g_field_pass"], bool)
        self.assertEqual(outcome["g_field_pass"],
            outcome["mcse_only_pass"] and outcome["cross_chain_pass"])


if __name__ == "__main__":
    unittest.main()
