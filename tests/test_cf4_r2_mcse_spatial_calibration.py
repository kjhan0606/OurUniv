import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cf4_r2_mcse_spatial_calibration import (
    correlated_gaussian_sequence,
    exact_reference_maps,
    run_spatial_calibration,
    synthetic_design,
)


class SpatialMCSECalibrationTests(unittest.TestCase):
    def test_fourier_fields_have_unit_marginals_and_spatial_correlation(self):
        fields = correlated_gaussian_sequence(np.random.default_rng(123),
            128, 8, 1.0)
        self.assertAlmostEqual(float(fields.mean()), 0.0, delta=0.04)
        self.assertAlmostEqual(float(fields.std()), 1.0, delta=0.04)
        neighbor = float(np.mean(fields * np.roll(fields, 1, axis=1)))
        distant = float(np.mean(fields * np.roll(fields, 4, axis=1)))
        self.assertGreater(neighbor, distant + 0.15)

    def test_occupancy_reference_reduces_to_finite_ar1_mean_error(self):
        shape = (1, 1, 1)
        probability = np.full(shape, 0.37)
        phi_occ = np.full(shape, 0.8)
        design = dict(probability=probability,
            phi_occupancy=phi_occ,
            phi_density=np.full(shape, 0.6),
            phi_velocity=np.full((3,) + shape, 0.7))
        report = exact_reference_maps(design, draws=32, chains=4)
        expected_variance_ratio = (
            32 + 2 * sum((32 - lag) * 0.8**lag for lag in range(1, 32))) / 32**2
        self.assertAlmostEqual(float(report["occupancy"][0, 0, 0]),
            np.sqrt(expected_variance_ratio / 4), places=13)

    def test_continuous_references_reduce_to_gaussian_ar1_limits(self):
        shape = (1, 1, 1)
        phi = 0.7
        design = dict(probability=np.ones(shape),
            phi_occupancy=np.full(shape, 0.6),
            phi_density=np.full(shape, phi),
            phi_velocity=np.full((3,) + shape, phi))
        report = exact_reference_maps(design, draws=128, chains=4,
            lognormal_sigma=0.01)
        mean_ratio = np.sqrt((1. + phi) / (1. - phi) / (4 * 128))
        sd_ratio = np.sqrt(2. * (1. + phi**2) / (1. - phi**2)
            / (4 * 128)) / 2.
        self.assertAlmostEqual(float(report["velocity_mean_x"][0, 0, 0]),
            mean_ratio, places=5)
        self.assertAlmostEqual(float(report["velocity_sd_x"][0, 0, 0]),
            sd_ratio, places=5)
        self.assertAlmostEqual(float(report["density_sd"][0, 0, 0]),
            sd_ratio, delta=3e-5)

    def test_small_spatial_run_is_reproducible_and_nonpromoting(self):
        kwargs = dict(size=8, draws=16, chains=2, replicates=2,
            seed=20261104, batch_sizes=(2, 4), minimum_batches=2,
            correlation_lengths=(0.0, 1.0))
        first = run_spatial_calibration(**kwargs)
        second = run_spatial_calibration(**kwargs)
        self.assertEqual(first, second)
        self.assertFalse(first["posterior_promoted"])
        self.assertFalse(first["heldout_read"])
        self.assertEqual(len(first["scenarios"]), 2)
        metrics = first["scenarios"][1]["mcse_p95_estimate_distribution"]
        self.assertIn("velocity_mean_x", metrics)
        self.assertIn("velocity_sd_z", metrics)
        self.assertIn("training_region", metrics["density_mean"]["4"])
        self.assertGreater(synthetic_design(8)["regions"][
            "training_region"].sum(), 1)


if __name__ == "__main__":
    unittest.main()
