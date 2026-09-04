import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cf4_b1_integrated_joint_calibration_v3 as calibration


def test_manifest_binding_and_latent_basis():
    assert calibration.BASIS.shape == (calibration.MODE_COUNT, calibration.GRID, calibration.GRID, calibration.GRID)
    assert calibration.MARK_JAC.shape == (16584, calibration.MODE_COUNT)
    assert calibration.MARKS["manifest"]["counts"]["secure_cf4_groups"] == 11610
    assert calibration.MARKS["excluded_target_rows"] == 17007
    assert calibration.seed_schedule(0)["truth"] == 2026083000
    assert calibration.seed_schedule(63)["truth"] == 2026083063
    assert calibration.seed_schedule(0)["count_split"] == 2026401000
    assert calibration.seed_schedule(63)["count_split"] == 2026401063


def test_one_member_is_joint_and_canonical_factor_agrees():
    row = calibration.run_mock(0, "A")
    metrics = row["metrics"]
    assert metrics["secure_rows"] == 16584
    assert metrics["secure_groups"] == 11610
    assert metrics["joint_log_likelihood_abs_error"] < 1.0e-7
    assert metrics["positive_support_fraction"] == 1.0
    assert np.isfinite(metrics["heldout_log_score_improvement"])


def test_harness_remains_pass_and_validation_firewall_closed():
    result = calibration.run_joint_harness()
    assert result["status"] == "PASS"
    assert result["validation_seeds_opened"] is False


def test_rsd_interpolation_is_periodic_and_edge_continuous():
    grid = calibration.base.GRID
    axis = np.indices((grid, grid, grid), dtype=np.float64)
    eta = (
        np.sin(2.0 * np.pi * axis[0] / grid)
        + 0.3 * np.cos(2.0 * np.pi * axis[1] / grid)
        + 0.2 * np.sin(2.0 * np.pi * axis[2] / grid)
    )
    # Whole-grid translations must reproduce the same interpolated value.
    coordinates = np.array(
        [[0.25, 0.25 + grid, 0.25 - grid],
         [7.75, 7.75 + grid, 7.75 - grid],
         [12.25, 12.25 + grid, 12.25 - grid]],
        dtype=np.float64,
    )
    translated = map_coordinates(eta, coordinates, order=1, mode="grid-wrap")
    np.testing.assert_allclose(translated[1:], translated[0], rtol=0.0, atol=1.0e-12)

    # The repaired RSD sampler must have a finite, continuous response as a
    # radial displacement crosses both periodic box faces.
    for point in ((0, grid // 2, grid // 2), (grid - 1, grid // 2, grid // 2)):
        values = []
        for sign in (-1.0, 1.0):
            velocity = np.zeros((grid, grid, grid, 3), dtype=np.float64)
            velocity[point] = (
                sign * calibration.base.HUBBLE / calibration.base.LITTLE_H
                * calibration.base.CELL_SIZE * 1.0e-5
                * calibration.base._RHAT[point]
            )
            values.append(calibration.base._spherical_rsd_field(eta, velocity)[point])
        assert np.all(np.isfinite(values))
        assert abs(values[1] - values[0]) < 1.0e-4


def test_shared_group_mark_fisher_matches_direct_covariance_inverse():
    jac = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]], dtype=float)
    sigma, tau = 7.0, 3.0
    covariance = sigma**2 * np.eye(3) + tau**2 * np.ones((3, 3))
    expected = jac.T @ np.linalg.inv(covariance) @ jac
    actual = calibration._single_group_mark_fisher(jac, sigma, tau)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_width_repair_does_not_change_mean_model():
    assert calibration.ARM_WIDTH_SCALE == {"A": 0.97, "B": 1.0, "C": 0.97, "D": 1.5}
    assert calibration.D_OVERDISPERSION_PHI == 0.35
    # Width calibration is allowed to change posterior draws, but it must not
    # move the deterministic MAP/mean latent estimate.  Exercise that
    # contract by perturbing one arm's draw scale and comparing the estimate.
    original = dict(calibration.ARM_WIDTH_SCALE)
    try:
        for arm in ("A", "D"):
            baseline = calibration.run_mock(0, arm)
            calibration.ARM_WIDTH_SCALE[arm] = 1.23
            perturbed = calibration.run_mock(0, arm)
            np.testing.assert_allclose(
                baseline["_estimate_coeff"], perturbed["_estimate_coeff"], rtol=0.0, atol=1.0e-12
            )
            assert not np.allclose(
                baseline["_draws_coeff"], perturbed["_draws_coeff"], rtol=0.0, atol=1.0e-12
            )
            calibration.ARM_WIDTH_SCALE[arm] = original[arm]
    finally:
        calibration.ARM_WIDTH_SCALE.clear()
        calibration.ARM_WIDTH_SCALE.update(original)
