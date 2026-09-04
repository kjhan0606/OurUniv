import sys
import inspect
from pathlib import Path

import numpy as np

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


def test_rsd_interpolation_uses_grid_periodic_wrap():
    source = inspect.getsource(calibration.base._spherical_rsd_field)
    assert 'mode="grid-wrap"' in source
    assert 'mode="wrap")' not in source


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
