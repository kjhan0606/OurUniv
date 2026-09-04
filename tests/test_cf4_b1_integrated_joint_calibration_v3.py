import sys
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
