from __future__ import annotations

import numpy as np

import cf4_b1_development_mock_calibration_v2 as calibration


def test_posterior_draw_count_and_correlated_draws_are_finite():
    mean = np.zeros((8, 8, 8), dtype=np.float64)
    variance = np.ones_like(mean)
    draws = calibration.posterior_draws(mean, variance, 2026300000)
    assert draws.shape == (16, 8, 8, 8)
    assert np.all(np.isfinite(draws))
    assert np.std(draws) > 0.0


def test_corrected_member_uses_train_conditioned_holdout_and_draw_coverage():
    first = calibration.run_mock(0, "A")
    second = calibration.run_mock(0, "A")
    assert first == second
    assert first["metrics"]["posterior_draw_count"] == 16
    assert first["metrics"]["profiled_nuisance_parameter_count"] == 12
    assert np.isfinite(first["metrics"]["heldout_log_score_improvement"])


def test_full_v2_calibration_runs_64_and_keeps_validation_closed():
    result = calibration.run_calibration()
    assert result["status"].endswith("NO_SCIENCE_CLAIM")
    assert result["joint_harness"]["status"] == "PASS"
    assert result["joint_factor_score_probe"]["independent_redshift_rejected"] is True
    assert result["aggregate"]["member_count"] == 64
    assert result["seed_firewall"]["validation_opened"] is False
    assert set(result["arms"]) == {"A", "B", "C", "D"}
    assert all(item["member_count"] == 16 for item in result["arms"].values())
    assert result["scientific_disposition"]["observational_z0_posterior"] == "NOT_CREATED"
