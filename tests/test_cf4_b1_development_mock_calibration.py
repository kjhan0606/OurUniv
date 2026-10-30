from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cf4_b1_development_mock_calibration as calibration  # noqa: E402


def test_seed_firewall_and_arm_assignment_are_exact():
    assert calibration.seed_schedule(0)["truth"] == 2026083000
    assert calibration.seed_schedule(63)["truth"] == 2026083063
    for bad in (-1, 64):
        try:
            calibration.seed_schedule(bad)
        except calibration.MockCalibrationError:
            pass
        else:
            raise AssertionError("out-of-range seed was accepted")


def test_synthetic_intensity_is_positive_and_counts_are_int64():
    eta, velocity = calibration._truth_field(2026083000)
    for arm in "ABCD":
        intensity = calibration._positive_intensity(eta, arm, velocity)
        assert intensity.shape == (6, 32, 32, 32)
        assert np.all(np.isfinite(intensity))
        assert np.all(intensity > 0.0)
        counts = calibration._draw_counts(intensity, arm, 2026100000)
        assert counts.dtype == np.dtype(np.int64)
        assert np.all(counts >= 0)


def test_single_member_is_reproducible_and_holdout_is_a_partition():
    first = calibration.run_mock(3, "C")
    second = calibration.run_mock(3, "C")
    assert first == second
    assert first["metrics"]["positive_support_fraction"] == 1.0


def test_full_calibration_consumes_only_64_development_seeds():
    result = calibration.run_calibration()
    assert result["status"].endswith("NO_SCIENCE_CLAIM")
    assert result["seed_firewall"]["development_count"] == 64
    assert result["seed_firewall"]["seed_start_inclusive"] == 2026083000
    assert result["seed_firewall"]["seed_stop_exclusive"] == 2026083064
    assert result["seed_firewall"]["validation_opened"] is False
    assert result["aggregate"]["member_count"] == 64
    assert set(result["arms"]) == {"A", "B", "C", "D"}
    assert all(item["member_count"] == 16 for item in result["arms"].values())
    assert result["scientific_disposition"]["observational_z0_posterior"] == "NOT_CREATED"
    assert result["scientific_disposition"]["0p3_cMpc_h_claim"] == "NOT_ALLOWED"


def test_result_json_round_trip(tmp_path):
    output = tmp_path / "result.json"
    assert calibration.main(["--output", str(output)]) == 0
    loaded = json.loads(output.read_text())
    assert loaded["aggregate"]["member_count"] == 64
