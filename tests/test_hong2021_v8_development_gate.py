from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hong2021_v8_development_gate import calibration_score


def test_calibration_score_uses_worst_log_deviation() -> None:
    metrics = {
        "fourier_log_density": {
            "generated_total_power_over_truth": {
                "3-6_h_mpc": 1.05,
                "6-10.0531_h_mpc": 0.8,
            }
        },
        "residual_calibration": {"generated_over_truth_rms": 1.1},
    }
    assert abs(calibration_score(metrics) - abs(__import__("math").log(0.8))) < 1e-12
