import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cf4_b1_calibration_diagnosis_v1 import gate_failures


def _metrics(**overrides):
    value = {
        "response": 1.0,
        "correlation_r": 0.99,
        "residual_power_ratio": 0.01,
        "coverage68": 0.6826894921370859,
        "coverage95": 0.9544997361036416,
        "heldout_log_score_improvement": 1.0,
        "fit_success": True,
        "joint_log_likelihood_abs_error": 1.0e-10,
    }
    value.update(overrides)
    return value


def test_frozen_gate_decomposition_is_named_and_fail_closed():
    assert gate_failures(_metrics()) == []
    assert gate_failures(_metrics(coverage68=0.9)) == ["coverage68"]
    assert gate_failures(_metrics(coverage95=0.8, fit_success=False)) == ["coverage95", "optimizer_convergence"]
