import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cf4_b1_mode_coverage_diagnosis_v1 as diagnosis
import numpy as np


def test_mode_coverage_metric_is_seed_frequency_not_gate_mutation():
    assert diagnosis.TARGET_68 == 0.6826894921370859
    assert diagnosis.TARGET_95 == 0.9544997361036416
    # The runner declares a separate diagnostic metric and never changes the
    # strict promotion gate in the integrated calibration module.
    assert diagnosis.integrated.MOCK_COUNT == 64


def test_mode_sigma_uses_analytic_laplace_covariance():
    estimate = np.zeros(diagnosis.integrated.MODE_COUNT)
    sigma = diagnosis.analytic_posterior_sigma(estimate, "D")
    precision = (
        np.eye(diagnosis.integrated.MODE_COUNT) / diagnosis.integrated.PRIOR_SIGMA**2
        + diagnosis.integrated._count_fisher(estimate, "D")
        + diagnosis.integrated.MARK_FISHER
    )
    expected = np.sqrt(np.diag(np.linalg.pinv(precision))) * diagnosis.integrated.ARM_WIDTH_SCALE["D"]
    np.testing.assert_allclose(sigma, expected, rtol=1.0e-12, atol=1.0e-12)
