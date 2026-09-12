import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cf4_b1_count_derivative_diagnosis_v2 as diagnosis


def test_derivative_ladder_and_seed_sample_are_frozen():
    assert diagnosis.DERIVATIVE_STEPS == (1.0e-3, 1.0e-4, 1.0e-5)
    assert len(diagnosis.SAMPLE_INDICES) == 16
    assert diagnosis.SAMPLE_INDICES[0:4] == (0, 1, 2, 3)
    assert diagnosis.SAMPLE_INDICES[-4:] == (48, 49, 50, 51)


def test_derivative_fisher_is_finite_for_zero_coefficients():
    fisher = diagnosis._count_derivative_fisher(
        np.zeros(diagnosis.integrated.MODE_COUNT), "A", 1.0e-4
    )
    assert fisher.shape == (diagnosis.integrated.MODE_COUNT, diagnosis.integrated.MODE_COUNT)
    assert np.all(np.isfinite(fisher))
    np.testing.assert_allclose(fisher, fisher.T, rtol=1.0e-12, atol=1.0e-12)
