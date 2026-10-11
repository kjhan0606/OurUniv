import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cf4_b1_count_information_diagnosis_v1 as diagnosis


def test_count_information_diagnosis_is_development_only_and_stratified():
    assert diagnosis.SAMPLE_INDICES == (
        0, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 48, 49, 50, 51
    )
    assert len(diagnosis.SAMPLE_INDICES) == 16
    assert diagnosis.FINITE_DIFFERENCE_STEP == 1.0e-4


def test_sandwich_ratio_is_finite_for_one_member():
    row = diagnosis.diagnose_member(0, "A")
    ratio = np.asarray(row["sandwich_to_expected_sigma_by_mode"])
    assert ratio.shape == (diagnosis.integrated.MODE_COUNT,)
    assert np.all(np.isfinite(ratio))
    assert np.all(ratio > 0.0)
