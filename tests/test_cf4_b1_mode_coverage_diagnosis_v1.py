import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cf4_b1_mode_coverage_diagnosis_v1 as diagnosis


def test_mode_coverage_metric_is_seed_frequency_not_gate_mutation():
    assert diagnosis.TARGET_68 == 0.6826894921370859
    assert diagnosis.TARGET_95 == 0.9544997361036416
    # The runner declares a separate diagnostic metric and never changes the
    # strict promotion gate in the integrated calibration module.
    assert diagnosis.integrated.MOCK_COUNT == 64
