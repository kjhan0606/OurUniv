import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cf4_b1_pooled_coverage_guard_v1 as guard


def test_guard_contract_is_clustered_and_two_level():
    assert guard.BOOTSTRAP_REPLICATES == 10_000
    assert guard.TARGETS["coverage68"] == 0.6826894921370859
    assert guard.TARGETS["coverage95"] == 0.9544997361036416
    assert guard.TOLERANCES == {"coverage68": 0.05, "coverage95": 0.02}


def test_guard_reproduces_committed_development_result():
    root = Path(__file__).resolve().parents[1]
    result = guard.evaluate(root / "config/cf4_b1_mode_coverage_diagnosis_result_v3.json")
    assert result["levels"]["coverage68"]["bootstrap_upper_97_5"] == 0.787109375
    assert result["levels"]["coverage95"]["bootstrap_upper_97_5"] == 0.982421875
    assert result["overall_pass"] is False
    assert result["primary_reference_pass"] is False
