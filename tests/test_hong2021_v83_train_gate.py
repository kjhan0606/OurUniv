from __future__ import annotations

from hong2021_v83_train_gate import calibration_pass


def _row() -> dict:
    return {
        "NLL_improvement_over_standard_normal": 0.01,
        "PIT_mean": 0.5,
        "backbone_quartile_PIT_means": [0.49, 0.50, 0.51, 0.52],
        "PIT_maximum_bin_mass_error": 0.01,
        "central_coverage": {"50": 0.5, "80": 0.8, "95": 0.95},
    }


def test_calibration_pass_requires_every_component() -> None:
    row = _row()
    assert calibration_pass(row)
    row["backbone_quartile_PIT_means"][3] = 0.8
    assert not calibration_pass(row)
    row = _row()
    row["NLL_improvement_over_standard_normal"] = 0.0
    assert not calibration_pass(row)
    row = _row()
    row["central_coverage"]["95"] = 0.89
    assert not calibration_pass(row)
