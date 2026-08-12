from __future__ import annotations

from hong2021_v84b_group_gate import calibration_pass


def passing_row() -> dict:
    return {
        "NLL_improvement_over_standard_normal": 0.01,
        "PIT_mean": 0.5,
        "PIT_total_variation_from_uniform": 0.005,
        "backbone_quartile_PIT_means": [0.5, 0.5, 0.5, 0.5],
        "central_coverage": {"50": 0.5, "80": 0.8, "95": 0.95},
        "tail_exceedance": {
            "0.001": {"lower_over_expected": 1.0, "upper_over_expected": 1.0},
            "0.0001": {"lower_over_expected": 1.0, "upper_over_expected": 1.0},
        },
    }


def test_group_gate_requires_all_central_and_extreme_tail_checks() -> None:
    row = passing_row()
    assert calibration_pass(row)
    row["tail_exceedance"]["0.0001"]["upper_over_expected"] = 1.26
    assert not calibration_pass(row)


def test_group_gate_rejects_good_tail_with_bad_nll_or_pit() -> None:
    row = passing_row()
    row["NLL_improvement_over_standard_normal"] = 0.0
    assert not calibration_pass(row)
    row = passing_row()
    row["PIT_total_variation_from_uniform"] = 0.011
    assert not calibration_pass(row)
