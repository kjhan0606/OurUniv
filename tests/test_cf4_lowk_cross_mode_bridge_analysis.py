import copy

import pytest

from cf4_lowk_cross_mode_bridge_analysis import analyze_bridge_result


def result():
    checkpoints = []
    for cycle, bridge_l1, control_l1, bridge_overlap, control_overlap in (
        (4, .40, .45, .10, .08),
        (8, .34, .41, .14, .09),
        (16, .30, .39, .18, .10),
    ):
        checkpoints.append({
            "bridge_cycle": cycle,
            "matched_mh_sweeps": 2 * cycle,
            "bridge_maximum_parent_L1": bridge_l1,
            "control_maximum_parent_L1": control_l1,
            "bridge_minimum_exact_overlap": bridge_overlap,
            "control_minimum_exact_overlap": control_overlap,
        })
    return {
        "schema": "ouruniv-cf4-lowk-cross-mode-bridge-pilot-v1",
        "status": "complete_diagnostic",
        "checkpoints": checkpoints,
        "groups": [
            {
                "group": group,
                "roundtrip_fraction": .1 * group,
                "swap_acceptance_fraction": [.2, .3, .4, .5, .6],
            }
            for group in range(4)
        ],
    }


def test_analysis_reports_bridge_evidence_without_promoting_parent():
    analysis = analyze_bridge_result(result())
    assert analysis["science_evidence"]["bridge_mechanism_supported"] is True
    assert analysis["science_evidence"]["frozen_parent_L1_reference_pass"] is True
    assert analysis["transport"]["cross_mode_transport_observed"] is True
    assert analysis["decision"]["parent_posterior_promotion_authorized"] is False


def test_analysis_rejects_unmatched_sweeps():
    value = copy.deepcopy(result())
    value["checkpoints"][1]["matched_mh_sweeps"] = 15
    with pytest.raises(ValueError, match="sweep matched"):
        analyze_bridge_result(value)
