import copy
import itertools

import numpy as np
import pytest

from cf4_lowk_cross_mode_bridge_analysis import analyze_bridge_result


def _maximum_l1(values):
    return max(
        np.abs(values[left] - values[right]).sum()
        for left, right in itertools.combinations(range(4), 2)
    )


def fixture():
    base = np.zeros(256, dtype=np.float64)
    base[:4] = 0.25
    bridge, control = [], []
    for checkpoint in range(3):
        bridge_rows, control_rows = [], []
        for group in range(4):
            bridge_row = base.copy()
            bridge_row[0] += group * (0.004 - checkpoint * 0.0005)
            bridge_row[1] -= group * (0.004 - checkpoint * 0.0005)
            bridge_rows.append(bridge_row)
            control_row = base.copy()
            control_row[2] += group * (0.005 + checkpoint * 0.0005)
            control_row[3] -= group * (0.005 + checkpoint * 0.0005)
            control_rows.append(control_row)
        bridge.append(bridge_rows)
        control.append(control_rows)
    bridge = np.asarray(bridge)
    control = np.asarray(control)
    checkpoints = []
    for index, cycle in enumerate((4, 8, 16)):
        checkpoints.append({
            "bridge_cycle": cycle,
            "matched_mh_sweeps": 2 * cycle,
            "bridge_maximum_parent_L1": _maximum_l1(bridge[index]),
            "control_maximum_parent_L1": _maximum_l1(control[index]),
            "bridge_minimum_exact_overlap": .1 + .02 * index,
            "control_minimum_exact_overlap": .08 + .01 * index,
        })
    result = {
        "schema": "ouruniv-cf4-lowk-cross-mode-bridge-pilot-v1",
        "status": "complete_diagnostic",
        "particles_per_group": 128,
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
    return result, bridge, control


def test_analysis_uses_particle_matched_null_without_promoting_parent():
    result, bridge, control = fixture()
    analysis = analyze_bridge_result(
        result, bridge, control, null_draws=500, null_seed=91
    )
    assert analysis["schema"].endswith("analysis-v2")
    assert analysis["science_evidence"][
        "all_bridge_checkpoints_pass_particle_matched_q999"
    ] is True
    assert analysis["science_evidence"][
        "original_2048_particle_resolution_reached"
    ] is False
    assert analysis["science_evidence"]["original_parent_incoherence_resolved"] is False
    assert analysis["transport"]["cross_mode_transport_observed"] is True
    assert analysis["decision"]["parent_posterior_promotion_authorized"] is False


def test_analysis_rejects_unmatched_sweeps():
    result, bridge, control = fixture()
    value = copy.deepcopy(result)
    value["checkpoints"][1]["matched_mh_sweeps"] = 15
    with pytest.raises(ValueError, match="sweep matched"):
        analyze_bridge_result(value, bridge, control, null_draws=10)
