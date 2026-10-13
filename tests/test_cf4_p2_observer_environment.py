from __future__ import annotations

import numpy as np

from cf4_p2_screen import find_pairs


def _halos(group_distance: float) -> dict:
    return {
        "pos": np.asarray(
            [[191.6, 192.0, 192.0], [192.4, 192.0, 192.0],
             [192.0, 192.0 + group_distance, 192.0]],
            dtype=np.float64,
        ),
        "mass": np.asarray([1.2e12, 1.0e12, 6.0e12], dtype=np.float64),
        "vel": np.zeros((3, 3), dtype=np.float64),
    }


def _screen(environment: bool) -> dict:
    result = {
        "pair_member_mass_range_msun_h": [5.0e11, 4.0e12],
        "pair_mass_ratio_max": 4.0,
        "pair_separation_range_mpc_h": [0.3, 1.2],
        "pair_midpoint_max_offset_mpc_h": 5.0,
        "isolation_mass_threshold_msun_h": 5.0e12,
        "isolation_radius_mpc_h": 3.0,
    }
    if environment:
        result["observer_environment_gate"] = {"radius_mpc_h": 8.0}
    return result


M33 = {"mass_range_msun_h": [3.0e10, 5.0e11]}
CENTRE = np.full(3, 192.0)


def test_legacy_screen_does_not_apply_new_environment_gate():
    assert len(find_pairs(_halos(6.0), CENTRE, _screen(False), M33)) == 1


def test_environment_gate_rejects_massive_group_near_observer():
    assert find_pairs(_halos(6.0), CENTRE, _screen(True), M33) == []


def test_environment_gate_retains_isolated_pair():
    assert len(find_pairs(_halos(10.0), CENTRE, _screen(True), M33)) == 1
