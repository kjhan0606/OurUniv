import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cf4_zoom_z0_gate_v2 import (  # noqa: E402
    evaluate_core_checks,
    evaluate_m33_peaks,
    load_extended_config,
    nearby_third_halo,
)


def test_extended_v11_keeps_frozen_definitive_thresholds():
    config = load_extended_config(
        ROOT / "config/p2_lg_targets_v11_bgc_inverse_peak.json")
    gate = config["definitive_zoom_gate"]
    assert gate["pair_member_mass_range_msun_h"] == [5e11, 3e12]
    assert gate["pair_separation_range_mpc_h"] == [0.45, 0.75]
    assert gate["maximum_contaminant_mass_fraction_within_r200c"] == 1e-4
    assert config["paired_small_scale_seeds"][-1] == 5108


def test_m33_unmerged_peak_gate_selects_physical_candidate():
    peaks = {
        "mass": np.array([1e12, 1.2e11, 8e10, 4e11]),
        "pos": np.array([[0, 0, 0], [0.2, 0, 0], [0.5, 0, 0], [0.1, 0, 0.]]),
        "group_id": np.arange(4),
        "n": np.array([1000, 100, 80, 400]),
        "contamination_fof": np.zeros(4),
    }
    gate = {
        "mass_range_msun_h": [3e10, 5e11],
        "m31_separation_range_mpc_h": [0.08, 0.35],
        "maximum_mass_fraction_of_m31": 0.3,
    }
    result = evaluate_m33_peaks(peaks, np.zeros(3), 1e12, gate, 10.0)
    assert result["passed"]
    assert result["candidate"]["catalog_index"] == 1


def test_third_halo_and_exact_core_thresholds():
    cat = {
        "mass": np.array([1.5e12, 1.0e12, 9e11]),
        "pos": np.array([[5.0, 5.0, 5.0], [5.6, 5.0, 5.0], [7.0, 5.0, 5.0]]),
    }
    third = nearby_third_halo(cat, (0, 1), np.array([5.3, 5, 5]),
                              2.5, 1e12, 10.0)
    assert third["passed"]
    profiles = [{"m200c_msun_h": 1.5e12}, {"m200c_msun_h": 1e12}]
    pair = {
        "separation_mpc_h": 0.6,
        "vtotal_kms": -100.0,
        "vtan_kms": 60.0,
        "midpoint_offset_mpc_h": 2.0,
        "massive_halo_isolation_mpc_h": 4.0,
    }
    target = {
        "pair_member_mass_range_msun_h": [5e11, 3e12],
        "pair_mass_ratio_max": 3.0,
        "pair_separation_range_mpc_h": [0.45, 0.75],
        "total_radial_velocity_range_km_s": [-200.0, -20.0],
        "maximum_tangential_velocity_km_s": 120.0,
        "pair_midpoint_max_offset_mpc_h": 5.0,
        "isolation_radius_mpc_h": 3.0,
    }
    assert all(evaluate_core_checks(profiles, pair, third, target).values())
