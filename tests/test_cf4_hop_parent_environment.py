from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cf4_hop_parent_environment",
    ROOT / "scripts" / "cf4_hop_parent_environment.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def selected_pair() -> dict:
    return {
        "position_i_mpc_h": [10.0, 10.0, 10.0],
        "position_j_mpc_h": [10.8, 10.0, 10.0],
        "m1_fof_msun_h": 1.6e12,
        "m2_fof_msun_h": 3.5e12,
        "npart": [45, 98],
    }


def test_exact_particle_union_is_diagnostic_not_independent_pair_support() -> None:
    result = MODULE.classify_selected_pair_crossmatch(
        selected_pair(),
        np.asarray([[10.55, 10.0, 10.0]]),
        np.asarray([5.1e12]),
        np.asarray([143]),
        384.0,
    )
    assert result["mode"] == "merged_hop_group_mass_consistent"
    assert result["pair_supported_for_parent_trace"] is False
    assert result["hop_independently_resolves_pair"] is False


def test_merged_group_with_extra_particles_is_not_accepted() -> None:
    result = MODULE.classify_selected_pair_crossmatch(
        selected_pair(),
        np.asarray([[10.55, 10.0, 10.0]]),
        np.asarray([5.3e12]),
        np.asarray([149]),
        384.0,
    )
    assert result["mode"] == "merged_hop_group_unverified"
    assert result["pair_supported_for_parent_trace"] is False


def test_distinct_nearby_hop_groups_remain_independently_resolved() -> None:
    result = MODULE.classify_selected_pair_crossmatch(
        selected_pair(),
        np.asarray([[10.0, 10.0, 10.0], [10.8, 10.0, 10.0]]),
        np.asarray([1.6e12, 3.5e12]),
        np.asarray([45, 98]),
        384.0,
    )
    assert result["mode"] == "distinct_hop_groups"
    assert result["pair_supported_for_parent_trace"] is True
    assert result["hop_independently_resolves_pair"] is True
