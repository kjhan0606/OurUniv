import hashlib
import json
from pathlib import Path

import numpy as np

from cf4_lg_mode_release_audit import (
    frozen_mode_errors,
    l3_max_stat_permutation,
    projection_errors,
    right_sided_ks,
    simultaneous_envelope,
)
from cf4_lg_peak_cr import free_rfft_mask


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/p2_lg_v8_cf4_mode_release_audit.json"


def test_right_sided_ks_handles_ties_and_direction():
    assert np.isclose(right_sided_ks(np.array([0.0, 1.0]), np.array([1.0, 2.0])), 0.5)
    assert np.isclose(right_sided_ks(np.array([1.0, 2.0]), np.array([0.0, 1.0])), 0.0)


def test_projection_and_frozen_errors_are_normalized_and_direction_free():
    rng = np.random.default_rng(4)
    parent = rng.normal(size=(8, 8, 8))
    exact = projection_errors(parent, parent.copy())
    assert exact["relative_RMS"] == 0.0
    assert exact["maximum_normalized_error"] == 0.0

    frozen = ~free_rfft_mask(8, 4)
    preserved = frozen_mode_errors(parent, parent, frozen)
    assert preserved["relative_RMS"] == 0.0
    assert preserved["maximum_normalized_error"] == 0.0
    changed = frozen_mode_errors(parent + 1.0e-4, parent, frozen)
    assert changed["relative_RMS"] > 0.0
    assert changed["maximum_normalized_error"] > 0.0


def test_l3_max_stat_permutation_is_deterministic_and_detects_worse_proposal():
    rng = np.random.default_rng(9)
    reference = rng.normal(0.0, 0.2, size=(30, 6))
    proposal = rng.normal(1.0, 0.2, size=(31, 6))
    q99 = float(np.quantile(reference[:, 0], 0.99, method="linear"))
    first = l3_max_stat_permutation(reference, proposal, q99, 499, 71, 73)
    second = l3_max_stat_permutation(reference, proposal, q99, 499, 71, 91)
    np.testing.assert_allclose(
        first["observed_coordinates"], second["observed_coordinates"]
    )
    assert first["one_sided_max_stat_pvalue"] == second[
        "one_sided_max_stat_pvalue"
    ]
    assert first["one_sided_max_stat_pvalue"] < 0.01


def test_simultaneous_envelope_uses_sealed_reference_scale():
    calibration = {
        "coordinate_q99": [0.5, 0.5],
        "reference_summary": [0.0] * 6,
        "bootstrap_studentization_scale": [1.0] * 6,
    }
    passed = simultaneous_envelope(np.zeros((20, 2)), calibration, critical=1.0)
    failed = simultaneous_envelope(np.full((20, 2), 2.0), calibration, critical=1.0)
    assert passed["pass"] is True
    assert failed["pass"] is False


def test_program_pins_implementation_and_keeps_all_subset_routes_closed():
    program = json.loads(PROGRAM.read_text())
    implementation = REPO / program["implementation"]["path"]
    assert hashlib.sha256(implementation.read_bytes()).hexdigest() == program[
        "implementation"
    ]["sha256"]
    assert program["proposal_seed_range_python"] == [5269, 5525]
    assert program["information_firewall"]["all_256_projections_required"] is True
    assert program["information_firewall"]["P1_P2_or_seed5422_subset_allowed"] is False
    assert program["geometry"] == {
        "proposal_N": 576,
        "projection_N": 192,
        "frozen_N": 64,
        "box_size_mpc_h": 384.0,
    }
    assert "norm='ortho' rFFT" in program["metric_implementation"][
        "L1_frozen_modes"
    ]
    assert program["decision"]["seed_promotion_authorized_now"] is False
    assert program["decision"]["RAMSES_authorized_now"] is False


def test_source_stops_after_L0_before_constructing_CF4_operator():
    source = (REPO / "src/cf4_lg_mode_release_audit.py").read_text()
    assert source.index("if not l0_pass:") < source.index(
        "forward, _, _, npdtype = build_forward"
    )
