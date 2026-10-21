import json
from pathlib import Path

import numpy as np

from cf4_lg_z0_likelihood import (
    enumerate_candidate_pairs,
    logmeanexp,
    pair_log_likelihood,
    periodic_delta,
)


REPO = Path(__file__).resolve().parents[1]
PROGRAM = REPO / "config/p2_lg_z0_forward_likelihood_v7_development.json"


def _program() -> dict:
    return json.loads(PROGRAM.read_text())


def test_periodic_delta_uses_short_image() -> None:
    np.testing.assert_allclose(
        periodic_delta([1.0, 383.0, 10.0], [383.0, 1.0, 12.0], 384.0),
        [2.0, -2.0, -2.0],
    )


def test_loose_pair_enumeration_records_total_radial_velocity() -> None:
    preselection = _program()["candidate_preselection"]
    positions = np.array([[191.7, 192.0, 192.0], [192.3, 192.0, 192.0]])
    velocities = np.array([[50.0, 0.0, 0.0], [-50.0, 0.0, 0.0]])
    masses = np.array([1.2e12, 1.0e12])
    pairs = enumerate_candidate_pairs(
        positions, velocities, masses,
        centre=np.full(3, 192.0), box_size=384.0,
        preselection=preselection,
    )
    assert len(pairs) == 1
    assert np.isclose(pairs[0]["separation_mpc_h"], 0.6)
    assert np.isclose(pairs[0]["total_radial_velocity_km_s"], -40.0)


def test_target_like_pair_scores_above_displaced_pair() -> None:
    likelihood = _program()["z0_likelihood"]
    target = {
        "masses_msun_h": [1.2e12, 1.2e12],
        "separation_mpc_h": 0.574,
        "midpoint_offset_vector_mpc_h": [0.0, 0.0, 0.0],
        "total_radial_velocity_km_s": -109.0,
        "tangential_velocity_km_s": 57.0,
        "isolation_mpc_h": 8.0,
    }
    displaced = dict(target, midpoint_offset_vector_mpc_h=[6.0, 0.0, 0.0])
    target_score, _ = pair_log_likelihood(target, likelihood)
    displaced_score, _ = pair_log_likelihood(displaced, likelihood)
    assert target_score > displaced_score


def test_pair_mixture_has_no_duplicate_count_bonus() -> None:
    score = -12.5
    assert np.isclose(logmeanexp(np.array([score])), score)
    assert np.isclose(logmeanexp(np.array([score, score, score])), score)


def test_v7_program_keeps_fresh_bank_locked_until_development_pass() -> None:
    program = _program()
    assert program["status"] == "frozen_before_v6_catalog_scoring"
    assert program["information_firewall"]["fresh_v7_fields_opened"] is False
    assert len(program["prospective_v7_bank"]["proposal_seeds"]) == 64
    assert "importance_correction" in program["prospective_v7_bank"]["latent_midpoint_sampling_proposal"]
