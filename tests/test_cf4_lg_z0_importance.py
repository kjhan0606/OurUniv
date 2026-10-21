import numpy as np

from cf4_lg_z0_importance import evaluate_importance_gate


def _pair(i, j, log_likelihood=-10.0):
    return {
        "halo_i": i,
        "halo_j": j,
        "log_likelihood": log_likelihood,
    }


def test_importance_gate_intersects_the_same_pair_identity():
    likelihood = {"rows": [{
        "parent_seed": 3429,
        "small_scale_seed": 1,
        "candidate_pairs": [_pair(1, 2, -8.0), _pair(3, 4, -10.0)],
        "midpoint_importance": {
            "log_target_prior_over_sampling_proposal": 0.2,
        },
    }]}
    preview = {"rows": [{
        "small_scale_seed": 1,
        "pair_rows": [
            {"screen_pair": _pair(2, 1), "preview_pass": False},
            {"screen_pair": _pair(4, 3), "preview_pass": True},
        ],
    }]}
    hard_p2 = {"results": [{
        "small_scale_seed": 1,
        "screen_pairs": [_pair(1, 2), _pair(3, 4)],
    }]}
    selection = {
        "minimum_jointly_eligible_realizations": 1,
        "minimum_fresh_importance_ESS": 1.0,
        "maximum_fresh_single_weight": 1.0,
    }
    result = evaluate_importance_gate(
        likelihood, preview, hard_p2, selection
    )
    row = result["rows"][0]
    assert row["n_jointly_eligible_pairs"] == 1
    assert row["best_eligible_pair"]["halo_i"] == 3
    assert np.isclose(row["unnormalized_log_importance_weight"], -9.8)
    assert result["passed"]


def test_importance_gate_fails_when_only_different_pairs_pass_each_gate():
    likelihood = {"rows": [{
        "parent_seed": 3429,
        "small_scale_seed": 1,
        "candidate_pairs": [_pair(1, 2), _pair(3, 4)],
        "midpoint_importance": {
            "log_target_prior_over_sampling_proposal": 0.0,
        },
    }]}
    preview = {"rows": [{
        "small_scale_seed": 1,
        "pair_rows": [{"screen_pair": _pair(1, 2), "preview_pass": True}],
    }]}
    hard_p2 = {"results": [{
        "small_scale_seed": 1,
        "screen_pairs": [_pair(3, 4)],
    }]}
    selection = {
        "minimum_jointly_eligible_realizations": 1,
        "minimum_fresh_importance_ESS": 1.0,
        "maximum_fresh_single_weight": 1.0,
    }
    result = evaluate_importance_gate(
        likelihood, preview, hard_p2, selection
    )
    assert result["n_jointly_eligible_realizations"] == 0
    assert not result["passed"]
