from __future__ import annotations

import copy
from pathlib import Path

from hong2021_v18_init import sha256_file
from hong2021_v27_latent_audit import (
    AUDIT_PROGRAM_SHA256,
    coarse_base_z_score,
    compare_to_v26,
    latent_support_score,
    mechanism_summary,
    numerical_stability,
)


REPO = Path(__file__).parents[1]


def _row(*, coarse: tuple[float, float], truth_tail: int, generated_tail: int):
    moments = lambda count, tail, maximum: {
        "count": count,
        "absolute_tail_counts": {"5.0": tail},
        "absolute_maximum": maximum,
    }
    return {
        "truth_base_z_coarse_to_fine": [
            {"standard_deviation": coarse[0]},
            {"standard_deviation": coarse[1]},
        ],
        "truth_latent": moments(1000, truth_tail, 5.0),
        "generated_latent": moments(16000, generated_tail, 6.0),
        "roundtrip": {
            "maximum_absolute_latent_error": 1.0e-3,
            "rms_latent_error": 1.0e-4,
            "maximum_absolute_logdet_cancellation_coarse_to_fine": [
                1.0e-6,
                2.0e-6,
                3.0e-6,
                4.0e-6,
                5.0e-6,
                6.0e-6,
            ],
        },
        "stored_ensemble_replay": {"maximum_absolute_y_difference": 1.0e-7},
    }


def test_audit_program_hash_is_frozen():
    assert (
        sha256_file(REPO / "config/hong2021_v27_latent_audit_program.json")
        == AUDIT_PROGRAM_SHA256
    )


def test_fixed_scores_reward_standard_normal_base_and_matching_tail_fraction():
    prior = _row(coarse=(1.6, 1.4), truth_tail=1, generated_tail=64)
    current = _row(coarse=(1.2, 1.1), truth_tail=1, generated_tail=24)
    assert coarse_base_z_score(current) < coarse_base_z_score(prior)
    assert latent_support_score(current) < latent_support_score(prior)
    comparison = compare_to_v26(
        {domain: current for domain in ("TNG100", "SIMBA", "Swift")},
        {domain: prior for domain in ("TNG100", "SIMBA", "Swift")},
    )
    assert comparison["all_domains_both_strictly_improved"] is True


def test_zero_tail_count_rule_is_finite():
    row = _row(coarse=(1.0, 1.0), truth_tail=0, generated_tail=0)
    assert latent_support_score(row) >= 0.0


def test_numerical_stability_and_fixed_classification_order():
    prior = _row(coarse=(1.6, 1.4), truth_tail=1, generated_tail=64)
    current = _row(coarse=(1.2, 1.1), truth_tail=1, generated_tail=24)
    final = {domain: copy.deepcopy(current) for domain in ("TNG100", "SIMBA", "Swift")}
    assert numerical_stability(final)["pass"] is True
    candidates = {"30000": final}
    v26 = {
        "candidates": {
            "30000": {
                "domains": {
                    domain: copy.deepcopy(prior)
                    for domain in ("TNG100", "SIMBA", "Swift")
                }
            }
        }
    }
    optimization = {
        "finest_scale_objective_fraction": 0.875,
        "relative_improvement_25000_to_30000": 0.006,
    }
    summary = mechanism_summary(candidates, optimization, v26)
    assert summary["classification"].startswith("parent_aligned_context_repairs")

    final["TNG100"]["roundtrip"]["rms_latent_error"] = 2.0e-3
    summary = mechanism_summary(candidates, optimization, v26)
    assert summary["classification"] == "v27_trained_flow_numerical_failure"


def test_one_domain_without_both_improvements_selects_empirical_control():
    prior = _row(coarse=(1.6, 1.4), truth_tail=1, generated_tail=64)
    current = _row(coarse=(1.2, 1.1), truth_tail=1, generated_tail=24)
    final = {domain: copy.deepcopy(current) for domain in ("TNG100", "SIMBA", "Swift")}
    final["TNG100"] = copy.deepcopy(prior)
    v26 = {
        "candidates": {
            "30000": {
                "domains": {
                    domain: copy.deepcopy(prior)
                    for domain in ("TNG100", "SIMBA", "Swift")
                }
            }
        }
    }
    summary = mechanism_summary(
        {"30000": final},
        {
            "finest_scale_objective_fraction": 0.875,
            "relative_improvement_25000_to_30000": 0.006,
        },
        v26,
    )
    assert summary["classification"] == (
        "explicit_conditional_flow_remains_latent_miscalibrated_after_phase_repair"
    )
    assert summary["next"] == (
        "freeze_and_test_train_only_empirical_joint_residual_control"
    )
