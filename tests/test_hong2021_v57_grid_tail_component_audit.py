import json
from pathlib import Path

import numpy as np
import torch

from hong2021_v18_init import sha256_file
from hong2021_v57_grid_tail_component_audit import (
    PROGRAM_SHA256,
    _bin_indices,
    _bin_labels,
    _domain_summary,
    _rank_components,
    classify,
)


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/hong2021_v57_grid_tail_component_audit_program.json"


def test_program_is_frozen_hash_bound_and_keeps_firewall_closed():
    row = json.loads(PROGRAM.read_text())
    parent = row["parent_evidence"]
    record = json.loads((ROOT / parent["v56_record"]).read_text())
    partition = row["fixed_threshold_partition"]
    assert sha256_file(PROGRAM) == PROGRAM_SHA256
    assert sha256_file(ROOT / parent["v56_record"]) == parent["v56_record_sha256"]
    assert row["status"] == "frozen_before_audit_implementation_or_execution"
    assert record["train_only_mechanism_decision"]["next"] == parent["required_next"]
    assert record["firewall"]["development_accessed"] is False
    assert row["firewall"]["development_access"] == "forbidden"
    assert row["firewall"]["historical_EAGLE_access"] == "forbidden"
    assert row["firewall"]["independent_gate_locked"] is True
    assert len(partition["scored_grid_thresholds_log10rho"]) == 16
    assert len(_bin_labels(16)) == 18
    assert np.all(
        np.diff(
            [
                partition["lower_anchor_log10rho"],
                *partition["scored_grid_thresholds_log10rho"],
            ]
        )
        > 0.0
    )


def test_strict_exceedance_edges_remain_in_lower_bin():
    thresholds = np.asarray([1.0, 2.0, 3.0])
    values = np.asarray([0.0, 1.0, 1.5, 2.0, 3.0, 3.1])
    assert _bin_indices(values, thresholds).tolist() == [0, 0, 1, 1, 2, 3]


def test_location_ranking_is_per_voxel_and_permutation_invariant():
    locations = torch.tensor(
        [[3.0, 0.0], [1.0, 4.0], [2.0, 1.0], [0.0, 2.0], [4.0, 3.0]]
    )
    weights = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    scales = weights + 100.0
    ranked_weights, ranked_locations, ranked_scales = _rank_components(
        weights, locations, scales
    )
    assert torch.equal(ranked_locations, torch.arange(5, dtype=torch.float32)[:, None].repeat(1, 2))
    assert ranked_weights[:, 0].tolist() == [6.0, 2.0, 4.0, 0.0, 8.0]
    assert ranked_weights[:, 1].tolist() == [1.0, 5.0, 7.0, 9.0, 3.0]
    assert torch.equal(ranked_scales - ranked_weights, torch.full_like(weights, 100.0))


def test_exact_synthetic_component_probability_amplitude_and_region_partition():
    thresholds = np.asarray([1.0, 2.0])
    truth_log10rho = np.asarray([0.5, 1.5, 2.5])
    truth_delta_squared = np.asarray([1.0, 2.0, 3.0])
    survival = np.zeros((2, 5, 3), dtype=np.float64)
    survival[0, 0] = [0.0, 1.0, 1.0]
    survival[1, 0] = [0.0, 0.0, 1.0]
    component_moment_bins = np.zeros((5, 3), dtype=np.float64)
    component_moment_bins[0] = truth_delta_squared
    component_probability_bins = np.zeros_like(component_moment_bins)
    component_probability_bins[0] = 1.0
    component_totals = np.asarray([6.0, 0.0, 0.0, 0.0, 0.0])
    component_mass = np.asarray([3.0, 0.0, 0.0, 0.0, 0.0])
    sealed = {
        "strata": {
            "q99_9_and_above": {
                "truth_mean_delta_squared": 2.0,
                "V56_quadrature_mean_delta_squared": 2.0,
            }
        }
    }
    numerics = {
        "maximum_complete_moment_relative_difference_from_v56_gate": 1e-12,
        "maximum_bin_partition_relative_error": 1e-12,
        "maximum_component_partition_relative_error": 1e-12,
        "maximum_log_ratio_identity_absolute_error": 1e-12,
        "maximum_32_to_64_complete_moment_relative_difference": 0.005,
        "maximum_32_to_64_tail_moment_relative_difference_for_classification": 0.02,
        "minimum_empirical_exceedance_count_for_threshold_classification": 1,
    }
    row = _domain_summary(
        truth_log10rho,
        truth_delta_squared,
        survival,
        component_moment_bins,
        component_probability_bins,
        component_totals,
        component_moment_bins.copy(),
        component_totals.copy(),
        component_mass,
        thresholds,
        np.asarray([1.0]),
        sealed,
        numerics,
    )
    assert row["numerical_requirements_pass"] is True
    assert row["complete_moment"]["predicted_over_truth"] == 1.0
    assert row["threshold_decomposition"]["q99_999_anchor"][
        "predicted_over_truth_probability"
    ] == 1.0
    assert row["threshold_decomposition"]["grid_01"][
        "predicted_over_truth_conditional_amplitude"
    ] == 1.0
    assert row["supported_grid_error_summary"][
        "weighted_mean_absolute_log_probability_ratio"
    ] == 0.0
    assert row["ranked_component_mean_mixture_mass"] == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert sum(value["positive_excess_share"] for value in row["regions"].values()) == 0.0


def test_classification_precedence_and_frozen_actions():
    base = {
        "regions": {
            "below_grid": {"positive_excess_share": 0.2},
            "inside_grid": {"positive_excess_share": 0.3},
            "beyond_grid": {"positive_excess_share": 0.5},
        },
        "supported_grid_error_summary": {
            "available": True,
            "weighted_mean_absolute_log_probability_ratio": 2.0,
            "weighted_mean_absolute_log_conditional_amplitude_ratio": 1.0,
        },
    }
    classification, next_action = classify(False, base)
    assert classification == "V56_grid_tail_component_decomposition_is_numerically_unresolved"
    assert "higher_accuracy_train_only" in next_action
    classification, next_action = classify(True, base)
    assert classification == "V56_TNG_moment_excess_lies_beyond_scored_global_train_maximum"
    assert "reachable_output_support" in next_action
    base["regions"]["beyond_grid"]["positive_excess_share"] = 0.1
    base["regions"]["below_grid"]["positive_excess_share"] = 0.6
    classification, _ = classify(True, base)
    assert classification == "V56_TNG_moment_excess_lies_below_the_upper_survival_grid"
    base["regions"]["below_grid"]["positive_excess_share"] = 0.1
    base["regions"]["inside_grid"]["positive_excess_share"] = 0.8
    classification, _ = classify(True, base)
    assert classification == "V56_TNG_scored_grid_survival_probabilities_remain_miscalibrated"
    base["supported_grid_error_summary"]["weighted_mean_absolute_log_probability_ratio"] = 0.5
    classification, _ = classify(True, base)
    assert classification == "V56_TNG_scored_grid_is_too_coarse_for_conditional_tail_amplitude"
