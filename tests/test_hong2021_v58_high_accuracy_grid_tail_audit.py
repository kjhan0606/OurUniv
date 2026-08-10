import json
from pathlib import Path

import numpy as np
import torch

from hong2021_v18_init import sha256_file
from hong2021_v58_high_accuracy_grid_tail_audit import (
    CONTROL_ORDER,
    PRIMARY_ORDER,
    PROGRAM_SHA256,
    _cdf_interval_bins,
    _domain_summary,
    classify,
)


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/hong2021_v58_high_accuracy_grid_tail_audit_program.json"


def test_program_is_frozen_hash_bound_and_preserves_v57_classification():
    row = json.loads(PROGRAM.read_text())
    parent = row["parent_evidence"]
    record = json.loads((ROOT / parent["v57_record"]).read_text())
    v57 = json.loads((ROOT / row["frozen_inputs"]["v57_program"]).read_text())
    assert sha256_file(PROGRAM) == PROGRAM_SHA256
    assert sha256_file(ROOT / parent["v57_record"]) == parent["v57_record_sha256"]
    assert row["status"] == "frozen_before_audit_implementation_or_execution"
    assert record["audit"]["classification"] == parent["required_classification"]
    assert row["classification"]["branches"][1:] == v57["classification"]["branches"][1:]
    assert row["integration"]["training_or_refit"] is False
    assert row["firewall"]["development_access"] == "forbidden"
    assert row["firewall"]["historical_EAGLE_access"] == "forbidden"
    assert row["firewall"]["independent_gate_locked"] is True


def test_cdf_interval_quadrature_has_exact_probability_partition_and_converges():
    probes = 4
    weights = torch.zeros((5, probes), dtype=torch.float64)
    weights[0] = 1.0
    locations = torch.zeros_like(weights)
    scales = torch.ones_like(weights)
    base = torch.tensor([-0.1, -0.05, 0.05, 0.1], dtype=torch.float64)
    thresholds = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    target_std = 0.1
    primary = _cdf_interval_bins(
        weights, locations, scales, base, target_std, thresholds, PRIMARY_ORDER
    )
    control = _cdf_interval_bins(
        weights, locations, scales, base, target_std, thresholds, CONTROL_ORDER
    )
    assert np.isclose(primary["component_probability_bins"].sum(), probes, atol=1e-12)
    assert np.all(primary["component_probability_bins"] >= 0.0)
    assert np.all(primary["component_moment_bins"] >= 0.0)
    primary_total = primary["component_moment_bins"].sum()
    control_total = control["component_moment_bins"].sum()
    assert abs(primary_total - control_total) / max(primary_total, control_total) < 0.005


def test_exact_synthetic_high_accuracy_summary_closes_all_identities():
    thresholds = np.asarray([1.0, 2.0])
    truth_log10rho = np.asarray([0.5, 1.5, 2.5])
    truth_delta_squared = np.asarray([1.0, 2.0, 3.0])
    component_moments = np.zeros((5, 3), dtype=np.float64)
    component_moments[0] = truth_delta_squared
    component_probabilities = np.zeros_like(component_moments)
    component_probabilities[0] = 1.0
    primary = {
        "component_moment_bins": component_moments,
        "component_probability_bins": component_probabilities,
        "component_total_moments": np.asarray([6.0, 0.0, 0.0, 0.0, 0.0]),
    }
    control = {
        "component_moment_bins": component_moments.copy(),
        "component_probability_bins": component_probabilities.copy(),
        "component_total_moments": np.asarray([6.0, 0.0, 0.0, 0.0, 0.0]),
    }
    sealed = {
        "strata": {
            "q99_9_and_above": {
                "truth_mean_delta_squared": 2.0,
                "V56_quadrature_mean_delta_squared": 2.0,
            }
        }
    }
    numerics = {
        "maximum_exact_V56_gate_reproduction_relative_difference": 1e-12,
        "maximum_64_to_128_complete_moment_relative_difference": 0.005,
        "maximum_64_to_128_tail_moment_relative_difference": 0.005,
        "maximum_high_accuracy_complete_moment_relative_difference_from_exact_V56_64": 0.005,
        "maximum_bin_partition_relative_error": 1e-12,
        "maximum_component_partition_relative_error": 1e-12,
        "maximum_log_ratio_identity_absolute_error": 1e-12,
        "minimum_empirical_exceedance_count_for_threshold_classification": 1,
    }
    row = _domain_summary(
        truth_log10rho,
        truth_delta_squared,
        truth_delta_squared.copy(),
        primary,
        control,
        np.asarray([3.0, 0.0, 0.0, 0.0, 0.0]),
        thresholds,
        np.asarray([1.0]),
        sealed,
        numerics,
    )
    assert row["numerical_requirements_pass"] is True
    assert row["complete_moment"]["high_accuracy_over_truth"] == 1.0
    assert row["threshold_decomposition"]["q99_999_anchor"][
        "predicted_over_truth_probability"
    ] == 1.0
    assert row["threshold_decomposition"]["grid_01"][
        "predicted_over_truth_conditional_amplitude"
    ] == 1.0
    assert row["supported_grid_error_summary"][
        "weighted_mean_absolute_log_probability_ratio"
    ] == 0.0


def test_classification_keeps_numerical_precedence_and_v57_branches():
    row = {
        "regions": {
            "below_grid": {"positive_excess_share": 0.1},
            "inside_grid": {"positive_excess_share": 0.2},
            "beyond_grid": {"positive_excess_share": 0.7},
        },
        "supported_grid_error_summary": {
            "available": True,
            "weighted_mean_absolute_log_probability_ratio": 0.5,
            "weighted_mean_absolute_log_conditional_amplitude_ratio": 1.0,
        },
    }
    classification, next_action = classify(False, row)
    assert classification == "V58_high_accuracy_grid_tail_decomposition_is_numerically_unresolved"
    assert "minimal_train_only_numerical_repair" in next_action
    classification, next_action = classify(True, row)
    assert classification == "V56_TNG_moment_excess_lies_beyond_scored_global_train_maximum"
    assert "reachable_output_support" in next_action
    row["regions"]["beyond_grid"]["positive_excess_share"] = 0.1
    row["regions"]["inside_grid"]["positive_excess_share"] = 0.8
    classification, _ = classify(True, row)
    assert classification == "V56_TNG_scored_grid_is_too_coarse_for_conditional_tail_amplitude"
