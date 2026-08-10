import json
from pathlib import Path

import numpy as np

from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v55_tail_amplitude_audit import (
    PROGRAM_SHA256,
    _bin_indices,
    _domain_summary,
    classify,
)


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/hong2021_v55_tail_amplitude_audit_program.json"


def test_program_is_frozen_and_hash_bound():
    row = json.loads(PROGRAM.read_text())
    assert sha256_file(PROGRAM) == PROGRAM_SHA256
    assert row["status"] == "frozen_before_audit_implementation_or_execution"
    assert row["firewall"]["development_access"] == "forbidden"
    assert row["firewall"]["historical_EAGLE_access"] == "forbidden"
    assert row["firewall"]["independent_gate_locked"] is True
    assert np.all(np.diff(row["fixed_output_thresholds"]["values"]) > 0.0)


def test_strict_exceedance_bin_edges_remain_in_lower_bin():
    thresholds = np.asarray([1.0, 2.0, 3.0, 4.0])
    values = np.asarray([0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 4.1])
    assert _bin_indices(values, thresholds).tolist() == [0, 0, 1, 1, 2, 3, 4]


def test_classification_precedence_and_frozen_actions():
    all_true = {domain: True for domain in DOMAIN_ORDER}
    all_false = {domain: False for domain in DOMAIN_ORDER}
    classification, next_action = classify(False, all_true, all_true)
    assert classification == "fixed_threshold_tail_decomposition_is_numerically_or_empirically_unresolved"
    assert next_action.startswith("freeze_a_higher_accuracy_train_only")
    classification, next_action = classify(True, all_true, all_false)
    assert classification == "V54_probability_score_leaves_beyond_highest_threshold_amplitudes_unconstrained"
    assert "proper_survival_score_grid" in next_action
    classification, _ = classify(True, all_false, all_true)
    assert classification == "V54_highest_fixed_threshold_exceedance_probability_remains_miscalibrated"


def test_mixed_domain_does_not_select_common_mechanism():
    amplitude = {domain: domain != "SIMBA" for domain in DOMAIN_ORDER}
    probability = {domain: domain == "SIMBA" for domain in DOMAIN_ORDER}
    classification, next_action = classify(True, amplitude, probability)
    assert classification == "V54_tail_failure_is_mixed_across_probability_amplitude_or_domain"
    assert next_action == "seal_the_domainwise_fixed_bin_decomposition_before_selecting_any_further_model"


def test_exact_synthetic_probability_amplitude_and_bin_decomposition():
    thresholds = np.asarray([1.0, 2.0, 3.0, 4.0])
    log10rho = np.asarray([0.0, 1.5, 2.5, 3.5, 4.5])
    delta_squared = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    predicted_probability = np.stack(
        [(log10rho > threshold).astype(np.float64) for threshold in thresholds]
    )
    sealed = {
        "strata": {
            "q99_9_and_above": {
                "truth_mean_delta_squared": 3.0,
                "V54_quadrature_mean_delta_squared": 3.0,
            }
        }
    }
    numerics = {
        "maximum_complete_moment_relative_difference_from_v54_gate": 1e-10,
        "maximum_bin_partition_relative_error": 1e-12,
        "maximum_log_ratio_identity_absolute_error": 1e-12,
        "maximum_32_to_64_complete_moment_relative_difference": 0.005,
        "maximum_32_to_64_tail_moment_relative_difference_for_classification": 0.02,
        "minimum_empirical_exceedance_count_for_classification": 1,
    }
    row = _domain_summary(
        log10rho,
        delta_squared,
        predicted_probability,
        delta_squared.copy(),
        np.ones(5),
        15.0,
        delta_squared.copy(),
        15.0,
        thresholds,
        sealed,
        numerics,
    )
    assert row["numerical_requirements_pass"] is True
    assert row["complete_moment"]["predicted_over_truth"] == 1.0
    assert row["threshold_decomposition"]["q99_999"]["predicted_over_truth_tail_moment"] == 1.0
    assert row["threshold_decomposition"]["q99_999"]["log_ratio_identity_absolute_error"] == 0.0
    assert sum(
        value["truth_mean_delta_squared_contribution"]
        for value in row["fixed_output_bins"].values()
    ) == 3.0
