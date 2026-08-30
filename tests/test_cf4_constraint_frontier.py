import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cf4_constraint_frontier import (  # noqa: E402
    OBSERVATION_CONSTRAINED,
    PRIOR_DOMINATED,
    STRUCTURE_CONDITIONED,
    classify_bins,
    contiguous_frontier,
    evaluate_field_frontiers,
    material_field_extensions,
    material_frontier_extension,
    strict_gate_mask,
)


def _all_true(size):
    return np.ones(size, dtype=bool)


def test_strict_gate_threshold_edges_are_inclusive():
    response = np.array([0.8, 1.2, 0.8, 1.2])
    correlation = np.array([0.7, 0.7, 0.7, 0.7])
    residual = np.array([0.5, 0.5, 0.5, 0.5])
    flags = _all_true(4)

    result = strict_gate_mask(
        response,
        correlation,
        residual,
        flags,
        flags,
        flags,
        flags,
        flags,
    )

    np.testing.assert_array_equal(result, flags)


def test_strict_gate_fails_each_value_just_outside_threshold():
    response = np.array([0.8 - 1e-12, 1.2 + 1e-12, 1.0, 1.0, 1.0])
    correlation = np.array([0.8, 0.8, 0.7 - 1e-12, 0.8, 0.8])
    residual = np.array([0.4, 0.4, 0.4, 0.5 + 1e-12, 0.4])
    flags = _all_true(5)
    heldout = flags.copy()
    heldout[-1] = False

    result = strict_gate_mask(
        response,
        correlation,
        residual,
        flags,
        flags,
        flags,
        flags,
        heldout,
    )

    assert not np.any(result)


def test_contiguous_frontier_ignores_hole_and_isolated_high_k_passes():
    k = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    gate_pass = np.array([True, True, False, True, True])

    result = contiguous_frontier(k, gate_pass)

    assert result.k_eff == pytest.approx(0.2)
    assert result.prefix_bin_count == 2
    assert result.first_failed_index == 2
    assert result.ignored_passing_indices == (3, 4)


def test_two_quarter_octave_extension_and_positive_bootstrap_lower_bound_go():
    k = 0.1 * 2.0 ** (np.arange(5) / 4.0)
    baseline = np.array([True, True, False, False, False])
    candidate = np.array([True, True, True, True, False])
    bootstrap_delta = np.linspace(0.01, 0.2, 101)

    result = material_frontier_extension(
        k, baseline, candidate, bootstrap_delta
    )

    assert result.go
    assert result.new_contiguous_quarter_octave_bins == 2
    assert result.k_eff_ratio == pytest.approx(np.sqrt(2.0))
    assert result.bootstrap_delta_log2_lower_95 > 0.0
    assert result.failed_requirements == ()


def test_one_bin_extension_and_narrow_no_baseline_cases_fail():
    k = 0.1 * 2.0 ** (np.arange(4) / 4.0)
    baseline = np.array([True, True, False, False])
    one_bin_candidate = np.array([True, True, True, False])
    positive_bootstrap = np.full(32, 0.1)

    one_bin = material_frontier_extension(
        k, baseline, one_bin_candidate, positive_bootstrap
    )
    narrow = material_frontier_extension(
        k,
        np.array([False, False, False, False]),
        np.array([True, True, False, False]),
        positive_bootstrap,
    )

    assert not one_bin.go
    assert one_bin.new_contiguous_quarter_octave_bins == 1
    assert "fewer_than_two_new_contiguous_quarter_octave_bins" in (
        one_bin.failed_requirements
    )
    assert not narrow.go
    assert "baseline_has_no_contiguous_frontier" in narrow.failed_requirements


def test_bootstrap_lower_bound_must_be_strictly_positive():
    k = 0.1 * 2.0 ** (np.arange(5) / 4.0)
    result = material_frontier_extension(
        k,
        np.array([True, True, False, False, False]),
        np.array([True, True, True, True, False]),
        np.zeros(64),
    )

    assert not result.go
    assert "bootstrap_95_percent_lower_bound_not_positive" in (
        result.failed_requirements
    )


def test_classification_separates_field_observations_from_structure_summaries():
    classes = classify_bins(
        all_data_pass=np.array([True, True, True, False, False]),
        field_observation_only_pass=np.array(
            [True, False, False, True, False]
        ),
        structure_leave_one_out_attribution_pass=np.array(
            [False, True, False, True, True]
        ),
    )

    np.testing.assert_array_equal(
        classes,
        np.array(
            [
                OBSERVATION_CONSTRAINED,
                STRUCTURE_CONDITIONED,
                PRIOR_DOMINATED,
                PRIOR_DOMINATED,
                PRIOR_DOMINATED,
            ]
        ),
    )


@pytest.mark.parametrize(
    "density_pass, velocity_pass, density_k, velocity_k, joint_k",
    [
        (
            np.array([True, True, True, False]),
            np.array([True, True, False, True]),
            0.3,
            0.2,
            0.2,
        ),
        (
            np.array([True, False, True, True]),
            np.array([True, True, True, False]),
            0.1,
            0.3,
            0.1,
        ),
    ],
)
def test_joint_field_frontier_is_minimum_without_cross_field_masking(
    density_pass, velocity_pass, density_k, velocity_k, joint_k
):
    result = evaluate_field_frontiers(
        np.array([0.1, 0.2, 0.3, 0.4]), density_pass, velocity_pass
    )

    assert result.density_delta.k_eff == pytest.approx(density_k)
    assert result.velocity_divergence_theta.k_eff == pytest.approx(velocity_k)
    assert result.joint.k_eff == pytest.approx(joint_k)


def test_field_material_go_decisions_are_independent_and_joint_is_diagnostic():
    k = 0.1 * 2.0 ** (np.arange(5) / 4.0)
    baseline_density = np.array([True, True, False, False, False])
    baseline_velocity = np.array([True, True, False, False, False])
    extended = np.array([True, True, True, True, False])
    not_extended = np.array([True, True, False, True, True])
    positive_bootstrap = np.full(64, 0.1)

    density_only = material_field_extensions(
        k,
        baseline_density,
        baseline_velocity,
        extended,
        not_extended,
        positive_bootstrap,
        positive_bootstrap,
        positive_bootstrap,
    )
    both_fields = material_field_extensions(
        k,
        baseline_density,
        baseline_velocity,
        extended,
        extended,
        positive_bootstrap,
        positive_bootstrap,
        positive_bootstrap,
    )

    assert density_only.candidate_fields.density_delta.k_eff == pytest.approx(
        k[3]
    )
    assert (
        density_only.candidate_fields.velocity_divergence_theta.k_eff
        == pytest.approx(k[1])
    )
    assert density_only.candidate_fields.joint.k_eff == pytest.approx(k[1])
    assert density_only.density_material_extension.go
    assert not density_only.theta_material_extension.go
    assert not density_only.joint_material_extension.go
    assert both_fields.candidate_fields.density_delta.k_eff == pytest.approx(k[3])
    assert (
        both_fields.candidate_fields.velocity_divergence_theta.k_eff
        == pytest.approx(k[3])
    )
    assert both_fields.density_material_extension.go
    assert both_fields.theta_material_extension.go
    assert both_fields.joint_material_extension.go


@pytest.mark.parametrize(
    "operation, match",
    [
        (
            lambda: contiguous_frontier(
                np.array([0.1, 0.1]), np.array([True, True])
            ),
            "strictly increasing",
        ),
        (
            lambda: contiguous_frontier(
                np.array([0.1, np.nan]), np.array([True, True])
            ),
            "finite",
        ),
        (
            lambda: contiguous_frontier(
                np.array([0.1, 0.2]), np.array([True])
            ),
            "shape",
        ),
        (
            lambda: contiguous_frontier(
                np.array([0.1, 0.2]), np.array([1, 0])
            ),
            "boolean dtype",
        ),
        (
            lambda: strict_gate_mask(
                np.array([1.0, np.inf]),
                np.array([0.8, 0.8]),
                np.array([0.4, 0.4]),
                _all_true(2),
                _all_true(2),
                _all_true(2),
                _all_true(2),
                _all_true(2),
            ),
            "finite",
        ),
        (
            lambda: strict_gate_mask(
                np.array([1.0]),
                np.array([1.01]),
                np.array([0.4]),
                _all_true(1),
                _all_true(1),
                _all_true(1),
                _all_true(1),
                _all_true(1),
            ),
            "within",
        ),
        (
            lambda: strict_gate_mask(
                np.array([1.0]),
                np.array([0.8]),
                np.array([-0.1]),
                _all_true(1),
                _all_true(1),
                _all_true(1),
                _all_true(1),
                _all_true(1),
            ),
            "non-negative",
        ),
        (
            lambda: classify_bins(
                np.array([True, False]),
                np.array([True]),
                np.array([False, False]),
            ),
            "shape",
        ),
        (
            lambda: evaluate_field_frontiers(
                np.array([0.1, 0.2]),
                np.array([True, True]),
                np.array([True]),
            ),
            "shape",
        ),
        (
            lambda: material_frontier_extension(
                np.array([0.1, 0.2]),
                np.array([True, False]),
                np.array([True, True]),
                np.array([0.1, np.nan]),
            ),
            "finite",
        ),
    ],
)
def test_invalid_arrays_fail_closed(operation, match):
    with pytest.raises(ValueError, match=match):
        operation()
