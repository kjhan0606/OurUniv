import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cf4_mock_calibration import (  # noqa: E402
    CalibrationError,
    NOT_EVALUATED,
    compute_development_smoke_metrics,
    development_upstream_gate_schema,
)


MANIFEST_SHA = "d" * 64


def _inputs():
    truth = np.array([[1.0, 2.0, 1.5, 0.5], [2.0, 1.0, 0.5, 1.5]])
    offsets = np.array([-0.1, 0.1])
    posterior = truth[:, None, :] + offsets[None, :, None]
    prior_variance = np.ones(4)
    mode_bins = np.array([0, 0, 1, 1])
    declared = np.array([0, 1])
    geometry = np.array([True, False])
    upstream = development_upstream_gate_schema(
        np.array([True, True]), np.array([True, True])
    )
    return truth, posterior, prior_variance, mode_bins, declared, geometry, upstream


def test_perfect_posterior_mean_metrics_and_geometry_intersection():
    result = compute_development_smoke_metrics(
        *_inputs(), domain_id="global_delta", bin_manifest_body_sha256=MANIFEST_SHA
    )
    np.testing.assert_allclose(result["response"], [1.0, 1.0])
    np.testing.assert_allclose(result["correlation_r"], [1.0, 1.0])
    np.testing.assert_allclose(result["residual_power_ratio"], [0.0, 0.0], atol=1e-31)
    np.testing.assert_allclose(result["posterior_prior_variance_ratio_median"], [0.02, 0.02])
    assert result["data_scope"] == "implementation_smoke_input_provenance_not_validated"
    assert result["CF4_selection_noise_truth_mock_provenance_validated"] is False
    assert result["development_science_metric_allowed"] is False
    assert result["geometry_supported_metric_intersection"] == [True, False]
    assert result["coverage68_status"] == NOT_EVALUATED
    assert result["coverage95_status"] == NOT_EVALUATED
    assert result["heldout_improvement_status"] == NOT_EVALUATED
    assert result["strict_gate_before_geometry"] == [False, False]
    assert result["strict_gate_intersection_with_geometry"] == [False, False]
    assert result["strict_frontier_or_science_claim_allowed"] is False


def test_response_correlation_and_residual_formulas_for_scaled_mean():
    truth, posterior, prior, mode_bins, declared, geometry, upstream = _inputs()
    posterior = 0.5 * truth[:, None, :] + np.array([-0.05, 0.05])[None, :, None]
    result = compute_development_smoke_metrics(
        truth,
        posterior,
        prior,
        mode_bins,
        declared,
        geometry,
        upstream,
        domain_id="Local_Group_delta",
        bin_manifest_body_sha256=MANIFEST_SHA,
    )
    np.testing.assert_allclose(result["response"], [0.5, 0.5])
    np.testing.assert_allclose(result["correlation_r"], [1.0, 1.0])
    np.testing.assert_allclose(result["residual_power_ratio"], [0.25, 0.25])


def test_complex_fourier_coefficients_are_supported():
    truth, posterior, prior, mode_bins, declared, geometry, upstream = _inputs()
    truth = truth.astype(complex) * (1.0 + 1.0j)
    posterior = truth[:, None, :] + np.array([-0.1, 0.1])[None, :, None]
    result = compute_development_smoke_metrics(
        truth,
        posterior,
        prior,
        mode_bins,
        declared,
        geometry,
        upstream,
        domain_id="global_theta",
        bin_manifest_body_sha256=MANIFEST_SHA,
    )
    np.testing.assert_allclose(result["response"], [1.0, 1.0])
    np.testing.assert_allclose(result["correlation_r"], [1.0, 1.0])


def test_roundoff_above_unit_correlation_is_clipped_not_rejected():
    truth, posterior, prior, mode_bins, declared, geometry, upstream = _inputs()
    truth = np.tile(truth, (2, 1))
    posterior = np.tile(posterior, (2, 1, 1))
    result = compute_development_smoke_metrics(
        truth,
        posterior,
        prior,
        mode_bins,
        declared,
        geometry,
        upstream,
        domain_id="global_delta",
        bin_manifest_body_sha256=MANIFEST_SHA,
    )
    assert all(-1.0 <= value <= 1.0 for value in result["correlation_r"])


def test_not_evaluated_boolean_contract_is_strictly_false():
    schema = development_upstream_gate_schema([True, False], [True, True])
    assert schema["coverage68_pass"] == [False, False]
    assert schema["coverage95_pass"] == [False, False]
    assert schema["heldout_improvement_pass"] == [False, False]
    schema["coverage68_pass"][0] = True
    with pytest.raises(CalibrationError, match="must all be false"):
        compute_development_smoke_metrics(
            *_inputs()[:-1],
            schema,
            domain_id="global_delta",
            bin_manifest_body_sha256=MANIFEST_SHA,
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda values: values.__setitem__(0, np.full((2, 4), np.nan)), "nonfinite"),
        (lambda values: values.__setitem__(1, np.ones((2, 1, 4))), "at least two draws"),
        (lambda values: values.__setitem__(2, np.array([1.0, 0.0, 1.0, 1.0])), "strictly positive"),
        (lambda values: values.__setitem__(3, np.array([0, 0, 0, 0])), "every declared bin"),
        (lambda values: values.__setitem__(5, np.array([1, 0])), "exact length-2 boolean"),
    ],
)
def test_missing_nonfinite_shape_and_dtype_inputs_fail_closed(mutation, match):
    values = list(_inputs())
    mutation(values)
    with pytest.raises(CalibrationError, match=match):
        compute_development_smoke_metrics(
            *values,
            domain_id="global_delta",
            bin_manifest_body_sha256=MANIFEST_SHA,
        )


def test_zero_power_bin_fails_closed():
    values = list(_inputs())
    values[0][:, 2:] = 0.0
    values[1][:, :, 2:] = 0.0
    with pytest.raises(CalibrationError, match="zero truth/posterior-mean power"):
        compute_development_smoke_metrics(
            *values,
            domain_id="global_delta",
            bin_manifest_body_sha256=MANIFEST_SHA,
        )


@pytest.mark.parametrize("mock_count", [65, 256])
def test_above_frozen_64_development_mocks_are_rejected(mock_count):
    values = list(_inputs())
    repeats = (mock_count + values[0].shape[0] - 1) // values[0].shape[0]
    values[0] = np.tile(values[0], (repeats, 1))[:mock_count]
    values[1] = np.tile(values[1], (repeats, 1, 1))[:mock_count]
    with pytest.raises(CalibrationError, match="frozen development-smoke maximum of 64"):
        compute_development_smoke_metrics(
            *values,
            domain_id="global_delta",
            bin_manifest_body_sha256=MANIFEST_SHA,
        )


def test_upstream_schema_rejects_numeric_truthiness_and_extra_keys():
    with pytest.raises(CalibrationError, match="exact boolean"):
        development_upstream_gate_schema([1, 0], [True, True])
    values = list(_inputs())
    values[-1]["unexpected"] = False
    with pytest.raises(CalibrationError, match="key set is not exact"):
        compute_development_smoke_metrics(
            *values,
            domain_id="global_delta",
            bin_manifest_body_sha256=MANIFEST_SHA,
        )


def test_manifest_binding_is_required_and_recorded():
    result = compute_development_smoke_metrics(
        *_inputs(), domain_id="global_delta", bin_manifest_body_sha256=MANIFEST_SHA
    )
    assert result["bin_manifest_body_sha256"] == MANIFEST_SHA
    with pytest.raises(CalibrationError, match="lowercase 64-hex"):
        compute_development_smoke_metrics(
            *_inputs(), domain_id="global_delta", bin_manifest_body_sha256="bad"
        )
