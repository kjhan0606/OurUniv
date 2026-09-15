"""Development-smoke CF4-only mock calibration metrics.

This module deliberately cannot run the untouched 256-mock validation.  It
computes descriptive per-bin metrics and prepares the exact booleans consumed
by ``cf4_constraint_frontier.strict_gate_mask``.  Coverage and held-out gates
remain NOT_EVALUATED and therefore fail every development-smoke strict gate.
"""

from __future__ import annotations

import math
import re
from typing import Mapping

import numpy as np

from cf4_constraint_frontier import strict_gate_mask


UPSTREAM_SCHEMA = "ouruniv-cf4-development-smoke-upstream-gates-v1"
NOT_EVALUATED = "NOT_EVALUATED"
MAX_DEVELOPMENT_SMOKE_MOCKS = 64
VARIANCE_RATIO_MEDIAN_MAX = 0.8


class CalibrationError(ValueError):
    """A calibration input is incomplete, nonfinite, or contract-invalid."""


def _field_array(value: object, ndim: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{label} must be a numeric {ndim}D array") from exc
    if array.ndim != ndim or any(size == 0 for size in array.shape):
        raise CalibrationError(f"{label} must be a non-empty {ndim}D array")
    if array.dtype == np.dtype(bool) or not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise CalibrationError(f"{label} must contain real or complex numbers")
    result = array.astype(np.complex128 if np.iscomplexobj(array) else np.float64)
    if not np.all(np.isfinite(result)):
        raise CalibrationError(f"{label} contains nonfinite values")
    return result


def _real_vector(value: object, size: int, label: str, *, positive: bool = False) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{label} must be a real vector") from exc
    if array.ndim != 1 or array.size != size or array.dtype == np.dtype(bool):
        raise CalibrationError(f"{label} must be a length-{size} real vector")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise CalibrationError(f"{label} must be a real vector")
    result = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise CalibrationError(f"{label} contains nonfinite values")
    if positive and np.any(result <= 0.0):
        raise CalibrationError(f"{label} must be strictly positive")
    return result


def _boolean_vector(value: object, size: int, label: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{label} must be a boolean vector") from exc
    if array.ndim != 1 or array.size != size or array.dtype != np.dtype(bool):
        raise CalibrationError(f"{label} must be an exact length-{size} boolean vector")
    return array


def development_upstream_gate_schema(
    phase_null_pass: object,
    variance_bootstrap_95_upper_below_one_pass: object,
) -> dict[str, object]:
    """Create the only accepted development-smoke upstream gate document."""

    phase = np.asarray(phase_null_pass)
    variance = np.asarray(variance_bootstrap_95_upper_below_one_pass)
    if phase.ndim != 1 or phase.size == 0 or phase.dtype != np.dtype(bool):
        raise CalibrationError("phase_null_pass must be a non-empty exact boolean vector")
    if variance.shape != phase.shape or variance.dtype != np.dtype(bool):
        raise CalibrationError("variance bootstrap pass must match phase boolean shape")
    false_values = [False] * int(phase.size)
    return {
        "schema": UPSTREAM_SCHEMA,
        "phase_null_pass": phase.tolist(),
        "variance_bootstrap_95_upper_below_one_pass": variance.tolist(),
        "coverage68_status": NOT_EVALUATED,
        "coverage68_pass": false_values.copy(),
        "coverage95_status": NOT_EVALUATED,
        "coverage95_pass": false_values.copy(),
        "heldout_improvement_status": NOT_EVALUATED,
        "heldout_improvement_pass": false_values.copy(),
    }


def _validate_upstream(value: object, size: int) -> dict[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise CalibrationError("upstream gates must be a mapping")
    exact_keys = {
        "schema",
        "phase_null_pass",
        "variance_bootstrap_95_upper_below_one_pass",
        "coverage68_status",
        "coverage68_pass",
        "coverage95_status",
        "coverage95_pass",
        "heldout_improvement_status",
        "heldout_improvement_pass",
    }
    if set(value) != exact_keys or value.get("schema") != UPSTREAM_SCHEMA:
        raise CalibrationError("upstream gate schema/key set is not exact")
    for status in (
        "coverage68_status",
        "coverage95_status",
        "heldout_improvement_status",
    ):
        if value.get(status) != NOT_EVALUATED:
            raise CalibrationError(
                "development smoke requires coverage and held-out status NOT_EVALUATED"
            )
    output = {
        "phase": _boolean_vector(value["phase_null_pass"], size, "phase_null_pass"),
        "variance_bootstrap": _boolean_vector(
            value["variance_bootstrap_95_upper_below_one_pass"],
            size,
            "variance bootstrap pass",
        ),
        "coverage68": _boolean_vector(value["coverage68_pass"], size, "coverage68_pass"),
        "coverage95": _boolean_vector(value["coverage95_pass"], size, "coverage95_pass"),
        "heldout": _boolean_vector(
            value["heldout_improvement_pass"], size, "heldout_improvement_pass"
        ),
    }
    if np.any(output["coverage68"]) or np.any(output["coverage95"]) or np.any(output["heldout"]):
        raise CalibrationError("NOT_EVALUATED coverage/held-out booleans must all be false")
    return output


def compute_development_smoke_metrics(
    truth: object,
    posterior_ensemble: object,
    prior_variance: object,
    mode_bin_index: object,
    declared_bin_ids: object,
    geometry_supported: object,
    upstream_gates: object,
    *,
    domain_id: str,
    bin_manifest_body_sha256: str,
    posterior_mean: object | None = None,
) -> dict[str, object]:
    """Compute per-bin CF4-only metrics without permitting a science gate.

    ``truth`` has shape ``(mock, mode)`` and ``posterior_ensemble`` has shape
    ``(mock, draw, mode)``.  Real or complex Fourier coefficients are allowed.
    ``posterior_mean`` may supply the analytic mean with shape ``(mock, mode)``;
    if omitted, the historical draw mean is retained for compatibility.  Draws
    always determine the ddof=1 posterior variance.  Every declared bin must
    own at least one mode; omission fails closed.
    """

    truth_array = _field_array(truth, 2, "truth")
    posterior = _field_array(posterior_ensemble, 3, "posterior_ensemble")
    if posterior.shape[0] != truth_array.shape[0] or posterior.shape[2] != truth_array.shape[1]:
        raise CalibrationError("posterior ensemble mock/mode axes must match truth")
    if posterior.shape[1] < 2:
        raise CalibrationError("posterior ensemble requires at least two draws")
    mock_count, mode_count = truth_array.shape
    if mock_count > MAX_DEVELOPMENT_SMOKE_MOCKS:
        raise CalibrationError(
            "mock_count exceeds frozen development-smoke maximum of 64"
        )
    if not isinstance(domain_id, str) or not domain_id:
        raise CalibrationError("domain_id must be a non-empty string")
    if (
        not isinstance(bin_manifest_body_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", bin_manifest_body_sha256) is None
    ):
        raise CalibrationError("bin_manifest_body_sha256 must be lowercase 64-hex")

    bins = np.asarray(declared_bin_ids)
    if (
        bins.ndim != 1
        or bins.size == 0
        or bins.dtype == np.dtype(bool)
        or not np.issubdtype(bins.dtype, np.integer)
    ):
        raise CalibrationError("declared_bin_ids must be a non-empty integer vector")
    bins = bins.astype(np.int64, copy=False)
    if np.any(np.diff(bins) <= 0):
        raise CalibrationError("declared_bin_ids must be strictly increasing")
    assignment = np.asarray(mode_bin_index)
    if (
        assignment.ndim != 1
        or assignment.size != mode_count
        or assignment.dtype == np.dtype(bool)
        or not np.issubdtype(assignment.dtype, np.integer)
    ):
        raise CalibrationError("mode_bin_index must be an integer vector matching modes")
    assignment = assignment.astype(np.int64, copy=False)
    if np.any(~np.isin(assignment, bins)):
        raise CalibrationError("mode_bin_index references an undeclared bin")
    if any(not np.any(assignment == bin_id) for bin_id in bins):
        raise CalibrationError("every declared bin must contain at least one mode")

    geometry = _boolean_vector(geometry_supported, bins.size, "geometry_supported")
    upstream = _validate_upstream(upstream_gates, bins.size)

    prior = np.asarray(prior_variance)
    if prior.ndim == 1:
        prior = _real_vector(prior, mode_count, "prior_variance", positive=True)
        prior = np.broadcast_to(prior[None, :], truth_array.shape)
    elif prior.ndim == 2:
        if prior.shape != truth_array.shape:
            raise CalibrationError("2D prior_variance must match truth shape")
        if prior.dtype == np.dtype(bool) or not np.issubdtype(prior.dtype, np.number):
            raise CalibrationError("prior_variance must be numeric")
        if np.iscomplexobj(prior):
            raise CalibrationError("prior_variance must be real")
        prior = prior.astype(np.float64, copy=False)
        if not np.all(np.isfinite(prior)) or np.any(prior <= 0.0):
            raise CalibrationError("prior_variance must be finite and strictly positive")
    else:
        raise CalibrationError("prior_variance must be 1D per-mode or 2D per-mock/mode")

    if posterior_mean is None:
        posterior_mean_array = posterior.mean(axis=1)
        posterior_mean_source = "ensemble_draw_mean_backward_compatible_default"
    else:
        posterior_mean_array = _field_array(posterior_mean, 2, "posterior_mean")
        if posterior_mean_array.shape != truth_array.shape:
            raise CalibrationError("explicit posterior_mean must match truth shape")
        posterior_mean_source = "explicit_analytic_posterior_mean"
    posterior_variance = np.var(posterior, axis=1, ddof=1)
    response: list[float] = []
    correlation: list[float] = []
    residual_ratio: list[float] = []
    variance_ratio: list[float] = []
    mode_counts: list[int] = []
    for bin_id in bins:
        mask = assignment == bin_id
        truth_bin = truth_array[:, mask].reshape(-1)
        mean_bin = posterior_mean_array[:, mask].reshape(-1)
        truth_power = float(np.sum(np.abs(truth_bin) ** 2))
        mean_power = float(np.sum(np.abs(mean_bin) ** 2))
        cross = float(np.real(np.vdot(truth_bin, mean_bin)))
        if truth_power <= 0.0 or mean_power <= 0.0:
            raise CalibrationError(f"zero truth/posterior-mean power in bin {int(bin_id)}")
        response.append(cross / truth_power)
        correlation.append(cross / math.sqrt(truth_power * mean_power))
        residual_ratio.append(
            float(np.sum(np.abs(mean_bin - truth_bin) ** 2)) / truth_power
        )
        ratios = (posterior_variance[:, mask] / prior[:, mask]).reshape(-1)
        if not np.all(np.isfinite(ratios)) or np.any(ratios < 0.0):
            raise CalibrationError(f"invalid posterior/prior variance ratio in bin {int(bin_id)}")
        variance_ratio.append(float(np.median(ratios)))
        mode_counts.append(int(np.count_nonzero(mask)))

    response_array = np.asarray(response, dtype=float)
    correlation_array = np.asarray(correlation, dtype=float)
    residual_array = np.asarray(residual_ratio, dtype=float)
    variance_array = np.asarray(variance_ratio, dtype=float)
    if not all(
        np.all(np.isfinite(array))
        for array in (response_array, correlation_array, residual_array, variance_array)
    ):
        raise CalibrationError("computed metrics contain nonfinite values")
    correlation_tolerance = 32.0 * np.finfo(np.float64).eps
    if np.any(
        (correlation_array < -1.0 - correlation_tolerance)
        | (correlation_array > 1.0 + correlation_tolerance)
    ):
        raise CalibrationError("computed r(k) lies outside [-1,1]")
    correlation_array = np.clip(correlation_array, -1.0, 1.0)

    variance_point_pass = variance_array <= VARIANCE_RATIO_MEDIAN_MAX
    variance_pass = variance_point_pass & upstream["variance_bootstrap"]
    strict_before_geometry = strict_gate_mask(
        response_array,
        correlation_array,
        residual_array,
        upstream["phase"],
        variance_pass,
        upstream["coverage68"],
        upstream["coverage95"],
        upstream["heldout"],
    )
    strict_with_geometry = strict_before_geometry & geometry
    if np.any(strict_before_geometry) or np.any(strict_with_geometry):
        raise CalibrationError("development smoke cannot pass strict coverage/held-out gates")

    return {
        "schema": "ouruniv-cf4-development-smoke-calibration-metrics-v1",
        "status": "DEVELOPMENT_SMOKE_NOT_VALIDATION",
        "data_scope": "implementation_smoke_input_provenance_not_validated",
        "CF4_selection_noise_truth_mock_provenance_validated": False,
        "development_science_metric_allowed": False,
        "domain_id": domain_id,
        "bin_manifest_body_sha256": bin_manifest_body_sha256,
        "mock_count": int(mock_count),
        "posterior_draw_count": int(posterior.shape[1]),
        "posterior_mean_source": posterior_mean_source,
        "declared_bin_ids": bins.tolist(),
        "modes_per_mock_by_bin": mode_counts,
        "response": response_array.tolist(),
        "correlation_r": correlation_array.tolist(),
        "residual_power_ratio": residual_array.tolist(),
        "posterior_prior_variance_ratio_median": variance_array.tolist(),
        "metric_definitions": {
            "response": "Re(sum truth_conjugate*posterior_mean)/sum(abs(truth)^2)",
            "correlation_r": (
                "Re(sum truth_conjugate*posterior_mean)/"
                "sqrt(sum(abs(truth)^2)*sum(abs(posterior_mean)^2))"
            ),
            "residual_power_ratio": "sum(abs(posterior_mean-truth)^2)/sum(abs(truth)^2)",
            "posterior_prior_variance_ratio_median": (
                "median_over_mock_mode_of("
                "sample_variance_across_draws_ddof_1/prior_variance)"
            ),
        },
        "variance_ratio_median_le_0_8_pass": variance_point_pass.tolist(),
        "phase_null_pass_upstream": upstream["phase"].tolist(),
        "variance_bootstrap_95_upper_below_one_pass_upstream": upstream[
            "variance_bootstrap"
        ].tolist(),
        "geometry_supported": geometry.tolist(),
        "metrics_available": [True] * int(bins.size),
        "geometry_supported_metric_intersection": geometry.tolist(),
        "strict_gate_before_geometry": strict_before_geometry.tolist(),
        "strict_gate_intersection_with_geometry": strict_with_geometry.tolist(),
        "coverage68_status": NOT_EVALUATED,
        "coverage68_pass": upstream["coverage68"].tolist(),
        "coverage95_status": NOT_EVALUATED,
        "coverage95_pass": upstream["coverage95"].tolist(),
        "heldout_improvement_status": NOT_EVALUATED,
        "heldout_improvement_pass": upstream["heldout"].tolist(),
        "upstream_gate_schema": UPSTREAM_SCHEMA,
        "strict_frontier_or_science_claim_allowed": False,
        "untouched_256_mock_validation_executed": False,
    }


__all__ = [
    "CalibrationError",
    "MAX_DEVELOPMENT_SMOKE_MOCKS",
    "NOT_EVALUATED",
    "UPSTREAM_SCHEMA",
    "VARIANCE_RATIO_MEDIAN_MAX",
    "compute_development_smoke_metrics",
    "development_upstream_gate_schema",
]
