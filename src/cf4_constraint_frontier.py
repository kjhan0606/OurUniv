"""Fail-closed constraint-frontier utilities for the CF4 KF-DESIGN contract.

The functions in this module consume the complete, immutable-manifest-bound
evaluation vector.  They never accept a science cutoff: every declared bin is
checked from the supported lowest k upward, and the first failed bin terminates
the usable contiguous frontier.  The caller must verify the bin-manifest
SHA256 before invoking this module; omitted or reordered bins are invalid
inputs, not an alternative analysis domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


RESPONSE_MIN = 0.8
RESPONSE_MAX = 1.2
CORRELATION_MIN = 0.7
RESIDUAL_POWER_RATIO_MAX = 0.5
QUARTER_OCTAVE_DELTA_LOG2 = 0.25
MINIMUM_NEW_QUARTER_OCTAVE_BINS = 2
MINIMUM_K_EFF_RATIO = float(np.sqrt(2.0))

OBSERVATION_CONSTRAINED = "observation_constrained"
STRUCTURE_CONDITIONED = "structure_conditioned"
PRIOR_DOMINATED = "prior_dominated"


@dataclass(frozen=True)
class FrontierResult:
    """The strict contiguous-prefix frontier for one declared domain."""

    k_eff: float | None
    prefix_bin_count: int
    first_failed_index: int | None
    ignored_passing_indices: tuple[int, ...]


@dataclass(frozen=True)
class MaterialExtensionResult:
    """Decision record for a candidate frontier versus its frozen baseline."""

    go: bool
    baseline: FrontierResult
    candidate: FrontierResult
    new_contiguous_quarter_octave_bins: int
    k_eff_ratio: float | None
    bootstrap_delta_log2_lower_95: float
    failed_requirements: tuple[str, ...]


@dataclass(frozen=True)
class FieldFrontierResult:
    """Separate density/velocity frontiers and their fail-closed joint value."""

    density_delta: FrontierResult
    velocity_divergence_theta: FrontierResult
    joint: FrontierResult


@dataclass(frozen=True)
class FieldExtensionResult:
    """Independent delta/theta decisions plus the coupled joint diagnostic."""

    baseline_fields: FieldFrontierResult
    candidate_fields: FieldFrontierResult
    density_material_extension: MaterialExtensionResult
    theta_material_extension: MaterialExtensionResult
    joint_material_extension: MaterialExtensionResult


def _numeric_1d(name: str, values: Iterable[float]) -> np.ndarray:
    """Return a finite, non-empty, real one-dimensional float array."""

    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional real array") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must contain real numeric values")
    result = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _boolean_1d(name: str, values: Iterable[bool]) -> np.ndarray:
    """Return a non-empty boolean vector without truthy numeric coercion."""

    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional boolean array") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if array.dtype != np.dtype(np.bool_):
        raise ValueError(f"{name} must have exact boolean dtype")
    return array


def _require_same_shape(reference_name: str, reference: np.ndarray, **others: np.ndarray) -> None:
    for name, array in others.items():
        if array.shape != reference.shape:
            raise ValueError(
                f"{name} shape {array.shape} does not match "
                f"{reference_name} shape {reference.shape}"
            )


def _validated_k(k: Iterable[float]) -> np.ndarray:
    values = _numeric_1d("k", k)
    if np.any(values <= 0.0):
        raise ValueError("k must be strictly positive")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("k must be strictly increasing")
    return values


def strict_gate_mask(
    response: Iterable[float],
    correlation: Iterable[float],
    residual_power_ratio: Iterable[float],
    phase_null_pass: Iterable[bool],
    variance_reduction_pass: Iterable[bool],
    coverage68_pass: Iterable[bool],
    coverage95_pass: Iterable[bool],
    heldout_improvement_pass: Iterable[bool],
) -> np.ndarray:
    """Return the inclusive v3 all-gate pass mask for every declared bin.

    Phase significance, material variance reduction, calibrated 68/95 percent
    coverage, and held-out improvement are preregistered upstream tests.  They
    are accepted here only as exact booleans so NaN or numeric truthiness cannot
    silently pass a bin.
    """

    response_array = _numeric_1d("response", response)
    correlation_array = _numeric_1d("correlation", correlation)
    residual_array = _numeric_1d("residual_power_ratio", residual_power_ratio)
    phase_array = _boolean_1d("phase_null_pass", phase_null_pass)
    variance_array = _boolean_1d(
        "variance_reduction_pass", variance_reduction_pass
    )
    coverage68_array = _boolean_1d("coverage68_pass", coverage68_pass)
    coverage95_array = _boolean_1d("coverage95_pass", coverage95_pass)
    heldout_array = _boolean_1d(
        "heldout_improvement_pass", heldout_improvement_pass
    )
    _require_same_shape(
        "response",
        response_array,
        correlation=correlation_array,
        residual_power_ratio=residual_array,
        phase_null_pass=phase_array,
        variance_reduction_pass=variance_array,
        coverage68_pass=coverage68_array,
        coverage95_pass=coverage95_array,
        heldout_improvement_pass=heldout_array,
    )
    if np.any((correlation_array < -1.0) | (correlation_array > 1.0)):
        raise ValueError("correlation must lie within [-1, 1]")
    if np.any(residual_array < 0.0):
        raise ValueError("residual_power_ratio must be non-negative")

    return (
        (response_array >= RESPONSE_MIN)
        & (response_array <= RESPONSE_MAX)
        & (correlation_array >= CORRELATION_MIN)
        & (residual_array <= RESIDUAL_POWER_RATIO_MAX)
        & phase_array
        & variance_array
        & coverage68_array
        & coverage95_array
        & heldout_array
    )


def contiguous_frontier(
    k: Iterable[float], gate_pass: Iterable[bool]
) -> FrontierResult:
    """Find k_eff from the strict passing prefix beginning at the lowest k.

    Any passing bins beyond the first failure are recorded but ignored.  This
    makes holes and isolated high-k successes fail closed.
    """

    k_array = _validated_k(k)
    pass_array = _boolean_1d("gate_pass", gate_pass)
    _require_same_shape("k", k_array, gate_pass=pass_array)

    failures = np.flatnonzero(~pass_array)
    first_failed = int(failures[0]) if failures.size else None
    prefix_count = len(k_array) if first_failed is None else first_failed
    k_eff = float(k_array[prefix_count - 1]) if prefix_count else None

    if first_failed is None:
        ignored = ()
    else:
        ignored = tuple(
            int(index)
            for index in np.flatnonzero(pass_array[first_failed + 1 :])
            + first_failed
            + 1
        )

    return FrontierResult(
        k_eff=k_eff,
        prefix_bin_count=prefix_count,
        first_failed_index=first_failed,
        ignored_passing_indices=ignored,
    )


def evaluate_field_frontiers(
    k: Iterable[float],
    density_delta_pass: Iterable[bool],
    velocity_divergence_theta_pass: Iterable[bool],
) -> FieldFrontierResult:
    """Evaluate delta and theta separately; the joint frontier is their min.

    A failure in either field terminates the joint prefix.  Consequently a
    density success cannot hide a velocity failure, or vice versa.  This is a
    z=0 field diagnostic only; an IC frontier is measured later in IC-HYBRID.
    """

    k_array = _validated_k(k)
    density_pass = _boolean_1d("density_delta_pass", density_delta_pass)
    velocity_pass = _boolean_1d(
        "velocity_divergence_theta_pass", velocity_divergence_theta_pass
    )
    _require_same_shape(
        "k",
        k_array,
        density_delta_pass=density_pass,
        velocity_divergence_theta_pass=velocity_pass,
    )

    density = contiguous_frontier(k_array, density_pass)
    velocity = contiguous_frontier(k_array, velocity_pass)
    joint = contiguous_frontier(k_array, density_pass & velocity_pass)
    expected_joint = (
        None
        if density.k_eff is None or velocity.k_eff is None
        else min(density.k_eff, velocity.k_eff)
    )
    if joint.k_eff != expected_joint:
        raise AssertionError("joint field frontier must equal min(delta, theta)")
    return FieldFrontierResult(
        density_delta=density,
        velocity_divergence_theta=velocity,
        joint=joint,
    )


def classify_bins(
    all_data_pass: Iterable[bool],
    field_observation_only_pass: Iterable[bool],
    structure_leave_one_out_attribution_pass: Iterable[bool],
) -> np.ndarray:
    """Classify all-D bins using field observations and structure summaries.

    Non-CF4 is deliberately not used as a synonym for structure: an
    independently proven galaxy-density catalog belongs to the field-observation
    subset even though it is non-CF4.
    """

    all_data = _boolean_1d("all_data_pass", all_data_pass)
    field_observation = _boolean_1d(
        "field_observation_only_pass", field_observation_only_pass
    )
    structure_attribution = _boolean_1d(
        "structure_leave_one_out_attribution_pass",
        structure_leave_one_out_attribution_pass,
    )
    _require_same_shape(
        "all_data_pass",
        all_data,
        field_observation_only_pass=field_observation,
        structure_leave_one_out_attribution_pass=structure_attribution,
    )

    result = np.full(all_data.shape, PRIOR_DOMINATED, dtype="<U23")
    result[all_data & field_observation] = OBSERVATION_CONSTRAINED
    result[
        all_data & ~field_observation & structure_attribution
    ] = STRUCTURE_CONDITIONED
    return result


def material_frontier_extension(
    k: Iterable[float],
    baseline_pass: Iterable[bool],
    candidate_pass: Iterable[bool],
    bootstrap_delta_log2: Iterable[float],
) -> MaterialExtensionResult:
    """Test the fixed material-extension contract against a frozen baseline.

    Both frontiers use the same declared k bins.  GO requires a valid baseline,
    at least two contiguous quarter-octaves of extension (equivalently a
    k_eff ratio of at least sqrt(2)), and a strictly positive one-sided 95%
    bootstrap lower bound for log2(k_eff_candidate/k_eff_baseline).  Isolated
    passes never enter either frontier.
    """

    k_array = _validated_k(k)
    baseline_array = _boolean_1d("baseline_pass", baseline_pass)
    candidate_array = _boolean_1d("candidate_pass", candidate_pass)
    _require_same_shape(
        "k",
        k_array,
        baseline_pass=baseline_array,
        candidate_pass=candidate_array,
    )
    bootstrap = _numeric_1d("bootstrap_delta_log2", bootstrap_delta_log2)

    baseline = contiguous_frontier(k_array, baseline_array)
    candidate = contiguous_frontier(k_array, candidate_array)
    lower_95 = float(np.quantile(bootstrap, 0.05, method="linear"))

    failed: list[str] = []
    ratio: float | None = None
    new_quarter_bins = 0
    if baseline.k_eff is None:
        failed.append("baseline_has_no_contiguous_frontier")
    elif candidate.k_eff is None:
        failed.append("candidate_has_no_contiguous_frontier")
    else:
        ratio = candidate.k_eff / baseline.k_eff
        delta_log2 = float(np.log2(ratio))
        new_quarter_bins = max(
            0,
            int(
                np.floor(
                    delta_log2 / QUARTER_OCTAVE_DELTA_LOG2 + 1.0e-12
                )
            ),
        )
        if new_quarter_bins < MINIMUM_NEW_QUARTER_OCTAVE_BINS:
            failed.append("fewer_than_two_new_contiguous_quarter_octave_bins")
        if ratio < MINIMUM_K_EFF_RATIO:
            failed.append("k_eff_ratio_below_sqrt_2")

    if not lower_95 > 0.0:
        failed.append("bootstrap_95_percent_lower_bound_not_positive")

    return MaterialExtensionResult(
        go=not failed,
        baseline=baseline,
        candidate=candidate,
        new_contiguous_quarter_octave_bins=new_quarter_bins,
        k_eff_ratio=ratio,
        bootstrap_delta_log2_lower_95=lower_95,
        failed_requirements=tuple(failed),
    )


def material_field_extensions(
    k: Iterable[float],
    baseline_density_delta_pass: Iterable[bool],
    baseline_velocity_divergence_theta_pass: Iterable[bool],
    candidate_density_delta_pass: Iterable[bool],
    candidate_velocity_divergence_theta_pass: Iterable[bool],
    bootstrap_density_delta_log2: Iterable[float],
    bootstrap_theta_delta_log2: Iterable[float],
    bootstrap_joint_delta_log2: Iterable[float],
) -> FieldExtensionResult:
    """Evaluate density, velocity, and coupled material extensions separately.

    Density is the primary density-method decision, theta is the independent
    velocity-method decision, and the logical-AND joint result is a coupled
    diagnostic.  No decision is copied from one field to another.
    """

    k_array = _validated_k(k)
    baseline_density = _boolean_1d(
        "baseline_density_delta_pass", baseline_density_delta_pass
    )
    baseline_velocity = _boolean_1d(
        "baseline_velocity_divergence_theta_pass",
        baseline_velocity_divergence_theta_pass,
    )
    candidate_density = _boolean_1d(
        "candidate_density_delta_pass", candidate_density_delta_pass
    )
    candidate_velocity = _boolean_1d(
        "candidate_velocity_divergence_theta_pass",
        candidate_velocity_divergence_theta_pass,
    )
    _require_same_shape(
        "k",
        k_array,
        baseline_density_delta_pass=baseline_density,
        baseline_velocity_divergence_theta_pass=baseline_velocity,
        candidate_density_delta_pass=candidate_density,
        candidate_velocity_divergence_theta_pass=candidate_velocity,
    )

    baseline_fields = evaluate_field_frontiers(
        k_array, baseline_density, baseline_velocity
    )
    candidate_fields = evaluate_field_frontiers(
        k_array, candidate_density, candidate_velocity
    )
    density_material = material_frontier_extension(
        k_array,
        baseline_density,
        candidate_density,
        bootstrap_density_delta_log2,
    )
    theta_material = material_frontier_extension(
        k_array,
        baseline_velocity,
        candidate_velocity,
        bootstrap_theta_delta_log2,
    )
    joint_material = material_frontier_extension(
        k_array,
        baseline_density & baseline_velocity,
        candidate_density & candidate_velocity,
        bootstrap_joint_delta_log2,
    )
    return FieldExtensionResult(
        baseline_fields=baseline_fields,
        candidate_fields=candidate_fields,
        density_material_extension=density_material,
        theta_material_extension=theta_material,
        joint_material_extension=joint_material,
    )


__all__ = [
    "CORRELATION_MIN",
    "FieldFrontierResult",
    "FieldExtensionResult",
    "FrontierResult",
    "MaterialExtensionResult",
    "MINIMUM_K_EFF_RATIO",
    "MINIMUM_NEW_QUARTER_OCTAVE_BINS",
    "OBSERVATION_CONSTRAINED",
    "PRIOR_DOMINATED",
    "RESIDUAL_POWER_RATIO_MAX",
    "RESPONSE_MAX",
    "RESPONSE_MIN",
    "STRUCTURE_CONDITIONED",
    "classify_bins",
    "contiguous_frontier",
    "evaluate_field_frontiers",
    "material_frontier_extension",
    "material_field_extensions",
    "strict_gate_mask",
]
