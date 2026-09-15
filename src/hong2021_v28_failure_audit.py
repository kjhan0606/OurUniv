#!/usr/bin/env python
"""Pure physical-tail summaries for the frozen V28 failure audit."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


AUDIT_PROGRAM_SHA256 = "d74d561a9d3772f525882807bf3aaeb6cb111e24496082a638fa6a125a5a55cb"
QUANTILE = 0.99999
TOP_VALUES_KEPT = 10_000


@dataclass
class PhysicalTailAccumulator:
    """Stream exact required moments while retaining enough values for q99.999."""

    count: int = 0
    delta_square_sum: float = 0.0
    maximum: float = -math.inf
    top: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def update(self, y: np.ndarray) -> None:
        log10rho = 4.5 * np.asarray(y, dtype=np.float64).reshape(-1)
        if not np.isfinite(log10rho).all():
            raise ValueError("physical tail audit received nonfinite density")
        self.count += len(log10rho)
        self.maximum = max(self.maximum, float(log10rho.max()))
        delta = np.power(10.0, log10rho) - 1.0
        self.delta_square_sum += float(np.square(delta).sum())
        combined = np.concatenate((self.top, log10rho))
        if len(combined) > TOP_VALUES_KEPT:
            combined = np.partition(combined, len(combined) - TOP_VALUES_KEPT)[
                -TOP_VALUES_KEPT:
            ]
        self.top = combined

    def report(self) -> dict[str, float | int]:
        if self.count <= 0:
            raise ValueError("physical tail accumulator is empty")
        ordered = np.sort(self.top)
        start = self.count - len(ordered)
        position = (self.count - 1) * QUANTILE
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low < start:
            raise RuntimeError("physical tail accumulator retained too few values")
        fraction = position - low
        quantile = (1.0 - fraction) * ordered[low - start] + fraction * ordered[
            high - start
        ]
        return {
            "voxels": self.count,
            "q99_999_log10rho": float(quantile),
            "maximum_log10rho": self.maximum,
            "mean_delta_squared": self.delta_square_sum / self.count,
        }


def compare_physical_tails(
    reference: Mapping[str, float | int], candidate: Mapping[str, float | int]
) -> dict[str, Any]:
    q_error = float(candidate["q99_999_log10rho"]) - float(
        reference["q99_999_log10rho"]
    )
    maximum_excess = float(candidate["maximum_log10rho"]) - float(
        reference["maximum_log10rho"]
    )
    variance_ratio = float(candidate["mean_delta_squared"]) / float(
        reference["mean_delta_squared"]
    )
    checks = {
        "absolute_q99_999_error_at_most_0.1_dex": abs(q_error) <= 0.1,
        "maximum_excess_at_most_0.3_dex": maximum_excess <= 0.3,
        "mean_delta_squared_ratio_at_most_1.5": variance_ratio <= 1.5,
    }
    return {
        "delta_q99_999_dex": q_error,
        "candidate_maximum_above_reference_dex": maximum_excess,
        "candidate_over_reference_mean_delta_squared": variance_ratio,
        "checks": checks,
        "pass": all(checks.values()),
    }


def classify(domains: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """Apply the mechanism order frozen before inspecting the audit output."""
    if not all(row["self_reconstruction_vs_truth"]["pass"] for row in domains.values()):
        return {
            "class": "V21_V14_representation_self_reconstruction_is_not_tail_preserving",
            "next": "repair_and_validate_lossless_physical_density_coordinate",
        }
    if not all(row["cross_generated_vs_selected_donor_truths"]["pass"] for row in domains.values()):
        return {
            "class": "cross_condition_inverse_inflates_intact_train_residuals",
            "next": "replace_latent_transplantation_with_condition_consistent_physical_residual_representation",
        }
    if not all(row["selected_donor_truths_vs_validation_truth"]["pass"] for row in domains.values()):
        return {
            "class": "target_free_condition_matching_selects_the_wrong_physical_population",
            "next": "redesign_local_observation_descriptor_and_deterministic_backbone",
        }
    if not all(row["two_point_improves_deterministic_all_scales"] for row in domains.values()):
        return {
            "class": "deterministic_backbone_or_local_phase_alignment_limits_2PCF",
            "next": "audit_and_replace_deterministic_current_density_backbone",
        }
    return {
        "class": "frozen_V28_failure_mechanisms_not_reproduced",
        "next": "stop_and_reconcile_with_V28_development_decision",
    }


__all__ = [
    "AUDIT_PROGRAM_SHA256",
    "PhysicalTailAccumulator",
    "classify",
    "compare_physical_tails",
]
