#!/usr/bin/env python
"""Frozen comparison logic for the post-failure V27 latent audit."""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from hong2021_v26 import DETAIL_DIMENSIONS_COARSE_TO_FINE


AUDIT_PROGRAM_SHA256 = (
    "ef06802a8a02f87821073021c7a16fda0d178dcd89526ddb09f9955baafa924b"
)
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
ROUNDTRIP_RMS_MAX = 1.0e-3
ROUNDTRIP_OVER_TAIL_EXCESS_MAX = 1.0e-2
LOGDET_CANCELLATION_PER_DIMENSION_MAX = 1.0e-4
STORED_REPLAY_FLOAT32_EPS_MAX = 2.0


def coarse_base_z_score(row: Mapping[str, Any]) -> float:
    """Score the first two truth base-z scales against a standard normal base."""
    scales = row["truth_base_z_coarse_to_fine"][:2]
    if len(scales) != 2:
        raise ValueError("latent audit requires two coarse base-z scales")
    standard_deviations = [float(scale["standard_deviation"]) for scale in scales]
    if any(not math.isfinite(value) or value <= 0.0 for value in standard_deviations):
        raise ValueError("invalid truth base-z standard deviation")
    return float(np.mean(np.abs(np.log(standard_deviations))))


def latent_support_score(row: Mapping[str, Any], threshold: str = "5.0") -> float:
    """Score generated/truth G21 tail occupancy, with the frozen zero-count rule."""
    truth = row["truth_latent"]
    generated = row["generated_latent"]
    truth_count = int(truth["absolute_tail_counts"][threshold])
    generated_count = int(generated["absolute_tail_counts"][threshold])
    truth_total = int(truth["count"])
    generated_total = int(generated["count"])
    if truth_total <= 0 or generated_total <= 0:
        raise ValueError("latent support score requires nonempty truth and generated fields")
    if truth_count == 0 or generated_count == 0:
        truth_fraction = (truth_count + 1) / (truth_total + 1)
        generated_fraction = (generated_count + 1) / (generated_total + 1)
    else:
        truth_fraction = truth_count / truth_total
        generated_fraction = generated_count / generated_total
    return abs(math.log(generated_fraction / truth_fraction))


def compare_to_v26(
    final: Mapping[str, Mapping[str, Any]],
    v26_final: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen strict per-domain latent-calibration comparison."""
    domains: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        current = final[domain]
        prior = v26_final[domain]
        v26_coarse = coarse_base_z_score(prior)
        v27_coarse = coarse_base_z_score(current)
        v26_support = latent_support_score(prior)
        v27_support = latent_support_score(current)
        domains[domain] = {
            "coarse_base_z_score_v26": v26_coarse,
            "coarse_base_z_score_v27": v27_coarse,
            "coarse_base_z_strictly_improved": v27_coarse < v26_coarse,
            "latent_support_score_v26": v26_support,
            "latent_support_score_v27": v27_support,
            "latent_support_strictly_improved": v27_support < v26_support,
        }
        domains[domain]["both_strictly_improved"] = bool(
            domains[domain]["coarse_base_z_strictly_improved"]
            and domains[domain]["latent_support_strictly_improved"]
        )
    return {
        "domains": domains,
        "all_domains_both_strictly_improved": all(
            row["both_strictly_improved"] for row in domains.values()
        ),
        "selection_role": "predeclared_post_failure_classification",
    }


def numerical_stability(final: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the numerical thresholds frozen before this audit was executed."""
    roundtrip_max = max(
        float(row["roundtrip"]["maximum_absolute_latent_error"])
        for row in final.values()
    )
    roundtrip_rms = max(
        float(row["roundtrip"]["rms_latent_error"]) for row in final.values()
    )
    replay_max = max(
        float(row["stored_ensemble_replay"]["maximum_absolute_y_difference"])
        for row in final.values()
    )
    maximum_logdet_per_dimension = max(
        float(value) / dimension
        for row in final.values()
        for value, dimension in zip(
            row["roundtrip"][
                "maximum_absolute_logdet_cancellation_coarse_to_fine"
            ],
            DETAIL_DIMENSIONS_COARSE_TO_FINE,
            strict=True,
        )
    )
    tail_rows: dict[str, Any] = {}
    ratios = []
    for domain, row in final.items():
        domain_roundtrip_max = float(
            row["roundtrip"]["maximum_absolute_latent_error"]
        )
        truth_maximum = float(row["truth_latent"]["absolute_maximum"])
        generated_maximum = float(row["generated_latent"]["absolute_maximum"])
        excess = generated_maximum - truth_maximum
        # A causal comparison must not combine one domain's numerical error
        # with another domain's physical tail excess.
        ratio = domain_roundtrip_max / max(excess, np.finfo(float).tiny)
        ratios.append(ratio)
        tail_rows[domain] = {
            "truth_absolute_latent_maximum": truth_maximum,
            "generated_absolute_latent_maximum": generated_maximum,
            "generated_minus_truth_absolute_maximum": excess,
            "maximum_absolute_latent_roundtrip_error": domain_roundtrip_max,
            "maximum_roundtrip_error_over_generated_tail_excess": ratio,
        }
    maximum_roundtrip_over_excess = max(ratios)
    checks = {
        "roundtrip_rms": roundtrip_rms <= ROUNDTRIP_RMS_MAX,
        "roundtrip_error_relative_to_tail_excess": (
            maximum_roundtrip_over_excess <= ROUNDTRIP_OVER_TAIL_EXCESS_MAX
        ),
        "logdet_cancellation_per_dimension": (
            maximum_logdet_per_dimension <= LOGDET_CANCELLATION_PER_DIMENSION_MAX
        ),
        "stored_physical_replay": (
            replay_max
            <= STORED_REPLAY_FLOAT32_EPS_MAX * float(np.finfo(np.float32).eps)
        ),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "maximum_absolute_latent_roundtrip_error": roundtrip_max,
        "maximum_latent_roundtrip_rms": roundtrip_rms,
        "maximum_roundtrip_error_over_generated_tail_excess": (
            maximum_roundtrip_over_excess
        ),
        "maximum_logdet_cancellation_per_dimension": maximum_logdet_per_dimension,
        "maximum_stored_physical_replay_difference_y": replay_max,
        "float32_machine_epsilon": float(np.finfo(np.float32).eps),
        "latent_tail_extrema": tail_rows,
    }


def mechanism_summary(
    candidates: Mapping[str, Mapping[str, Mapping[str, Any]]],
    optimization: Mapping[str, Any],
    v26_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify the failed V27 run in the order frozen by the audit program."""
    final = candidates["30000"]
    v26_final = v26_audit["candidates"]["30000"]["domains"]
    stability = numerical_stability(final)
    comparison = compare_to_v26(final, v26_final)
    if not stability["pass"]:
        classification = "v27_trained_flow_numerical_failure"
        next_step = "repair_numerical_inversion_before_any_new_statistical_model"
    elif comparison["all_domains_both_strictly_improved"]:
        classification = (
            "parent_aligned_context_repairs_latent_calibration_but_field_"
            "morphology_still_fails"
        )
        next_step = (
            "audit_deterministic_current_density_backbone_before_changing_"
            "residual_density_family"
        )
    else:
        classification = (
            "explicit_conditional_flow_remains_latent_miscalibrated_after_phase_repair"
        )
        next_step = "freeze_and_test_train_only_empirical_joint_residual_control"
    return {
        "classification": classification,
        "next": next_step,
        "numerical_stability": stability,
        "comparison_to_v26": comparison,
        "finest_scale_objective_fraction": optimization[
            "finest_scale_objective_fraction"
        ],
        "relative_validation_nll_improvement_25000_to_30000": optimization[
            "relative_improvement_25000_to_30000"
        ],
        "posthoc_tuning": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }


__all__ = [
    "AUDIT_PROGRAM_SHA256",
    "DOMAIN_ORDER",
    "coarse_base_z_score",
    "compare_to_v26",
    "latent_support_score",
    "mechanism_summary",
    "numerical_stability",
]
