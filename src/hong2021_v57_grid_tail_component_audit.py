#!/usr/bin/env python
"""Train-only V56 grid-tail and ranked-mixture-component decomposition."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v46_tail_occupancy_audit import EXPECTED_OBJECTS, PROBE_VOXELS, _probe_indices
from hong2021_v50_network import mixture_parameters
from hong2021_v51_bounded_support_audit import (
    LOWER_SUPPORT,
    SUPPORT_RANGE,
    _physical_delta_squared,
    _relative_difference,
    _truth_probe,
)
from hong2021_v54_train import CONTROL_QUADRATURE_ORDER, PRIMARY_QUADRATURE_ORDER
from hong2021_v56_train import load_cache, load_program as load_v56_program
from hong2021_v56_train_gate import _load_fit


PROGRAM_SHA256 = "a8d7e63dd1c2f95c642962bf0c755581172bb9a99a1e26c17ced061a3dbe187c"
PROGRAM_SCHEMA = "hong2021-v57-grid-tail-component-audit-program-v1"
SCHEMA = "hong2021-v57-grid-tail-component-audit-v1"
COMPONENTS = 5


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V57 {label} hash differs")
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _path(repo: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (repo / candidate).resolve()


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)


def _bin_indices(log10rho: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Implement strict exceedance: equality remains in the lower bin."""
    return np.searchsorted(thresholds, log10rho, side="left")


def _bin_labels(cells: int) -> list[str]:
    if cells < 1:
        raise ValueError("V57 grid requires at least one cell")
    labels = ["at_or_below_q99_999", "q99_999_to_grid_01"]
    labels.extend(f"grid_{index:02d}_to_grid_{index + 1:02d}" for index in range(1, cells))
    labels.append("above_global_train_maximum")
    return labels


def _threshold_labels(cells: int) -> list[str]:
    return ["q99_999_anchor", *(f"grid_{index:02d}" for index in range(1, cells + 1))]


def _rank_components(
    weights: torch.Tensor, locations: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        weights.ndim != 2
        or weights.shape != locations.shape
        or weights.shape != scales.shape
        or weights.shape[0] != COMPONENTS
    ):
        raise ValueError("V57 component shape differs")
    order = torch.argsort(locations, dim=0)
    return (
        torch.gather(weights, 0, order),
        torch.gather(locations, 0, order),
        torch.gather(scales, 0, order),
    )


def _component_survival(
    weights: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
    base: torch.Tensor,
    target_std: float,
    thresholds: torch.Tensor,
) -> np.ndarray:
    if base.ndim != 1 or base.shape[0] != weights.shape[1] or target_std <= 0.0:
        raise ValueError("V57 survival input differs")
    standardized = (thresholds[:, None].double() / 4.5 - base.double()[None]) / float(
        target_std
    )
    result = torch.zeros(
        (len(thresholds), COMPONENTS, len(base)),
        dtype=torch.float64,
        device=weights.device,
    )
    below = standardized <= LOWER_SUPPORT
    above = standardized >= LOWER_SUPPORT + SUPPORT_RANGE
    for threshold_index in range(len(thresholds)):
        lower_mask = below[threshold_index]
        if bool(lower_mask.any()):
            result[threshold_index, :, lower_mask] = weights[:, lower_mask].double()
        interior = ~(lower_mask | above[threshold_index])
        if bool(interior.any()):
            coordinate = (
                standardized[threshold_index, interior] - LOWER_SUPPORT
            ) / SUPPORT_RANGE
            latent = torch.log(coordinate) - torch.log1p(-coordinate)
            survival = 1.0 - torch.special.ndtr(
                (latent[None] - locations[:, interior].double())
                / scales[:, interior].double()
            )
            result[threshold_index, :, interior] = (
                weights[:, interior].double() * survival
            )
    return torch.clamp(result, 0.0, 1.0).cpu().numpy()


def _quadrature_bins(
    weights: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
    base: torch.Tensor,
    target_std: float,
    nodes: torch.Tensor,
    quadrature_weights: torch.Tensor,
    thresholds: torch.Tensor,
) -> dict[str, np.ndarray]:
    if base.ndim != 1 or base.shape[0] != weights.shape[1]:
        raise ValueError("V57 quadrature input differs")
    normalized = quadrature_weights.double() / math.sqrt(math.pi)
    bins = len(thresholds) + 1
    moment_bins = torch.zeros(
        (COMPONENTS, bins), dtype=torch.float64, device=weights.device
    )
    probability_bins = torch.zeros_like(moment_bins)
    component_totals = torch.zeros(COMPONENTS, dtype=torch.float64, device=weights.device)
    for component in range(COMPONENTS):
        latent = locations[component].double()[:, None] + math.sqrt(2.0) * (
            scales[component].double()[:, None] * nodes.double()[None]
        )
        standardized = LOWER_SUPPORT + SUPPORT_RANGE * torch.sigmoid(latent)
        physical_y = base.double()[:, None] + float(target_std) * standardized
        log10rho = 4.5 * physical_y
        rho = torch.exp(math.log(10.0) * log10rho)
        delta_squared = torch.square(rho - 1.0)
        mass = weights[component].double()[:, None] * normalized[None]
        component_totals[component] = torch.sum(mass * delta_squared)
        assignments = torch.bucketize(log10rho, thresholds.double(), right=False)
        for bin_index in range(bins):
            mask = assignments == bin_index
            probability_bins[component, bin_index] = torch.sum(mass * mask)
            moment_bins[component, bin_index] = torch.sum(mass * delta_squared * mask)
    return {
        "component_moment_bins": moment_bins.cpu().numpy(),
        "component_probability_bins": probability_bins.cpu().numpy(),
        "component_total_moments": component_totals.cpu().numpy(),
    }


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_audit_implementation_or_execution"
    ):
        raise ValueError("V57 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _path(repo, parent["v56_record"]), parent["v56_record_sha256"], "V56 record"
    )
    decision = record.get("train_only_mechanism_decision", {})
    firewall = record.get("firewall", {})
    observed_pass = {
        domain: record.get("train_only_physical_moment_ratios", {})
        .get(domain, {})
        .get("top_backbone_pass")
        for domain in DOMAIN_ORDER
    }
    if (
        record.get("status") != parent["required_status"]
        or decision.get("classification") != parent["required_classification"]
        or decision.get("next") != parent["required_next"]
        or observed_pass != parent["required_domain_top_backbone_pass"]
        or firewall.get("development_accessed")
        is not parent["required_development_accessed"]
        or firewall.get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
        or firewall.get("Astrid_accessed") is not False
        or firewall.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V57 parent conclusion or firewall differs")
    frozen = program["frozen_inputs"]
    for key in (
        "v56_program",
        "v56_checkpoint",
        "v56_training_report",
        "v56_grid",
        "v56_preflight",
        "v56_train_gate",
        "v54_threshold_selection",
        "conditioning_cache",
        "support_selection",
    ):
        if sha256_file(_path(repo, frozen[key])) != frozen[f"{key}_sha256"]:
            raise ValueError(f"V57 frozen input differs: {key}")
    report = _verified_json(
        _path(repo, frozen["v56_training_report"]),
        frozen["v56_training_report_sha256"],
        "V56 report",
    )
    grid = _verified_json(
        _path(repo, frozen["v56_grid"]), frozen["v56_grid_sha256"], "V56 grid"
    )
    gate = _verified_json(
        _path(repo, frozen["v56_train_gate"]),
        frozen["v56_train_gate_sha256"],
        "V56 gate",
    )
    if (
        canonical_digest(report) != frozen["v56_training_report_decision_digest_sha256"]
        or canonical_digest(grid) != frozen["v56_grid_decision_digest_sha256"]
        or canonical_digest(gate) != frozen["v56_train_gate_decision_digest_sha256"]
        or gate.get("train_mechanism_pass") is not False
        or gate.get("development_accessed") is not False
        or gate.get("independent_gate_locked") is not True
    ):
        raise ValueError("V57 V56 report, grid, or gate digest differs")
    partition = program["fixed_threshold_partition"]
    selected = _verified_json(
        _path(repo, frozen["v54_threshold_selection"]),
        frozen["v54_threshold_selection_sha256"],
        "V54 thresholds",
    )
    anchor = float(partition["lower_anchor_log10rho"])
    grid_thresholds = np.asarray(
        partition["scored_grid_thresholds_log10rho"], dtype=np.float64
    )
    grid_weights = np.asarray(
        partition["scored_grid_physical_moment_weights"], dtype=np.float64
    )
    if (
        anchor != float(selected["common_log10rho_thresholds"][-1])
        or not np.array_equal(grid_thresholds, np.asarray(grid["thresholds_log10rho"]))
        or not np.array_equal(grid_weights, np.asarray(grid["physical_moment_weights"]))
        or not np.all(np.diff(np.concatenate(([anchor], grid_thresholds))) > 0.0)
        or len(grid_thresholds) != 16
        or len(_bin_labels(len(grid_thresholds))) != 18
    ):
        raise ValueError("V57 fixed threshold partition differs")
    return program, gate


def classify(numerical_requirements_pass: bool, tng: dict[str, Any]) -> tuple[str, str]:
    if not numerical_requirements_pass:
        return (
            "V56_grid_tail_component_decomposition_is_numerically_unresolved",
            "freeze_a_higher_accuracy_train_only_grid_tail_audit_without_training_or_development_access",
        )
    regions = tng["regions"]
    if regions["beyond_grid"]["positive_excess_share"] >= 0.5:
        return (
            "V56_TNG_moment_excess_lies_beyond_scored_global_train_maximum",
            "freeze_one_matched_train_only_model_that_extends_the_proper_survival_grid_over_the_immutable_reachable_output_support_without_changing_other_model_or_training_choices",
        )
    if regions["below_grid"]["positive_excess_share"] >= 0.5:
        return (
            "V56_TNG_moment_excess_lies_below_the_upper_survival_grid",
            "freeze_one_matched_train_only_model_that_extends_the_proper_survival_grid_downward_to_the_immutable_q99_9_output_threshold_without_changing_other_model_or_training_choices",
        )
    summary = tng["supported_grid_error_summary"]
    if regions["inside_grid"]["positive_excess_share"] >= 0.5 and summary["available"]:
        if (
            summary["weighted_mean_absolute_log_probability_ratio"]
            > summary["weighted_mean_absolute_log_conditional_amplitude_ratio"]
        ):
            return (
                "V56_TNG_scored_grid_survival_probabilities_remain_miscalibrated",
                "freeze_one_matched_train_only_model_that_changes_only_the_predeclared_upper_survival_score_coefficient",
            )
        return (
            "V56_TNG_scored_grid_is_too_coarse_for_conditional_tail_amplitude",
            "freeze_one_matched_train_only_model_that_changes_only_the_predeclared_upper_survival_grid_resolution",
        )
    return (
        "V56_TNG_remaining_moment_excess_is_mixed_across_grid_regions",
        "seal_the_domainwise_grid_and_component_decomposition_before_selecting_any_further_model",
    )


def _domain_summary(
    truth_log10rho: np.ndarray,
    truth_delta_squared: np.ndarray,
    predicted_component_survival: np.ndarray,
    primary_component_moment_bins: np.ndarray,
    primary_component_probability_bins: np.ndarray,
    primary_component_totals: np.ndarray,
    control_component_moment_bins: np.ndarray,
    control_component_totals: np.ndarray,
    component_mass_sums: np.ndarray,
    thresholds: np.ndarray,
    grid_weights: np.ndarray,
    sealed: dict[str, Any],
    numerics: dict[str, Any],
) -> dict[str, Any]:
    count = len(truth_log10rho)
    threshold_count = len(thresholds)
    bin_count = threshold_count + 1
    labels = _bin_labels(threshold_count - 1)
    threshold_labels = _threshold_labels(threshold_count - 1)
    expected_survival_shape = (threshold_count, COMPONENTS, count)
    expected_bin_shape = (COMPONENTS, bin_count)
    if (
        predicted_component_survival.shape != expected_survival_shape
        or primary_component_moment_bins.shape != expected_bin_shape
        or primary_component_probability_bins.shape != expected_bin_shape
        or control_component_moment_bins.shape != expected_bin_shape
        or primary_component_totals.shape != (COMPONENTS,)
        or control_component_totals.shape != (COMPONENTS,)
        or component_mass_sums.shape != (COMPONENTS,)
        or len(labels) != bin_count
        or grid_weights.shape != (threshold_count - 1,)
    ):
        raise ValueError("V57 domain accumulator shape differs")

    truth_bins = _bin_indices(truth_log10rho, thresholds)
    truth_counts = np.bincount(truth_bins, minlength=bin_count).astype(np.int64)
    truth_moment_sums = np.bincount(
        truth_bins, weights=truth_delta_squared, minlength=bin_count
    ).astype(np.float64)
    primary_moment_bins = primary_component_moment_bins.sum(axis=0)
    primary_probability_bins = primary_component_probability_bins.sum(axis=0)
    control_moment_bins = control_component_moment_bins.sum(axis=0)
    primary_total = float(primary_component_totals.sum(dtype=np.float64))
    control_total = float(control_component_totals.sum(dtype=np.float64))
    truth_total = float(truth_moment_sums.sum(dtype=np.float64))
    truth_mean = truth_total / count
    predicted_mean = primary_total / count
    sealed_top = sealed["strata"]["q99_9_and_above"]
    reproduction = {
        "truth_relative_difference_from_V56_gate": _relative_error(
            truth_mean, float(sealed_top["truth_mean_delta_squared"])
        ),
        "predicted_relative_difference_from_V56_gate": _relative_error(
            predicted_mean, float(sealed_top["V56_quadrature_mean_delta_squared"])
        ),
    }

    truth_bin_means = truth_moment_sums / count
    primary_bin_means = primary_moment_bins / count
    control_bin_means = control_moment_bins / count
    positive_bin_excess = np.maximum(primary_bin_means - truth_bin_means, 0.0)
    positive_excess_sum = float(positive_bin_excess.sum(dtype=np.float64))
    positive_bin_shares = (
        positive_bin_excess / positive_excess_sum
        if positive_excess_sum > 0.0
        else np.zeros(bin_count, dtype=np.float64)
    )
    bins: dict[str, Any] = {}
    for bin_index, label in enumerate(labels):
        component_moments = primary_component_moment_bins[:, bin_index] / count
        component_probabilities = primary_component_probability_bins[:, bin_index] / count
        component_total = float(component_moments.sum(dtype=np.float64))
        bins[label] = {
            "truth_count": int(truth_counts[bin_index]),
            "truth_probability": float(truth_counts[bin_index] / count),
            "truth_mean_delta_squared_contribution": float(truth_bin_means[bin_index]),
            "predicted_quadrature_probability_64": float(
                primary_probability_bins[bin_index] / count
            ),
            "predicted_mean_delta_squared_contribution_64": float(
                primary_bin_means[bin_index]
            ),
            "predicted_mean_delta_squared_contribution_32": float(
                control_bin_means[bin_index]
            ),
            "positive_excess_share": float(positive_bin_shares[bin_index]),
            "ranked_component_probability_contributions_64": component_probabilities.tolist(),
            "ranked_component_moment_contributions_64": component_moments.tolist(),
            "ranked_component_moment_shares_64": (
                component_moments / component_total
                if component_total > 0.0
                else np.zeros(COMPONENTS, dtype=np.float64)
            ).tolist(),
        }

    regions: dict[str, Any] = {}
    region_indices = {
        "below_grid": np.asarray([0], dtype=np.int64),
        "inside_grid": np.arange(1, threshold_count, dtype=np.int64),
        "beyond_grid": np.asarray([threshold_count], dtype=np.int64),
    }
    for label, indices in region_indices.items():
        truth_region = float(truth_bin_means[indices].sum(dtype=np.float64))
        predicted_region = float(primary_bin_means[indices].sum(dtype=np.float64))
        component_region = primary_component_moment_bins[:, indices].sum(axis=1) / count
        component_total = float(component_region.sum(dtype=np.float64))
        regions[label] = {
            "truth_mean_delta_squared_contribution": truth_region,
            "predicted_mean_delta_squared_contribution_64": predicted_region,
            "predicted_over_truth": (
                predicted_region / truth_region if truth_region > 0.0 else None
            ),
            "positive_excess_share": (
                float(positive_bin_excess[indices].sum(dtype=np.float64) / positive_excess_sum)
                if positive_excess_sum > 0.0
                else 0.0
            ),
            "ranked_component_moment_contributions_64": component_region.tolist(),
            "ranked_component_moment_shares_64": (
                component_region / component_total
                if component_total > 0.0
                else np.zeros(COMPONENTS, dtype=np.float64)
            ).tolist(),
        }

    threshold_rows: dict[str, Any] = {}
    tail_convergence_pass = True
    identity_pass = True
    analytic_component_partition_error = 0.0
    supported_grid_indices: list[int] = []
    for threshold_index, label in enumerate(threshold_labels):
        tail_slice = slice(threshold_index + 1, None)
        truth_tail_count = int(truth_counts[tail_slice].sum())
        truth_probability = truth_tail_count / count
        analytic_components = np.mean(
            predicted_component_survival[threshold_index], axis=1, dtype=np.float64
        )
        analytic_probability = float(analytic_components.sum(dtype=np.float64))
        quadrature_probability_components = (
            primary_component_probability_bins[:, tail_slice].sum(axis=1) / count
        )
        quadrature_probability = float(
            quadrature_probability_components.sum(dtype=np.float64)
        )
        primary_tail_components = (
            primary_component_moment_bins[:, tail_slice].sum(axis=1) / count
        )
        control_tail_components = (
            control_component_moment_bins[:, tail_slice].sum(axis=1) / count
        )
        primary_tail_moment = float(primary_tail_components.sum(dtype=np.float64))
        control_tail_moment = float(control_tail_components.sum(dtype=np.float64))
        truth_tail_moment = float(truth_moment_sums[tail_slice].sum() / count)
        convergence = _relative_difference(primary_tail_moment, control_tail_moment)
        tail_convergence_pass = bool(
            tail_convergence_pass
            and convergence
            <= float(
                numerics[
                    "maximum_32_to_64_tail_moment_relative_difference_for_classification"
                ]
            )
        )
        empirical_supported = truth_tail_count >= int(
            numerics["minimum_empirical_exceedance_count_for_threshold_classification"]
        )
        ratio_available = bool(
            empirical_supported
            and min(
                truth_probability,
                analytic_probability,
                truth_tail_moment,
                primary_tail_moment,
            )
            > 0.0
        )
        if ratio_available:
            truth_conditional = truth_tail_moment / truth_probability
            predicted_conditional = primary_tail_moment / analytic_probability
            probability_ratio = analytic_probability / truth_probability
            conditional_ratio = predicted_conditional / truth_conditional
            tail_ratio = primary_tail_moment / truth_tail_moment
            log_probability = math.log(probability_ratio)
            log_conditional = math.log(conditional_ratio)
            log_tail = math.log(tail_ratio)
            identity_error = abs(log_tail - log_probability - log_conditional)
            identity_pass = bool(
                identity_pass
                and identity_error
                <= float(numerics["maximum_log_ratio_identity_absolute_error"])
            )
            if threshold_index > 0:
                supported_grid_indices.append(threshold_index)
        else:
            truth_conditional = predicted_conditional = None
            probability_ratio = conditional_ratio = tail_ratio = None
            log_probability = log_conditional = log_tail = None
            identity_error = None
            if empirical_supported:
                identity_pass = False
        tail_component_total = float(primary_tail_components.sum(dtype=np.float64))
        probability_component_total = float(analytic_components.sum(dtype=np.float64))
        analytic_component_partition_error = max(
            analytic_component_partition_error,
            _relative_error(analytic_probability, probability_component_total),
        )
        moment_shares = (
            primary_tail_components / tail_component_total
            if tail_component_total > 0.0
            else np.zeros(COMPONENTS, dtype=np.float64)
        )
        probability_shares = (
            analytic_components / probability_component_total
            if probability_component_total > 0.0
            else np.zeros(COMPONENTS, dtype=np.float64)
        )
        dominant_rank = int(np.argmax(moment_shares))
        threshold_rows[label] = {
            "log10rho_threshold": float(thresholds[threshold_index]),
            "truth_exceedance_count": truth_tail_count,
            "truth_exceedance_probability": float(truth_probability),
            "predicted_analytic_exceedance_probability": analytic_probability,
            "predicted_quadrature_exceedance_probability_64": quadrature_probability,
            "analytic_to_quadrature_probability_relative_difference": _relative_difference(
                analytic_probability, quadrature_probability
            ),
            "truth_mean_delta_squared_tail_contribution": truth_tail_moment,
            "predicted_mean_delta_squared_tail_contribution_64": primary_tail_moment,
            "predicted_mean_delta_squared_tail_contribution_32": control_tail_moment,
            "tail_moment_32_to_64_relative_difference": convergence,
            "truth_conditional_mean_delta_squared": truth_conditional,
            "predicted_conditional_mean_delta_squared": predicted_conditional,
            "predicted_over_truth_probability": probability_ratio,
            "predicted_over_truth_conditional_amplitude": conditional_ratio,
            "predicted_over_truth_tail_moment": tail_ratio,
            "log_probability_ratio": log_probability,
            "log_conditional_amplitude_ratio": log_conditional,
            "log_tail_moment_ratio": log_tail,
            "log_ratio_identity_absolute_error": identity_error,
            "empirical_support_pass": empirical_supported,
            "ratio_available": ratio_available,
            "ranked_component_analytic_probability_contributions": analytic_components.tolist(),
            "ranked_component_analytic_probability_shares": probability_shares.tolist(),
            "ranked_component_quadrature_probability_contributions_64": quadrature_probability_components.tolist(),
            "ranked_component_moment_contributions_64": primary_tail_components.tolist(),
            "ranked_component_moment_contributions_32": control_tail_components.tolist(),
            "ranked_component_moment_shares_64": moment_shares.tolist(),
            "dominant_location_rank": dominant_rank,
            "single_component_tail_moment_dominates": bool(moment_shares[dominant_rank] >= 0.5),
        }

    if supported_grid_indices:
        local_indices = np.asarray([index - 1 for index in supported_grid_indices])
        supported_weights = grid_weights[local_indices]
        normalized_weights = supported_weights / supported_weights.sum(dtype=np.float64)
        probability_errors = np.asarray(
            [
                abs(float(threshold_rows[threshold_labels[index]]["log_probability_ratio"]))
                for index in supported_grid_indices
            ]
        )
        amplitude_errors = np.asarray(
            [
                abs(
                    float(
                        threshold_rows[threshold_labels[index]][
                            "log_conditional_amplitude_ratio"
                        ]
                    )
                )
                for index in supported_grid_indices
            ]
        )
        supported_summary = {
            "available": True,
            "supported_grid_labels": [threshold_labels[index] for index in supported_grid_indices],
            "renormalized_physical_moment_weights": normalized_weights.tolist(),
            "weighted_mean_absolute_log_probability_ratio": float(
                np.sum(normalized_weights * probability_errors, dtype=np.float64)
            ),
            "weighted_mean_absolute_log_conditional_amplitude_ratio": float(
                np.sum(normalized_weights * amplitude_errors, dtype=np.float64)
            ),
        }
    else:
        supported_summary = {
            "available": False,
            "supported_grid_labels": [],
            "renormalized_physical_moment_weights": [],
            "weighted_mean_absolute_log_probability_ratio": None,
            "weighted_mean_absolute_log_conditional_amplitude_ratio": None,
        }

    complete_convergence = _relative_difference(primary_total, control_total)
    bin_partition_error = max(
        _relative_error(float(primary_moment_bins.sum()), primary_total),
        _relative_error(float(control_moment_bins.sum()), control_total),
        _relative_error(float(truth_moment_sums.sum()), truth_total),
    )
    component_partition_error = max(
        _relative_error(float(primary_component_moment_bins.sum()), primary_total),
        _relative_error(float(control_component_moment_bins.sum()), control_total),
        analytic_component_partition_error,
    )
    reproduction_pass = max(reproduction.values()) <= float(
        numerics["maximum_complete_moment_relative_difference_from_v56_gate"]
    )
    anchor_supported = bool(threshold_rows["q99_999_anchor"]["ratio_available"])
    numerical_pass = bool(
        reproduction_pass
        and anchor_supported
        and bin_partition_error <= float(numerics["maximum_bin_partition_relative_error"])
        and component_partition_error
        <= float(numerics["maximum_component_partition_relative_error"])
        and complete_convergence
        <= float(numerics["maximum_32_to_64_complete_moment_relative_difference"])
        and tail_convergence_pass
        and identity_pass
    )
    return {
        "top_backbone_probe_voxels": count,
        "complete_moment": {
            "truth_mean_delta_squared": truth_mean,
            "predicted_mean_delta_squared_64": predicted_mean,
            "predicted_mean_delta_squared_32": control_total / count,
            "predicted_over_truth": predicted_mean / truth_mean,
            "quadrature_32_to_64_relative_difference": complete_convergence,
            **reproduction,
        },
        "ranked_component_mean_mixture_mass": (component_mass_sums / count).tolist(),
        "fixed_output_bins": bins,
        "regions": regions,
        "threshold_decomposition": threshold_rows,
        "supported_grid_error_summary": supported_summary,
        "positive_excess_sum": positive_excess_sum,
        "bin_partition_relative_error": bin_partition_error,
        "component_partition_relative_error": component_partition_error,
        "tail_quadrature_convergence_pass": tail_convergence_pass,
        "log_ratio_identity_pass": identity_pass,
        "anchor_empirical_support_pass": anchor_supported,
        "reproduces_V56_gate": reproduction_pass,
        "numerical_requirements_pass": numerical_pass,
    }


@torch.inference_mode()
def _domain(
    model: torch.nn.Module,
    device: torch.device,
    v35: dict[str, Any],
    prepared: h5py.File,
    gate: dict[str, Any],
    domain: str,
    domain_index: int,
    thresholds: np.ndarray,
    grid_weights: np.ndarray,
    numerics: dict[str, Any],
) -> dict[str, Any]:
    row = v35["development_domains"][domain]
    objects = int(row["train_objects"])
    if objects != EXPECTED_OBJECTS[domain]:
        raise RuntimeError("V57 train object count differs")
    truth_probe = _truth_probe(v35, prepared, domain, domain_index)
    boundary = float(gate["domains"][domain]["backbone_boundaries"][2])
    top = truth_probe["backbone_base"].astype(np.float64) >= boundary
    expected_top = int(gate["domains"][domain]["strata"]["q99_9_and_above"]["count"])
    if int(top.sum()) != expected_top:
        raise ValueError("V57 top-backbone probe differs")
    truth_y = truth_probe["physical_y"][top]
    truth_log10rho = 4.5 * truth_y.astype(np.float64)
    truth_delta_squared = _physical_delta_squared(truth_y)
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    nodes64, weights64 = np.polynomial.hermite.hermgauss(PRIMARY_QUADRATURE_ORDER)
    nodes32, weights32 = np.polynomial.hermite.hermgauss(CONTROL_QUADRATURE_ORDER)
    nodes64_t = torch.from_numpy(nodes64).to(device)
    weights64_t = torch.from_numpy(weights64).to(device)
    nodes32_t = torch.from_numpy(nodes32).to(device)
    weights32_t = torch.from_numpy(weights32).to(device)
    thresholds_t = torch.from_numpy(thresholds).to(device)
    bin_count = len(thresholds) + 1
    survival_parts: list[np.ndarray] = []
    primary_component_moment_bins = np.zeros((COMPONENTS, bin_count), dtype=np.float64)
    primary_component_probability_bins = np.zeros_like(primary_component_moment_bins)
    control_component_moment_bins = np.zeros_like(primary_component_moment_bins)
    primary_component_totals = np.zeros(COMPONENTS, dtype=np.float64)
    control_component_totals = np.zeros(COMPONENTS, dtype=np.float64)
    component_mass_sums = np.zeros(COMPONENTS, dtype=np.float64)
    data, cache = _open_split(row, "train")
    try:
        from hong2021_v48_train import condition_cube

        for object_index in range(objects):
            condition, _, backbone = condition_cube(
                data, cache, prepared, domain, "train", object_index
            )
            parameter = model(torch.from_numpy(condition[None]).to(device))
            indices = _probe_indices(domain_index, object_index)
            start = object_index * PROBE_VOXELS
            selected = indices[top[start : start + PROBE_VOXELS]]
            if len(selected):
                index_tensor = torch.from_numpy(selected).to(device)
                flat = (
                    parameter.reshape(1, 15, -1)
                    .index_select(2, index_tensor)
                    .reshape(1, 15, 1, 1, -1)
                )
                logits, locations, scales = mixture_parameters(flat)
                mixture_weights = torch.softmax(logits, dim=1)[0, :, 0, 0]
                locations = locations[0, :, 0, 0]
                scales = scales[0, :, 0, 0]
                mixture_weights, locations, scales = _rank_components(
                    mixture_weights, locations, scales
                )
                component_mass_sums += mixture_weights.double().sum(dim=1).cpu().numpy()
                base = torch.from_numpy(
                    backbone.reshape(-1)[selected].astype(np.float64) + target_mean
                ).to(device)
                survival_parts.append(
                    _component_survival(
                        mixture_weights,
                        locations,
                        scales,
                        base,
                        target_std,
                        thresholds_t,
                    )
                )
                primary = _quadrature_bins(
                    mixture_weights,
                    locations,
                    scales,
                    base,
                    target_std,
                    nodes64_t,
                    weights64_t,
                    thresholds_t,
                )
                control = _quadrature_bins(
                    mixture_weights,
                    locations,
                    scales,
                    base,
                    target_std,
                    nodes32_t,
                    weights32_t,
                    thresholds_t,
                )
                primary_component_moment_bins += primary["component_moment_bins"]
                primary_component_probability_bins += primary[
                    "component_probability_bins"
                ]
                primary_component_totals += primary["component_total_moments"]
                control_component_moment_bins += control["component_moment_bins"]
                control_component_totals += control["component_total_moments"]
            if (object_index + 1) % 16 == 0 or object_index + 1 == objects:
                print(f"[v57-audit] {domain} {object_index + 1}/{objects}", flush=True)
    finally:
        data.close()
        cache.close()
    predicted_component_survival = np.concatenate(survival_parts, axis=2)
    if predicted_component_survival.shape != (len(thresholds), COMPONENTS, expected_top):
        raise RuntimeError("V57 predicted survival probe differs")
    return _domain_summary(
        truth_log10rho,
        truth_delta_squared,
        predicted_component_survival,
        primary_component_moment_bins,
        primary_component_probability_bins,
        primary_component_totals,
        control_component_moment_bins,
        control_component_totals,
        component_mass_sums,
        thresholds,
        grid_weights,
        gate["domains"][domain],
        numerics,
    )


def audit(program_path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, gate = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if not clean or socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V57 audit requires clean Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V57 audit requires Ada")
    frozen = program["frozen_inputs"]
    v56_program, v35, _ = load_v56_program(_path(repo, frozen["v56_program"]), repo)
    development = Path(v56_program["output_roots"]["development"])
    if development.exists():
        raise RuntimeError("V57 refuses a pre-existing V56 development directory")
    model, checkpoint = _load_fit(
        v56_program,
        _path(repo, frozen["v56_checkpoint"]),
        frozen["v56_checkpoint_sha256"],
        _path(repo, frozen["v56_training_report"]),
        frozen["v56_training_report_sha256"],
        _path(repo, frozen["v56_grid"]),
        frozen["v56_grid_sha256"],
        frozen["v54_threshold_selection_sha256"],
        _path(repo, frozen["v56_preflight"]),
        frozen["v56_preflight_sha256"],
        frozen["conditioning_cache_sha256"],
        repo,
        commit,
    )
    model = model.to("cuda").eval()
    prepared = load_cache(
        _path(repo, frozen["conditioning_cache"]),
        frozen["conditioning_cache_sha256"],
        str(checkpoint["code_commit"]),
    )
    partition = program["fixed_threshold_partition"]
    thresholds = np.asarray(
        [
            partition["lower_anchor_log10rho"],
            *partition["scored_grid_thresholds_log10rho"],
        ],
        dtype=np.float64,
    )
    grid_weights = np.asarray(
        partition["scored_grid_physical_moment_weights"], dtype=np.float64
    )
    domains: dict[str, Any] = {}
    try:
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            domains[domain] = _domain(
                model,
                torch.device("cuda"),
                v35,
                prepared,
                gate,
                domain,
                domain_index,
                thresholds,
                grid_weights,
                program["numerics"],
            )
    finally:
        prepared.close()
    numerical_pass = all(row["numerical_requirements_pass"] for row in domains.values())
    classification, next_action = classify(numerical_pass, domains["TNG100"])
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_train_only_grid_tail_component_audit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "checkpoint_sha256": frozen["v56_checkpoint_sha256"],
        "training_report_sha256": frozen["v56_training_report_sha256"],
        "grid_sha256": frozen["v56_grid_sha256"],
        "v56_train_gate_sha256": frozen["v56_train_gate_sha256"],
        "combined_thresholds_log10rho": thresholds.tolist(),
        "domains": domains,
        "numerical_requirements_pass": numerical_pass,
        "classification": classification,
        "next": next_action,
        "training_or_refit_performed": False,
        "validation_accessed": False,
        "development_accessed": False,
        "new_development_sample_generated": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("V57 refuses an existing audit")
    result = audit(args.program, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
