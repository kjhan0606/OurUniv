#!/usr/bin/env python
"""High-accuracy train-only re-audit of the V56 grid tail."""
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
    _quadrature_object,
    _relative_difference,
    _truth_probe,
)
from hong2021_v54_train import PRIMARY_QUADRATURE_ORDER
from hong2021_v56_train import load_cache, load_program as load_v56_program
from hong2021_v56_train_gate import _load_fit
from hong2021_v57_grid_tail_component_audit import (
    COMPONENTS,
    _bin_indices,
    _bin_labels,
    _rank_components,
    _threshold_labels,
)


PROGRAM_SHA256 = "ef14565fe5de172151dd8a4aa91c28a07ddeab990c1062b380eaa07023ab5db9"
PROGRAM_SCHEMA = "hong2021-v58-high-accuracy-grid-tail-audit-program-v1"
SCHEMA = "hong2021-v58-high-accuracy-grid-tail-audit-v1"
PRIMARY_ORDER = 128
CONTROL_ORDER = 64


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V58 {label} hash differs")
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _path(repo: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (repo / candidate).resolve()


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)


def load_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_audit_implementation_or_execution"
    ):
        raise ValueError("V58 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _path(repo, parent["v57_record"]), parent["v57_record_sha256"], "V57 record"
    )
    audit_row = record.get("audit", {})
    firewall = record.get("firewall", {})
    if (
        record.get("status") != parent["required_status"]
        or audit_row.get("classification") != parent["required_classification"]
        or audit_row.get("next") != parent["required_next"]
        or firewall.get("development_accessed")
        is not parent["required_development_accessed"]
        or firewall.get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
        or firewall.get("Astrid_accessed") is not False
        or firewall.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V58 parent conclusion or firewall differs")
    frozen = program["frozen_inputs"]
    for key in (
        "v57_program",
        "v57_audit",
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
            raise ValueError(f"V58 frozen input differs: {key}")
    v57_program = _verified_json(
        _path(repo, frozen["v57_program"]), frozen["v57_program_sha256"], "V57 program"
    )
    v57_audit = _verified_json(
        _path(repo, frozen["v57_audit"]), frozen["v57_audit_sha256"], "V57 audit"
    )
    gate = _verified_json(
        _path(repo, frozen["v56_train_gate"]),
        frozen["v56_train_gate_sha256"],
        "V56 gate",
    )
    if (
        canonical_digest(v57_audit) != frozen["v57_audit_decision_digest_sha256"]
        or v57_audit.get("classification") != parent["required_classification"]
        or v57_audit.get("numerical_requirements_pass") is not False
        or v57_audit.get("development_accessed") is not False
        or gate.get("train_mechanism_pass") is not False
        or gate.get("development_accessed") is not False
        or gate.get("independent_gate_locked") is not True
        or program["classification"]["branches"][1:]
        != v57_program["classification"]["branches"][1:]
    ):
        raise ValueError("V58 V57 audit, V56 gate, or classification binding differs")
    return program, v57_program, v57_audit, gate


def _cdf_boundaries(
    location: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    target_std: float,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    standardized = (thresholds[:, None].double() / 4.5 - base.double()[None]) / float(
        target_std
    )
    cdf = torch.empty_like(standardized)
    below = standardized <= LOWER_SUPPORT
    above = standardized >= LOWER_SUPPORT + SUPPORT_RANGE
    interior = ~(below | above)
    cdf[below] = 0.0
    cdf[above] = 1.0
    if bool(interior.any()):
        coordinate = torch.clamp(
            (standardized - LOWER_SUPPORT) / SUPPORT_RANGE,
            min=torch.finfo(torch.float64).tiny,
            max=1.0 - torch.finfo(torch.float64).eps,
        )
        latent = torch.log(coordinate) - torch.log1p(-coordinate)
        expanded_location = location.double()[None].expand_as(standardized)
        expanded_scale = scale.double()[None].expand_as(standardized)
        cdf[interior] = torch.special.ndtr(
            ((latent - expanded_location) / expanded_scale)[interior]
        )
    boundaries = torch.cat(
        [
            torch.zeros((1, len(base)), dtype=torch.float64, device=base.device),
            cdf,
            torch.ones((1, len(base)), dtype=torch.float64, device=base.device),
        ],
        dim=0,
    )
    if bool((torch.diff(boundaries, dim=0) < -8.0 * torch.finfo(torch.float64).eps).any()):
        raise RuntimeError("V58 analytic CDF boundaries are not monotone")
    return boundaries


def _cdf_interval_bins(
    weights: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
    base: torch.Tensor,
    target_std: float,
    thresholds: torch.Tensor,
    order: int,
) -> dict[str, np.ndarray]:
    if (
        weights.shape != locations.shape
        or weights.shape != scales.shape
        or weights.shape[0] != COMPONENTS
        or base.ndim != 1
        or base.shape[0] != weights.shape[1]
        or order <= 0
    ):
        raise ValueError("V58 interval quadrature input differs")
    nodes, quadrature_weights = np.polynomial.legendre.leggauss(order)
    nodes_t = torch.from_numpy(nodes).to(base.device).double()
    quadrature_weights_t = torch.from_numpy(quadrature_weights).to(base.device).double()
    bins = len(thresholds) + 1
    moment_bins = torch.zeros(
        (COMPONENTS, bins), dtype=torch.float64, device=base.device
    )
    probability_bins = torch.zeros_like(moment_bins)
    lower_endpoint = torch.nextafter(
        torch.tensor(0.0, dtype=torch.float64, device=base.device),
        torch.tensor(1.0, dtype=torch.float64, device=base.device),
    )
    upper_endpoint = torch.nextafter(
        torch.tensor(1.0, dtype=torch.float64, device=base.device),
        torch.tensor(0.0, dtype=torch.float64, device=base.device),
    )
    for component in range(COMPONENTS):
        boundaries = _cdf_boundaries(
            locations[component], scales[component], base, target_std, thresholds
        )
        for bin_index in range(bins):
            lower = boundaries[bin_index]
            upper = boundaries[bin_index + 1]
            width = torch.clamp(upper - lower, min=0.0)
            probability_bins[component, bin_index] = torch.sum(
                weights[component].double() * width
            )
            half = 0.5 * width
            coordinate = 0.5 * (upper + lower)[:, None] + half[:, None] * nodes_t[None]
            coordinate = torch.clamp(
                coordinate, min=lower_endpoint, max=upper_endpoint
            )
            normal = math.sqrt(2.0) * torch.erfinv(2.0 * coordinate - 1.0)
            latent = locations[component].double()[:, None] + (
                scales[component].double()[:, None] * normal
            )
            standardized = LOWER_SUPPORT + SUPPORT_RANGE * torch.sigmoid(latent)
            physical_y = base.double()[:, None] + float(target_std) * standardized
            rho = torch.exp(4.5 * math.log(10.0) * physical_y)
            delta_squared = torch.square(rho - 1.0)
            integral = half * torch.sum(
                quadrature_weights_t[None] * delta_squared, dim=1
            )
            moment_bins[component, bin_index] = torch.sum(
                weights[component].double() * integral
            )
    return {
        "component_probability_bins": probability_bins.cpu().numpy(),
        "component_moment_bins": moment_bins.cpu().numpy(),
        "component_total_moments": moment_bins.sum(dim=1).cpu().numpy(),
    }


def classify(numerical_requirements_pass: bool, tng: dict[str, Any]) -> tuple[str, str]:
    if not numerical_requirements_pass:
        return (
            "V58_high_accuracy_grid_tail_decomposition_is_numerically_unresolved",
            "freeze_only_the_minimal_train_only_numerical_repair_without_training_or_development_access",
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
    exact_v56_values: np.ndarray,
    primary: dict[str, np.ndarray],
    control: dict[str, np.ndarray],
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
    primary_moment_components = np.asarray(primary["component_moment_bins"])
    control_moment_components = np.asarray(control["component_moment_bins"])
    probability_components = np.asarray(primary["component_probability_bins"])
    expected = (COMPONENTS, bin_count)
    if (
        primary_moment_components.shape != expected
        or control_moment_components.shape != expected
        or probability_components.shape != expected
        or exact_v56_values.shape != (count,)
        or component_mass_sums.shape != (COMPONENTS,)
        or grid_weights.shape != (threshold_count - 1,)
    ):
        raise ValueError("V58 domain accumulator shape differs")
    truth_bins = _bin_indices(truth_log10rho, thresholds)
    truth_counts = np.bincount(truth_bins, minlength=bin_count).astype(np.int64)
    truth_moment_sums = np.bincount(
        truth_bins, weights=truth_delta_squared, minlength=bin_count
    ).astype(np.float64)
    primary_bins = primary_moment_components.sum(axis=0)
    control_bins = control_moment_components.sum(axis=0)
    probability_bins = probability_components.sum(axis=0)
    truth_total = float(truth_moment_sums.sum(dtype=np.float64))
    primary_total = float(primary_bins.sum(dtype=np.float64))
    control_total = float(control_bins.sum(dtype=np.float64))
    exact_v56_total = float(exact_v56_values.sum(dtype=np.float64))
    sealed_top = sealed["strata"]["q99_9_and_above"]
    truth_mean = truth_total / count
    exact_v56_mean = exact_v56_total / count
    primary_mean = primary_total / count
    reproduction = {
        "truth_relative_difference_from_V56_gate": _relative_error(
            truth_mean, float(sealed_top["truth_mean_delta_squared"])
        ),
        "exact_unranked_64_relative_difference_from_V56_gate": _relative_error(
            exact_v56_mean, float(sealed_top["V56_quadrature_mean_delta_squared"])
        ),
    }
    high_accuracy_vs_v56 = _relative_difference(primary_total, exact_v56_total)
    complete_convergence = _relative_difference(primary_total, control_total)

    truth_bin_means = truth_moment_sums / count
    primary_bin_means = primary_bins / count
    control_bin_means = control_bins / count
    positive_bin_excess = np.maximum(primary_bin_means - truth_bin_means, 0.0)
    positive_excess_sum = float(positive_bin_excess.sum(dtype=np.float64))
    bins: dict[str, Any] = {}
    for bin_index, label in enumerate(labels):
        component_moments = primary_moment_components[:, bin_index] / count
        component_probabilities = probability_components[:, bin_index] / count
        component_total = float(component_moments.sum(dtype=np.float64))
        bins[label] = {
            "truth_count": int(truth_counts[bin_index]),
            "truth_probability": float(truth_counts[bin_index] / count),
            "truth_mean_delta_squared_contribution": float(truth_bin_means[bin_index]),
            "predicted_analytic_probability": float(probability_bins[bin_index] / count),
            "predicted_mean_delta_squared_contribution_128": float(primary_bin_means[bin_index]),
            "predicted_mean_delta_squared_contribution_64": float(control_bin_means[bin_index]),
            "positive_excess_share": (
                float(positive_bin_excess[bin_index] / positive_excess_sum)
                if positive_excess_sum > 0.0
                else 0.0
            ),
            "ranked_component_probability_contributions": component_probabilities.tolist(),
            "ranked_component_moment_contributions_128": component_moments.tolist(),
            "ranked_component_moment_shares_128": (
                component_moments / component_total
                if component_total > 0.0
                else np.zeros(COMPONENTS, dtype=np.float64)
            ).tolist(),
        }

    region_indices = {
        "below_grid": np.asarray([0], dtype=np.int64),
        "inside_grid": np.arange(1, threshold_count, dtype=np.int64),
        "beyond_grid": np.asarray([threshold_count], dtype=np.int64),
    }
    regions: dict[str, Any] = {}
    for label, indices in region_indices.items():
        truth_region = float(truth_bin_means[indices].sum(dtype=np.float64))
        primary_region = float(primary_bin_means[indices].sum(dtype=np.float64))
        component_region = primary_moment_components[:, indices].sum(axis=1) / count
        component_total = float(component_region.sum(dtype=np.float64))
        regions[label] = {
            "truth_mean_delta_squared_contribution": truth_region,
            "predicted_mean_delta_squared_contribution_128": primary_region,
            "predicted_over_truth": primary_region / truth_region if truth_region > 0.0 else None,
            "positive_excess_share": (
                float(positive_bin_excess[indices].sum(dtype=np.float64) / positive_excess_sum)
                if positive_excess_sum > 0.0
                else 0.0
            ),
            "ranked_component_moment_contributions_128": component_region.tolist(),
            "ranked_component_moment_shares_128": (
                component_region / component_total
                if component_total > 0.0
                else np.zeros(COMPONENTS, dtype=np.float64)
            ).tolist(),
        }

    threshold_rows: dict[str, Any] = {}
    identity_pass = True
    tail_convergence_pass = True
    supported_grid_indices: list[int] = []
    for threshold_index, label in enumerate(threshold_labels):
        tail_slice = slice(threshold_index + 1, None)
        truth_count = int(truth_counts[tail_slice].sum())
        truth_probability = truth_count / count
        analytic_components = probability_components[:, tail_slice].sum(axis=1) / count
        analytic_probability = float(analytic_components.sum(dtype=np.float64))
        primary_components = primary_moment_components[:, tail_slice].sum(axis=1) / count
        control_components = control_moment_components[:, tail_slice].sum(axis=1) / count
        primary_tail = float(primary_components.sum(dtype=np.float64))
        control_tail = float(control_components.sum(dtype=np.float64))
        truth_tail = float(truth_moment_sums[tail_slice].sum() / count)
        convergence = _relative_difference(primary_tail, control_tail)
        tail_convergence_pass = bool(
            tail_convergence_pass
            and convergence
            <= float(numerics["maximum_64_to_128_tail_moment_relative_difference"])
        )
        supported = truth_count >= int(
            numerics["minimum_empirical_exceedance_count_for_threshold_classification"]
        )
        available = bool(
            supported
            and min(truth_probability, analytic_probability, truth_tail, primary_tail) > 0.0
        )
        if available:
            truth_conditional = truth_tail / truth_probability
            predicted_conditional = primary_tail / analytic_probability
            probability_ratio = analytic_probability / truth_probability
            conditional_ratio = predicted_conditional / truth_conditional
            tail_ratio = primary_tail / truth_tail
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
            if supported:
                identity_pass = False
        moment_total = float(primary_components.sum(dtype=np.float64))
        probability_total = float(analytic_components.sum(dtype=np.float64))
        moment_shares = (
            primary_components / moment_total
            if moment_total > 0.0
            else np.zeros(COMPONENTS, dtype=np.float64)
        )
        probability_shares = (
            analytic_components / probability_total
            if probability_total > 0.0
            else np.zeros(COMPONENTS, dtype=np.float64)
        )
        dominant_rank = int(np.argmax(moment_shares))
        threshold_rows[label] = {
            "log10rho_threshold": float(thresholds[threshold_index]),
            "truth_exceedance_count": truth_count,
            "truth_exceedance_probability": float(truth_probability),
            "predicted_analytic_exceedance_probability": analytic_probability,
            "truth_mean_delta_squared_tail_contribution": truth_tail,
            "predicted_mean_delta_squared_tail_contribution_128": primary_tail,
            "predicted_mean_delta_squared_tail_contribution_64": control_tail,
            "tail_moment_64_to_128_relative_difference": convergence,
            "truth_conditional_mean_delta_squared": truth_conditional,
            "predicted_conditional_mean_delta_squared": predicted_conditional,
            "predicted_over_truth_probability": probability_ratio,
            "predicted_over_truth_conditional_amplitude": conditional_ratio,
            "predicted_over_truth_tail_moment": tail_ratio,
            "log_probability_ratio": log_probability,
            "log_conditional_amplitude_ratio": log_conditional,
            "log_tail_moment_ratio": log_tail,
            "log_ratio_identity_absolute_error": identity_error,
            "empirical_support_pass": supported,
            "ratio_available": available,
            "ranked_component_analytic_probability_contributions": analytic_components.tolist(),
            "ranked_component_analytic_probability_shares": probability_shares.tolist(),
            "ranked_component_moment_contributions_128": primary_components.tolist(),
            "ranked_component_moment_contributions_64": control_components.tolist(),
            "ranked_component_moment_shares_128": moment_shares.tolist(),
            "dominant_location_rank": dominant_rank,
            "single_component_tail_moment_dominates": bool(moment_shares[dominant_rank] >= 0.5),
        }

    if supported_grid_indices:
        local_indices = np.asarray([index - 1 for index in supported_grid_indices])
        weights = grid_weights[local_indices]
        weights = weights / weights.sum(dtype=np.float64)
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
            "renormalized_physical_moment_weights": weights.tolist(),
            "weighted_mean_absolute_log_probability_ratio": float(
                np.sum(weights * probability_errors, dtype=np.float64)
            ),
            "weighted_mean_absolute_log_conditional_amplitude_ratio": float(
                np.sum(weights * amplitude_errors, dtype=np.float64)
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

    bin_partition_error = max(
        _relative_error(float(primary_bins.sum()), primary_total),
        _relative_error(float(control_bins.sum()), control_total),
        _relative_error(float(truth_moment_sums.sum()), truth_total),
        _relative_error(float(probability_bins.sum()), count),
    )
    component_partition_error = max(
        _relative_error(float(primary_moment_components.sum()), primary_total),
        _relative_error(float(control_moment_components.sum()), control_total),
        _relative_error(float(probability_components.sum()), count),
    )
    reproduction_pass = max(reproduction.values()) <= float(
        numerics["maximum_exact_V56_gate_reproduction_relative_difference"]
    )
    numerical_pass = bool(
        reproduction_pass
        and threshold_rows["q99_999_anchor"]["ratio_available"]
        and complete_convergence
        <= float(numerics["maximum_64_to_128_complete_moment_relative_difference"])
        and high_accuracy_vs_v56
        <= float(
            numerics[
                "maximum_high_accuracy_complete_moment_relative_difference_from_exact_V56_64"
            ]
        )
        and tail_convergence_pass
        and identity_pass
        and bin_partition_error <= float(numerics["maximum_bin_partition_relative_error"])
        and component_partition_error
        <= float(numerics["maximum_component_partition_relative_error"])
    )
    return {
        "top_backbone_probe_voxels": count,
        "complete_moment": {
            "truth_mean_delta_squared": truth_mean,
            "exact_V56_unranked_GH64_mean_delta_squared": exact_v56_mean,
            "high_accuracy_GL128_mean_delta_squared": primary_mean,
            "control_GL64_mean_delta_squared": control_total / count,
            "exact_V56_over_truth": exact_v56_mean / truth_mean,
            "high_accuracy_over_truth": primary_mean / truth_mean,
            "GL64_to_GL128_relative_difference": complete_convergence,
            "GL128_to_exact_V56_GH64_relative_difference": high_accuracy_vs_v56,
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
        "reproduces_V56_gate": reproduction_pass,
        "numerical_requirements_pass": numerical_pass,
    }


@torch.inference_mode()
def _domain(
    model: torch.nn.Module,
    device: torch.device,
    v35: dict[str, Any],
    prepared: h5py.File,
    support: dict[str, Any],
    gate: dict[str, Any],
    domain: str,
    domain_index: int,
    thresholds: np.ndarray,
    grid_weights: np.ndarray,
    numerics: dict[str, Any],
    primary_order: int = PRIMARY_ORDER,
    control_order: int = CONTROL_ORDER,
    summary_function: Any = None,
    progress_label: str = "v58-audit",
) -> dict[str, Any]:
    row = v35["development_domains"][domain]
    objects = int(row["train_objects"])
    if objects != EXPECTED_OBJECTS[domain]:
        raise RuntimeError("V58 train object count differs")
    truth_probe = _truth_probe(v35, prepared, domain, domain_index)
    boundary = float(gate["domains"][domain]["backbone_boundaries"][2])
    top = truth_probe["backbone_base"].astype(np.float64) >= boundary
    expected_top = int(gate["domains"][domain]["strata"]["q99_9_and_above"]["count"])
    if int(top.sum()) != expected_top:
        raise ValueError("V58 top-backbone probe differs")
    truth_y = truth_probe["physical_y"][top]
    truth_log10rho = 4.5 * truth_y.astype(np.float64)
    truth_delta_squared = _physical_delta_squared(truth_y)
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    domain_maximum = float(support["domains"][domain]["maximum_standardized_residual"])
    gh_nodes, gh_weights = np.polynomial.hermite.hermgauss(PRIMARY_QUADRATURE_ORDER)
    gh_nodes_t = torch.from_numpy(gh_nodes).to(device)
    gh_weights_t = torch.from_numpy(gh_weights).to(device)
    exact_parts: list[np.ndarray] = []
    ranked_weight_parts: list[torch.Tensor] = []
    ranked_location_parts: list[torch.Tensor] = []
    ranked_scale_parts: list[torch.Tensor] = []
    base_parts: list[torch.Tensor] = []
    data, cache = _open_split(row, "train")
    try:
        from hong2021_v48_train import condition_cube

        for object_index in range(objects):
            condition, _, backbone = condition_cube(
                data, cache, prepared, domain, "train", object_index
            )
            parameter = model(torch.from_numpy(condition[None]).to(device))
            indices = _probe_indices(domain_index, object_index)
            index_tensor = torch.from_numpy(indices).to(device)
            flat = (
                parameter.reshape(1, 15, -1)
                .index_select(2, index_tensor)
                .reshape(1, 15, 1, 1, -1)
            )
            logits, locations, scales = mixture_parameters(flat)
            mixture_weights = torch.softmax(logits, dim=1)[0, :, 0, 0]
            locations = locations[0, :, 0, 0]
            scales = scales[0, :, 0, 0]
            base = torch.from_numpy(
                backbone.reshape(-1)[indices].astype(np.float64) + target_mean
            ).to(device)
            exact = _quadrature_object(
                mixture_weights,
                locations,
                scales,
                base,
                target_std,
                gh_nodes_t,
                gh_weights_t,
                domain_maximum,
            )["delta_squared"].sum(axis=0)
            start = object_index * PROBE_VOXELS
            local_top = top[start : start + PROBE_VOXELS]
            exact_parts.append(exact[local_top])
            selected = torch.from_numpy(np.flatnonzero(local_top)).to(device)
            if len(selected):
                selected_weights = mixture_weights.index_select(1, selected)
                selected_locations = locations.index_select(1, selected)
                selected_scales = scales.index_select(1, selected)
                selected_weights, selected_locations, selected_scales = _rank_components(
                    selected_weights, selected_locations, selected_scales
                )
                ranked_weight_parts.append(selected_weights.cpu())
                ranked_location_parts.append(selected_locations.cpu())
                ranked_scale_parts.append(selected_scales.cpu())
                base_parts.append(base.index_select(0, selected).cpu())
            if (object_index + 1) % 16 == 0 or object_index + 1 == objects:
                print(f"[{progress_label}] {domain} {object_index + 1}/{objects}", flush=True)
    finally:
        data.close()
        cache.close()
    exact_values = np.concatenate(exact_parts)
    ranked_weights = torch.cat(ranked_weight_parts, dim=1)
    ranked_locations = torch.cat(ranked_location_parts, dim=1)
    ranked_scales = torch.cat(ranked_scale_parts, dim=1)
    bases = torch.cat(base_parts)
    if exact_values.shape != (expected_top,) or ranked_weights.shape != (
        COMPONENTS,
        expected_top,
    ):
        raise RuntimeError("V58 collected top probe differs")
    thresholds_t = torch.from_numpy(thresholds)
    primary = _cdf_interval_bins(
        ranked_weights,
        ranked_locations,
        ranked_scales,
        bases,
        target_std,
        thresholds_t,
        primary_order,
    )
    control = _cdf_interval_bins(
        ranked_weights,
        ranked_locations,
        ranked_scales,
        bases,
        target_std,
        thresholds_t,
        control_order,
    )
    component_mass_sums = ranked_weights.double().sum(dim=1).cpu().numpy()
    summarize = _domain_summary if summary_function is None else summary_function
    return summarize(
        truth_log10rho,
        truth_delta_squared,
        exact_values,
        primary,
        control,
        component_mass_sums,
        thresholds,
        grid_weights,
        gate["domains"][domain],
        numerics,
    )


def audit(program_path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, v57_program, _, gate = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if not clean or socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V58 audit requires clean Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V58 audit requires Ada")
    frozen = program["frozen_inputs"]
    v56_program, v35, _ = load_v56_program(_path(repo, frozen["v56_program"]), repo)
    development = Path(v56_program["output_roots"]["development"])
    if development.exists():
        raise RuntimeError("V58 refuses a pre-existing V56 development directory")
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
    support = _verified_json(
        _path(repo, frozen["support_selection"]),
        frozen["support_selection_sha256"],
        "support selection",
    )
    partition = v57_program["fixed_threshold_partition"]
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
                support,
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
        "status": "complete_high_accuracy_train_only_grid_tail_audit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "checkpoint_sha256": frozen["v56_checkpoint_sha256"],
        "v57_audit_sha256": frozen["v57_audit_sha256"],
        "v56_train_gate_sha256": frozen["v56_train_gate_sha256"],
        "primary_interval_quadrature_order": PRIMARY_ORDER,
        "control_interval_quadrature_order": CONTROL_ORDER,
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
        raise FileExistsError("V58 refuses an existing audit")
    result = audit(args.program, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
