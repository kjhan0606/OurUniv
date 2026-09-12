#!/usr/bin/env python
"""V51 train-only audit of V50 bounded support, PIT, and physical tails."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v46_tail_occupancy_audit import (
    EXPECTED_OBJECTS,
    PROBE_VOXELS,
    _effective,
    _probe_indices,
    _quantiles,
)
from hong2021_v50_network import (
    LOWER_SUPPORT,
    SUPPORT_RANGE,
    UPPER_SUPPORT,
    LocalMixtureUNet,
    bounded_mixture_inverse,
    bounded_mixture_log_probability,
    bounded_to_latent,
    mixture_parameters,
    parameter_count,
)
from hong2021_v50_train import (
    CHECKPOINT_SCHEMA,
    PARAMETERS,
    PROGRAM_SHA256 as V50_PROGRAM_SHA256,
    condition_cube,
    load_cache,
    load_program as load_v50_program,
)


PROGRAM_SCHEMA = "hong2021-v51-train-only-bounded-support-calibration-audit-program-v1"
PROGRAM_SHA256 = "ba30af32cfd147e97ebd62118536cab5b0c9b30cdb81a17538fed5305f1928ba"
RESULT_SCHEMA = "hong2021-v51-train-only-bounded-support-calibration-audit-v1"
BOUNDARY_LAYERS = (0.05, 0.01, 0.001, 0.0001)
PIT_TAILS = (0.01, 0.001, 0.0001, 0.00001, 0.000001)
TAIL_TARGET_QUANTILES = (0.99, 0.999)
PREDICTIVE_QUANTILES = (0.99, 0.999, 0.9999, 0.99999)
PREDICTIVE_MEMBERS = 16
PREDICTIVE_SEED = 151051
STRATUM_QUANTILES = (0.9, 0.99, 0.999)
PRIMARY_QUADRATURE_ORDER = 64
CONTROL_QUADRATURE_ORDER = 32
GLOBAL_TRAIN_MINIMUM = -12.695331409918287
GLOBAL_TRAIN_MAXIMUM = 9.165971027770349
MINIMUM_COMMON_DOMAINS = 2
MINIMUM_EXPECTED_BEYOND_GLOBAL_MAXIMUM = 10.0
DOMINANT_SECOND_MOMENT_FRACTION = 0.5
PIT_CLASSIFICATION_TAIL = 0.00001
PIT_RATIO_MINIMUM = 0.5
PIT_RATIO_MAXIMUM = 2.0
Q99_999_DEX_THRESHOLD = 0.10
MOMENT_RATIO_MINIMUM = 2.0 / 3.0
MOMENT_RATIO_MAXIMUM = 1.5
RISK_TOP_STRATUM_MOMENT_RATIO_MAXIMUM = 1.5
EFFECTIVE_COMPONENT_THRESHOLD = 3.0
WEIGHT_THRESHOLD = 0.02
RESPONSIBILITY_RATIO_THRESHOLD = 0.5
QUADRATURE_RELATIVE_DIFFERENCE_MAXIMUM = 0.005
INVERSE_ERROR_MAXIMUM = 2.0e-6


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V51 {label} hash differs")
    return json.loads(path.read_text())


def load_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_audit_implementation_or_execution"
    ):
        raise ValueError("V51 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        (repo / parent["v50_record"]).resolve(),
        parent["v50_record_sha256"],
        "V50 result record",
    )
    decision_record = record.get("development_decision", {})
    if (
        record.get("status") != parent["required_status"]
        or decision_record.get("classification") != parent["required_classification"]
        or decision_record.get("next") != parent["required_next"]
        or decision_record.get("development_pass") is not parent["required_development_pass"]
        or decision_record.get("candidate_high_k_power_and_residual_RMS_all_domains")
        is not parent["required_high_k_power_and_residual_RMS_all_domains"]
        or record.get("firewall", {}).get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V51 V50 conclusion or firewall differs")
    frozen = program["frozen_inputs"]
    for key, digest_key in (
        ("v50_program", "v50_program_sha256"),
        ("support_selection", "support_selection_sha256"),
        ("checkpoint", "checkpoint_sha256"),
        ("training_report", "training_report_sha256"),
        ("conditioning_cache", "conditioning_cache_sha256"),
        ("preflight", "preflight_sha256"),
        ("development_decision", "development_decision_sha256"),
    ):
        candidate = Path(frozen[key])
        if not candidate.is_absolute():
            candidate = repo / candidate
        if sha256_file(candidate.resolve()) != frozen[digest_key]:
            raise ValueError(f"V51 frozen {key} hash differs")
    support = json.loads(Path(frozen["support_selection"]).read_text())
    report_text = Path(frozen["training_report"]).read_text()
    report = json.loads(report_text)
    decision = json.loads(Path(frozen["development_decision"]).read_text())
    if (
        canonical_digest(support)
        != frozen["support_selection_decision_digest_sha256"]
        or canonical_digest(report) != frozen["training_report_decision_digest_sha256"]
        or canonical_digest(decision)
        != frozen["development_decision_digest_sha256"]
        or report_text.count("Infinity") != 1
        or support.get("support", {}).get("lower_support") != LOWER_SUPPORT
        or support.get("support", {}).get("upper_support") != UPPER_SUPPORT
        or support.get("support", {}).get("global_minimum") != GLOBAL_TRAIN_MINIMUM
        or support.get("support", {}).get("global_maximum") != GLOBAL_TRAIN_MAXIMUM
        or support.get("all_train_values_strictly_interior") is not True
        or decision.get("classification") != parent["required_classification"]
        or decision.get("next") != parent["required_next"]
        or decision.get("independent_gate_locked") is not True
        or decision.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V51 frozen digest, support, anomaly, or decision differs")
    _, v35, _ = load_v50_program((repo / frozen["v50_program"]).resolve(), repo)
    return program, v35, support, decision


def classify(
    boundary_dominated: bool,
    risk_amplified: bool,
    upper_tail_overdispersed: bool,
    component_failure: bool,
    moment_underconstrained: bool,
    train_tail_calibrated: bool,
) -> tuple[str, str]:
    if boundary_dominated:
        return (
            "train_unoccupied_fixed_support_margin_dominates_the_bounded_physical_tail",
            "freeze_a_train_only_bounded_rational_quadratic_spline_likelihood_on_the_same_support_with_endpoint_mass_gates_before_fit",
        )
    if risk_amplified:
        return (
            "structure_risk_conditioning_amplifies_the_bounded_extreme_tail",
            "freeze_a_matched_train_only_no-risk-channel_model_ablation_with_all_other_V50_information_support_and_gates_unchanged",
        )
    if upper_tail_overdispersed:
        return (
            "bounded_logit_mixture_is_overdispersed_in_the_train_upper_tail",
            "freeze_a_train_only_bounded_rational_quadratic_spline_likelihood_on_the_same_support_and_gates",
        )
    if component_failure:
        return (
            "V50_bounded_latent_mixture_has_effective_component_failure",
            "stop_finite_mixtures_and_freeze_a_single_train_only_bounded_monotone_spline_likelihood_on_the_same_support",
        )
    if moment_underconstrained:
        return (
            "bounded_voxel_log_score_calibrates_train_ranks_but_not_the_physical_second_moment",
            "freeze_a_train_only_strictly_proper_physical_tail_scoring_comparison_before_changing_the_generator",
        )
    if train_tail_calibrated:
        return (
            "train_bounded_marginal_is_calibrated_but_empirical_rank_copula_or_query_shift_breaks_development_extremes",
            "audit_only_train_empirical_rank_tail_dependence_and_train_to_development_condition_shift_without_changing_the_bounded_likelihood",
        )
    return (
        "bounded_support_extreme_failure_is_mixed_or_not_identified",
        "seal_boundary_PIT_component_risk_strata_and_bounded_moment_evidence_before_selecting_one_new_model",
    )


def _load_model(
    program: dict[str, Any], repo: Path, commit: str
) -> tuple[LocalMixtureUNet, dict[str, Any]]:
    frozen = program["frozen_inputs"]
    checkpoint = torch.load(frozen["checkpoint"], map_location="cpu", weights_only=False)
    source_commit = str(checkpoint.get("code_commit"))
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("program_sha256") != V50_PROGRAM_SHA256
        or checkpoint.get("step") != 12_000
        or checkpoint.get("parameters") != PARAMETERS
        or checkpoint.get("conditioning_cache_sha256")
        != frozen["conditioning_cache_sha256"]
        or checkpoint.get("support_selection_sha256")
        != frozen["support_selection_sha256"]
        or checkpoint.get("open_standardized_support")
        != [LOWER_SUPPORT, UPPER_SUPPORT]
        or checkpoint.get("sample_clipping") is not False
        or checkpoint.get("component_scale_cap") is not False
        or checkpoint.get("validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection")
        is not False
        or checkpoint.get("Astrid_accessed") is not False
        or checkpoint.get("historical_EAGLE_accessed") is not False
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, commit],
            cwd=repo,
            capture_output=True,
        ).returncode
    ):
        raise ValueError("V51 V50 checkpoint binding differs")
    model = LocalMixtureUNet()
    if parameter_count(model) != PARAMETERS:
        raise RuntimeError("V51 V50 parameter count differs")
    model.load_state_dict(checkpoint["ema_state_dict"])
    return model, checkpoint


def _support_threshold(layer: float, upper: bool) -> float:
    if not 0.0 < layer < 0.5:
        raise ValueError("V51 support layer differs")
    coordinate = 1.0 - layer if upper else layer
    return LOWER_SUPPORT + SUPPORT_RANGE * coordinate


def _bounded_mixture_cdf64(
    parameters: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Accurate float64 bounded CDF for PIT and probabilities down to 1e-6."""
    logits, locations, scales = mixture_parameters(parameters)
    if value.shape != (len(parameters), 1, *parameters.shape[-3:]):
        raise ValueError("V51 bounded CDF value shape differs")
    latent = bounded_to_latent(value).double()
    weights = torch.softmax(logits.double(), dim=1)
    standardized = (latent - locations.double()) / scales.double()
    component_cdf = torch.special.ndtr(standardized)
    result = torch.sum(weights * component_cdf, dim=1, keepdim=True)
    return torch.clamp(result, 0.0, 1.0)


def _physical_delta_squared(y: np.ndarray) -> np.ndarray:
    value = np.power(10.0, 4.5 * y.astype(np.float64)) - 1.0
    result = np.square(value)
    if not np.isfinite(result).all():
        raise RuntimeError("V51 empirical physical moment differs")
    return result


def _relative_difference(primary: float, control: float) -> float:
    denominator = max(abs(primary), abs(control), 1.0e-300)
    return abs(primary - control) / denominator


def _truth_probe(
    v35: dict[str, Any], prepared: h5py.File, domain: str, domain_index: int
) -> dict[str, np.ndarray]:
    row = v35["development_domains"][domain]
    objects = int(row["train_objects"])
    if objects != EXPECTED_OBJECTS[domain]:
        raise RuntimeError("V51 train object count differs")
    values: dict[str, list[np.ndarray]] = {
        "standardized": [],
        "physical_y": [],
        "backbone_base": [],
        "risk": [],
    }
    data, cache = _open_split(row, "train")
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    try:
        for object_index in range(objects):
            indices = _probe_indices(domain_index, object_index)
            condition, target, backbone = condition_cube(
                data, cache, prepared, domain, "train", object_index
            )
            flat_target = target.reshape(-1)[indices]
            flat_backbone = backbone.reshape(-1)[indices]
            truth = np.asarray(data["target"][object_index, 0], dtype=np.float32)
            exact_standardized = (
                truth.reshape(-1)[indices].astype(np.float64)
                - flat_backbone.astype(np.float64)
                - target_mean
            ) / target_std
            values["standardized"].append(exact_standardized.astype(np.float32))
            values["physical_y"].append(
                truth.reshape(-1)[indices].astype(np.float32)
            )
            values["backbone_base"].append(
                (flat_backbone + target_mean).astype(np.float32)
            )
            values["risk"].append(
                condition[5].reshape(-1)[indices].astype(np.float32)
            )
    finally:
        data.close()
        cache.close()
    result = {name: np.concatenate(parts) for name, parts in values.items()}
    expected = objects * PROBE_VOXELS
    if any(len(value) != expected or not np.isfinite(value).all() for value in result.values()):
        raise RuntimeError("V51 truth probe differs")
    return result


def _quadrature_object(
    weights: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
    base: torch.Tensor,
    target_std: float,
    nodes: torch.Tensor,
    quadrature_weights: torch.Tensor,
    domain_maximum: float,
) -> dict[str, Any]:
    """Evaluate bounded physical moments for one fixed-probe object."""
    if weights.ndim != 2 or weights.shape != locations.shape or scales.shape != weights.shape:
        raise ValueError("V51 quadrature component shape differs")
    if base.ndim != 1 or base.shape[0] != weights.shape[1]:
        raise ValueError("V51 quadrature base shape differs")
    normalized = quadrature_weights.double() / math.sqrt(math.pi)
    if nodes.ndim != 1 or normalized.shape != nodes.shape:
        raise ValueError("V51 quadrature rule differs")
    components, probes = weights.shape
    first = torch.zeros((components, probes), dtype=torch.float64, device=weights.device)
    second = torch.zeros_like(first)
    delta_squared = torch.zeros_like(first)
    upper_second = {layer: 0.0 for layer in BOUNDARY_LAYERS}
    upper_delta_squared = {layer: 0.0 for layer in BOUNDARY_LAYERS}
    global_second = 0.0
    global_delta_squared = 0.0
    domain_second = 0.0
    domain_delta_squared = 0.0
    coefficient = 4.5 * math.log(10.0)
    for component in range(components):
        latent = locations[component].double()[:, None] + math.sqrt(2.0) * (
            scales[component].double()[:, None] * nodes.double()[None]
        )
        standardized = LOWER_SUPPORT + SUPPORT_RANGE * torch.sigmoid(latent)
        physical_y = base.double()[:, None] + float(target_std) * standardized
        rho = torch.exp(coefficient * physical_y)
        rho_squared = torch.square(rho)
        component_delta_squared = torch.square(rho - 1.0)
        mixture_quadrature = (
            weights[component].double()[:, None] * normalized[None]
        )
        first[component] = torch.sum(mixture_quadrature * rho, dim=1)
        second[component] = torch.sum(mixture_quadrature * rho_squared, dim=1)
        delta_squared[component] = torch.sum(
            mixture_quadrature * component_delta_squared, dim=1
        )
        for layer in BOUNDARY_LAYERS:
            mask = standardized >= _support_threshold(layer, True)
            upper_second[layer] += float(
                torch.sum(mixture_quadrature * rho_squared * mask).cpu()
            )
            upper_delta_squared[layer] += float(
                torch.sum(mixture_quadrature * component_delta_squared * mask).cpu()
            )
        global_mask = standardized > GLOBAL_TRAIN_MAXIMUM
        global_second += float(
            torch.sum(mixture_quadrature * rho_squared * global_mask).cpu()
        )
        global_delta_squared += float(
            torch.sum(mixture_quadrature * component_delta_squared * global_mask).cpu()
        )
        domain_mask = standardized > float(domain_maximum)
        domain_second += float(
            torch.sum(mixture_quadrature * rho_squared * domain_mask).cpu()
        )
        domain_delta_squared += float(
            torch.sum(mixture_quadrature * component_delta_squared * domain_mask).cpu()
        )
    if not (
        torch.isfinite(first).all()
        and torch.isfinite(second).all()
        and torch.isfinite(delta_squared).all()
    ):
        raise RuntimeError("V51 bounded quadrature moment is nonfinite")
    return {
        "first": first.cpu().numpy(),
        "second": second.cpu().numpy(),
        "delta_squared": delta_squared.cpu().numpy(),
        "upper_second": upper_second,
        "upper_delta_squared": upper_delta_squared,
        "above_global_second": global_second,
        "above_global_delta_squared": global_delta_squared,
        "above_domain_second": domain_second,
        "above_domain_delta_squared": domain_delta_squared,
    }


def _strata_summary(
    variable: np.ndarray,
    truth_delta_squared: np.ndarray,
    predicted_delta_squared: np.ndarray,
    probability_above_global_maximum: np.ndarray,
    upper_one_percent_probability: np.ndarray,
) -> dict[str, Any]:
    if not (
        variable.shape
        == truth_delta_squared.shape
        == predicted_delta_squared.shape
        == probability_above_global_maximum.shape
        == upper_one_percent_probability.shape
    ):
        raise ValueError("V51 stratum input shape differs")
    boundaries = np.quantile(variable.astype(np.float64), STRATUM_QUANTILES)
    masks = (
        variable < boundaries[0],
        (variable >= boundaries[0]) & (variable < boundaries[1]),
        (variable >= boundaries[1]) & (variable < boundaries[2]),
        variable >= boundaries[2],
    )
    labels = ("below_q90", "q90_to_q99", "q99_to_q99_9", "q99_9_and_above")
    rows: dict[str, Any] = {}
    for label, mask in zip(labels, masks, strict=True):
        count = int(mask.sum())
        if count <= 0:
            raise RuntimeError("V51 empty conditional stratum")
        truth = float(np.mean(truth_delta_squared[mask], dtype=np.float64))
        predicted = float(np.mean(predicted_delta_squared[mask], dtype=np.float64))
        if not math.isfinite(truth) or not math.isfinite(predicted) or truth <= 0.0:
            raise RuntimeError("V51 conditional moment differs")
        rows[label] = {
            "count": count,
            "truth_mean_delta_squared": truth,
            "quadrature_mean_delta_squared": predicted,
            "quadrature_over_truth_mean_delta_squared": predicted / truth,
            "mean_probability_above_global_train_maximum": float(
                np.mean(probability_above_global_maximum[mask], dtype=np.float64)
            ),
            "mean_upper_one_percent_boundary_probability": float(
                np.mean(upper_one_percent_probability[mask], dtype=np.float64)
            ),
        }
    return {"boundaries": boundaries.tolist(), "strata": rows}


@torch.inference_mode()
def _audit_domain(
    model: LocalMixtureUNet,
    device: torch.device,
    v35: dict[str, Any],
    prepared: h5py.File,
    support: dict[str, Any],
    domain: str,
    domain_index: int,
    truth_probe: dict[str, np.ndarray],
) -> dict[str, Any]:
    row = v35["development_domains"][domain]
    objects = int(row["train_objects"])
    total_voxels = objects * 64**3
    expected_voxels = int(support["domains"][domain]["native_voxels"])
    if objects != EXPECTED_OBJECTS[domain] or total_voxels != expected_voxels:
        raise RuntimeError("V51 train population differs")
    domain_minimum = float(support["domains"][domain]["minimum_standardized_residual"])
    domain_maximum = float(support["domains"][domain]["maximum_standardized_residual"])
    tail_thresholds = np.quantile(
        truth_probe["standardized"].astype(np.float64), TAIL_TARGET_QUANTILES
    )
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    nll_sum = 0.0
    tail_nll_sum = np.zeros(len(TAIL_TARGET_QUANTILES), dtype=np.float64)
    tail_count = np.zeros(len(TAIL_TARGET_QUANTILES), dtype=np.int64)
    pit_lower = np.zeros(len(PIT_TAILS), dtype=np.int64)
    pit_upper = np.zeros(len(PIT_TAILS), dtype=np.int64)
    truth_boundary_lower = {layer: 0 for layer in BOUNDARY_LAYERS}
    truth_boundary_upper = {layer: 0 for layer in BOUNDARY_LAYERS}
    model_boundary_lower_sum = {layer: 0.0 for layer in BOUNDARY_LAYERS}
    model_boundary_upper_sum = {layer: 0.0 for layer in BOUNDARY_LAYERS}
    model_beyond_sum = {
        "below_global_minimum": 0.0,
        "above_global_maximum": 0.0,
        "below_domain_minimum": 0.0,
        "above_domain_maximum": 0.0,
    }
    truth_beyond_count = {name: 0 for name in model_beyond_sum}
    weight_sum = np.zeros(5, dtype=np.float64)
    responsibility_sum = np.zeros(5, dtype=np.float64)
    tail_responsibility_sum = np.zeros((2, 5), dtype=np.float64)
    primary_component_first_sum = np.zeros(5, dtype=np.float64)
    primary_component_second_sum = np.zeros(5, dtype=np.float64)
    primary_component_delta_squared_sum = np.zeros(5, dtype=np.float64)
    control_first_sum = 0.0
    control_second_sum = 0.0
    control_delta_squared_sum = 0.0
    primary_upper_second = {layer: 0.0 for layer in BOUNDARY_LAYERS}
    primary_upper_delta_squared = {layer: 0.0 for layer in BOUNDARY_LAYERS}
    primary_beyond = {
        "above_global_second": 0.0,
        "above_global_delta_squared": 0.0,
        "above_domain_second": 0.0,
        "above_domain_delta_squared": 0.0,
    }
    probe_quadrature_delta_squared: list[np.ndarray] = []
    probe_probability_above_global: list[np.ndarray] = []
    probe_probability_upper_one_percent: list[np.ndarray] = []
    predictive_standardized: list[np.ndarray] = []
    predictive_y: list[np.ndarray] = []
    maximum_inverse_error = 0.0
    nodes64_np, weights64_np = np.polynomial.hermite.hermgauss(PRIMARY_QUADRATURE_ORDER)
    nodes32_np, weights32_np = np.polynomial.hermite.hermgauss(CONTROL_QUADRATURE_ORDER)
    nodes64 = torch.from_numpy(nodes64_np).to(device)
    weights64 = torch.from_numpy(weights64_np).to(device)
    nodes32 = torch.from_numpy(nodes32_np).to(device)
    weights32 = torch.from_numpy(weights32_np).to(device)
    data, cache = _open_split(row, "train")
    try:
        for object_index in range(objects):
            condition, target, backbone = condition_cube(
                data, cache, prepared, domain, "train", object_index
            )
            condition_tensor = torch.from_numpy(condition[None]).to(device)
            observed = torch.from_numpy(target[None]).to(device)
            parameter = model(condition_tensor)
            log_probability = bounded_mixture_log_probability(parameter, observed)
            cdf = _bounded_mixture_cdf64(parameter, observed)
            logits, locations, scales = mixture_parameters(parameter)
            weights = F.softmax(logits, dim=1)
            latent = bounded_to_latent(observed).float()
            standardized_latent = (latent - locations) / scales
            component_log_probability = (
                -0.5 * torch.square(standardized_latent)
                - torch.log(scales)
                - 0.5 * math.log(2.0 * math.pi)
            )
            responsibilities = torch.softmax(
                F.log_softmax(logits, dim=1) + component_log_probability, dim=1
            )
            if (
                not torch.isfinite(parameter).all()
                or not torch.isfinite(log_probability).all()
                or not torch.isfinite(cdf).all()
                or float(torch.max(torch.abs(weights.sum(dim=1) - 1.0)).cpu())
                > 1.0e-5
                or float(
                    torch.max(torch.abs(responsibilities.sum(dim=1) - 1.0)).cpu()
                )
                > 1.0e-5
            ):
                raise RuntimeError("V51 bounded likelihood integrity differs")
            nll = -log_probability
            nll_sum += float(nll.double().sum().cpu())
            flat_target_tensor = observed.reshape(-1)
            truth_native = np.asarray(
                data["target"][object_index, 0], dtype=np.float32
            )
            flat_target_numpy = (
                truth_native.astype(np.float64)
                - backbone[0].astype(np.float64)
                - target_mean
            ).reshape(-1) / target_std
            if not (
                np.isfinite(flat_target_numpy).all()
                and np.all(
                    (flat_target_numpy > LOWER_SUPPORT)
                    & (flat_target_numpy < UPPER_SUPPORT)
                )
            ):
                raise RuntimeError("V51 exact train support occupancy differs")
            for tail_index, threshold in enumerate(tail_thresholds):
                mask = flat_target_tensor >= float(threshold)
                count = int(mask.sum().cpu())
                tail_count[tail_index] += count
                tail_nll_sum[tail_index] += float(
                    nll.reshape(-1)[mask].double().sum().cpu()
                )
                tail_responsibility_sum[tail_index] += (
                    responsibilities.permute(0, 2, 3, 4, 1)
                    .reshape(-1, 5)[mask]
                    .double()
                    .sum(dim=0)
                    .cpu()
                    .numpy()
                )
            for pit_index, probability in enumerate(PIT_TAILS):
                pit_lower[pit_index] += int((cdf < probability).sum().cpu())
                pit_upper[pit_index] += int((cdf > 1.0 - probability).sum().cpu())
            for layer in BOUNDARY_LAYERS:
                lower_threshold = _support_threshold(layer, False)
                upper_threshold = _support_threshold(layer, True)
                truth_boundary_lower[layer] += int(
                    np.count_nonzero(flat_target_numpy <= lower_threshold)
                )
                truth_boundary_upper[layer] += int(
                    np.count_nonzero(flat_target_numpy >= upper_threshold)
                )
                lower_value = torch.full_like(observed, lower_threshold)
                upper_value = torch.full_like(observed, upper_threshold)
                model_boundary_lower_sum[layer] += float(
                    _bounded_mixture_cdf64(parameter, lower_value).sum().cpu()
                )
                model_boundary_upper_sum[layer] += float(
                    (1.0 - _bounded_mixture_cdf64(parameter, upper_value))
                    .sum()
                    .cpu()
                )
            thresholds = {
                "below_global_minimum": (GLOBAL_TRAIN_MINIMUM, False),
                "above_global_maximum": (GLOBAL_TRAIN_MAXIMUM, True),
                "below_domain_minimum": (domain_minimum, False),
                "above_domain_maximum": (domain_maximum, True),
            }
            for name, (threshold, upper) in thresholds.items():
                value = torch.full_like(observed, threshold)
                probability = _bounded_mixture_cdf64(parameter, value)
                if upper:
                    probability = 1.0 - probability
                    truth_beyond_count[name] += int(
                        np.count_nonzero(flat_target_numpy > threshold)
                    )
                else:
                    truth_beyond_count[name] += int(
                        np.count_nonzero(flat_target_numpy < threshold)
                    )
                model_beyond_sum[name] += float(probability.double().sum().cpu())
            weight_sum += weights.double().sum(dim=(0, 2, 3, 4)).cpu().numpy()
            responsibility_sum += (
                responsibilities.double().sum(dim=(0, 2, 3, 4)).cpu().numpy()
            )

            indices = _probe_indices(domain_index, object_index)
            index_tensor = torch.from_numpy(indices).to(device)
            flat_parameter = (
                parameter.reshape(1, 15, -1)
                .index_select(2, index_tensor)
                .reshape(1, 15, 1, 1, -1)
            )
            probe_logits, probe_locations, probe_scales = mixture_parameters(flat_parameter)
            probe_weights = torch.softmax(probe_logits, dim=1)[0, :, 0, 0]
            probe_locations = probe_locations[0, :, 0, 0]
            probe_scales = probe_scales[0, :, 0, 0]
            probe_base = torch.from_numpy(
                backbone.reshape(-1)[indices].astype(np.float64) + target_mean
            ).to(device)
            primary = _quadrature_object(
                probe_weights,
                probe_locations,
                probe_scales,
                probe_base,
                target_std,
                nodes64,
                weights64,
                domain_maximum,
            )
            control = _quadrature_object(
                probe_weights,
                probe_locations,
                probe_scales,
                probe_base,
                target_std,
                nodes32,
                weights32,
                domain_maximum,
            )
            primary_component_first_sum += primary["first"].sum(axis=1)
            primary_component_second_sum += primary["second"].sum(axis=1)
            primary_component_delta_squared_sum += primary["delta_squared"].sum(axis=1)
            control_first_sum += float(control["first"].sum())
            control_second_sum += float(control["second"].sum())
            control_delta_squared_sum += float(control["delta_squared"].sum())
            for layer in BOUNDARY_LAYERS:
                primary_upper_second[layer] += primary["upper_second"][layer]
                primary_upper_delta_squared[layer] += primary["upper_delta_squared"][layer]
            for name in primary_beyond:
                primary_beyond[name] += primary[name]
            probe_quadrature_delta_squared.append(
                primary["delta_squared"].sum(axis=0)
            )
            global_value = torch.full(
                (1, 1, 1, 1, PROBE_VOXELS),
                GLOBAL_TRAIN_MAXIMUM,
                dtype=flat_parameter.dtype,
                device=device,
            )
            upper_one_value = torch.full_like(
                global_value, _support_threshold(0.01, True)
            )
            probe_probability_above_global.append(
                (1.0 - _bounded_mixture_cdf64(flat_parameter, global_value))
                .cpu()
                .numpy()
                .reshape(-1)
            )
            probe_probability_upper_one_percent.append(
                (1.0 - _bounded_mixture_cdf64(flat_parameter, upper_one_value))
                .cpu()
                .numpy()
                .reshape(-1)
            )
            flat_backbone = backbone.reshape(-1)[indices]
            for member in range(PREDICTIVE_MEMBERS):
                generator = np.random.default_rng(
                    PREDICTIVE_SEED
                    + member
                    + 10_000_000 * domain_index
                    + 10_000 * object_index
                )
                rank = generator.random(PROBE_VOXELS, dtype=np.float32).reshape(
                    1, 1, 1, 1, -1
                )
                rank_tensor = torch.from_numpy(rank).to(device)
                draw = bounded_mixture_inverse(flat_parameter, rank_tensor)
                error = float(
                    torch.max(
                        torch.abs(
                            _bounded_mixture_cdf64(flat_parameter, draw)
                            - rank_tensor.double()
                        )
                    ).cpu()
                )
                maximum_inverse_error = max(maximum_inverse_error, error)
                draw_numpy = draw.cpu().numpy().reshape(-1)
                predictive_standardized.append(draw_numpy.astype(np.float32))
                predictive_y.append(
                    (flat_backbone + target_mean + target_std * draw_numpy).astype(
                        np.float32
                    )
                )
            if (object_index + 1) % 16 == 0 or object_index + 1 == objects:
                print(f"[v51-audit] {domain} {object_index + 1}/{objects}", flush=True)
    finally:
        data.close()
        cache.close()

    if tail_count.min() <= 0 or maximum_inverse_error > INVERSE_ERROR_MAXIMUM:
        raise RuntimeError("V51 tail count or inverse error differs")
    mean_weight = weight_sum / total_voxels
    mean_responsibility = responsibility_sum / total_voxels
    responsibility_ratio = mean_responsibility / mean_weight
    probe_voxels = objects * PROBE_VOXELS
    primary_first = primary_component_first_sum / probe_voxels
    primary_second = primary_component_second_sum / probe_voxels
    primary_delta_squared = primary_component_delta_squared_sum / probe_voxels
    primary_first_total = float(primary_first.sum())
    primary_second_total = float(primary_second.sum())
    primary_delta_squared_total = float(primary_delta_squared.sum())
    control_first = control_first_sum / probe_voxels
    control_second = control_second_sum / probe_voxels
    control_delta_squared = control_delta_squared_sum / probe_voxels
    quadrature_relative = {
        "mean_rho": _relative_difference(primary_first_total, control_first),
        "mean_rho_squared": _relative_difference(primary_second_total, control_second),
        "mean_delta_squared": _relative_difference(
            primary_delta_squared_total, control_delta_squared
        ),
    }
    if max(quadrature_relative.values()) > QUADRATURE_RELATIVE_DIFFERENCE_MAXIMUM:
        raise RuntimeError("V51 quadrature convergence differs")
    truth_standardized = truth_probe["standardized"]
    truth_y = truth_probe["physical_y"]
    truth_delta_squared = _physical_delta_squared(truth_y)
    predicted_standardized = np.concatenate(predictive_standardized)
    predicted_y = np.concatenate(predictive_y)
    predicted_delta_squared = _physical_delta_squared(predicted_y)
    quadrature_delta_squared = np.concatenate(probe_quadrature_delta_squared)
    probability_above_global = np.concatenate(probe_probability_above_global)
    probability_upper_one_percent = np.concatenate(
        probe_probability_upper_one_percent
    )
    if any(
        value.shape != truth_standardized.shape
        for value in (
            quadrature_delta_squared,
            probability_above_global,
            probability_upper_one_percent,
        )
    ):
        raise RuntimeError("V51 probe alignment differs")
    truth_log10rho_quantiles = _quantiles(4.5 * truth_y, PREDICTIVE_QUANTILES)
    predicted_log10rho_quantiles = _quantiles(
        4.5 * predicted_y, PREDICTIVE_QUANTILES
    )
    truth_probe_moment = float(np.mean(truth_delta_squared, dtype=np.float64))
    sampled_probe_moment = float(
        np.mean(predicted_delta_squared, dtype=np.float64)
    )
    pit = {
        str(probability): {
            "lower_count": int(pit_lower[index]),
            "upper_count": int(pit_upper[index]),
            "lower_fraction": float(pit_lower[index] / total_voxels),
            "upper_fraction": float(pit_upper[index] / total_voxels),
            "lower_observed_over_expected": float(
                pit_lower[index] / total_voxels / probability
            ),
            "upper_observed_over_expected": float(
                pit_upper[index] / total_voxels / probability
            ),
        }
        for index, probability in enumerate(PIT_TAILS)
    }
    boundary_occupancy = {
        str(layer): {
            "lower_threshold": _support_threshold(layer, False),
            "upper_threshold": _support_threshold(layer, True),
            "truth_lower_count": truth_boundary_lower[layer],
            "truth_upper_count": truth_boundary_upper[layer],
            "truth_lower_fraction": truth_boundary_lower[layer] / total_voxels,
            "truth_upper_fraction": truth_boundary_upper[layer] / total_voxels,
            "model_mean_lower_probability": model_boundary_lower_sum[layer]
            / total_voxels,
            "model_mean_upper_probability": model_boundary_upper_sum[layer]
            / total_voxels,
            "quadrature_rho_squared_fraction_from_upper_layer": primary_upper_second[
                layer
            ]
            / (primary_second_total * probe_voxels),
            "quadrature_delta_squared_fraction_from_upper_layer": primary_upper_delta_squared[
                layer
            ]
            / (primary_delta_squared_total * probe_voxels),
        }
        for layer in BOUNDARY_LAYERS
    }
    beyond_extrema = {
        name: {
            "truth_count": truth_beyond_count[name],
            "model_mean_probability": model_beyond_sum[name] / total_voxels,
            "model_expected_native_voxels": model_beyond_sum[name],
        }
        for name in model_beyond_sum
    }
    beyond_extrema["above_global_maximum"].update(
        {
            "quadrature_rho_squared_fraction": primary_beyond[
                "above_global_second"
            ]
            / (primary_second_total * probe_voxels),
            "quadrature_delta_squared_fraction": primary_beyond[
                "above_global_delta_squared"
            ]
            / (primary_delta_squared_total * probe_voxels),
        }
    )
    beyond_extrema["above_domain_maximum"].update(
        {
            "quadrature_rho_squared_fraction": primary_beyond[
                "above_domain_second"
            ]
            / (primary_second_total * probe_voxels),
            "quadrature_delta_squared_fraction": primary_beyond[
                "above_domain_delta_squared"
            ]
            / (primary_delta_squared_total * probe_voxels),
        }
    )
    unsupported_components = [
        component
        for component in range(5)
        if mean_weight[component] >= WEIGHT_THRESHOLD
        and responsibility_ratio[component] < RESPONSIBILITY_RATIO_THRESHOLD
    ]
    return {
        "train_objects": objects,
        "total_native_voxels": total_voxels,
        "probe_voxels": probe_voxels,
        "posterior_predictive_values": int(len(predicted_y)),
        "domain_train_minimum": domain_minimum,
        "domain_train_maximum": domain_maximum,
        "mean_NLL": float(nll_sum / total_voxels),
        "upper_tail_NLL": {
            "q99": float(tail_nll_sum[0] / tail_count[0]),
            "q99_9": float(tail_nll_sum[1] / tail_count[1]),
        },
        "PIT": pit,
        "support_boundary_occupancy": boundary_occupancy,
        "beyond_train_extrema": beyond_extrema,
        "mean_mixture_weight": mean_weight.tolist(),
        "mean_posterior_responsibility": mean_responsibility.tolist(),
        "responsibility_to_weight_ratio": responsibility_ratio.tolist(),
        "upper_tail_mean_posterior_responsibility": {
            "q99": (tail_responsibility_sum[0] / tail_count[0]).tolist(),
            "q99_9": (tail_responsibility_sum[1] / tail_count[1]).tolist(),
        },
        "global_effective_responsibility_components": _effective(
            mean_responsibility
        ),
        "unsupported_component_indices": unsupported_components,
        "bounded_quadrature_probe": {
            "primary_order": PRIMARY_QUADRATURE_ORDER,
            "control_order": CONTROL_QUADRATURE_ORDER,
            "primary_mean_rho_by_component": primary_first.tolist(),
            "primary_mean_rho_squared_by_component": primary_second.tolist(),
            "primary_mean_delta_squared_by_component": primary_delta_squared.tolist(),
            "rho_squared_component_fraction": (
                primary_second / primary_second_total
            ).tolist(),
            "delta_squared_component_fraction": (
                primary_delta_squared / primary_delta_squared_total
            ).tolist(),
            "primary_mean_rho": primary_first_total,
            "primary_mean_rho_squared": primary_second_total,
            "primary_mean_delta_squared": primary_delta_squared_total,
            "control_mean_rho": control_first,
            "control_mean_rho_squared": control_second,
            "control_mean_delta_squared": control_delta_squared,
            "relative_difference_32_to_64": quadrature_relative,
            "truth_mean_delta_squared": truth_probe_moment,
            "primary_over_truth_mean_delta_squared": primary_delta_squared_total
            / truth_probe_moment,
        },
        "posterior_predictive_probe": {
            "standardized_truth_quantiles": _quantiles(
                truth_standardized, PREDICTIVE_QUANTILES
            ),
            "standardized_predicted_quantiles": _quantiles(
                predicted_standardized, PREDICTIVE_QUANTILES
            ),
            "log10rho_truth_quantiles": truth_log10rho_quantiles,
            "log10rho_predicted_quantiles": predicted_log10rho_quantiles,
            "delta_q99_999_log10rho_dex": predicted_log10rho_quantiles[3]
            - truth_log10rho_quantiles[3],
            "truth_mean_delta_squared": truth_probe_moment,
            "sampled_mean_delta_squared": sampled_probe_moment,
            "sampled_over_truth_mean_delta_squared": sampled_probe_moment
            / truth_probe_moment,
            "members_per_probed_voxel": PREDICTIVE_MEMBERS,
            "maximum_inverse_CDF_error": maximum_inverse_error,
        },
        "conditional_strata": {
            "backbone": _strata_summary(
                truth_probe["backbone_base"],
                truth_delta_squared,
                quadrature_delta_squared,
                probability_above_global,
                probability_upper_one_percent,
            ),
            "structure_risk": _strata_summary(
                truth_probe["risk"],
                truth_delta_squared,
                quadrature_delta_squared,
                probability_above_global,
                probability_upper_one_percent,
            ),
        },
    }


def _assert_finite_tree(value: Any, path: str = "result") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"V51 nonfinite result at {path}")


def audit(program_path: Path, repo: Path, output: Path) -> dict[str, Any]:
    program, v35, support, decision = load_program(program_path, repo.resolve())
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V51 audit requires a clean committed worktree")
    if socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V51 audit requires Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V51 audit requires the Lageunha Ada GPU")
    if output.exists():
        raise FileExistsError("V51 refuses existing output")
    device = torch.device("cuda")
    model, checkpoint = _load_model(program, repo.resolve(), commit)
    model = model.to(device).eval()
    frozen = program["frozen_inputs"]
    prepared = load_cache(
        Path(frozen["conditioning_cache"]),
        frozen["conditioning_cache_sha256"],
        str(checkpoint["code_commit"]),
    )
    domains: dict[str, Any] = {}
    try:
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            truth = _truth_probe(v35, prepared, domain, domain_index)
            domains[domain] = _audit_domain(
                model,
                device,
                v35,
                prepared,
                support,
                domain,
                domain_index,
                truth,
            )
    finally:
        prepared.close()
    if sum(row["total_native_voxels"] for row in domains.values()) != int(
        program["audit_population"]["total_native_voxels"]
    ):
        raise RuntimeError("V51 total train population differs")
    boundary_dominated_domains = [
        domain
        for domain, row in domains.items()
        if row["beyond_train_extrema"]["above_global_maximum"][
            "model_expected_native_voxels"
        ]
        >= MINIMUM_EXPECTED_BEYOND_GLOBAL_MAXIMUM
        and row["beyond_train_extrema"]["above_global_maximum"][
            "quadrature_rho_squared_fraction"
        ]
        >= DOMINANT_SECOND_MOMENT_FRACTION
    ]
    boundary_dominated = len(boundary_dominated_domains) >= MINIMUM_COMMON_DOMAINS
    ablation_improves_every_domain = all(
        decision["arms"]["structure_risk_ablation"]["domains"][domain][
            "mechanism_Q3_Q4"
        ]["generated_over_truth_mean_delta_squared"]
        < decision["arms"]["bounded_query_local_mixture_copula"]["domains"][domain][
            "mechanism_Q3_Q4"
        ]["generated_over_truth_mean_delta_squared"]
        for domain in DOMAIN_ORDER
    )
    risk_amplified_domains = [
        domain
        for domain, row in domains.items()
        if row["conditional_strata"]["structure_risk"]["strata"][
            "q99_9_and_above"
        ]["quadrature_over_truth_mean_delta_squared"]
        > RISK_TOP_STRATUM_MOMENT_RATIO_MAXIMUM
    ]
    risk_amplified = (
        len(risk_amplified_domains) >= MINIMUM_COMMON_DOMAINS
        and ablation_improves_every_domain
    )
    pit_key = str(PIT_CLASSIFICATION_TAIL)
    upper_overdispersed_domains = [
        domain
        for domain, row in domains.items()
        if row["PIT"][pit_key]["upper_observed_over_expected"] < PIT_RATIO_MINIMUM
    ]
    every_domain_extreme_failure = all(
        abs(row["posterior_predictive_probe"]["delta_q99_999_log10rho_dex"])
        > Q99_999_DEX_THRESHOLD
        or not (
            MOMENT_RATIO_MINIMUM
            <= row["bounded_quadrature_probe"][
                "primary_over_truth_mean_delta_squared"
            ]
            <= MOMENT_RATIO_MAXIMUM
        )
        for row in domains.values()
    )
    upper_tail_overdispersed = (
        len(upper_overdispersed_domains) >= MINIMUM_COMMON_DOMAINS
        and every_domain_extreme_failure
    )
    component_failure_domains = [
        domain
        for domain, row in domains.items()
        if row["global_effective_responsibility_components"]
        < EFFECTIVE_COMPONENT_THRESHOLD
        or bool(row["unsupported_component_indices"])
    ]
    component_failure = bool(component_failure_domains)
    quantile_calibrated = all(
        PIT_RATIO_MINIMUM
        <= row["PIT"][pit_key]["lower_observed_over_expected"]
        <= PIT_RATIO_MAXIMUM
        and PIT_RATIO_MINIMUM
        <= row["PIT"][pit_key]["upper_observed_over_expected"]
        <= PIT_RATIO_MAXIMUM
        and abs(row["posterior_predictive_probe"]["delta_q99_999_log10rho_dex"])
        <= Q99_999_DEX_THRESHOLD
        for row in domains.values()
    )
    moment_calibrated = all(
        MOMENT_RATIO_MINIMUM
        <= row["bounded_quadrature_probe"]["primary_over_truth_mean_delta_squared"]
        <= MOMENT_RATIO_MAXIMUM
        for row in domains.values()
    )
    moment_underconstrained = quantile_calibrated and not moment_calibrated
    train_tail_calibrated = quantile_calibrated and moment_calibrated
    classification, next_step = classify(
        boundary_dominated,
        risk_amplified,
        upper_tail_overdispersed,
        component_failure,
        moment_underconstrained,
        train_tail_calibrated,
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete_train_only_bounded_support_calibration_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "audit_code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "domains": domains,
        "branch_conditions": {
            "boundary_dominated_domains": boundary_dominated_domains,
            "train_unoccupied_support_margin_dominates_common_tail": boundary_dominated,
            "structure_risk_ablation_improves_development_Q4_every_domain": ablation_improves_every_domain,
            "risk_amplified_domains": risk_amplified_domains,
            "structure_risk_conditioning_amplifies_common_tail": risk_amplified,
            "upper_PIT_overdispersed_domains": upper_overdispersed_domains,
            "every_domain_posterior_predictive_or_moment_failure": every_domain_extreme_failure,
            "bounded_upper_tail_overdispersed": upper_tail_overdispersed,
            "component_failure_domains": component_failure_domains,
            "effective_component_failure": component_failure,
            "extreme_quantiles_calibrated": quantile_calibrated,
            "physical_second_moment_calibrated": moment_calibrated,
            "physical_second_moment_underconstrained": moment_underconstrained,
            "train_bounded_tail_calibrated": train_tail_calibrated,
        },
        "classification": classification,
        "next": next_step,
        "acknowledged_training_anomaly": {
            "logged_AMP_gradient_overflow_count": 1,
            "step": 8400,
            "sealed_training_report_literal_Infinity_occurrences": 1,
            "sealed_training_report_modified": False,
        },
        "training_or_refit_performed": False,
        "new_development_sample_generated": False,
        "validation_inputs_opened": False,
        "validation_truth_opened": False,
        "development_arrays_opened": False,
        "support_changed": False,
        "threshold_changed_after_diagnostic": False,
        "posthoc_scale_or_clipping_used": False,
        "posthoc_DC_or_Ak_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    _assert_finite_tree(result)
    result["decision_digest_sha256"] = canonical_digest(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, output)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit(args.program, args.repo, args.out)


if __name__ == "__main__":
    main()
