#!/usr/bin/env python
"""V49 train-only audit of V48 Gaussian extreme calibration and physical moments."""
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
    _truth_probe,
)
from hong2021_v48_network import (
    LocalMixtureUNet,
    gaussian_mixture_cdf,
    gaussian_mixture_inverse,
    gaussian_mixture_log_probability,
    mixture_parameters,
    parameter_count,
)
from hong2021_v48_train import (
    CHECKPOINT_SCHEMA,
    PARAMETERS,
    PROGRAM_SHA256 as V48_PROGRAM_SHA256,
    condition_cube,
    load_cache,
    load_program as load_v48_program,
)


PROGRAM_SCHEMA = "hong2021-v49-train-only-Gaussian-extreme-calibration-audit-program-v1"
PROGRAM_SHA256 = "7595330346f0a1fcc8a195f99ca8df805b47879b1d6cad1f9e3a9e85559b6c5a"
RESULT_SCHEMA = "hong2021-v49-train-only-Gaussian-extreme-calibration-audit-v1"
PIT_TAILS = (0.01, 0.001, 0.0001, 0.00001)
TAIL_TARGET_QUANTILES = (0.99, 0.999)
PREDICTIVE_QUANTILES = (0.99, 0.999, 0.9999, 0.99999)
SUMMARY_QUANTILES = (0.5, 0.9, 0.99, 0.999, 1.0)
PREDICTIVE_MEMBERS = 16
PREDICTIVE_SEED = 146049
GAUSSIAN_Q99_999 = 4.264890793922825
WEIGHT_THRESHOLD = 0.02
RESPONSIBILITY_RATIO_THRESHOLD = 0.5
DOMINANT_MOMENT_FRACTION = 0.5
EFFECTIVE_COMPONENT_THRESHOLD = 3.0
PIT_RATIO_MINIMUM = 0.5
PIT_RATIO_MAXIMUM = 2.0
Q99_999_DEX_THRESHOLD = 0.10
MOMENT_RATIO_MINIMUM = 2.0 / 3.0
MOMENT_RATIO_MAXIMUM = 1.5
INVERSE_ERROR_MAXIMUM = 2.0e-6


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V49 {label} hash differs")
    return json.loads(path.read_text())


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_audit_implementation_or_execution"
    ):
        raise ValueError("V49 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        (repo / parent["v48_record"]).resolve(),
        parent["v48_record_sha256"],
        "V48 result record",
    )
    decision = record.get("development_decision", {})
    if (
        decision.get("classification") != parent["required_classification"]
        or decision.get("next") != parent["required_next"]
        or decision.get("development_pass") is not parent["required_development_pass"]
        or decision.get("candidate_high_k_power_and_residual_RMS_all_domains")
        is not parent["required_high_k_power_and_residual_RMS_all_domains"]
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
        or record.get("firewall", {}).get("independent_gate_locked") is not True
    ):
        raise ValueError("V49 V48 conclusion or firewall differs")
    frozen = program["frozen_inputs"]
    for key, digest_key in (
        ("v48_program", "v48_program_sha256"),
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
            raise ValueError(f"V49 frozen {key} hash differs")
    _, v35, _ = load_v48_program((repo / frozen["v48_program"]).resolve(), repo)
    return program, v35


def classify(
    unsupported_component: bool,
    globally_overdispersed: bool,
    component_collapse: bool,
    moment_underconstrained: bool,
    train_tail_calibrated: bool,
) -> tuple[str, str]:
    spline = (
        "freeze_a_train_only_monotone_bounded_support_conditional_spline_likelihood_"
        "with_support_and_gates_fixed_before_fit"
    )
    if unsupported_component:
        return (
            "unsupported_Gaussian_component_mass_dominates_the_physical_train_tail",
            spline,
        )
    if globally_overdispersed:
        return (
            "Gaussian_mixture_is_globally_overdispersed_in_the_train_extreme_tail",
            spline,
        )
    if component_collapse:
        return (
            "V48_Gaussian_mixture_has_effective_component_collapse",
            "stop_finite_mixtures_and_freeze_a_train_only_monotone_bounded_support_conditional_spline_likelihood",
        )
    if moment_underconstrained:
        return (
            "voxel_log_score_calibrates_extreme_quantiles_but_not_the_physical_second_moment",
            "freeze_a_train_only_tail_weighted_strictly_proper_scoring_audit_before_changing_the_generator",
        )
    if train_tail_calibrated:
        return (
            "train_Gaussian_tail_is_calibrated_but_the_empirical_rank_copula_breaks_development_extremes",
            "audit_only_the_train_empirical_conditional_rank_tail_and_query_coupling_without_changing_the_Gaussian_likelihood",
        )
    return (
        "Gaussian_train_extreme_failure_is_mixed_or_not_identified",
        "seal_the_PIT_component_and_analytic_moment_evidence_before_selecting_one_new_likelihood",
    )


def _load_model(
    program: dict[str, Any], repo: Path, commit: str
) -> tuple[LocalMixtureUNet, dict[str, Any]]:
    frozen = program["frozen_inputs"]
    checkpoint = torch.load(frozen["checkpoint"], map_location="cpu", weights_only=False)
    source_commit = str(checkpoint.get("code_commit"))
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("program_sha256") != V48_PROGRAM_SHA256
        or checkpoint.get("step") != 12_000
        or checkpoint.get("parameters") != PARAMETERS
        or checkpoint.get("conditioning_cache_sha256")
        != frozen["conditioning_cache_sha256"]
        or checkpoint.get("spatial_rank_transport") is not False
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
        raise ValueError("V49 V48 checkpoint binding differs")
    model = LocalMixtureUNet()
    if parameter_count(model) != PARAMETERS:
        raise RuntimeError("V49 V48 parameter count differs")
    model.load_state_dict(checkpoint["ema_state_dict"])
    return model, checkpoint


def _gaussian_component_moments(
    weights: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
    base: torch.Tensor,
    target_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted component E[rho], E[rho^2], and E[(rho-1)^2]."""
    value_weights = weights.double()
    value_locations = locations.double()
    value_scales = scales.double()
    value_base = base.double()
    density_coefficient = 4.5 * math.log(10.0)
    residual_coefficient = density_coefficient * float(target_std)
    log_first = (
        density_coefficient * value_base
        + residual_coefficient * value_locations
        + 0.5 * residual_coefficient**2 * torch.square(value_scales)
    )
    log_second = (
        2.0 * density_coefficient * value_base
        + 2.0 * residual_coefficient * value_locations
        + 2.0 * residual_coefficient**2 * torch.square(value_scales)
    )
    first = value_weights * torch.exp(log_first)
    second = value_weights * torch.exp(log_second)
    delta_squared = second - 2.0 * first + value_weights
    if (
        not torch.isfinite(first).all()
        or not torch.isfinite(second).all()
        or not torch.isfinite(delta_squared).all()
        or float(delta_squared.min().cpu()) < -1.0e-10
    ):
        raise RuntimeError("V49 analytic Gaussian physical moment differs")
    return first, second, delta_squared


def _physical_delta_squared(y: np.ndarray) -> np.ndarray:
    value = np.power(10.0, 4.5 * y.astype(np.float64)) - 1.0
    result = np.square(value)
    if not np.isfinite(result).all():
        raise RuntimeError("V49 empirical physical moment differs")
    return result


@torch.inference_mode()
def _audit_domain(
    model: LocalMixtureUNet,
    device: torch.device,
    v35: dict[str, Any],
    prepared: h5py.File,
    domain: str,
    domain_index: int,
    truth_probe: dict[str, np.ndarray],
) -> dict[str, Any]:
    row = v35["development_domains"][domain]
    objects = int(row["train_objects"])
    if objects != EXPECTED_OBJECTS[domain]:
        raise RuntimeError("V49 train object count differs")
    total_voxels = objects * 64**3
    thresholds = np.quantile(
        truth_probe["standardized"].astype(np.float64), TAIL_TARGET_QUANTILES
    )
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    nll_sum = 0.0
    tail_nll_sum = np.zeros(2, dtype=np.float64)
    tail_count = np.zeros(2, dtype=np.int64)
    pit_lower = np.zeros(len(PIT_TAILS), dtype=np.int64)
    pit_upper = np.zeros(len(PIT_TAILS), dtype=np.int64)
    weight_sum = np.zeros(5, dtype=np.float64)
    responsibility_sum = np.zeros(5, dtype=np.float64)
    tail_responsibility_sum = np.zeros((2, 5), dtype=np.float64)
    analytic_first_sum = np.zeros(5, dtype=np.float64)
    analytic_second_sum = np.zeros(5, dtype=np.float64)
    analytic_delta_squared_sum = np.zeros(5, dtype=np.float64)
    empirical_delta_squared_sum = 0.0
    sampled: dict[str, list[list[np.ndarray]]] = {
        name: [[] for _ in range(5)]
        for name in (
            "weight",
            "responsibility",
            "location",
            "scale",
            "upper_q99_999",
            "log10_weighted_rho_squared",
        )
    }
    probe_analytic_delta_squared: list[np.ndarray] = []
    predictive_standardized: list[np.ndarray] = []
    predictive_y: list[np.ndarray] = []
    maximum_inverse_error = 0.0
    data, cache = _open_split(row, "train")
    try:
        for object_index in range(objects):
            condition, target, backbone = condition_cube(
                data, cache, prepared, domain, "train", object_index
            )
            parameter = model(torch.from_numpy(condition[None]).to(device)).float()
            observed = torch.from_numpy(target[None]).to(device).float()
            log_probability = gaussian_mixture_log_probability(parameter, observed)
            cdf = gaussian_mixture_cdf(parameter, observed)
            logits, locations, scales = mixture_parameters(parameter)
            weights = torch.softmax(logits, dim=1)
            standardized = (observed - locations) / scales
            component_log_probability = (
                -0.5 * torch.square(standardized)
                - torch.log(scales)
                - 0.5 * math.log(2.0 * math.pi)
            )
            responsibilities = torch.softmax(
                F.log_softmax(logits, dim=1) + component_log_probability, dim=1
            )
            if (
                float(torch.max(torch.abs(weights.sum(dim=1) - 1.0)).cpu()) > 1.0e-5
                or float(
                    torch.max(torch.abs(responsibilities.sum(dim=1) - 1.0)).cpu()
                )
                > 1.0e-5
                or not torch.isfinite(parameter).all()
                or not torch.isfinite(log_probability).all()
                or not torch.isfinite(cdf).all()
            ):
                raise RuntimeError("V49 Gaussian likelihood integrity differs")
            nll = -log_probability
            nll_sum += float(nll.double().sum().cpu())
            flat_target = observed.reshape(-1)
            for tail_index, threshold in enumerate(thresholds):
                mask = flat_target >= float(threshold)
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
            weight_sum += weights.double().sum(dim=(0, 2, 3, 4)).cpu().numpy()
            responsibility_sum += (
                responsibilities.double().sum(dim=(0, 2, 3, 4)).cpu().numpy()
            )
            base = torch.from_numpy(backbone[None]).to(device).double() + target_mean
            first, second, delta_squared = _gaussian_component_moments(
                weights, locations, scales, base, target_std
            )
            analytic_first_sum += first.sum(dim=(0, 2, 3, 4)).cpu().numpy()
            analytic_second_sum += second.sum(dim=(0, 2, 3, 4)).cpu().numpy()
            analytic_delta_squared_sum += (
                delta_squared.sum(dim=(0, 2, 3, 4)).cpu().numpy()
            )
            truth_y = (
                backbone.reshape(-1).astype(np.float64)
                + target_mean
                + target_std * target.reshape(-1).astype(np.float64)
            )
            empirical_delta_squared_sum += float(_physical_delta_squared(truth_y).sum())

            indices = _probe_indices(domain_index, object_index)
            index_tensor = torch.from_numpy(indices).to(device)
            flat_parameter = (
                parameter.reshape(1, 15, -1)
                .index_select(2, index_tensor)
                .reshape(1, 15, 1, 1, -1)
            )
            flat_responsibility = responsibilities.reshape(1, 5, -1).index_select(
                2, index_tensor
            )
            flat_second = second.reshape(1, 5, -1).index_select(2, index_tensor)
            flat_delta_squared = delta_squared.reshape(1, 5, -1).index_select(
                2, index_tensor
            )
            probe_analytic_delta_squared.append(
                flat_delta_squared.sum(dim=1).reshape(-1).cpu().numpy()
            )
            probe_logits, probe_locations, probe_scales = mixture_parameters(flat_parameter)
            probe_weights = torch.softmax(probe_logits, dim=1)
            endpoint = probe_locations + GAUSSIAN_Q99_999 * probe_scales
            for component in range(5):
                sampled["weight"][component].append(
                    probe_weights[0, component].cpu().numpy().reshape(-1)
                )
                sampled["responsibility"][component].append(
                    flat_responsibility[0, component].cpu().numpy().reshape(-1)
                )
                sampled["location"][component].append(
                    probe_locations[0, component].cpu().numpy().reshape(-1)
                )
                sampled["scale"][component].append(
                    probe_scales[0, component].cpu().numpy().reshape(-1)
                )
                sampled["upper_q99_999"][component].append(
                    endpoint[0, component].cpu().numpy().reshape(-1)
                )
                sampled["log10_weighted_rho_squared"][component].append(
                    torch.log10(flat_second[0, component])
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
                draw = gaussian_mixture_inverse(flat_parameter, rank_tensor)
                error = float(
                    torch.max(
                        torch.abs(
                            gaussian_mixture_cdf(flat_parameter, draw) - rank_tensor
                        )
                    ).cpu()
                )
                maximum_inverse_error = max(maximum_inverse_error, error)
                draw_numpy = draw.cpu().numpy().reshape(-1)
                predictive_standardized.append(draw_numpy.astype(np.float32))
                residual = draw_numpy * target_std + target_mean
                predictive_y.append((flat_backbone + residual).astype(np.float32))
            if (object_index + 1) % 16 == 0 or object_index + 1 == objects:
                print(f"[v49-audit] {domain} {object_index + 1}/{objects}", flush=True)
    finally:
        data.close()
        cache.close()

    if tail_count.min() <= 0 or maximum_inverse_error > INVERSE_ERROR_MAXIMUM:
        raise RuntimeError("V49 tail count or inverse error differs")
    mean_weight = weight_sum / total_voxels
    mean_responsibility = responsibility_sum / total_voxels
    responsibility_ratio = mean_responsibility / mean_weight
    analytic_first = analytic_first_sum / total_voxels
    analytic_second = analytic_second_sum / total_voxels
    analytic_delta_squared = analytic_delta_squared_sum / total_voxels
    analytic_delta_squared_total = float(analytic_delta_squared.sum())
    empirical_delta_squared = float(empirical_delta_squared_sum / total_voxels)
    second_fraction = analytic_second / analytic_second.sum()
    delta_squared_fraction = analytic_delta_squared / analytic_delta_squared.sum()
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
    sampled_summary = {
        name: [
            {
                "quantiles": _quantiles(
                    np.concatenate(component_values), SUMMARY_QUANTILES
                )
            }
            for component_values in by_component
        ]
        for name, by_component in sampled.items()
    }
    predicted_standardized = np.concatenate(predictive_standardized)
    predicted_y = np.concatenate(predictive_y)
    truth_standardized = truth_probe["standardized"]
    truth_y = truth_probe["physical_y"]
    standardized_truth_quantiles = _quantiles(
        truth_standardized, PREDICTIVE_QUANTILES
    )
    standardized_predicted_quantiles = _quantiles(
        predicted_standardized, PREDICTIVE_QUANTILES
    )
    log10rho_truth_quantiles = _quantiles(4.5 * truth_y, PREDICTIVE_QUANTILES)
    log10rho_predicted_quantiles = _quantiles(
        4.5 * predicted_y, PREDICTIVE_QUANTILES
    )
    probe_truth_moment = float(_physical_delta_squared(truth_y).mean())
    probe_sampled_moment = float(_physical_delta_squared(predicted_y).mean())
    probe_analytic_moment = float(np.concatenate(probe_analytic_delta_squared).mean())
    unsupported = [
        component
        for component in range(5)
        if mean_weight[component] >= WEIGHT_THRESHOLD
        and responsibility_ratio[component] < RESPONSIBILITY_RATIO_THRESHOLD
        and second_fraction[component] >= DOMINANT_MOMENT_FRACTION
    ]
    return {
        "train_objects": objects,
        "total_native_voxels": total_voxels,
        "probe_voxels": int(len(truth_standardized)),
        "posterior_predictive_values": int(len(predicted_y)),
        "mean_NLL": float(nll_sum / total_voxels),
        "upper_tail_NLL": {
            "q99": float(tail_nll_sum[0] / tail_count[0]),
            "q99_9": float(tail_nll_sum[1] / tail_count[1]),
        },
        "PIT": pit,
        "mean_mixture_weight": mean_weight.tolist(),
        "mean_posterior_responsibility": mean_responsibility.tolist(),
        "responsibility_to_weight_ratio": responsibility_ratio.tolist(),
        "upper_tail_mean_posterior_responsibility": {
            "q99": (tail_responsibility_sum[0] / tail_count[0]).tolist(),
            "q99_9": (tail_responsibility_sum[1] / tail_count[1]).tolist(),
        },
        "global_effective_responsibility_components": _effective(mean_responsibility),
        "sampled_component_summaries": sampled_summary,
        "analytic_native_physical_moments": {
            "mean_rho_by_component": analytic_first.tolist(),
            "mean_rho_squared_by_component": analytic_second.tolist(),
            "mean_delta_squared_by_component": analytic_delta_squared.tolist(),
            "rho_squared_component_fraction": second_fraction.tolist(),
            "delta_squared_component_fraction": delta_squared_fraction.tolist(),
            "predicted_mean_delta_squared": analytic_delta_squared_total,
            "truth_mean_delta_squared": empirical_delta_squared,
            "predicted_over_truth_mean_delta_squared": float(
                analytic_delta_squared_total / empirical_delta_squared
            ),
        },
        "posterior_predictive_probe": {
            "standardized_truth_quantiles": standardized_truth_quantiles,
            "standardized_predicted_quantiles": standardized_predicted_quantiles,
            "log10rho_truth_quantiles": log10rho_truth_quantiles,
            "log10rho_predicted_quantiles": log10rho_predicted_quantiles,
            "delta_q99_999_log10rho_dex": float(
                log10rho_predicted_quantiles[3] - log10rho_truth_quantiles[3]
            ),
            "truth_mean_delta_squared": probe_truth_moment,
            "sampled_mean_delta_squared": probe_sampled_moment,
            "sampled_over_truth_mean_delta_squared": float(
                probe_sampled_moment / probe_truth_moment
            ),
            "analytic_mean_delta_squared": probe_analytic_moment,
            "analytic_over_truth_mean_delta_squared": float(
                probe_analytic_moment / probe_truth_moment
            ),
            "members_per_probed_voxel": PREDICTIVE_MEMBERS,
            "maximum_inverse_CDF_error": maximum_inverse_error,
        },
        "unsupported_component_indices": unsupported,
    }


def audit(program_path: Path, repo: Path, output: Path) -> dict[str, Any]:
    program, v35 = load_program(program_path, repo.resolve())
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V49 audit requires a clean committed worktree")
    if socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V49 audit requires Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V49 audit requires the Lageunha Ada GPU")
    if output.exists():
        raise FileExistsError("V49 refuses existing output")
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
                model, device, v35, prepared, domain, domain_index, truth
            )
    finally:
        prepared.close()
    unsupported = any(row["unsupported_component_indices"] for row in domains.values())
    overdispersed = all(
        row["posterior_predictive_probe"]["delta_q99_999_log10rho_dex"]
        > Q99_999_DEX_THRESHOLD
        or row["analytic_native_physical_moments"][
            "predicted_over_truth_mean_delta_squared"
        ]
        > MOMENT_RATIO_MAXIMUM
        for row in domains.values()
    ) and sum(
        row["PIT"][str(PIT_TAILS[-1])]["upper_observed_over_expected"]
        < PIT_RATIO_MINIMUM
        for row in domains.values()
    ) >= 2
    collapse = any(
        row["global_effective_responsibility_components"]
        < EFFECTIVE_COMPONENT_THRESHOLD
        for row in domains.values()
    )
    quantile_calibrated = all(
        PIT_RATIO_MINIMUM
        <= row["PIT"][str(PIT_TAILS[-1])]["lower_observed_over_expected"]
        <= PIT_RATIO_MAXIMUM
        and PIT_RATIO_MINIMUM
        <= row["PIT"][str(PIT_TAILS[-1])]["upper_observed_over_expected"]
        <= PIT_RATIO_MAXIMUM
        and abs(
            row["posterior_predictive_probe"]["delta_q99_999_log10rho_dex"]
        )
        <= Q99_999_DEX_THRESHOLD
        for row in domains.values()
    )
    moment_calibrated = all(
        MOMENT_RATIO_MINIMUM
        <= row["analytic_native_physical_moments"][
            "predicted_over_truth_mean_delta_squared"
        ]
        <= MOMENT_RATIO_MAXIMUM
        for row in domains.values()
    )
    moment_underconstrained = quantile_calibrated and not moment_calibrated
    train_tail_calibrated = quantile_calibrated and moment_calibrated
    classification, next_step = classify(
        unsupported,
        overdispersed,
        collapse,
        moment_underconstrained,
        train_tail_calibrated,
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete_train_only_Gaussian_extreme_calibration_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "audit_code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "domains": domains,
        "branch_conditions": {
            "unsupported_component_mass": unsupported,
            "globally_overdispersed_train_extreme_tail": overdispersed,
            "component_collapse": collapse,
            "extreme_quantiles_calibrated": quantile_calibrated,
            "physical_second_moment_calibrated": moment_calibrated,
            "physical_second_moment_underconstrained": moment_underconstrained,
            "train_tail_calibrated": train_tail_calibrated,
        },
        "classification": classification,
        "next": next_step,
        "training_or_refit_performed": False,
        "new_development_sample_generated": False,
        "validation_inputs_opened": False,
        "validation_truth_opened": False,
        "development_arrays_opened": False,
        "threshold_changed_after_diagnostic": False,
        "posthoc_scale_or_clipping_used": False,
        "posthoc_DC_or_Ak_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(result, indent=2), flush=True)
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
