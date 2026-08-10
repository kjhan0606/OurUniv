#!/usr/bin/env python
"""Train-only fixed-threshold probability/amplitude decomposition for V54."""
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
from hong2021_v54_train import (
    CONTROL_QUADRATURE_ORDER,
    PRIMARY_QUADRATURE_ORDER,
    load_cache,
    load_program as load_v54_program,
)
from hong2021_v54_train_gate import _load_fit


PROGRAM_SHA256 = "b87b1f8f6626117cf11ab33b8ee23725a5007cbcc25b225abdec5ad5be17d97c"
PROGRAM_SCHEMA = "hong2021-v55-fixed-threshold-tail-amplitude-audit-program-v1"
SCHEMA = "hong2021-v55-fixed-threshold-tail-amplitude-audit-v1"
BIN_LABELS = (
    "at_or_below_q99",
    "q99_to_q99_9",
    "q99_9_to_q99_99",
    "q99_99_to_q99_999",
    "above_q99_999",
)
THRESHOLD_LABELS = ("q99", "q99_9", "q99_99", "q99_999")


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V55 {label} hash differs")
    return json.loads(path.read_text())


def _path(repo: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (repo / candidate).resolve()


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_audit_implementation_or_execution"
    ):
        raise ValueError("V55 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _path(repo, parent["v54_record"]), parent["v54_record_sha256"], "V54 result record"
    )
    decision = record.get("train_only_mechanism_decision", {})
    firewall = record.get("firewall", {})
    if (
        record.get("status") != parent["required_status"]
        or decision.get("classification") != parent["required_classification"]
        or decision.get("next") != parent["required_next"]
        or firewall.get("development_access") is not parent["required_development_accessed"]
        or firewall.get("independent_gate_locked") is not parent["required_independent_gate_locked"]
        or firewall.get("Astrid_accessed") is not False
        or firewall.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V55 parent conclusion or firewall differs")
    frozen = program["frozen_inputs"]
    for key, digest_key in (
        ("v54_program", "v54_program_sha256"),
        ("v54_checkpoint", "v54_checkpoint_sha256"),
        ("v54_training_report", "v54_training_report_sha256"),
        ("v54_threshold_selection", "v54_threshold_selection_sha256"),
        ("v54_preflight", "v54_preflight_sha256"),
        ("v54_train_gate", "v54_train_gate_sha256"),
        ("conditioning_cache", "conditioning_cache_sha256"),
        ("support_selection", "support_selection_sha256"),
    ):
        if sha256_file(_path(repo, frozen[key])) != frozen[digest_key]:
            raise ValueError(f"V55 {key} hash differs")
    report = json.loads(_path(repo, frozen["v54_training_report"]).read_text())
    gate = json.loads(_path(repo, frozen["v54_train_gate"]).read_text())
    if (
        canonical_digest(report) != frozen["v54_training_report_decision_digest_sha256"]
        or canonical_digest(gate) != frozen["v54_train_gate_decision_digest_sha256"]
        or gate.get("train_mechanism_pass") is not False
        or gate.get("development_accessed") is not False
        or gate.get("independent_gate_locked") is not True
    ):
        raise ValueError("V55 V54 report or gate digest differs")
    thresholds = json.loads(_path(repo, frozen["v54_threshold_selection"]).read_text())
    fixed = np.asarray(program["fixed_output_thresholds"]["values"], dtype=np.float64)
    if (
        fixed.shape != (4,)
        or not np.all(np.diff(fixed) > 0.0)
        or not np.array_equal(fixed, np.asarray(thresholds["common_log10rho_thresholds"]))
    ):
        raise ValueError("V55 fixed output thresholds differ")
    return program, record, gate


def _bin_indices(log10rho: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Bins implement threshold strictness: equality remains below the threshold."""
    return np.searchsorted(thresholds, log10rho, side="left")


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)


def _survival_probabilities(
    weights: torch.Tensor,
    locations: torch.Tensor,
    scales: torch.Tensor,
    base: torch.Tensor,
    target_std: float,
    thresholds: torch.Tensor,
) -> np.ndarray:
    standardized = (thresholds[:, None] / 4.5 - base.double()[None]) / float(target_std)
    result = torch.empty_like(standardized)
    below = standardized <= LOWER_SUPPORT
    above = standardized >= LOWER_SUPPORT + SUPPORT_RANGE
    interior = ~(below | above)
    result[below] = 1.0
    result[above] = 0.0
    for threshold_index in range(len(thresholds)):
        mask = interior[threshold_index]
        if bool(mask.any()):
            coordinate = (standardized[threshold_index, mask] - LOWER_SUPPORT) / SUPPORT_RANGE
            latent = torch.log(coordinate) - torch.log1p(-coordinate)
            component_cdf = torch.special.ndtr(
                (latent[None] - locations[:, mask].double()) / scales[:, mask].double()
            )
            cdf = torch.sum(weights[:, mask].double() * component_cdf, dim=0)
            result[threshold_index, mask] = 1.0 - cdf
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
) -> dict[str, np.ndarray | float]:
    normalized = quadrature_weights.double() / math.sqrt(math.pi)
    moment_bins = torch.zeros(5, dtype=torch.float64, device=weights.device)
    probability_bins = torch.zeros_like(moment_bins)
    total_moment = torch.zeros((), dtype=torch.float64, device=weights.device)
    for component in range(weights.shape[0]):
        latent = locations[component].double()[:, None] + math.sqrt(2.0) * (
            scales[component].double()[:, None] * nodes.double()[None]
        )
        standardized = LOWER_SUPPORT + SUPPORT_RANGE * torch.sigmoid(latent)
        physical_y = base.double()[:, None] + float(target_std) * standardized
        log10rho = 4.5 * physical_y
        rho = torch.exp(math.log(10.0) * log10rho)
        delta_squared = torch.square(rho - 1.0)
        mass = weights[component].double()[:, None] * normalized[None]
        total_moment += torch.sum(mass * delta_squared)
        bins = torch.bucketize(log10rho, thresholds.double(), right=False)
        for bin_index in range(5):
            mask = bins == bin_index
            probability_bins[bin_index] += torch.sum(mass * mask)
            moment_bins[bin_index] += torch.sum(mass * delta_squared * mask)
    return {
        "moment_bins": moment_bins.cpu().numpy(),
        "probability_bins": probability_bins.cpu().numpy(),
        "total_moment": float(total_moment.cpu()),
    }


def classify(
    numerical_requirements_pass: bool,
    amplitude_dominates: dict[str, bool],
    probability_dominates: dict[str, bool],
) -> tuple[str, str]:
    if not numerical_requirements_pass:
        return (
            "fixed_threshold_tail_decomposition_is_numerically_or_empirically_unresolved",
            "freeze_a_higher_accuracy_train_only_tail_integration_audit_without_training_or_development_access",
        )
    if all(amplitude_dominates.get(domain, False) for domain in DOMAIN_ORDER):
        return (
            "V54_probability_score_leaves_beyond_highest_threshold_amplitudes_unconstrained",
            "freeze_one_matched_train_only_model_with_a_proper_survival_score_grid_spanning_q99_999_to_the_immutable_train_maximum_while_retaining_the_unchanged_bounded_NLL",
        )
    if all(probability_dominates.get(domain, False) for domain in DOMAIN_ORDER):
        return (
            "V54_highest_fixed_threshold_exceedance_probability_remains_miscalibrated",
            "freeze_one_matched_train_only_model_that_changes_only_the_predeclared_proper_tail_score_weighting_and_reuses_the_same_thresholds",
        )
    return (
        "V54_tail_failure_is_mixed_across_probability_amplitude_or_domain",
        "seal_the_domainwise_fixed_bin_decomposition_before_selecting_any_further_model",
    )


def _domain_summary(
    truth_log10rho: np.ndarray,
    truth_delta_squared: np.ndarray,
    predicted_probability: np.ndarray,
    primary_moment_bins: np.ndarray,
    primary_probability_bins: np.ndarray,
    primary_total_moment: float,
    control_moment_bins: np.ndarray,
    control_total_moment: float,
    thresholds: np.ndarray,
    sealed: dict[str, Any],
    numerics: dict[str, Any],
) -> dict[str, Any]:
    count = len(truth_log10rho)
    truth_bins = _bin_indices(truth_log10rho, thresholds)
    truth_counts = np.bincount(truth_bins, minlength=5).astype(np.int64)
    truth_moment_sums = np.bincount(
        truth_bins, weights=truth_delta_squared, minlength=5
    ).astype(np.float64)
    truth_total = float(np.sum(truth_delta_squared, dtype=np.float64))
    if not (
        predicted_probability.shape == (4, count)
        and primary_moment_bins.shape == (5,)
        and control_moment_bins.shape == (5,)
    ):
        raise ValueError("V55 domain accumulator shape differs")
    truth_mean = truth_total / count
    predicted_mean = primary_total_moment / count
    sealed_truth = float(sealed["strata"]["q99_9_and_above"]["truth_mean_delta_squared"])
    sealed_predicted = float(
        sealed["strata"]["q99_9_and_above"]["V54_quadrature_mean_delta_squared"]
    )
    reproduction = {
        "truth_relative_difference_from_V54_gate": _relative_error(truth_mean, sealed_truth),
        "predicted_relative_difference_from_V54_gate": _relative_error(
            predicted_mean, sealed_predicted
        ),
    }
    truth_bin_means = truth_moment_sums / count
    primary_bin_means = primary_moment_bins / count
    control_bin_means = control_moment_bins / count
    positive_excess = np.maximum(primary_bin_means - truth_bin_means, 0.0)
    positive_sum = float(positive_excess.sum(dtype=np.float64))
    positive_shares = (
        positive_excess / positive_sum if positive_sum > 0.0 else np.zeros(5, dtype=np.float64)
    )
    bins: dict[str, Any] = {}
    for index, label in enumerate(BIN_LABELS):
        bins[label] = {
            "truth_count": int(truth_counts[index]),
            "truth_probability": float(truth_counts[index] / count),
            "truth_mean_delta_squared_contribution": float(truth_bin_means[index]),
            "predicted_quadrature_probability_64": float(primary_probability_bins[index] / count),
            "predicted_mean_delta_squared_contribution_64": float(primary_bin_means[index]),
            "predicted_mean_delta_squared_contribution_32": float(control_bin_means[index]),
            "positive_excess_share": float(positive_shares[index]),
        }
    threshold_rows: dict[str, Any] = {}
    empirical_support_pass = True
    tail_convergence_pass = True
    identity_pass = True
    for threshold_index, label in enumerate(THRESHOLD_LABELS):
        tail_slice = slice(threshold_index + 1, None)
        truth_tail_count = int(truth_counts[tail_slice].sum())
        truth_probability = truth_tail_count / count
        analytic_probability = float(np.mean(predicted_probability[threshold_index], dtype=np.float64))
        quadrature_probability = float(primary_probability_bins[tail_slice].sum() / count)
        truth_tail_moment = float(truth_moment_sums[tail_slice].sum() / count)
        primary_tail_moment = float(primary_moment_bins[tail_slice].sum() / count)
        control_tail_moment = float(control_moment_bins[tail_slice].sum() / count)
        empirical_pass = truth_tail_count >= int(
            numerics["minimum_empirical_exceedance_count_for_classification"]
        )
        values_positive = bool(
            min(
                truth_probability,
                analytic_probability,
                truth_tail_moment,
                primary_tail_moment,
            )
            > 0.0
        )
        empirical_support_pass = bool(
            empirical_support_pass and empirical_pass and values_positive
        )
        if values_positive:
            truth_conditional = truth_tail_moment / truth_probability
            predicted_conditional = primary_tail_moment / analytic_probability
            probability_ratio = analytic_probability / truth_probability
            conditional_ratio = predicted_conditional / truth_conditional
            tail_ratio = primary_tail_moment / truth_tail_moment
            log_probability = math.log(probability_ratio)
            log_conditional = math.log(conditional_ratio)
            log_tail = math.log(tail_ratio)
            identity_error = abs(log_tail - log_probability - log_conditional)
        else:
            truth_conditional = predicted_conditional = None
            probability_ratio = conditional_ratio = tail_ratio = None
            log_probability = log_conditional = log_tail = None
            identity_error = math.inf
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
        identity_pass = bool(
            identity_pass
            and identity_error
            <= float(numerics["maximum_log_ratio_identity_absolute_error"])
        )
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
            "empirical_support_pass": empirical_pass,
        }
    complete_convergence = _relative_difference(primary_total_moment, control_total_moment)
    partition_error = max(
        _relative_error(float(primary_moment_bins.sum()), primary_total_moment),
        _relative_error(float(truth_moment_sums.sum()), truth_total),
    )
    reproduction_pass = max(reproduction.values()) <= float(
        numerics["maximum_complete_moment_relative_difference_from_v54_gate"]
    )
    numerical_pass = bool(
        reproduction_pass
        and partition_error <= float(numerics["maximum_bin_partition_relative_error"])
        and complete_convergence
        <= float(numerics["maximum_32_to_64_complete_moment_relative_difference"])
        and tail_convergence_pass
        and identity_pass
        and empirical_support_pass
    )
    highest = threshold_rows["q99_999"]
    amplitude_dominates = bool(
        numerical_pass
        and positive_shares[-1] >= 0.5
        and abs(float(highest["log_conditional_amplitude_ratio"]))
        > abs(float(highest["log_probability_ratio"]))
    )
    probability_dominates = bool(
        numerical_pass
        and abs(float(highest["log_probability_ratio"]))
        > abs(float(highest["log_conditional_amplitude_ratio"]))
    )
    return {
        "top_backbone_probe_voxels": count,
        "complete_moment": {
            "truth_mean_delta_squared": truth_mean,
            "predicted_mean_delta_squared_64": predicted_mean,
            "predicted_mean_delta_squared_32": control_total_moment / count,
            "predicted_over_truth": predicted_mean / truth_mean,
            "quadrature_32_to_64_relative_difference": complete_convergence,
            **reproduction,
        },
        "fixed_output_bins": bins,
        "threshold_decomposition": threshold_rows,
        "bin_partition_relative_error": partition_error,
        "positive_excess_sum": positive_sum,
        "empirical_support_pass": empirical_support_pass,
        "tail_quadrature_convergence_pass": tail_convergence_pass,
        "log_ratio_identity_pass": identity_pass,
        "reproduces_V54_gate": reproduction_pass,
        "numerical_requirements_pass": numerical_pass,
        "beyond_q99_999_amplitude_dominates": amplitude_dominates,
        "q99_999_probability_dominates": probability_dominates,
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
    numerics: dict[str, Any],
) -> dict[str, Any]:
    row = v35["development_domains"][domain]
    objects = int(row["train_objects"])
    if objects != EXPECTED_OBJECTS[domain]:
        raise RuntimeError("V55 train object count differs")
    truth_probe = _truth_probe(v35, prepared, domain, domain_index)
    boundary = float(gate["domains"][domain]["backbone_boundaries"][2])
    top = truth_probe["backbone_base"].astype(np.float64) >= boundary
    expected_top = int(gate["domains"][domain]["strata"]["q99_9_and_above"]["count"])
    if int(top.sum()) != expected_top:
        raise ValueError("V55 top-backbone probe differs")
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
    probability_parts: list[np.ndarray] = []
    primary_moment_bins = np.zeros(5, dtype=np.float64)
    primary_probability_bins = np.zeros(5, dtype=np.float64)
    control_moment_bins = np.zeros(5, dtype=np.float64)
    primary_total = control_total = 0.0
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
            local_top = top[start : start + PROBE_VOXELS]
            selected = indices[local_top]
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
                base = torch.from_numpy(
                    backbone.reshape(-1)[selected].astype(np.float64) + target_mean
                ).to(device)
                probability_parts.append(
                    _survival_probabilities(
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
                primary_moment_bins += np.asarray(primary["moment_bins"])
                primary_probability_bins += np.asarray(primary["probability_bins"])
                primary_total += float(primary["total_moment"])
                control_moment_bins += np.asarray(control["moment_bins"])
                control_total += float(control["total_moment"])
            if (object_index + 1) % 16 == 0 or object_index + 1 == objects:
                print(f"[v55-audit] {domain} {object_index + 1}/{objects}", flush=True)
    finally:
        data.close()
        cache.close()
    predicted_probability = np.concatenate(probability_parts, axis=1)
    if predicted_probability.shape != (4, expected_top):
        raise RuntimeError("V55 predicted probability probe differs")
    return _domain_summary(
        truth_log10rho,
        truth_delta_squared,
        predicted_probability,
        primary_moment_bins,
        primary_probability_bins,
        primary_total,
        control_moment_bins,
        control_total,
        thresholds,
        gate["domains"][domain],
        numerics,
    )


def audit(program_path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, _, gate = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if not clean or socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V55 audit requires clean Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V55 audit requires Ada")
    frozen = program["frozen_inputs"]
    v54_program_path = _path(repo, frozen["v54_program"])
    v54_program, v35, _ = load_v54_program(v54_program_path, repo)
    development = Path(v54_program["output_roots"]["development"])
    if development.exists():
        raise RuntimeError("V55 refuses a pre-existing V54 development directory")
    model, checkpoint = _load_fit(
        v54_program,
        _path(repo, frozen["v54_checkpoint"]),
        frozen["v54_checkpoint_sha256"],
        _path(repo, frozen["v54_training_report"]),
        frozen["v54_training_report_sha256"],
        _path(repo, frozen["v54_threshold_selection"]),
        frozen["v54_threshold_selection_sha256"],
        _path(repo, frozen["v54_preflight"]),
        frozen["v54_preflight_sha256"],
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
    thresholds = np.asarray(program["fixed_output_thresholds"]["values"], dtype=np.float64)
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
                program["numerics"],
            )
    finally:
        prepared.close()
    numerical_pass = all(row["numerical_requirements_pass"] for row in domains.values())
    amplitude = {
        domain: bool(row["beyond_q99_999_amplitude_dominates"])
        for domain, row in domains.items()
    }
    probability = {
        domain: bool(row["q99_999_probability_dominates"])
        for domain, row in domains.items()
    }
    classification, next_action = classify(numerical_pass, amplitude, probability)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_train_only_fixed_threshold_tail_amplitude_audit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "checkpoint_sha256": frozen["v54_checkpoint_sha256"],
        "training_report_sha256": frozen["v54_training_report_sha256"],
        "threshold_selection_sha256": frozen["v54_threshold_selection_sha256"],
        "v54_train_gate_sha256": frozen["v54_train_gate_sha256"],
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
        raise FileExistsError("V55 refuses an existing audit")
    result = audit(args.program, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
