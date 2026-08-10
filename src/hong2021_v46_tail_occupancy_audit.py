#!/usr/bin/env python
"""Train-only V46 audit of V45 mixture tails and component occupancy."""
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
from hong2021_v35_spectrum_phase import _backbone, _open_split
from hong2021_v45_network import (
    LocalMixtureUNet,
    logistic_mixture_cdf,
    logistic_mixture_inverse,
    logistic_mixture_log_probability,
    mixture_parameters,
    parameter_count,
)
from hong2021_v45_train import (
    CHECKPOINT_SCHEMA,
    PARAMETERS,
    PROGRAM_SHA256 as V45_PROGRAM_SHA256,
    condition_cube,
    load_cache,
    load_program as load_v45_program,
)


PROGRAM_SCHEMA = "hong2021-v46-train-only-mixture-tail-occupancy-audit-program-v1"
PROGRAM_SHA256 = "85f429131257e617391345ab08e1743d4683a35681d1e10d4c79d4d381d049ad"
RESULT_SCHEMA = "hong2021-v46-train-only-mixture-tail-occupancy-audit-v1"
SEED = 146046
PROBE_VOXELS = 4096
PIT_TAILS = (0.01, 0.001, 0.0001)
SUMMARY_QUANTILES = (0.5, 0.9, 0.99, 0.999, 1.0)
PREDICTIVE_QUANTILES = (0.99, 0.999, 0.9999)
EXPECTED_OBJECTS = {"TNG100": 432, "SIMBA": 202, "Swift": 409}


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V46 {label} hash differs")
    return json.loads(path.read_text())


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status")
        != "frozen_before_audit_implementation_or_diagnostic_execution"
    ):
        raise ValueError("V46 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        (repo / parent["v45_record"]).resolve(),
        parent["v45_record_sha256"],
        "V45 record",
    )
    decision = record.get("development_decision", {})
    if (
        decision.get("classification") != parent["required_classification"]
        or decision.get("next") != parent["required_next"]
        or decision.get("development_pass") is not parent["required_development_pass"]
        or decision.get("candidate_high_k_power_and_residual_RMS_all_domains")
        is not parent["required_high_k_power_and_residual_RMS_all_domains"]
        or record.get("firewall", {}).get("independent_gate_locked") is not True
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V46 parent conclusion or firewall differs")
    frozen = program["frozen_inputs"]
    for key, digest_key in (
        ("v45_program", "v45_program_sha256"),
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
            raise ValueError(f"V46 frozen {key} hash differs")
    v45, v35, _ = load_v45_program((repo / frozen["v45_program"]).resolve(), repo)
    return program, v35


def _probe_indices(domain_index: int, object_index: int) -> np.ndarray:
    generator = np.random.default_rng(SEED + 100_000 * domain_index + object_index)
    return np.sort(
        generator.choice(64**3, size=PROBE_VOXELS, replace=False).astype(np.int64)
    )


def _quantiles(value: np.ndarray, probabilities: tuple[float, ...]) -> list[float]:
    if value.size == 0 or not np.isfinite(value).all():
        raise RuntimeError("V46 quantile input differs")
    return np.quantile(value.astype(np.float64, copy=False), probabilities).tolist()


def _density_moment(y: np.ndarray) -> float:
    delta = np.power(10.0, 4.5 * y.astype(np.float64)) - 1.0
    result = float(np.mean(np.square(delta), dtype=np.float64))
    if not math.isfinite(result):
        raise RuntimeError("V46 density moment is nonfinite")
    return result


def _effective(probability: np.ndarray) -> float:
    value = probability / probability.sum()
    return float(np.exp(-np.sum(value * np.log(np.clip(value, 1.0e-30, None)))))


def classify(
    unsupported_component: bool,
    globally_overdispersed: bool,
    component_collapse: bool,
    train_tail_calibrated: bool,
) -> tuple[str, str]:
    if unsupported_component:
        return (
            "unsupported_low_responsibility_component_mass_drives_the_train_tail",
            "replace_free_logistic_mixture_with_a_proper_train_only_distribution_that_cannot_hide_mass_in_unoccupied_components",
        )
    if globally_overdispersed:
        return (
            "mixture_likelihood_is_globally_overdispersed_in_the_train_upper_tail",
            "replace_logistic_tail_shape_with_a_train_only_tail_identifiable_proper_likelihood",
        )
    if component_collapse:
        return (
            "identifiable_initialization_did_not_prevent_component_collapse",
            "stop_finite_mixture_parameterization_and_use_a_monotone_train_only_conditional_quantile_model",
        )
    if train_tail_calibrated:
        return (
            "train_mixture_tail_is_calibrated_but_empirical_rank_copula_breaks_development_extremes",
            "audit_only_the_train_empirical_conditional_rank_tail_and_query_coupling",
        )
    return (
        "mixture_tail_failure_is_mixed_or_not_identified",
        "compare_train_tail_NLL_PIT_occupancy_and_posterior_predictive_failures_before_any_new_generator",
    )


def _load_model(program: dict[str, Any], repo: Path, commit: str) -> LocalMixtureUNet:
    frozen = program["frozen_inputs"]
    checkpoint = torch.load(frozen["checkpoint"], map_location="cpu", weights_only=False)
    source_commit = str(checkpoint.get("code_commit"))
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("program_sha256") != V45_PROGRAM_SHA256
        or checkpoint.get("step") != 12_000
        or checkpoint.get("parameters") != PARAMETERS
        or checkpoint.get("conditioning_cache_sha256")
        != frozen["conditioning_cache_sha256"]
        or checkpoint.get("spatial_rank_transport") is not False
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, commit],
            cwd=repo,
            capture_output=True,
        ).returncode
    ):
        raise ValueError("V46 V45 checkpoint binding differs")
    model = LocalMixtureUNet()
    if parameter_count(model) != PARAMETERS:
        raise RuntimeError("V46 V45 parameter count differs")
    model.load_state_dict(checkpoint["ema_state_dict"])
    return model


def _truth_probe(
    v35: dict[str, Any], prepared: h5py.File, domain: str, domain_index: int
) -> dict[str, np.ndarray]:
    row = v35["development_domains"][domain]
    objects = int(row["train_objects"])
    if objects != EXPECTED_OBJECTS[domain]:
        raise RuntimeError("V46 train object count differs")
    data, cache = _open_split(row, "train")
    standardized, physical = [], []
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    try:
        for object_index in range(objects):
            indices = _probe_indices(domain_index, object_index)
            truth = np.asarray(data["target"][object_index, 0], dtype=np.float32)
            backbone = _backbone(cache, object_index).astype(np.float32)
            residual = truth - backbone
            standardized.append(((residual.reshape(-1)[indices] - target_mean) / target_std).astype(np.float32))
            physical.append(truth.reshape(-1)[indices].astype(np.float32))
    finally:
        data.close()
        cache.close()
    result = {
        "standardized": np.concatenate(standardized),
        "physical_y": np.concatenate(physical),
    }
    if any(len(value) != objects * PROBE_VOXELS for value in result.values()):
        raise RuntimeError("V46 truth probe count differs")
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
    thresholds = np.quantile(truth_probe["standardized"], (0.99, 0.999))
    total_voxels = objects * 64**3
    nll_sum = 0.0
    tail_nll_sum = np.zeros(2, dtype=np.float64)
    tail_counts = np.zeros(2, dtype=np.int64)
    pit_lower = np.zeros(len(PIT_TAILS), dtype=np.int64)
    pit_upper = np.zeros(len(PIT_TAILS), dtype=np.int64)
    weight_sum = np.zeros(5, dtype=np.float64)
    responsibility_sum = np.zeros(5, dtype=np.float64)
    tail_responsibility_sum = np.zeros((2, 5), dtype=np.float64)
    entropy_sum = 0.0
    sampled = {name: [[] for _ in range(5)] for name in ("weight", "location", "scale", "responsibility", "upper_q99_99")}
    predictive_residual: list[np.ndarray] = []
    predictive_y: list[np.ndarray] = []
    maximum_inverse_error = 0.0
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    data, cache = _open_split(row, "train")
    try:
        for object_index in range(objects):
            condition, target, backbone = condition_cube(
                data, cache, prepared, domain, "train", object_index
            )
            parameter = model(torch.from_numpy(condition[None]).to(device)).float()
            observed = torch.from_numpy(target[None]).to(device).float()
            log_probability = logistic_mixture_log_probability(parameter, observed)
            cdf = logistic_mixture_cdf(parameter, observed)
            logits, locations, scales = mixture_parameters(parameter)
            weights = torch.softmax(logits, dim=1)
            z = (observed - locations) / scales
            component_log_probability = -z - 2.0 * F.softplus(-z) - torch.log(scales)
            responsibilities = torch.softmax(
                F.log_softmax(logits, dim=1) + component_log_probability, dim=1
            )
            if (
                float(torch.max(torch.abs(weights.sum(dim=1) - 1.0))) > 1.0e-5
                or float(torch.max(torch.abs(responsibilities.sum(dim=1) - 1.0)))
                > 1.0e-5
            ):
                raise RuntimeError("V46 mixture normalization differs")
            nll = -log_probability
            nll_sum += float(nll.double().sum().cpu())
            flat_target = observed.reshape(-1)
            for tail_index, threshold in enumerate(thresholds):
                mask = flat_target >= float(threshold)
                count = int(mask.sum().cpu())
                tail_counts[tail_index] += count
                tail_nll_sum[tail_index] += float(
                    nll.reshape(-1)[mask].double().sum().cpu()
                )
                tail_responsibility_sum[tail_index] += (
                    responsibilities.permute(0, 2, 3, 4, 1).reshape(-1, 5)[mask]
                    .double()
                    .sum(dim=0)
                    .cpu()
                    .numpy()
                )
            for tail_index, probability in enumerate(PIT_TAILS):
                pit_lower[tail_index] += int((cdf < probability).sum().cpu())
                pit_upper[tail_index] += int((cdf > 1.0 - probability).sum().cpu())
            weight_sum += weights.double().sum(dim=(0, 2, 3, 4)).cpu().numpy()
            responsibility_sum += (
                responsibilities.double().sum(dim=(0, 2, 3, 4)).cpu().numpy()
            )
            entropy_sum += float(
                (-(weights * torch.log(weights.clamp_min(1.0e-30))).sum(dim=1))
                .double()
                .sum()
                .cpu()
            )

            indices = _probe_indices(domain_index, object_index)
            index_tensor = torch.from_numpy(indices).to(device)
            flat_parameter = parameter.reshape(1, 15, -1).index_select(2, index_tensor).reshape(1, 15, 1, 1, -1)
            flat_responsibility = responsibilities.reshape(1, 5, -1).index_select(2, index_tensor)
            probe_logits, probe_locations, probe_scales = mixture_parameters(flat_parameter)
            probe_weights = torch.softmax(probe_logits, dim=1)
            endpoint = probe_locations + math.log(9999.0) * probe_scales
            for component in range(5):
                sampled["weight"][component].append(probe_weights[0, component].cpu().numpy())
                sampled["location"][component].append(probe_locations[0, component].cpu().numpy())
                sampled["scale"][component].append(probe_scales[0, component].cpu().numpy())
                sampled["responsibility"][component].append(flat_responsibility[0, component].cpu().numpy())
                sampled["upper_q99_99"][component].append(endpoint[0, component].cpu().numpy())
            flat_backbone = backbone.reshape(-1)[indices]
            for member in range(4):
                generator = np.random.default_rng(
                    SEED + member + 10_000_000 * domain_index + 10_000 * object_index
                )
                rank = generator.random(PROBE_VOXELS, dtype=np.float32).reshape(1, 1, 1, 1, -1)
                rank_tensor = torch.from_numpy(rank).to(device)
                draw = logistic_mixture_inverse(flat_parameter, rank_tensor)
                error = float(torch.max(torch.abs(logistic_mixture_cdf(flat_parameter, draw) - rank_tensor)).cpu())
                maximum_inverse_error = max(maximum_inverse_error, error)
                residual = draw.cpu().numpy().reshape(-1) * target_std + target_mean
                predictive_residual.append(residual.astype(np.float32))
                predictive_y.append((flat_backbone + residual).astype(np.float32))
            if (object_index + 1) % 16 == 0 or object_index + 1 == objects:
                print(f"[v46-audit] {domain} {object_index + 1}/{objects}", flush=True)
    finally:
        data.close()
        cache.close()

    if tail_counts.min() <= 0 or maximum_inverse_error > 2.0e-6:
        raise RuntimeError("V46 tail count or inverse error differs")
    mean_weight = weight_sum / total_voxels
    mean_responsibility = responsibility_sum / total_voxels
    sampled_summary: dict[str, list[dict[str, Any]]] = {}
    for name, by_component in sampled.items():
        sampled_summary[name] = [
            {"quantiles": _quantiles(np.concatenate(values), SUMMARY_QUANTILES)}
            for values in by_component
        ]
    predicted_residual = np.concatenate(predictive_residual)
    predicted_y = np.concatenate(predictive_y)
    truth_residual_physical = truth_probe["standardized"] * target_std + target_mean
    truth_y = truth_probe["physical_y"]
    residual_truth_q = _quantiles(truth_residual_physical, PREDICTIVE_QUANTILES)
    residual_predicted_q = _quantiles(predicted_residual, PREDICTIVE_QUANTILES)
    y_truth_q = _quantiles(4.5 * truth_y, PREDICTIVE_QUANTILES)
    y_predicted_q = _quantiles(4.5 * predicted_y, PREDICTIVE_QUANTILES)
    truth_moment = _density_moment(truth_y)
    predicted_moment = _density_moment(predicted_y)
    pit = {
        str(probability): {
            "lower_fraction": float(pit_lower[index] / total_voxels),
            "upper_fraction": float(pit_upper[index] / total_voxels),
            "lower_observed_over_expected": float(pit_lower[index] / total_voxels / probability),
            "upper_observed_over_expected": float(pit_upper[index] / total_voxels / probability),
        }
        for index, probability in enumerate(PIT_TAILS)
    }
    unsupported = []
    truth_standardized_q99_99 = float(
        np.quantile(truth_probe["standardized"].astype(np.float64), 0.9999)
    )
    for component in range(5):
        ratio = float(mean_responsibility[component] / mean_weight[component])
        median_endpoint = sampled_summary["upper_q99_99"][component]["quantiles"][0]
        if mean_weight[component] >= 0.02 and ratio < 0.5 and median_endpoint > truth_standardized_q99_99:
            unsupported.append(component)
    result = {
        "train_objects": objects,
        "total_native_voxels": total_voxels,
        "probe_voxels": int(len(truth_probe["standardized"])),
        "mean_NLL": float(nll_sum / total_voxels),
        "upper_tail_NLL": {
            "q99": float(tail_nll_sum[0] / tail_counts[0]),
            "q99_9": float(tail_nll_sum[1] / tail_counts[1]),
        },
        "PIT": pit,
        "mean_mixture_weight": mean_weight.tolist(),
        "mean_posterior_responsibility": mean_responsibility.tolist(),
        "responsibility_to_weight_ratio": (mean_responsibility / mean_weight).tolist(),
        "upper_tail_mean_posterior_responsibility": {
            "q99": (tail_responsibility_sum[0] / tail_counts[0]).tolist(),
            "q99_9": (tail_responsibility_sum[1] / tail_counts[1]).tolist(),
        },
        "mean_weight_entropy": float(entropy_sum / total_voxels),
        "mean_effective_weight_components": float(math.exp(entropy_sum / total_voxels)),
        "global_effective_responsibility_components": _effective(mean_responsibility),
        "sampled_component_summaries": sampled_summary,
        "truth_standardized_residual_q99_99": truth_standardized_q99_99,
        "unsupported_component_indices": unsupported,
        "posterior_predictive": {
            "residual_truth_quantiles": residual_truth_q,
            "residual_predicted_quantiles": residual_predicted_q,
            "log10rho_truth_quantiles": y_truth_q,
            "log10rho_predicted_quantiles": y_predicted_q,
            "delta_q99_99_log10rho_dex": float(y_predicted_q[2] - y_truth_q[2]),
            "truth_mean_delta_squared": truth_moment,
            "predicted_mean_delta_squared": predicted_moment,
            "predicted_over_truth_mean_delta_squared": float(predicted_moment / truth_moment),
            "members_per_probed_voxel": 4,
            "maximum_inverse_CDF_error": maximum_inverse_error,
        },
    }
    return result


def audit(program_path: Path, repo: Path, output: Path) -> dict[str, Any]:
    program, v35 = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V46 audit requires a clean committed worktree")
    if socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V46 audit requires Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V46 audit requires the Lageunha Ada GPU")
    if output.exists():
        raise FileExistsError("V46 refuses existing output")
    device = torch.device("cuda")
    model = _load_model(program, repo.resolve(), commit).to(device).eval()
    frozen = program["frozen_inputs"]
    checkpoint = torch.load(frozen["checkpoint"], map_location="cpu", weights_only=False)
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
        row["posterior_predictive"]["delta_q99_99_log10rho_dex"] > 0.10
        or row["posterior_predictive"]["predicted_over_truth_mean_delta_squared"] > 1.5
        for row in domains.values()
    ) and sum(row["PIT"]["0.001"]["upper_observed_over_expected"] < 0.5 for row in domains.values()) >= 2
    collapse = any(
        row["global_effective_responsibility_components"] < 3.0
        for row in domains.values()
    )
    calibrated = all(
        0.5 <= row["PIT"]["0.001"]["lower_observed_over_expected"] <= 2.0
        and 0.5 <= row["PIT"]["0.001"]["upper_observed_over_expected"] <= 2.0
        and abs(row["posterior_predictive"]["delta_q99_99_log10rho_dex"]) <= 0.10
        and 2.0 / 3.0
        <= row["posterior_predictive"]["predicted_over_truth_mean_delta_squared"]
        <= 1.5
        for row in domains.values()
    )
    classification, next_step = classify(unsupported, overdispersed, collapse, calibrated)
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete_train_only_diagnostic",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "audit_code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "domains": domains,
        "branch_conditions": {
            "unsupported_component_mass": unsupported,
            "globally_overdispersed_train_upper_tail": overdispersed,
            "component_collapse": collapse,
            "train_tail_calibrated": calibrated,
        },
        "classification": classification,
        "next": next_step,
        "training_or_refit_performed": False,
        "new_development_sample_generated": False,
        "validation_inputs_opened": False,
        "validation_truth_opened": False,
        "development_arrays_reopened": False,
        "threshold_changed_after_diagnostic": False,
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
