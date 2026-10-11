#!/usr/bin/env python
"""No-refit train-only gradient audit for a direct conditional physical moment."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v48_train import condition_cube, load_cache
from hong2021_v50_network import (
    LOWER_SUPPORT,
    SUPPORT_RANGE,
    LocalMixtureUNet,
    bounded_mixture_log_probability,
    mixture_parameters,
    parameter_count,
)
from hong2021_v50_train import PARAMETERS
from hong2021_v54_train import _same_seed_model
from hong2021_v56_train import (
    CHECKPOINT_SCHEMA as V56_CHECKPOINT_SCHEMA,
    GRID_COEFFICIENT,
    TAIL_COEFFICIENT,
    composite_loss as v56_composite_loss,
    load_program as load_v56_program,
    upper_survival_grid_score,
)
from hong2021_v61_preflight import _checkpointed_weighted_score
from hong2021_v61_train import CHECKPOINT_SCHEMA as V61_CHECKPOINT_SCHEMA


PROGRAM_SHA256 = "f8b75e0b69937931f60cbd51036c89275c01c1d69e8a508dd94deedefcd5fb2a"
PROGRAM_SCHEMA = "hong2021-v62-conditional-physical-moment-gradient-audit-program-v1"
SCHEMA = "hong2021-v62-conditional-physical-moment-gradient-audit-v1"


def _path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V62 {label} hash differs")
    return _json(path)


def load_program(
    path: Path, repo: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    repo = repo.resolve()
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_audit_implementation_or_execution"
    ):
        raise ValueError("V62 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _path(repo, parent["v61_record"]), parent["v61_record_sha256"], "V61 record"
    )
    decision = record.get("train_only_mechanism_decision", {})
    firewall = record.get("firewall", {})
    if (
        record.get("status") != parent["required_status"]
        or decision.get("classification") != parent["required_classification"]
        or record.get("selected_next_step", {}).get("action")
        != parent["required_next_action"]
        or firewall.get("development_accessed")
        is not parent["required_development_accessed"]
        or firewall.get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
        or firewall.get("Astrid_accessed") is not False
        or firewall.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V62 parent conclusion or firewall differs")
    frozen = program["frozen_inputs"]
    for key, value in frozen.items():
        if key.endswith("_sha256"):
            continue
        digest = frozen.get(f"{key}_sha256")
        if digest is not None and sha256_file(_path(repo, value)) != digest:
            raise ValueError(f"V62 frozen input differs: {key}")
    v56_gate = _json(_path(repo, frozen["v56_train_gate"]))
    v61_gate = _json(_path(repo, frozen["v61_train_gate"]))
    v61_preflight = _json(_path(repo, frozen["v61_preflight"]))
    if (
        v56_gate.get("train_mechanism_pass") is not False
        or v61_gate.get("train_mechanism_pass") is not False
        or v56_gate.get("development_accessed") is not False
        or v61_gate.get("development_accessed") is not False
        or v56_gate.get("independent_gate_locked") is not True
        or v61_gate.get("independent_gate_locked") is not True
        or canonical_digest(v61_gate) != v61_gate.get("decision_digest_sha256")
        or float(v61_preflight.get("appended_score_parameter_gradient_maximum_absolute"))
        != float(record["preflight"]["appended_score_parameter_gradient_maximum_absolute"])
    ):
        raise ValueError("V62 sealed gate or preflight differs")
    _, v35, _ = load_v56_program(_path(repo, frozen["v56_program"]), repo)
    v56_grid = _json(_path(repo, frozen["v56_grid"]))
    v60_grid = _json(_path(repo, frozen["v60_grid"]))
    return program, v35, v56_grid, v60_grid, v61_gate


def conditional_log_moment_score(
    predicted: torch.Tensor, truth: torch.Tensor
) -> torch.Tensor:
    if (
        predicted.ndim != 1
        or truth.shape != predicted.shape
        or not bool(torch.isfinite(predicted).all().detach().cpu())
        or not bool(torch.isfinite(truth).all().detach().cpu())
        or not bool(((predicted > 0.0) & (truth > 0.0)).all().detach().cpu())
    ):
        raise ValueError("V62 physical moment score input differs")
    log_ratio = torch.log(predicted / truth)
    return 0.5 * torch.mean(torch.square(log_ratio))


def gradient_metrics(
    gradient: torch.Tensor, selected_voxels: int
) -> dict[str, float]:
    if selected_voxels <= 0 or not bool(torch.isfinite(gradient).all().detach().cpu()):
        raise ValueError("V62 output gradient differs")
    return {
        "L2": float(torch.linalg.vector_norm(gradient.double()).detach().cpu()),
        "L2_per_selected_voxel": float(
            (torch.linalg.vector_norm(gradient.double()) / selected_voxels)
            .detach()
            .cpu()
        ),
        "maximum_absolute": float(gradient.detach().abs().max().cpu()),
    }


def classify(
    numerical_pass: bool,
    masks_nonempty: bool,
    gradient_ratio_pass: bool,
    memory_pass: bool,
) -> tuple[str, str, bool]:
    if not numerical_pass:
        return (
            "direct_conditional_physical_moment_objective_is_numerically_unresolved",
            "stop_before_refit_and_freeze_a_higher_accuracy_train_only_numerical_audit",
            False,
        )
    if masks_nonempty and gradient_ratio_pass and memory_pass:
        return (
            "direct_conditional_log_physical_moment_objective_has_gate_aligned_optimization_scale",
            "freeze_one_V63_model_that_retains_the_complete_V56_objective_and_adds_only_the_coefficient_0.1_conditional_log_physical_moment_term",
            True,
        )
    return (
        "direct_conditional_physical_moment_objective_is_not_optimization_feasible",
        "stop_before_refit_and_audit_a_bounded_train_only_moment_estimating_score_without_development_access",
        False,
    )


def _model_from_checkpoint(
    path: Path,
    expected_schema: str,
    expected_program_sha256: str,
    device: torch.device,
) -> LocalMixtureUNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != expected_schema
        or checkpoint.get("program_sha256") != expected_program_sha256
        or checkpoint.get("step") != 12_000
        or checkpoint.get("parameters") != PARAMETERS
        or checkpoint.get("independent_gate_locked") is not True
        or checkpoint.get(
            "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection"
        )
        is not False
    ):
        raise ValueError("V62 checkpoint binding differs")
    model = LocalMixtureUNet().to(device)
    if parameter_count(model) != PARAMETERS:
        raise RuntimeError("V62 architecture differs")
    model.load_state_dict(checkpoint["ema_state_dict"])
    return model


def _real_batch(
    v35: dict[str, Any],
    prepared: Any,
    device: torch.device,
    object_index: int,
    isometry_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    conditions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    backbones: list[np.ndarray] = []
    axes, reflections = CUBE_ISOMETRIES[isometry_index]
    for domain in DOMAIN_ORDER:
        data, cache = _open_split(v35["development_domains"][domain], "train")
        try:
            condition, target, backbone = condition_cube(
                data, cache, prepared, domain, "train", object_index
            )
        finally:
            data.close()
            cache.close()
        conditions.append(apply_cube_isometry(condition, axes, reflections))
        targets.append(apply_cube_isometry(target, axes, reflections))
        backbones.append(apply_cube_isometry(backbone, axes, reflections))
    return (
        torch.from_numpy(np.stack(conditions)).to(device),
        torch.from_numpy(np.stack(targets)).to(device),
        torch.from_numpy(np.stack(backbones)).to(device),
    )


def _quadrature_rule(order: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return (
        torch.from_numpy(nodes).to(device),
        torch.from_numpy(weights).to(device) / math.sqrt(math.pi),
    )


def conditional_physical_moments(
    parameters: torch.Tensor,
    target: torch.Tensor,
    backbone: torch.Tensor,
    target_mean: float,
    target_std: float,
    boundaries: torch.Tensor,
    nodes: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if (
        parameters.shape[0] != len(DOMAIN_ORDER)
        or target.shape != (len(DOMAIN_ORDER), 1, *parameters.shape[-3:])
        or backbone.shape != target.shape
        or boundaries.shape != (len(DOMAIN_ORDER),)
        or nodes.ndim != 1
        or weights.shape != nodes.shape
        or target_std <= 0.0
    ):
        raise ValueError("V62 conditional moment input differs")
    coefficient = 4.5 * math.log(10.0)
    predicted_rows: list[torch.Tensor] = []
    truth_rows: list[torch.Tensor] = []
    counts: list[int] = []
    for index in range(len(DOMAIN_ORDER)):
        base = backbone[index, 0].double().reshape(-1) + float(target_mean)
        mask = base >= boundaries[index].double()
        count = int(mask.sum().detach().cpu())
        if count <= 0:
            raise RuntimeError("V62 empty high-backbone mask")
        selected = (
            parameters[index : index + 1]
            .reshape(1, 15, -1)[:, :, mask]
            .reshape(1, 15, 1, 1, count)
        )
        logits, locations, scales = mixture_parameters(selected)
        mixture_weights = torch.softmax(logits, dim=1)[0, :, 0, 0]
        locations = locations[0, :, 0, 0]
        scales = scales[0, :, 0, 0]
        predicted = torch.zeros(count, dtype=torch.float64, device=parameters.device)
        for component in range(mixture_weights.shape[0]):
            latent = locations[component].double()[:, None] + math.sqrt(2.0) * (
                scales[component].double()[:, None] * nodes.double()[None]
            )
            standardized = LOWER_SUPPORT + SUPPORT_RANGE * torch.sigmoid(latent)
            physical_y = base[mask, None] + float(target_std) * standardized
            delta_squared = torch.square(torch.exp(coefficient * physical_y) - 1.0)
            mass = mixture_weights[component].double()[:, None] * weights.double()[None]
            predicted += torch.sum(mass * delta_squared, dim=1)
        exact_y = base[mask] + float(target_std) * target[index, 0].double().reshape(-1)[mask]
        truth = torch.square(torch.exp(coefficient * exact_y) - 1.0)
        predicted_rows.append(predicted.mean())
        truth_rows.append(truth.mean())
        counts.append(count)
    predicted_tensor = torch.stack(predicted_rows)
    truth_tensor = torch.stack(truth_rows)
    if not bool(
        (
            torch.isfinite(predicted_tensor)
            & torch.isfinite(truth_tensor)
            & (predicted_tensor > 0.0)
            & (truth_tensor > 0.0)
        )
        .all()
        .detach()
        .cpu()
    ):
        raise RuntimeError("V62 nonfinite physical moment")
    return predicted_tensor, truth_tensor, counts


def _local_gradient(
    output: torch.Tensor,
    closure: Callable[[torch.Tensor], torch.Tensor],
    selected_voxels: int,
) -> tuple[float, dict[str, float]]:
    value = output.detach().requires_grad_(True)
    objective = closure(value)
    gradient = torch.autograd.grad(objective, value)[0]
    if not bool(torch.isfinite(objective).detach().cpu()):
        raise RuntimeError("V62 nonfinite local objective")
    return float(objective.detach().cpu()), gradient_metrics(gradient, selected_voxels)


def audit(program_path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, v35, v56_grid, v60_grid, v61_gate = load_program(
        program_path, repo
    )
    commit, clean = git_state(repo)
    if not clean or socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V62 audit requires clean Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V62 audit requires the Lageunha Ada GPU")
    device = torch.device("cuda")
    frozen = program["frozen_inputs"]
    prepared = load_cache(
        _path(repo, frozen["conditioning_cache"]),
        frozen["conditioning_cache_sha256"],
        commit,
    )
    batch_rule = program["fixed_real_train_batch"]
    condition, target, backbone = _real_batch(
        v35,
        prepared,
        device,
        int(batch_rule["object_index"]),
        int(batch_rule["signed_cube_isometry_index"]),
    )
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    prepared.close()
    boundaries = torch.tensor(
        [
            v61_gate["domains"][domain]["backbone_boundaries"][2]
            for domain in DOMAIN_ORDER
        ],
        dtype=torch.float64,
        device=device,
    )
    primary_order = int(program["numerical_controls"]["primary_Gauss_Hermite_order"])
    control_order = int(program["numerical_controls"]["control_Gauss_Hermite_order"])
    nodes64, weights64 = _quadrature_rule(primary_order, device)
    nodes32, weights32 = _quadrature_rule(control_order, device)
    v56_thresholds = torch.tensor(v56_grid["thresholds_log10rho"], device=device)
    v56_weights = torch.tensor(v56_grid["physical_moment_weights"], device=device)
    v60_thresholds = torch.tensor(v60_grid["thresholds_log10rho"], device=device)
    v60_weights = torch.tensor(v60_grid["physical_moment_weights"], device=device)
    v54 = _json(_path(repo, frozen["v54_threshold_selection"]))
    v54_thresholds = torch.tensor(v54["common_log10rho_thresholds"], device=device)
    models = {
        "same_seed_initialization": _same_seed_model(device),
        "V56_step12000_EMA": _model_from_checkpoint(
            _path(repo, frozen["v56_checkpoint"]),
            V56_CHECKPOINT_SCHEMA,
            frozen["v56_program_sha256"],
            device,
        ),
        "V61_step12000_EMA": _model_from_checkpoint(
            _path(repo, frozen["v61_checkpoint"]),
            V61_CHECKPOINT_SCHEMA,
            frozen["v61_program_sha256"],
            device,
        ),
    }
    rows: dict[str, Any] = {}
    quadrature_limit = float(
        program["numerical_controls"]["maximum_32_to_64_relative_difference"]
    )
    numerical_pass = True
    ratios: list[float] = []
    for label, model in models.items():
        model.eval()
        with torch.no_grad():
            output = model(condition)
            primary, truth, counts = conditional_physical_moments(
                output,
                target,
                backbone,
                target_mean,
                target_std,
                boundaries,
                nodes64,
                weights64,
            )
            control, control_truth, control_counts = conditional_physical_moments(
                output,
                target,
                backbone,
                target_mean,
                target_std,
                boundaries,
                nodes32,
                weights32,
            )
        if counts != control_counts or not torch.equal(truth, control_truth):
            raise RuntimeError("V62 quadrature control binding differs")
        convergence = torch.abs(primary - control) / torch.maximum(
            torch.maximum(primary.abs(), control.abs()),
            torch.full_like(primary, 1.0e-300),
        )
        numerical_pass = bool(
            numerical_pass and float(convergence.max().cpu()) <= quadrature_limit
        )
        selected = sum(counts)

        def candidate(value: torch.Tensor) -> torch.Tensor:
            predicted, observed, _ = conditional_physical_moments(
                value,
                target,
                backbone,
                target_mean,
                target_std,
                boundaries,
                nodes64,
                weights64,
            )
            return conditional_log_moment_score(predicted, observed)

        def nll(value: torch.Tensor) -> torch.Tensor:
            return -bounded_mixture_log_probability(value, target).mean()

        def old_grid(value: torch.Tensor) -> torch.Tensor:
            return upper_survival_grid_score(
                value,
                target,
                backbone,
                target_mean,
                target_std,
                v56_thresholds,
                v56_weights,
            )[0]

        def appended_grid(value: torch.Tensor) -> torch.Tensor:
            return _checkpointed_weighted_score(
                value,
                target,
                backbone,
                target_mean,
                target_std,
                v60_thresholds[16:],
                v60_weights[16:],
            )

        candidate_value, candidate_gradient = _local_gradient(
            output, candidate, selected
        )
        nll_value, nll_gradient = _local_gradient(output, nll, selected)
        old_value, old_gradient = _local_gradient(output, old_grid, selected)
        appended_value, appended_gradient = _local_gradient(
            output, appended_grid, selected
        )
        gradient_ratio = candidate_gradient["L2_per_selected_voxel"] / max(
            appended_gradient["L2_per_selected_voxel"], 1.0e-300
        )
        ratios.append(gradient_ratio)
        domains: dict[str, Any] = {}
        for index, domain in enumerate(DOMAIN_ORDER):
            ratio = float((primary[index] / truth[index]).cpu())
            domains[domain] = {
                "selected_voxels": counts[index],
                "truth_mean_delta_squared": float(truth[index].cpu()),
                "predicted_mean_delta_squared_64": float(primary[index].cpu()),
                "predicted_mean_delta_squared_32": float(control[index].cpu()),
                "predicted_over_truth_64": ratio,
                "log_predicted_over_truth_64": math.log(ratio),
                "quadrature_32_to_64_relative_difference": float(
                    convergence[index].cpu()
                ),
            }
        rows[label] = {
            "domains": domains,
            "candidate_score": candidate_value,
            "candidate_output_gradient": candidate_gradient,
            "bounded_NLL": nll_value,
            "bounded_NLL_output_gradient": nll_gradient,
            "V56_grid_score": old_value,
            "V56_grid_output_gradient": old_gradient,
            "V61_appended_118_score": appended_value,
            "V61_appended_118_output_gradient": appended_gradient,
            "candidate_to_V61_appended_output_gradient_L2_per_selected_voxel_ratio": gradient_ratio,
        }
        print(f"[v62-audit] completed {label}", flush=True)

    initialization = models["same_seed_initialization"]
    initialization.train()
    initialization.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    output = initialization(condition)
    base = v56_composite_loss(
        output,
        target,
        backbone,
        target_mean,
        target_std,
        v54_thresholds,
        v56_thresholds,
        v56_weights,
    )
    predicted, observed, _ = conditional_physical_moments(
        output,
        target,
        backbone,
        target_mean,
        target_std,
        boundaries,
        nodes64,
        weights64,
    )
    candidate = conditional_log_moment_score(predicted, observed)
    coefficient = float(program["candidate_objective"]["future_model_coefficient_if_selected"])
    composite = base[0] + coefficient * candidate
    composite.backward()
    squared_norm = torch.zeros((), dtype=torch.float64, device=device)
    maximum = 0.0
    finite = True
    for parameter in initialization.parameters():
        if parameter.grad is None:
            continue
        finite = bool(finite and torch.isfinite(parameter.grad).all().item())
        squared_norm += torch.sum(torch.square(parameter.grad.double()))
        maximum = max(maximum, float(parameter.grad.detach().abs().max().cpu()))
    full_model = {
        "V56_composite": float(base[0].detach().cpu()),
        "bounded_NLL": float(base[1].detach().cpu()),
        "V54_tail_score": float(base[2].detach().cpu()),
        "V56_grid_score": float(base[4].detach().cpu()),
        "candidate_score": float(candidate.detach().cpu()),
        "candidate_coefficient": coefficient,
        "candidate_composite": float(composite.detach().cpu()),
        "gradient_finite": finite,
        "gradient_L2": float(torch.sqrt(squared_norm).detach().cpu()),
        "gradient_maximum_absolute": maximum,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    memory_pass = full_model["peak_allocated_bytes"] < int(
        program["gradient_comparison"]["peak_allocated_bytes_limit"]
    )
    minimum_ratio = min(ratios)
    ratio_pass = minimum_ratio >= float(
        program["gradient_comparison"][
            "minimum_candidate_to_V61_appended_output_gradient_L2_ratio"
        ]
    )
    masks_nonempty = all(
        row["selected_voxels"] > 0
        for model_row in rows.values()
        for row in model_row["domains"].values()
    )
    numerical_pass = bool(
        numerical_pass
        and finite
        and full_model["gradient_L2"] > 0.0
        and math.isfinite(full_model["gradient_L2"])
    )
    classification, next_step, selected = classify(
        numerical_pass, masks_nonempty, ratio_pass, memory_pass
    )
    for model in models.values():
        del model
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_no_refit_train_only_objective_gradient_audit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "fixed_real_train_batch": batch_rule,
        "q99_9_backbone_boundaries": {
            domain: float(boundaries[index].cpu())
            for index, domain in enumerate(DOMAIN_ORDER)
        },
        "models": rows,
        "initialization_full_model_candidate_composite": full_model,
        "maximum_quadrature_relative_difference": max(
            row["quadrature_32_to_64_relative_difference"]
            for model_row in rows.values()
            for row in model_row["domains"].values()
        ),
        "quadrature_limit": quadrature_limit,
        "minimum_candidate_to_V61_appended_output_gradient_ratio": minimum_ratio,
        "required_minimum_gradient_ratio": program["gradient_comparison"][
            "minimum_candidate_to_V61_appended_output_gradient_L2_ratio"
        ],
        "numerical_pass": numerical_pass,
        "masks_nonempty": masks_nonempty,
        "gradient_ratio_pass": ratio_pass,
        "memory_pass": memory_pass,
        "candidate_selected": selected,
        "classification": classification,
        "next": next_step,
        "training_or_refit_performed": False,
        "validation_accessed": False,
        "development_accessed": False,
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
        raise FileExistsError("V62 refuses existing audit output")
    result = audit(args.program, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
