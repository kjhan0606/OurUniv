#!/usr/bin/env python
"""Train the preflight-approved V63 conditional physical-moment model."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import h5py
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
    UPPER_SUPPORT,
    LocalMixtureUNet,
    parameter_count,
)
from hong2021_v50_train import PARAMETERS
from hong2021_v54_train import _learning_rate, _same_seed_model
from hong2021_v56_train import (
    GRID_CELLS,
    GRID_COEFFICIENT,
    LIKELIHOOD_FAMILY,
    SEED,
    STEPS,
    TAIL_COEFFICIENT,
    VALIDATION_STEPS,
    composite_loss as v56_composite_loss,
)
from hong2021_v62_conditional_moment_gradient_audit import (
    _quadrature_rule,
    conditional_log_moment_score,
    conditional_physical_moments,
)
from hong2021_v63_preflight import (
    PROGRAM_SHA256,
    SCHEMA as PREFLIGHT_SCHEMA,
    _path,
    _verified_json,
    load_program,
)


CHECKPOINT_SCHEMA = "hong2021-v63-conditional-log-physical-moment-checkpoint-v1"
REPORT_SCHEMA = "hong2021-v63-conditional-log-physical-moment-training-report-v1"
PREFLIGHT_SHA256 = "00d7e8fd1ad182645597d52db10b773c151d2e54669775090f55c92dcc76d4db"
PREFLIGHT_DECISION_DIGEST = (
    "fa68bbb71797ed58403d2a3f5da1118e1d1f4bb9740c1e3b659e8d0498b93aa5"
)
PREFLIGHT_CODE_COMMIT = "b1b0d4904c0f436b3b9d24de4da1be1fdb1335bd"
PREFLIGHT_IMPLEMENTATION_SHA256 = (
    "a48b1fe58f9df34bff7ad5c2d625b1565f455395aa4eb506a459adcd60f7eb27"
)
CANDIDATE_IMPLEMENTATION_SHA256 = (
    "c4a0b00385269bc121a7624b01dcbe18bc32007cf6e6ffdfe58bf7cb2b65c9f5"
)
PREFLIGHT_RECORD = "config/hong2021_v63_preflight_record.json"
PREFLIGHT_RECORD_SHA256 = (
    "cf0abd8ec0c94db8b7489c9a0200510e1ba60af6e9524a958a96d6fa934208bb"
)
QUADRATURE_ORDER = 64
MOMENT_COEFFICIENT = 0.1


def _is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _load_training_inputs(
    program_path: Path,
    repo: Path,
    cache_path: Path,
    cache_sha: str,
    threshold_path: Path,
    threshold_sha: str,
    grid_path: Path,
    grid_sha: str,
    preflight_path: Path,
    preflight_sha: str,
    commit: str,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]
]:
    program, v35, grid, thresholds, _, _, _ = load_program(program_path, repo)
    frozen = program["frozen_inputs"]
    bindings = (
        (
            cache_path.resolve(),
            _path(repo, frozen["conditioning_cache"]),
            cache_sha,
            frozen["conditioning_cache_sha256"],
        ),
        (
            threshold_path.resolve(),
            _path(repo, frozen["v54_threshold_selection"]),
            threshold_sha,
            frozen["v54_threshold_selection_sha256"],
        ),
        (
            grid_path.resolve(),
            _path(repo, frozen["v56_grid"]),
            grid_sha,
            frozen["v56_grid_sha256"],
        ),
        (
            preflight_path.resolve(),
            Path(program["output_roots"]["preflight"]).resolve(),
            preflight_sha,
            PREFLIGHT_SHA256,
        ),
    )
    if any(
        actual != expected or digest != expected_digest
        for actual, expected, digest, expected_digest in bindings
    ):
        raise ValueError("V63 training input binding differs")
    record_path = repo / PREFLIGHT_RECORD
    if (
        sha256_file(repo / "src/hong2021_v63_preflight.py")
        != PREFLIGHT_IMPLEMENTATION_SHA256
        or sha256_file(repo / "src/hong2021_v62_conditional_moment_gradient_audit.py")
        != CANDIDATE_IMPLEMENTATION_SHA256
        or sha256_file(record_path) != PREFLIGHT_RECORD_SHA256
    ):
        raise ValueError("V63 approved implementation or record differs")
    record = _verified_json(record_path, PREFLIGHT_RECORD_SHA256, "preflight record")
    preflight = _verified_json(preflight_path, preflight_sha, "preflight")
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("status") != "pass"
        or preflight.get("program_sha256") != PROGRAM_SHA256
        or preflight.get("code_commit") != PREFLIGHT_CODE_COMMIT
        or canonical_digest(preflight) != PREFLIGHT_DECISION_DIGEST
        or preflight.get("V56_base_reproduction_pass") is not True
        or preflight.get("candidate_reproduces_V62") is not True
        or preflight.get("train_mask_occupancy_pass") is not True
        or preflight.get("quadrature_convergence_pass") is not True
        or preflight.get("gradient_scale_pass") is not True
        or preflight.get("gradient_pass") is not True
        or preflight.get("memory_pass") is not True
        or preflight.get("training_performed") is not False
        or preflight.get("validation_accessed") is not False
        or preflight.get("development_accessed") is not False
        or preflight.get("independent_gate_locked") is not True
        or record.get("status") != "complete_hard_preflight_pass_training_authorized"
        or record.get("preflight", {}).get("sha256") != preflight_sha
        or record.get("authorization", {}).get("training_allowed") is not True
        or record.get("authorization", {}).get("training_steps") != STEPS
        or record.get("authorization", {}).get("development_access_allowed") is not False
        or not _is_ancestor(repo, PREFLIGHT_CODE_COMMIT, commit)
    ):
        raise ValueError("V63 preflight authorization differs")
    boundaries = {
        domain: float(program["sealed_q99_9_backbone_boundaries"][domain])
        for domain in DOMAIN_ORDER
    }
    return program, v35, grid, thresholds, boundaries


def conditional_moment_score(
    parameters: torch.Tensor,
    target: torch.Tensor,
    backbone: torch.Tensor,
    target_mean: float,
    target_std: float,
    boundaries: torch.Tensor,
    nodes: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    predicted, truth, counts = conditional_physical_moments(
        parameters,
        target,
        backbone,
        target_mean,
        target_std,
        boundaries,
        nodes,
        weights,
    )
    return conditional_log_moment_score(predicted, truth), predicted, truth, counts


def composite_training_loss(
    parameters: torch.Tensor,
    target: torch.Tensor,
    backbone: torch.Tensor,
    target_mean: float,
    target_std: float,
    v54_thresholds: torch.Tensor,
    grid_thresholds: torch.Tensor,
    grid_weights: torch.Tensor,
    boundaries: torch.Tensor,
    nodes: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[Any, ...]:
    base = v56_composite_loss(
        parameters,
        target,
        backbone,
        target_mean,
        target_std,
        v54_thresholds,
        grid_thresholds,
        grid_weights,
    )
    moment, predicted, truth, counts = conditional_moment_score(
        parameters,
        target,
        backbone,
        target_mean,
        target_std,
        boundaries,
        nodes,
        weights,
    )
    total = base[0] + MOMENT_COEFFICIENT * moment
    return total, *base[1:], moment, predicted, truth, counts


@torch.no_grad()
def _validation_scores(
    model: LocalMixtureUNet,
    v35: dict[str, Any],
    prepared: h5py.File,
    device: torch.device,
    v54_thresholds: torch.Tensor,
    grid_thresholds: torch.Tensor,
    grid_weights: torch.Tensor,
    boundary_values: dict[str, float],
    nodes: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    result: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        data, cache = _open_split(v35["development_domains"][domain], "validation")
        rows: list[tuple[float, ...]] = []
        counts: list[int] = []
        try:
            objects = int(v35["development_domains"][domain]["validation_objects"])
            for index in range(objects):
                condition, target, backbone = condition_cube(
                    data, cache, prepared, domain, "validation", index
                )
                parameter = model(torch.from_numpy(condition[None]).to(device))
                target_tensor = torch.from_numpy(target[None]).to(device)
                backbone_tensor = torch.from_numpy(backbone[None]).to(device)
                base = v56_composite_loss(
                    parameter,
                    target_tensor,
                    backbone_tensor,
                    float(prepared["target_mean"][()]),
                    float(prepared["target_std"][()]),
                    v54_thresholds,
                    grid_thresholds,
                    grid_weights,
                )
                repeated_parameter = parameter.expand(3, -1, -1, -1, -1)
                repeated_target = target_tensor.expand(3, -1, -1, -1, -1)
                repeated_backbone = backbone_tensor.expand(3, -1, -1, -1, -1)
                repeated_boundary = torch.full(
                    (3,), boundary_values[domain], dtype=torch.float64, device=device
                )
                moment, predicted, truth, selected = conditional_moment_score(
                    repeated_parameter,
                    repeated_target,
                    repeated_backbone,
                    float(prepared["target_mean"][()]),
                    float(prepared["target_std"][()]),
                    repeated_boundary,
                    nodes,
                    weights,
                )
                total = base[0] + MOMENT_COEFFICIENT * moment
                rows.append(
                    (
                        float(total.cpu()),
                        float(base[1].cpu()),
                        float(base[2].cpu()),
                        float(base[4].cpu()),
                        float(moment.cpu()),
                        float((predicted[0] / truth[0]).cpu()),
                    )
                )
                counts.append(selected[0])
        finally:
            data.close()
            cache.close()
        result[domain] = {
            "composite": float(np.mean([row[0] for row in rows])),
            "bounded_NLL": float(np.mean([row[1] for row in rows])),
            "V54_tail_score": float(np.mean([row[2] for row in rows])),
            "V56_grid_score": float(np.mean([row[3] for row in rows])),
            "conditional_moment_score": float(np.mean([row[4] for row in rows])),
            "geometric_mean_predicted_over_truth_moment": float(
                np.exp(np.mean(np.log([row[5] for row in rows])))
            ),
            "minimum_selected_voxels": min(counts),
        }
    model.train()
    return result


def train(
    program_path: Path,
    repo: Path,
    cache_path: Path,
    cache_sha: str,
    threshold_path: Path,
    threshold_sha: str,
    grid_path: Path,
    grid_sha: str,
    preflight_path: Path,
    preflight_sha: str,
    checkpoint_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    commit, clean = git_state(repo)
    if (
        not clean
        or socket.gethostname().split(".")[0].lower() != "lageunha"
        or checkpoint_path.exists()
        or report_path.exists()
    ):
        raise RuntimeError("V63 training requires clean Lageunha and new outputs")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V63 training requires the Lageunha Ada GPU")
    program, v35, grid, selected, boundary_values = _load_training_inputs(
        program_path,
        repo,
        cache_path,
        cache_sha,
        threshold_path,
        threshold_sha,
        grid_path,
        grid_sha,
        preflight_path,
        preflight_sha,
        commit,
    )
    prepared = load_cache(cache_path, cache_sha, commit)
    device = torch.device("cuda")
    model = _same_seed_model(device)
    ema_model = _same_seed_model(device)
    if parameter_count(model) != PARAMETERS:
        raise RuntimeError("V63 architecture differs")
    ema_model.load_state_dict(model.state_dict())
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler("cuda")
    generator = np.random.default_rng(SEED)
    v54_thresholds = torch.tensor(selected["common_log10rho_thresholds"], device=device)
    grid_thresholds = torch.tensor(grid["thresholds_log10rho"], device=device)
    grid_weights = torch.tensor(grid["physical_moment_weights"], device=device)
    boundaries = torch.tensor(
        [boundary_values[domain] for domain in DOMAIN_ORDER],
        dtype=torch.float64,
        device=device,
    )
    nodes, weights = _quadrature_rule(QUADRATURE_ORDER, device)
    handles = {
        domain: _open_split(v35["development_domains"][domain], "train")
        for domain in DOMAIN_ORDER
    }
    log_rows: list[dict[str, Any]] = []
    validation_rows: dict[str, Any] = {}
    overflow_events: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        for step in range(1, STEPS + 1):
            conditions, targets, backbones = [], [], []
            for domain in DOMAIN_ORDER:
                source = v35["development_domains"][domain]
                index = int(generator.integers(int(source["train_objects"])))
                data, cache = handles[domain]
                condition, target, backbone = condition_cube(
                    data, cache, prepared, domain, "train", index
                )
                axes, reflections = CUBE_ISOMETRIES[
                    int(generator.integers(len(CUBE_ISOMETRIES)))
                ]
                conditions.append(apply_cube_isometry(condition, axes, reflections))
                targets.append(apply_cube_isometry(target, axes, reflections))
                backbones.append(apply_cube_isometry(backbone, axes, reflections))
            condition_tensor = torch.from_numpy(np.stack(conditions)).to(device)
            target_tensor = torch.from_numpy(np.stack(targets)).to(device)
            backbone_tensor = torch.from_numpy(np.stack(backbones)).to(device)
            learning_rate = _learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                parameters = model(condition_tensor)
            scores = composite_training_loss(
                parameters,
                target_tensor,
                backbone_tensor,
                float(prepared["target_mean"][()]),
                float(prepared["target_std"][()]),
                v54_thresholds,
                grid_thresholds,
                grid_weights,
                boundaries,
                nodes,
                weights,
            )
            total, nll, tail, tail_components, upper, upper_components = scores[:6]
            moment, predicted, truth, counts = scores[6:]
            scale_before = float(scaler.get_scale())
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gradient_finite = bool(torch.isfinite(gradient_norm).cpu())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            overflow = scale_after < scale_before
            if overflow:
                overflow_events.append(
                    {
                        "step": step,
                        "loss": float(total.detach().cpu()),
                        "gradient_norm_is_finite": gradient_finite,
                        "scale_before": scale_before,
                        "scale_after": scale_after,
                    }
                )
            with torch.no_grad():
                for ema, current in zip(
                    ema_model.parameters(), model.parameters(), strict=True
                ):
                    ema.lerp_(current.detach(), 1.0 - 0.999)
                for ema, current in zip(ema_model.buffers(), model.buffers(), strict=True):
                    ema.copy_(current)
            if step == 1 or step % 50 == 0:
                row = {
                    "step": step,
                    "composite_loss": float(total.detach().cpu()),
                    "bounded_NLL": float(nll.detach().cpu()),
                    "V54_tail_score": float(tail.detach().cpu()),
                    "V54_Brier_components": tail_components.detach().cpu().tolist(),
                    "V56_grid_score": float(upper.detach().cpu()),
                    "V56_grid_raw_Brier_components": upper_components.detach().cpu().tolist(),
                    "conditional_moment_score": float(moment.detach().cpu()),
                    "conditional_predicted_over_truth": {
                        domain: float((predicted[index] / truth[index]).detach().cpu())
                        for index, domain in enumerate(DOMAIN_ORDER)
                    },
                    "conditional_selected_voxels": {
                        domain: counts[index]
                        for index, domain in enumerate(DOMAIN_ORDER)
                    },
                    "learning_rate": learning_rate,
                    "gradient_norm_before_clip": (
                        float(gradient_norm.detach().cpu())
                        if gradient_finite
                        else "nonfinite"
                    ),
                    "AMP_scale_before_update": scale_before,
                    "AMP_scale_after_update": scale_after,
                    "AMP_update_skipped": overflow,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                }
                log_rows.append(row)
                print("[v63-train] " + json.dumps(row), flush=True)
            if step in VALIDATION_STEPS:
                validation_rows[str(step)] = _validation_scores(
                    ema_model,
                    v35,
                    prepared,
                    device,
                    v54_thresholds,
                    grid_thresholds,
                    grid_weights,
                    boundary_values,
                    nodes,
                    weights,
                )
                print(
                    f"[v63-validation] step={step} "
                    + json.dumps(validation_rows[str(step)]),
                    flush=True,
                )
    finally:
        for data, cache in handles.values():
            data.close()
            cache.close()
        prepared.close()
    peak = int(torch.cuda.max_memory_allocated(device))
    common = {
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "step": STEPS,
        "parameters": PARAMETERS,
        "likelihood_family": LIKELIHOOD_FAMILY,
        "open_standardized_support": [LOWER_SUPPORT, UPPER_SUPPORT],
        "v54_threshold_selection_sha256": threshold_sha,
        "common_log10rho_thresholds": selected["common_log10rho_thresholds"],
        "grid_sha256": grid_sha,
        "grid_thresholds_log10rho": grid["thresholds_log10rho"],
        "grid_physical_moment_weights": grid["physical_moment_weights"],
        "grid_cells": GRID_CELLS,
        "tail_coefficient": TAIL_COEFFICIENT,
        "grid_coefficient": GRID_COEFFICIENT,
        "moment_coefficient": MOMENT_COEFFICIENT,
        "moment_quadrature_order": QUADRATURE_ORDER,
        "q99_9_backbone_boundaries": boundary_values,
        "conditioning_cache_sha256": cache_sha,
        "preflight_sha256": preflight_sha,
        "preflight_decision_digest_sha256": PREFLIGHT_DECISION_DIGEST,
        "preflight_implementation_sha256": PREFLIGHT_IMPLEMENTATION_SHA256,
        "candidate_implementation_sha256": CANDIDATE_IMPLEMENTATION_SHA256,
        "preflight_record_sha256": PREFLIGHT_RECORD_SHA256,
        "support_selection_sha256": program["frozen_inputs"][
            "support_selection_sha256"
        ],
        "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
        "sample_clipping": False,
        "component_scale_cap": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        **common,
        "ema_decay": 0.999,
        "ema_state_dict": {
            key: value.detach().cpu() for key, value in ema_model.state_dict().items()
        },
        "AMP_overflow_events": overflow_events,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    partial_checkpoint = checkpoint_path.with_suffix(checkpoint_path.suffix + ".partial")
    torch.save(checkpoint, partial_checkpoint)
    os.replace(partial_checkpoint, checkpoint_path)
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete_fixed_12000_step_conditional_moment_fit",
        **common,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "steps": STEPS,
        "training_log": log_rows,
        "AMP_overflow_events": overflow_events,
        "AMP_overflow_count": len(overflow_events),
        "peak_allocated_bytes": peak,
        "fixed_validation_diagnostics": validation_rows,
        "development_accessed": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    partial_report = report_path.with_suffix(report_path.suffix + ".partial")
    partial_report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial_report, report_path)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--thresholds-sha256", required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--grid-sha256", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    train(
        args.program,
        args.repo,
        args.cache,
        args.cache_sha256,
        args.thresholds,
        args.thresholds_sha256,
        args.grid,
        args.grid_sha256,
        args.preflight,
        args.preflight_sha256,
        args.checkpoint,
        args.report,
    )


if __name__ == "__main__":
    main()
