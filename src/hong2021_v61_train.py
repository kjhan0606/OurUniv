#!/usr/bin/env python
"""Train the preflight-approved V61 reachable-support survival model."""
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
    bounded_mixture_log_probability,
    parameter_count,
)
from hong2021_v50_train import PARAMETERS
from hong2021_v54_train import (
    TAIL_COEFFICIENT,
    _learning_rate,
    _same_seed_model,
    physical_tail_brier_score,
)
from hong2021_v56_train import (
    GRID_COEFFICIENT,
    LIKELIHOOD_FAMILY,
    REFERENCE_PROBABILITY,
    SEED,
    STEPS,
    VALIDATION_STEPS,
    load_program as load_v56_program,
)
from hong2021_v61_preflight import (
    EXISTING_CELLS,
    GRID_CELLS,
    PROGRAM_SHA256,
    SCORE_CHUNK_CELLS,
    SCHEMA as PREFLIGHT_SCHEMA,
    _checkpointed_weighted_score,
    _path,
    _raw_survival_components,
    _verified_json,
    load_program,
)


CHECKPOINT_SCHEMA = "hong2021-v61-reachable-support-checkpoint-v1"
REPORT_SCHEMA = "hong2021-v61-reachable-support-training-report-v1"
PREFLIGHT_SHA256 = "d900088be814bc969a94e4e4e2ff8b85d358665ab52dc9c31539e17bacb43063"
PREFLIGHT_DECISION_DIGEST = (
    "02ba6a5fb7111bc2babc5f37a2701c2a7761b085298026035e9d6e5b9074e64c"
)
PREFLIGHT_CODE_COMMIT = "c439b11059a3f949f574e4743d4dc4fdbc14ac0a"
PREFLIGHT_IMPLEMENTATION_SHA256 = (
    "fc03c6a7d9e4fd9c49f87b02cbc9c4cd68b966bd24c389010ea7774a3a8e22f2"
)
PREFLIGHT_RECORD = "config/hong2021_v61_preflight_record.json"
PREFLIGHT_RECORD_SHA256 = (
    "3ec5bca83a1088aa733f0c0b8534d9ea09affe46f1d8777e433b57ae2c9cce17"
)


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
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    program, _, grid = load_program(program_path, repo)
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
            _path(repo, frozen["v60_grid"]),
            grid_sha,
            frozen["v60_grid_sha256"],
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
        raise ValueError("V61 training input binding differs")
    implementation = repo / "src/hong2021_v61_preflight.py"
    record_path = repo / PREFLIGHT_RECORD
    if (
        sha256_file(implementation) != PREFLIGHT_IMPLEMENTATION_SHA256
        or sha256_file(record_path) != PREFLIGHT_RECORD_SHA256
    ):
        raise ValueError("V61 approved score implementation or record differs")
    record = _verified_json(record_path, PREFLIGHT_RECORD_SHA256, "preflight record")
    preflight = _verified_json(preflight_path, preflight_sha, "preflight")
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("status") != "pass"
        or preflight.get("program_sha256") != PROGRAM_SHA256
        or preflight.get("code_commit") != PREFLIGHT_CODE_COMMIT
        or canonical_digest(preflight) != PREFLIGHT_DECISION_DIGEST
        or preflight.get("grid_sha256") != grid_sha
        or preflight.get("grid_cells") != GRID_CELLS
        or preflight.get("appended_cells") != GRID_CELLS - EXISTING_CELLS
        or preflight.get("score_checkpoint_chunk_cells") != SCORE_CHUNK_CELLS
        or preflight.get("training_performed") is not False
        or preflight.get("development_accessed") is not False
        or preflight.get("historical_EAGLE_accessed") is not False
        or preflight.get("independent_gate_locked") is not True
        or record.get("status") != "complete_hard_preflight_pass_training_authorized"
        or record.get("preflight", {}).get("sha256") != preflight_sha
        or record.get("authorization", {}).get("training_allowed") is not True
        or record.get("authorization", {}).get("training_steps") != STEPS
        or not _is_ancestor(repo, PREFLIGHT_CODE_COMMIT, commit)
    ):
        raise ValueError("V61 preflight authorization differs")
    selected = _verified_json(threshold_path, threshold_sha, "V54 thresholds")
    thresholds = np.asarray(selected.get("common_log10rho_thresholds"), dtype=np.float64)
    if thresholds.shape != (4,) or not np.all(np.diff(thresholds) > 0.0):
        raise ValueError("V61 V54 thresholds differ")
    _, v35, _ = load_v56_program(_path(repo, frozen["v56_program"]), repo)
    return program, v35, grid, selected


def training_grid_score(
    parameters: torch.Tensor,
    target: torch.Tensor,
    backbone: torch.Tensor,
    target_mean: float,
    target_std: float,
    thresholds: torch.Tensor,
    weights: torch.Tensor,
    *,
    diagnostics: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if (
        target.shape != (len(parameters), 1, *parameters.shape[-3:])
        or backbone.shape != target.shape
        or thresholds.shape != (GRID_CELLS,)
        or weights.shape != thresholds.shape
        or target_std <= 0.0
    ):
        raise ValueError("V61 training grid score input differs")
    if torch.is_grad_enabled():
        score = _checkpointed_weighted_score(
            parameters,
            target,
            backbone,
            target_mean,
            target_std,
            thresholds,
            weights,
        )
    else:
        raw = _raw_survival_components(
            parameters, target, backbone, target_mean, target_std, thresholds
        )
        score = torch.sum(weights.double() * raw) / (
            REFERENCE_PROBABILITY * (1.0 - REFERENCE_PROBABILITY)
        )
        return score, raw if diagnostics else None
    components = None
    if diagnostics:
        with torch.no_grad():
            components = _raw_survival_components(
                parameters, target, backbone, target_mean, target_std, thresholds
            )
    return score, components


def composite_training_loss(
    parameters: torch.Tensor,
    target: torch.Tensor,
    backbone: torch.Tensor,
    target_mean: float,
    target_std: float,
    v54_thresholds: torch.Tensor,
    grid_thresholds: torch.Tensor,
    grid_weights: torch.Tensor,
    *,
    diagnostics: bool,
) -> tuple[torch.Tensor, ...]:
    nll = -bounded_mixture_log_probability(parameters, target).mean()
    tail, tail_components = physical_tail_brier_score(
        parameters, target, backbone, target_mean, target_std, v54_thresholds
    )
    grid, grid_components = training_grid_score(
        parameters,
        target,
        backbone,
        target_mean,
        target_std,
        grid_thresholds,
        grid_weights,
        diagnostics=diagnostics,
    )
    total = nll + TAIL_COEFFICIENT * tail + GRID_COEFFICIENT * grid
    return total, nll, tail, tail_components, grid, grid_components


@torch.no_grad()
def _validation_scores(
    model: LocalMixtureUNet,
    v35: dict[str, Any],
    prepared: h5py.File,
    device: torch.device,
    v54_thresholds: torch.Tensor,
    grid_thresholds: torch.Tensor,
    grid_weights: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    result: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        data, cache = _open_split(v35["development_domains"][domain], "validation")
        values = []
        try:
            count = int(v35["development_domains"][domain]["validation_objects"])
            for index in range(count):
                condition, target, backbone = condition_cube(
                    data, cache, prepared, domain, "validation", index
                )
                scores = composite_training_loss(
                    model(torch.from_numpy(condition[None]).to(device)),
                    torch.from_numpy(target[None]).to(device),
                    torch.from_numpy(backbone[None]).to(device),
                    float(prepared["target_mean"][()]),
                    float(prepared["target_std"][()]),
                    v54_thresholds,
                    grid_thresholds,
                    grid_weights,
                    diagnostics=False,
                )
                values.append(tuple(float(scores[i].cpu()) for i in (0, 1, 2, 4)))
        finally:
            data.close()
            cache.close()
        result[domain] = {
            "composite": float(np.mean([row[0] for row in values])),
            "bounded_NLL": float(np.mean([row[1] for row in values])),
            "V54_tail_score": float(np.mean([row[2] for row in values])),
            "reachable_grid_score": float(np.mean([row[3] for row in values])),
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
        raise RuntimeError("V61 training requires clean Lageunha and new outputs")
    if (
        not torch.cuda.is_available()
        or "ada" not in torch.cuda.get_device_name(0).lower()
    ):
        raise RuntimeError("V61 training requires Ada")
    program, v35, grid, selected = _load_training_inputs(
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
        raise RuntimeError("V61 architecture differs")
    ema_model.load_state_dict(model.state_dict())
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler("cuda")
    generator = np.random.default_rng(SEED)
    v54_thresholds = torch.tensor(selected["common_log10rho_thresholds"], device=device)
    grid_thresholds = torch.tensor(grid["thresholds_log10rho"], device=device)
    grid_weights = torch.tensor(grid["physical_moment_weights"], device=device)
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
            log_step = step == 1 or step % 50 == 0
            scores = composite_training_loss(
                parameters,
                target_tensor,
                backbone_tensor,
                float(prepared["target_mean"][()]),
                float(prepared["target_std"][()]),
                v54_thresholds,
                grid_thresholds,
                grid_weights,
                diagnostics=log_step,
            )
            total, nll, tail, tail_components, upper, upper_components = scores
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
            if log_step:
                assert upper_components is not None
                row = {
                    "step": step,
                    "composite_loss": float(total.detach().cpu()),
                    "bounded_NLL": float(nll.detach().cpu()),
                    "V54_tail_score": float(tail.detach().cpu()),
                    "V54_Brier_components": tail_components.detach().cpu().tolist(),
                    "reachable_grid_score": float(upper.detach().cpu()),
                    "reachable_grid_raw_Brier_components": (
                        upper_components.detach().cpu().tolist()
                    ),
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
                print("[v61-train] " + json.dumps(row), flush=True)
            if step in VALIDATION_STEPS:
                validation_rows[str(step)] = _validation_scores(
                    ema_model,
                    v35,
                    prepared,
                    device,
                    v54_thresholds,
                    grid_thresholds,
                    grid_weights,
                )
                print(
                    f"[v61-validation] step={step} "
                    + json.dumps(validation_rows[str(step)]),
                    flush=True,
                )
    finally:
        for data, cache in handles.values():
            data.close()
            cache.close()
        prepared.close()
    peak = int(torch.cuda.max_memory_allocated(device))
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
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
        "score_checkpoint_chunk_cells": SCORE_CHUNK_CELLS,
        "tail_coefficient": TAIL_COEFFICIENT,
        "grid_coefficient": GRID_COEFFICIENT,
        "ema_decay": 0.999,
        "ema_state_dict": {
            key: value.detach().cpu() for key, value in ema_model.state_dict().items()
        },
        "conditioning_cache_sha256": cache_sha,
        "preflight_sha256": preflight_sha,
        "preflight_decision_digest_sha256": PREFLIGHT_DECISION_DIGEST,
        "approved_score_implementation_sha256": PREFLIGHT_IMPLEMENTATION_SHA256,
        "support_selection_sha256": program["frozen_inputs"][
            "support_selection_sha256"
        ],
        "AMP_overflow_events": overflow_events,
        "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
        "sample_clipping": False,
        "component_scale_cap": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    partial_checkpoint = checkpoint_path.with_suffix(checkpoint_path.suffix + ".partial")
    torch.save(checkpoint, partial_checkpoint)
    os.replace(partial_checkpoint, checkpoint_path)
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete_fixed_12000_step_reachable_support_fit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "conditioning_cache_sha256": cache_sha,
        "preflight_sha256": preflight_sha,
        "preflight_decision_digest_sha256": PREFLIGHT_DECISION_DIGEST,
        "approved_score_implementation_sha256": PREFLIGHT_IMPLEMENTATION_SHA256,
        "support_selection_sha256": program["frozen_inputs"][
            "support_selection_sha256"
        ],
        "v54_threshold_selection_sha256": threshold_sha,
        "grid_sha256": grid_sha,
        "grid_cells": GRID_CELLS,
        "score_checkpoint_chunk_cells": SCORE_CHUNK_CELLS,
        "common_log10rho_thresholds": selected["common_log10rho_thresholds"],
        "grid_thresholds_log10rho": grid["thresholds_log10rho"],
        "grid_physical_moment_weights": grid["physical_moment_weights"],
        "tail_coefficient": TAIL_COEFFICIENT,
        "grid_coefficient": GRID_COEFFICIENT,
        "open_standardized_support": [LOWER_SUPPORT, UPPER_SUPPORT],
        "steps": STEPS,
        "parameters": PARAMETERS,
        "training_log": log_rows,
        "AMP_overflow_events": overflow_events,
        "AMP_overflow_count": len(overflow_events),
        "peak_allocated_bytes": peak,
        "fixed_validation_diagnostics": validation_rows,
        "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
        "likelihood_family": LIKELIHOOD_FAMILY,
        "sample_clipping": False,
        "component_scale_cap": False,
        "development_accessed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
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
