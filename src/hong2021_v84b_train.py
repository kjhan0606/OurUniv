#!/usr/bin/env python
"""Fixed train-only group-fit training loop for the V84B spliced marginal."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v48_train import condition_cube, load_cache
from hong2021_v84b_contract import DOMAIN_ORDER, load_program, validate_train_artifacts
from hong2021_v84b_network import (
    ConditionalSplicedTailUNet,
    conditional_log_probability,
    parameter_count,
)


CHECKPOINT_SCHEMA = "hong2021-v84b-group-held-out-spliced-tail-checkpoint-v1"
REPORT_SCHEMA = "hong2021-v84b-group-held-out-spliced-tail-training-report-v1"
STEPS = 12_000
SEED = 840144
EMA_DECAY = 0.999
INITIAL_LR = 2.0e-4
FINAL_LR = 2.0e-5
WEIGHT_DECAY = 1.0e-4
GRADIENT_CLIP = 1.0


def _learning_rate(step: int) -> float:
    fraction = (step - 1) / max(STEPS - 1, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return FINAL_LR + (INITIAL_LR - FINAL_LR) * cosine


def seeded_model(device: torch.device) -> ConditionalSplicedTailUNet:
    previous = torch.random.get_rng_state()
    torch.manual_seed(SEED)
    model = ConditionalSplicedTailUNet().to(device)
    torch.random.set_rng_state(previous)
    return model


def train(
    program_path: Path,
    repo: Path,
    conditioning_cache: Path,
    cache_sha256: str,
    preflight_path: Path,
    preflight_sha256: str,
    checkpoint_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    commit, clean = git_state(repo)
    if (
        not clean
        or socket.gethostname().split(".")[0].lower() != "lageunha"
        or not torch.cuda.is_available()
        or "ada" not in torch.cuda.get_device_name(0).lower()
    ):
        raise RuntimeError("V84B training requires clean frozen Lageunha Ada")
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError("V84B training refuses existing outputs")
    program, v35, partition = load_program(program_path, repo, commit)
    validate_train_artifacts(program, v35)
    frozen = program["frozen_inputs"]
    if (
        conditioning_cache.resolve() != Path(frozen["conditioning_cache"]).resolve()
        or cache_sha256 != frozen["conditioning_cache_sha256"]
        or sha256_file(conditioning_cache) != cache_sha256
    ):
        raise ValueError("V84B conditioning cache differs")
    preflight = json.loads(preflight_path.read_text())
    if (
        sha256_file(preflight_path) != preflight_sha256
        or preflight_path.resolve() != Path(program["output_roots"]["preflight"]).resolve()
        or preflight.get("schema") != "hong2021-v84b-spliced-tail-hard-preflight-v1"
        or preflight.get("status") != "pass"
        or preflight.get("program_sha256") != sha256_file(program_path)
        or preflight.get("training_performed") is not False
        or preflight.get("group_holdout_payload_accessed") is not False
        or preflight.get("independent_gate_locked") is not True
    ):
        raise ValueError("V84B preflight authorization differs")
    if (
        int(program["training"]["steps"]) != STEPS
        or int(program["training"]["seed"]) != SEED
        or program["training"]["checkpoint_selection"] != "fixed_step_12000_EMA_only"
        or program["training"]["objective"] != "proper_spliced_conditional_NLL_only"
    ):
        raise ValueError("V84B frozen training contract differs")
    device = torch.device("cuda")
    model = seeded_model(device)
    ema_model = seeded_model(device)
    expected_parameters = int(program["model"]["trainable_parameters"])
    if parameter_count(model) != expected_parameters:
        raise RuntimeError("V84B parameter count differs")
    ema_model.load_state_dict(model.state_dict())
    ema_model.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda")
    generator = np.random.default_rng(SEED)
    prepared = load_cache(conditioning_cache, cache_sha256, commit)
    handles = {
        domain: _open_split(v35["development_domains"][domain], "train")
        for domain in DOMAIN_ORDER
    }
    log_rows: list[dict[str, Any]] = []
    overflow_events: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        for step in range(1, STEPS + 1):
            conditions: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            selected: dict[str, int] = {}
            isometries: dict[str, int] = {}
            for domain in DOMAIN_ORDER:
                fit = partition[domain]["fit"]
                index = int(fit[int(generator.integers(len(fit)))])
                isometry = int(generator.integers(len(CUBE_ISOMETRIES)))
                axes, reflections = CUBE_ISOMETRIES[isometry]
                data, cache = handles[domain]
                condition, target, _ = condition_cube(
                    data, cache, prepared, domain, "train", index
                )
                conditions.append(apply_cube_isometry(condition, axes, reflections))
                targets.append(apply_cube_isometry(target, axes, reflections))
                selected[domain] = index
                isometries[domain] = isometry
            condition_tensor = torch.from_numpy(np.stack(conditions)).to(device)
            target_tensor = torch.from_numpy(np.stack(targets)).to(device)
            learning_rate = _learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                parameters = model(condition_tensor)
            log_probability = conditional_log_probability(parameters, target_tensor)
            loss = -log_probability.mean()
            domain_nll = -log_probability.mean(dim=(1, 2, 3, 4))
            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP
            )
            finite_gradient = bool(torch.isfinite(gradient_norm).detach().cpu())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            overflow = scale_after < scale_before
            if overflow:
                overflow_events.append(
                    {
                        "step": step,
                        "scale_before": scale_before,
                        "scale_after": scale_after,
                        "gradient_norm_is_finite": finite_gradient,
                    }
                )
            with torch.no_grad():
                for ema, current in zip(
                    ema_model.parameters(), model.parameters(), strict=True
                ):
                    ema.lerp_(current.detach(), 1.0 - EMA_DECAY)
                for ema, current in zip(
                    ema_model.buffers(), model.buffers(), strict=True
                ):
                    ema.copy_(current)
            if step == 1 or step % 50 == 0:
                row = {
                    "step": step,
                    "conditional_NLL": float(loss.detach().cpu()),
                    "domain_NLL": {
                        domain: float(domain_nll[position].detach().cpu())
                        for position, domain in enumerate(DOMAIN_ORDER)
                    },
                    "selected_group_fit_indices": selected,
                    "cube_isometries": isometries,
                    "learning_rate": learning_rate,
                    "gradient_norm_before_clip": (
                        float(gradient_norm.detach().cpu())
                        if finite_gradient
                        else "nonfinite"
                    ),
                    "AMP_scale_before_update": scale_before,
                    "AMP_scale_after_update": scale_after,
                    "AMP_update_skipped": overflow,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                }
                log_rows.append(row)
                print("[v84b-train] " + json.dumps(row), flush=True)
    finally:
        for data, cache in handles.values():
            data.close()
            cache.close()
        prepared.close()
    common = {
        "program_sha256": sha256_file(program_path),
        "code_commit": commit,
        "steps": STEPS,
        "seed": SEED,
        "parameters": expected_parameters,
        "objective": "proper_spliced_conditional_NLL_only",
        "partition_sha256": program["partition"]["sha256"],
        "conditioning_cache_sha256": cache_sha256,
        "preflight_sha256": preflight_sha256,
        "EMA_decay": EMA_DECAY,
        "group_holdout_payload_accessed": False,
        "validation_payload_accessed": False,
        "consumed_development_payload_accessed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        **common,
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
        "status": "complete_fixed_12000_step_group_fit_spliced_tail",
        **common,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_log": log_rows,
        "AMP_overflow_count": len(overflow_events),
        "AMP_overflow_events": overflow_events,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "holdout_used_for_stopping_or_checkpoint_selection": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial_report = report_path.with_suffix(report_path.suffix + ".partial")
    partial_report.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial_report, report_path)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--conditioning-cache", type=Path, required=True)
    parser.add_argument("--conditioning-cache-sha256", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    train(
        args.program,
        args.repo,
        args.conditioning_cache,
        args.conditioning_cache_sha256,
        args.preflight,
        args.preflight_sha256,
        args.checkpoint,
        args.report,
    )


if __name__ == "__main__":
    main()
