#!/usr/bin/env python
"""Measure V26 scale-gradient conditioning after the frozen failure.

This is a read-only development audit.  It evaluates deterministic first-two
object batches from each of the three already-open train and validation
domains, and never opens Astrid or historical EAGLE.  The purpose is to decide
whether the Haar-scale failure can be attributed to the registered
dimension-weighted optimizer geometry before designing another model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from hong2021_v14_edm import V14ResidualDataset
from hong2021_v15_edm import git_state
from hong2021_v18_init import sha256_file
from hong2021_v26 import (
    CANDIDATE_STEPS,
    DETAIL_DIMENSIONS_COARSE_TO_FINE,
    MODEL_SCHEMA,
    PARAMETERS,
    REGISTRY_SHA256,
    _paths,
    _validate_checkpoint,
    build_model,
    load_frozen_program,
)


SCHEMA = "hong2021-v26-gradient-scale-conditioning-audit-v1"
TRAINING = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/training/"
    "tng100_simba_swift_v26_e14_conditional_haar_flow"
)
MECHANISM_AUDIT = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/"
    "tng100_simba_swift_v26_e14_conditional_haar_flow/"
    "trained_flow_mechanism_audit_v2.json"
)
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift-EAGLE")


def _norm(values: Iterable[torch.Tensor]) -> float:
    square = sum(float(value.detach().double().square().sum()) for value in values)
    return math.sqrt(square)


def summarize_gradient_scales(
    normalized_gradient_norms: list[float],
    parameter_norms: list[float],
    *,
    dimensions: tuple[int, ...] = DETAIL_DIMENSIONS_COARSE_TO_FINE,
    weight_decay: float,
    clip_threshold: float,
) -> dict[str, Any]:
    """Convert equal-scale gradients to the registered objective geometry."""
    if not (
        len(normalized_gradient_norms) == len(parameter_norms) == len(dimensions)
    ):
        raise ValueError("gradient, parameter, and dimension scales differ")
    total_dimensions = float(sum(dimensions))
    shares = [dimension / total_dimensions for dimension in dimensions]
    registered = [
        gradient * share
        for gradient, share in zip(normalized_gradient_norms, shares, strict=True)
    ]
    global_norm = math.sqrt(sum(value * value for value in registered))
    clip_factor = min(1.0, clip_threshold / max(global_norm, np.finfo(float).tiny))
    rows = []
    for index, (dimension, share, normalized, parameter, actual) in enumerate(
        zip(
            dimensions,
            shares,
            normalized_gradient_norms,
            parameter_norms,
            registered,
            strict=True,
        )
    ):
        decay = weight_decay * parameter
        rows.append(
            {
                "coarse_to_fine_index": index,
                "dimensions": dimension,
                "objective_dimension_fraction": share,
                "equal_scale_nll_gradient_norm": normalized,
                "registered_objective_gradient_norm": actual,
                "registered_gradient_fraction_of_global_l2": actual
                / max(global_norm, np.finfo(float).tiny),
                "parameter_norm": parameter,
                "weight_decay_gradient_norm_proxy": decay,
                "weight_decay_over_registered_gradient_norm_proxy": decay
                / max(actual, np.finfo(float).tiny),
                "post_global_clip_gradient_norm": actual * clip_factor,
            }
        )
    return {
        "global_registered_gradient_norm": global_norm,
        "fixed_clip_threshold": clip_threshold,
        "global_clip_factor": clip_factor,
        "scales": rows,
    }


def _fixed_batch(
    paths: dict[str, tuple[str, str, str, str]], split: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[int]]]:
    offset = 0 if split == "train" else 2
    conditions = []
    latents = []
    indices = {}
    for domain in DOMAIN_ORDER:
        dataset = V14ResidualDataset(
            paths[domain][offset], paths[domain][offset + 1], False
        )
        selected = [0, 1]
        indices[domain] = selected
        for index in selected:
            condition, latent, _, _ = dataset[index]
            conditions.append(condition)
            latents.append(latent)
    return (
        torch.stack(conditions).to(device),
        torch.stack(latents).to(device),
        indices,
    )


def _parameter_norms(model: torch.nn.Module) -> list[float]:
    return [_norm(flow.parameters()) for flow in model.flows]


def _batch_gradient_audit(
    model: torch.nn.Module,
    condition: torch.Tensor,
    latent: torch.Tensor,
    *,
    weight_decay: float,
    clip_threshold: float,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    log_prob, diagnostic = model.log_prob(latent, condition)
    scale_log_prob = diagnostic["scale_log_prob_coarse_to_fine"]
    normalized_losses = []
    gradient_norms = []
    for scale, (flow, dimension) in enumerate(
        zip(model.flows, DETAIL_DIMENSIONS_COARSE_TO_FINE, strict=True)
    ):
        loss = -scale_log_prob[:, scale].mean() / dimension
        gradients = torch.autograd.grad(
            loss,
            tuple(flow.parameters()),
            retain_graph=scale + 1 < len(model.flows),
            allow_unused=False,
        )
        normalized_losses.append(float(loss.detach()))
        gradient_norms.append(_norm(gradients))
    summary = summarize_gradient_scales(
        gradient_norms,
        _parameter_norms(model),
        weight_decay=weight_decay,
        clip_threshold=clip_threshold,
    )
    summary.update(
        {
            "objects": len(latent),
            "registered_nll_per_non_dc_dimension": float(
                (-log_prob.mean() / sum(DETAIL_DIMENSIONS_COARSE_TO_FINE)).detach()
            ),
            "equal_scale_nll_coarse_to_fine": normalized_losses,
            "all_losses_finite": bool(
                torch.isfinite(log_prob).all()
                and torch.isfinite(scale_log_prob).all()
            ),
        }
    )
    return summary


def _interpret(candidates: dict[str, Any]) -> dict[str, Any]:
    final = candidates["30000"]
    rows = final["train"]["scales"]
    coarse_decay_ratios = [
        float(row["weight_decay_over_registered_gradient_norm_proxy"])
        for row in rows[:2]
    ]
    finest_decay_ratio = float(
        rows[-1]["weight_decay_over_registered_gradient_norm_proxy"]
    )
    registered_fractions = [
        float(row["registered_gradient_fraction_of_global_l2"]) for row in rows
    ]
    coarse_gradient_suppressed = max(registered_fractions[:2]) < 0.05
    decay_disproportionate = min(coarse_decay_ratios) > 10.0 * finest_decay_ratio
    if coarse_gradient_suppressed and decay_disproportionate:
        classification = "registered_optimizer_geometry_suppresses_coarse_scale_learning"
        next_step = (
            "a separately frozen per-scale optimizer preconditioning control is "
            "eligible before abandoning the conditional representation"
        )
    elif coarse_gradient_suppressed:
        classification = "coarse_scale_gradient_suppressed_without_decay_dominance"
        next_step = "audit global clipping and Adam second moments before a new model"
    else:
        classification = "coarse_failure_not_explained_by_registered_gradient_magnitude"
        next_step = "reassess conditional context and train-validation representation shift"
    return {
        "classification": classification,
        "next": next_step,
        "coarse_registered_gradient_fraction_of_global_l2": registered_fractions[:2],
        "finest_registered_gradient_fraction_of_global_l2": registered_fractions[-1],
        "coarse_weight_decay_over_gradient_proxy": coarse_decay_ratios,
        "finest_weight_decay_over_gradient_proxy": finest_decay_ratio,
        "coarse_gradient_suppressed": coarse_gradient_suppressed,
        "coarse_decay_disproportionate": decay_disproportionate,
        "final_train_global_clip_factor": final["train"]["global_clip_factor"],
        "final_validation_global_clip_factor": final["validation"][
            "global_clip_factor"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--training", type=Path, default=TRAINING)
    parser.add_argument("--mechanism-audit", type=Path, default=MECHANISM_AUDIT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if socket.gethostname().lower() != "lageunha":
        raise RuntimeError("V26 gradient-scale audit requires Lageunha")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V26 gradient-scale audit requires the Ada GPU")
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V26 gradient-scale audit requires a clean worktree")
    output = args.out.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise RuntimeError(f"refusing to overwrite V26 gradient audit: {output}")
    registry, artifacts, v20, _, haar = load_frozen_program(
        args.registry.resolve(), repo
    )
    mechanism = json.loads(args.mechanism_audit.read_text())
    if (
        mechanism.get("schema") != "hong2021-v26-trained-flow-mechanism-audit-v2"
        or mechanism.get("registry_sha256") != REGISTRY_SHA256
        or mechanism.get("Astrid_accessed") is not False
    ):
        raise ValueError("V26 mechanism-audit provenance differs")
    run = json.loads((args.training / "run.json").read_text())
    if (
        run.get("schema") != MODEL_SCHEMA
        or run.get("status") != "complete"
        or run.get("parameters") != PARAMETERS
    ):
        raise ValueError("V26 completed training provenance differs")
    protocol = registry["training_protocol"]
    device = torch.device(args.device)
    paths = _paths(artifacts, v20)
    train_condition, train_latent, train_indices = _fixed_batch(
        paths, "train", device
    )
    validation_condition, validation_latent, validation_indices = _fixed_batch(
        paths, "validation", device
    )
    candidates = {}
    for step in CANDIDATE_STEPS:
        checkpoint_path = (
            args.training / "validation_checkpoints" / f"step_{step:06d}.pt"
        )
        checkpoint, checkpoint_sha = _validate_checkpoint(
            checkpoint_path, step=step, artifacts=artifacts
        )
        model = build_model(
            haar, checkpoint["observable_context_features"], device=device
        )
        model.load_state_dict(checkpoint["ema_model"])
        model.eval()
        candidates[str(step)] = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "train": _batch_gradient_audit(
                model,
                train_condition,
                train_latent,
                weight_decay=float(protocol["weight_decay"]),
                clip_threshold=float(protocol["gradient_clip"]),
            ),
            "validation": _batch_gradient_audit(
                model,
                validation_condition,
                validation_latent,
                weight_decay=float(protocol["weight_decay"]),
                clip_threshold=float(protocol["gradient_clip"]),
            ),
        }
        print(f"[gradient-audit] step={step}", flush=True)
        del model
        torch.cuda.empty_cache()
    report = {
        "schema": SCHEMA,
        "status": "complete_development_only_post_failure_audit",
        "registry": str(args.registry.resolve()),
        "registry_sha256": REGISTRY_SHA256,
        "training": str(args.training.resolve()),
        "training_run_sha256": sha256_file(args.training / "run.json"),
        "mechanism_audit": str(args.mechanism_audit.resolve()),
        "mechanism_audit_sha256": sha256_file(args.mechanism_audit),
        "fixed_batch_rule": "indices 0 and 1 from each source, no augmentation",
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "execution_host": socket.gethostname(),
        "execution_gpu": torch.cuda.get_device_name(0),
        "audit_code_commit": commit,
        "worktree_clean_at_audit": clean,
        "candidates": candidates,
        "interpretation": _interpret(candidates),
        "model_or_optimizer_changed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(report["interpretation"], indent=2), flush=True)
    print(f"[out] {output}", flush=True)


if __name__ == "__main__":
    main()
