#!/usr/bin/env python
"""Hard preflight for the frozen V61 reachable-support survival score."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v48_train import condition_cube
from hong2021_v50_network import (
    INITIAL_BIASES,
    LOWER_SUPPORT,
    UPPER_SUPPORT,
    bounded_mixture_cdf,
    bounded_mixture_log_probability,
    parameter_count,
)
from hong2021_v54_train import TAIL_COEFFICIENT, _same_seed_model, physical_tail_brier_score
from hong2021_v56_train import (
    GRID_COEFFICIENT,
    PARAMETERS,
    REFERENCE_PROBABILITY,
    load_cache,
    load_program as load_v56_program,
)


PROGRAM_SHA256 = "327d750774a82885888ff08313e829462d43c877d45effc990f1035358f04cd1"
PROGRAM_SCHEMA = "hong2021-v61-reachable-support-bounded-mixture-program-v1"
SCHEMA = "hong2021-v61-reachable-support-hard-preflight-v1"
GRID_CELLS = 134
EXISTING_CELLS = 16
SCORE_CHUNK_CELLS = 8


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V61 {label} hash differs")
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _path(repo: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (repo / candidate).resolve()


def load_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status")
        != "frozen_before_preflight_model_implementation_training_or_evaluation"
    ):
        raise ValueError("V61 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _path(repo, parent["v60_record"]), parent["v60_record_sha256"], "V60 record"
    )
    grid_row = record.get("grid", {})
    firewall = record.get("firewall", {})
    if (
        record.get("status") != parent["required_status"]
        or grid_row.get("total_cells") != parent["required_total_cells"]
        or grid_row.get("existing_thresholds_byte_equal")
        is not parent["required_existing_thresholds_byte_equal"]
        or grid_row.get("final_threshold_equals_global_reachable_upper")
        is not parent["required_final_threshold_equals_global_reachable_upper"]
        or firewall.get("development_accessed")
        is not parent["required_development_accessed"]
        or firewall.get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
    ):
        raise ValueError("V61 parent grid or firewall differs")
    frozen = program["frozen_inputs"]
    for key in (
        "v56_program",
        "v56_grid",
        "v56_preflight",
        "v60_grid",
        "v54_threshold_selection",
        "conditioning_cache",
        "support_selection",
    ):
        if sha256_file(_path(repo, frozen[key])) != frozen[f"{key}_sha256"]:
            raise ValueError(f"V61 frozen input differs: {key}")
    v56_preflight = _verified_json(
        _path(repo, frozen["v56_preflight"]),
        frozen["v56_preflight_sha256"],
        "V56 preflight",
    )
    v56_grid = _verified_json(
        _path(repo, frozen["v56_grid"]), frozen["v56_grid_sha256"], "V56 grid"
    )
    v60_grid = _verified_json(
        _path(repo, frozen["v60_grid"]), frozen["v60_grid_sha256"], "V60 grid"
    )
    if (
        canonical_digest(v56_preflight)
        != frozen["v56_preflight_decision_digest_sha256"]
        or canonical_digest(v60_grid) != frozen["v60_grid_decision_digest_sha256"]
        or v56_preflight.get("status") != "pass"
        or v56_preflight.get("development_accessed") is not False
        or v60_grid.get("status") != "complete_reachable_support_grid"
        or v60_grid.get("total_cells") != GRID_CELLS
        or v60_grid.get("development_accessed") is not False
        or v60_grid.get("independent_gate_locked") is not True
    ):
        raise ValueError("V61 inherited preflight or grid digest differs")
    old_thresholds = np.asarray(v56_grid["thresholds_log10rho"], dtype=np.float64)
    thresholds = np.asarray(v60_grid["thresholds_log10rho"], dtype=np.float64)
    weights = np.asarray(v60_grid["physical_moment_weights"], dtype=np.float64)
    change = program["single_model_change"]
    if (
        thresholds.shape != (GRID_CELLS,)
        or weights.shape != thresholds.shape
        or not np.array_equal(thresholds[:EXISTING_CELLS], old_thresholds)
        or thresholds[-1] != float(change["final_threshold_log10rho"])
        or not np.all(np.diff(thresholds) > 0.0)
        or not np.all(weights > 0.0)
        or abs(float(weights.sum(dtype=np.float64)) - 1.0) > 1.0e-12
        or float(change["upper_survival_score_coefficient"]) != GRID_COEFFICIENT
        or float(change["reference_probability"]) != REFERENCE_PROBABILITY
    ):
        raise ValueError("V61 single grid change differs")
    return program, v56_preflight, v60_grid


def _raw_survival_components(
    parameters: torch.Tensor,
    target: torch.Tensor,
    backbone: torch.Tensor,
    target_mean: float,
    target_std: float,
    thresholds_log10rho: torch.Tensor,
) -> torch.Tensor:
    components = []
    for threshold in thresholds_log10rho:
        physical_y = threshold.double() / 4.5
        standardized = (physical_y - backbone.double() - float(target_mean)) / float(
            target_std
        )
        below = standardized <= LOWER_SUPPORT
        above = standardized >= UPPER_SUPPORT
        interior = standardized.clamp(LOWER_SUPPORT + 1.0e-6, UPPER_SUPPORT - 1.0e-6)
        exceedance = 1.0 - bounded_mixture_cdf(parameters, interior)
        exceedance = torch.where(below, torch.ones_like(exceedance), exceedance)
        exceedance = torch.where(above, torch.zeros_like(exceedance), exceedance)
        observed = (target.double() > standardized).float()
        raw = torch.square(exceedance - observed).mean()
        if not torch.isfinite(raw):
            raise RuntimeError("V61 nonfinite survival Brier component")
        components.append(raw)
    return torch.stack(components)


def _checkpointed_weighted_score(
    parameters: torch.Tensor,
    target: torch.Tensor,
    backbone: torch.Tensor,
    target_mean: float,
    target_std: float,
    thresholds: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    normalization = REFERENCE_PROBABILITY * (1.0 - REFERENCE_PROBABILITY)
    chunks = []
    for start in range(0, len(thresholds), SCORE_CHUNK_CELLS):
        stop = min(start + SCORE_CHUNK_CELLS, len(thresholds))
        threshold_chunk = thresholds[start:stop]
        weight_chunk = weights[start:stop]

        def weighted_chunk(
            value: torch.Tensor,
            selected_thresholds: torch.Tensor = threshold_chunk,
            selected_weights: torch.Tensor = weight_chunk,
        ) -> torch.Tensor:
            raw = _raw_survival_components(
                value,
                target,
                backbone,
                target_mean,
                target_std,
                selected_thresholds,
            )
            return torch.sum(selected_weights.double() * raw) / normalization

        chunks.append(checkpoint(weighted_chunk, parameters, use_reentrant=False))
    return torch.stack(chunks).sum()


def reachable_survival_grid_score(
    parameters: torch.Tensor,
    target: torch.Tensor,
    backbone: torch.Tensor,
    target_mean: float,
    target_std: float,
    thresholds_log10rho: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        target.shape != (len(parameters), 1, *parameters.shape[-3:])
        or backbone.shape != target.shape
        or thresholds_log10rho.shape != (GRID_CELLS,)
        or weights.shape != thresholds_log10rho.shape
        or target_std <= 0.0
    ):
        raise ValueError("V61 reachable survival score input differs")
    existing = _checkpointed_weighted_score(
        parameters,
        target,
        backbone,
        target_mean,
        target_std,
        thresholds_log10rho[:EXISTING_CELLS],
        weights[:EXISTING_CELLS],
    )
    appended = _checkpointed_weighted_score(
        parameters,
        target,
        backbone,
        target_mean,
        target_std,
        thresholds_log10rho[EXISTING_CELLS:],
        weights[EXISTING_CELLS:],
    )
    with torch.no_grad():
        stacked = _raw_survival_components(
            parameters,
            target,
            backbone,
            target_mean,
            target_std,
            thresholds_log10rho,
        )
    return existing + appended, stacked, existing, appended


def composite_loss(
    parameters: torch.Tensor,
    target: torch.Tensor,
    backbone: torch.Tensor,
    target_mean: float,
    target_std: float,
    v54_thresholds: torch.Tensor,
    grid_thresholds: torch.Tensor,
    grid_weights: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    nll = -bounded_mixture_log_probability(parameters, target).mean()
    tail, tail_components = physical_tail_brier_score(
        parameters, target, backbone, target_mean, target_std, v54_thresholds
    )
    grid, grid_components, existing, appended = reachable_survival_grid_score(
        parameters,
        target,
        backbone,
        target_mean,
        target_std,
        grid_thresholds,
        grid_weights,
    )
    total = nll + TAIL_COEFFICIENT * tail + GRID_COEFFICIENT * grid
    return total, nll, tail, tail_components, grid, grid_components, existing, appended


def preflight(program_path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, v56_preflight, grid = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if not clean or socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V61 preflight requires clean Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V61 preflight requires Ada")
    frozen = program["frozen_inputs"]
    _, v35, _ = load_v56_program(_path(repo, frozen["v56_program"]), repo)
    prepared = load_cache(
        _path(repo, frozen["conditioning_cache"]),
        frozen["conditioning_cache_sha256"],
        commit,
    )
    device = torch.device("cuda")
    handles = []
    conditions, targets, backbones = [], [], []
    try:
        for domain in DOMAIN_ORDER:
            data, cache = _open_split(v35["development_domains"][domain], "train")
            handles.append((data, cache))
            condition, target, backbone = condition_cube(
                data, cache, prepared, domain, "train", 0
            )
            axes, reflections = CUBE_ISOMETRIES[7]
            conditions.append(apply_cube_isometry(condition, axes, reflections))
            targets.append(apply_cube_isometry(target, axes, reflections))
            backbones.append(apply_cube_isometry(backbone, axes, reflections))
        torch.cuda.reset_peak_memory_stats(device)
        model = _same_seed_model(device)
        if parameter_count(model) != PARAMETERS or not all(
            parameter.requires_grad for parameter in model.parameters()
        ):
            raise RuntimeError("V61 architecture differs")
        condition_tensor = torch.from_numpy(np.stack(conditions)).to(device)
        target_tensor = torch.from_numpy(np.stack(targets)).to(device)
        backbone_tensor = torch.from_numpy(np.stack(backbones)).to(device)
        selected = _verified_json(
            _path(repo, frozen["v54_threshold_selection"]),
            frozen["v54_threshold_selection_sha256"],
            "V54 thresholds",
        )
        v54_thresholds = torch.tensor(selected["common_log10rho_thresholds"], device=device)
        grid_thresholds = torch.tensor(grid["thresholds_log10rho"], device=device)
        grid_weights = torch.tensor(grid["physical_moment_weights"], device=device)
        parameters = model(condition_tensor)
        expected = torch.tensor(INITIAL_BIASES, device=device).reshape(1, 15, 1, 1, 1)
        initialization_error = float(
            torch.max(torch.abs(parameters - expected)).detach().cpu()
        )
        scores = composite_loss(
            parameters,
            target_tensor,
            backbone_tensor,
            float(prepared["target_mean"][()]),
            float(prepared["target_std"][()]),
            v54_thresholds,
            grid_thresholds,
            grid_weights,
        )
        total, nll, tail, tail_components, upper, upper_components, existing, appended = scores
        base = nll + TAIL_COEFFICIENT * tail
        identity_error = float(torch.abs(total - base - GRID_COEFFICIENT * upper).cpu())
        base_differences = {
            "bounded_NLL": abs(
                float(nll.detach().cpu())
                - float(v56_preflight["real_source_balanced_bounded_NLL"])
            ),
            "V54_tail_score": abs(
                float(tail.detach().cpu())
                - float(v56_preflight["real_source_balanced_V54_tail_score"])
            ),
            "V54_Brier_components": float(
                np.max(
                    np.abs(
                        tail_components.detach().cpu().numpy()
                        - np.asarray(v56_preflight["real_source_balanced_V54_Brier_components"])
                    )
                )
            ),
        }
        appended_gradient = torch.autograd.grad(
            appended, parameters, retain_graph=True, allow_unused=False
        )[0]
        appended_gradient_finite = bool(torch.isfinite(appended_gradient).all().cpu())
        appended_gradient_maximum = float(torch.max(torch.abs(appended_gradient)).cpu())
        total.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        peak = int(torch.cuda.max_memory_allocated(device))
        limit = int(program["hard_preflight"]["peak_allocated_bytes_limit"])
        if (
            initialization_error > 1.0e-7
            or torch.count_nonzero(model.output.weight)
            or not torch.isfinite(total)
            or identity_error > 1.0e-7
            or max(base_differences.values()) > 1.0e-7
            or not appended_gradient_finite
            or appended_gradient_maximum == 0.0
            or not gradients
            or not all(torch.isfinite(gradient).all() for gradient in gradients)
            or not any(torch.count_nonzero(gradient) for gradient in gradients)
            or peak >= limit
        ):
            raise RuntimeError("V61 score, gradient, identity, base reproduction, or memory differs")
    finally:
        for data, cache in handles:
            data.close()
            cache.close()
        prepared.close()
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pass",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "parameters": PARAMETERS,
        "grid_sha256": frozen["v60_grid_sha256"],
        "grid_cells": GRID_CELLS,
        "existing_cells": EXISTING_CELLS,
        "appended_cells": GRID_CELLS - EXISTING_CELLS,
        "score_checkpoint_chunk_cells": SCORE_CHUNK_CELLS,
        "grid_thresholds_log10rho": grid["thresholds_log10rho"],
        "grid_physical_moment_weights": grid["physical_moment_weights"],
        "tail_coefficient": TAIL_COEFFICIENT,
        "grid_coefficient": GRID_COEFFICIENT,
        "real_source_balanced_composite_loss": float(total.detach().cpu()),
        "real_source_balanced_bounded_NLL": float(nll.detach().cpu()),
        "real_source_balanced_V54_tail_score": float(tail.detach().cpu()),
        "real_source_balanced_V54_Brier_components": tail_components.detach().cpu().tolist(),
        "real_source_balanced_reachable_grid_score": float(upper.detach().cpu()),
        "real_source_balanced_existing_16_weighted_score": float(existing.detach().cpu()),
        "real_source_balanced_appended_118_weighted_score": float(appended.detach().cpu()),
        "real_source_balanced_grid_raw_Brier_components": upper_components.detach().cpu().tolist(),
        "base_score_absolute_differences_from_V56_preflight": base_differences,
        "appended_score_parameter_gradient_finite": appended_gradient_finite,
        "appended_score_parameter_gradient_maximum_absolute": appended_gradient_maximum,
        "composite_identity_absolute_error": identity_error,
        "initial_output_maximum_error": initialization_error,
        "peak_allocated_bytes": peak,
        "peak_allocated_bytes_limit": limit,
        "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
        "training_performed": False,
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
        raise FileExistsError("V61 refuses an existing preflight")
    result = preflight(args.program, args.repo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
