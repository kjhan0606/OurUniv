#!/usr/bin/env python
"""Materialize, preflight, and train the frozen V56 upper survival grid."""
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

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v48_train import condition_cube, load_cache
from hong2021_v50_network import (
    INITIAL_BIASES,
    LOWER_SUPPORT,
    UPPER_SUPPORT,
    LocalMixtureUNet,
    bounded_mixture_cdf,
    bounded_mixture_log_probability,
    parameter_count,
)
from hong2021_v50_train import PARAMETERS, SUPPORT_SHA256, _device
from hong2021_v54_train import (
    QUANTILES,
    TAIL_COEFFICIENT,
    _learning_rate,
    _same_seed_model,
    load_program as load_v54_program,
    physical_tail_brier_score,
)


PROGRAM_SCHEMA = "hong2021-v56-upper-survival-grid-bounded-mixture-program-v1"
PROGRAM_SHA256 = "ec93d8d0894292793279edafad5e8243a272e2c55d52d4c27b3a0cd9ef714f40"
GRID_SCHEMA = "hong2021-v56-upper-survival-grid-v1"
PREFLIGHT_SCHEMA = "hong2021-v56-upper-survival-grid-hard-preflight-v1"
CHECKPOINT_SCHEMA = "hong2021-v56-upper-survival-grid-checkpoint-v1"
REPORT_SCHEMA = "hong2021-v56-upper-survival-grid-training-report-v1"
LIKELIHOOD_FAMILY = "bounded_logit_Gaussian_mixture"
GRID_CELLS = 16
GRID_COEFFICIENT = 0.1
REFERENCE_PROBABILITY = 1.0e-5
STEPS = 12_000
VALIDATION_STEPS = (4_000, 8_000, 12_000)
SEED = 144_044


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V56 {label} hash differs")
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _resolve(repo: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (repo / candidate).resolve()


def load_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status")
        != "frozen_before_grid_materialization_model_implementation_training_or_evaluation"
    ):
        raise ValueError("V56 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _resolve(repo, parent["v55_record"]), parent["v55_record_sha256"], "V55 record"
    )
    audit_row = record.get("audit", {})
    decomposition = record.get("fixed_threshold_probability_amplitude_decomposition", {})
    common_amplitude = all(
        decomposition.get(domain, {}).get("beyond_q99_999_amplitude_dominates") is True
        for domain in DOMAIN_ORDER
    )
    if (
        record.get("status") != parent["required_status"]
        or audit_row.get("classification") != parent["required_classification"]
        or audit_row.get("next") != parent["required_next"]
        or common_amplitude is not parent["required_common_amplitude_dominance"]
        or record.get("firewall", {}).get("development_accessed")
        is not parent["required_development_accessed"]
        or record.get("firewall", {}).get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
    ):
        raise ValueError("V56 V55 parent evidence differs")
    frozen = program["frozen_inputs"]
    for key in (
        "v54_program",
        "v54_record",
        "v54_threshold_selection",
        "v54_preflight",
        "v54_train_gate",
        "v55_audit",
        "conditioning_cache",
        "support_selection",
        "v50_checkpoint",
        "v50_training_report",
        "v50_development_decision",
        "v52_development_decision",
    ):
        if sha256_file(_resolve(repo, frozen[key])) != frozen[f"{key}_sha256"]:
            raise ValueError(f"V56 frozen input differs: {key}")
    for key, digest_key in (
        ("v54_threshold_selection", "v54_threshold_selection_decision_digest_sha256"),
        ("v55_audit", "v55_audit_decision_digest_sha256"),
    ):
        value = _verified_json(_resolve(repo, frozen[key]), frozen[f"{key}_sha256"], key)
        if canonical_digest(value) != frozen[digest_key]:
            raise ValueError(f"V56 frozen digest differs: {key}")
    audit = json.loads(_resolve(repo, frozen["v55_audit"]).read_text())
    if (
        audit.get("classification") != parent["required_classification"]
        or audit.get("development_accessed") is not False
        or audit.get("historical_EAGLE_accessed") is not False
        or audit.get("independent_gate_locked") is not True
    ):
        raise ValueError("V56 V55 audit firewall differs")
    v54, v35, v41 = load_v54_program(_resolve(repo, frozen["v54_program"]), repo)
    effective = dict(program)
    effective["inherited_inputs"] = v54["inherited_inputs"]
    return effective, v35, v41


def grid_values(lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
        raise ValueError("V56 grid edges differ")
    edges = np.linspace(lower, upper, GRID_CELLS + 1, dtype=np.float64)
    thresholds = edges[1:]
    proxy = np.square(np.power(10.0, edges) - 1.0)
    weights = np.diff(proxy) / (proxy[-1] - proxy[0])
    if (
        thresholds.shape != (GRID_CELLS,)
        or weights.shape != thresholds.shape
        or not np.all(np.diff(thresholds) > 0.0)
        or not np.all(weights > 0.0)
        or abs(float(weights.sum(dtype=np.float64)) - 1.0) > 1.0e-12
        or thresholds[-1] != upper
    ):
        raise RuntimeError("V56 grid materialization differs")
    return thresholds, weights


def materialize_grid(
    program_path: Path,
    repo: Path,
    threshold_path: Path,
    threshold_sha: str,
    output: Path,
) -> dict[str, Any]:
    program, v35, _ = load_program(program_path, repo.resolve())
    commit, clean = git_state(repo.resolve())
    if not clean or output.exists():
        raise RuntimeError("V56 grid materialization requires clean new output")
    frozen = program["frozen_inputs"]
    if (
        threshold_path.resolve() != _resolve(repo, frozen["v54_threshold_selection"])
        or threshold_sha != frozen["v54_threshold_selection_sha256"]
    ):
        raise ValueError("V56 threshold binding differs")
    selected = _verified_json(threshold_path, threshold_sha, "V54 thresholds")
    lower = float(selected["common_log10rho_thresholds"][3])
    maxima = [float(selected["domains"][domain]["maximum_log10rho"]) for domain in DOMAIN_ORDER]
    upper = max(maxima)
    rule = program["upper_survival_grid"]
    if lower != rule["lower_edge_value"] or upper != rule["upper_edge_value"]:
        raise ValueError("V56 immutable grid edge differs")
    thresholds, weights = grid_values(lower, upper)
    domains: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        data, cache = _open_split(row, "train")
        try:
            objects = int(row["train_objects"])
            target = np.asarray(data["target"][:objects, 0], dtype=np.float32)
        finally:
            data.close()
            cache.close()
        log10rho = 4.5 * target.astype(np.float64)
        maximum = float(log10rho.max())
        exceedances = int(np.count_nonzero(log10rho > upper))
        if maximum != maxima[DOMAIN_ORDER.index(domain)] or exceedances != 0:
            raise RuntimeError("V56 immutable train maximum confirmation differs")
        domains[domain] = {
            "train_objects": objects,
            "native_voxels": int(log10rho.size),
            "maximum_log10rho": maximum,
            "strict_exceedances_above_global_train_maximum": exceedances,
        }
        print(f"[v56-grid] {domain} objects={objects}", flush=True)
    result: dict[str, Any] = {
        "schema": GRID_SCHEMA,
        "status": "complete_fixed_upper_survival_grid",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "v54_threshold_selection_sha256": threshold_sha,
        "lower_edge_log10rho": lower,
        "upper_edge_log10rho": upper,
        "cells": GRID_CELLS,
        "thresholds_log10rho": thresholds.tolist(),
        "physical_moment_weights": weights.tolist(),
        "physical_moment_weight_sum": float(weights.sum(dtype=np.float64)),
        "reference_probability": REFERENCE_PROBABILITY,
        "normalization": REFERENCE_PROBABILITY * (1.0 - REFERENCE_PROBABILITY),
        "coefficient": GRID_COEFFICIENT,
        "domains": domains,
        "final_threshold_truth_strict_exceedances": 0,
        "validation_accessed": False,
        "development_accessed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, output)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def load_grid(path: Path, digest: str, commit: str, threshold_sha: str) -> dict[str, Any]:
    value = _verified_json(path, digest, "grid")
    if (
        value.get("schema") != GRID_SCHEMA
        or value.get("status") != "complete_fixed_upper_survival_grid"
        or value.get("program_sha256") != PROGRAM_SHA256
        or value.get("code_commit") != commit
        or value.get("worktree_clean") is not True
        or value.get("v54_threshold_selection_sha256") != threshold_sha
        or value.get("cells") != GRID_CELLS
        or value.get("final_threshold_truth_strict_exceedances") != 0
        or value.get("development_accessed") is not False
        or value.get("independent_gate_locked") is not True
    ):
        raise ValueError("V56 grid binding differs")
    thresholds, weights = grid_values(
        float(value["lower_edge_log10rho"]), float(value["upper_edge_log10rho"])
    )
    if not np.array_equal(thresholds, np.asarray(value["thresholds_log10rho"])) or not np.array_equal(
        weights, np.asarray(value["physical_moment_weights"])
    ):
        raise ValueError("V56 grid values differ")
    return value


def upper_survival_grid_score(
    parameters: torch.Tensor,
    target: torch.Tensor,
    backbone: torch.Tensor,
    target_mean: float,
    target_std: float,
    thresholds_log10rho: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        target.shape != (len(parameters), 1, *parameters.shape[-3:])
        or backbone.shape != target.shape
        or thresholds_log10rho.shape != (GRID_CELLS,)
        or weights.shape != thresholds_log10rho.shape
        or target_std <= 0.0
    ):
        raise ValueError("V56 upper survival score input differs")
    components = []
    for index, threshold in enumerate(thresholds_log10rho):
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
            raise RuntimeError(f"V56 nonfinite survival Brier score at grid cell {index}")
        components.append(raw)
    stacked = torch.stack(components)
    score = torch.sum(weights.double() * stacked) / (
        REFERENCE_PROBABILITY * (1.0 - REFERENCE_PROBABILITY)
    )
    return score, stacked


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
    grid, grid_components = upper_survival_grid_score(
        parameters,
        target,
        backbone,
        target_mean,
        target_std,
        grid_thresholds,
        grid_weights,
    )
    total = nll + TAIL_COEFFICIENT * tail + GRID_COEFFICIENT * grid
    return total, nll, tail, tail_components, grid, grid_components


def _thresholds(path: Path, digest: str) -> dict[str, Any]:
    value = _verified_json(path, digest, "V54 thresholds")
    thresholds = np.asarray(value.get("common_log10rho_thresholds"), dtype=np.float64)
    if thresholds.shape != (4,) or not np.all(np.diff(thresholds) > 0.0):
        raise ValueError("V56 V54 thresholds differ")
    return value


def preflight(
    program_path: Path,
    repo: Path,
    cache_path: Path,
    cache_sha: str,
    threshold_path: Path,
    threshold_sha: str,
    grid_path: Path,
    grid_sha: str,
    output: Path,
) -> dict[str, Any]:
    program, v35, _ = load_program(program_path, repo.resolve())
    commit, clean = git_state(repo.resolve())
    if not clean or output.exists():
        raise RuntimeError("V56 preflight requires clean new output")
    frozen = program["frozen_inputs"]
    if (
        cache_path.resolve() != _resolve(repo, frozen["conditioning_cache"])
        or cache_sha != frozen["conditioning_cache_sha256"]
        or threshold_sha != frozen["v54_threshold_selection_sha256"]
    ):
        raise ValueError("V56 preflight frozen binding differs")
    selected = _thresholds(threshold_path, threshold_sha)
    grid = load_grid(grid_path, grid_sha, commit, threshold_sha)
    prepared = load_cache(cache_path, cache_sha, commit)
    device = _device()
    handles: list[tuple[h5py.File, h5py.File]] = []
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
            raise RuntimeError("V56 architecture differs")
        condition_tensor = torch.from_numpy(np.stack(conditions)).to(device)
        target_tensor = torch.from_numpy(np.stack(targets)).to(device)
        backbone_tensor = torch.from_numpy(np.stack(backbones)).to(device)
        v54_thresholds = torch.tensor(selected["common_log10rho_thresholds"], device=device)
        grid_thresholds = torch.tensor(grid["thresholds_log10rho"], device=device)
        grid_weights = torch.tensor(grid["physical_moment_weights"], device=device)
        parameters = model(condition_tensor)
        expected = torch.tensor(INITIAL_BIASES, device=device).reshape(1, 15, 1, 1, 1)
        initialization_error = float(torch.max(torch.abs(parameters - expected)).cpu())
        if initialization_error > 1.0e-7 or torch.count_nonzero(model.output.weight):
            raise RuntimeError("V56 initialization differs")
        total, nll, tail, tail_components, upper, upper_components = composite_loss(
            parameters,
            target_tensor,
            backbone_tensor,
            float(prepared["target_mean"][()]),
            float(prepared["target_std"][()]),
            v54_thresholds,
            grid_thresholds,
            grid_weights,
        )
        base = nll + TAIL_COEFFICIENT * tail
        identity_error = float(torch.abs(total - base - GRID_COEFFICIENT * upper).cpu())
        total.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if (
            not torch.isfinite(total)
            or identity_error > 1.0e-7
            or not gradients
            or not all(torch.isfinite(gradient).all() for gradient in gradients)
            or not any(torch.count_nonzero(gradient) for gradient in gradients)
        ):
            raise RuntimeError("V56 score, identity, or gradient differs")
        peak = int(torch.cuda.max_memory_allocated(device))
        if peak >= 24 * 1024**3:
            raise RuntimeError("V56 preflight memory differs")
    finally:
        for data, cache in handles:
            data.close()
            cache.close()
        prepared.close()
    v54_preflight = _verified_json(
        _resolve(repo, frozen["v54_preflight"]), frozen["v54_preflight_sha256"], "V54 preflight"
    )
    if (
        v54_preflight.get("status") != "pass"
        or v54_preflight.get("development_accessed") is not False
        or v54_preflight.get("independent_gate_locked") is not True
    ):
        raise ValueError("V56 inherited paired-sample preflight differs")
    result: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "pass",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "parameters": PARAMETERS,
        "v54_threshold_selection_sha256": threshold_sha,
        "grid_sha256": grid_sha,
        "grid_thresholds_log10rho": grid["thresholds_log10rho"],
        "grid_physical_moment_weights": grid["physical_moment_weights"],
        "tail_coefficient": TAIL_COEFFICIENT,
        "grid_coefficient": GRID_COEFFICIENT,
        "real_source_balanced_composite_loss": float(total.detach().cpu()),
        "real_source_balanced_bounded_NLL": float(nll.detach().cpu()),
        "real_source_balanced_V54_tail_score": float(tail.detach().cpu()),
        "real_source_balanced_V54_Brier_components": tail_components.detach().cpu().tolist(),
        "real_source_balanced_upper_grid_score": float(upper.detach().cpu()),
        "real_source_balanced_upper_grid_raw_Brier_components": upper_components.detach().cpu().tolist(),
        "composite_identity_absolute_error": identity_error,
        "initial_output_maximum_error": initialization_error,
        "peak_allocated_bytes": peak,
        "inherited_V54_paired_sample_residual_DC": v54_preflight[
            "maximum_real_sample_residual_DC"
        ],
        "cache_sha256": cache_sha,
        "support_selection_sha256": SUPPORT_SHA256,
        "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
        "development_accessed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, output)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


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
            for index in range(int(v35["development_domains"][domain]["validation_objects"])):
                condition, target, backbone = condition_cube(
                    data, cache, prepared, domain, "validation", index
                )
                parameter = model(torch.from_numpy(condition[None]).to(device))
                scores = composite_loss(
                    parameter,
                    torch.from_numpy(target[None]).to(device),
                    torch.from_numpy(backbone[None]).to(device),
                    float(prepared["target_mean"][()]),
                    float(prepared["target_std"][()]),
                    v54_thresholds,
                    grid_thresholds,
                    grid_weights,
                )
                values.append(
                    tuple(float(value.cpu()) for value in (scores[0], scores[1], scores[2], scores[4]))
                )
        finally:
            data.close()
            cache.close()
        result[domain] = {
            "composite": float(np.mean([value[0] for value in values])),
            "bounded_NLL": float(np.mean([value[1] for value in values])),
            "V54_tail_score": float(np.mean([value[2] for value in values])),
            "upper_grid_score": float(np.mean([value[3] for value in values])),
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
    _, v35, _ = load_program(program_path, repo.resolve())
    commit, clean = git_state(repo.resolve())
    if not clean or checkpoint_path.exists() or report_path.exists():
        raise RuntimeError("V56 training requires clean new outputs")
    checked = _verified_json(preflight_path, preflight_sha, "preflight")
    if (
        checked.get("schema") != PREFLIGHT_SCHEMA
        or checked.get("status") != "pass"
        or checked.get("code_commit") != commit
        or checked.get("grid_sha256") != grid_sha
        or checked.get("v54_threshold_selection_sha256") != threshold_sha
    ):
        raise ValueError("V56 preflight binding differs")
    selected = _thresholds(threshold_path, threshold_sha)
    grid = load_grid(grid_path, grid_sha, commit, threshold_sha)
    prepared = load_cache(cache_path, cache_sha, commit)
    device = _device()
    model = _same_seed_model(device)
    ema_model = _same_seed_model(device)
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
    try:
        model.train()
        for step in range(1, STEPS + 1):
            conditions, targets, backbones = [], [], []
            for domain in DOMAIN_ORDER:
                row = v35["development_domains"][domain]
                index = int(generator.integers(int(row["train_objects"])))
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
            if step == 1 or step % 50 == 0:
                row = {
                    "step": step,
                    "composite_loss": float(total.detach().cpu()),
                    "bounded_NLL": float(nll.detach().cpu()),
                    "V54_tail_score": float(tail.detach().cpu()),
                    "V54_Brier_components": tail_components.detach().cpu().tolist(),
                    "upper_grid_score": float(upper.detach().cpu()),
                    "upper_grid_raw_Brier_components": upper_components.detach().cpu().tolist(),
                    "learning_rate": learning_rate,
                    "gradient_norm_before_clip": (
                        float(gradient_norm.detach().cpu()) if gradient_finite else "nonfinite"
                    ),
                    "AMP_scale_before_update": scale_before,
                    "AMP_scale_after_update": scale_after,
                    "AMP_update_skipped": overflow,
                }
                log_rows.append(row)
                print("[v56-train] " + json.dumps(row), flush=True)
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
                    f"[v56-validation] step={step} "
                    + json.dumps(validation_rows[str(step)]),
                    flush=True,
                )
    finally:
        for data, cache in handles.values():
            data.close()
            cache.close()
        prepared.close()
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "step": STEPS,
        "parameters": PARAMETERS,
        "likelihood_family": LIKELIHOOD_FAMILY,
        "open_standardized_support": [LOWER_SUPPORT, UPPER_SUPPORT],
        "support_selection_sha256": SUPPORT_SHA256,
        "v54_threshold_selection_sha256": threshold_sha,
        "common_log10rho_thresholds": selected["common_log10rho_thresholds"],
        "grid_sha256": grid_sha,
        "grid_thresholds_log10rho": grid["thresholds_log10rho"],
        "grid_physical_moment_weights": grid["physical_moment_weights"],
        "tail_coefficient": TAIL_COEFFICIENT,
        "grid_coefficient": GRID_COEFFICIENT,
        "ema_decay": 0.999,
        "ema_state_dict": {
            key: value.detach().cpu() for key, value in ema_model.state_dict().items()
        },
        "conditioning_cache_sha256": cache_sha,
        "preflight_sha256": preflight_sha,
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
        "status": "complete_fixed_12000_step_upper_survival_grid_fit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "conditioning_cache_sha256": cache_sha,
        "preflight_sha256": preflight_sha,
        "v54_threshold_selection_sha256": threshold_sha,
        "grid_sha256": grid_sha,
        "common_log10rho_thresholds": selected["common_log10rho_thresholds"],
        "grid_thresholds_log10rho": grid["thresholds_log10rho"],
        "grid_physical_moment_weights": grid["physical_moment_weights"],
        "tail_coefficient": TAIL_COEFFICIENT,
        "grid_coefficient": GRID_COEFFICIENT,
        "support_selection_sha256": SUPPORT_SHA256,
        "open_standardized_support": [LOWER_SUPPORT, UPPER_SUPPORT],
        "steps": STEPS,
        "parameters": PARAMETERS,
        "training_log": log_rows,
        "AMP_overflow_events": overflow_events,
        "AMP_overflow_count": len(overflow_events),
        "fixed_validation_diagnostics": validation_rows,
        "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
        "likelihood_family": LIKELIHOOD_FAMILY,
        "sample_clipping": False,
        "component_scale_cap": False,
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
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize-grid")
    materialize.add_argument("--program", type=Path, required=True)
    materialize.add_argument("--repo", type=Path, required=True)
    materialize.add_argument("--thresholds", type=Path, required=True)
    materialize.add_argument("--thresholds-sha256", required=True)
    materialize.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("preflight")
    fit = commands.add_parser("train")
    for command in (check, fit):
        command.add_argument("--program", type=Path, required=True)
        command.add_argument("--repo", type=Path, required=True)
        command.add_argument("--cache", type=Path, required=True)
        command.add_argument("--cache-sha256", required=True)
        command.add_argument("--thresholds", type=Path, required=True)
        command.add_argument("--thresholds-sha256", required=True)
        command.add_argument("--grid", type=Path, required=True)
        command.add_argument("--grid-sha256", required=True)
        command.add_argument("--preflight", type=Path, required=command is fit)
        command.add_argument("--preflight-sha256", required=command is fit)
        command.add_argument("--checkpoint", type=Path, required=command is fit)
        command.add_argument("--report", type=Path, required=command is fit)
        command.add_argument("--out", type=Path, required=command is check)
    args = parser.parse_args()
    if args.command == "materialize-grid":
        materialize_grid(
            args.program, args.repo, args.thresholds, args.thresholds_sha256, args.out
        )
    elif args.command == "preflight":
        preflight(
            args.program,
            args.repo,
            args.cache,
            args.cache_sha256,
            args.thresholds,
            args.thresholds_sha256,
            args.grid,
            args.grid_sha256,
            args.out,
        )
    else:
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
