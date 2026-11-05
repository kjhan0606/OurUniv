#!/usr/bin/env python
"""Select train thresholds, preflight, and train the frozen V54 tail score."""
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
from hong2021_v31_copula import conditional_forward, load_model
from hong2021_v35_spectrum_phase import _backbone, _open_split
from hong2021_v37_query_alignment import _selection_arrays
from hong2021_v48_train import condition_cube, load_cache
from hong2021_v50_network import (
    INITIAL_BIASES,
    LOWER_SUPPORT,
    UPPER_SUPPORT,
    LocalMixtureUNet,
    bounded_mixture_cdf,
    bounded_mixture_inverse,
    bounded_mixture_log_probability,
    parameter_count,
)
from hong2021_v50_train import (
    PARAMETERS,
    SUPPORT_DECISION_DIGEST,
    SUPPORT_SHA256,
    _device,
    load_program as load_v50_program,
)


PROGRAM_SCHEMA = "hong2021-v54-physical-tail-brier-bounded-mixture-program-v1"
PROGRAM_SHA256 = "84e06be3980deeb63456fb53a56d05d22eca68fea663d023b6be5e75460fbd90"
THRESHOLD_SCHEMA = "hong2021-v54-train-only-physical-tail-thresholds-v1"
PREFLIGHT_SCHEMA = "hong2021-v54-physical-tail-brier-hard-preflight-v1"
CHECKPOINT_SCHEMA = "hong2021-v54-physical-tail-brier-checkpoint-v1"
REPORT_SCHEMA = "hong2021-v54-physical-tail-brier-training-report-v1"
LIKELIHOOD_FAMILY = "bounded_logit_Gaussian_mixture"
QUANTILES = (0.99, 0.999, 0.9999, 0.99999)
NOMINAL_EXCEEDANCE = tuple(1.0 - value for value in QUANTILES)
TAIL_COEFFICIENT = 0.1
PRIMARY_QUADRATURE_ORDER = 64
CONTROL_QUADRATURE_ORDER = 32
STEPS = 12_000
VALIDATION_STEPS = (4_000, 8_000, 12_000)
SEED = 144_044


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V54 {label} hash differs")
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status")
        != "frozen_before_threshold_selection_model_implementation_training_or_evaluation"
    ):
        raise ValueError("V54 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _resolve(repo, parent["v53_record"]),
        parent["v53_record_sha256"],
        "V53 record",
    )
    if (
        record.get("status") != parent["required_status"]
        or record.get("audit", {}).get("classification")
        != parent["required_classification"]
        or record.get("audit", {}).get("next") != parent["required_next"]
        or not record.get("selected_next_model", {})
        .get("single_changed_factor", "")
        .startswith(parent["required_single_changed_factor_prefix"])
        or record.get("firewall", {}).get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
    ):
        raise ValueError("V54 V53 parent evidence differs")
    frozen = program["frozen_inputs"]
    for key in (
        "v50_program",
        "v50_record",
        "v50_checkpoint",
        "v50_training_report",
        "v50_development_decision",
        "v52_record",
        "v52_development_decision",
        "v53_audit",
        "conditioning_cache",
        "support_selection",
    ):
        if sha256_file(_resolve(repo, frozen[key])) != frozen[f"{key}_sha256"]:
            raise ValueError(f"V54 frozen input differs: {key}")
    digest_keys = {
        "v50_development_decision": "v50_development_decision_digest_sha256",
        "v52_development_decision": "v52_development_decision_digest_sha256",
        "v53_audit": "v53_audit_decision_digest_sha256",
    }
    for key, digest_key in digest_keys.items():
        value = _verified_json(
            _resolve(repo, frozen[key]), frozen[f"{key}_sha256"], key
        )
        if canonical_digest(value) != frozen[digest_key]:
            raise ValueError(f"V54 frozen digest differs: {key}")
        if (
            value.get("independent_gate_locked") is not True
            or value.get("historical_EAGLE_accessed") is not False
        ):
            raise ValueError(f"V54 frozen firewall differs: {key}")
    support = _verified_json(
        Path(frozen["support_selection"]),
        frozen["support_selection_sha256"],
        "support selection",
    )
    if (
        canonical_digest(support) != frozen["support_selection_decision_digest_sha256"]
        or frozen["support_selection_decision_digest_sha256"]
        != SUPPORT_DECISION_DIGEST
        or support.get("support", {}).get("lower_support") != LOWER_SUPPORT
        or support.get("support", {}).get("upper_support") != UPPER_SUPPORT
    ):
        raise ValueError("V54 support binding differs")
    v50, v35, v41 = load_v50_program(_resolve(repo, frozen["v50_program"]), repo)
    effective = dict(program)
    effective["inherited_inputs"] = v50["inherited_inputs"]
    return effective, v35, v41


def select_thresholds(program_path: Path, repo: Path, output: Path) -> dict[str, Any]:
    _, v35, _ = load_program(program_path, repo.resolve())
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V54 threshold selection requires a clean worktree")
    if output.exists():
        raise FileExistsError("V54 refuses existing threshold selection")
    domains: dict[str, Any] = {}
    values: list[np.ndarray] = []
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        data, cache = _open_split(row, "train")
        try:
            objects = int(row["train_objects"])
            target = np.asarray(data["target"][:objects, 0], dtype=np.float32)
        finally:
            data.close()
            cache.close()
        expected = objects * 64**3
        if target.size != expected or not np.isfinite(target).all():
            raise RuntimeError("V54 train threshold population differs")
        log_density = 4.5 * target.astype(np.float64)
        quantiles = np.quantile(log_density, QUANTILES, method="linear")
        if not np.all(np.diff(quantiles) > 0.0):
            raise RuntimeError("V54 domain threshold order differs")
        values.append(quantiles)
        domains[domain] = {
            "train_objects": objects,
            "native_voxels": expected,
            "log10rho_quantiles": quantiles.tolist(),
            "minimum_log10rho": float(log_density.min()),
            "maximum_log10rho": float(log_density.max()),
        }
        print(f"[v54-thresholds] {domain} objects={objects}", flush=True)
    common = np.median(np.stack(values), axis=0)
    if not np.all(np.diff(common) > 0.0):
        raise RuntimeError("V54 common threshold order differs")
    result: dict[str, Any] = {
        "schema": THRESHOLD_SCHEMA,
        "status": "complete_train_only_fixed_threshold_selection",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "quantile_probabilities": list(QUANTILES),
        "nominal_exceedance_probabilities": list(NOMINAL_EXCEEDANCE),
        "normalized_Brier_weights": [
            1.0 / (value * (1.0 - value)) for value in QUANTILES
        ],
        "domains": domains,
        "common_log10rho_thresholds": common.tolist(),
        "common_physical_y_thresholds": (common / 4.5).tolist(),
        "selection_rule": "median_of_three_exact_domainwise_linear_train_quantiles",
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


def load_thresholds(path: Path, digest: str, commit: str) -> dict[str, Any]:
    value = _verified_json(path, digest, "threshold selection")
    if (
        value.get("schema") != THRESHOLD_SCHEMA
        or value.get("status") != "complete_train_only_fixed_threshold_selection"
        or value.get("program_sha256") != PROGRAM_SHA256
        or value.get("code_commit") != commit
        or value.get("worktree_clean") is not True
        or tuple(value.get("quantile_probabilities", ())) != QUANTILES
        or value.get("validation_accessed") is not False
        or value.get("development_accessed") is not False
        or value.get("independent_gate_locked") is not True
    ):
        raise ValueError("V54 threshold selection binding differs")
    thresholds = np.asarray(value["common_log10rho_thresholds"], dtype=np.float64)
    weights = np.asarray(value["normalized_Brier_weights"], dtype=np.float64)
    expected = np.asarray([1.0 / (q * (1.0 - q)) for q in QUANTILES])
    if (
        thresholds.shape != (4,)
        or not np.all(np.isfinite(thresholds))
        or not np.all(np.diff(thresholds) > 0.0)
        or not np.array_equal(weights, expected)
    ):
        raise ValueError("V54 threshold values or weights differ")
    return value


def physical_tail_brier_score(
    parameters: torch.Tensor,
    target: torch.Tensor,
    backbone: torch.Tensor,
    target_mean: float,
    target_std: float,
    thresholds_log10rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        target.shape != (len(parameters), 1, *parameters.shape[-3:])
        or backbone.shape != target.shape
        or thresholds_log10rho.shape != (4,)
        or target_std <= 0.0
    ):
        raise ValueError("V54 physical tail score input differs")
    scores = []
    for index, (quantile, threshold) in enumerate(
        zip(QUANTILES, thresholds_log10rho, strict=True)
    ):
        physical_y = threshold.double() / 4.5
        standardized = (
            physical_y - backbone.double() - float(target_mean)
        ) / float(target_std)
        below = standardized <= LOWER_SUPPORT
        above = standardized >= UPPER_SUPPORT
        interior = standardized.clamp(LOWER_SUPPORT + 1.0e-6, UPPER_SUPPORT - 1.0e-6)
        exceedance = 1.0 - bounded_mixture_cdf(parameters, interior)
        exceedance = torch.where(below, torch.ones_like(exceedance), exceedance)
        exceedance = torch.where(above, torch.zeros_like(exceedance), exceedance)
        observed = (target.double() > standardized).float()
        score = torch.square(exceedance - observed).mean() / (
            quantile * (1.0 - quantile)
        )
        if not torch.isfinite(score):
            raise RuntimeError(f"V54 nonfinite Brier score at threshold {index}")
        scores.append(score)
    stacked = torch.stack(scores)
    return stacked.mean(), stacked


def composite_loss(
    parameters: torch.Tensor,
    target: torch.Tensor,
    backbone: torch.Tensor,
    target_mean: float,
    target_std: float,
    thresholds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    nll = -bounded_mixture_log_probability(parameters, target).mean()
    tail, components = physical_tail_brier_score(
        parameters, target, backbone, target_mean, target_std, thresholds
    )
    total = nll + TAIL_COEFFICIENT * tail
    return total, nll, tail, components


def _same_seed_model(device: torch.device) -> LocalMixtureUNet:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return LocalMixtureUNet().to(device)


def preflight(
    program_path: Path,
    repo: Path,
    cache_path: Path,
    cache_sha: str,
    threshold_path: Path,
    threshold_sha: str,
    output: Path,
) -> dict[str, Any]:
    program, v35, _ = load_program(program_path, repo.resolve())
    commit, clean = git_state(repo.resolve())
    if not clean or output.exists():
        raise RuntimeError("V54 preflight requires clean new output")
    frozen = program["frozen_inputs"]
    if cache_path.resolve() != Path(frozen["conditioning_cache"]).resolve() or cache_sha != frozen["conditioning_cache_sha256"]:
        raise ValueError("V54 cache binding differs")
    selected = load_thresholds(threshold_path, threshold_sha, commit)
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
            raise RuntimeError("V54 architecture differs")
        condition_tensor = torch.from_numpy(np.stack(conditions)).to(device)
        target_tensor = torch.from_numpy(np.stack(targets)).to(device)
        backbone_tensor = torch.from_numpy(np.stack(backbones)).to(device)
        thresholds = torch.tensor(
            selected["common_log10rho_thresholds"], device=device
        )
        parameters = model(condition_tensor)
        expected = torch.tensor(INITIAL_BIASES, device=device).reshape(1, 15, 1, 1, 1)
        initialization_error = float(torch.max(torch.abs(parameters - expected)).cpu())
        if initialization_error > 1.0e-7 or torch.count_nonzero(model.output.weight):
            raise RuntimeError("V54 initialization differs")
        total, nll, tail, components = composite_loss(
            parameters,
            target_tensor,
            backbone_tensor,
            float(prepared["target_mean"][()]),
            float(prepared["target_std"][()]),
            thresholds,
        )
        total.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if (
            not torch.isfinite(total)
            or not gradients
            or not all(torch.isfinite(g).all() for g in gradients)
            or not any(torch.count_nonzero(g) for g in gradients)
        ):
            raise RuntimeError("V54 score or gradient differs")
        peak = int(torch.cuda.max_memory_allocated(device))
        if peak >= 24 * 1024**3:
            raise RuntimeError("V54 preflight memory differs")
        selections = _selection_arrays(v35)
        domain = DOMAIN_ORDER[0]
        query_index = int(selections[domain]["source_index"][0])
        donor_source = DOMAIN_ORDER[int(selections[domain]["donor_source"][0, 0])]
        donor_index = int(selections[domain]["donor_index"][0, 0])
        isometry = int(selections[domain]["donor_isometry"][0, 0])
        query_data, query_cache = _open_split(v35["development_domains"][domain], "validation")
        donor_data, donor_cache = _open_split(v35["development_domains"][donor_source], "train")
        try:
            query_condition, _, query_backbone = condition_cube(
                query_data, query_cache, prepared, domain, "validation", query_index
            )
            donor_backbone = _backbone(donor_cache, donor_index)[None]
            donor_truth = np.asarray(donor_data["target"][donor_index], dtype=np.float32)
            copula = load_model(
                Path(program["inherited_inputs"]["conditional_copula_artifact"]),
                program["inherited_inputs"]["conditional_copula_artifact_sha256"],
            )
            rank = conditional_forward(donor_truth - donor_backbone, donor_backbone, copula)
            axes, reflections = CUBE_ISOMETRIES[isometry]
            rank = apply_cube_isometry(rank, axes, reflections)
            with torch.no_grad():
                query_parameters = model(torch.from_numpy(query_condition[None]).to(device))
                standardized = bounded_mixture_inverse(
                    query_parameters, torch.from_numpy(rank[None]).to(device)
                ).cpu().numpy()[0]
            if not np.all((standardized > LOWER_SUPPORT) & (standardized < UPPER_SUPPORT)):
                raise RuntimeError("V54 preflight paired sample left support")
            residual = standardized * float(prepared["target_std"][()]) + float(prepared["target_mean"][()])
            residual -= residual.mean(dtype=np.float64)
            sample = query_backbone + residual
            real_dc = float(abs(residual.mean(dtype=np.float64)))
            if not np.isfinite(sample).all() or real_dc > 1.0e-7:
                raise RuntimeError("V54 preflight paired sample differs")
        finally:
            query_data.close()
            query_cache.close()
            donor_data.close()
            donor_cache.close()
    finally:
        for data, cache in handles:
            data.close()
            cache.close()
        prepared.close()
    result: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "pass",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "parameters": PARAMETERS,
        "threshold_selection": str(threshold_path.resolve()),
        "threshold_selection_sha256": threshold_sha,
        "common_log10rho_thresholds": selected["common_log10rho_thresholds"],
        "tail_coefficient": TAIL_COEFFICIENT,
        "real_source_balanced_composite_loss": float(total.detach().cpu()),
        "real_source_balanced_bounded_NLL": float(nll.detach().cpu()),
        "real_source_balanced_tail_score": float(tail.detach().cpu()),
        "real_source_balanced_Brier_components": components.detach().cpu().tolist(),
        "initial_output_maximum_error": initialization_error,
        "peak_allocated_bytes": peak,
        "maximum_real_sample_residual_DC": real_dc,
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


def _learning_rate(step: int) -> float:
    fraction = step / STEPS
    return 2.0e-5 + 0.5 * (2.0e-4 - 2.0e-5) * (
        1.0 + math.cos(math.pi * fraction)
    )


@torch.no_grad()
def _validation_scores(
    model: LocalMixtureUNet,
    v35: dict[str, Any],
    prepared: h5py.File,
    device: torch.device,
    thresholds: torch.Tensor,
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
                total, nll, tail, components = composite_loss(
                    parameter,
                    torch.from_numpy(target[None]).to(device),
                    torch.from_numpy(backbone[None]).to(device),
                    float(prepared["target_mean"][()]),
                    float(prepared["target_std"][()]),
                    thresholds,
                )
                values.append((float(total.cpu()), float(nll.cpu()), float(tail.cpu()), components.cpu().numpy()))
        finally:
            data.close()
            cache.close()
        result[domain] = {
            "composite": float(np.mean([v[0] for v in values])),
            "bounded_NLL": float(np.mean([v[1] for v in values])),
            "tail_score": float(np.mean([v[2] for v in values])),
            "Brier_components": np.mean([v[3] for v in values], axis=0).tolist(),
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
    preflight_path: Path,
    preflight_sha: str,
    checkpoint_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    program, v35, _ = load_program(program_path, repo.resolve())
    commit, clean = git_state(repo.resolve())
    if not clean or checkpoint_path.exists() or report_path.exists():
        raise RuntimeError("V54 training requires clean new outputs")
    checked = _verified_json(preflight_path, preflight_sha, "preflight")
    if (
        checked.get("schema") != PREFLIGHT_SCHEMA
        or checked.get("status") != "pass"
        or checked.get("code_commit") != commit
        or checked.get("threshold_selection_sha256") != threshold_sha
    ):
        raise ValueError("V54 preflight binding differs")
    selected = load_thresholds(threshold_path, threshold_sha, commit)
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
    thresholds = torch.tensor(selected["common_log10rho_thresholds"], device=device)
    handles = {domain: _open_split(v35["development_domains"][domain], "train") for domain in DOMAIN_ORDER}
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
                condition, target, backbone = condition_cube(data, cache, prepared, domain, "train", index)
                axes, reflections = CUBE_ISOMETRIES[int(generator.integers(len(CUBE_ISOMETRIES)))]
                conditions.append(apply_cube_isometry(condition, axes, reflections))
                targets.append(apply_cube_isometry(target, axes, reflections))
                backbones.append(apply_cube_isometry(backbone, axes, reflections))
            condition_tensor = torch.from_numpy(np.stack(conditions)).to(device)
            target_tensor = torch.from_numpy(np.stack(targets)).to(device)
            backbone_tensor = torch.from_numpy(np.stack(backbones)).to(device)
            lr = _learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                parameters = model(condition_tensor)
            total, nll, tail, components = composite_loss(
                parameters,
                target_tensor,
                backbone_tensor,
                float(prepared["target_mean"][()]),
                float(prepared["target_std"][()]),
                thresholds,
            )
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
                overflow_events.append({
                    "step": step,
                    "loss": float(total.detach().cpu()),
                    "gradient_norm_is_finite": gradient_finite,
                    "scale_before": scale_before,
                    "scale_after": scale_after,
                })
            with torch.no_grad():
                for ema, current in zip(ema_model.parameters(), model.parameters(), strict=True):
                    ema.lerp_(current.detach(), 1.0 - 0.999)
                for ema, current in zip(ema_model.buffers(), model.buffers(), strict=True):
                    ema.copy_(current)
            if step == 1 or step % 50 == 0:
                row = {
                    "step": step,
                    "composite_loss": float(total.detach().cpu()),
                    "bounded_NLL": float(nll.detach().cpu()),
                    "tail_score": float(tail.detach().cpu()),
                    "Brier_components": components.detach().cpu().tolist(),
                    "learning_rate": lr,
                    "gradient_norm_before_clip": float(gradient_norm.detach().cpu()) if gradient_finite else "nonfinite",
                    "AMP_scale_before_update": scale_before,
                    "AMP_scale_after_update": scale_after,
                    "AMP_update_skipped": overflow,
                }
                log_rows.append(row)
                print("[v54-train] " + json.dumps(row), flush=True)
            if step in VALIDATION_STEPS:
                validation_rows[str(step)] = _validation_scores(ema_model, v35, prepared, device, thresholds)
                print(f"[v54-validation] step={step} " + json.dumps(validation_rows[str(step)]), flush=True)
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
        "threshold_selection": str(threshold_path.resolve()),
        "threshold_selection_sha256": threshold_sha,
        "common_log10rho_thresholds": selected["common_log10rho_thresholds"],
        "tail_coefficient": TAIL_COEFFICIENT,
        "ema_decay": 0.999,
        "ema_state_dict": {key: value.detach().cpu() for key, value in ema_model.state_dict().items()},
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
        "status": "complete_fixed_12000_step_physical_tail_brier_fit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "conditioning_cache_sha256": cache_sha,
        "preflight_sha256": preflight_sha,
        "threshold_selection_sha256": threshold_sha,
        "common_log10rho_thresholds": selected["common_log10rho_thresholds"],
        "tail_coefficient": TAIL_COEFFICIENT,
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
    select = commands.add_parser("select-thresholds")
    select.add_argument("--program", type=Path, required=True)
    select.add_argument("--repo", type=Path, required=True)
    select.add_argument("--out", type=Path, required=True)
    check = commands.add_parser("preflight")
    fit = commands.add_parser("train")
    for command in (check, fit):
        command.add_argument("--program", type=Path, required=True)
        command.add_argument("--repo", type=Path, required=True)
        command.add_argument("--cache", type=Path, required=True)
        command.add_argument("--cache-sha256", required=True)
        command.add_argument("--thresholds", type=Path, required=True)
        command.add_argument("--thresholds-sha256", required=True)
        command.add_argument("--preflight", type=Path, required=command is fit)
        command.add_argument("--preflight-sha256", required=command is fit)
        command.add_argument("--checkpoint", type=Path, required=command is fit)
        command.add_argument("--report", type=Path, required=command is fit)
        command.add_argument("--out", type=Path, required=command is check)
    args = parser.parse_args()
    if args.command == "select-thresholds":
        select_thresholds(args.program, args.repo, args.out)
    elif args.command == "preflight":
        preflight(args.program, args.repo, args.cache, args.cache_sha256, args.thresholds, args.thresholds_sha256, args.out)
    else:
        train(args.program, args.repo, args.cache, args.cache_sha256, args.thresholds, args.thresholds_sha256, args.preflight, args.preflight_sha256, args.checkpoint, args.report)


if __name__ == "__main__":
    main()
