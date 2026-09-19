#!/usr/bin/env python
"""Preflight and train the frozen V48 Gaussian-mixture likelihood."""
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
from hong2021_v45_train import load_program as load_v45_program
from hong2021_v48_network import (
    BISECTION_STEPS,
    INITIAL_BIASES,
    INITIAL_LOCATIONS,
    INITIAL_RAW_SCALE,
    INITIAL_SCALE,
    INPUT_CHANNELS,
    LocalMixtureUNet,
    gaussian_mixture_cdf,
    gaussian_mixture_inverse,
    gaussian_mixture_log_probability,
    mixture_parameters,
    parameter_count,
)


PROGRAM_SCHEMA = "hong2021-v48-identifiable-query-local-Gaussian-mixture-copula-development-program-v1"
PROGRAM_SHA256 = "aa667f40d9db4723547a11916f56e0c5b2f99be0546f526356dfb5c52eecc4f9"
V45_CACHE_SCHEMA = "hong2021-v45-train-only-conditioning-cache-v1"
V45_CACHE_SHA256 = "f62a074927a1ee67eb8b2a43fd36f0db024bb56545c049af93578abca9412153"
V45_CACHE_COMMIT = "e780bfb55c2c62c3b11397b525316ec353781403"
V45_PROGRAM_SHA256 = "30a5461d2c9bbd2464b08787ca6381467645a8d08f77efac2a0fb5a1932567ac"
PREFLIGHT_SCHEMA = "hong2021-v48-Gaussian-mixture-copula-hard-preflight-v1"
CHECKPOINT_SCHEMA = "hong2021-v48-Gaussian-mixture-copula-checkpoint-v1"
REPORT_SCHEMA = "hong2021-v48-Gaussian-mixture-copula-training-report-v1"
STEPS = 12_000
VALIDATION_STEPS = (4_000, 8_000, 12_000)
PARAMETERS = 8_490_415
EMA_DECAY = 0.999

def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"{label} hash differs")
    return json.loads(path.read_text())


def load_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "V48 program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status")
        != "frozen_before_implementation_training_sampling_or_development_evaluation"
    ):
        raise ValueError("V48 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        (repo / parent["v47_record"]).resolve(),
        parent["v47_record_sha256"],
        "V48 V47 result record",
    )
    if (
        record.get("audit", {}).get("classification")
        != parent["required_classification"]
        or record.get("audit", {}).get("next") != parent["required_next"]
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
        or record.get("firewall", {}).get("independent_gate_locked") is not True
    ):
        raise ValueError("V48 V47 conclusion or firewall differs")
    v45_record = _verified_json(
        (repo / parent["v45_record"]).resolve(),
        parent["v45_record_sha256"],
        "V48 V45 result record",
    )
    if (
        v45_record.get("development_decision", {}).get("classification")
        != "identifiable_mixture_body_is_supported_but_extreme_likelihood_is_insufficient"
        or v45_record.get("firewall", {}).get("independent_gate_locked") is not True
    ):
        raise ValueError("V48 V45 result or firewall differs")
    inherited = program["inherited_experiment"]
    v45_path = (repo / inherited["v45_program"]).resolve()
    if (
        sha256_file(v45_path) != inherited["v45_program_sha256"]
        or inherited["v45_program_sha256"] != V45_PROGRAM_SHA256
        or inherited["v45_conditioning_cache_sha256"] != V45_CACHE_SHA256
    ):
        raise ValueError("V48 inherited V45 hashes differ")
    v45, v35, v41 = load_v45_program(v45_path, repo)
    effective = dict(program)
    effective["inherited_inputs"] = v45["inherited_inputs"]
    return effective, v35, v41


def _radius(voxel_mpc_h: float) -> np.ndarray:
    coordinate = (np.arange(64, dtype=np.float64) + 0.5) * voxel_mpc_h - 10.0
    return (
        np.sqrt(
            np.square(coordinate[:, None, None])
            + np.square(coordinate[None, :, None])
            + np.square(coordinate[None, None, :])
        )
        / 10.0
    ).astype(np.float32)


def load_cache(path: Path, digest: str, commit: str) -> h5py.File:
    del commit
    if digest != V45_CACHE_SHA256 or sha256_file(path) != digest:
        raise ValueError("V48 conditioning cache hash differs")
    handle = h5py.File(path, "r")
    if (
        str(handle.attrs.get("schema")) != V45_CACHE_SCHEMA
        or str(handle.attrs.get("program_sha256")) != V45_PROGRAM_SHA256
        or str(handle.attrs.get("code_commit")) != V45_CACHE_COMMIT
        or not bool(handle.attrs.get("complete", False))
        or bool(handle.attrs.get("validation_truth_opened", True))
    ):
        handle.close()
        raise ValueError("V48 conditioning cache metadata differs")
    return handle


def condition_cube(
    data: h5py.File,
    cache: h5py.File,
    prepared: h5py.File,
    domain: str,
    split: str,
    index: int,
    *,
    risk_ablation: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = np.asarray(data["input"][index, 0], dtype=np.float32)
    velocity = np.asarray(data["input"][index, 1], dtype=np.float32)
    dispersion = np.asarray(data["input"][index, 2], dtype=np.float32)
    backbone = _backbone(cache, index).astype(np.float32)
    radius = _radius(float(data.attrs["voxel_mpc_h"]))
    risk = np.asarray(prepared[f"{domain}/{split}/block_risk"][index], dtype=np.float32)
    risk = risk.repeat(4, axis=0).repeat(4, axis=1).repeat(4, axis=2)
    amplitude = float(prepared[f"{domain}/{split}/object_amplitude"][index])
    raw = np.stack(
        (
            np.log1p(count),
            velocity,
            dispersion,
            backbone,
            radius,
            risk,
            np.full_like(backbone, amplitude),
        )
    ).astype(np.float32)
    mean = np.asarray(prepared["condition_mean"], dtype=np.float32)[:, None, None, None]
    std = np.asarray(prepared["condition_std"], dtype=np.float32)[:, None, None, None]
    condition = (raw - mean) / std
    if risk_ablation:
        condition[5] = 0.0
    truth = np.asarray(data["target"][index, 0], dtype=np.float32)
    target = (
        truth - backbone - np.float32(prepared["target_mean"][()])
    ) / np.float32(prepared["target_std"][()])
    if not np.isfinite(condition).all() or not np.isfinite(target).all():
        raise RuntimeError("V48 condition or target is nonfinite")
    return condition, target[None], backbone[None]


def _device() -> torch.device:
    if socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V48 requires Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V48 requires the Lageunha Ada GPU")
    return torch.device("cuda")


def _component_gradient_spreads(gradient: torch.Tensor) -> dict[str, float]:
    return {
        name: float((gradient[start:stop] - gradient[start]).abs().max().detach().cpu())
        for name, start, stop in (
            ("mixture_logits", 0, 5),
            ("locations", 5, 10),
            ("raw_scales", 10, 15),
        )
    }


def _identifiability_probe(device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(144044)
    target = torch.randn(1, 1, 4, 4, 4, generator=generator, device=device)

    symmetric_bias = torch.zeros(15, device=device, requires_grad=True)
    symmetric_parameters = symmetric_bias.reshape(1, 15, 1, 1, 1).expand(
        1, 15, 4, 4, 4
    )
    symmetric_loss = -gaussian_mixture_log_probability(
        symmetric_parameters, target
    ).mean()
    symmetric_loss.backward()
    symmetric_spreads = _component_gradient_spreads(symmetric_bias.grad)
    if max(symmetric_spreads.values()) > 1.0e-7:
        raise RuntimeError("V48 symmetric Gaussian control differs")

    bias = torch.nn.Parameter(torch.tensor(INITIAL_BIASES, device=device))
    initial = bias.reshape(1, 15, 1, 1, 1)
    logits, locations, scales = mixture_parameters(initial)
    weights = torch.softmax(logits, dim=1)
    mean = torch.sum(weights * locations, dim=1)
    variance = torch.sum(
        weights
        * (
            torch.square(locations - mean[:, None])
            + torch.square(scales)
        ),
        dim=1,
    )
    initial_mean = float(mean.detach().cpu())
    initial_variance = float(variance.detach().cpu())
    initial_location_spread = float(
        (locations.max() - locations.min()).detach().cpu()
    )
    if (
        abs(initial_mean) > 1.0e-6
        or abs(initial_variance - 1.0) > 1.0e-6
        or initial_location_spread <= 0.0
        or torch.unique(locations).numel() != 5
    ):
        raise RuntimeError("V48 initial mixture moments or locations differ")

    optimizer = torch.optim.AdamW([bias], lr=2.0e-4, weight_decay=1.0e-4)
    v48_spreads: dict[str, float] | None = None
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        parameters = bias.reshape(1, 15, 1, 1, 1).expand(1, 15, 4, 4, 4)
        loss = -gaussian_mixture_log_probability(parameters, target).mean()
        loss.backward()
        if v48_spreads is None:
            v48_spreads = _component_gradient_spreads(bias.grad)
        optimizer.step()
    assert v48_spreads is not None
    final = bias.detach().reshape(1, 15, 1, 1, 1)
    _, final_locations, final_scales = mixture_parameters(final)
    final_location_spread = float(
        (final_locations.max() - final_locations.min()).cpu()
    )
    final_scale_spread = float((final_scales.max() - final_scales.min()).cpu())
    if (
        min(v48_spreads.values()) <= 1.0e-8
        or final_location_spread <= 1.0e-3
        or final_scale_spread <= 1.0e-6
    ):
        raise RuntimeError("V48 components are not dynamically identifiable")
    return {
        "symmetric_Gaussian_same_role_component_gradient_spread": symmetric_spreads,
        "V48_same_role_component_gradient_spread": v48_spreads,
        "initial_mixture_mean": initial_mean,
        "initial_mixture_variance": initial_variance,
        "initial_component_location_spread": initial_location_spread,
        "component_location_spread_after_two_steps": final_location_spread,
        "component_scale_spread_after_two_steps": final_scale_spread,
    }


@torch.no_grad()
def _finite_physical_log_moments(
    parameters: torch.Tensor, target_std: float
) -> dict[str, float]:
    """Prove finite Gaussian exponential moments without exponentiating them."""
    logits, locations, scales = mixture_parameters(parameters)
    if not all(torch.isfinite(value).all() for value in (logits, locations, scales)):
        raise RuntimeError("V48 predicted Gaussian parameters are nonfinite")
    log_weights = torch.log_softmax(logits, dim=1)
    result: dict[str, float] = {}
    for order in (1, 2):
        coefficient = order * 4.5 * math.log(10.0) * target_std
        component = (
            coefficient * locations
            + 0.5 * coefficient**2 * torch.square(scales)
        )
        log_moment = torch.logsumexp(log_weights + component, dim=1)
        if not torch.isfinite(log_moment).all():
            raise RuntimeError("V48 finite Gaussian moment proof differs")
        result[f"maximum_standardized_residual_log_moment_order_{order}"] = float(
            log_moment.max().cpu()
        )
    return result


def preflight(
    program_path: Path,
    repo: Path,
    cache_path: Path,
    cache_sha: str,
    output: Path,
) -> dict[str, Any]:
    program, v35, _ = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V48 preflight requires a clean worktree")
    if (
        cache_path.resolve()
        != Path(program["inherited_experiment"]["v45_conditioning_cache"]).resolve()
        or cache_sha != V45_CACHE_SHA256
    ):
        raise ValueError("V48 must use the frozen V45 conditioning cache")
    device = _device()
    prepared = load_cache(cache_path, cache_sha, commit)
    handles: list[tuple[h5py.File, h5py.File]] = []
    try:
        conditions, targets = [], []
        for domain in DOMAIN_ORDER:
            data, cache = _open_split(v35["development_domains"][domain], "train")
            handles.append((data, cache))
            condition, target, _ = condition_cube(data, cache, prepared, domain, "train", 0)
            axes, reflections = CUBE_ISOMETRIES[7]
            transformed_condition = apply_cube_isometry(condition, axes, reflections)
            transformed_target = apply_cube_isometry(target, axes, reflections)
            restored_condition = apply_cube_isometry(
                transformed_condition,
                tuple(np.argsort(axes)),
                tuple(reflections[axes.index(i)] for i in range(3)),
            )
            if not np.array_equal(np.sort(condition.reshape(-1)), np.sort(restored_condition.reshape(-1))):
                raise RuntimeError("V48 isometry coverage differs")
            conditions.append(transformed_condition)
            targets.append(transformed_target)
        torch.cuda.reset_peak_memory_stats(device)
        model = LocalMixtureUNet().to(device)
        if parameter_count(model) != PARAMETERS:
            raise RuntimeError("V48 parameter count differs")
        identifiability = _identifiability_probe(device)
        condition_tensor = torch.from_numpy(np.stack(conditions)).to(device)
        target_tensor = torch.from_numpy(np.stack(targets)).to(device)
        parameters = model(condition_tensor)
        expected_bias = torch.tensor(
            INITIAL_BIASES, device=device, dtype=parameters.dtype
        ).reshape(1, 15, 1, 1, 1)
        initial_output_maximum_error = float(
            torch.max(torch.abs(parameters - expected_bias)).detach().cpu()
        )
        if (
            torch.count_nonzero(model.output.weight).item() != 0
            or initial_output_maximum_error > 1.0e-7
        ):
            raise RuntimeError("V48 final initialization differs")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=2.0e-4, weight_decay=1.0e-4
        )
        loss = -gaussian_mixture_log_probability(parameters, target_tensor).mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError("V48 real-data loss or gradients differ")
        if not any(torch.count_nonzero(g).item() for g in gradients):
            raise RuntimeError("V48 gradients are all zero")
        real_output_bias_gradient_spreads = _component_gradient_spreads(
            model.output.bias.grad
        )
        optimizer.step()
        with torch.no_grad():
            real_parameters_after_step = model(condition_tensor)
        physical_moment_proof = _finite_physical_log_moments(
            real_parameters_after_step, float(prepared["target_std"][()])
        )
        small_parameters = torch.randn(2, 15, 4, 4, 4, device=device)
        uniform = torch.linspace(1.0e-4, 1.0 - 1.0e-4, 2 * 4**3, device=device).reshape(2, 1, 4, 4, 4)
        inverse = gaussian_mixture_inverse(small_parameters, uniform)
        cdf_error = float(torch.max(torch.abs(gaussian_mixture_cdf(small_parameters, inverse) - uniform)))
        peak = int(torch.cuda.max_memory_allocated(device))
        if cdf_error > 2.0e-6 or peak >= 24 * 1024**3:
            raise RuntimeError("V48 mixture inverse or memory preflight differs")

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
            with torch.no_grad():
                query_parameters = model(torch.from_numpy(query_condition[None]).to(device))
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
                standardized = gaussian_mixture_inverse(
                    query_parameters, torch.from_numpy(rank[None]).to(device)
                ).cpu().numpy()[0]
            residual = standardized * float(prepared["target_std"][()]) + float(
                prepared["target_mean"][()]
            )
            residual -= residual.mean(dtype=np.float64)
            sample = query_backbone + residual
            real_dc = float(abs(residual.mean(dtype=np.float64)))
            if not np.isfinite(sample).all() or real_dc > 1.0e-7:
                raise RuntimeError("V48 real paired-donor sample differs")
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
        "real_source_balanced_loss": float(loss.detach().cpu()),
        "initial_output_maximum_error": initial_output_maximum_error,
        "identifiability_probe": identifiability,
        "finite_physical_Gaussian_moment_proof": physical_moment_proof,
        "real_output_bias_same_role_component_gradient_spread": real_output_bias_gradient_spreads,
        "mixture_CDF_inverse_maximum_error": cdf_error,
        "mixture_bisection_steps": BISECTION_STEPS,
        "peak_allocated_bytes": peak,
        "maximum_real_sample_residual_DC": real_dc,
        "cache": str(cache_path.resolve()),
        "cache_sha256": cache_sha,
        "validation_truth_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
        "spatial_rank_transport": False,
        "density_or_residual_clipping": False,
        "component_scale_cap": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    if output.exists():
        raise FileExistsError("V48 refuses existing preflight")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(result, indent=2), flush=True)
    return result


def _learning_rate(step: int) -> float:
    fraction = step / STEPS
    return 2.0e-5 + 0.5 * (2.0e-4 - 2.0e-5) * (1.0 + math.cos(math.pi * fraction))


@torch.no_grad()
def _validation_nll(
    model: LocalMixtureUNet,
    v35: dict[str, Any],
    prepared: h5py.File,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    result: dict[str, float] = {}
    for domain in DOMAIN_ORDER:
        data, cache = _open_split(v35["development_domains"][domain], "validation")
        values = []
        try:
            for index in range(int(v35["development_domains"][domain]["validation_objects"])):
                condition, target, _ = condition_cube(
                    data, cache, prepared, domain, "validation", index
                )
                parameter = model(torch.from_numpy(condition[None]).to(device))
                value = torch.from_numpy(target[None]).to(device)
                values.append(float((-gaussian_mixture_log_probability(parameter, value)).mean().cpu()))
        finally:
            data.close()
            cache.close()
        result[domain] = float(np.mean(values))
    model.train()
    return result


def train(
    program_path: Path,
    repo: Path,
    cache_path: Path,
    cache_sha: str,
    preflight_path: Path,
    preflight_sha: str,
    checkpoint_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    program, v35, _ = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V48 training requires a clean worktree")
    if (
        cache_path.resolve()
        != Path(program["inherited_experiment"]["v45_conditioning_cache"]).resolve()
        or cache_sha != V45_CACHE_SHA256
    ):
        raise ValueError("V48 must use the frozen V45 conditioning cache")
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError("V48 refuses existing training outputs")
    checked = _verified_json(preflight_path, preflight_sha, "V48 preflight")
    if (
        checked.get("schema") != PREFLIGHT_SCHEMA
        or checked.get("status") != "pass"
        or checked.get("code_commit") != commit
        or checked.get("cache_sha256") != cache_sha
    ):
        raise ValueError("V48 preflight binding differs")
    device = _device()
    prepared = load_cache(cache_path, cache_sha, commit)
    torch.manual_seed(144044)
    torch.cuda.manual_seed_all(144044)
    generator = np.random.default_rng(144044)
    model = LocalMixtureUNet().to(device)
    ema_model = LocalMixtureUNet().to(device)
    ema_model.load_state_dict(model.state_dict())
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.0e-4, weight_decay=1.0e-4
    )
    scaler = torch.amp.GradScaler("cuda")
    handles = {
        domain: _open_split(v35["development_domains"][domain], "train")
        for domain in DOMAIN_ORDER
    }
    log_rows: list[dict[str, float | int]] = []
    validation_rows: dict[str, Any] = {}
    try:
        model.train()
        for step in range(1, STEPS + 1):
            conditions, targets = [], []
            for domain in DOMAIN_ORDER:
                row = v35["development_domains"][domain]
                index = int(generator.integers(int(row["train_objects"])))
                data, cache = handles[domain]
                condition, target, _ = condition_cube(
                    data, cache, prepared, domain, "train", index
                )
                isometry = int(generator.integers(len(CUBE_ISOMETRIES)))
                axes, reflections = CUBE_ISOMETRIES[isometry]
                conditions.append(apply_cube_isometry(condition, axes, reflections))
                targets.append(apply_cube_isometry(target, axes, reflections))
            condition_tensor = torch.from_numpy(np.stack(conditions)).to(device)
            target_tensor = torch.from_numpy(np.stack(targets)).to(device)
            lr = _learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                parameters = model(condition_tensor)
            loss = -gaussian_mixture_log_probability(parameters, target_tensor).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
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
                    "loss": float(loss.detach().cpu()),
                    "learning_rate": lr,
                    "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                }
                log_rows.append(row)
                print("[v48-train] " + json.dumps(row), flush=True)
            if step in VALIDATION_STEPS:
                validation_rows[str(step)] = _validation_nll(
                    ema_model, v35, prepared, device
                )
                print(
                    f"[v48-validation] step={step} {json.dumps(validation_rows[str(step)])}",
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
        "input_channels": INPUT_CHANNELS,
        "mixtures": 5,
        "likelihood_family": "Gaussian",
        "ema_decay": EMA_DECAY,
        "ema_state_dict": {
            key: value.detach().cpu() for key, value in ema_model.state_dict().items()
        },
        "conditioning_cache": str(cache_path.resolve()),
        "conditioning_cache_sha256": cache_sha,
        "preflight": str(preflight_path.resolve()),
        "preflight_sha256": preflight_sha,
        "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
        "spatial_rank_transport": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    partial_checkpoint = checkpoint_path.with_suffix(checkpoint_path.suffix + ".partial")
    torch.save(checkpoint, partial_checkpoint)
    os.replace(partial_checkpoint, checkpoint_path)
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete_fixed_12000_step_train_only_fit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "conditioning_cache": str(cache_path.resolve()),
        "conditioning_cache_sha256": cache_sha,
        "preflight": str(preflight_path.resolve()),
        "preflight_sha256": preflight_sha,
        "steps": STEPS,
        "parameters": PARAMETERS,
        "training_log": log_rows,
        "fixed_validation_NLL_diagnostic": validation_rows,
        "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
        "density_or_tail_weighted_loss": False,
        "likelihood_family": "Gaussian",
        "component_scale_cap": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial_report = report_path.with_suffix(report_path.suffix + ".partial")
    partial_report.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial_report, report_path)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--program", type=Path, required=True)
    check.add_argument("--repo", type=Path, required=True)
    check.add_argument("--cache", type=Path, required=True)
    check.add_argument("--cache-sha256", required=True)
    check.add_argument("--out", type=Path, required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--program", type=Path, required=True)
    fit.add_argument("--repo", type=Path, required=True)
    fit.add_argument("--cache", type=Path, required=True)
    fit.add_argument("--cache-sha256", required=True)
    fit.add_argument("--preflight", type=Path, required=True)
    fit.add_argument("--preflight-sha256", required=True)
    fit.add_argument("--checkpoint", type=Path, required=True)
    fit.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.program, args.repo, args.cache, args.cache_sha256, args.out)
    else:
        train(
            args.program,
            args.repo,
            args.cache,
            args.cache_sha256,
            args.preflight,
            args.preflight_sha256,
            args.checkpoint,
            args.report,
        )


if __name__ == "__main__":
    main()
