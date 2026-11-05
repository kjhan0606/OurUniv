#!/usr/bin/env python
"""Preflight and train the frozen V50 bounded-logit mixture likelihood."""
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
from hong2021_v48_train import (
    V45_CACHE_SHA256,
    condition_cube,
    load_cache,
    load_program as load_v48_program,
)
from hong2021_v50_network import (
    BISECTION_STEPS,
    INITIAL_BIASES,
    INPUT_CHANNELS,
    LOWER_SUPPORT,
    QUADRATURE_INITIAL_STANDARDIZED_MEAN,
    QUADRATURE_INITIAL_STANDARDIZED_VARIANCE,
    UPPER_SUPPORT,
    LocalMixtureUNet,
    bounded_mixture_cdf,
    bounded_mixture_inverse,
    bounded_mixture_log_probability,
    bounded_to_latent,
    initial_standardized_quadrature,
    latent_to_bounded,
    mixture_parameters,
    parameter_count,
)


PROGRAM_SCHEMA = "hong2021-v50-bounded-logit-Gaussian-mixture-copula-development-program-v1"
PROGRAM_SHA256 = "f67e8f5a9eacf40eb138774632add896e922e84d944263bb245ff2a71344d85f"
PREFLIGHT_SCHEMA = "hong2021-v50-bounded-logit-mixture-copula-hard-preflight-v1"
CHECKPOINT_SCHEMA = "hong2021-v50-bounded-logit-mixture-copula-checkpoint-v1"
REPORT_SCHEMA = "hong2021-v50-bounded-logit-mixture-copula-training-report-v1"
SUPPORT_SHA256 = "3c339a76a5f1172ed30b5aeb5fe14e958a123ec10452e8386d52fbd1269e76b2"
SUPPORT_DECISION_DIGEST = "91335a4be726aaae1cc8354aaa96fedc338bc65c9f2afd8ed4aedbb438a9dfce"
STEPS = 12_000
VALIDATION_STEPS = (4_000, 8_000, 12_000)
PARAMETERS = 8_490_415
EMA_DECAY = 0.999
LIKELIHOOD_FAMILY = "bounded_logit_Gaussian_mixture"


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V50 {label} hash differs")
    return json.loads(path.read_text())


def load_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status")
        != "frozen_before_model_implementation_training_sampling_or_development_evaluation"
    ):
        raise ValueError("V50 program schema or status differs")
    parent = program["parent_evidence"]
    v49 = _verified_json(
        (repo / parent["v49_record"]).resolve(),
        parent["v49_record_sha256"],
        "V49 record",
    )
    support_record = _verified_json(
        (repo / parent["v50_support_record"]).resolve(),
        parent["v50_support_record_sha256"],
        "support record",
    )
    v48_record = _verified_json(
        (repo / parent["v48_record"]).resolve(),
        parent["v48_record_sha256"],
        "V48 record",
    )
    if (
        v49.get("status") != parent["required_status"]
        or v49.get("selected_next_likelihood", {}).get("family")
        != parent["required_family"]
        or support_record.get("status") != parent["required_support_status"]
        or support_record.get("support", {}).get("lower") != LOWER_SUPPORT
        or support_record.get("support", {}).get("upper") != UPPER_SUPPORT
        or support_record.get("firewall", {}).get("independent_gate_locked") is not True
        or v48_record.get("firewall", {}).get("independent_gate_locked") is not True
    ):
        raise ValueError("V50 parent evidence differs")
    immutable = program["immutable_inputs"]
    v48_path = (repo / immutable["v48_program"]).resolve()
    support_path = Path(immutable["support_selection"]).resolve()
    if (
        sha256_file(v48_path) != immutable["v48_program_sha256"]
        or sha256_file(support_path) != SUPPORT_SHA256
        or immutable["support_selection_sha256"] != SUPPORT_SHA256
    ):
        raise ValueError("V50 immutable input hash differs")
    support = json.loads(support_path.read_text())
    if (
        support.get("decision_digest_sha256") != SUPPORT_DECISION_DIGEST
        or immutable["support_selection_decision_digest_sha256"]
        != SUPPORT_DECISION_DIGEST
        or support.get("support", {}).get("lower_support") != LOWER_SUPPORT
        or support.get("support", {}).get("upper_support") != UPPER_SUPPORT
        or support.get("all_train_values_strictly_interior") is not True
    ):
        raise ValueError("V50 support artifact differs")
    v48, v35, v41 = load_v48_program(v48_path, repo)
    effective = dict(program)
    effective["inherited_inputs"] = v48["inherited_inputs"]
    return effective, v35, v41


def _device() -> torch.device:
    if socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V50 requires Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V50 requires the Lageunha Ada GPU")
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
    symmetric = symmetric_bias.reshape(1, 15, 1, 1, 1).expand(1, 15, 4, 4, 4)
    symmetric_loss = -bounded_mixture_log_probability(symmetric, target).mean()
    symmetric_loss.backward()
    symmetric_spreads = _component_gradient_spreads(symmetric_bias.grad)
    if max(symmetric_spreads.values()) > 1.0e-7:
        raise RuntimeError("V50 symmetric latent Gaussian control differs")

    bias = torch.nn.Parameter(torch.tensor(INITIAL_BIASES, device=device))
    initial = bias.reshape(1, 15, 1, 1, 1)
    _, locations, _ = mixture_parameters(initial)
    if torch.unique(locations).numel() != 5:
        raise RuntimeError("V50 initial latent locations differ")
    nodes, weights = np.polynomial.hermite.hermgauss(128)
    mean, variance = initial_standardized_quadrature(
        torch.from_numpy(nodes).to(device), torch.from_numpy(weights).to(device)
    )
    if (
        abs(mean - QUADRATURE_INITIAL_STANDARDIZED_MEAN) > 1.0e-6
        or abs(variance - QUADRATURE_INITIAL_STANDARDIZED_VARIANCE) > 1.0e-6
    ):
        raise RuntimeError("V50 frozen initial quadrature differs")

    optimizer = torch.optim.AdamW([bias], lr=2.0e-4, weight_decay=1.0e-4)
    gradient_spreads: dict[str, float] | None = None
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        parameters = bias.reshape(1, 15, 1, 1, 1).expand(1, 15, 4, 4, 4)
        loss = -bounded_mixture_log_probability(parameters, target).mean()
        loss.backward()
        if gradient_spreads is None:
            gradient_spreads = _component_gradient_spreads(bias.grad)
        optimizer.step()
    assert gradient_spreads is not None
    _, final_locations, final_scales = mixture_parameters(
        bias.detach().reshape(1, 15, 1, 1, 1)
    )
    if (
        min(gradient_spreads.values()) <= 1.0e-8
        or float((final_locations.max() - final_locations.min()).cpu()) <= 1.0e-3
        or float((final_scales.max() - final_scales.min()).cpu()) <= 1.0e-6
    ):
        raise RuntimeError("V50 components are not dynamically identifiable")
    return {
        "symmetric_same_role_component_gradient_spread": symmetric_spreads,
        "V50_same_role_component_gradient_spread": gradient_spreads,
        "quadrature_initial_standardized_mean": mean,
        "quadrature_initial_standardized_variance": variance,
        "component_location_spread_after_two_steps": float(
            (final_locations.max() - final_locations.min()).cpu()
        ),
        "component_scale_spread_after_two_steps": float(
            (final_scales.max() - final_scales.min()).cpu()
        ),
    }


def _bounded_physical_moment_proof(target_mean: float, target_std: float) -> dict[str, float]:
    lower = target_mean + target_std * LOWER_SUPPORT
    upper = target_mean + target_std * UPPER_SUPPORT
    if not all(math.isfinite(value) for value in (lower, upper)) or upper <= lower:
        raise RuntimeError("V50 physical bound differs")
    coefficient = 4.5 * math.log(10.0)
    return {
        "minimum_generated_standardized_residual": LOWER_SUPPORT,
        "maximum_generated_standardized_residual": UPPER_SUPPORT,
        "minimum_generated_physical_residual": lower,
        "maximum_generated_physical_residual": upper,
        "upper_log_density_moment_bound_order_1": coefficient * upper,
        "upper_log_density_moment_bound_order_2": 2.0 * coefficient * upper,
    }


def preflight(
    program_path: Path,
    repo: Path,
    cache_path: Path,
    cache_sha: str,
    output: Path,
) -> dict[str, Any]:
    program, v35, _ = load_program(program_path, repo.resolve())
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V50 preflight requires a clean worktree")
    if (
        cache_path.resolve()
        != Path(program["immutable_inputs"]["v45_conditioning_cache"]).resolve()
        or cache_sha != V45_CACHE_SHA256
    ):
        raise ValueError("V50 must use the immutable V45 conditioning cache")
    device = _device()
    prepared = load_cache(cache_path, cache_sha, commit)
    handles: list[tuple[h5py.File, h5py.File]] = []
    try:
        conditions, targets = [], []
        minimum_target = math.inf
        maximum_target = -math.inf
        for domain in DOMAIN_ORDER:
            data, cache = _open_split(v35["development_domains"][domain], "train")
            handles.append((data, cache))
            condition, target, _ = condition_cube(
                data, cache, prepared, domain, "train", 0
            )
            if not np.all((target > LOWER_SUPPORT) & (target < UPPER_SUPPORT)):
                raise RuntimeError("V50 real train target lies outside support")
            minimum_target = min(minimum_target, float(target.min()))
            maximum_target = max(maximum_target, float(target.max()))
            axes, reflections = CUBE_ISOMETRIES[7]
            conditions.append(apply_cube_isometry(condition, axes, reflections))
            targets.append(apply_cube_isometry(target, axes, reflections))
        torch.cuda.reset_peak_memory_stats(device)
        model = LocalMixtureUNet().to(device)
        if parameter_count(model) != PARAMETERS:
            raise RuntimeError("V50 parameter count differs")
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
            raise RuntimeError("V50 final initialization differs")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=2.0e-4, weight_decay=1.0e-4
        )
        loss = -bounded_mixture_log_probability(parameters, target_tensor).mean()
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if (
            not torch.isfinite(loss)
            or not gradients
            or not all(torch.isfinite(gradient).all() for gradient in gradients)
            or not any(torch.count_nonzero(gradient).item() for gradient in gradients)
        ):
            raise RuntimeError("V50 real-data loss or gradients differ")
        real_gradient_spreads = _component_gradient_spreads(model.output.bias.grad)
        optimizer.step()
        proof = _bounded_physical_moment_proof(
            float(prepared["target_mean"][()]), float(prepared["target_std"][()])
        )
        support_values = torch.linspace(
            LOWER_SUPPORT + 0.01,
            UPPER_SUPPORT - 0.01,
            1024,
            device=device,
        )
        transform_error = float(
            torch.max(
                torch.abs(latent_to_bounded(bounded_to_latent(support_values)) - support_values)
            ).cpu()
        )
        small_parameters = torch.randn(2, 15, 4, 4, 4, device=device)
        uniform = torch.linspace(
            1.0e-4, 1.0 - 1.0e-4, 2 * 4**3, device=device
        ).reshape(2, 1, 4, 4, 4)
        inverse = bounded_mixture_inverse(small_parameters, uniform)
        cdf_error = float(
            torch.max(torch.abs(bounded_mixture_cdf(small_parameters, inverse) - uniform)).cpu()
        )
        peak = int(torch.cuda.max_memory_allocated(device))
        if transform_error > 2.0e-6 or cdf_error > 2.0e-6 or peak >= 24 * 1024**3:
            raise RuntimeError("V50 transform, inverse, or memory preflight differs")

        selections = _selection_arrays(v35)
        domain = DOMAIN_ORDER[0]
        query_index = int(selections[domain]["source_index"][0])
        donor_source = DOMAIN_ORDER[int(selections[domain]["donor_source"][0, 0])]
        donor_index = int(selections[domain]["donor_index"][0, 0])
        isometry = int(selections[domain]["donor_isometry"][0, 0])
        query_data, query_cache = _open_split(
            v35["development_domains"][domain], "validation"
        )
        donor_data, donor_cache = _open_split(
            v35["development_domains"][donor_source], "train"
        )
        try:
            query_condition, _, query_backbone = condition_cube(
                query_data, query_cache, prepared, domain, "validation", query_index
            )
            with torch.no_grad():
                query_parameters = model(
                    torch.from_numpy(query_condition[None]).to(device)
                )
            donor_backbone = _backbone(donor_cache, donor_index)[None]
            donor_truth = np.asarray(donor_data["target"][donor_index], dtype=np.float32)
            copula = load_model(
                Path(program["inherited_inputs"]["conditional_copula_artifact"]),
                program["inherited_inputs"]["conditional_copula_artifact_sha256"],
            )
            rank = conditional_forward(
                donor_truth - donor_backbone, donor_backbone, copula
            )
            axes, reflections = CUBE_ISOMETRIES[isometry]
            rank = apply_cube_isometry(rank, axes, reflections)
            with torch.no_grad():
                standardized = bounded_mixture_inverse(
                    query_parameters, torch.from_numpy(rank[None]).to(device)
                ).cpu().numpy()[0]
            if not np.all(
                (standardized > LOWER_SUPPORT) & (standardized < UPPER_SUPPORT)
            ):
                raise RuntimeError("V50 real sample left bounded support")
            residual = standardized * float(prepared["target_std"][()]) + float(
                prepared["target_mean"][()]
            )
            residual -= residual.mean(dtype=np.float64)
            sample = query_backbone + residual
            real_dc = float(abs(residual.mean(dtype=np.float64)))
            if not np.isfinite(sample).all() or real_dc > 1.0e-7:
                raise RuntimeError("V50 real paired-donor sample differs")
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
        "support_selection_sha256": SUPPORT_SHA256,
        "support_selection_decision_digest_sha256": SUPPORT_DECISION_DIGEST,
        "open_standardized_support": [LOWER_SUPPORT, UPPER_SUPPORT],
        "real_source_balanced_loss": float(loss.detach().cpu()),
        "real_source_balanced_target_extrema": [minimum_target, maximum_target],
        "initial_output_maximum_error": initial_output_maximum_error,
        "identifiability_probe": identifiability,
        "bounded_physical_moment_proof": proof,
        "support_transform_maximum_error": transform_error,
        "real_output_bias_same_role_component_gradient_spread": real_gradient_spreads,
        "mixture_CDF_inverse_maximum_error": cdf_error,
        "mixture_bisection_steps": BISECTION_STEPS,
        "peak_allocated_bytes": peak,
        "maximum_real_sample_residual_DC": real_dc,
        "cache": str(cache_path.resolve()),
        "cache_sha256": cache_sha,
        "validation_truth_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
        "spatial_rank_transport": False,
        "sample_clipping": False,
        "component_scale_cap": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    if output.exists():
        raise FileExistsError("V50 refuses existing preflight")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(result, indent=2), flush=True)
    return result


def _learning_rate(step: int) -> float:
    fraction = step / STEPS
    return 2.0e-5 + 0.5 * (2.0e-4 - 2.0e-5) * (
        1.0 + math.cos(math.pi * fraction)
    )


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
            for index in range(
                int(v35["development_domains"][domain]["validation_objects"])
            ):
                condition, target, _ = condition_cube(
                    data, cache, prepared, domain, "validation", index
                )
                if not np.all((target > LOWER_SUPPORT) & (target < UPPER_SUPPORT)):
                    raise RuntimeError("V50 validation diagnostic lies outside support")
                parameters = model(torch.from_numpy(condition[None]).to(device))
                observed = torch.from_numpy(target[None]).to(device)
                values.append(
                    float((-bounded_mixture_log_probability(parameters, observed)).mean().cpu())
                )
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
    program, v35, _ = load_program(program_path, repo.resolve())
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V50 training requires a clean worktree")
    if (
        cache_path.resolve()
        != Path(program["immutable_inputs"]["v45_conditioning_cache"]).resolve()
        or cache_sha != V45_CACHE_SHA256
    ):
        raise ValueError("V50 must use the immutable V45 conditioning cache")
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError("V50 refuses existing training outputs")
    checked = _verified_json(preflight_path, preflight_sha, "preflight")
    if (
        checked.get("schema") != PREFLIGHT_SCHEMA
        or checked.get("status") != "pass"
        or checked.get("code_commit") != commit
        or checked.get("cache_sha256") != cache_sha
        or checked.get("support_selection_sha256") != SUPPORT_SHA256
    ):
        raise ValueError("V50 preflight binding differs")
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
            loss = -bounded_mixture_log_probability(parameters, target_tensor).mean()
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
                print("[v50-train] " + json.dumps(row), flush=True)
            if step in VALIDATION_STEPS:
                validation_rows[str(step)] = _validation_nll(
                    ema_model, v35, prepared, device
                )
                print(
                    f"[v50-validation] step={step} {json.dumps(validation_rows[str(step)])}",
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
        "likelihood_family": LIKELIHOOD_FAMILY,
        "open_standardized_support": [LOWER_SUPPORT, UPPER_SUPPORT],
        "support_selection_sha256": SUPPORT_SHA256,
        "ema_decay": EMA_DECAY,
        "ema_state_dict": {
            key: value.detach().cpu()
            for key, value in ema_model.state_dict().items()
        },
        "conditioning_cache": str(cache_path.resolve()),
        "conditioning_cache_sha256": cache_sha,
        "preflight": str(preflight_path.resolve()),
        "preflight_sha256": preflight_sha,
        "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
        "spatial_rank_transport": False,
        "sample_clipping": False,
        "component_scale_cap": False,
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
        "support_selection_sha256": SUPPORT_SHA256,
        "open_standardized_support": [LOWER_SUPPORT, UPPER_SUPPORT],
        "steps": STEPS,
        "parameters": PARAMETERS,
        "training_log": log_rows,
        "fixed_validation_NLL_diagnostic": validation_rows,
        "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
        "density_or_tail_weighted_loss": False,
        "likelihood_family": LIKELIHOOD_FAMILY,
        "sample_clipping": False,
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
