#!/usr/bin/env python
"""Preflight and train the frozen V52 matched no-risk bounded mixture."""
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
from hong2021_v48_train import V45_CACHE_SHA256, condition_cube, load_cache
from hong2021_v50_network import (
    BISECTION_STEPS,
    INITIAL_BIASES,
    INPUT_CHANNELS,
    LOWER_SUPPORT,
    UPPER_SUPPORT,
    LocalMixtureUNet,
    bounded_mixture_cdf,
    bounded_mixture_inverse,
    bounded_mixture_log_probability,
    bounded_to_latent,
    latent_to_bounded,
    parameter_count,
)
from hong2021_v50_train import (
    _bounded_physical_moment_proof,
    _component_gradient_spreads,
    _device,
    _identifiability_probe,
    load_program as load_v50_program,
)


PROGRAM_SCHEMA = "hong2021-v52-matched-no-risk-bounded-mixture-development-program-v1"
PROGRAM_SHA256 = "4831c79c1c0e2a06a62d48d8665b3b8a57eea32e92b3bf1a4ae16acd67437413"
PREFLIGHT_SCHEMA = "hong2021-v52-matched-no-risk-bounded-mixture-hard-preflight-v1"
CHECKPOINT_SCHEMA = "hong2021-v52-matched-no-risk-bounded-mixture-checkpoint-v1"
REPORT_SCHEMA = "hong2021-v52-matched-no-risk-bounded-mixture-training-report-v1"
SUPPORT_SHA256 = "3c339a76a5f1172ed30b5aeb5fe14e958a123ec10452e8386d52fbd1269e76b2"
SUPPORT_DECISION_DIGEST = "91335a4be726aaae1cc8354aaa96fedc338bc65c9f2afd8ed4aedbb438a9dfce"
STEPS = 12_000
VALIDATION_STEPS = (4_000, 8_000, 12_000)
PARAMETERS = 8_490_415
EMA_DECAY = 0.999
LIKELIHOOD_FAMILY = "bounded_logit_Gaussian_mixture"
SEED = 144044


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V52 {label} hash differs")
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
        raise ValueError("V52 program schema or status differs")
    parent = program["parent_evidence"]
    v51 = _verified_json(
        (repo / parent["v51_record"]).resolve(),
        parent["v51_record_sha256"],
        "V51 record",
    )
    if (
        v51.get("status") != parent["required_status"]
        or v51.get("audit", {}).get("classification")
        != parent["required_classification"]
        or v51.get("audit", {}).get("next") != parent["required_next"]
        or v51.get("selected_next_model", {}).get("model")
        != parent["required_selected_model"]
        or v51.get("firewall", {}).get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
        or v51.get("firewall", {}).get("Astrid_accessed") is not False
        or v51.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V52 V51 evidence or firewall differs")
    frozen = program["frozen_inputs"]
    v50_record = _verified_json(
        (repo / frozen["v50_result_record"]).resolve(),
        frozen["v50_result_record_sha256"],
        "V50 result record",
    )
    for key, digest_key in (
        ("v50_program", "v50_program_sha256"),
        ("v50_result_record", "v50_result_record_sha256"),
        ("v50_checkpoint", "v50_checkpoint_sha256"),
        ("v50_training_report", "v50_training_report_sha256"),
        ("v50_preflight", "v50_preflight_sha256"),
        ("v50_development_decision", "v50_development_decision_sha256"),
        ("v51_audit", "v51_audit_sha256"),
        ("conditioning_cache", "conditioning_cache_sha256"),
        ("support_selection", "support_selection_sha256"),
    ):
        candidate = Path(frozen[key])
        if not candidate.is_absolute():
            candidate = repo / candidate
        if sha256_file(candidate.resolve()) != frozen[digest_key]:
            raise ValueError(f"V52 frozen {key} hash differs")
    support = json.loads(Path(frozen["support_selection"]).read_text())
    v50_report = json.loads(Path(frozen["v50_training_report"]).read_text())
    v50_decision = json.loads(Path(frozen["v50_development_decision"]).read_text())
    v51_audit = json.loads(Path(frozen["v51_audit"]).read_text())
    if (
        canonical_digest(support) != SUPPORT_DECISION_DIGEST
        or frozen["support_selection_decision_digest_sha256"]
        != SUPPORT_DECISION_DIGEST
        or canonical_digest(v50_decision)
        != frozen["v50_development_decision_digest_sha256"]
        or canonical_digest(v50_report)
        != v50_record.get("training", {}).get("report_decision_digest_sha256")
        or canonical_digest(v51_audit) != frozen["v51_audit_decision_digest_sha256"]
        or support.get("support", {}).get("lower_support") != LOWER_SUPPORT
        or support.get("support", {}).get("upper_support") != UPPER_SUPPORT
        or v51_audit.get("classification") != parent["required_classification"]
        or v51_audit.get("next") != parent["required_next"]
        or v51_audit.get("independent_gate_locked") is not True
        or v50_record.get("development_decision", {}).get("classification")
        != "bounded_marginal_support_preserves_the_field_body_but_is_not_sufficient_for_extreme_calibration"
        or v50_record.get("firewall", {}).get("independent_gate_locked") is not True
    ):
        raise ValueError("V52 support, V50 decision, or V51 audit digest differs")
    v50, v35, v41 = load_v50_program(
        (repo / frozen["v50_program"]).resolve(), repo
    )
    effective = dict(program)
    effective["inherited_inputs"] = v50["inherited_inputs"]
    return effective, v35, v41


def no_risk_condition_cube(
    data: h5py.File,
    cache: h5py.File,
    prepared: h5py.File,
    domain: str,
    split: str,
    index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    condition, target, backbone = condition_cube(
        data,
        cache,
        prepared,
        domain,
        split,
        index,
        risk_ablation=True,
    )
    if np.count_nonzero(condition[5]) != 0:
        raise RuntimeError("V52 risk channel is not exact standardized zero")
    return condition, target, backbone


def _learning_rate(step: int) -> float:
    fraction = step / STEPS
    return 2.0e-5 + 0.5 * (2.0e-4 - 2.0e-5) * (
        1.0 + math.cos(math.pi * fraction)
    )


def _same_seed_initialization(device: torch.device) -> tuple[LocalMixtureUNet, float]:
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    reference = LocalMixtureUNet().to(device)
    reference_state = {
        key: value.detach().clone() for key, value in reference.state_dict().items()
    }
    del reference
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = LocalMixtureUNet().to(device)
    maximum = max(
        float(torch.max(torch.abs(value - reference_state[key])).cpu())
        for key, value in model.state_dict().items()
    )
    return model, maximum


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
        raise RuntimeError("V52 preflight requires a clean worktree")
    if (
        cache_path.resolve() != Path(program["frozen_inputs"]["conditioning_cache"]).resolve()
        or cache_sha != V45_CACHE_SHA256
    ):
        raise ValueError("V52 must use the immutable V45 conditioning cache")
    if output.exists():
        raise FileExistsError("V52 refuses existing preflight")
    device = _device()
    prepared = load_cache(cache_path, cache_sha, commit)
    handles: list[tuple[h5py.File, h5py.File]] = []
    conditions, targets = [], []
    active_maximum_error = 0.0
    target_maximum_error = 0.0
    backbone_maximum_error = 0.0
    original_risk_minimum = math.inf
    original_risk_maximum = -math.inf
    minimum_target = math.inf
    maximum_target = -math.inf
    try:
        for domain in DOMAIN_ORDER:
            for split in ("train", "validation"):
                data, cache = _open_split(v35["development_domains"][domain], split)
                handles.append((data, cache))
                original, original_target, original_backbone = condition_cube(
                    data, cache, prepared, domain, split, 0
                )
                ablated, target, backbone = no_risk_condition_cube(
                    data, cache, prepared, domain, split, 0
                )
                active = (0, 1, 2, 3, 4, 6)
                active_maximum_error = max(
                    active_maximum_error,
                    float(np.max(np.abs(ablated[list(active)] - original[list(active)]))),
                )
                target_maximum_error = max(
                    target_maximum_error, float(np.max(np.abs(target - original_target)))
                )
                backbone_maximum_error = max(
                    backbone_maximum_error,
                    float(np.max(np.abs(backbone - original_backbone))),
                )
                original_risk_minimum = min(
                    original_risk_minimum, float(original[5].min())
                )
                original_risk_maximum = max(
                    original_risk_maximum, float(original[5].max())
                )
                if not np.all((target > LOWER_SUPPORT) & (target < UPPER_SUPPORT)):
                    raise RuntimeError("V52 real target lies outside frozen support")
                if split == "train":
                    minimum_target = min(minimum_target, float(target.min()))
                    maximum_target = max(maximum_target, float(target.max()))
                    axes, reflections = CUBE_ISOMETRIES[7]
                    conditions.append(apply_cube_isometry(ablated, axes, reflections))
                    targets.append(apply_cube_isometry(target, axes, reflections))
        if (
            active_maximum_error != 0.0
            or target_maximum_error != 0.0
            or backbone_maximum_error != 0.0
            or original_risk_maximum <= original_risk_minimum
        ):
            raise RuntimeError("V52 intervention is not isolated or substantive")
        torch.cuda.reset_peak_memory_stats(device)
        model, initialization_error = _same_seed_initialization(device)
        if (
            parameter_count(model) != PARAMETERS
            or initialization_error != 0.0
            or not all(parameter.requires_grad for parameter in model.parameters())
        ):
            raise RuntimeError("V52 matched architecture or initialization differs")
        identifiability = _identifiability_probe(device)
        condition_tensor = torch.from_numpy(np.stack(conditions)).to(device)
        target_tensor = torch.from_numpy(np.stack(targets)).to(device)
        if torch.count_nonzero(condition_tensor[:, 5]).item() != 0:
            raise RuntimeError("V52 augmented risk channel differs")
        parameters = model(condition_tensor)
        expected_bias = torch.tensor(
            INITIAL_BIASES, device=device, dtype=parameters.dtype
        ).reshape(1, 15, 1, 1, 1)
        output_error = float(torch.max(torch.abs(parameters - expected_bias)).cpu())
        if torch.count_nonzero(model.output.weight).item() != 0 or output_error > 1.0e-7:
            raise RuntimeError("V52 output initialization differs")
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=2.0e-4, weight_decay=1.0e-4
        )
        loss = -bounded_mixture_log_probability(parameters, target_tensor).mean()
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if (
            not torch.isfinite(loss)
            or not gradients
            or not all(torch.isfinite(gradient).all() for gradient in gradients)
            or not any(torch.count_nonzero(gradient).item() for gradient in gradients)
        ):
            raise RuntimeError("V52 real-data loss or gradients differ")
        gradient_spreads = _component_gradient_spreads(model.output.bias.grad)
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
                torch.abs(
                    latent_to_bounded(bounded_to_latent(support_values))
                    - support_values
                )
            ).cpu()
        )
        small_parameters = torch.randn(2, 15, 4, 4, 4, device=device)
        uniform = torch.linspace(
            1.0e-4, 1.0 - 1.0e-4, 2 * 4**3, device=device
        ).reshape(2, 1, 4, 4, 4)
        inverse = bounded_mixture_inverse(small_parameters, uniform)
        cdf_error = float(
            torch.max(
                torch.abs(bounded_mixture_cdf(small_parameters, inverse) - uniform)
            ).cpu()
        )
        peak = int(torch.cuda.max_memory_allocated(device))
        if transform_error > 2.0e-6 or cdf_error > 2.0e-6 or peak >= 24 * 1024**3:
            raise RuntimeError("V52 transform, inverse, or memory preflight differs")

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
            query_condition, _, query_backbone = no_risk_condition_cube(
                query_data, query_cache, prepared, domain, "validation", query_index
            )
            with torch.no_grad():
                query_parameters = model(
                    torch.from_numpy(query_condition[None]).to(device)
                )
            donor_backbone = _backbone(donor_cache, donor_index)[None]
            donor_truth = np.asarray(
                donor_data["target"][donor_index], dtype=np.float32
            )
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
                raise RuntimeError("V52 paired-donor sample left frozen support")
            residual = standardized * float(prepared["target_std"][()]) + float(
                prepared["target_mean"][()]
            )
            residual -= residual.mean(dtype=np.float64)
            sample = query_backbone + residual
            real_dc = float(abs(residual.mean(dtype=np.float64)))
            if not np.isfinite(sample).all() or real_dc > 1.0e-7:
                raise RuntimeError("V52 paired-donor sample differs")
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
        "same_seed_initialization_maximum_error": initialization_error,
        "risk_channel_exact_standardized_zero": True,
        "active_condition_maximum_error_from_V50_preprocessing": active_maximum_error,
        "target_maximum_error_from_V50_preprocessing": target_maximum_error,
        "backbone_maximum_error_from_V50_preprocessing": backbone_maximum_error,
        "original_risk_channel_extrema": [
            original_risk_minimum,
            original_risk_maximum,
        ],
        "support_selection_sha256": SUPPORT_SHA256,
        "support_selection_decision_digest_sha256": SUPPORT_DECISION_DIGEST,
        "open_standardized_support": [LOWER_SUPPORT, UPPER_SUPPORT],
        "real_source_balanced_loss": float(loss.detach().cpu()),
        "real_source_balanced_target_extrema": [minimum_target, maximum_target],
        "initial_output_maximum_error": output_error,
        "identifiability_probe": identifiability,
        "bounded_physical_moment_proof": proof,
        "support_transform_maximum_error": transform_error,
        "real_output_bias_same_role_component_gradient_spread": gradient_spreads,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, output)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


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
                condition, target, _ = no_risk_condition_cube(
                    data, cache, prepared, domain, "validation", index
                )
                if not np.all((target > LOWER_SUPPORT) & (target < UPPER_SUPPORT)):
                    raise RuntimeError("V52 validation diagnostic lies outside support")
                parameters = model(torch.from_numpy(condition[None]).to(device))
                observed = torch.from_numpy(target[None]).to(device)
                values.append(
                    float(
                        (-bounded_mixture_log_probability(parameters, observed))
                        .mean()
                        .cpu()
                    )
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
        raise RuntimeError("V52 training requires a clean worktree")
    if (
        cache_path.resolve() != Path(program["frozen_inputs"]["conditioning_cache"]).resolve()
        or cache_sha != V45_CACHE_SHA256
    ):
        raise ValueError("V52 must use the immutable V45 conditioning cache")
    if checkpoint_path.exists() or report_path.exists():
        raise FileExistsError("V52 refuses existing training outputs")
    checked = _verified_json(preflight_path, preflight_sha, "preflight")
    if (
        checked.get("schema") != PREFLIGHT_SCHEMA
        or checked.get("status") != "pass"
        or checked.get("code_commit") != commit
        or checked.get("cache_sha256") != cache_sha
        or checked.get("support_selection_sha256") != SUPPORT_SHA256
        or checked.get("risk_channel_exact_standardized_zero") is not True
    ):
        raise ValueError("V52 preflight binding differs")
    device = _device()
    prepared = load_cache(cache_path, cache_sha, commit)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    generator = np.random.default_rng(SEED)
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
    log_rows: list[dict[str, Any]] = []
    overflow_events: list[dict[str, Any]] = []
    validation_rows: dict[str, Any] = {}
    try:
        model.train()
        for step in range(1, STEPS + 1):
            conditions, targets = [], []
            for domain in DOMAIN_ORDER:
                row = v35["development_domains"][domain]
                index = int(generator.integers(int(row["train_objects"])))
                data, cache = handles[domain]
                condition, target, _ = no_risk_condition_cube(
                    data, cache, prepared, domain, "train", index
                )
                isometry = int(generator.integers(len(CUBE_ISOMETRIES)))
                axes, reflections = CUBE_ISOMETRIES[isometry]
                conditions.append(apply_cube_isometry(condition, axes, reflections))
                targets.append(apply_cube_isometry(target, axes, reflections))
            condition_tensor = torch.from_numpy(np.stack(conditions)).to(device)
            target_tensor = torch.from_numpy(np.stack(targets)).to(device)
            if torch.count_nonzero(condition_tensor[:, 5]).item() != 0:
                raise RuntimeError("V52 training risk channel differs")
            lr = _learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                parameters = model(condition_tensor)
            loss = -bounded_mixture_log_probability(parameters, target_tensor).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("V52 training loss is nonfinite")
            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gradient_value = float(gradient_norm.detach().cpu())
            gradient_is_finite = math.isfinite(gradient_value)
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            overflow = scale_after < scale_before
            if overflow:
                overflow_events.append(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "gradient_norm_is_finite": gradient_is_finite,
                        "scale_before": scale_before,
                        "scale_after": scale_after,
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
                    "loss": float(loss.detach().cpu()),
                    "learning_rate": lr,
                    "gradient_norm_before_clip": (
                        gradient_value if gradient_is_finite else "nonfinite"
                    ),
                    "gradient_norm_is_finite": gradient_is_finite,
                    "AMP_scale_before_update": scale_before,
                    "AMP_scale_after_update": scale_after,
                    "AMP_update_skipped": overflow,
                }
                log_rows.append(row)
                print("[v52-train] " + json.dumps(row), flush=True)
            if step in VALIDATION_STEPS:
                validation_rows[str(step)] = _validation_nll(
                    ema_model, v35, prepared, device
                )
                print(
                    f"[v52-validation] step={step} "
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
        "input_channels": INPUT_CHANNELS,
        "mixtures": 5,
        "likelihood_family": LIKELIHOOD_FAMILY,
        "open_standardized_support": [LOWER_SUPPORT, UPPER_SUPPORT],
        "support_selection_sha256": SUPPORT_SHA256,
        "risk_channel_exact_standardized_zero": True,
        "ema_decay": EMA_DECAY,
        "ema_state_dict": {
            key: value.detach().cpu()
            for key, value in ema_model.state_dict().items()
        },
        "conditioning_cache": str(cache_path.resolve()),
        "conditioning_cache_sha256": cache_sha,
        "preflight": str(preflight_path.resolve()),
        "preflight_sha256": preflight_sha,
        "AMP_overflow_events": overflow_events,
        "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection": False,
        "spatial_rank_transport": False,
        "sample_clipping": False,
        "component_scale_cap": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    partial_checkpoint = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".partial"
    )
    torch.save(checkpoint, partial_checkpoint)
    os.replace(partial_checkpoint, checkpoint_path)
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete_fixed_12000_step_matched_no_risk_train_only_fit",
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
        "risk_channel_exact_standardized_zero": True,
        "steps": STEPS,
        "parameters": PARAMETERS,
        "training_log": log_rows,
        "AMP_overflow_events": overflow_events,
        "AMP_overflow_count": len(overflow_events),
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
    partial_report.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    os.replace(partial_report, report_path)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
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
