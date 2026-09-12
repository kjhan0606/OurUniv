#!/usr/bin/env python
"""Run the single fixed V70 source-balanced latent spatial EDM fit."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import socket
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v48_train import condition_cube, load_cache
from hong2021_v63_preflight import _path
from hong2021_v63_train import _is_ancestor
from hong2021_v70_latent_cache import CACHE_SCHEMA, REPORT_SCHEMA as CACHE_REPORT_SCHEMA
from hong2021_v70_network import LatentSpatialUNet, edm_loss, parameter_count
from hong2021_v70_preflight import PROGRAM_FREEZE_COMMIT, PROGRAM_SHA256, load_program


CHECKPOINT_SCHEMA = "hong2021-v70-query-aligned-latent-spatial-checkpoint-v1"
REPORT_SCHEMA = "hong2021-v70-query-aligned-latent-spatial-training-report-v1"
CACHE_RESULT_RECORD = "config/hong2021_v70_latent_cache_result_record.json"
CACHE_RESULT_RECORD_SHA256 = (
    "3419206ce239546d7a2742ead01f20c9e6495c311dda0e4b82da6944a799ef76"
)
PREFLIGHT_SHA256 = "5b708473534954ff45f19ae0711249dd2d7305fa7288458467b71a78b853a3c4"
SEED = 170070
STEPS = 30_000
LOG_EVERY = 50
CHECKPOINT_EVERY = 250
LEARNING_RATE = 1.0e-4
MINIMUM_LEARNING_RATE = 1.0e-5
WEIGHT_DECAY = 1.0e-4
EMA_DECAY = 0.9995
SIGMA_DATA = 1.0
LOG_SIGMA_MEAN = -0.8
LOG_SIGMA_STD = 1.2
GRADIENT_CLIP = 1.0


def _json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def learning_rate(step: int) -> float:
    if not 1 <= step <= STEPS:
        raise ValueError("V70 training step differs")
    fraction = (step - 1) / (STEPS - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return MINIMUM_LEARNING_RATE + cosine * (
        LEARNING_RATE - MINIMUM_LEARNING_RATE
    )


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module) -> None:
    for ema, current in zip(
        ema_model.parameters(), model.parameters(), strict=True
    ):
        ema.lerp_(current.detach(), 1.0 - EMA_DECAY)
    for ema, current in zip(ema_model.buffers(), model.buffers(), strict=True):
        ema.copy_(current)


def atomic_torch_save(value: Any, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(value, partial)
    os.replace(partial, path)


def atomic_json(value: Any, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(partial, path)


def _load_authorization(
    program: dict[str, Any], repo: Path, commit: str, cache: Path, cache_sha: str,
    cache_report: Path, cache_report_sha: str,
) -> dict[str, Any]:
    record_path = repo / CACHE_RESULT_RECORD
    if sha256_file(record_path) != CACHE_RESULT_RECORD_SHA256:
        raise ValueError("V70 latent-cache result record hash differs")
    record = _json(record_path)
    if (
        record.get("status") != "complete_full_scan_pass_authorized_fixed_training"
        or record.get("program_sha256") != PROGRAM_SHA256
        or record.get("cache", {}).get("sha256") != cache_sha
        or record.get("cache", {}).get("report_sha256") != cache_report_sha
        or record.get("cache", {}).get("complete_scan_pass") is not True
        or record.get("cache", {}).get("fixed_training_authorized") is not True
        or record.get("firewall", {}).get("optimizer_constructed") is not False
        or record.get("firewall", {}).get("optimizer_step_performed") is not False
        or record.get("firewall", {}).get("validation_accessed") is not False
        or record.get("firewall", {}).get("development_accessed") is not False
        or record.get("firewall", {}).get("independent_gate_locked") is not True
        or not _is_ancestor(repo, str(record.get("code_commit")), commit)
    ):
        raise ValueError("V70 latent-cache authorization differs")
    if sha256_file(cache) != cache_sha or sha256_file(cache_report) != cache_report_sha:
        raise ValueError("V70 latent cache or report hash differs")
    report = _json(cache_report)
    if (
        report.get("schema") != CACHE_REPORT_SCHEMA
        or report.get("status") != "pass"
        or report.get("program_sha256") != PROGRAM_SHA256
        or report.get("cache_sha256") != cache_sha
        or report.get("complete_scan_pass") is not True
        or report.get("fixed_training_authorized") is not True
        or report.get("optimizer_constructed") is not False
        or report.get("optimizer_step_performed") is not False
        or report.get("validation_accessed") is not False
        or report.get("development_accessed") is not False
        or report.get("independent_gate_locked") is not True
        or canonical_digest(report) != report.get("decision_digest_sha256")
        or not _is_ancestor(repo, str(report.get("code_commit")), commit)
    ):
        raise ValueError("V70 latent-cache report authorization differs")
    with h5py.File(cache, "r") as handle:
        if (
            str(handle.attrs.get("schema")) != CACHE_SCHEMA
            or str(handle.attrs.get("program_sha256")) != PROGRAM_SHA256
            or str(handle.attrs.get("preflight_sha256"))
            != PREFLIGHT_SHA256
            or not bool(handle.attrs.get("complete", False))
            or bool(handle.attrs.get("validation_accessed", True))
            or bool(handle.attrs.get("development_accessed", True))
            or not bool(handle.attrs.get("independent_gate_locked", False))
        ):
            raise ValueError("V70 latent-cache metadata differs")
        for domain in DOMAIN_ORDER:
            expected = int(program["immutable_train_partition"]["fit_objects"][domain])
            fit = np.asarray(handle[f"{domain}/fit_indices"], dtype=np.int64)
            held = np.asarray(
                handle[f"{domain}/mechanism_holdout_indices"], dtype=np.int64
            )
            if (
                fit.size != expected
                or held.size != 16
                or np.intersect1d(fit, held).size
            ):
                raise ValueError("V70 cache train partition differs")
    return report


def _training_batch(
    handles: dict[str, tuple[h5py.File, h5py.File]],
    latent_cache: h5py.File,
    prepared: h5py.File,
    v35: dict[str, Any],
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    conditions: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    indices: dict[str, int] = {}
    isometries: dict[str, int] = {}
    for domain in DOMAIN_ORDER:
        fit = latent_cache[f"{domain}/fit_indices"]
        position = int(generator.integers(len(fit)))
        index = int(fit[position])
        transform = int(generator.integers(len(CUBE_ISOMETRIES)))
        data, cache = handles[domain]
        condition, _, _ = condition_cube(
            data, cache, prepared, domain, "train", index
        )
        latent = np.asarray(
            latent_cache[f"{domain}/latent"][index], dtype=np.float32
        )
        axes, reflections = CUBE_ISOMETRIES[transform]
        joined = apply_cube_isometry(
            np.concatenate((condition, latent), axis=0), axes, reflections
        )
        conditions.append(joined[:7])
        latents.append(joined[7:])
        indices[domain] = index
        isometries[domain] = transform
    condition_batch = np.ascontiguousarray(np.stack(conditions), dtype=np.float32)
    latent_batch = np.ascontiguousarray(np.stack(latents), dtype=np.float32)
    if not np.isfinite(condition_batch).all() or not np.isfinite(latent_batch).all():
        raise RuntimeError("V70 training batch is nonfinite")
    return condition_batch, latent_batch, indices, isometries


def _resume_state(
    path: Path,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    numpy_generator: np.random.Generator,
    noise_generator: torch.Generator,
    repo: Path,
    commit: str,
    training_source_sha: str,
    network_source_sha: str,
    cache_sha: str,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_commit = str(checkpoint.get("initial_code_commit"))
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("program_sha256") != PROGRAM_SHA256
        or checkpoint.get("training_source_sha256") != training_source_sha
        or checkpoint.get("network_source_sha256") != network_source_sha
        or checkpoint.get("cache_sha256") != cache_sha
        or checkpoint.get("steps") != STEPS
        or not 0 < int(checkpoint.get("step", 0)) < STEPS
        or not _is_ancestor(repo, source_commit, commit)
    ):
        raise ValueError("V70 resume checkpoint differs")
    model.load_state_dict(checkpoint["model_state_dict"])
    ema_model.load_state_dict(checkpoint["ema_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    numpy_generator.bit_generator.state = checkpoint["numpy_generator_state"]
    noise_generator.set_state(checkpoint["noise_generator_state"])
    torch.set_rng_state(checkpoint["torch_CPU_rng_state"])
    torch.cuda.set_rng_state_all(checkpoint["torch_CUDA_rng_state_all"])
    return (
        int(checkpoint["step"]),
        list(checkpoint["history"]),
        list(checkpoint["overflow_events"]),
        source_commit,
    )


def _checkpoint(
    step: int,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    numpy_generator: np.random.Generator,
    noise_generator: torch.Generator,
    history: list[dict[str, Any]],
    overflow_events: list[dict[str, Any]],
    initial_commit: str,
    training_source_sha: str,
    network_source_sha: str,
    cache_sha: str,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "program_sha256": PROGRAM_SHA256,
        "initial_code_commit": initial_commit,
        "training_source_sha256": training_source_sha,
        "network_source_sha256": network_source_sha,
        "cache_sha256": cache_sha,
        "steps": STEPS,
        "step": step,
        "parameters": parameter_count(model),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "ema_state_dict": {
            key: value.detach().cpu() for key, value in ema_model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "numpy_generator_state": numpy_generator.bit_generator.state,
        "noise_generator_state": noise_generator.get_state().cpu(),
        "torch_CPU_rng_state": torch.get_rng_state(),
        "torch_CUDA_rng_state_all": torch.cuda.get_rng_state_all(),
        "history": history,
        "overflow_events": overflow_events,
        "validation_accessed": False,
        "development_accessed": False,
        "independent_gate_locked": True,
    }


def train(
    program_path: Path,
    repo: Path,
    cache_path: Path,
    cache_sha: str,
    cache_report_path: Path,
    cache_report_sha: str,
    output: Path,
    resume: Path | None,
) -> dict[str, Any]:
    repo = repo.resolve()
    program, v35, _ = load_program(program_path, repo)
    commit, clean = git_state(repo)
    final_checkpoint = output / "step30000.pt"
    report_path = output / "training_report.json"
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
        or final_checkpoint.exists()
        or report_path.exists()
    ):
        raise RuntimeError("V70 training requires clean Lageunha and no final fit")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V70 training requires the Lageunha Ada GPU")
    _load_authorization(
        program, repo, commit, cache_path, cache_sha,
        cache_report_path, cache_report_sha,
    )
    if resume is None:
        if output.exists():
            raise FileExistsError("V70 refuses an existing initial training directory")
        output.mkdir(parents=True)
    elif not output.exists() or resume.resolve() != (output / "last.pt").resolve():
        raise ValueError("V70 resume must use the bound last checkpoint")
    training_source_sha = sha256_file(repo / "src/hong2021_v70_train.py")
    network_source_sha = sha256_file(repo / "src/hong2021_v70_network.py")
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = LatentSpatialUNet().to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.amp.GradScaler("cuda")
    numpy_generator = np.random.default_rng(SEED)
    noise_generator = torch.Generator(device=device).manual_seed(SEED + 1)
    history: list[dict[str, Any]] = []
    overflow_events: list[dict[str, Any]] = []
    initial_commit = commit
    start_step = 0
    if resume is not None:
        start_step, history, overflow_events, initial_commit = _resume_state(
            resume, model, ema_model, optimizer, scaler, numpy_generator,
            noise_generator, repo, commit, training_source_sha,
            network_source_sha, cache_sha,
        )
    run = {
        "schema": REPORT_SCHEMA,
        "status": "training",
        "program_sha256": PROGRAM_SHA256,
        "initial_code_commit": initial_commit,
        "current_code_commit": commit,
        "training_source_sha256": training_source_sha,
        "network_source_sha256": network_source_sha,
        "cache_sha256": cache_sha,
        "cache_report_sha256": cache_report_sha,
        "parameters": parameter_count(model),
        "steps": STEPS,
        "start_step": start_step,
        "source_balance_per_step": {domain: 1 for domain in DOMAIN_ORDER},
        "fit_objects": program["immutable_train_partition"]["fit_objects"],
        "mechanism_holdout_objects_per_domain": 16,
        "seed": SEED,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "minimum_learning_rate": MINIMUM_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "EMA_decay": EMA_DECAY,
        "sigma_data": SIGMA_DATA,
        "log_sigma_mean": LOG_SIGMA_MEAN,
        "log_sigma_standard_deviation": LOG_SIGMA_STD,
        "augmentation": "independent uniform frozen 48 signed cube isometries",
        "checkpoint_selection": "fixed step-30000 EMA only",
        "validation_used_for_stopping_or_selection": False,
        "pair_or_2PCF_loss": False,
        "Fourier_or_Ak_loss": False,
        "device": str(device),
    }
    atomic_json(run, output / "run.json")
    prepared = load_cache(
        _path(repo, program["frozen_inputs"]["conditioning_cache"]),
        program["frozen_inputs"]["conditioning_cache_sha256"],
        commit,
    )
    latent_cache = h5py.File(cache_path, "r")
    handles = {
        domain: _open_split(v35["development_domains"][domain], "train")
        for domain in DOMAIN_ORDER
    }
    interval_total = 0.0
    interval_domain = np.zeros(len(DOMAIN_ORDER), dtype=np.float64)
    interval_steps = 0
    start_time = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model.train()
        for step in range(start_step + 1, STEPS + 1):
            condition_numpy, latent_numpy, indices, isometries = _training_batch(
                handles, latent_cache, prepared, v35, numpy_generator
            )
            condition = torch.from_numpy(condition_numpy).to(device)
            latent = torch.from_numpy(latent_numpy).to(device)
            sigma = torch.exp(
                torch.randn(
                    len(DOMAIN_ORDER), device=device, generator=noise_generator
                )
                * LOG_SIGMA_STD
                + LOG_SIGMA_MEAN
            )
            noise = torch.randn(
                latent.shape, device=device, generator=noise_generator
            )
            current_lr = learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss, per_object = edm_loss(
                    model, latent, condition, sigma, noise, SIGMA_DATA
                )
            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            gradient_finite = bool(torch.isfinite(gradient_norm).cpu())
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
                        "gradient_finite": gradient_finite,
                    }
                )
            else:
                update_ema(ema_model, model)
            weight = (sigma.square() + SIGMA_DATA**2) / (
                sigma * SIGMA_DATA
            ).square()
            weighted = (weight.float() * per_object.detach()).cpu().numpy()
            interval_total += float(loss.detach().cpu())
            interval_domain += weighted
            interval_steps += 1
            if step == 1 or step % LOG_EVERY == 0:
                row = {
                    "step": step,
                    "source_balanced_EDM_loss": interval_total / interval_steps,
                    "domain_EDM_loss": {
                        domain: float(interval_domain[index] / interval_steps)
                        for index, domain in enumerate(DOMAIN_ORDER)
                    },
                    "learning_rate": current_lr,
                    "gradient_norm_before_clip": (
                        float(gradient_norm.detach().cpu())
                        if gradient_finite
                        else "nonfinite"
                    ),
                    "AMP_scale_before_update": scale_before,
                    "AMP_scale_after_update": scale_after,
                    "AMP_update_skipped": overflow,
                    "sample_indices": indices,
                    "signed_cube_isometries": isometries,
                    "peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(device)
                    ),
                    "elapsed_seconds_this_process": time.time() - start_time,
                }
                history.append(row)
                atomic_json(history, output / "history.json")
                print("[v70-train] " + json.dumps(row), flush=True)
                interval_total = 0.0
                interval_domain.fill(0.0)
                interval_steps = 0
            if step % CHECKPOINT_EVERY == 0 and step < STEPS:
                atomic_torch_save(
                    _checkpoint(
                        step, model, ema_model, optimizer, scaler,
                        numpy_generator, noise_generator, history,
                        overflow_events, initial_commit, training_source_sha,
                        network_source_sha, cache_sha,
                    ),
                    output / "last.pt",
                )
    finally:
        for data, cache in handles.values():
            data.close()
            cache.close()
        latent_cache.close()
        prepared.close()
    peak = int(torch.cuda.max_memory_allocated(device))
    sealed = {
        "schema": CHECKPOINT_SCHEMA,
        "program_sha256": PROGRAM_SHA256,
        "initial_code_commit": initial_commit,
        "completion_code_commit": commit,
        "training_source_sha256": training_source_sha,
        "network_source_sha256": network_source_sha,
        "cache_sha256": cache_sha,
        "step": STEPS,
        "steps": STEPS,
        "parameters": parameter_count(model),
        "ema_decay": EMA_DECAY,
        "ema_state_dict": {
            key: value.detach().cpu() for key, value in ema_model.state_dict().items()
        },
        "overflow_events": overflow_events,
        "validation_accessed": False,
        "development_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    atomic_torch_save(sealed, final_checkpoint)
    result: dict[str, Any] = {
        **run,
        "status": "complete_fixed_30000_step_fit",
        "completion_code_commit": commit,
        "checkpoint": str(final_checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(final_checkpoint),
        "history": history,
        "overflow_events": overflow_events,
        "peak_allocated_bytes": peak,
        "training_complete": True,
        "train_only_mechanism_gate_run": False,
        "validation_accessed": False,
        "development_accessed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    atomic_json(result, report_path)
    run["status"] = "complete_fixed_30000_step_fit"
    run["checkpoint_sha256"] = result["checkpoint_sha256"]
    atomic_json(run, output / "run.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--cache-report-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    result = train(
        args.program,
        args.repo,
        args.cache,
        args.cache_sha256,
        args.cache_report,
        args.cache_report_sha256,
        args.out,
        args.resume,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
