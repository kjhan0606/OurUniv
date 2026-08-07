#!/usr/bin/env python
"""Train and directly sample the frozen V26 conditional Haar spline flow."""
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

from hong2021_residual_v12_gaussianized import inverse_gaussianize_torch
from hong2021_residual_v6 import atomic_save, seed_everything, update_ema
from hong2021_residual_v8_context import cycling, make_loader
from hong2021_v14_edm import (
    ENSEMBLE_SCHEMA,
    V14ResidualDataset,
    source_balanced_feature_standardization,
)
from hong2021_v14_mean_correction import DOMAINS
from hong2021_v14_multiscale import inverse_standardized_residual
from hong2021_v15_development_gate import canonical_digest
from hong2021_v15_edm import git_state
from hong2021_v18_edm import _indices
from hong2021_v18_init import sha256_file
from hong2021_v21_conditional_affine import invert_profile_torch
from hong2021_v21_edm import ARTIFACT_SHA256
from hong2021_v25_edm import load_frozen_program as load_v25_program
from hong2021_v26_flow import ConditionalHaarSplineFlow


REGISTRY_SCHEMA = "hong2021-v26-conditional-haar-spline-flow-development-program-v1"
REGISTRY_SHA256 = "810e7ce430c0b9a1a7322bad3ef0ea650af80165cd26a16fc84e8d9b9e96ac2c"
DESIGN_AUDIT_SHA256 = "4f134592209ea2f142b726a0bb296279cba6a2d204323f4b61e42c514ff33b2e"
HAAR_ARTIFACT_SHA256 = "32975f030ee014e15edb749dec8b26a0885e71ebab491f45f5d51e0f38782a24"
MODEL_SCHEMA = "hong2021-v26-conditional-haar-spline-flow-v1"
PARAMETERS = 3_206_424
NON_DC_DIMENSIONS = 262_143
CANDIDATE_STEPS = (10_000, 20_000, 30_000)
DOMAIN_KEYS = {"TNG100": "tng", "SIMBA": "simba_dev", "Swift": "swift_dev"}
CACHE_KEYS = {
    "TNG100": {"train": "TNG100_train", "validation": "TNG100_validation"},
    "SIMBA": {"train": "SIMBA_train", "validation": "SIMBA_validation"},
    "Swift": {"train": "Swift_train", "validation": "Swift_validation"},
}
DETAIL_DIMENSIONS_COARSE_TO_FINE = (7, 56, 448, 3584, 28672, 229376)


def _resolve(value: str, repo: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_frozen_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = path.resolve()
    if sha256_file(path) != REGISTRY_SHA256:
        raise ValueError("V26 registry differs from its frozen hash")
    registry = json.loads(path.read_text())
    if (
        registry.get("schema") != REGISTRY_SCHEMA
        or registry.get("status")
        != "frozen_before_candidate_training_or_development_evaluation"
    ):
        raise ValueError("V26 registry schema or status mismatch")
    design = registry["design_audit"]
    design_path = _resolve(design["path"], repo)
    if (
        design.get("sha256") != DESIGN_AUDIT_SHA256
        or sha256_file(design_path) != DESIGN_AUDIT_SHA256
    ):
        raise ValueError("V26 design audit hash mismatch")
    design_payload = json.loads(design_path.read_text())
    if (
        design_payload.get("selected_likelihood", {}).get("name")
        != "coarse_to_fine_conditional_haar_rational_quadratic_spline_flow"
        or design_payload.get("firewall", {}).get("Astrid_accessed") is not False
        or design_payload.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V26 design audit selection or firewall mismatch")
    parent = registry["parent_evidence"]
    v25_path = _resolve(parent["v25_registry"], repo)
    if sha256_file(v25_path) != parent["v25_registry_sha256"]:
        raise ValueError("V26 V25-parent registry hash mismatch")
    _, artifacts, v20, _ = load_v25_program(v25_path, repo)
    decision_path = Path(parent["v25_decision"])
    decision = json.loads(decision_path.read_text())
    if (
        sha256_file(decision_path) != parent["v25_decision_sha256"]
        or canonical_digest(decision) != parent["v25_decision_digest_sha256"]
        or decision.get("development_pass") is not False
        or decision.get("next") != parent["required_next"]
    ):
        raise ValueError("V26 V25-parent decision mismatch")
    coordinate = registry["coordinate_system"]
    haar_path = Path(coordinate["standardization_artifact"])
    if (
        coordinate["standardization_artifact_sha256"] != HAAR_ARTIFACT_SHA256
        or sha256_file(haar_path) != HAAR_ARTIFACT_SHA256
    ):
        raise ValueError("V26 Haar standardization artifact hash mismatch")
    haar = json.loads(haar_path.read_text())
    if (
        haar.get("non_dc_dimensions") != NON_DC_DIMENSIONS
        or haar.get("Astrid_accessed") is not False
        or haar.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V26 Haar artifact dimension or firewall mismatch")
    likelihood = registry["likelihood"]
    exact = {
        "levels": 6,
        "detail_channels_per_level": 7,
        "coupling_layers_per_level": 4,
        "bins": 8,
        "tail_bound_standardized_units": 6.0,
        "parameters": PARAMETERS,
        "target_or_density_dependent_weights": False,
        "auxiliary_field_or_tail_losses": False,
    }
    for key, value in exact.items():
        if likelihood.get(key) != value:
            raise ValueError(f"V26 frozen likelihood differs: {key}")
    training = registry["training_protocol"]
    if (
        training.get("batch") != 6
        or training.get("steps") != 30_000
        or training.get("candidate_steps") != list(CANDIDATE_STEPS)
        or training.get("source_balance_per_batch")
        != {"TNG100": 2, "SIMBA": 2, "Swift": 2}
    ):
        raise ValueError("V26 training protocol differs from its freeze")
    if sha256_file(repo / "config/hong2021_v21_derived_artifacts.json") != ARTIFACT_SHA256:
        raise ValueError("V26 inherited V21 artifact attestation differs")
    return registry, artifacts, v20, decision, haar


def _paths(
    artifacts: dict[str, Any], v20: dict[str, Any]
) -> dict[str, tuple[str, str, str, str]]:
    data = v20["e8_gaussianized_marginal_retrain"]["data"]
    return {
        model_domain: (
            data[("Swift" if model_domain == "Swift-EAGLE" else model_domain)]["train_data"]["path"],
            artifacts["caches"][CACHE_KEYS[("Swift" if model_domain == "Swift-EAGLE" else model_domain)]["train"]]["path"],
            data[("Swift" if model_domain == "Swift-EAGLE" else model_domain)]["validation_data"]["path"],
            artifacts["caches"][CACHE_KEYS[("Swift" if model_domain == "Swift-EAGLE" else model_domain)]["validation"]]["path"],
        )
        for model_domain in DOMAINS
    }


def build_model(
    haar: dict[str, Any], feature_fit: dict[str, Any], *, device: torch.device
) -> ConditionalHaarSplineFlow:
    model = ConditionalHaarSplineFlow(
        detail_mean=haar["source_balanced_mean"],
        detail_std=haar["source_balanced_standard_deviation"],
        context_mean=feature_fit["mean"],
        context_std=feature_fit["std"],
        condition_channels=4,
        hidden_channels=32,
        levels=6,
        couplings=4,
        bins=8,
        tail_bound=6.0,
    ).to(device)
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != PARAMETERS:
        raise RuntimeError(f"V26 parameter count changed: {parameters}")
    return model


@torch.inference_mode()
def fixed_validation(
    model: ConditionalHaarSplineFlow,
    loader: Any,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total = 0.0
    scale_total = np.zeros(6, dtype=np.float64)
    maximum_dc = 0.0
    samples = 0
    for condition, residual, _, _ in loader:
        condition = condition.to(device, non_blocking=True)
        residual = residual.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            log_prob, diagnostic = model.log_prob(residual, condition)
        batch = len(residual)
        total += float((-log_prob / NON_DC_DIMENSIONS).sum())
        scale = -diagnostic["scale_log_prob_coarse_to_fine"].double()
        scale /= torch.as_tensor(
            DETAIL_DIMENSIONS_COARSE_TO_FINE,
            device=scale.device,
            dtype=scale.dtype,
        )[None]
        scale_total += scale.sum(dim=0).cpu().numpy()
        maximum_dc = max(
            maximum_dc, float(diagnostic["coarsest_dc"].abs().max())
        )
        samples += batch
    if not samples:
        raise ValueError("V26 validation loader is empty")
    return {
        "nll_per_non_dc_dimension": total / samples,
        "scale_nll_coarse_to_fine": (scale_total / samples).tolist(),
        "maximum_absolute_float32_haar_dc": maximum_dc,
        "objects": samples,
    }


def train(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    if socket.gethostname().lower() != "lageunha":
        raise RuntimeError("V26 training requires Lageunha")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V26 training requires the Lageunha Ada CUDA device")
    gpu = torch.cuda.get_device_name(0)
    if "ada" not in gpu.lower():
        raise RuntimeError(f"V26 training requires an Ada GPU, found {gpu}")
    registry, artifacts, v20, _, haar = load_frozen_program(
        args.registry.resolve(), repo
    )
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V26 training requires a clean committed worktree")
    preflight_path = args.preflight.resolve()
    if not preflight_path.is_file():
        raise RuntimeError("V26 hard preflight is absent")
    preflight = json.loads(preflight_path.read_text())
    expected_preflight = {
        "schema": "hong2021-v26-hard-preflight-v1",
        "status": "pass",
        "registry_sha256": REGISTRY_SHA256,
        "code_commit": commit,
        "host": socket.gethostname(),
        "gpu": gpu,
        "parameters": PARAMETERS,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    for key, value in expected_preflight.items():
        if preflight.get(key) != value:
            raise RuntimeError(f"V26 hard preflight mismatch: {key}")
    output = args.out.resolve()
    if output.exists():
        raise RuntimeError(f"V26 refuses a pre-existing training output: {output}")
    seed_everything(144021)
    device = torch.device(args.device)
    paths = _paths(artifacts, v20)
    feature_datasets = {
        domain: V14ResidualDataset(row[0], row[1], False)
        for domain, row in paths.items()
    }
    feature_fit = source_balanced_feature_standardization(feature_datasets)
    train_datasets = {
        domain: V14ResidualDataset(row[0], row[1], True)
        for domain, row in paths.items()
    }
    validation_datasets = {
        domain: V14ResidualDataset(row[2], row[3], False)
        for domain, row in paths.items()
    }
    train_loaders = {
        domain: make_loader(
            dataset, 2, 1, True, 144021 + index, device
        )
        for index, (domain, dataset) in enumerate(train_datasets.items())
    }
    validation_loaders = {
        domain: make_loader(
            dataset, 6, 1, False, 144031 + index, device
        )
        for index, (domain, dataset) in enumerate(validation_datasets.items())
    }
    model = build_model(haar, feature_fit, device=device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    protocol = registry["training_protocol"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(protocol["learning_rate"]),
        weight_decay=float(protocol["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(protocol["steps"]),
        eta_min=float(protocol["minimum_learning_rate"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    iterators = {domain: cycling(loader) for domain, loader in train_loaders.items()}
    output.mkdir(parents=True)
    checkpoints = output / "validation_checkpoints"
    checkpoints.mkdir()
    metadata = {
        "schema": MODEL_SCHEMA,
        "status": "training",
        "experiment_registry": str(args.registry.resolve()),
        "experiment_registry_sha256": REGISTRY_SHA256,
        "design_audit_sha256": DESIGN_AUDIT_SHA256,
        "v21_artifacts_sha256": ARTIFACT_SHA256,
        "haar_artifact": registry["coordinate_system"]["standardization_artifact"],
        "haar_artifact_sha256": HAAR_ARTIFACT_SHA256,
        "data": {
            domain: {
                "train": row[0], "train_cache": row[1],
                "validation": row[2], "validation_cache": row[3],
            }
            for domain, row in paths.items()
        },
        "observable_context_features": feature_fit,
        "model": registry["likelihood"],
        "parameters": PARAMETERS,
        "non_dc_dimensions": NON_DC_DIMENSIONS,
        "steps": int(protocol["steps"]),
        "candidate_steps": protocol["candidate_steps"],
        "batch": int(protocol["batch"]),
        "source_balance_per_batch": protocol["source_balance_per_batch"],
        "optimizer": "AdamW",
        "learning_rate": float(protocol["learning_rate"]),
        "minimum_learning_rate": float(protocol["minimum_learning_rate"]),
        "weight_decay": float(protocol["weight_decay"]),
        "gradient_clip": float(protocol["gradient_clip"]),
        "ema_decay": float(protocol["ema_decay"]),
        "validation_every": int(protocol["validation_every"]),
        "seed": int(protocol["seed"]),
        "augmentation": protocol["augmentation"],
        "objective": registry["likelihood"]["objective"],
        "target_or_density_dependent_weights": False,
        "nondevelopment_data_used": False,
        "hard_preflight": str(preflight_path),
        "hard_preflight_sha256": sha256_file(preflight_path),
        "execution_host": socket.gethostname(),
        "execution_gpu": gpu,
        "code_commit_at_launch": commit,
        "worktree_clean_at_launch": clean,
        "Astrid_used": False,
        "EAGLE_RefL0100N1504_used": False,
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    history = []
    interval_loss = 0.0
    interval_scale = np.zeros(6, dtype=np.float64)
    interval_samples = 0
    gradient_sum = 0.0
    gradient_activations = 0
    updates = 0
    started = time.time()
    model.train()
    for step in range(1, int(protocol["steps"]) + 1):
        batches = [next(iterators[domain]) for domain in DOMAINS]
        condition = torch.cat([row[0] for row in batches]).to(device, non_blocking=True)
        residual = torch.cat([row[1] for row in batches]).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=True):
            log_prob, diagnostic = model.log_prob(residual, condition)
            loss = -log_prob.mean() / NON_DC_DIMENSIONS
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient = float(
            nn.utils.clip_grad_norm_(model.parameters(), float(protocol["gradient_clip"]))
        )
        gradient_sum += gradient
        gradient_activations += int(gradient > float(protocol["gradient_clip"]))
        updates += 1
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        update_ema(ema_model, model, float(protocol["ema_decay"]))
        batch = len(residual)
        interval_loss += float(loss.detach()) * batch
        scale_nll = -diagnostic["scale_log_prob_coarse_to_fine"].detach().double()
        scale_nll /= torch.as_tensor(
            DETAIL_DIMENSIONS_COARSE_TO_FINE,
            device=device,
            dtype=torch.float64,
        )[None]
        interval_scale += scale_nll.sum(dim=0).cpu().numpy()
        interval_samples += batch
        if step % int(protocol["validation_every"]) == 0:
            validation = {
                domain: fixed_validation(ema_model, loader, device)
                for domain, loader in validation_loaders.items()
            }
            row = {
                "step": step,
                "train_nll_per_non_dc_dimension": interval_loss / interval_samples,
                "train_scale_nll_coarse_to_fine": (
                    interval_scale / interval_samples
                ).tolist(),
                "fixed_validation": validation,
                "balanced_validation_nll": float(np.mean([
                    value["nll_per_non_dc_dimension"]
                    for value in validation.values()
                ])),
                "worst_validation_nll": float(max(
                    value["nll_per_non_dc_dimension"]
                    for value in validation.values()
                )),
                "gradient_diagnostic": {
                    "mean_norm_before_fixed_clip": gradient_sum / updates,
                    "fixed_clip_activation_fraction": gradient_activations / updates,
                    "fixed_clip_threshold": float(protocol["gradient_clip"]),
                    "selection_role": "none",
                },
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - started,
            }
            history.append(row)
            (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            checkpoint = {**metadata, **row, "ema_model": ema_model.state_dict()}
            atomic_save(checkpoint, checkpoints / f"step_{step:06d}.pt")
            atomic_save(
                {
                    **checkpoint,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                output / "last.pt",
            )
            print(
                f"step={step:06d} train_nll={row['train_nll_per_non_dc_dimension']:.7f} "
                + " ".join(
                    f"{domain}={value['nll_per_non_dc_dimension']:.7f}"
                    for domain, value in validation.items()
                )
                + f" elapsed={row['elapsed_seconds']:.0f}s",
                flush=True,
            )
            interval_loss = 0.0
            interval_scale[:] = 0.0
            interval_samples = 0
            gradient_sum = 0.0
            gradient_activations = 0
            updates = 0
            model.train()
    metadata["status"] = "complete"
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _validate_checkpoint(
    path: Path,
    *,
    step: int,
    artifacts: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    digest = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("schema") != MODEL_SCHEMA
        or int(checkpoint.get("step", -1)) != step
        or checkpoint.get("experiment_registry_sha256") != REGISTRY_SHA256
        or checkpoint.get("design_audit_sha256") != DESIGN_AUDIT_SHA256
        or checkpoint.get("v21_artifacts_sha256") != ARTIFACT_SHA256
        or checkpoint.get("haar_artifact_sha256") != HAAR_ARTIFACT_SHA256
        or checkpoint.get("parameters") != PARAMETERS
        or checkpoint.get("non_dc_dimensions") != NON_DC_DIMENSIONS
        or checkpoint.get("steps") != 30_000
        or checkpoint.get("candidate_steps") != list(CANDIDATE_STEPS)
        or checkpoint.get("batch") != 6
        or checkpoint.get("target_or_density_dependent_weights") is not False
        or checkpoint.get("nondevelopment_data_used") is not False
        or checkpoint.get("worktree_clean_at_launch") is not True
    ):
        raise ValueError("V26 checkpoint protocol or provenance mismatch")
    preflight_path = Path(str(checkpoint.get("hard_preflight", "")))
    if (
        not preflight_path.is_file()
        or sha256_file(preflight_path) != checkpoint.get("hard_preflight_sha256")
    ):
        raise ValueError("V26 checkpoint preflight seal mismatch")
    return checkpoint, digest


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    registry, artifacts, v20, _, haar = load_frozen_program(
        args.registry.resolve(), repo
    )
    commit, clean = git_state(repo)
    if not clean:
        raise RuntimeError("V26 sampling requires a clean committed worktree")
    step = int(args.step)
    if step not in CANDIDATE_STEPS:
        raise ValueError("V26 sampling step is not preregistered")
    checkpoint_path = (
        args.training_root.resolve()
        / "validation_checkpoints"
        / f"step_{step:06d}.pt"
    )
    checkpoint, checkpoint_sha = _validate_checkpoint(
        checkpoint_path, step=step, artifacts=artifacts
    )
    domain = args.domain
    experiment = v20["e8_gaussianized_marginal_retrain"]
    data = experiment["data"][domain]["validation_data"]
    cache = artifacts["caches"][CACHE_KEYS[domain]["validation"]]
    indices = _indices(experiment["development_objects"][domain], repo)
    seed = int(registry["training_protocol"]["sampling_seeds"][domain])
    seed_everything(seed)
    device = torch.device(args.device)
    model = build_model(
        haar, checkpoint["observable_context_features"], device=device
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval()
    dataset = V14ResidualDataset(data["path"], cache["path"], False)
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    transform = json.loads(Path(artifacts["gaussianization"]["path"]).read_text())
    centers = torch.as_tensor(profile["centers"], dtype=torch.float64, device=device)
    mu = torch.as_tensor(profile["mu"], dtype=torch.float64, device=device)
    log_sigma = torch.as_tensor(
        profile["log_sigma"], dtype=torch.float64, device=device
    )
    z_knots = torch.as_tensor(
        transform["z_knots"], dtype=torch.float32, device=device
    )
    residual_knots = torch.as_tensor(
        transform["residual_value_knots"], dtype=torch.float32, device=device
    )
    output = args.out.resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError(f"V26 refuses to overwrite ensemble: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(seed)
    maximum_pre_center_dc = 0.0
    maximum_post_center_dc = 0.0
    try:
        with h5py.File(partial, "w") as handle:
            sample_ds = handle.create_dataset(
                "sample",
                shape=(16, 16, 1, 64, 64, 64),
                dtype="f4",
                chunks=(1, 1, 1, 64, 64, 64),
                compression="lzf",
            )
            mean_ds = handle.create_dataset(
                "conditional_mean", shape=(16, 1, 64, 64, 64),
                dtype="f4", compression="lzf",
            )
            truth_ds = handle.create_dataset(
                "truth", shape=(16, 1, 64, 64, 64),
                dtype="f4", compression="lzf",
            )
            handle.create_dataset("source_index", data=np.asarray(indices, dtype=np.int64))
            location_ds = handle.create_dataset("predicted_residual_dc", shape=(16,), dtype="f4")
            scale_ds = handle.create_dataset("predicted_band_scales", shape=(16, 4), dtype="f4")
            for output_index, data_index in enumerate(indices):
                condition, _, corrected_mean, truth = dataset[data_index]
                location, scales = dataset.predicted_location_scales(data_index)
                condition_batch = condition[None].to(device).expand(16, -1, -1, -1, -1)
                latent, dc = model.sample_with_diagnostics(
                    condition_batch, generator=generator
                )
                maximum_pre_center_dc = max(
                    maximum_pre_center_dc, float(dc["pre_center_mean"].abs().max())
                )
                maximum_post_center_dc = max(
                    maximum_post_center_dc, float(dc["post_center_mean"].abs().max())
                )
                u = inverse_gaussianize_torch(latent, z_knots, residual_knots)
                mean_batch = corrected_mean[None].to(device).expand(16, -1, -1, -1, -1)
                standardized = invert_profile_torch(
                    u, mean_batch, centers, mu, log_sigma
                )
                values = standardized[:, 0].float().cpu().numpy()
                physical = np.stack([
                    inverse_standardized_residual(
                        value,
                        predicted_location=location,
                        predicted_scales=scales,
                        voxel_mpc_h=dataset.voxel_mpc_h,
                    )
                    for value in values
                ]).astype(np.float32)
                sample_ds[output_index, :, 0] = corrected_mean.numpy()[0] + physical
                mean_ds[output_index] = corrected_mean.numpy() + np.float32(location)
                truth_ds[output_index] = truth.numpy()
                location_ds[output_index] = location
                scale_ds[output_index] = scales
                print(f"[sample] V26 {domain} {output_index + 1}/16", flush=True)
            handle.attrs.update({
                "schema": ENSEMBLE_SCHEMA,
                "method": "conditional_haar_spline_flow",
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_step": step,
                "checkpoint_schema": MODEL_SCHEMA,
                "source_cache": str(Path(cache["path"]).resolve()),
                "source_cache_sha256": cache["sha256"],
                "source_data_sha256": data["sha256"],
                "v26_registry_sha256": REGISTRY_SHA256,
                "v21_artifact_attestation_sha256": ARTIFACT_SHA256,
                "v21_profile_sha256": artifacts["profile"]["sha256"],
                "v21_gaussianization_sha256": artifacts["gaussianization"]["sha256"],
                "haar_artifact_sha256": HAAR_ARTIFACT_SHA256,
                "ensemble_members": 16,
                "seed": seed,
                "diagnostic_k_h_mpc": 1.0,
                "location_scale_uses_target": False,
                "direct_sampling": True,
                "modeled_non_dc_dimensions": NON_DC_DIMENSIONS,
                "maximum_pre_center_latent_dc": maximum_pre_center_dc,
                "maximum_post_center_latent_dc": maximum_post_center_dc,
                "sampling_code_commit": commit,
                "worktree_clean_at_sampling": clean,
                "Astrid_accessed": False,
                "historical_EAGLE_accessed": False,
                "complete": True,
            })
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    print(f"[out] {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    training = sub.add_parser("train")
    sampling = sub.add_parser("sample")
    for item in (training, sampling):
        item.add_argument("--registry", type=Path, required=True)
        item.add_argument("--repo", type=Path, required=True)
        item.add_argument("--device", default="cuda")
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--preflight", type=Path, required=True)
    sampling.add_argument("--training-root", type=Path, required=True)
    sampling.add_argument("--domain", choices=tuple(DOMAIN_KEYS), required=True)
    sampling.add_argument("--step", type=int, choices=CANDIDATE_STEPS, required=True)
    sampling.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args) if args.mode == "train" else sample(args)


if __name__ == "__main__":
    main()
