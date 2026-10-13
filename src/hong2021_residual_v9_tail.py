#!/usr/bin/env python
"""Train-only density-tail-balanced continuation of the V8 context EDM.

Uniform voxel denoising reproduces variance but underweights the rare cells
that define extreme peaks and voids.  V9 derives five density-bin weights from
TNG and SIMBA development *training* truths only.  The source-balanced bin
probability receives an inverse-square-root weight, capped at ten and mixed
50:50 with the original unweighted EDM objective.  No validation, historical
SIMBA CV0-15, or EAGLE truth enters the weights or optimizer.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Subset

from hong2021_residual_v6 import (
    V6ResidualDataset,
    atomic_save,
    cache_scale,
    edm_denoise,
    fixed_validation_loss,
    sample_edm,
    seed_everything,
    update_ema,
)
from hong2021_residual_v8_context import (
    ObservableContextUNet,
    cycling,
    make_loader,
    validation_pair,
)


SCHEMA = "hong2021-conditional-laplacian-residual-v9-tail-balanced"
DENSITY_SCALE = 4.5
DENSITY_LOG10_BOUNDARIES = (-1.0, 0.0, 1.0, 2.0)
DENSITY_BIN_LABELS = (
    "rho_lt_0.1",
    "rho_0.1_to_1",
    "rho_1_to_10",
    "rho_10_to_100",
    "rho_ge_100",
)


def density_bin_counts(path: str | Path, chunk: int = 4) -> np.ndarray:
    counts = np.zeros(len(DENSITY_BIN_LABELS), dtype=np.int64)
    with h5py.File(path, "r") as handle:
        target = handle["target"]
        for start in range(0, len(target), chunk):
            value = DENSITY_SCALE * np.asarray(
                target[start : start + chunk], dtype=np.float32
            )
            index = np.digitize(value, DENSITY_LOG10_BOUNDARIES)
            counts += np.bincount(index.ravel(), minlength=len(counts))
    if np.any(counts == 0):
        raise ValueError(f"empty density bin in training data: {counts.tolist()}")
    return counts


def balanced_tail_weights(
    tng_counts: np.ndarray,
    simba_counts: np.ndarray,
    exponent: float = 0.5,
    maximum: float = 10.0,
) -> dict[str, Any]:
    tng = np.asarray(tng_counts, dtype=np.float64)
    simba = np.asarray(simba_counts, dtype=np.float64)
    if tng.shape != (5,) or simba.shape != (5,) or np.any(tng <= 0) or np.any(simba <= 0):
        raise ValueError("density counts must be positive five-vectors")
    probability_tng = tng / tng.sum()
    probability_simba = simba / simba.sum()
    probability = 0.5 * (probability_tng + probability_simba)
    raw = np.power(probability, -exponent)
    raw /= np.sum(probability * raw)
    clipped = np.minimum(raw, maximum)
    weights = clipped / np.sum(probability * clipped)
    return {
        "bin_labels": list(DENSITY_BIN_LABELS),
        "log10_density_boundaries": list(DENSITY_LOG10_BOUNDARIES),
        "counts": {"tng": tng_counts.tolist(), "simba": simba_counts.tolist()},
        "probability": {
            "tng": probability_tng.tolist(),
            "simba": probability_simba.tolist(),
            "equal_source": probability.tolist(),
        },
        "inverse_probability_exponent": exponent,
        "pre_normalization_maximum": maximum,
        "weights": weights.astype(np.float32).tolist(),
        "weight_expectation_equal_source": float(np.sum(probability * weights)),
        "unweighted_objective_fraction": 0.5,
        "tail_balanced_objective_fraction": 0.5,
    }


def voxel_tail_weights(
    truth: torch.Tensor, bin_weights: torch.Tensor
) -> torch.Tensor:
    log_density = DENSITY_SCALE * truth
    boundaries = torch.tensor(
        DENSITY_LOG10_BOUNDARIES,
        device=truth.device,
        dtype=truth.dtype,
    )
    index = torch.bucketize(log_density, boundaries, right=True)
    return bin_weights[index]


def tail_balanced_edm_loss(
    model: nn.Module,
    residual: torch.Tensor,
    condition: torch.Tensor,
    truth: torch.Tensor,
    bin_weights: torch.Tensor,
    generator: torch.Generator,
    sigma_data: float,
    edm_p_mean: float,
    edm_p_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = len(residual)
    sigma = torch.exp(
        torch.randn(batch, device=residual.device, generator=generator)
        * edm_p_std
        + edm_p_mean
    )
    noise = torch.randn(
        residual.shape, device=residual.device, generator=generator
    )
    noisy = residual + sigma[:, None, None, None, None] * noise
    denoised = edm_denoise(model, noisy, condition, sigma, sigma_data)
    edm_weight = (sigma.square() + sigma_data**2) / (
        sigma * sigma_data
    ).square()
    error2 = (denoised - residual).square()
    unweighted = (edm_weight * error2.mean(dim=(1, 2, 3, 4))).mean()
    tail_weight = voxel_tail_weights(truth, bin_weights)
    weighted_per_sample = (error2 * tail_weight).sum(dim=(1, 2, 3, 4)) / (
        tail_weight.sum(dim=(1, 2, 3, 4)).clamp_min(1.0)
    )
    weighted = (edm_weight * weighted_per_sample).mean()
    return 0.5 * (unweighted + weighted), unweighted, weighted


@torch.inference_mode()
def fixed_tail_validation_loss(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    bin_weights: torch.Tensor,
    seed: int,
    sigma_data: float,
    edm_p_mean: float,
    edm_p_std: float,
) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    samples = 0
    for condition, residual, _, truth in loader:
        condition = condition.to(device, non_blocking=True)
        residual = residual.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss, _, _ = tail_balanced_edm_loss(
                model, residual, condition, truth, bin_weights, generator,
                sigma_data, edm_p_mean, edm_p_std,
            )
        total += float(loss) * len(residual)
        samples += len(residual)
    return total / samples


def train(args: argparse.Namespace) -> None:
    if args.batch % 2:
        raise ValueError("source-balanced training requires an even --batch")
    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(args.initialize, map_location="cpu", weights_only=False)
    if parent.get("schema") != "hong2021-conditional-laplacian-residual-v8-observable-context":
        raise ValueError("V9 must initialize from the selected V8 checkpoint")
    residual_scale = cache_scale(args.tng_train_cache)
    if not np.isclose(residual_scale, float(parent["residual_scale"])):
        raise ValueError("parent and frozen TNG residual scales differ")
    tail_fit = balanced_tail_weights(
        density_bin_counts(args.tng_train_data),
        density_bin_counts(args.simba_train_data),
        args.tail_exponent,
        args.tail_maximum,
    )
    bin_weights = torch.tensor(tail_fit["weights"], device=device)
    feature_fit = parent["observable_context_features"]
    model = ObservableContextUNet(
        base_channels=int(parent["base_channels"]),
        context_mean=feature_fit["mean"], context_std=feature_fit["std"],
    ).to(device)
    model.load_state_dict(parent["ema_model"])
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    datasets: dict[str, Dataset] = {
        "tng_train": V6ResidualDataset(
            args.tng_train_data, args.tng_train_cache, residual_scale, True
        ),
        "simba_train": V6ResidualDataset(
            args.simba_train_data, args.simba_train_cache, residual_scale, True
        ),
        "tng_validation": V6ResidualDataset(
            args.tng_validation_data, args.tng_validation_cache, residual_scale, False
        ),
        "simba_validation": V6ResidualDataset(
            args.simba_validation_data, args.simba_validation_cache,
            residual_scale, False,
        ),
    }
    if args.smoke_limit is not None:
        datasets = {
            key: Subset(value, range(min(args.smoke_limit, len(value))))
            for key, value in datasets.items()
        }
    half = args.batch // 2
    train_tng = make_loader(
        datasets["tng_train"], half, args.workers, True, args.seed, device
    )
    train_simba = make_loader(
        datasets["simba_train"], half, args.workers, True, args.seed + 1, device
    )
    val_tng = make_loader(
        datasets["tng_validation"], args.validation_batch, args.workers,
        False, args.seed + 2, device,
    )
    val_simba = make_loader(
        datasets["simba_validation"], args.validation_batch, args.workers,
        False, args.seed + 3, device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.min_lr
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    noise = torch.Generator(device=device).manual_seed(args.seed + 100)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    candidates = output / "validation_checkpoints"
    candidates.mkdir(exist_ok=True)
    if (output / "last.pt").exists():
        raise SystemExit(f"refusing to overwrite existing run: {output}")
    initial_tng, initial_simba = validation_pair(
        ema_model, val_tng, val_simba, device, args.validation_seed,
        args.sigma_data, args.edm_p_mean, args.edm_p_std,
    )
    metadata = {
        "schema": SCHEMA,
        "status": "training",
        "method": "edm",
        "initialize": str(Path(args.initialize).resolve()),
        "initialize_step": int(parent["step"]),
        "observable_context_features": feature_fit,
        "tail_weight_fit": tail_fit,
        "tail_weight_fit_uses_training_truth_only": True,
        "nondevelopment_data_used": {
            "historical_simba_cv0_15": False,
            "sealed_eagle": False,
        },
        "tng_train_data": args.tng_train_data,
        "tng_train_cache": args.tng_train_cache,
        "simba_train_data": args.simba_train_data,
        "simba_train_cache": args.simba_train_cache,
        "tng_validation_data": args.tng_validation_data,
        "tng_validation_cache": args.tng_validation_cache,
        "simba_validation_data": args.simba_validation_data,
        "simba_validation_cache": args.simba_validation_cache,
        "residual_scale": residual_scale,
        "base_channels": int(parent["base_channels"]),
        "parameters": sum(value.numel() for value in model.parameters()),
        "steps": args.steps,
        "batch": args.batch,
        "source_balance_per_batch": {"tng": half, "simba": half},
        "ema_decay": args.ema_decay,
        "validation_every": args.validation_every,
        "validation_seed": args.validation_seed,
        "initial_unweighted_validation_loss": {
            "tng": initial_tng, "simba": initial_simba,
        },
        "sigma_data": args.sigma_data,
        "edm_p_mean": args.edm_p_mean,
        "edm_p_std": args.edm_p_std,
        "learning_rate": args.lr,
        "minimum_learning_rate": args.min_lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)

    iterator_tng, iterator_simba = cycling(train_tng), cycling(train_simba)
    history = []
    interval = {"total": 0.0, "unweighted": 0.0, "tail": 0.0, "samples": 0}
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        tng_condition, tng_residual, _, tng_truth = next(iterator_tng)
        simba_condition, simba_residual, _, simba_truth = next(iterator_simba)
        condition = torch.cat((tng_condition, simba_condition)).to(device, non_blocking=True)
        residual = torch.cat((tng_residual, simba_residual)).to(device, non_blocking=True)
        truth = torch.cat((tng_truth, simba_truth)).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss, unweighted, tail = tail_balanced_edm_loss(
                model, residual, condition, truth, bin_weights, noise,
                args.sigma_data, args.edm_p_mean, args.edm_p_std,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        update_ema(ema_model, model, args.ema_decay)
        interval["total"] += float(loss.detach()) * len(residual)
        interval["unweighted"] += float(unweighted.detach()) * len(residual)
        interval["tail"] += float(tail.detach()) * len(residual)
        interval["samples"] += len(residual)

        if step % args.validation_every == 0 or step == args.steps:
            val_unweighted_tng, val_unweighted_simba = validation_pair(
                ema_model, val_tng, val_simba, device, args.validation_seed,
                args.sigma_data, args.edm_p_mean, args.edm_p_std,
            )
            val_tail_tng = fixed_tail_validation_loss(
                ema_model, val_tng, device, bin_weights,
                args.validation_seed + 10, args.sigma_data,
                args.edm_p_mean, args.edm_p_std,
            )
            val_tail_simba = fixed_tail_validation_loss(
                ema_model, val_simba, device, bin_weights,
                args.validation_seed + 11, args.sigma_data,
                args.edm_p_mean, args.edm_p_std,
            )
            n = interval["samples"]
            row = {
                "step": step,
                "train_combined_loss": interval["total"] / n,
                "train_unweighted_loss": interval["unweighted"] / n,
                "train_tail_weighted_loss": interval["tail"] / n,
                "validation_unweighted_tng": val_unweighted_tng,
                "validation_unweighted_simba": val_unweighted_simba,
                "validation_tail_balanced_tng": val_tail_tng,
                "validation_tail_balanced_simba": val_tail_simba,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - start,
            }
            history.append(row)
            (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            checkpoint = {**metadata, **row, "ema_model": ema_model.state_dict()}
            atomic_save(checkpoint, candidates / f"step_{step:06d}.pt")
            full = {
                **checkpoint,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            atomic_save(full, output / "last.pt")
            print(
                f"step={step:06d} train={row['train_combined_loss']:.6f} "
                f"val_u={val_unweighted_tng:.6f}/{val_unweighted_simba:.6f} "
                f"val_tail={val_tail_tng:.6f}/{val_tail_simba:.6f} "
                f"elapsed={row['elapsed_seconds']:.0f}s", flush=True,
            )
            interval = {"total": 0.0, "unweighted": 0.0, "tail": 0.0, "samples": 0}
            model.train()
    metadata["status"] = "complete"
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


def sample(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != SCHEMA:
        raise ValueError("not a V9 checkpoint")
    feature_fit = checkpoint["observable_context_features"]
    model = ObservableContextUNet(
        base_channels=int(checkpoint["base_channels"]),
        context_mean=feature_fit["mean"], context_std=feature_fit["std"],
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval().to(device)
    scale = float(checkpoint["residual_scale"])
    dataset = V6ResidualDataset(args.data, args.cache, scale, False)
    indices = [int(value) for value in args.indices.split(",")]
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError("invalid --indices")
    output = Path(args.out)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    try:
        with h5py.File(temporary, "w") as handle:
            shape = (len(indices), args.ensemble, 1, dataset.grid, dataset.grid, dataset.grid)
            samples_ds = handle.create_dataset(
                "sample", shape=shape, dtype="f4",
                chunks=(1, 1, 1, dataset.grid, dataset.grid, dataset.grid), compression="lzf",
            )
            mean_ds = handle.create_dataset(
                "conditional_mean", shape=(len(indices), 1, dataset.grid, dataset.grid, dataset.grid),
                dtype="f4", compression="lzf",
            )
            truth_ds = handle.create_dataset(
                "truth", shape=(len(indices), 1, dataset.grid, dataset.grid, dataset.grid),
                dtype="f4", compression="lzf",
            )
            handle.create_dataset("source_index", data=np.asarray(indices, dtype=np.int64))
            for output_index, data_index in enumerate(indices):
                condition, _, mean, truth = dataset[data_index]
                condition = condition[None].to(device).expand(args.ensemble, -1, -1, -1, -1)
                residual = sample_edm(
                    model, condition, generator, args.sampling_steps,
                    args.sigma_min, args.sigma_max, args.rho,
                    float(checkpoint["sigma_data"]),
                )
                residual = residual - residual.mean(
                    dim=(-3, -2, -1), keepdim=True
                )
                samples_ds[output_index] = (mean[None].to(device) + scale * residual).cpu().numpy()
                mean_ds[output_index] = mean.numpy()
                truth_ds[output_index] = truth.numpy()
                print(f"[sample] V9 EDM {output_index+1}/{len(indices)}", flush=True)
            handle.attrs.update(
                {
                    "schema": SCHEMA,
                    "method": "edm",
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "checkpoint_step": int(checkpoint["step"]),
                    "source_cache": str(Path(args.cache).resolve()),
                    "sigma_cells": dataset.sigma_cells,
                    "residual_scale": scale,
                    "sampling_steps": args.sampling_steps,
                    "seed": args.seed,
                    "complete": True,
                }
            )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    training = sub.add_parser("train")
    training.add_argument("--initialize", required=True)
    for prefix in ("tng-train", "simba-train", "tng-validation", "simba-validation"):
        training.add_argument(f"--{prefix}-data", required=True)
        training.add_argument(f"--{prefix}-cache", required=True)
    training.add_argument("--out", required=True)
    training.add_argument("--steps", type=int, default=5000)
    training.add_argument("--batch", type=int, default=6)
    training.add_argument("--validation-batch", type=int, default=6)
    training.add_argument("--workers", type=int, default=1)
    training.add_argument("--lr", type=float, default=1.0e-5)
    training.add_argument("--min-lr", type=float, default=1.0e-6)
    training.add_argument("--weight-decay", type=float, default=1.0e-4)
    training.add_argument("--ema-decay", type=float, default=0.999)
    training.add_argument("--validation-every", type=int, default=500)
    training.add_argument("--validation-seed", type=int, default=108173)
    training.add_argument("--sigma-data", type=float, default=1.0)
    training.add_argument("--edm-p-mean", type=float, default=-0.8)
    training.add_argument("--edm-p-std", type=float, default=1.2)
    training.add_argument("--tail-exponent", type=float, default=0.5)
    training.add_argument("--tail-maximum", type=float, default=10.0)
    training.add_argument("--seed", type=int, default=5021)
    training.add_argument("--device", default="cuda")
    training.add_argument("--smoke-limit", type=int)
    sampling = sub.add_parser("sample")
    sampling.add_argument("--data", required=True)
    sampling.add_argument("--cache", required=True)
    sampling.add_argument("--checkpoint", required=True)
    sampling.add_argument("--out", required=True)
    sampling.add_argument("--indices", required=True)
    sampling.add_argument("--ensemble", type=int, default=16)
    sampling.add_argument("--sampling-steps", type=int, default=40)
    sampling.add_argument("--sigma-min", type=float, default=0.002)
    sampling.add_argument("--sigma-max", type=float, default=40.0)
    sampling.add_argument("--rho", type=float, default=7.0)
    sampling.add_argument("--seed", type=int, default=5777)
    sampling.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    {"train": train, "sample": sample}[args.mode](args)


if __name__ == "__main__":
    main()
