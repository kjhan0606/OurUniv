#!/usr/bin/env python
"""Two-component residual: learned low-pass mean plus V8 stochastic detail."""
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

from hong2021_residual_diffusion import ConditionalResidualUNet
from hong2021_residual_v6 import (
    V6ResidualDataset,
    atomic_save,
    cache_scale,
    sample_edm,
    seed_everything,
    update_ema,
)
from hong2021_residual_v8_context import (
    ObservableContextUNet,
    cycling,
    make_loader,
)
from hong2021_residual_v9_tail import (
    balanced_tail_weights,
    density_bin_counts,
    voxel_tail_weights,
)


SCHEMA = "hong2021-two-component-residual-v10-lowpass-mean-v8-edm"


class LowpassDataset(Dataset):
    def __init__(
        self, data: str, cache: str, residual_scale: float, augment: bool
    ) -> None:
        self.base = V6ResidualDataset(data, cache, residual_scale, augment)
        self.residual_scale = float(residual_scale)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        condition, laplacian_standardized, mean, truth = self.base[index]
        target = truth - mean - self.residual_scale * laplacian_standardized
        target = target - target.mean(dim=(-3, -2, -1), keepdim=True)
        return condition, target, truth


def correction_forward(
    model: ConditionalResidualUNet, condition: torch.Tensor
) -> torch.Tensor:
    batch = len(condition)
    zero_field = torch.zeros(
        (batch, 1) + tuple(condition.shape[-3:]),
        device=condition.device,
        dtype=condition.dtype,
    )
    zero_time = torch.zeros(batch, device=condition.device, dtype=condition.dtype)
    value = model(zero_field, condition, zero_time)
    return value - value.mean(dim=(-3, -2, -1), keepdim=True)


def correction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    truth: torch.Tensor,
    bin_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    error2 = (prediction - target).square()
    unweighted = error2.mean()
    weights = voxel_tail_weights(truth, bin_weights)
    weighted = (error2 * weights).sum() / weights.sum().clamp_min(1.0)
    return 0.5 * (unweighted + weighted), unweighted, weighted


@torch.inference_mode()
def validate(
    model: ConditionalResidualUNet,
    loader: Any,
    device: torch.device,
    bin_weights: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    total = np.zeros(3, dtype=np.float64)
    samples = 0
    for condition, target, truth in loader:
        condition = condition.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            values = correction_loss(
                correction_forward(model, condition), target, truth, bin_weights
            )
        total += np.asarray([float(value) for value in values]) * len(target)
        samples += len(target)
    return dict(
        zip(("combined", "unweighted", "tail_weighted"), (total / samples).tolist())
    )


def train(args: argparse.Namespace) -> None:
    if args.batch % 2:
        raise ValueError("source-balanced training requires an even batch")
    seed_everything(args.seed)
    device = torch.device(args.device)
    residual_scale = cache_scale(args.tng_train_cache)
    tail_fit = balanced_tail_weights(
        density_bin_counts(args.tng_train_data),
        density_bin_counts(args.simba_train_data),
    )
    bin_weights = torch.tensor(tail_fit["weights"], device=device)
    datasets: dict[str, Dataset] = {
        "tng_train": LowpassDataset(
            args.tng_train_data, args.tng_train_cache, residual_scale, True
        ),
        "simba_train": LowpassDataset(
            args.simba_train_data, args.simba_train_cache, residual_scale, True
        ),
        "tng_validation": LowpassDataset(
            args.tng_validation_data, args.tng_validation_cache, residual_scale, False
        ),
        "simba_validation": LowpassDataset(
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
    loaders = {
        "tng_train": make_loader(
            datasets["tng_train"], half, args.workers, True, args.seed, device
        ),
        "simba_train": make_loader(
            datasets["simba_train"], half, args.workers, True, args.seed + 1, device
        ),
        "tng_validation": make_loader(
            datasets["tng_validation"], args.validation_batch, args.workers,
            False, args.seed + 2, device,
        ),
        "simba_validation": make_loader(
            datasets["simba_validation"], args.validation_batch, args.workers,
            False, args.seed + 3, device,
        ),
    }
    model = ConditionalResidualUNet(base_channels=args.base_channels).to(device)
    nn.init.zeros_(model.output.weight)
    nn.init.zeros_(model.output.bias)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.min_lr
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    candidates = output / "validation_checkpoints"
    candidates.mkdir(exist_ok=True)
    if (output / "last.pt").exists():
        raise SystemExit(f"refusing to overwrite {output}")
    metadata = {
        "schema": SCHEMA,
        "status": "training",
        "component": "deterministic zero-DC omitted low-pass residual mean",
        "target": "truth - frozen_mean - Laplacian(truth-frozen_mean), cube mean removed",
        "stochastic_checkpoint": str(Path(args.stochastic_checkpoint).resolve()),
        "tail_weight_fit": tail_fit,
        "nondevelopment_data_used": {
            "historical_simba_cv0_15": False, "sealed_eagle": False,
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
        "base_channels": args.base_channels,
        "parameters": sum(value.numel() for value in model.parameters()),
        "steps": args.steps,
        "batch": args.batch,
        "source_balance_per_batch": {"tng": half, "simba": half},
        "learning_rate": args.lr,
        "minimum_learning_rate": args.min_lr,
        "ema_decay": args.ema_decay,
        "validation_every": args.validation_every,
        "exact_cube_dc_projection": True,
        "device": str(device),
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    iter_tng = cycling(loaders["tng_train"])
    iter_simba = cycling(loaders["simba_train"])
    history = []
    interval = np.zeros(4, dtype=np.float64)
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        tc, tt, ty = next(iter_tng)
        sc, st, sy = next(iter_simba)
        condition = torch.cat((tc, sc)).to(device, non_blocking=True)
        target = torch.cat((tt, st)).to(device, non_blocking=True)
        truth = torch.cat((ty, sy)).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            values = correction_loss(
                correction_forward(model, condition), target, truth, bin_weights
            )
        scaler.scale(values[0]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        update_ema(ema_model, model, args.ema_decay)
        interval[:3] += np.asarray([float(value.detach()) for value in values]) * len(target)
        interval[3] += len(target)
        if step % args.validation_every == 0 or step == args.steps:
            tng_val = validate(
                ema_model, loaders["tng_validation"], device, bin_weights
            )
            simba_val = validate(
                ema_model, loaders["simba_validation"], device, bin_weights
            )
            row = {
                "step": step,
                "train_combined": interval[0] / interval[3],
                "train_unweighted": interval[1] / interval[3],
                "train_tail_weighted": interval[2] / interval[3],
                "validation_tng": tng_val,
                "validation_simba": simba_val,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - start,
            }
            history.append(row)
            (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            checkpoint = {**metadata, **row, "ema_model": ema_model.state_dict()}
            atomic_save(checkpoint, candidates / f"step_{step:06d}.pt")
            atomic_save(
                {
                    **checkpoint,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                output / "last.pt",
            )
            print(
                f"step={step:06d} train={row['train_combined']:.6e} "
                f"val_tng={tng_val['combined']:.6e} "
                f"val_simba={simba_val['combined']:.6e} "
                f"elapsed={row['elapsed_seconds']:.0f}s", flush=True,
            )
            interval[:] = 0
            model.train()
    metadata["status"] = "complete"
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    correction_checkpoint = torch.load(
        args.correction_checkpoint, map_location="cpu", weights_only=False
    )
    stochastic_checkpoint = torch.load(
        args.stochastic_checkpoint, map_location="cpu", weights_only=False
    )
    correction_model = ConditionalResidualUNet(
        base_channels=int(correction_checkpoint["base_channels"])
    )
    correction_model.load_state_dict(correction_checkpoint["ema_model"])
    correction_model.eval().to(device)
    features = stochastic_checkpoint["observable_context_features"]
    stochastic_model = ObservableContextUNet(
        base_channels=int(stochastic_checkpoint["base_channels"]),
        context_mean=features["mean"], context_std=features["std"],
    )
    stochastic_model.load_state_dict(stochastic_checkpoint["ema_model"])
    stochastic_model.eval().to(device)
    scale = float(stochastic_checkpoint["residual_scale"])
    dataset = V6ResidualDataset(args.data, args.cache, scale, False)
    indices = [int(value) for value in args.indices.split(",")]
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError("invalid indices")
    output = Path(args.out)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    try:
        with h5py.File(temporary, "w") as handle:
            shape = (len(indices), args.ensemble, 1, dataset.grid, dataset.grid, dataset.grid)
            generated_ds = handle.create_dataset(
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
            for out_index, data_index in enumerate(indices):
                condition, _, mean, truth = dataset[data_index]
                condition_device = condition[None].to(device)
                correction = correction_forward(correction_model, condition_device)
                corrected_mean = mean[None].to(device) + correction
                ensemble_condition = condition_device.expand(args.ensemble, -1, -1, -1, -1)
                residual = sample_edm(
                    stochastic_model, ensemble_condition, generator,
                    args.sampling_steps, args.sigma_min, args.sigma_max,
                    args.rho, float(stochastic_checkpoint["sigma_data"]),
                )
                residual = residual - residual.mean(dim=(-3, -2, -1), keepdim=True)
                generated_ds[out_index] = (
                    corrected_mean + scale * residual
                ).cpu().numpy()
                mean_ds[out_index] = corrected_mean.cpu().numpy()
                truth_ds[out_index] = truth.numpy()
                print(f"[sample] V10 {out_index+1}/{len(indices)}", flush=True)
            handle.attrs.update(
                {
                    "schema": SCHEMA,
                    "method": "edm",
                    "checkpoint": str(Path(args.correction_checkpoint).resolve()),
                    "checkpoint_step": int(correction_checkpoint["step"]),
                    "stochastic_checkpoint": str(Path(args.stochastic_checkpoint).resolve()),
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
    training.add_argument("--stochastic-checkpoint", required=True)
    for prefix in ("tng-train", "simba-train", "tng-validation", "simba-validation"):
        training.add_argument(f"--{prefix}-data", required=True)
        training.add_argument(f"--{prefix}-cache", required=True)
    training.add_argument("--out", required=True)
    training.add_argument("--steps", type=int, default=5000)
    training.add_argument("--batch", type=int, default=6)
    training.add_argument("--validation-batch", type=int, default=6)
    training.add_argument("--workers", type=int, default=1)
    training.add_argument("--base-channels", type=int, default=16)
    training.add_argument("--lr", type=float, default=1.0e-4)
    training.add_argument("--min-lr", type=float, default=1.0e-5)
    training.add_argument("--weight-decay", type=float, default=1.0e-4)
    training.add_argument("--ema-decay", type=float, default=0.999)
    training.add_argument("--validation-every", type=int, default=500)
    training.add_argument("--seed", type=int, default=6021)
    training.add_argument("--device", default="cuda")
    training.add_argument("--smoke-limit", type=int)
    sampling = sub.add_parser("sample")
    sampling.add_argument("--data", required=True)
    sampling.add_argument("--cache", required=True)
    sampling.add_argument("--correction-checkpoint", required=True)
    sampling.add_argument("--stochastic-checkpoint", required=True)
    sampling.add_argument("--out", required=True)
    sampling.add_argument("--indices", required=True)
    sampling.add_argument("--ensemble", type=int, default=16)
    sampling.add_argument("--sampling-steps", type=int, default=40)
    sampling.add_argument("--sigma-min", type=float, default=0.002)
    sampling.add_argument("--sigma-max", type=float, default=40.0)
    sampling.add_argument("--rho", type=float, default=7.0)
    sampling.add_argument("--seed", type=int, default=13777)
    sampling.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    {"train": train, "sample": sample}[args.mode](args)


if __name__ == "__main__":
    main()
