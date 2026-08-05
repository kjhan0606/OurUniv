#!/usr/bin/env python
"""Observable-context multi-domain EDM for Hong-style density residuals.

V7 used one stationary conditional residual distribution for every observer
cube.  That leaves a visible cross-code failure when tracer abundance and
velocity scatter shift.  V8 conditions the same U-Net on eight rotation- and
mirror-invariant statistics computed from quantities that are available for
real observations.  It never receives a simulation identity.

The context standardization is fit with equal weight on TNG and the designated
SIMBA development training set.  Historically inspected SIMBA CV0-15 and all
sealed EAGLE confirmation cubes are forbidden for fitting/model selection.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from hong2021_residual_diffusion import ConditionalResidualUNet, worker_seed
from hong2021_residual_v6 import (
    V6ResidualDataset,
    atomic_save,
    cache_scale,
    fixed_validation_loss,
    method_loss,
    sample_edm,
    seed_everything,
    update_ema,
)


SCHEMA = "hong2021-conditional-laplacian-residual-v8-observable-context"
FEATURE_NAMES = (
    "log1p_occupied_cells",
    "log1p_standardized_count_sum",
    "occupied_standardized_count_mean",
    "occupied_standardized_count_std",
    "occupied_standardized_velocity_mean",
    "occupied_standardized_velocity_std",
    "deterministic_mean_cube_mean",
    "deterministic_mean_cube_std",
)


def observable_context_features(condition: torch.Tensor) -> torch.Tensor:
    """Return cube-isometry-invariant features without using density truth."""
    if condition.ndim != 5 or condition.shape[1] < 3:
        raise ValueError(f"expected condition shape (B,>=3,N,N,N), got {condition.shape}")
    count = condition[:, 0]
    velocity = condition[:, 1]
    deterministic_mean = condition[:, 2]
    dimensions = (-3, -2, -1)
    occupied = count > 0
    weight = occupied.to(count.dtype)
    occupied_cells = weight.sum(dim=dimensions).clamp_min(1.0)
    count_sum = count.sum(dim=dimensions).clamp_min(0.0)
    count_mean = count_sum / occupied_cells
    count_second = count.square().sum(dim=dimensions) / occupied_cells
    count_std = (count_second - count_mean.square()).clamp_min(0.0).sqrt()
    velocity_sum = (velocity * weight).sum(dim=dimensions)
    velocity_mean = velocity_sum / occupied_cells
    velocity_second = (velocity.square() * weight).sum(dim=dimensions) / occupied_cells
    velocity_std = (
        velocity_second - velocity_mean.square()
    ).clamp_min(0.0).sqrt()
    mean_mean = deterministic_mean.mean(dim=dimensions)
    mean_std = deterministic_mean.std(dim=dimensions, correction=0)
    return torch.stack(
        (
            occupied_cells.log1p(),
            count_sum.log1p(),
            count_mean,
            count_std,
            velocity_mean,
            velocity_std,
            mean_mean,
            mean_std,
        ),
        dim=1,
    )


def source_feature_moments(
    dataset: Dataset, batch_size: int = 8
) -> tuple[torch.Tensor, torch.Tensor, int]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total = torch.zeros(len(FEATURE_NAMES), dtype=torch.float64)
    square = torch.zeros_like(total)
    samples = 0
    for condition, _, _, _ in loader:
        features = observable_context_features(condition).to(torch.float64)
        total += features.sum(dim=0)
        square += features.square().sum(dim=0)
        samples += len(features)
    if samples == 0:
        raise ValueError("cannot fit context features on an empty dataset")
    return total / samples, square / samples, samples


def balanced_feature_standardization(
    tng_dataset: Dataset, simba_dataset: Dataset, batch_size: int = 8
) -> dict[str, Any]:
    """Fit moments with equal source weight, irrespective of sample counts."""
    tng_mean, tng_second, tng_samples = source_feature_moments(
        tng_dataset, batch_size
    )
    simba_mean, simba_second, simba_samples = source_feature_moments(
        simba_dataset, batch_size
    )
    mean = 0.5 * (tng_mean + simba_mean)
    second = 0.5 * (tng_second + simba_second)
    std = (second - mean.square()).clamp_min(1.0e-12).sqrt()
    return {
        "feature_names": list(FEATURE_NAMES),
        "mean": mean.to(torch.float32).tolist(),
        "std": std.to(torch.float32).tolist(),
        "source_samples": {"tng": tng_samples, "simba": simba_samples},
        "source_mean": {
            "tng": tng_mean.tolist(),
            "simba": simba_mean.tolist(),
        },
        "source_std": {
            "tng": (tng_second - tng_mean.square()).clamp_min(0).sqrt().tolist(),
            "simba": (
                simba_second - simba_mean.square()
            ).clamp_min(0).sqrt().tolist(),
        },
        "weighting": "equal 0.5 TNG + 0.5 SIMBA development-train moments",
    }


class ObservableContextUNet(ConditionalResidualUNet):
    """V6 U-Net with global observable context added to its time embedding."""

    def __init__(
        self,
        condition_channels: int = 4,
        base_channels: int = 32,
        context_mean: list[float] | torch.Tensor | None = None,
        context_std: list[float] | torch.Tensor | None = None,
    ) -> None:
        super().__init__(condition_channels, base_channels)
        feature_count = len(FEATURE_NAMES)
        mean = torch.zeros(feature_count) if context_mean is None else torch.as_tensor(context_mean)
        std = torch.ones(feature_count) if context_std is None else torch.as_tensor(context_std)
        mean = mean.to(torch.float32)
        std = std.to(torch.float32)
        if mean.shape != (feature_count,) or std.shape != (feature_count,):
            raise ValueError("invalid observable-context standardization shape")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0):
            raise ValueError("invalid observable-context standardization values")
        self.register_buffer("context_mean", mean.clone())
        self.register_buffer("context_std", std.clone())
        time_channels = self.base_channels * 4
        self.context = nn.Sequential(
            nn.Linear(feature_count, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        # Loading a V6/V7 backbone then starts as the exact parent model.
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def standardized_context(self, condition: torch.Tensor) -> torch.Tensor:
        raw = observable_context_features(condition)
        return (raw - self.context_mean) / self.context_std

    def forward(
        self,
        noisy_residual: torch.Tensor,
        condition: torch.Tensor,
        time_fraction: torch.Tensor,
    ) -> torch.Tensor:
        if condition.shape[1] != self.condition_channels:
            raise ValueError(
                f"expected {self.condition_channels} condition channels, "
                f"received {condition.shape[1]}"
            )
        time = self.time(time_fraction) + self.context(
            self.standardized_context(condition)
        )
        level0 = self.level0(
            self.input(torch.cat((noisy_residual, condition), dim=1)), time
        )
        level1 = self.level1(self.down1(level0), time)
        middle = self.middle2(self.middle1(self.down2(level1), time), time)
        up1 = F.interpolate(middle, size=level1.shape[-3:], mode="nearest")
        up1 = self.up1(up1)
        up1 = self.decode1(torch.cat((up1, level1), dim=1), time)
        up0 = F.interpolate(up1, size=level0.shape[-3:], mode="nearest")
        up0 = self.up0(up0)
        up0 = self.decode0(torch.cat((up0, level0), dim=1), time)
        return self.output(F.silu(self.output_norm(up0)))


def initialize_parent(
    model: ObservableContextUNet, parent_state: dict[str, torch.Tensor]
) -> dict[str, list[str]]:
    result = model.load_state_dict(parent_state, strict=False)
    allowed_missing = {
        "context_mean",
        "context_std",
        "context.0.weight",
        "context.0.bias",
        "context.2.weight",
        "context.2.bias",
    }
    if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
        raise ValueError(
            "parent checkpoint is incompatible: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def cycling(loader: DataLoader) -> Iterator[Any]:
    while True:
        yield from loader


def make_loader(
    dataset: Dataset,
    batch: int,
    workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(seed),
    )


def validation_pair(
    model: nn.Module,
    tng_loader: DataLoader,
    simba_loader: DataLoader,
    device: torch.device,
    seed: int,
    sigma_data: float,
    edm_p_mean: float,
    edm_p_std: float,
) -> tuple[float, float]:
    tng = fixed_validation_loss(
        "edm", model, tng_loader, device, seed,
        sigma_data, edm_p_mean, edm_p_std,
    )
    simba = fixed_validation_loss(
        "edm", model, simba_loader, device, seed + 1,
        sigma_data, edm_p_mean, edm_p_std,
    )
    return tng, simba


def inference_checkpoint(
    metadata: dict[str, Any], row: dict[str, float | int], ema_model: nn.Module
) -> dict[str, Any]:
    return {**metadata, **row, "ema_model": ema_model.state_dict()}


def train(
    args: argparse.Namespace,
    *,
    dataset_type: type[Dataset] = V6ResidualDataset,
    run_schema: str = SCHEMA,
    residual_scale_override: float | None = None,
    require_parent_scale: bool = True,
    parent_initializer: Any = initialize_parent,
    metadata_extra: dict[str, Any] | None = None,
) -> None:
    """Train the observable-context EDM.

    The keyword-only hooks keep the published V8 CLI unchanged while allowing
    later, explicitly versioned experiments to reuse the identical optimizer
    and validation protocol with a different residual cache.  They are not
    exposed as command-line switches, so a V8 run cannot silently change its
    target semantics.
    """
    if args.batch % 2:
        raise ValueError("source-balanced training requires an even --batch")
    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(args.initialize, map_location="cpu", weights_only=False)
    if parent.get("method") != "edm":
        raise ValueError("V8 requires an EDM parent checkpoint")
    residual_scale = (
        cache_scale(args.tng_train_cache)
        if residual_scale_override is None
        else float(residual_scale_override)
    )
    if require_parent_scale and not np.isclose(
        float(parent["residual_scale"]), residual_scale
    ):
        raise ValueError("parent and frozen TNG residual scales differ")

    feature_tng = dataset_type(
        args.tng_train_data, args.tng_train_cache, residual_scale, augment=False
    )
    feature_simba = dataset_type(
        args.simba_train_data, args.simba_train_cache, residual_scale, augment=False
    )
    feature_fit = balanced_feature_standardization(
        feature_tng, feature_simba, args.feature_batch
    )
    datasets: dict[str, Dataset] = {
        "tng_train": dataset_type(
            args.tng_train_data, args.tng_train_cache, residual_scale, augment=True
        ),
        "simba_train": dataset_type(
            args.simba_train_data, args.simba_train_cache, residual_scale, augment=True
        ),
        "tng_validation": dataset_type(
            args.tng_validation_data, args.tng_validation_cache,
            residual_scale, augment=False,
        ),
        "simba_validation": dataset_type(
            args.simba_validation_data, args.simba_validation_cache,
            residual_scale, augment=False,
        ),
    }
    if args.smoke_limit is not None:
        for key, dataset in list(datasets.items()):
            datasets[key] = Subset(
                dataset, range(min(args.smoke_limit, len(dataset)))
            )
    half_batch = args.batch // 2
    train_tng = make_loader(
        datasets["tng_train"], half_batch, args.workers, True, args.seed, device
    )
    train_simba = make_loader(
        datasets["simba_train"], half_batch, args.workers, True,
        args.seed + 1, device,
    )
    validation_tng = make_loader(
        datasets["tng_validation"], args.validation_batch, args.workers,
        False, args.seed + 2, device,
    )
    validation_simba = make_loader(
        datasets["simba_validation"], args.validation_batch, args.workers,
        False, args.seed + 3, device,
    )

    base_channels = int(parent["base_channels"])
    model = ObservableContextUNet(
        base_channels=base_channels,
        context_mean=feature_fit["mean"],
        context_std=feature_fit["std"],
    ).to(device)
    parent_load = parent_initializer(model, parent["ema_model"])
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    context_parameters = list(model.context.parameters())
    context_ids = {id(value) for value in context_parameters}
    backbone_parameters = [
        value for value in model.parameters() if id(value) not in context_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_lr},
            {"params": context_parameters, "lr": args.context_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.min_lr
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    training_noise = torch.Generator(device=device).manual_seed(args.seed + 100)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    candidates = output / "validation_checkpoints"
    candidates.mkdir(exist_ok=True)
    if (output / "last.pt").exists():
        raise SystemExit(f"refusing to overwrite existing run: {output}")

    initial_tng, initial_simba = validation_pair(
        ema_model, validation_tng, validation_simba, device,
        args.validation_seed, args.sigma_data, args.edm_p_mean, args.edm_p_std,
    )
    metadata = {
        "schema": run_schema,
        "status": "training",
        "method": "edm",
        "initialize": str(Path(args.initialize).resolve()),
        "initialize_step": int(parent["step"]),
        "source_balance_per_batch": {"tng": half_batch, "simba": half_batch},
        "simulation_label_conditioning": False,
        "target_or_truth_context_features": False,
        "observable_context_features": feature_fit,
        "nondevelopment_data_used_for_fitting_or_selection": {
            "historically_inspected_simba_cv0_15": False,
            "sealed_eagle_confirmation": False,
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
        "base_channels": base_channels,
        "parameters": sum(value.numel() for value in model.parameters()),
        "parent_load": parent_load,
        "steps": args.steps,
        "batch": args.batch,
        "examples_budget": args.steps * args.batch,
        "examples_per_source": args.steps * half_batch,
        "ema_decay": args.ema_decay,
        "fixed_validation_seed": args.validation_seed,
        "validation_every": args.validation_every,
        "initial_fixed_validation_loss_ema": {
            "tng": initial_tng,
            "simba": initial_simba,
        },
        "checkpoint_selection": (
            "minimum max(TNG/initial_TNG, SIMBA-development/initial_SIMBA) "
            "fixed EMA denoising loss; field candidates retained for a "
            "predeclared development-only gate"
        ),
        "sigma_data": args.sigma_data,
        "edm_p_mean": args.edm_p_mean,
        "edm_p_std": args.edm_p_std,
        "backbone_learning_rate": args.backbone_lr,
        "context_learning_rate": args.context_lr,
        "minimum_learning_rate": args.min_lr,
        "weight_decay": args.weight_decay,
        "augmentation": "independent uniform random 48 cube isometries",
        "device": str(device),
    }
    if metadata_extra:
        overlap = set(metadata).intersection(metadata_extra)
        if overlap:
            raise ValueError(f"metadata_extra would overwrite keys: {sorted(overlap)}")
        metadata.update(metadata_extra)
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)

    tng_iterator = cycling(train_tng)
    simba_iterator = cycling(train_simba)
    history: list[dict[str, float | int]] = []
    best_worst_relative = float("inf")
    best_balanced = float("inf")
    interval_loss = 0.0
    interval_samples = 0
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        tng_condition, tng_residual, _, _ = next(tng_iterator)
        simba_condition, simba_residual, _, _ = next(simba_iterator)
        condition = torch.cat((tng_condition, simba_condition), dim=0).to(
            device, non_blocking=True
        )
        residual = torch.cat((tng_residual, simba_residual), dim=0).to(
            device, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = method_loss(
                "edm", model, residual, condition, training_noise,
                args.sigma_data, args.edm_p_mean, args.edm_p_std,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        update_ema(ema_model, model, args.ema_decay)
        interval_loss += float(loss.detach()) * len(residual)
        interval_samples += len(residual)

        if step % args.validation_every == 0 or step == args.steps:
            validation_tng_loss, validation_simba_loss = validation_pair(
                ema_model, validation_tng, validation_simba, device,
                args.validation_seed, args.sigma_data,
                args.edm_p_mean, args.edm_p_std,
            )
            balanced = 0.5 * (validation_tng_loss + validation_simba_loss)
            worst_relative = max(
                validation_tng_loss / initial_tng,
                validation_simba_loss / initial_simba,
            )
            row = {
                "step": step,
                "examples_seen": step * args.batch,
                "examples_seen_per_source": step * half_batch,
                "train_loss_interval": interval_loss / interval_samples,
                "fixed_validation_loss_ema_tng": validation_tng_loss,
                "fixed_validation_loss_ema_simba": validation_simba_loss,
                "balanced_validation_loss_ema": balanced,
                "worst_relative_validation_loss_ema": worst_relative,
                "backbone_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "context_learning_rate": float(optimizer.param_groups[1]["lr"]),
                "elapsed_seconds": time.time() - start,
            }
            history.append(row)
            (output / "history.json").write_text(
                json.dumps(history, indent=2) + "\n"
            )
            checkpoint = {
                **metadata,
                **row,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            atomic_save(checkpoint, output / "last.pt")
            atomic_save(
                inference_checkpoint(metadata, row, ema_model),
                candidates / f"step_{step:06d}.pt",
            )
            if worst_relative < best_worst_relative:
                best_worst_relative = worst_relative
                atomic_save(checkpoint, output / "minimum_validation.pt")
            if balanced < best_balanced:
                best_balanced = balanced
                atomic_save(checkpoint, output / "minimum_balanced_validation.pt")
            print(
                f"step={step:06d} train={row['train_loss_interval']:.6f} "
                f"val_tng={validation_tng_loss:.6f} "
                f"val_simba={validation_simba_loss:.6f} "
                f"worst_rel={worst_relative:.6f} "
                f"lr_backbone={optimizer.param_groups[0]['lr']:.3e} "
                f"lr_context={optimizer.param_groups[1]['lr']:.3e} "
                f"elapsed={row['elapsed_seconds']:.0f}s",
                flush=True,
            )
            interval_loss = 0.0
            interval_samples = 0
            model.train()
    metadata["status"] = "complete"
    metadata["best_worst_relative_validation_loss_ema"] = best_worst_relative
    metadata["best_balanced_validation_loss_ema"] = best_balanced
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


def sample(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != SCHEMA:
        raise ValueError(f"not a V8 checkpoint: {checkpoint.get('schema')}")
    feature_fit = checkpoint["observable_context_features"]
    model = ObservableContextUNet(
        base_channels=int(checkpoint["base_channels"]),
        context_mean=feature_fit["mean"],
        context_std=feature_fit["std"],
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval().to(device)
    scale = float(checkpoint["residual_scale"])
    dataset = V6ResidualDataset(args.data, args.cache, scale, augment=False)
    indices = [int(value) for value in args.indices.split(",")]
    if (
        not indices
        or len(set(indices)) != len(indices)
        or min(indices) < 0
        or max(indices) >= len(dataset)
    ):
        raise ValueError("invalid --indices")
    output = Path(args.out)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    try:
        with h5py.File(temporary, "w") as handle:
            shape = (
                len(indices), args.ensemble, 1,
                dataset.grid, dataset.grid, dataset.grid,
            )
            samples_ds = handle.create_dataset(
                "sample", shape=shape, dtype="f4",
                chunks=(1, 1, 1, dataset.grid, dataset.grid, dataset.grid),
                compression="lzf",
            )
            mean_ds = handle.create_dataset(
                "conditional_mean",
                shape=(len(indices), 1, dataset.grid, dataset.grid, dataset.grid),
                dtype="f4", compression="lzf",
            )
            truth_ds = handle.create_dataset(
                "truth",
                shape=(len(indices), 1, dataset.grid, dataset.grid, dataset.grid),
                dtype="f4", compression="lzf",
            )
            handle.create_dataset(
                "source_index", data=np.asarray(indices, dtype=np.int64)
            )
            for output_index, data_index in enumerate(indices):
                condition, _, mean, truth = dataset[data_index]
                condition = condition[None].to(device).expand(
                    args.ensemble, -1, -1, -1, -1
                )
                residual = sample_edm(
                    model, condition, generator, args.sampling_steps,
                    args.sigma_min, args.sigma_max, args.rho,
                    float(checkpoint["sigma_data"]),
                )
                residual = residual - residual.mean(
                    dim=(-3, -2, -1), keepdim=True
                )
                generated = mean[None].to(device) + scale * residual
                samples_ds[output_index] = generated.cpu().numpy()
                mean_ds[output_index] = mean.numpy()
                truth_ds[output_index] = truth.numpy()
                print(
                    f"[sample] V8 EDM {output_index+1}/{len(indices)}",
                    flush=True,
                )
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
                    "simulation_label_conditioning": False,
                    "complete": True,
                }
            )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    print(f"[out] {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    training = sub.add_parser("train")
    training.add_argument("--initialize", required=True)
    training.add_argument("--tng-train-data", required=True)
    training.add_argument("--tng-train-cache", required=True)
    training.add_argument("--simba-train-data", required=True)
    training.add_argument("--simba-train-cache", required=True)
    training.add_argument("--tng-validation-data", required=True)
    training.add_argument("--tng-validation-cache", required=True)
    training.add_argument("--simba-validation-data", required=True)
    training.add_argument("--simba-validation-cache", required=True)
    training.add_argument("--out", required=True)
    training.add_argument("--steps", type=int, default=10_000)
    training.add_argument("--batch", type=int, default=6)
    training.add_argument("--validation-batch", type=int, default=6)
    training.add_argument("--feature-batch", type=int, default=8)
    training.add_argument("--workers", type=int, default=1)
    training.add_argument("--backbone-lr", type=float, default=2.0e-5)
    training.add_argument("--context-lr", type=float, default=2.0e-4)
    training.add_argument("--min-lr", type=float, default=2.0e-6)
    training.add_argument("--weight-decay", type=float, default=1.0e-4)
    training.add_argument("--ema-decay", type=float, default=0.999)
    training.add_argument("--validation-every", type=int, default=500)
    training.add_argument("--validation-seed", type=int, default=98173)
    training.add_argument("--sigma-data", type=float, default=1.0)
    training.add_argument("--edm-p-mean", type=float, default=-0.8)
    training.add_argument("--edm-p-std", type=float, default=1.2)
    training.add_argument("--seed", type=int, default=4021)
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
    sampling.add_argument("--seed", type=int, default=4777)
    sampling.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    {"train": train, "sample": sample}[args.mode](args)


if __name__ == "__main__":
    main()
