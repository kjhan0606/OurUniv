#!/usr/bin/env python
"""Train and sample the frozen V14 three-domain multiscale residual EDM."""
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

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_residual_diffusion import radial_geometry
from hong2021_residual_v6 import (
    atomic_save,
    sample_edm,
    seed_everything,
    update_ema,
)
from hong2021_residual_v8_context import (
    FEATURE_NAMES,
    ObservableContextUNet,
    cycling,
    make_loader,
    source_feature_moments,
)
from hong2021_residual_v9_tail import (
    density_bin_counts,
    fixed_tail_validation_loss,
    tail_balanced_edm_loss,
)
from hong2021_train import apply_input_preprocessing
from hong2021_v14_mean_correction import DOMAINS, source_balanced_tail_weights
from hong2021_v14_multiscale import inverse_standardized_residual
from hong2021_v14_residual_cache import STANDARDIZED_SCHEMA


SCHEMA = "hong2021-v14-three-domain-multiscale-observable-context-edm-v1"
ENSEMBLE_SCHEMA = "hong2021-v14-multiscale-location-scale-edm-ensemble-v1"


class V14ResidualDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        cache_path: str | Path,
        augment: bool,
    ) -> None:
        self.data_path = str(data_path)
        self.cache_path = str(cache_path)
        self.augment = bool(augment)
        self._data: h5py.File | None = None
        self._cache: h5py.File | None = None
        with h5py.File(self.data_path, "r") as data, h5py.File(
            self.cache_path, "r"
        ) as cache:
            if str(cache.attrs.get("schema")) != STANDARDIZED_SCHEMA:
                raise ValueError(f"not a V14 standardized residual cache: {cache_path}")
            if not bool(cache.attrs.get("complete", False)):
                raise ValueError("V14 standardized residual cache is incomplete")
            if bool(cache.attrs.get("scale_prediction_uses_target", True)):
                raise ValueError("V14 scale prediction is not target-free")
            self.n = int(data["input"].shape[0])
            self.grid = int(data["input"].shape[-1])
            expected = (self.n, 1, self.grid, self.grid, self.grid)
            for name in ("conditional_mean", "standardized_residual"):
                if tuple(cache[name].shape) != expected:
                    raise ValueError(f"invalid {name} shape")
            if tuple(data["target"].shape) != expected:
                raise ValueError("data target shape differs from residual cache")
            if cache["predicted_residual_dc"].shape != (self.n,):
                raise ValueError("invalid predicted residual DC shape")
            if cache["predicted_band_scales"].shape != (self.n, 4):
                raise ValueError("invalid predicted band scale shape")
            self.preprocessing = json.loads(cache.attrs["input_preprocessing"])
            self.voxel_mpc_h = float(cache.attrs["voxel_mpc_h"])
            self.cache_rms = float(cache.attrs["standardized_residual_rms"])
        if not np.isfinite(self.cache_rms) or self.cache_rms <= 0:
            raise ValueError("invalid standardized residual RMS")
        self.radial = radial_geometry(self.grid)[None]

    def __len__(self) -> int:
        return self.n

    def _open(self) -> tuple[h5py.File, h5py.File]:
        if self._data is None:
            self._data = h5py.File(self.data_path, "r")
            self._cache = h5py.File(self.cache_path, "r")
        assert self._cache is not None
        return self._data, self._cache

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data, cache = self._open()
        observable = apply_input_preprocessing(
            np.asarray(data["input"][index], dtype=np.float32), self.preprocessing
        )
        mean = np.asarray(cache["conditional_mean"][index], dtype=np.float32)
        residual = np.asarray(cache["standardized_residual"][index], dtype=np.float32)
        truth = np.asarray(data["target"][index], dtype=np.float32)
        condition = np.concatenate((observable, mean, self.radial), axis=0)
        if self.augment:
            transform = int(np.random.randint(len(CUBE_ISOMETRIES)))
            permutation, reflections = CUBE_ISOMETRIES[transform]
            joined = apply_cube_isometry(
                np.concatenate((condition, residual, truth), axis=0),
                permutation,
                reflections,
            )
            channels = condition.shape[0]
            condition = joined[:channels]
            residual = joined[channels : channels + 1]
            truth = joined[channels + 1 :]
            mean = condition[2:3]
        return tuple(
            torch.from_numpy(np.ascontiguousarray(value))
            for value in (condition, residual, mean, truth)
        )

    def predicted_location_scales(self, index: int) -> tuple[float, np.ndarray]:
        _, cache = self._open()
        location = float(cache["predicted_residual_dc"][index])
        scales = np.asarray(cache["predicted_band_scales"][index], dtype=np.float64)
        if not np.isfinite(location) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise RuntimeError("cached location-scale prediction is invalid")
        return location, scales


def source_balanced_sigma_data(datasets: dict[str, V14ResidualDataset]) -> float:
    if set(datasets) != set(DOMAINS):
        raise ValueError("sigma_data requires exactly the three V14 sources")
    return float(np.sqrt(np.mean([value.cache_rms**2 for value in datasets.values()])))


def source_balanced_feature_standardization(
    datasets: dict[str, Dataset], batch_size: int = 8
) -> dict[str, Any]:
    if set(datasets) != set(DOMAINS):
        raise ValueError("feature standardization requires exactly three sources")
    rows = {}
    for name in DOMAINS:
        mean, second, samples = source_feature_moments(
            datasets[name], batch_size=batch_size
        )
        rows[name] = (mean, second, samples)
    mean = torch.stack([rows[name][0] for name in DOMAINS]).mean(dim=0)
    second = torch.stack([rows[name][1] for name in DOMAINS]).mean(dim=0)
    std = (second - mean.square()).clamp_min(1.0e-12).sqrt()
    return {
        "feature_names": list(FEATURE_NAMES),
        "mean": mean.to(torch.float32).tolist(),
        "std": std.to(torch.float32).tolist(),
        "source_samples": {name: rows[name][2] for name in DOMAINS},
        "source_mean": {name: rows[name][0].tolist() for name in DOMAINS},
        "source_std": {
            name: (rows[name][1] - rows[name][0].square()).clamp_min(0).sqrt().tolist()
            for name in DOMAINS
        },
        "source_weight": 1.0 / len(DOMAINS),
        "uses_density_truth": False,
    }


def train(args: argparse.Namespace) -> None:
    if args.batch % len(DOMAINS):
        raise ValueError("batch must divide equally across the three sources")
    candidate_steps = [int(value) for value in args.candidate_steps.split(",")]
    if (
        not candidate_steps
        or candidate_steps != sorted(set(candidate_steps))
        or candidate_steps[-1] > args.steps
        or any(step % args.validation_every for step in candidate_steps)
    ):
        raise ValueError("candidate steps must be unique ordered validation steps")
    seed_everything(args.seed)
    device = torch.device(args.device)
    paths = {
        "TNG100": (args.tng_train_data, args.tng_train_cache, args.tng_validation_data, args.tng_validation_cache),
        "SIMBA": (args.simba_train_data, args.simba_train_cache, args.simba_validation_data, args.simba_validation_cache),
        "Swift-EAGLE": (args.swift_train_data, args.swift_train_cache, args.swift_validation_data, args.swift_validation_cache),
    }
    train_datasets = {
        name: V14ResidualDataset(value[0], value[1], True)
        for name, value in paths.items()
    }
    validation_datasets = {
        name: V14ResidualDataset(value[2], value[3], False)
        for name, value in paths.items()
    }
    sigma_data = source_balanced_sigma_data(train_datasets)
    feature_fit = source_balanced_feature_standardization(train_datasets)
    tail_fit = source_balanced_tail_weights(
        {name: density_bin_counts(value[0]) for name, value in paths.items()},
        args.tail_exponent,
        args.tail_maximum,
    )
    if args.smoke_limit is not None:
        train_datasets = {
            name: Subset(value, range(min(args.smoke_limit, len(value))))
            for name, value in train_datasets.items()
        }
        validation_datasets = {
            name: Subset(value, range(min(args.smoke_limit, len(value))))
            for name, value in validation_datasets.items()
        }
    bin_weights = torch.tensor(tail_fit["weights"], device=device)
    source_batch = args.batch // len(DOMAINS)
    train_loaders = {
        name: make_loader(
            train_datasets[name], source_batch, args.workers, True,
            args.seed + index, device,
        )
        for index, name in enumerate(DOMAINS)
    }
    validation_loaders = {
        name: make_loader(
            validation_datasets[name], args.validation_batch, args.workers,
            False, args.seed + 10 + index, device,
        )
        for index, name in enumerate(DOMAINS)
    }
    model = ObservableContextUNet(
        base_channels=args.base_channels,
        context_mean=feature_fit["mean"],
        context_std=feature_fit["std"],
    ).to(device)
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
    noise = torch.Generator(device=device).manual_seed(args.seed + 100)
    validation_seeds = dict(zip(DOMAINS, args.validation_seeds, strict=True))
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / "validation_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    if (output / "last.pt").exists():
        raise RuntimeError(f"refusing to overwrite existing run: {output}")
    metadata = {
        "schema": SCHEMA,
        "status": "training",
        "target": "zero-DC multiscale-standardized corrected residual",
        "data": {
            name: {
                "train": paths[name][0], "train_cache": paths[name][1],
                "validation": paths[name][2], "validation_cache": paths[name][3],
            }
            for name in DOMAINS
        },
        "nondevelopment_data_used": False,
        "observable_context_features": feature_fit,
        "tail_weight_fit": tail_fit,
        "sigma_data": sigma_data,
        "sigma_data_fit": "sqrt(equal-source mean of training-cache residual RMS squared)",
        "base_channels": args.base_channels,
        "parameters": sum(value.numel() for value in model.parameters()),
        "steps": args.steps,
        "batch": args.batch,
        "source_balance_per_batch": {name: source_batch for name in DOMAINS},
        "learning_rate": args.lr,
        "minimum_learning_rate": args.min_lr,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "gradient_clip": args.gradient_clip,
        "edm_p_mean": args.edm_p_mean,
        "edm_p_std": args.edm_p_std,
        "validation_every": args.validation_every,
        "validation_seeds": validation_seeds,
        "candidate_steps": candidate_steps,
        "augmentation": "48 signed cube isometries",
        "seed": args.seed,
        "device": str(device),
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    iterators = {name: cycling(train_loaders[name]) for name in DOMAINS}
    history = []
    interval = np.zeros(4, dtype=np.float64)
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        batches = [next(iterators[name]) for name in DOMAINS]
        condition = torch.cat([value[0] for value in batches]).to(device, non_blocking=True)
        residual = torch.cat([value[1] for value in batches]).to(device, non_blocking=True)
        truth = torch.cat([value[3] for value in batches]).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            values = tail_balanced_edm_loss(
                model, residual, condition, truth, bin_weights, noise,
                sigma_data, args.edm_p_mean, args.edm_p_std,
            )
        scaler.scale(values[0]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        update_ema(ema_model, model, args.ema_decay)
        interval[:3] += np.asarray([float(value.detach()) for value in values]) * len(residual)
        interval[3] += len(residual)
        if step % args.validation_every == 0 or step == args.steps:
            validation = {
                name: fixed_tail_validation_loss(
                    ema_model, validation_loaders[name], device, bin_weights,
                    validation_seeds[name], sigma_data, args.edm_p_mean, args.edm_p_std,
                )
                for name in DOMAINS
            }
            row = {
                "step": step,
                "train": dict(
                    zip(
                        ("combined", "unweighted", "tail_weighted"),
                        (interval[:3] / interval[3]).tolist(),
                    )
                ),
                "fixed_tail_validation": validation,
                "balanced_validation": float(np.mean(list(validation.values()))),
                "worst_validation": float(max(validation.values())),
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
                },
                output / "last.pt",
            )
            print(
                f"step={step:06d} train={row['train']['combined']:.6e} "
                + " ".join(f"{name}={validation[name]:.6e}" for name in DOMAINS)
                + f" elapsed={row['elapsed_seconds']:.0f}s",
                flush=True,
            )
            interval[:] = 0
            model.train()
    metadata["status"] = "complete"
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != SCHEMA:
        raise ValueError("not a V14 multiscale EDM checkpoint")
    feature_fit = checkpoint["observable_context_features"]
    model = ObservableContextUNet(
        base_channels=int(checkpoint["base_channels"]),
        context_mean=feature_fit["mean"], context_std=feature_fit["std"],
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval().to(device)
    dataset = V14ResidualDataset(args.data, args.cache, False)
    indices = [int(value) for value in args.indices.split(",")]
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError("invalid sample indices")
    output = Path(args.out)
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite ensemble: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    try:
        with h5py.File(partial, "w") as handle:
            shape = (len(indices), args.ensemble, 1, dataset.grid, dataset.grid, dataset.grid)
            sample_ds = handle.create_dataset(
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
            location_ds = handle.create_dataset("predicted_residual_dc", shape=(len(indices),), dtype="f4")
            scale_ds = handle.create_dataset("predicted_band_scales", shape=(len(indices), 4), dtype="f4")
            for output_index, data_index in enumerate(indices):
                condition, _, corrected_mean, truth = dataset[data_index]
                location, scales = dataset.predicted_location_scales(data_index)
                condition_batch = condition[None].to(device).expand(
                    args.ensemble, -1, -1, -1, -1
                )
                standardized = sample_edm(
                    model, condition_batch, generator,
                    args.sampling_steps, args.sigma_min, args.sigma_max,
                    args.rho, float(checkpoint["sigma_data"]),
                )
                standardized -= standardized.mean(dim=(-3, -2, -1), keepdim=True)
                standardized_numpy = standardized[:, 0].float().cpu().numpy()
                physical = np.stack(
                    [
                        inverse_standardized_residual(
                            value,
                            predicted_location=location,
                            predicted_scales=scales,
                            voxel_mpc_h=dataset.voxel_mpc_h,
                        )
                        for value in standardized_numpy
                    ]
                ).astype(np.float32)
                mean_with_location = corrected_mean.numpy() + np.float32(location)
                sample_ds[output_index, :, 0] = corrected_mean.numpy()[0] + physical
                mean_ds[output_index] = mean_with_location
                truth_ds[output_index] = truth.numpy()
                location_ds[output_index] = location
                scale_ds[output_index] = scales
                print(f"[sample] V14 {output_index + 1}/{len(indices)}", flush=True)
            handle.attrs.update(
                {
                    "schema": ENSEMBLE_SCHEMA,
                    "method": "edm",
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "checkpoint_step": int(checkpoint["step"]),
                    "source_cache": str(Path(args.cache).resolve()),
                    "sigma_data": float(checkpoint["sigma_data"]),
                    "sampling_steps": args.sampling_steps,
                    "sigma_min": args.sigma_min,
                    "sigma_max": args.sigma_max,
                    "rho": args.rho,
                    "seed": args.seed,
                    "diagnostic_k_h_mpc": 1.0,
                    "location_scale_uses_target": False,
                    "complete": True,
                }
            )
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
    for prefix in (
        "tng-train", "tng-validation", "simba-train", "simba-validation",
        "swift-train", "swift-validation",
    ):
        training.add_argument(f"--{prefix}-data", required=True)
        training.add_argument(f"--{prefix}-cache", required=True)
    training.add_argument("--out", required=True)
    training.add_argument("--steps", type=int, default=10000)
    training.add_argument("--candidate-steps", default="2000,5000,10000")
    training.add_argument("--batch", type=int, default=6)
    training.add_argument("--validation-batch", type=int, default=6)
    training.add_argument("--workers", type=int, default=1)
    training.add_argument("--base-channels", type=int, default=32)
    training.add_argument("--lr", type=float, default=2.0e-4)
    training.add_argument("--min-lr", type=float, default=2.0e-5)
    training.add_argument("--weight-decay", type=float, default=1.0e-4)
    training.add_argument("--ema-decay", type=float, default=0.999)
    training.add_argument("--gradient-clip", type=float, default=1.0)
    training.add_argument("--edm-p-mean", type=float, default=-0.8)
    training.add_argument("--edm-p-std", type=float, default=1.2)
    training.add_argument("--tail-exponent", type=float, default=0.5)
    training.add_argument("--tail-maximum", type=float, default=10.0)
    training.add_argument("--validation-every", type=int, default=500)
    training.add_argument("--validation-seeds", type=int, nargs=3, default=[99173, 99174, 99175])
    training.add_argument("--seed", type=int, default=144021)
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
    sampling.add_argument("--seed", type=int, required=True)
    sampling.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    {"train": train, "sample": sample}[args.mode](args)


if __name__ == "__main__":
    main()
