#!/usr/bin/env python
"""Fair conditional residual comparison: EDM versus rectified flow.

V6 corrects the V5 pilot's short 3,600-update budget, stochastic validation
checkpointing, absent EMA, singular cosine-DDPM endpoint, and periodic/doubly
applied high-pass filter.  Both methods use the same observer conditions,
compact 3-D U-Net, full 48 cube isometries, Laplacian residual target, batch,
and optimizer-update budget.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_residual_diffusion import (
    ConditionalResidualUNet,
    radial_geometry,
    worker_seed,
)
from hong2021_train import apply_input_preprocessing


SCHEMA = "hong2021-conditional-laplacian-residual-v6"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def laplacian_residual(value: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Non-periodic high-frequency band with a constant-preserving DC null."""
    array = np.asarray(value, dtype=np.float32)
    smooth = gaussian_filter(
        array,
        sigma=(0,) * (array.ndim - 3) + (sigma_cells,) * 3,
        mode="reflect",
    )
    result = array - smooth
    result -= result.mean(axis=(-3, -2, -1), keepdims=True)
    return result.astype(np.float32)


def prepare_cache(args: argparse.Namespace) -> None:
    output = Path(args.out)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists():
        raise SystemExit(f"refusing to overwrite {output} or {temporary}")
    output.parent.mkdir(parents=True, exist_ok=True)
    square_sum = 0.0
    value_count = 0
    sample_count = 0
    start = time.time()
    try:
        with h5py.File(args.data, "r") as data, h5py.File(
            args.mean_cache, "r"
        ) as mean_cache, h5py.File(temporary, "w") as handle:
            target = data["target"]
            mean_source = mean_cache["conditional_mean"]
            if target.shape != mean_source.shape:
                raise ValueError("target and conditional mean shapes differ")
            sample_count = len(target)
            mean_ds = handle.create_dataset(
                "conditional_mean",
                shape=target.shape,
                dtype="f4",
                chunks=(1,) + target.shape[1:],
                compression="lzf",
            )
            residual_ds = handle.create_dataset(
                "laplacian_residual",
                shape=target.shape,
                dtype="f4",
                chunks=(1,) + target.shape[1:],
                compression="lzf",
            )
            for start_index in range(0, len(target), args.chunk):
                stop = min(start_index + args.chunk, len(target))
                mean = np.asarray(mean_source[start_index:stop], dtype=np.float32)
                truth = np.asarray(target[start_index:stop], dtype=np.float32)
                residual = laplacian_residual(
                    truth - mean, sigma_cells=args.sigma_cells
                )
                mean_ds[start_index:stop] = mean
                residual_ds[start_index:stop] = residual
                residual64 = residual.astype(np.float64)
                square_sum += float(np.square(residual64).sum())
                value_count += residual.size
                print(f"[prepare] {stop}/{len(target)}", flush=True)
            rms = math.sqrt(square_sum / value_count)
            handle.attrs.update(
                {
                    "schema": SCHEMA,
                    "source_data": str(Path(args.data).resolve()),
                    "source_mean_cache": str(Path(args.mean_cache).resolve()),
                    "input_preprocessing": mean_cache.attrs["input_preprocessing"],
                    "sigma_cells": args.sigma_cells,
                    "residual_rms": rms,
                    "voxel_mpc_h": float(data.attrs["voxel_mpc_h"]),
                    "complete": True,
                }
            )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    print(
        json.dumps(
            {
                "out": str(output),
                "samples": sample_count,
                "residual_rms": rms,
                "elapsed_seconds": time.time() - start,
            },
            indent=2,
        ),
        flush=True,
    )


class V6ResidualDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        cache_path: str | Path,
        residual_scale: float,
        augment: bool,
    ) -> None:
        self.data_path = str(data_path)
        self.cache_path = str(cache_path)
        self.residual_scale = float(residual_scale)
        self.augment = bool(augment)
        self._data: h5py.File | None = None
        self._cache: h5py.File | None = None
        with h5py.File(self.data_path, "r") as data, h5py.File(
            self.cache_path, "r"
        ) as cache:
            self.n = int(data["input"].shape[0])
            self.grid = int(data["input"].shape[-1])
            if cache.attrs["schema"] != SCHEMA:
                raise ValueError(f"unsupported cache schema: {cache.attrs['schema']}")
            if cache["conditional_mean"].shape[0] != self.n:
                raise ValueError("data/cache length mismatch")
            self.preprocessing = json.loads(cache.attrs["input_preprocessing"])
            self.sigma_cells = float(cache.attrs["sigma_cells"])
        if self.residual_scale <= 0:
            raise ValueError("residual scale must be positive")
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
            np.asarray(data["input"][index], dtype=np.float32),
            self.preprocessing,
        )
        mean = np.asarray(cache["conditional_mean"][index], dtype=np.float32)
        residual = (
            np.asarray(cache["laplacian_residual"][index], dtype=np.float32)
            / self.residual_scale
        )
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


def edm_coefficients(
    sigma: torch.Tensor, sigma_data: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma2 = sigma.square()
    data2 = sigma_data**2
    denominator = sigma2 + data2
    c_skip = data2 / denominator
    c_out = sigma * sigma_data / denominator.sqrt()
    c_in = denominator.rsqrt()
    c_noise = sigma.log() / 4.0
    return c_skip, c_out, c_in, c_noise


def edm_denoise(
    model: ConditionalResidualUNet,
    noisy: torch.Tensor,
    condition: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float,
) -> torch.Tensor:
    c_skip, c_out, c_in, c_noise = edm_coefficients(sigma, sigma_data)
    shape = (len(sigma),) + (1,) * (noisy.ndim - 1)
    network = model(
        c_in.reshape(shape) * noisy,
        condition,
        c_noise,
    )
    return c_skip.reshape(shape) * noisy + c_out.reshape(shape) * network


def method_loss(
    method: str,
    model: ConditionalResidualUNet,
    data: torch.Tensor,
    condition: torch.Tensor,
    generator: torch.Generator,
    sigma_data: float,
    edm_p_mean: float,
    edm_p_std: float,
) -> torch.Tensor:
    batch = len(data)
    if method == "edm":
        sigma = torch.exp(
            torch.randn(batch, device=data.device, generator=generator) * edm_p_std
            + edm_p_mean
        )
        noise = torch.randn(data.shape, device=data.device, generator=generator)
        noisy = data + sigma[:, None, None, None, None] * noise
        denoised = edm_denoise(model, noisy, condition, sigma, sigma_data)
        weight = (sigma.square() + sigma_data**2) / (
            sigma * sigma_data
        ).square()
        per_sample = (denoised - data).square().mean(dim=(1, 2, 3, 4))
        return (weight * per_sample).mean()
    if method == "flow":
        noise = torch.randn(data.shape, device=data.device, generator=generator)
        time_value = torch.rand(batch, device=data.device, generator=generator)
        shape = (batch,) + (1,) * (data.ndim - 1)
        path = (1.0 - time_value.reshape(shape)) * noise + time_value.reshape(
            shape
        ) * data
        target_velocity = data - noise
        predicted_velocity = model(path, condition, time_value)
        return F.mse_loss(predicted_velocity, target_velocity)
    raise ValueError(f"unknown method: {method}")


@torch.no_grad()
def update_ema(
    ema_model: nn.Module, model: nn.Module, decay: float
) -> None:
    for ema_parameter, parameter in zip(
        ema_model.parameters(), model.parameters(), strict=True
    ):
        ema_parameter.lerp_(parameter.detach(), 1.0 - decay)
    for ema_buffer, buffer in zip(
        ema_model.buffers(), model.buffers(), strict=True
    ):
        ema_buffer.copy_(buffer)


def fixed_validation_loss(
    method: str,
    model: ConditionalResidualUNet,
    loader: DataLoader,
    device: torch.device,
    seed: int,
    sigma_data: float,
    edm_p_mean: float,
    edm_p_std: float,
) -> float:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = 0.0
    samples = 0
    with torch.inference_mode():
        for condition, residual, _, _ in loader:
            condition = condition.to(device, non_blocking=True)
            residual = residual.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, enabled=device.type == "cuda"
            ):
                loss = method_loss(
                    method,
                    model,
                    residual,
                    condition,
                    generator,
                    sigma_data,
                    edm_p_mean,
                    edm_p_std,
                )
            total += float(loss) * len(residual)
            samples += len(residual)
    return total / samples


def atomic_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def cache_scale(path: str | Path) -> float:
    with h5py.File(path, "r") as handle:
        return float(handle.attrs["residual_rms"])


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    residual_scale = cache_scale(args.train_cache)
    train_set: Dataset = V6ResidualDataset(
        args.train_data, args.train_cache, residual_scale, augment=True
    )
    validation_set: Dataset = V6ResidualDataset(
        args.validation_data, args.validation_cache, residual_scale, augment=False
    )
    if args.smoke_limit:
        train_set = Subset(train_set, range(min(args.smoke_limit, len(train_set))))
        validation_set = Subset(
            validation_set, range(min(args.smoke_limit, len(validation_set)))
        )
    common = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "worker_init_fn": worker_seed,
    }
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
        **common,
    )
    validation_loader = DataLoader(
        validation_set, shuffle=False, drop_last=False, **common
    )
    model = ConditionalResidualUNet(base_channels=args.base_channels).to(device)
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
    training_noise = torch.Generator(device=device).manual_seed(args.seed + 100)
    iterator = iter(train_loader)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists():
        raise SystemExit(f"refusing to overwrite existing run: {output}")
    parameters = sum(value.numel() for value in model.parameters())
    metadata = {
        "schema": SCHEMA,
        "status": "training",
        "method": args.method,
        "train_data": args.train_data,
        "validation_data": args.validation_data,
        "train_cache": args.train_cache,
        "validation_cache": args.validation_cache,
        "residual_scale": residual_scale,
        "base_channels": args.base_channels,
        "parameters": parameters,
        "steps": args.steps,
        "batch": args.batch,
        "examples_budget": args.steps * args.batch,
        "ema_decay": args.ema_decay,
        "fixed_validation_seed": args.validation_seed,
        "validation_every": args.validation_every,
        "sigma_data": args.sigma_data,
        "edm_p_mean": args.edm_p_mean,
        "edm_p_std": args.edm_p_std,
        "augmentation": "uniform random full 48 signed cube isometries",
        "double_filter_at_sampling": False,
        "device": str(device),
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    history: list[dict[str, float | int]] = []
    interval_loss = 0.0
    interval_samples = 0
    best_validation = float("inf")
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        try:
            condition, residual, _, _ = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            condition, residual, _, _ = next(iterator)
        condition = condition.to(device, non_blocking=True)
        residual = residual.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = method_loss(
                args.method,
                model,
                residual,
                condition,
                training_noise,
                args.sigma_data,
                args.edm_p_mean,
                args.edm_p_std,
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
            train_loss = interval_loss / interval_samples
            validation_loss = fixed_validation_loss(
                args.method,
                ema_model,
                validation_loader,
                device,
                args.validation_seed,
                args.sigma_data,
                args.edm_p_mean,
                args.edm_p_std,
            )
            row = {
                "step": step,
                "examples_seen": step * args.batch,
                "train_loss_interval": train_loss,
                "fixed_validation_loss_ema": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - start,
            }
            history.append(row)
            (output / "history.json").write_text(
                json.dumps(history, indent=2) + "\n"
            )
            checkpoint = {
                **metadata,
                "step": step,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "fixed_validation_loss_ema": validation_loss,
            }
            atomic_save(checkpoint, output / "last.pt")
            if validation_loss < best_validation:
                best_validation = validation_loss
                atomic_save(checkpoint, output / "minimum_validation.pt")
            print(
                f"step={step:06d} train={train_loss:.6f} "
                f"validation_ema={validation_loss:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"elapsed={row['elapsed_seconds']:.0f}s",
                flush=True,
            )
            interval_loss = 0.0
            interval_samples = 0
            model.train()
    metadata["status"] = "complete"
    metadata["best_fixed_validation_loss_ema"] = best_validation
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


def karras_sigmas(
    steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    ramp = torch.linspace(0, 1, steps, device=device)
    minimum = sigma_min ** (1.0 / rho)
    maximum = sigma_max ** (1.0 / rho)
    sigma = (maximum + ramp * (minimum - maximum)) ** rho
    return torch.cat((sigma, torch.zeros(1, device=device)))


@torch.inference_mode()
def sample_edm(
    model: ConditionalResidualUNet,
    condition: torch.Tensor,
    generator: torch.Generator,
    steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    sigma_data: float,
    *,
    init_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    sigmas = karras_sigmas(
        steps, sigma_min, sigma_max, rho, condition.device
    )
    noise = torch.randn(
        (len(condition), 1) + tuple(condition.shape[-3:]),
        device=condition.device,
        generator=generator,
    )
    value = noise * sigmas[0] if init_transform is None else init_transform(noise)
    if value.shape != noise.shape or value.dtype != noise.dtype or value.device != noise.device:
        raise ValueError("EDM initialization transform changed tensor shape, dtype, or device")
    if not torch.isfinite(value).all():
        raise ValueError("EDM initialization transform produced non-finite values")
    for index in range(steps):
        sigma = sigmas[index]
        sigma_next = sigmas[index + 1]
        sigma_batch = torch.full(
            (len(condition),), float(sigma), device=condition.device
        )
        denoised = edm_denoise(
            model, value, condition, sigma_batch, sigma_data
        )
        derivative = (value - denoised) / sigma
        euler = value + (sigma_next - sigma) * derivative
        if sigma_next > 0:
            next_batch = torch.full(
                (len(condition),), float(sigma_next), device=condition.device
            )
            next_denoised = edm_denoise(
                model, euler, condition, next_batch, sigma_data
            )
            next_derivative = (euler - next_denoised) / sigma_next
            value = value + (sigma_next - sigma) * 0.5 * (
                derivative + next_derivative
            )
        else:
            value = euler
    return value


@torch.inference_mode()
def sample_flow(
    model: ConditionalResidualUNet,
    condition: torch.Tensor,
    generator: torch.Generator,
    steps: int,
) -> torch.Tensor:
    value = torch.randn(
        (len(condition), 1) + tuple(condition.shape[-3:]),
        device=condition.device,
        generator=generator,
    )
    dt = 1.0 / steps
    for index in range(steps):
        time_value = torch.full(
            (len(condition),), index / steps, device=condition.device
        )
        velocity = model(value, condition, time_value)
        euler = value + dt * velocity
        next_time = torch.full(
            (len(condition),), (index + 1) / steps, device=condition.device
        )
        next_velocity = model(euler, condition, next_time)
        value = value + 0.5 * dt * (velocity + next_velocity)
    return value


def sample(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ConditionalResidualUNet(
        condition_channels=4,
        base_channels=int(checkpoint["base_channels"]),
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval().to(device)
    method = str(checkpoint["method"])
    scale = float(checkpoint["residual_scale"])
    dataset = V6ResidualDataset(args.data, args.cache, scale, augment=False)
    indices = [int(value) for value in args.indices.split(",")]
    if len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= len(dataset):
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
                len(indices),
                args.ensemble,
                1,
                dataset.grid,
                dataset.grid,
                dataset.grid,
            )
            samples_ds = handle.create_dataset(
                "sample",
                shape=shape,
                dtype="f4",
                chunks=(1, 1, 1, dataset.grid, dataset.grid, dataset.grid),
                compression="lzf",
            )
            mean_ds = handle.create_dataset(
                "conditional_mean",
                shape=(len(indices), 1, dataset.grid, dataset.grid, dataset.grid),
                dtype="f4",
                compression="lzf",
            )
            truth_ds = handle.create_dataset(
                "truth",
                shape=(len(indices), 1, dataset.grid, dataset.grid, dataset.grid),
                dtype="f4",
                compression="lzf",
            )
            handle.create_dataset("source_index", data=np.asarray(indices, dtype=np.int64))
            for output_index, data_index in enumerate(indices):
                condition, _, mean, truth = dataset[data_index]
                condition = condition[None].to(device).expand(
                    args.ensemble, -1, -1, -1, -1
                )
                if method == "edm":
                    residual = sample_edm(
                        model,
                        condition,
                        generator,
                        args.sampling_steps,
                        args.sigma_min,
                        args.sigma_max,
                        args.rho,
                        float(checkpoint["sigma_data"]),
                    )
                elif method == "flow":
                    residual = sample_flow(
                        model, condition, generator, args.sampling_steps
                    )
                else:
                    raise ValueError(f"unknown checkpoint method: {method}")
                # Only the exact constant mode is projected out.  Unlike V5,
                # the already-band-limited generated target is not filtered a
                # second time.
                residual = residual - residual.mean(
                    dim=(-3, -2, -1), keepdim=True
                )
                generated = mean[None].to(device) + scale * residual
                samples_ds[output_index] = generated.cpu().numpy()
                mean_ds[output_index] = mean.numpy()
                truth_ds[output_index] = truth.numpy()
                print(
                    f"[sample] method={method} {output_index+1}/{len(indices)}",
                    flush=True,
                )
            handle.attrs.update(
                {
                    "schema": SCHEMA,
                    "method": method,
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "checkpoint_step": int(checkpoint["step"]),
                    "source_cache": str(Path(args.cache).resolve()),
                    "sigma_cells": dataset.sigma_cells,
                    "residual_scale": scale,
                    "sampling_steps": args.sampling_steps,
                    "seed": args.seed,
                    "double_filter_at_sampling": False,
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
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--mean-cache", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--sigma-cells", type=float, default=2.0)
    prepare.add_argument("--chunk", type=int, default=4)

    training = sub.add_parser("train")
    training.add_argument("--method", choices=("edm", "flow"), required=True)
    training.add_argument("--train-data", required=True)
    training.add_argument("--train-cache", required=True)
    training.add_argument("--validation-data", required=True)
    training.add_argument("--validation-cache", required=True)
    training.add_argument("--out", required=True)
    training.add_argument("--steps", type=int, default=20_000)
    training.add_argument("--batch", type=int, default=6)
    training.add_argument("--workers", type=int, default=1)
    training.add_argument("--base-channels", type=int, default=32)
    training.add_argument("--lr", type=float, default=2.0e-4)
    training.add_argument("--min-lr", type=float, default=2.0e-5)
    training.add_argument("--weight-decay", type=float, default=1.0e-4)
    training.add_argument("--ema-decay", type=float, default=0.999)
    training.add_argument("--validation-every", type=int, default=500)
    training.add_argument("--validation-seed", type=int, default=99173)
    training.add_argument("--sigma-data", type=float, default=1.0)
    training.add_argument("--edm-p-mean", type=float, default=-0.8)
    training.add_argument("--edm-p-std", type=float, default=1.2)
    training.add_argument("--seed", type=int, default=2021)
    training.add_argument("--device", default="cuda")
    training.add_argument("--smoke-limit", type=int, default=None)

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
    sampling.add_argument("--seed", type=int, default=777)
    sampling.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    {"prepare": prepare_cache, "train": train, "sample": sample}[args.mode](args)


if __name__ == "__main__":
    main()
