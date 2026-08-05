#!/usr/bin/env python
"""Conditional diffusion for unresolved Hong present-density residuals.

The frozen deterministic GroupNorm model supplies the conditional mean.  This
successor learns only the high-k residual in normalized log density and draws
an ensemble rather than claiming one unconstrained small-scale phase field is
the truth.  The observer-centred inputs admit the full 48 signed cube-axis
isometries (24 rotations and 24 mirrors), but not translations.

Modes
-----
prepare
    Cache deterministic predictions and high-pass residuals for one HDF5 split.
train
    Train a compact GroupNorm 3-D epsilon-prediction diffusion model.
sample
    Draw validation ensembles while exactly preserving the frozen low-k mean.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_evaluate import load_checkpoint_model
from hong2021_train import (
    AugmentedH5Dataset,
    apply_input_preprocessing,
)


SCHEMA = "hong2021-conditional-high-k-residual-diffusion-v1"


def cosine_transition(value: np.ndarray, low: float, high: float) -> np.ndarray:
    if not 0 <= low < high:
        raise ValueError("transition requires 0 <= low < high")
    phase = np.clip((value - low) / (high - low), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * phase)


def highpass_mask(
    grid: int,
    voxel_mpc_h: float,
    k_low_h_mpc: float,
    k_high_h_mpc: float,
) -> np.ndarray:
    """Smooth rFFT high-pass mask in physical angular wavenumber h/Mpc."""
    if grid <= 1 or voxel_mpc_h <= 0:
        raise ValueError("grid and voxel size must be positive")
    kxy = 2.0 * np.pi * np.fft.fftfreq(grid, d=voxel_mpc_h)
    kz = 2.0 * np.pi * np.fft.rfftfreq(grid, d=voxel_mpc_h)
    magnitude = np.sqrt(
        kxy[:, None, None] ** 2
        + kxy[None, :, None] ** 2
        + kz[None, None, :] ** 2
    )
    return cosine_transition(magnitude, k_low_h_mpc, k_high_h_mpc)


def highpass_numpy(value: np.ndarray, mask: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape[-3:-1] != mask.shape[:2] or array.shape[-1] // 2 + 1 != mask.shape[-1]:
        raise ValueError(f"field/mask shape mismatch: {array.shape}, {mask.shape}")
    transformed = np.fft.rfftn(array, axes=(-3, -2, -1))
    result = np.fft.irfftn(
        transformed * mask, s=array.shape[-3:], axes=(-3, -2, -1)
    )
    return result.astype(np.float32)


def radial_geometry(grid: int) -> np.ndarray:
    # The reflection centre of an even-sized voxel array lies between its two
    # central cells.  Using (N-1)/2 makes this channel exactly invariant under
    # every signed array-axis permutation used for augmentation.
    coordinate = np.arange(grid, dtype=np.float32) - (grid - 1) / 2.0
    x, y, z = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    return (np.sqrt(x * x + y * y + z * z) / (grid / 2.0)).astype(np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def prepare_cache(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, metadata = load_checkpoint_model(args.checkpoint, device)
    preprocessing = metadata["input_preprocessing"]
    dataset = AugmentedH5Dataset(
        args.data, augment=False, preprocessing=preprocessing
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    with h5py.File(args.data, "r") as source:
        samples = int(source["target"].shape[0])
        target_shape = tuple(int(v) for v in source["target"].shape[1:])
        voxel_mpc_h = float(source.attrs["voxel_mpc_h"])
    if target_shape[0] != 1 or len(set(target_shape[1:])) != 1:
        raise ValueError(f"unexpected target shape: {target_shape}")
    grid = target_shape[-1]
    mask = highpass_mask(
        grid, voxel_mpc_h, args.k_low_h_mpc, args.k_high_h_mpc
    )
    output = Path(args.out)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists():
        raise SystemExit(f"refusing to overwrite cache: {output} or {temporary}")
    output.parent.mkdir(parents=True, exist_ok=True)
    residual_square_sum = 0.0
    residual_count = 0
    offset = 0
    start = time.time()
    try:
        with h5py.File(temporary, "w") as handle:
            prediction_ds = handle.create_dataset(
                "conditional_mean",
                shape=(samples,) + target_shape,
                dtype="f4",
                chunks=(1,) + target_shape,
                compression="lzf",
            )
            residual_ds = handle.create_dataset(
                "highpass_residual",
                shape=(samples,) + target_shape,
                dtype="f4",
                chunks=(1,) + target_shape,
                compression="lzf",
            )
            model.eval()
            with torch.inference_mode():
                for x, target in loader:
                    prediction = model(
                        x.to(device, non_blocking=True)
                    ).cpu().numpy()
                    truth = target.numpy()
                    residual = highpass_numpy(truth - prediction, mask)
                    stop = offset + len(prediction)
                    prediction_ds[offset:stop] = prediction
                    residual_ds[offset:stop] = residual
                    residual64 = residual.astype(np.float64)
                    residual_square_sum += float(np.square(residual64).sum())
                    residual_count += residual.size
                    offset = stop
                    print(
                        f"[prepare] {offset}/{samples} elapsed={time.time()-start:.0f}s",
                        flush=True,
                    )
            if offset != samples:
                raise RuntimeError(f"prepared {offset} samples, expected {samples}")
            residual_rms = math.sqrt(residual_square_sum / residual_count)
            handle.attrs.update(
                {
                    "schema": SCHEMA,
                    "source_data": str(Path(args.data).resolve()),
                    "deterministic_checkpoint": str(Path(args.checkpoint).resolve()),
                    "deterministic_epoch": int(metadata["epoch"]),
                    "input_preprocessing": json.dumps(preprocessing),
                    "voxel_mpc_h": voxel_mpc_h,
                    "k_low_h_mpc": args.k_low_h_mpc,
                    "k_high_h_mpc": args.k_high_h_mpc,
                    "highpass_residual_rms": residual_rms,
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
                "samples": samples,
                "highpass_residual_rms": residual_rms,
                "elapsed_seconds": time.time() - start,
            },
            indent=2,
        ),
        flush=True,
    )


class ResidualDataset(Dataset):
    """Hong inputs, frozen mean, geometry, and standardized high-k residual."""

    def __init__(
        self,
        data_path: str | Path,
        cache_path: str | Path,
        residual_scale: float,
        augment: bool,
        seed: int = 0,
    ) -> None:
        self.data_path = str(data_path)
        self.cache_path = str(cache_path)
        self.residual_scale = float(residual_scale)
        self.augment = bool(augment)
        self.seed = int(seed)
        self._data: h5py.File | None = None
        self._cache: h5py.File | None = None
        with h5py.File(self.data_path, "r") as data, h5py.File(
            self.cache_path, "r"
        ) as cache:
            self.n = int(data["input"].shape[0])
            self.grid = int(data["input"].shape[-1])
            if cache["conditional_mean"].shape[0] != self.n:
                raise ValueError("data/cache sample counts differ")
            self.preprocessing = json.loads(cache.attrs["input_preprocessing"])
            self.cache_schema = str(cache.attrs["schema"])
            self.k_low_h_mpc = float(cache.attrs["k_low_h_mpc"])
            self.k_high_h_mpc = float(cache.attrs["k_high_h_mpc"])
        if self.cache_schema != SCHEMA:
            raise ValueError(f"unsupported residual cache: {self.cache_schema}")
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
            np.asarray(cache["highpass_residual"][index], dtype=np.float32)
            / self.residual_scale
        )
        truth = np.asarray(data["target"][index], dtype=np.float32)
        condition = np.concatenate((observable, mean, self.radial), axis=0)
        if self.augment:
            # The random choice is worker-local and changes on every access.  All
            # channels and the residual receive exactly the same signed isometry.
            transform = int(np.random.randint(len(CUBE_ISOMETRIES)))
            permutation, reflections = CUBE_ISOMETRIES[transform]
            joined = np.concatenate((condition, residual, truth), axis=0)
            joined = apply_cube_isometry(joined, permutation, reflections)
            n_condition = condition.shape[0]
            condition = joined[:n_condition]
            residual = joined[n_condition : n_condition + 1]
            truth = joined[n_condition + 1 :]
            mean = condition[2:3]
        return tuple(
            torch.from_numpy(np.ascontiguousarray(value))
            for value in (condition, residual, mean, truth)
        )


def group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    raise RuntimeError("no GroupNorm divisor")


class TimeEmbedding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.dimension = dimension
        self.network = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.SiLU(),
            nn.Linear(dimension * 4, dimension),
        )

    def forward(self, time_fraction: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequency = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=time_fraction.device)
            / max(half - 1, 1)
        )
        phase = time_fraction[:, None] * frequency[None] * 1000.0
        embedding = torch.cat((phase.sin(), phase.cos()), dim=1)
        if embedding.shape[1] < self.dimension:
            embedding = F.pad(embedding, (0, self.dimension - embedding.shape[1]))
        return self.network(embedding)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, value: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        hidden = hidden + self.time(time)[:, :, None, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(value)


class ConditionalResidualUNet(nn.Module):
    """Compact two-scale 3-D U-Net for residual epsilon prediction."""

    def __init__(self, condition_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        c = int(base_channels)
        self.condition_channels = int(condition_channels)
        self.base_channels = c
        self.time = TimeEmbedding(c * 4)
        self.input = nn.Conv3d(1 + condition_channels, c, 3, padding=1)
        self.level0 = ResidualBlock(c, c, c * 4)
        self.down1 = nn.Conv3d(c, 2 * c, 4, stride=2, padding=1)
        self.level1 = ResidualBlock(2 * c, 2 * c, c * 4)
        self.down2 = nn.Conv3d(2 * c, 4 * c, 4, stride=2, padding=1)
        self.middle1 = ResidualBlock(4 * c, 4 * c, c * 4)
        self.middle2 = ResidualBlock(4 * c, 4 * c, c * 4)
        self.up1 = nn.Conv3d(4 * c, 2 * c, 3, padding=1)
        self.decode1 = ResidualBlock(4 * c, 2 * c, c * 4)
        self.up0 = nn.Conv3d(2 * c, c, 3, padding=1)
        self.decode0 = ResidualBlock(2 * c, c, c * 4)
        self.output_norm = nn.GroupNorm(group_count(c), c)
        self.output = nn.Conv3d(c, 1, 3, padding=1)

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
        time = self.time(time_fraction)
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


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    if steps < 2:
        raise ValueError("diffusion steps must be at least two")
    position = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(
        ((position / steps + offset) / (1.0 + offset)) * math.pi / 2.0
    ) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    beta = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return beta.clamp(1.0e-5, 0.999).float()


def diffusion_arrays(steps: int, device: torch.device) -> dict[str, torch.Tensor]:
    beta = cosine_beta_schedule(steps).to(device)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    alpha_bar_previous = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
    return {
        "beta": beta,
        "alpha": alpha,
        "alpha_bar": alpha_bar,
        "alpha_bar_previous": alpha_bar_previous,
        "posterior_variance": beta
        * (1.0 - alpha_bar_previous)
        / (1.0 - alpha_bar),
    }


def extract(values: torch.Tensor, index: torch.Tensor, ndim: int = 5) -> torch.Tensor:
    result = values[index]
    return result.reshape((len(index),) + (1,) * (ndim - 1))


def diffusion_loss(
    model: ConditionalResidualUNet,
    residual: torch.Tensor,
    condition: torch.Tensor,
    schedule: dict[str, torch.Tensor],
    step: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    alpha_bar = extract(schedule["alpha_bar"], step)
    noisy = alpha_bar.sqrt() * residual + (1.0 - alpha_bar).sqrt() * noise
    predicted = model(noisy, condition, step.float() / len(schedule["beta"]))
    return F.mse_loss(predicted, noise)


def cache_residual_scale(path: str | Path) -> float:
    with h5py.File(path, "r") as handle:
        return float(handle.attrs["highpass_residual_rms"])


def worker_seed(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def run_epoch(
    model: ConditionalResidualUNet,
    loader: DataLoader,
    schedule: dict[str, torch.Tensor],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    seed: int,
) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    samples = 0
    generator = torch.Generator(device=device).manual_seed(seed)
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for condition, residual, _, _ in loader:
            condition = condition.to(device, non_blocking=True)
            residual = residual.to(device, non_blocking=True)
            batch = len(residual)
            step = torch.randint(
                len(schedule["beta"]), (batch,), device=device, generator=generator
            )
            noise = torch.randn(
                residual.shape, device=device, generator=generator
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            amp_enabled = device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss = diffusion_loss(
                    model, residual, condition, schedule, step, noise
                )
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            total += float(loss.detach()) * batch
            samples += batch
    return total / samples


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_scale = cache_residual_scale(args.train_cache)
    train_set: Dataset = ResidualDataset(
        args.train_data,
        args.train_cache,
        residual_scale=train_scale,
        augment=True,
        seed=args.seed,
    )
    validation_set: Dataset = ResidualDataset(
        args.validation_data,
        args.validation_cache,
        residual_scale=train_scale,
        augment=False,
        seed=args.seed + 1,
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
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, shuffle=True, generator=generator, drop_last=False, **common
    )
    validation_loader = DataLoader(
        validation_set, shuffle=False, drop_last=False, **common
    )
    model = ConditionalResidualUNet(base_channels=args.base_channels).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    schedule = diffusion_arrays(args.diffusion_steps, device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "schema": SCHEMA,
        "status": "training",
        "train_data": args.train_data,
        "validation_data": args.validation_data,
        "train_cache": args.train_cache,
        "validation_cache": args.validation_cache,
        "residual_scale": train_scale,
        "condition_channels": [
            "standardized_galaxy_count",
            "standardized_mean_radial_velocity",
            "frozen_deterministic_mean",
            "radial_geometry",
        ],
        "augmentation": {
            "training": "uniform random full 48 signed axis permutations",
            "rotations": 24,
            "mirrors": 24,
            "translations": 0,
            "validation": "none",
        },
        "base_channels": args.base_channels,
        "parameters": parameters,
        "diffusion_steps": args.diffusion_steps,
        "epochs": args.epochs,
        "batch": args.batch,
        "seed": args.seed,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "minimum_learning_rate": args.min_lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
    }
    (out / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    print(json.dumps(run_metadata, indent=2), flush=True)
    history: list[dict[str, float | int]] = []
    best = float("inf")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = run_epoch(
            model,
            train_loader,
            schedule,
            device,
            optimizer,
            scaler,
            seed=args.seed + epoch * 2,
        )
        validation_loss = run_epoch(
            model,
            validation_loader,
            schedule,
            device,
            optimizer=None,
            scaler=None,
            seed=args.seed + epoch * 2 + 1,
        )
        scheduler_lr.step()
        row = {
            "epoch": epoch,
            "train_epsilon_mse": train_loss,
            "validation_epsilon_mse": validation_loss,
            "learning_rate": learning_rate,
            "elapsed_seconds": time.time() - start,
        }
        history.append(row)
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        checkpoint = {
            "schema": SCHEMA,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_epsilon_mse": train_loss,
            "validation_epsilon_mse": validation_loss,
            "base_channels": args.base_channels,
            "condition_channels": model.condition_channels,
            "diffusion_steps": args.diffusion_steps,
            "residual_scale": train_scale,
            "train_cache": args.train_cache,
        }
        atomic_torch_save(checkpoint, out / "last_epoch.pt")
        if validation_loss < best:
            best = validation_loss
            atomic_torch_save(checkpoint, out / "minimum_validation_loss.pt")
        print(
            f"epoch={epoch:03d} train={train_loss:.6f} "
            f"validation={validation_loss:.6f} lr={learning_rate:.3e} "
            f"elapsed={row['elapsed_seconds']:.0f}s",
            flush=True,
        )
    run_metadata["status"] = "complete"
    run_metadata["best_validation_epsilon_mse"] = best
    (out / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")


@torch.inference_mode()
def draw_residual(
    model: ConditionalResidualUNet,
    condition: torch.Tensor,
    schedule: dict[str, torch.Tensor],
    generator: torch.Generator,
    x0_clip: float,
) -> torch.Tensor:
    value = torch.randn(
        (len(condition), 1) + tuple(condition.shape[-3:]),
        device=condition.device,
        generator=generator,
    )
    steps = len(schedule["beta"])
    for index in range(steps - 1, -1, -1):
        step = torch.full(
            (len(condition),), index, device=condition.device, dtype=torch.long
        )
        predicted_noise = model(value, condition, step.float() / steps)
        beta = extract(schedule["beta"], step)
        alpha = extract(schedule["alpha"], step)
        alpha_bar = extract(schedule["alpha_bar"], step)
        alpha_bar_previous = extract(schedule["alpha_bar_previous"], step)
        predicted_x0 = (
            value - torch.sqrt(1.0 - alpha_bar) * predicted_noise
        ) / torch.sqrt(alpha_bar)
        # The final cosine-schedule beta is nearly one.  A small epsilon error
        # there otherwise magnifies x0 by O(10^3), even when validation epsilon
        # MSE is good.  The standardized training residual is almost entirely
        # inside [-6,6] (absolute extrema -12.4/8.0), so this broad support bound
        # prevents numerical reverse-chain explosions without image-style
        # clipping to [-1,1].
        predicted_x0 = predicted_x0.clamp(-x0_clip, x0_clip)
        coefficient_x0 = (
            beta * torch.sqrt(alpha_bar_previous) / (1.0 - alpha_bar)
        )
        coefficient_xt = (
            (1.0 - alpha_bar_previous)
            * torch.sqrt(alpha)
            / (1.0 - alpha_bar)
        )
        mean = coefficient_x0 * predicted_x0 + coefficient_xt * value
        if index > 0:
            variance = extract(schedule["posterior_variance"], step)
            noise = torch.randn(
                value.shape, device=value.device, generator=generator
            )
            value = mean + torch.sqrt(variance.clamp_min(1.0e-20)) * noise
        else:
            value = mean
    return value


def sample(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ConditionalResidualUNet(
        condition_channels=int(checkpoint["condition_channels"]),
        base_channels=int(checkpoint["base_channels"]),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    residual_scale = float(checkpoint["residual_scale"])
    dataset = ResidualDataset(
        args.data,
        args.cache,
        residual_scale=residual_scale,
        augment=False,
        seed=args.seed,
    )
    if args.indices:
        selected = [int(value) for value in args.indices.split(",")]
        if len(set(selected)) != len(selected):
            raise ValueError("--indices must not contain duplicates")
        if not selected or min(selected) < 0 or max(selected) >= len(dataset):
            raise ValueError("--indices contains an out-of-range validation index")
    else:
        selected = list(
            range(args.start, min(args.start + args.samples, len(dataset)))
        )
    schedule = diffusion_arrays(int(checkpoint["diffusion_steps"]), device)
    mask = torch.from_numpy(
        highpass_mask(
            dataset.grid,
            args.voxel_mpc_h,
            dataset.k_low_h_mpc,
            dataset.k_high_h_mpc,
        )
    ).to(device)
    output = Path(args.out)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists():
        raise SystemExit(f"refusing to overwrite samples: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    try:
        with h5py.File(temporary, "w") as handle:
            count = len(selected)
            handle.create_dataset("source_index", data=np.asarray(selected, dtype=np.int64))
            shape = (count, args.ensemble, 1, dataset.grid, dataset.grid, dataset.grid)
            generated_ds = handle.create_dataset(
                "sample", shape=shape, dtype="f4", chunks=(1, 1, 1, dataset.grid, dataset.grid, dataset.grid), compression="lzf"
            )
            mean_ds = handle.create_dataset(
                "conditional_mean", shape=(count, 1, dataset.grid, dataset.grid, dataset.grid), dtype="f4", compression="lzf"
            )
            truth_ds = handle.create_dataset(
                "truth", shape=(count, 1, dataset.grid, dataset.grid, dataset.grid), dtype="f4", compression="lzf"
            )
            for output_index, data_index in enumerate(selected):
                condition, _, mean, truth = dataset[data_index]
                condition = condition[None].to(device)
                ensemble_condition = condition.expand(args.ensemble, -1, -1, -1, -1)
                standardized = draw_residual(
                    model,
                    ensemble_condition,
                    schedule,
                    generator,
                    x0_clip=args.x0_clip,
                )
                # Numerical/model leakage below the transition is removed again.
                transformed = torch.fft.rfftn(standardized, dim=(-3, -2, -1))
                standardized = torch.fft.irfftn(
                    transformed * mask,
                    s=standardized.shape[-3:],
                    dim=(-3, -2, -1),
                )
                generated = mean[None].to(device) + residual_scale * standardized
                generated_ds[output_index] = generated.cpu().numpy()
                mean_ds[output_index] = mean.numpy()
                truth_ds[output_index] = truth.numpy()
                print(
                    f"[sample] {output_index+1}/{count} ensemble={args.ensemble}",
                    flush=True,
                )
            handle.attrs.update(
                {
                    "schema": SCHEMA,
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "source_data": str(Path(args.data).resolve()),
                    "source_cache": str(Path(args.cache).resolve()),
                    "ensemble": args.ensemble,
                    "seed": args.seed,
                    "x0_clip_standardized_residual": args.x0_clip,
                    "k_low_h_mpc": dataset.k_low_h_mpc,
                    "k_high_h_mpc": dataset.k_high_h_mpc,
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
    subparsers = parser.add_subparsers(dest="mode", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--batch", type=int, default=6)
    prepare.add_argument("--workers", type=int, default=2)
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--k-low-h-mpc", type=float, default=2.0)
    prepare.add_argument("--k-high-h-mpc", type=float, default=4.0)

    training = subparsers.add_parser("train")
    training.add_argument("--train-data", required=True)
    training.add_argument("--train-cache", required=True)
    training.add_argument("--validation-data", required=True)
    training.add_argument("--validation-cache", required=True)
    training.add_argument("--out", required=True)
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--batch", type=int, default=2)
    training.add_argument("--workers", type=int, default=2)
    training.add_argument("--base-channels", type=int, default=32)
    training.add_argument("--diffusion-steps", type=int, default=200)
    training.add_argument("--lr", type=float, default=2.0e-4)
    training.add_argument("--min-lr", type=float, default=2.0e-5)
    training.add_argument("--weight-decay", type=float, default=1.0e-4)
    training.add_argument("--seed", type=int, default=2021)
    training.add_argument("--device", default="cuda")
    training.add_argument("--smoke-limit", type=int, default=None)

    sampling = subparsers.add_parser("sample")
    sampling.add_argument("--data", required=True)
    sampling.add_argument("--cache", required=True)
    sampling.add_argument("--checkpoint", required=True)
    sampling.add_argument("--out", required=True)
    sampling.add_argument("--samples", type=int, default=8)
    sampling.add_argument("--start", type=int, default=0)
    sampling.add_argument(
        "--indices",
        default=None,
        help="comma-separated validation indices; overrides --start/--samples",
    )
    sampling.add_argument("--ensemble", type=int, default=8)
    sampling.add_argument("--seed", type=int, default=2021)
    sampling.add_argument("--device", default="cuda")
    sampling.add_argument("--voxel-mpc-h", type=float, default=0.3125)
    sampling.add_argument(
        "--x0-clip",
        type=float,
        default=8.0,
        help="broad support bound in units of the training residual RMS",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    {"prepare": prepare_cache, "train": train, "sample": sample}[args.mode](args)


if __name__ == "__main__":
    main()
