#!/usr/bin/env python
"""Train the frozen V14 three-domain zero-DC deterministic mean correction."""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, Subset

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_residual_diffusion import ConditionalResidualUNet, radial_geometry
from hong2021_residual_v6 import atomic_save, seed_everything, update_ema
from hong2021_residual_v8_context import cycling, make_loader
from hong2021_residual_v9_tail import (
    DENSITY_BIN_LABELS,
    DENSITY_LOG10_BOUNDARIES,
    density_bin_counts,
    voxel_tail_weights,
)
from hong2021_train import apply_input_preprocessing
from hong2021_v14_baseline_audit import SCHEMA as BASELINE_CACHE_SCHEMA


SCHEMA = "hong2021-v14-three-domain-zero-dc-mean-correction-v1"
DOMAINS = ("TNG100", "SIMBA", "Swift-EAGLE")


class MeanCorrectionDataset(Dataset):
    """V14 conditions and exact centered residual target from a baseline audit."""

    def __init__(self, data_path: str | Path, cache_path: str | Path, augment: bool) -> None:
        self.data_path = str(data_path)
        self.cache_path = str(cache_path)
        self.augment = bool(augment)
        self._data: h5py.File | None = None
        self._cache: h5py.File | None = None
        with h5py.File(self.data_path, "r") as data, h5py.File(
            self.cache_path, "r"
        ) as cache:
            if str(cache.attrs.get("schema")) != BASELINE_CACHE_SCHEMA:
                raise ValueError(f"not a V14 baseline cache: {cache_path}")
            if bool(cache.attrs.get("feature_uses_target", True)):
                raise ValueError("baseline condition features are not target-free")
            self.n = int(data["input"].shape[0])
            self.grid = int(data["input"].shape[-1])
            expected = (self.n, 1, self.grid, self.grid, self.grid)
            if tuple(cache["conditional_mean"].shape) != expected:
                raise ValueError("data and baseline mean shapes differ")
            if tuple(cache["centered_residual"].shape) != expected:
                raise ValueError("invalid baseline centered residual shape")
            if tuple(data["target"].shape) != expected:
                raise ValueError("invalid density target shape")
            self.preprocessing = json.loads(cache.attrs["input_preprocessing"])
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data, cache = self._open()
        observable = apply_input_preprocessing(
            np.asarray(data["input"][index], dtype=np.float32), self.preprocessing
        )
        mean = np.asarray(cache["conditional_mean"][index], dtype=np.float32)
        target = np.asarray(cache["centered_residual"][index], dtype=np.float32)
        truth = np.asarray(data["target"][index], dtype=np.float32)
        condition = np.concatenate((observable, mean, self.radial), axis=0)
        if self.augment:
            transform = int(np.random.randint(len(CUBE_ISOMETRIES)))
            permutation, reflections = CUBE_ISOMETRIES[transform]
            joined = apply_cube_isometry(
                np.concatenate((condition, target, truth), axis=0),
                permutation,
                reflections,
            )
            channels = condition.shape[0]
            condition = joined[:channels]
            target = joined[channels : channels + 1]
            truth = joined[channels + 1 :]
        return tuple(
            torch.from_numpy(np.ascontiguousarray(value))
            for value in (condition, target, truth)
        )


def source_balanced_tail_weights(
    counts: dict[str, np.ndarray],
    exponent: float = 0.5,
    maximum: float = 10.0,
) -> dict[str, Any]:
    if len(counts) < 2 or not np.isfinite(exponent) or exponent < 0:
        raise ValueError("at least two sources and a nonnegative exponent are required")
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("tail-weight cap must be positive")
    probabilities = {}
    for name, value in counts.items():
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (len(DENSITY_BIN_LABELS),) or np.any(array <= 0):
            raise ValueError(f"invalid density-bin counts for {name}")
        probabilities[name] = array / array.sum()
    balanced = np.mean(list(probabilities.values()), axis=0)
    raw = np.power(balanced, -exponent)
    raw /= np.sum(balanced * raw)
    clipped = np.minimum(raw, maximum)
    weights = clipped / np.sum(balanced * clipped)
    return {
        "bin_labels": list(DENSITY_BIN_LABELS),
        "log10_density_boundaries": list(DENSITY_LOG10_BOUNDARIES),
        "counts": {name: np.asarray(value).tolist() for name, value in counts.items()},
        "probability": {
            **{name: value.tolist() for name, value in probabilities.items()},
            "equal_source": balanced.tolist(),
        },
        "source_weight": 1.0 / len(counts),
        "inverse_probability_exponent": exponent,
        "pre_normalization_maximum": maximum,
        "weights": weights.astype(np.float32).tolist(),
        "weight_expectation_equal_source": float(np.sum(balanced * weights)),
        "unweighted_objective_fraction": 0.5,
        "tail_balanced_objective_fraction": 0.5,
    }


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
    model: ConditionalResidualUNet | None,
    loader: Any,
    device: torch.device,
    bin_weights: torch.Tensor,
) -> dict[str, float]:
    if model is not None:
        model.eval()
    total = np.zeros(3, dtype=np.float64)
    samples = 0
    for condition, target, truth in loader:
        condition = condition.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        truth = truth.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = (
                torch.zeros_like(target)
                if model is None
                else correction_forward(model, condition)
            )
            values = correction_loss(prediction, target, truth, bin_weights)
        total += np.asarray([float(value) for value in values]) * len(target)
        samples += len(target)
    if samples == 0:
        raise ValueError("cannot validate an empty dataset")
    return dict(
        zip(("combined", "unweighted", "tail_weighted"), (total / samples).tolist())
    )


def select_correction_candidate(
    history: list[dict[str, Any]],
    baseline: dict[str, dict[str, float]],
    candidate_steps: list[int],
) -> dict[str, Any]:
    rows = {int(row["step"]): row for row in history}
    candidates = []
    for step in candidate_steps:
        if step not in rows:
            raise ValueError(f"candidate step {step} has no validation row")
        validation = rows[step]["validation"]
        relative = {
            domain: float(validation[domain]["combined"] / baseline[domain]["combined"])
            for domain in DOMAINS
        }
        candidates.append(
            {
                "step": step,
                "relative_combined_mse": relative,
                "improves_every_domain": all(value < 1.0 for value in relative.values()),
                "worst_relative_combined_mse": max(relative.values()),
            }
        )
    eligible = [row for row in candidates if row["improves_every_domain"]]
    selected = min(
        eligible,
        key=lambda row: (row["worst_relative_combined_mse"], row["step"]),
    ) if eligible else None
    return {
        "schema": "hong2021-v14-mean-correction-selection-v1",
        "rule": (
            "among candidates improving every source over exact zero correction, "
            "minimize worst source-relative combined validation MSE; smallest step tie break"
        ),
        "candidates": candidates,
        "selected_step": None if selected is None else selected["step"],
        "development_pass": selected is not None,
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
    tail_fit = source_balanced_tail_weights(
        {name: density_bin_counts(value[0]) for name, value in paths.items()},
        args.tail_exponent,
        args.tail_maximum,
    )
    bin_weights = torch.tensor(tail_fit["weights"], device=device)
    datasets: dict[str, Dataset] = {}
    for name, (train_data, train_cache, validation_data, validation_cache) in paths.items():
        datasets[f"{name}_train"] = MeanCorrectionDataset(train_data, train_cache, True)
        datasets[f"{name}_validation"] = MeanCorrectionDataset(
            validation_data, validation_cache, False
        )
    if args.smoke_limit is not None:
        datasets = {
            key: Subset(value, range(min(args.smoke_limit, len(value))))
            for key, value in datasets.items()
        }
    source_batch = args.batch // len(DOMAINS)
    loaders = {}
    for index, name in enumerate(DOMAINS):
        loaders[f"{name}_train"] = make_loader(
            datasets[f"{name}_train"], source_batch, args.workers, True,
            args.seed + index, device,
        )
        loaders[f"{name}_validation"] = make_loader(
            datasets[f"{name}_validation"], args.validation_batch, args.workers,
            False, args.seed + 10 + index, device,
        )
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
    checkpoints = output / "validation_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    if (output / "last.pt").exists():
        raise RuntimeError(f"refusing to overwrite existing run: {output}")
    baseline = {
        name: validate(
            None, loaders[f"{name}_validation"], device, bin_weights
        )
        for name in DOMAINS
    }
    metadata = {
        "schema": SCHEMA,
        "status": "training",
        "component": "deterministic exact-zero-DC residual mean correction",
        "target": "centered truth minus frozen V4 conditional mean",
        "tail_weight_fit": tail_fit,
        "data": {
            name: {
                "train": paths[name][0], "train_cache": paths[name][1],
                "validation": paths[name][2], "validation_cache": paths[name][3],
            }
            for name in DOMAINS
        },
        "nondevelopment_data_used": False,
        "base_channels": args.base_channels,
        "parameters": sum(value.numel() for value in model.parameters()),
        "steps": args.steps,
        "batch": args.batch,
        "source_balance_per_batch": {name: source_batch for name in DOMAINS},
        "learning_rate": args.lr,
        "minimum_learning_rate": args.min_lr,
        "weight_decay": args.weight_decay,
        "ema_decay": args.ema_decay,
        "validation_every": args.validation_every,
        "candidate_steps": candidate_steps,
        "zero_correction_validation_baseline": baseline,
        "augmentation": "48 signed cube isometries",
        "exact_cube_dc_projection": True,
        "seed": args.seed,
        "device": str(device),
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    iterators = {
        name: cycling(loaders[f"{name}_train"]) for name in DOMAINS
    }
    history = []
    interval = np.zeros(4, dtype=np.float64)
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        batches = [next(iterators[name]) for name in DOMAINS]
        condition = torch.cat([value[0] for value in batches]).to(device, non_blocking=True)
        target = torch.cat([value[1] for value in batches]).to(device, non_blocking=True)
        truth = torch.cat([value[2] for value in batches]).to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            values = correction_loss(
                correction_forward(model, condition), target, truth, bin_weights
            )
        scaler.scale(values[0]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        update_ema(ema_model, model, args.ema_decay)
        interval[:3] += np.asarray([float(value.detach()) for value in values]) * len(target)
        interval[3] += len(target)
        if step % args.validation_every == 0 or step == args.steps:
            validation = {
                name: validate(
                    ema_model, loaders[f"{name}_validation"], device, bin_weights
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
                "validation": validation,
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
            relative = {
                name: validation[name]["combined"] / baseline[name]["combined"]
                for name in DOMAINS
            }
            print(
                f"step={step:06d} train={row['train']['combined']:.6e} "
                + " ".join(f"{name}={relative[name]:.4f}x" for name in DOMAINS)
                + f" elapsed={row['elapsed_seconds']:.0f}s",
                flush=True,
            )
            interval[:] = 0
            model.train()
    selection = select_correction_candidate(history, baseline, candidate_steps)
    if selection["selected_step"] is not None:
        selection["selected_checkpoint"] = str(
            (checkpoints / f"step_{selection['selected_step']:06d}.pt").resolve()
        )
    else:
        selection["selected_checkpoint"] = None
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    metadata["status"] = "complete" if selection["development_pass"] else "failed_selection"
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(selection, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in (
        "tng-train", "tng-validation", "simba-train", "simba-validation",
        "swift-train", "swift-validation",
    ):
        parser.add_argument(f"--{prefix}-data", required=True)
        parser.add_argument(f"--{prefix}-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--candidate-steps", default="1000,3000,5000")
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--validation-batch", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--tail-exponent", type=float, default=0.5)
    parser.add_argument("--tail-maximum", type=float, default=10.0)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=142021)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-limit", type=int)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
