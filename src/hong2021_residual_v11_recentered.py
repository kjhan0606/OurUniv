#!/usr/bin/env python
"""Corrected-mean, full-band stochastic residual EDM (V11).

V10 learned the low-frequency part missing from the frozen deterministic mean,
but continued drawing the old Laplacian residual around the old mean.  That is
not a centered two-component model: uncertainty left after the correction can
live below the Laplacian transition, and the old residual distribution need
not remain calibrated after the conditional mean changes.

V11 first freezes a V10 correction and materializes

    corrected_mean = original_mean + correction(observables)
    residual = truth - corrected_mean

with the residual cube mean projected out.  It then retrains the V8
observable-context EDM on that *full-band* centered residual using the original
uniform EDM likelihood and equal TNG/SIMBA source balance.  No simulation label,
historical SIMBA stress cube, or sealed EAGLE cube is used for fitting.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_residual_diffusion import ConditionalResidualUNet, radial_geometry
from hong2021_residual_v6 import (
    V6ResidualDataset,
    cache_scale,
    sample_edm,
    seed_everything,
)
from hong2021_residual_v8_context import (
    ObservableContextUNet,
    train as train_observable_context,
)
from hong2021_residual_v10_twocomponent import (
    SCHEMA as V10_SCHEMA,
    correction_forward,
)
from hong2021_train import apply_input_preprocessing


SCHEMA = "hong2021-corrected-mean-fullband-residual-v11-observable-context-edm"
CACHE_SCHEMA = "hong2021-corrected-mean-fullband-residual-cache-v11"


def centered_residual(truth: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    value = truth - mean
    return value - value.mean(dim=(-3, -2, -1), keepdim=True)


def prepare_cache(args: argparse.Namespace) -> None:
    """Materialize a frozen correction without exposing target data to it."""
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.correction_checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema") != V10_SCHEMA:
        raise ValueError(
            f"expected a V10 correction checkpoint, got {checkpoint.get('schema')}"
        )
    model = ConditionalResidualUNet(
        base_channels=int(checkpoint["base_channels"])
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    old_scale = cache_scale(args.original_cache)
    dataset = V6ResidualDataset(
        args.data, args.original_cache, old_scale, augment=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    output = Path(args.out)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists():
        raise SystemExit(f"refusing to overwrite {output} or {temporary}")
    output.parent.mkdir(parents=True, exist_ok=True)
    square_sum = 0.0
    value_count = 0
    offset = 0
    try:
        with h5py.File(args.data, "r") as source, h5py.File(
            temporary, "w"
        ) as handle:
            target_shape = source["target"].shape
            mean_ds = handle.create_dataset(
                "conditional_mean",
                shape=target_shape,
                dtype="f4",
                chunks=(1,) + target_shape[1:],
                compression="lzf",
            )
            residual_ds = handle.create_dataset(
                "centered_residual",
                shape=target_shape,
                dtype="f4",
                chunks=(1,) + target_shape[1:],
                compression="lzf",
            )
            for condition, _, old_mean, truth in loader:
                condition_device = condition.to(device, non_blocking=True)
                old_mean_device = old_mean.to(device, non_blocking=True)
                truth_device = truth.to(device, non_blocking=True)
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type, enabled=device.type == "cuda"
                ):
                    correction = correction_forward(model, condition_device)
                corrected = old_mean_device + correction.to(old_mean_device.dtype)
                residual = centered_residual(truth_device, corrected)
                corrected_np = corrected.float().cpu().numpy()
                residual_np = residual.float().cpu().numpy()
                stop = offset + len(residual_np)
                mean_ds[offset:stop] = corrected_np
                residual_ds[offset:stop] = residual_np
                residual64 = residual_np.astype(np.float64)
                square_sum += float(np.square(residual64).sum())
                value_count += residual_np.size
                offset = stop
                print(f"[prepare-v11] {offset}/{len(dataset)}", flush=True)
            if offset != len(dataset):
                raise RuntimeError("prepared sample count is incomplete")
            rms = math.sqrt(square_sum / value_count)
            handle.attrs.update(
                {
                    "schema": CACHE_SCHEMA,
                    "source_data": str(Path(args.data).resolve()),
                    "source_original_cache": str(
                        Path(args.original_cache).resolve()
                    ),
                    "correction_checkpoint": str(
                        Path(args.correction_checkpoint).resolve()
                    ),
                    "correction_checkpoint_step": int(checkpoint["step"]),
                    "input_preprocessing": json.dumps(dataset.preprocessing),
                    "target": (
                        "truth - (original_mean + frozen_V10_correction), "
                        "with exact per-cube DC projection"
                    ),
                    "residual_rms": rms,
                    "old_laplacian_residual_rms": old_scale,
                    "voxel_mpc_h": float(source.attrs["voxel_mpc_h"]),
                    "exact_cube_dc_projection": True,
                    "target_used_by_correction_model": False,
                    "correction_inference_precision": (
                        "CUDA autocast" if device.type == "cuda" else "float32"
                    ),
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
            {"out": str(output), "samples": offset, "residual_rms": rms},
            indent=2,
        ),
        flush=True,
    )


class V11ResidualDataset(Dataset):
    """Observables, corrected mean, and standardized full-band residual."""

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
            if str(cache.attrs.get("schema")) != CACHE_SCHEMA:
                raise ValueError(f"unsupported V11 cache: {cache.attrs.get('schema')}")
            self.n = int(data["input"].shape[0])
            self.grid = int(data["input"].shape[-1])
            expected = data["target"].shape
            if cache["conditional_mean"].shape != expected:
                raise ValueError("data/cache conditional-mean shape mismatch")
            if cache["centered_residual"].shape != expected:
                raise ValueError("data/cache centered-residual shape mismatch")
            self.preprocessing = json.loads(cache.attrs["input_preprocessing"])
        if not np.isfinite(self.residual_scale) or self.residual_scale <= 0:
            raise ValueError("residual scale must be positive and finite")
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
            np.asarray(cache["centered_residual"][index], dtype=np.float32)
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


def balanced_residual_scale(tng_cache: str | Path, simba_cache: str | Path) -> dict[str, Any]:
    with h5py.File(tng_cache, "r") as handle:
        if str(handle.attrs.get("schema")) != CACHE_SCHEMA:
            raise ValueError("TNG is not a V11 cache")
        tng = float(handle.attrs["residual_rms"])
    with h5py.File(simba_cache, "r") as handle:
        if str(handle.attrs.get("schema")) != CACHE_SCHEMA:
            raise ValueError("SIMBA is not a V11 cache")
        simba = float(handle.attrs["residual_rms"])
    balanced = math.sqrt(0.5 * (tng * tng + simba * simba))
    return {
        "tng_rms": tng,
        "simba_rms": simba,
        "balanced_rms": balanced,
        "weighting": "equal 0.5 TNG + 0.5 SIMBA development-train second moment",
    }


def initialize_recentered_parent(
    model: ObservableContextUNet, parent_state: dict[str, torch.Tensor]
) -> dict[str, list[str]]:
    """Load V8 weights but retain V11's freshly fitted feature moments."""
    excluded = {"context_mean", "context_std"}
    filtered = {key: value for key, value in parent_state.items() if key not in excluded}
    result = model.load_state_dict(filtered, strict=False)
    if set(result.missing_keys) != excluded or result.unexpected_keys:
        raise ValueError(
            "V8 parent is incompatible with V11: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "retained_new_buffers": sorted(excluded),
    }


def train(args: argparse.Namespace) -> None:
    parent = torch.load(args.initialize, map_location="cpu", weights_only=False)
    if parent.get("schema") != "hong2021-conditional-laplacian-residual-v8-observable-context":
        raise ValueError("V11 must initialize from the frozen V8 observable-context EDM")
    scale_fit = balanced_residual_scale(
        args.tng_train_cache, args.simba_train_cache
    )
    train_observable_context(
        args,
        dataset_type=V11ResidualDataset,
        run_schema=SCHEMA,
        residual_scale_override=float(scale_fit["balanced_rms"]),
        require_parent_scale=False,
        parent_initializer=initialize_recentered_parent,
        metadata_extra={
            "residual_target": (
                "full-band truth - frozen corrected conditional mean, exact "
                "cube-DC projection"
            ),
            "residual_scale_fit": scale_fit,
            "likelihood_weighting": "uniform EDM voxel likelihood",
            "correction_checkpoint": _common_correction_checkpoint(
                args.tng_train_cache,
                args.simba_train_cache,
                args.tng_validation_cache,
                args.simba_validation_cache,
            ),
            "low_frequency_stochastic_residual_allowed": True,
        },
    )


def _common_correction_checkpoint(*cache_paths: str | Path) -> str:
    checkpoints = set()
    for path in cache_paths:
        with h5py.File(path, "r") as handle:
            if str(handle.attrs.get("schema")) != CACHE_SCHEMA:
                raise ValueError(f"not a V11 cache: {path}")
            checkpoints.add(str(handle.attrs["correction_checkpoint"]))
    if len(checkpoints) != 1:
        raise ValueError("V11 caches were made with different correction checkpoints")
    return checkpoints.pop()


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != SCHEMA:
        raise ValueError(f"not a V11 checkpoint: {checkpoint.get('schema')}")
    features = checkpoint["observable_context_features"]
    model = ObservableContextUNet(
        base_channels=int(checkpoint["base_channels"]),
        context_mean=features["mean"],
        context_std=features["std"],
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval().to(device)
    scale = float(checkpoint["residual_scale"])
    correction_model: ConditionalResidualUNet | None = None
    if args.cache is not None:
        if args.original_cache is not None or args.correction_checkpoint is not None:
            raise ValueError(
                "use either --cache or --original-cache plus --correction-checkpoint"
            )
        dataset: Dataset = V11ResidualDataset(
            args.data, args.cache, scale, augment=False
        )
        source_cache = args.cache
    else:
        if args.original_cache is None or args.correction_checkpoint is None:
            raise ValueError(
                "sampling without --cache requires --original-cache and "
                "--correction-checkpoint"
            )
        correction_checkpoint = torch.load(
            args.correction_checkpoint, map_location="cpu", weights_only=False
        )
        if correction_checkpoint.get("schema") != V10_SCHEMA:
            raise ValueError("--correction-checkpoint is not a V10 checkpoint")
        correction_model = ConditionalResidualUNet(
            base_channels=int(correction_checkpoint["base_channels"])
        )
        correction_model.load_state_dict(correction_checkpoint["ema_model"])
        correction_model.eval().to(device)
        dataset = V6ResidualDataset(
            args.data,
            args.original_cache,
            cache_scale(args.original_cache),
            augment=False,
        )
        source_cache = args.original_cache
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
            handle.create_dataset(
                "source_index", data=np.asarray(indices, dtype=np.int64)
            )
            for output_index, data_index in enumerate(indices):
                condition, _, mean, truth = dataset[data_index]
                condition_single = condition[None].to(device)
                if correction_model is not None:
                    # Match the precision used when development caches were
                    # materialized, so sealed on-the-fly domains do not acquire
                    # a hardware-path shift in their corrected means.
                    with torch.autocast(
                        device_type=device.type, enabled=device.type == "cuda"
                    ):
                        correction = correction_forward(
                            correction_model, condition_single
                        )
                    correction = correction.to(condition_single.dtype)
                    corrected_mean = mean[None].to(device) + correction
                    condition_single = condition_single.clone()
                    condition_single[:, 2:3] = corrected_mean
                    mean = corrected_mean[0].cpu()
                condition_device = condition_single.expand(
                    args.ensemble, -1, -1, -1, -1
                )
                residual = sample_edm(
                    model,
                    condition_device,
                    generator,
                    args.sampling_steps,
                    args.sigma_min,
                    args.sigma_max,
                    args.rho,
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
                    f"[sample] V11 EDM {output_index + 1}/{len(indices)}",
                    flush=True,
                )
            handle.attrs.update(
                {
                    "schema": SCHEMA,
                    "method": "edm",
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "checkpoint_step": int(checkpoint["step"]),
                    "source_cache": str(Path(source_cache).resolve()),
                    "corrected_mean_source": (
                        "materialized V11 cache"
                        if correction_model is None
                        else "on-the-fly frozen V10 correction from observables only"
                    ),
                    "correction_checkpoint": (
                        checkpoint["correction_checkpoint"]
                        if correction_model is None
                        else str(Path(args.correction_checkpoint).resolve())
                    ),
                    "correction_inference_precision": (
                        "materialized cache"
                        if correction_model is None
                        else (
                            "CUDA autocast"
                            if device.type == "cuda"
                            else "float32"
                        )
                    ),
                    "residual_scale": scale,
                    "sampling_steps": args.sampling_steps,
                    "seed": args.seed,
                    "diagnostic_k_h_mpc": 0.3,
                    "residual_operator": "full-band residual with exact cube-DC projection",
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


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--initialize", required=True)
    for prefix in ("tng-train", "simba-train", "tng-validation", "simba-validation"):
        parser.add_argument(f"--{prefix}-data", required=True)
        parser.add_argument(f"--{prefix}-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--validation-batch", type=int, default=6)
    parser.add_argument("--feature-batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=2.0e-5)
    parser.add_argument("--context-lr", type=float, default=2.0e-4)
    parser.add_argument("--min-lr", type=float, default=2.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--validation-seed", type=int, default=108173)
    parser.add_argument("--sigma-data", type=float, default=1.0)
    parser.add_argument("--edm-p-mean", type=float, default=-0.8)
    parser.add_argument("--edm-p-std", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=7021)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-limit", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--original-cache", required=True)
    prepare.add_argument("--correction-checkpoint", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--batch", type=int, default=4)
    prepare.add_argument("--workers", type=int, default=1)
    prepare.add_argument("--seed", type=int, default=7021)
    prepare.add_argument("--device", default="cuda")
    training = sub.add_parser("train")
    add_training_arguments(training)
    sampling = sub.add_parser("sample")
    sampling.add_argument("--data", required=True)
    sampling.add_argument("--cache")
    sampling.add_argument("--original-cache")
    sampling.add_argument("--correction-checkpoint")
    sampling.add_argument("--checkpoint", required=True)
    sampling.add_argument("--out", required=True)
    sampling.add_argument("--indices", required=True)
    sampling.add_argument("--ensemble", type=int, default=16)
    sampling.add_argument("--sampling-steps", type=int, default=40)
    sampling.add_argument("--sigma-min", type=float, default=0.002)
    sampling.add_argument("--sigma-max", type=float, default=40.0)
    sampling.add_argument("--rho", type=float, default=7.0)
    sampling.add_argument("--seed", type=int, default=18777)
    sampling.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    {"prepare": prepare_cache, "train": train, "sample": sample}[args.mode](args)


if __name__ == "__main__":
    main()
