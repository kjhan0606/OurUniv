#!/usr/bin/env python
"""Train the centered residual EDM in a train-only Gaussianized coordinate.

V11 matches power and RMS but slightly broadens the negative SIMBA tail while
shrinking the positive tail.  V12 removes that hard score-learning problem by
fitting one monotone residual-to-normal map from an equal-weight mixture of
TNG and SIMBA *training* residuals.  The EDM operates in that latent coordinate;
sampling applies the frozen inverse map and then projects only the cube DC.

The map uses no validation, historical SIMBA, EAGLE, simulation label, or
target-dependent inference feature.  A single shared map is used for every
domain and real observation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy.special import ndtr
from torch.utils.data import Dataset

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_residual_diffusion import ConditionalResidualUNet, radial_geometry
from hong2021_residual_v6 import V6ResidualDataset, cache_scale, sample_edm, seed_everything
from hong2021_residual_v8_context import ObservableContextUNet, train as train_context_edm
from hong2021_residual_v10_twocomponent import SCHEMA as V10_SCHEMA, correction_forward
from hong2021_residual_v11_recentered import (
    CACHE_SCHEMA as V11_CACHE_SCHEMA,
    SCHEMA as V11_SCHEMA,
)
from hong2021_train import apply_input_preprocessing


SCHEMA = "hong2021-gaussianized-fullband-residual-v12-observable-context-edm"
CACHE_SCHEMA = "hong2021-gaussianized-fullband-residual-cache-v12"
TRANSFORM_SCHEMA = "hong2021-balanced-train-residual-gaussianization-v1"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def residual_extrema(path: str | Path) -> tuple[float, float, int]:
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("schema")) != V11_CACHE_SCHEMA:
            raise ValueError(f"not a V11 cache: {path}")
        data = handle["centered_residual"]
        low = float("inf")
        high = float("-inf")
        count = 0
        for index in range(len(data)):
            value = np.asarray(data[index], dtype=np.float32)
            low = min(low, float(value.min()))
            high = max(high, float(value.max()))
            count += value.size
    return low, high, count


def residual_histogram(
    path: str | Path, edges: np.ndarray
) -> tuple[np.ndarray, int]:
    histogram = np.zeros(len(edges) - 1, dtype=np.int64)
    count = 0
    with h5py.File(path, "r") as handle:
        data = handle["centered_residual"]
        for index in range(len(data)):
            value = np.asarray(data[index], dtype=np.float32)
            histogram += np.histogram(value, bins=edges)[0]
            count += value.size
    if int(histogram.sum()) != count:
        raise RuntimeError("histogram did not contain every residual voxel")
    return histogram, count


def fit_transform(args: argparse.Namespace) -> None:
    output = Path(args.out)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    tng_low, tng_high, tng_count = residual_extrema(args.tng_train_cache)
    simba_low, simba_high, simba_count = residual_extrema(args.simba_train_cache)
    low = min(tng_low, simba_low)
    high = max(tng_high, simba_high)
    padding = max(1.0e-6, (high - low) * 1.0e-6)
    edges = np.linspace(low - padding, high + padding, args.histogram_bins + 1)
    tng_histogram, checked_tng_count = residual_histogram(
        args.tng_train_cache, edges
    )
    simba_histogram, checked_simba_count = residual_histogram(
        args.simba_train_cache, edges
    )
    if checked_tng_count != tng_count or checked_simba_count != simba_count:
        raise RuntimeError("residual count changed between transform passes")
    tng_probability = tng_histogram.astype(np.float64) / tng_count
    simba_probability = simba_histogram.astype(np.float64) / simba_count
    probability_mass = 0.5 * (tng_probability + simba_probability)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cdf_midpoint = np.cumsum(probability_mass) - 0.5 * probability_mass
    z_knots = np.linspace(-args.z_limit, args.z_limit, args.knots)
    probabilities = ndtr(z_knots)
    value_knots = np.interp(
        probabilities,
        cdf_midpoint,
        centers,
        left=centers[0],
        right=centers[-1],
    )
    # Histogram plateaus can repeat an extreme quantile.  A monotone transform
    # needs strictly increasing abscissae for its inverse interpolation.
    value_knots = np.maximum.accumulate(value_knots)
    for index in range(1, len(value_knots)):
        if value_knots[index] <= value_knots[index - 1]:
            value_knots[index] = np.nextafter(
                value_knots[index - 1], float("inf")
            )
    report = {
        "schema": TRANSFORM_SCHEMA,
        "fitting_data": {
            "tng_train_cache": str(Path(args.tng_train_cache).resolve()),
            "tng_train_sha256": file_sha256(args.tng_train_cache),
            "tng_voxels": tng_count,
            "simba_train_cache": str(Path(args.simba_train_cache).resolve()),
            "simba_train_sha256": file_sha256(args.simba_train_cache),
            "simba_voxels": simba_count,
        },
        "source_weighting": {"tng": 0.5, "simba_development_train": 0.5},
        "validation_or_test_data_used": False,
        "histogram_bins": args.histogram_bins,
        "histogram_range": [float(edges[0]), float(edges[-1])],
        "z_limit": args.z_limit,
        "z_knots": z_knots.tolist(),
        "residual_value_knots": value_knots.tolist(),
        "tail_behavior": "clip latent z to fixed train-fitted knot support",
        "inverse_postprocessing": "exact per-cube residual DC projection",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "out": str(output),
                "tng_voxels": tng_count,
                "simba_voxels": simba_count,
                "range": report["histogram_range"],
                "knots": args.knots,
            },
            indent=2,
        ),
        flush=True,
    )


def load_transform(path: str | Path) -> dict[str, Any]:
    report = json.loads(Path(path).read_text())
    if report.get("schema") != TRANSFORM_SCHEMA:
        raise ValueError(f"not a V12 transform: {path}")
    z = np.asarray(report["z_knots"], dtype=np.float64)
    value = np.asarray(report["residual_value_knots"], dtype=np.float64)
    if z.ndim != 1 or value.shape != z.shape or len(z) < 3:
        raise ValueError("invalid transform knot shape")
    if not np.all(np.diff(z) > 0) or not np.all(np.diff(value) > 0):
        raise ValueError("transform knots must be strictly increasing")
    return report


def gaussianize_numpy(value: np.ndarray, transform: dict[str, Any]) -> np.ndarray:
    return np.interp(
        np.asarray(value, dtype=np.float32),
        np.asarray(transform["residual_value_knots"], dtype=np.float64),
        np.asarray(transform["z_knots"], dtype=np.float64),
    ).astype(np.float32)


def inverse_gaussianize_numpy(
    value: np.ndarray, transform: dict[str, Any]
) -> np.ndarray:
    return np.interp(
        np.asarray(value, dtype=np.float32),
        np.asarray(transform["z_knots"], dtype=np.float64),
        np.asarray(transform["residual_value_knots"], dtype=np.float64),
    ).astype(np.float32)


def inverse_gaussianize_torch(
    value: torch.Tensor, z_knots: torch.Tensor, residual_knots: torch.Tensor
) -> torch.Tensor:
    clipped = value.clamp(float(z_knots[0]), float(z_knots[-1]))
    upper = torch.searchsorted(z_knots, clipped).clamp(1, len(z_knots) - 1)
    lower = upper - 1
    z0 = z_knots[lower]
    z1 = z_knots[upper]
    r0 = residual_knots[lower]
    r1 = residual_knots[upper]
    fraction = (clipped - z0) / (z1 - z0)
    return r0 + fraction * (r1 - r0)


def prepare_cache(args: argparse.Namespace) -> None:
    transform = load_transform(args.transform)
    output = Path(args.out)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists():
        raise SystemExit(f"refusing to overwrite {output} or {temporary}")
    output.parent.mkdir(parents=True, exist_ok=True)
    square_sum = 0.0
    value_count = 0
    try:
        with h5py.File(args.v11_cache, "r") as source, h5py.File(
            temporary, "w"
        ) as handle:
            if str(source.attrs.get("schema")) != V11_CACHE_SCHEMA:
                raise ValueError("--v11-cache is not a V11 cache")
            shape = source["centered_residual"].shape
            mean_ds = handle.create_dataset(
                "conditional_mean",
                shape=shape,
                dtype="f4",
                chunks=(1,) + shape[1:],
                compression="lzf",
            )
            latent_ds = handle.create_dataset(
                "gaussianized_residual",
                shape=shape,
                dtype="f4",
                chunks=(1,) + shape[1:],
                compression="lzf",
            )
            for index in range(len(source["centered_residual"])):
                mean_ds[index] = source["conditional_mean"][index]
                latent = gaussianize_numpy(
                    source["centered_residual"][index], transform
                )
                latent_ds[index] = latent
                latent64 = latent.astype(np.float64)
                square_sum += float(np.square(latent64).sum())
                value_count += latent.size
                if (index + 1) % 16 == 0 or index + 1 == len(latent_ds):
                    print(f"[prepare-v12] {index + 1}/{len(latent_ds)}", flush=True)
            latent_rms = math.sqrt(square_sum / value_count)
            handle.attrs.update(
                {
                    "schema": CACHE_SCHEMA,
                    "source_v11_cache": str(Path(args.v11_cache).resolve()),
                    "source_data": source.attrs["source_data"],
                    "correction_checkpoint": source.attrs["correction_checkpoint"],
                    "input_preprocessing": source.attrs["input_preprocessing"],
                    "transform": str(Path(args.transform).resolve()),
                    "transform_sha256": file_sha256(args.transform),
                    "latent_rms": latent_rms,
                    "residual_rms": 1.0,
                    "target": "train-only rank-Gaussianized V11 centered residual",
                    "complete": True,
                }
            )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    print(json.dumps({"out": str(output), "latent_rms": latent_rms}, indent=2))


class V12ResidualDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        cache_path: str | Path,
        residual_scale: float,
        augment: bool,
    ) -> None:
        if not np.isclose(residual_scale, 1.0):
            raise ValueError("V12 latent residual scale is fixed to one")
        self.data_path = str(data_path)
        self.cache_path = str(cache_path)
        self.residual_scale = 1.0
        self.augment = bool(augment)
        self._data: h5py.File | None = None
        self._cache: h5py.File | None = None
        with h5py.File(self.data_path, "r") as data, h5py.File(
            self.cache_path, "r"
        ) as cache:
            if str(cache.attrs.get("schema")) != CACHE_SCHEMA:
                raise ValueError(f"not a V12 cache: {cache_path}")
            self.n = int(data["input"].shape[0])
            self.grid = int(data["input"].shape[-1])
            expected = data["target"].shape
            if cache["conditional_mean"].shape != expected:
                raise ValueError("data/cache mean shape mismatch")
            if cache["gaussianized_residual"].shape != expected:
                raise ValueError("data/cache latent shape mismatch")
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        data, cache = self._open()
        observable = apply_input_preprocessing(
            np.asarray(data["input"][index], dtype=np.float32), self.preprocessing
        )
        mean = np.asarray(cache["conditional_mean"][index], dtype=np.float32)
        latent = np.asarray(
            cache["gaussianized_residual"][index], dtype=np.float32
        )
        truth = np.asarray(data["target"][index], dtype=np.float32)
        condition = np.concatenate((observable, mean, self.radial), axis=0)
        if self.augment:
            permutation, reflections = CUBE_ISOMETRIES[
                int(np.random.randint(len(CUBE_ISOMETRIES)))
            ]
            joined = apply_cube_isometry(
                np.concatenate((condition, latent, truth), axis=0),
                permutation,
                reflections,
            )
            channels = condition.shape[0]
            condition = joined[:channels]
            latent = joined[channels : channels + 1]
            truth = joined[channels + 1 :]
            mean = condition[2:3]
        return tuple(
            torch.from_numpy(np.ascontiguousarray(value))
            for value in (condition, latent, mean, truth)
        )


def initialize_v11_parent(
    model: ObservableContextUNet, parent_state: dict[str, torch.Tensor]
) -> dict[str, list[str]]:
    excluded = {"context_mean", "context_std"}
    state = {key: value for key, value in parent_state.items() if key not in excluded}
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) != excluded or result.unexpected_keys:
        raise ValueError(
            f"incompatible V11 parent: {result.missing_keys}, {result.unexpected_keys}"
        )
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
        "retained_new_buffers": sorted(excluded),
    }


def common_cache_metadata(key: str, *paths: str | Path) -> str:
    values = set()
    for path in paths:
        with h5py.File(path, "r") as handle:
            if str(handle.attrs.get("schema")) != CACHE_SCHEMA:
                raise ValueError(f"not a V12 cache: {path}")
            values.add(str(handle.attrs[key]))
    if len(values) != 1:
        raise ValueError(f"V12 caches disagree on {key}")
    return values.pop()


def train(args: argparse.Namespace) -> None:
    parent = torch.load(args.initialize, map_location="cpu", weights_only=False)
    if parent.get("schema") != V11_SCHEMA:
        raise ValueError("V12 must initialize from a frozen V11 checkpoint")
    paths = (
        args.tng_train_cache,
        args.simba_train_cache,
        args.tng_validation_cache,
        args.simba_validation_cache,
    )
    transform_path = common_cache_metadata("transform", *paths)
    transform = load_transform(transform_path)
    train_context_edm(
        args,
        dataset_type=V12ResidualDataset,
        run_schema=SCHEMA,
        residual_scale_override=1.0,
        require_parent_scale=False,
        parent_initializer=initialize_v11_parent,
        metadata_extra={
            "residual_target": "train-only Gaussianized V11 full-band residual",
            "gaussianization": transform,
            "gaussianization_path": transform_path,
            "gaussianization_sha256": file_sha256(transform_path),
            "likelihood_weighting": "uniform EDM likelihood in Gaussianized coordinate",
            "correction_checkpoint": common_cache_metadata(
                "correction_checkpoint", *paths
            ),
            "inverse_exact_cube_dc_projection": True,
        },
    )


@torch.inference_mode()
def sample(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != SCHEMA:
        raise ValueError(f"not a V12 checkpoint: {checkpoint.get('schema')}")
    transform = checkpoint["gaussianization"]
    features = checkpoint["observable_context_features"]
    model = ObservableContextUNet(
        base_channels=int(checkpoint["base_channels"]),
        context_mean=features["mean"],
        context_std=features["std"],
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval().to(device)
    z_knots = torch.tensor(transform["z_knots"], device=device, dtype=torch.float32)
    residual_knots = torch.tensor(
        transform["residual_value_knots"], device=device, dtype=torch.float32
    )

    correction_model: ConditionalResidualUNet | None = None
    if args.cache is not None:
        if args.original_cache is not None or args.correction_checkpoint is not None:
            raise ValueError("use either --cache or the on-the-fly correction pair")
        dataset: Dataset = V12ResidualDataset(args.data, args.cache, 1.0, False)
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
            raise ValueError("not a V10 correction checkpoint")
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
    grid = dataset.grid  # type: ignore[attr-defined]
    try:
        with h5py.File(temporary, "w") as handle:
            samples_ds = handle.create_dataset(
                "sample",
                shape=(len(indices), args.ensemble, 1, grid, grid, grid),
                dtype="f4",
                chunks=(1, 1, 1, grid, grid, grid),
                compression="lzf",
            )
            mean_ds = handle.create_dataset(
                "conditional_mean",
                shape=(len(indices), 1, grid, grid, grid),
                dtype="f4",
                compression="lzf",
            )
            truth_ds = handle.create_dataset(
                "truth",
                shape=(len(indices), 1, grid, grid, grid),
                dtype="f4",
                compression="lzf",
            )
            handle.create_dataset("source_index", data=np.asarray(indices, dtype=np.int64))
            for output_index, data_index in enumerate(indices):
                condition, _, mean, truth = dataset[data_index]
                condition_single = condition[None].to(device)
                if correction_model is not None:
                    with torch.autocast(
                        device_type=device.type, enabled=device.type == "cuda"
                    ):
                        correction = correction_forward(
                            correction_model, condition_single
                        )
                    corrected = mean[None].to(device) + correction.to(
                        condition_single.dtype
                    )
                    condition_single = condition_single.clone()
                    condition_single[:, 2:3] = corrected
                    mean = corrected[0].cpu()
                condition_batch = condition_single.expand(
                    args.ensemble, -1, -1, -1, -1
                )
                latent = sample_edm(
                    model,
                    condition_batch,
                    generator,
                    args.sampling_steps,
                    args.sigma_min,
                    args.sigma_max,
                    args.rho,
                    float(checkpoint["sigma_data"]),
                )
                residual = inverse_gaussianize_torch(
                    latent, z_knots, residual_knots
                )
                residual -= residual.mean(dim=(-3, -2, -1), keepdim=True)
                generated = mean[None].to(device) + residual
                samples_ds[output_index] = generated.cpu().numpy()
                mean_ds[output_index] = mean.numpy()
                truth_ds[output_index] = truth.numpy()
                print(f"[sample] V12 EDM {output_index + 1}/{len(indices)}", flush=True)
            handle.attrs.update(
                {
                    "schema": SCHEMA,
                    "method": "edm",
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "checkpoint_step": int(checkpoint["step"]),
                    "source_cache": str(Path(source_cache).resolve()),
                    "sampling_steps": args.sampling_steps,
                    "seed": args.seed,
                    "diagnostic_k_h_mpc": 0.3,
                    "residual_operator": (
                        "inverse train-only Gaussianization, then exact cube DC projection"
                    ),
                    "gaussianization_sha256": checkpoint["gaussianization_sha256"],
                    "correction_checkpoint": checkpoint["correction_checkpoint"],
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
    parser.add_argument("--validation-seed", type=int, default=118173)
    parser.add_argument("--sigma-data", type=float, default=1.0)
    parser.add_argument("--edm-p-mean", type=float, default=-0.8)
    parser.add_argument("--edm-p-std", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=8021)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-limit", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fitting = sub.add_parser("fit-transform")
    fitting.add_argument("--tng-train-cache", required=True)
    fitting.add_argument("--simba-train-cache", required=True)
    fitting.add_argument("--out", required=True)
    fitting.add_argument("--histogram-bins", type=int, default=131_072)
    fitting.add_argument("--knots", type=int, default=8193)
    fitting.add_argument("--z-limit", type=float, default=5.0)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--v11-cache", required=True)
    prepare.add_argument("--transform", required=True)
    prepare.add_argument("--out", required=True)
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
    sampling.add_argument("--seed", type=int, default=23777)
    sampling.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    {
        "fit-transform": fit_transform,
        "prepare": prepare_cache,
        "train": train,
        "sample": sample,
    }[args.mode](args)


if __name__ == "__main__":
    main()
