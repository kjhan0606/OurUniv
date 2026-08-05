#!/usr/bin/env python
"""Prepare corrected and multiscale-standardized residual caches for V14."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from hong2021_residual_diffusion import ConditionalResidualUNet
from hong2021_residual_v8_context import FEATURE_NAMES, observable_context_features
from hong2021_v14_baseline_audit import (
    BAND_EDGES_H_MPC,
    centered_fourier_band_rms,
)
from hong2021_v14_mean_correction import (
    DOMAINS,
    SCHEMA as CORRECTION_SCHEMA,
    MeanCorrectionDataset,
    correction_forward,
)
from hong2021_v14_multiscale import standardize_residual


CORRECTED_SCHEMA = "hong2021-v14-corrected-residual-cache-v1"
STANDARDIZED_SCHEMA = "hong2021-v14-multiscale-standardized-residual-cache-v1"


def _summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "minimum": np.min(array, axis=0).tolist(),
        "maximum": np.max(array, axis=0).tolist(),
    }


def prepare_corrected(args: argparse.Namespace) -> dict[str, Any]:
    if args.domain not in DOMAINS:
        raise ValueError(f"not a V14 development domain: {args.domain}")
    checkpoint = torch.load(
        args.correction_checkpoint, map_location="cpu", weights_only=False
    )
    if checkpoint.get("schema") != CORRECTION_SCHEMA:
        raise ValueError("not a V14 mean-correction checkpoint")
    device = torch.device(args.device)
    model = ConditionalResidualUNet(
        base_channels=int(checkpoint["base_channels"])
    )
    model.load_state_dict(checkpoint["ema_model"])
    model.eval().to(device)
    dataset = MeanCorrectionDataset(args.data, args.baseline_cache, False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    with h5py.File(args.data, "r") as data:
        shape = tuple(data["target"].shape)
        voxel_mpc_h = float(data.attrs["voxel_mpc_h"])
    output = Path(args.out)
    partial = output.with_suffix(output.suffix + ".partial")
    report_path = output.with_suffix(".json")
    if any(path.exists() for path in (output, partial, report_path)):
        raise RuntimeError(f"refusing to overwrite corrected cache: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = len(dataset)
    offset = 0
    try:
        with h5py.File(partial, "w") as handle:
            mean_ds = handle.create_dataset(
                "conditional_mean", shape=shape, dtype="f4",
                chunks=(1,) + shape[1:], compression="lzf",
            )
            residual_ds = handle.create_dataset(
                "centered_residual", shape=shape, dtype="f4",
                chunks=(1,) + shape[1:], compression="lzf",
            )
            feature_ds = handle.create_dataset(
                "observable_context_features", shape=(samples, len(FEATURE_NAMES)), dtype="f4"
            )
            dc_ds = handle.create_dataset("residual_dc", shape=(samples,), dtype="f4")
            rms_ds = handle.create_dataset(
                "residual_centered_rms", shape=(samples,), dtype="f4"
            )
            band_ds = handle.create_dataset(
                "residual_band_rms", shape=(samples, len(BAND_EDGES_H_MPC) - 1), dtype="f4"
            )
            with torch.inference_mode():
                for condition, _, truth in loader:
                    condition_device = condition.to(device, non_blocking=True)
                    correction = correction_forward(model, condition_device).float().cpu()
                    corrected_mean = condition[:, 2:3] + correction
                    corrected_condition = condition.clone()
                    corrected_condition[:, 2:3] = corrected_mean
                    features = observable_context_features(corrected_condition).numpy()
                    residual = truth.numpy() - corrected_mean.numpy()
                    for local in range(len(condition)):
                        index = offset + local
                        location, band_rms, centered = centered_fourier_band_rms(
                            residual[local, 0], voxel_mpc_h=voxel_mpc_h
                        )
                        mean_ds[index] = corrected_mean[local].numpy()
                        residual_ds[index, 0] = centered
                        feature_ds[index] = features[local]
                        dc_ds[index] = location
                        rms_ds[index] = np.sqrt(
                            np.mean(centered.astype(np.float64) ** 2)
                        )
                        band_ds[index] = band_rms
                    offset += len(condition)
                    print(f"[corrected-cache] {args.domain} {offset}/{samples}", flush=True)
            if offset != samples:
                raise RuntimeError("corrected cache sample count is incomplete")
            handle.attrs.update(
                {
                    "schema": CORRECTED_SCHEMA,
                    "domain": args.domain,
                    "source_data": str(Path(args.data).resolve()),
                    "baseline_cache": str(Path(args.baseline_cache).resolve()),
                    "correction_checkpoint": str(Path(args.correction_checkpoint).resolve()),
                    "correction_step": int(checkpoint["step"]),
                    "input_preprocessing": json.dumps(dataset.preprocessing),
                    "feature_names": json.dumps(list(FEATURE_NAMES)),
                    "band_edges_h_mpc": json.dumps(BAND_EDGES_H_MPC),
                    "voxel_mpc_h": voxel_mpc_h,
                    "feature_uses_target": False,
                    "correction_exact_zero_dc": True,
                    "complete": True,
                }
            )
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    with h5py.File(output, "r") as handle:
        report = {
            "schema": CORRECTED_SCHEMA,
            "domain": args.domain,
            "output": str(output.resolve()),
            "samples": samples,
            "residual_dc": _summary(handle["residual_dc"][:]),
            "residual_centered_rms": _summary(handle["residual_centered_rms"][:]),
            "residual_band_rms": _summary(handle["residual_band_rms"][:]),
            "feature_uses_target": False,
        }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def prepare_standardized(args: argparse.Namespace) -> dict[str, Any]:
    from hong2021_v14_location_scale import load_model, predict_location_scales

    model = load_model(args.location_scale_model)
    source = Path(args.corrected_cache)
    output = Path(args.out)
    partial = output.with_suffix(output.suffix + ".partial")
    report_path = output.with_suffix(".json")
    if any(path.exists() for path in (output, partial, report_path)):
        raise RuntimeError(f"refusing to overwrite standardized cache: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    square_sum = 0.0
    value_count = 0
    try:
        with h5py.File(source, "r") as old, h5py.File(partial, "w") as new:
            if str(old.attrs.get("schema")) != CORRECTED_SCHEMA:
                raise ValueError("not a V14 corrected residual cache")
            if bool(old.attrs.get("feature_uses_target", True)):
                raise ValueError("corrected cache context is not target-free")
            residual = old["centered_residual"]
            shape = tuple(residual.shape)
            voxel_mpc_h = float(old.attrs["voxel_mpc_h"])
            standardized_ds = new.create_dataset(
                "standardized_residual", shape=shape, dtype="f4",
                chunks=(1,) + shape[1:], compression="lzf",
            )
            old.copy("conditional_mean", new)
            old.copy("observable_context_features", new)
            location_ds = new.create_dataset(
                "predicted_residual_dc", shape=(len(residual),), dtype="f4"
            )
            scale_ds = new.create_dataset(
                "predicted_band_scales", shape=(len(residual), len(BAND_EDGES_H_MPC) - 1), dtype="f4"
            )
            for start in range(0, len(residual), args.chunk):
                stop = min(start + args.chunk, len(residual))
                features = np.asarray(
                    old["observable_context_features"][start:stop], dtype=np.float64
                )
                locations, scales = predict_location_scales(features, model)
                for local, index in enumerate(range(start, stop)):
                    _, standardized = standardize_residual(
                        np.asarray(residual[index, 0], dtype=np.float64),
                        predicted_scales=scales[local],
                        voxel_mpc_h=voxel_mpc_h,
                    )
                    standardized = standardized.astype(np.float32)
                    standardized_ds[index, 0] = standardized
                    square_sum += float(
                        np.square(standardized.astype(np.float64)).sum()
                    )
                    value_count += standardized.size
                location_ds[start:stop] = locations
                scale_ds[start:stop] = scales
                print(f"[standardized-cache] {stop}/{len(residual)}", flush=True)
            rms = float(np.sqrt(square_sum / value_count))
            for key, value in old.attrs.items():
                new.attrs[key] = value
            new.attrs.update(
                {
                    "schema": STANDARDIZED_SCHEMA,
                    "corrected_cache": str(source.resolve()),
                    "location_scale_model": str(Path(args.location_scale_model).resolve()),
                    "standardized_residual_rms": rms,
                    "location_prediction_uses_target": False,
                    "scale_prediction_uses_target": False,
                    "complete": True,
                }
            )
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    report = {
        "schema": STANDARDIZED_SCHEMA,
        "output": str(output.resolve()),
        "samples": shape[0],
        "standardized_residual_rms": rms,
        "location_scale_model": str(Path(args.location_scale_model).resolve()),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    corrected = sub.add_parser("corrected")
    corrected.add_argument("--domain", choices=DOMAINS, required=True)
    corrected.add_argument("--data", required=True)
    corrected.add_argument("--baseline-cache", required=True)
    corrected.add_argument("--correction-checkpoint", required=True)
    corrected.add_argument("--out", required=True)
    corrected.add_argument("--batch", type=int, default=6)
    corrected.add_argument("--workers", type=int, default=1)
    corrected.add_argument("--device", default="cuda")
    standardized = sub.add_parser("standardized")
    standardized.add_argument("--corrected-cache", required=True)
    standardized.add_argument("--location-scale-model", required=True)
    standardized.add_argument("--out", required=True)
    standardized.add_argument("--chunk", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    {"corrected": prepare_corrected, "standardized": prepare_standardized}[args.mode](args)


if __name__ == "__main__":
    main()
