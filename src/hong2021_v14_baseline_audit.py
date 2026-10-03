#!/usr/bin/env python
"""Cache and diagnose the frozen deterministic mean on V14 CIC domains.

For every development cube this records the unchanged V4 conditional mean,
eight target-free observable context features, the residual DC, and centered
residual RMS in four fixed Fourier bands.  These diagnostics distinguish a
location error from conditional scale/shape errors before the exact V14 model
family is frozen.  The CLI rejects EAGLE and Astrid domains.
"""
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

from hong2021_residual_diffusion import (
    AugmentedH5Dataset,
    load_checkpoint_model,
    radial_geometry,
)
from hong2021_residual_v8_context import FEATURE_NAMES, observable_context_features


SCHEMA = "hong2021-v14-common-cic-baseline-mean-audit-v1"
BAND_EDGES_H_MPC = (0.0, 1.0, 3.0, 6.0, float("inf"))
ALLOWED_DOMAINS = ("TNG100", "CAMELS-SIMBA", "CAMELS-Swift-EAGLE")


def centered_fourier_band_rms(
    residual: np.ndarray,
    *,
    voxel_mpc_h: float,
    edges: tuple[float, ...] = BAND_EDGES_H_MPC,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return residual DC, centered RMS, and an exhaustive band-RMS split."""
    value = np.asarray(residual, dtype=np.float64)
    if value.ndim != 3 or len(set(value.shape)) != 1:
        raise ValueError(f"expected a cubic 3-D residual, got {value.shape}")
    if not np.isfinite(value).all() or voxel_mpc_h <= 0:
        raise ValueError("residual and voxel size must be finite")
    if len(edges) < 2 or edges[0] != 0 or not all(
        first < second for first, second in zip(edges[:-1], edges[1:], strict=True)
    ):
        raise ValueError("band edges must increase from zero")
    location = float(value.mean(dtype=np.float64))
    centered = value - location
    grid = value.shape[0]
    frequency = 2.0 * np.pi * np.fft.fftfreq(grid, d=voxel_mpc_h)
    radius = np.sqrt(
        frequency[:, None, None] ** 2
        + frequency[None, :, None] ** 2
        + frequency[None, None, :] ** 2
    )
    spectrum = np.fft.fftn(centered, norm="ortho")
    power = np.abs(spectrum) ** 2
    band_rms = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = (radius >= lower) & (radius < upper) & (radius > 0)
        band_rms.append(float(np.sqrt(power[selected].sum() / value.size)))
    centered_rms = float(np.sqrt(np.mean(centered**2)))
    if not np.isclose(
        np.square(band_rms).sum(), centered_rms**2, rtol=2e-12, atol=1e-15
    ):
        raise RuntimeError("Fourier bands do not exhaust centered residual power")
    return location, np.asarray(band_rms), centered.astype(np.float32)


def _summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "minimum": np.min(array, axis=0).tolist(),
        "maximum": np.max(array, axis=0).tolist(),
    }


def _prepare_allowed(
    args: argparse.Namespace, allowed_domains: tuple[str, ...]
) -> dict[str, Any]:
    if args.domain not in allowed_domains:
        raise ValueError(f"domain is not allowed for this V14 preparation: {args.domain}")
    output = Path(args.out)
    partial = output.with_suffix(output.suffix + ".partial")
    report_path = output.with_suffix(".json")
    if any(path.exists() for path in (output, partial, report_path)):
        raise RuntimeError(f"refusing to overwrite output for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, metadata = load_checkpoint_model(Path(args.checkpoint), device)
    preprocessing = metadata["input_preprocessing"]
    dataset = AugmentedH5Dataset(args.data, augment=False, preprocessing=preprocessing)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    with h5py.File(args.data, "r") as source:
        samples = len(source["target"])
        shape = source["target"].shape
        voxel_mpc_h = float(source.attrs["voxel_mpc_h"])
    radial = torch.from_numpy(radial_geometry(shape[-1])[None, None])
    rows = []
    offset = 0
    try:
        with h5py.File(partial, "w") as handle:
            mean_data = handle.create_dataset(
                "conditional_mean", shape=shape, dtype="f4", chunks=(1,) + shape[1:], compression="lzf"
            )
            residual_data = handle.create_dataset(
                "centered_residual", shape=shape, dtype="f4", chunks=(1,) + shape[1:], compression="lzf"
            )
            feature_data = handle.create_dataset(
                "observable_context_features", shape=(samples, len(FEATURE_NAMES)), dtype="f4"
            )
            dc_data = handle.create_dataset("residual_dc", shape=(samples,), dtype="f4")
            rms_data = handle.create_dataset("residual_centered_rms", shape=(samples,), dtype="f4")
            band_data = handle.create_dataset(
                "residual_band_rms", shape=(samples, len(BAND_EDGES_H_MPC) - 1), dtype="f4"
            )
            model.eval()
            with torch.inference_mode():
                for observable, truth in loader:
                    prediction = model(observable.to(device, non_blocking=True)).float().cpu()
                    radial_batch = radial.expand(len(observable), -1, -1, -1, -1)
                    condition = torch.cat((observable, prediction, radial_batch), dim=1)
                    features = observable_context_features(condition).numpy()
                    truth_numpy = truth.numpy()
                    prediction_numpy = prediction.numpy()
                    for local in range(len(observable)):
                        index = offset + local
                        residual = truth_numpy[local, 0] - prediction_numpy[local, 0]
                        location, band_rms, centered = centered_fourier_band_rms(
                            residual, voxel_mpc_h=voxel_mpc_h
                        )
                        centered_rms = float(np.sqrt(np.mean(centered.astype(np.float64) ** 2)))
                        mean_data[index] = prediction_numpy[local]
                        residual_data[index, 0] = centered
                        feature_data[index] = features[local]
                        dc_data[index] = location
                        rms_data[index] = centered_rms
                        band_data[index] = band_rms
                        rows.append(
                            {
                                "index": index,
                                "residual_dc": location,
                                "residual_centered_rms": centered_rms,
                                "residual_band_rms": band_rms.tolist(),
                            }
                        )
                    offset += len(observable)
                    print(f"[baseline-audit] {offset}/{samples}", flush=True)
            if offset != samples:
                raise RuntimeError("baseline audit sample count is incomplete")
            handle.attrs.update(
                {
                    "schema": SCHEMA,
                    "domain": args.domain,
                    "source_data": str(Path(args.data).resolve()),
                    "deterministic_checkpoint": str(Path(args.checkpoint).resolve()),
                    "deterministic_epoch": int(metadata["epoch"]),
                    "input_preprocessing": json.dumps(preprocessing),
                    "feature_names": json.dumps(list(FEATURE_NAMES)),
                    "band_edges_h_mpc": json.dumps(BAND_EDGES_H_MPC),
                    "feature_uses_target": False,
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
            "schema": SCHEMA,
            "domain": args.domain,
            "source_data": str(Path(args.data).resolve()),
            "output": str(output.resolve()),
            "samples": samples,
            "feature_uses_target": False,
            "observable_context_features": _summary(handle["observable_context_features"][:]),
            "residual_dc": _summary(handle["residual_dc"][:]),
            "residual_centered_rms": _summary(handle["residual_centered_rms"][:]),
            "residual_band_rms": _summary(handle["residual_band_rms"][:]),
            "rows": rows,
        }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare development caches; the public CLI remains development-only."""
    return _prepare_allowed(args, ALLOWED_DOMAINS)


def prepare_astrid_independent(args: argparse.Namespace) -> dict[str, Any]:
    """Prepare the already-unsealed Astrid baseline for the one-shot gate.

    Callers must verify the committed V14 seal before invoking this function.
    It is deliberately absent from this module's CLI.
    """
    return _prepare_allowed(args, ("CAMELS-Astrid",))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
