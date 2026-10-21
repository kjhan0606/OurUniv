#!/usr/bin/env python
"""Audit BatchNorm training/inference fidelity of Hong TNG100 checkpoints.

The Hong network contains BatchNorm at every scale.  A low loss measured in
``model.train()`` is not sufficient if the saved running statistics yield a
different function in ``model.eval()``.  This audit compares saved inference,
mini-batch statistics, and PyTorch's cumulative batch-moment recalibration on
the unaugmented train and validation cubes.  The latter is not an exact pooled
population variance because it does not include between-batch mean variance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from hong2021_model import Hong2021Net, PAPER_CHANNELS
from hong2021_train import AugmentedH5Dataset


def loader(
    path: Path,
    batch: int,
    workers: int,
    limit: int | None = None,
    preprocessing: dict[str, object] | None = None,
) -> DataLoader:
    dataset: Any = AugmentedH5Dataset(
        path, augment=False, preprocessing=preprocessing
    )
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def evaluate(model: nn.Module, data: DataLoader, device: torch.device) -> dict[str, Any]:
    sample_mse: list[float] = []
    error_sum = 0.0
    voxel_count = 0
    sum_truth = sum_prediction = 0.0
    sum_truth2 = sum_prediction2 = sum_product = 0.0
    saturated_low = saturated_high = 0
    with torch.inference_mode():
        for x, truth in data:
            prediction = model(x.to(device, non_blocking=True)).cpu()
            error = prediction - truth
            sample_mse.extend(
                torch.mean(torch.square(error), dim=(1, 2, 3, 4)).numpy().tolist()
            )
            error_sum += float(torch.square(error).sum(dtype=torch.float64))
            voxel_count += truth.numel()
            t = truth.double()
            p = prediction.double()
            sum_truth += float(t.sum())
            sum_prediction += float(p.sum())
            sum_truth2 += float(torch.square(t).sum())
            sum_prediction2 += float(torch.square(p).sum())
            sum_product += float((t * p).sum())
            saturated_low += int((prediction <= -0.99).sum())
            saturated_high += int((prediction >= 0.99).sum())
    covariance = sum_product - sum_truth * sum_prediction / voxel_count
    variance_truth = sum_truth2 - sum_truth**2 / voxel_count
    variance_prediction = sum_prediction2 - sum_prediction**2 / voxel_count
    values = np.asarray(sample_mse, dtype=np.float64)
    return {
        "samples": int(values.size),
        "voxel_mse": error_sum / voxel_count,
        "voxel_pearson": covariance / np.sqrt(variance_truth * variance_prediction),
        "sample_mse_min_median_p84_max": np.percentile(
            values, [0, 50, 84, 100]
        ).tolist(),
        "sample_mse_above_0.05": int(np.sum(values > 0.05)),
        "saturation_fraction_low": saturated_low / voxel_count,
        "saturation_fraction_high": saturated_high / voxel_count,
    }


def first_predictions(
    model: nn.Module,
    path: Path,
    batch: int,
    workers: int,
    device: torch.device,
    samples: int = 12,
    preprocessing: dict[str, object] | None = None,
) -> np.ndarray:
    data = loader(
        path,
        batch,
        workers,
        limit=samples,
        preprocessing=preprocessing,
    )
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for x, _ in data:
            values.append(model(x.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(values)


def reset_and_recalibrate(
    model: nn.Module, data: DataLoader, device: torch.device
) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.reset_running_stats()
            module.momentum = None
    model.train()
    with torch.inference_mode():
        for x, _ in data:
            model(x.to(device, non_blocking=True))
    model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("normalization", "batch") != "batch":
        raise SystemExit("BatchNorm audit requires a BatchNorm checkpoint")
    channels = tuple(int(v) for v in checkpoint.get("paper_channels", PAPER_CHANNELS))
    model = Hong2021Net(channels=channels)
    model.load_state_dict(checkpoint["model"])
    checkpoint_metadata = {
        "path": str(args.checkpoint),
        "epoch": int(checkpoint["epoch"]),
        "train_loss": float(checkpoint["train_loss"]),
        "validation_loss": float(checkpoint["validation_loss"]),
    }
    del checkpoint
    model.to(device)

    report: dict[str, Any] = {
        "schema": "hong2021-batchnorm-audit-v1",
        "checkpoint": checkpoint_metadata,
        "batch": args.batch,
        "modes": {},
    }
    print("[audit] saved running statistics", flush=True)
    model.eval()
    report["modes"]["saved_eval"] = {
        "train": evaluate(model, loader(args.train, args.batch, args.workers), device),
        "validation": evaluate(
            model, loader(args.validation, args.batch, args.workers), device
        ),
    }
    prediction_1 = first_predictions(
        model, args.validation, 1, args.workers, device
    )
    prediction_6 = first_predictions(
        model, args.validation, 6, args.workers, device
    )
    report["saved_eval_batch_invariance_first_12"] = {
        "max_abs": float(np.max(np.abs(prediction_1 - prediction_6))),
        "rms": float(np.sqrt(np.mean(np.square(prediction_1 - prediction_6)))),
    }

    print("[audit] mini-batch statistics", flush=True)
    model.train()
    report["modes"]["batch_stats"] = {
        "train": evaluate(model, loader(args.train, args.batch, args.workers), device),
        "validation": evaluate(
            model, loader(args.validation, args.batch, args.workers), device
        ),
    }

    print("[audit] cumulative recalibration on unaugmented training cubes", flush=True)
    reset_and_recalibrate(
        model, loader(args.train, args.batch, args.workers), device
    )
    report["modes"]["recalibrated_eval"] = {
        "train": evaluate(model, loader(args.train, args.batch, args.workers), device),
        "validation": evaluate(
            model, loader(args.validation, args.batch, args.workers), device
        ),
    }
    report["diagnosis"] = {
        "saved_eval_to_logged_train_mse_ratio": (
            report["modes"]["saved_eval"]["train"]["voxel_mse"]
            / checkpoint_metadata["train_loss"]
        ),
        "batch_stats_to_logged_train_mse_ratio": (
            report["modes"]["batch_stats"]["train"]["voxel_mse"]
            / checkpoint_metadata["train_loss"]
        ),
        "inference_fidelity_pass": bool(
            abs(
                report["modes"]["saved_eval"]["train"]["voxel_mse"]
                / checkpoint_metadata["train_loss"]
                - 1.0
            )
            <= 0.1
            and report["modes"]["saved_eval"]["train"][
                "sample_mse_above_0.05"
            ]
            == 0
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
