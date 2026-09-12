#!/usr/bin/env python
"""Audit a GroupNorm Hong checkpoint without any batch-statistics fallback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from hong2021_bn_audit import evaluate, first_predictions
from hong2021_model import Hong2021Net, PAPER_CHANNELS
from hong2021_train import AugmentedH5Dataset


def loader(
    path: Path,
    augment: bool,
    batch: int,
    workers: int,
    preprocessing: dict[str, object],
) -> DataLoader:
    return DataLoader(
        AugmentedH5Dataset(
            path, augment=augment, preprocessing=preprocessing
        ),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


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

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    normalization = checkpoint.get("normalization", "batch")
    if normalization != "group":
        raise SystemExit("GroupNorm audit requires a GroupNorm checkpoint")
    channels = tuple(int(value) for value in checkpoint.get("paper_channels", PAPER_CHANNELS))
    preprocessing = checkpoint.get(
        "input_preprocessing",
        {"mode": "faithful", "schema": "hong2021-input-preprocessing-v1"},
    )
    model = Hong2021Net(channels=channels, normalization="group")
    model.load_state_dict(checkpoint["model"])
    metadata = {
        "path": str(args.checkpoint),
        "epoch": int(checkpoint["epoch"]),
        "train_loss": float(checkpoint["train_loss"]),
        "validation_loss": float(checkpoint["validation_loss"]),
    }
    del checkpoint
    device = torch.device(args.device)
    model.to(device)

    report: dict[str, Any] = {
        "schema": "hong2021-groupnorm-audit-v1",
        "checkpoint": metadata,
        "batch": args.batch,
        "normalization": "group",
        "modes": {},
    }
    model.eval()
    print("[audit] unaugmented train and validation", flush=True)
    report["modes"]["unaugmented_eval"] = {
        "train": evaluate(
            model,
            loader(
                args.train,
                False,
                args.batch,
                args.workers,
                preprocessing,
            ),
            device,
        ),
        "validation": evaluate(
            model,
            loader(
                args.validation,
                False,
                args.batch,
                args.workers,
                preprocessing,
            ),
            device,
        ),
    }
    print("[audit] exact 24-fold augmented train and validation", flush=True)
    report["modes"]["augmented_eval"] = {
        "train": evaluate(
            model,
            loader(
                args.train,
                True,
                args.batch,
                args.workers,
                preprocessing,
            ),
            device,
        ),
        "validation": evaluate(
            model,
            loader(
                args.validation,
                True,
                args.batch,
                args.workers,
                preprocessing,
            ),
            device,
        ),
    }

    model.eval()
    eval_batch_1 = first_predictions(
        model,
        args.validation,
        1,
        args.workers,
        device,
        preprocessing=preprocessing,
    )
    eval_batch_6 = first_predictions(
        model,
        args.validation,
        6,
        args.workers,
        device,
        preprocessing=preprocessing,
    )
    model.train()
    train_batch_1 = first_predictions(
        model,
        args.validation,
        1,
        args.workers,
        device,
        preprocessing=preprocessing,
    )
    report["invariance_first_12"] = {
        "batch_1_vs_6_max_abs": float(np.max(np.abs(eval_batch_1 - eval_batch_6))),
        "batch_1_vs_6_rms": float(
            np.sqrt(np.mean(np.square(eval_batch_1 - eval_batch_6)))
        ),
        "train_vs_eval_max_abs": float(
            np.max(np.abs(train_batch_1 - eval_batch_1))
        ),
        "train_vs_eval_rms": float(
            np.sqrt(np.mean(np.square(train_batch_1 - eval_batch_1)))
        ),
    }
    report["diagnosis"] = {
        "batch_independence_pass": bool(
            report["invariance_first_12"]["batch_1_vs_6_max_abs"] <= 1.0e-5
        ),
        "mode_independence_pass": bool(
            report["invariance_first_12"]["train_vs_eval_max_abs"] <= 1.0e-5
        ),
        "augmented_final_weight_train_to_logged_train_mse_ratio": (
            report["modes"]["augmented_eval"]["train"]["voxel_mse"]
            / metadata["train_loss"]
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
