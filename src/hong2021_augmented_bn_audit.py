#!/usr/bin/env python
"""Audit frozen Hong weights on the exact 24-fold augmented distribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from hong2021_bn_audit import evaluate
from hong2021_model import Hong2021Net, PAPER_CHANNELS
from hong2021_train import AugmentedH5Dataset


def make_loader(
    path: Path,
    augment: bool,
    batch: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        AugmentedH5Dataset(path, augment=augment),
        batch_size=batch,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def reset_and_augmented_recalibration(
    model: nn.Module,
    data: DataLoader,
    device: torch.device,
) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.reset_running_stats()
            module.momentum = None
    model.train()
    with torch.inference_mode():
        for batch_index, (x, _) in enumerate(data, start=1):
            model(x.to(device, non_blocking=True))
            if batch_index % 200 == 0:
                print(f"[recalibration] batches={batch_index}", flush=True)
    model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    channels = tuple(int(value) for value in checkpoint.get("paper_channels", PAPER_CHANNELS))
    if checkpoint.get("normalization", "batch") != "batch":
        raise SystemExit("augmented BN audit requires a BatchNorm checkpoint")
    state = checkpoint["model"]
    metadata = {
        "path": str(args.checkpoint),
        "epoch": int(checkpoint["epoch"]),
        "train_loss": float(checkpoint["train_loss"]),
        "validation_loss": float(checkpoint["validation_loss"]),
    }
    del checkpoint
    device = torch.device(args.device)
    model = Hong2021Net(channels=channels).to(device)

    def restore() -> None:
        model.load_state_dict(state)

    report: dict[str, Any] = {
        "schema": "hong2021-augmented-bn-audit-v1",
        "checkpoint": metadata,
        "batch": args.batch,
        "seed": args.seed,
        "augmentation_factor": 24,
        "modes": {},
    }
    print("[audit] saved eval on full augmented training set", flush=True)
    restore()
    model.eval()
    report["modes"]["saved_eval_augmented_ordered"] = evaluate(
        model,
        make_loader(args.train, True, args.batch, args.workers, False, args.seed),
        device,
    )

    print("[audit] batch statistics on shuffled augmented training set", flush=True)
    restore()
    model.train()
    report["modes"]["batch_stats_augmented_shuffled"] = evaluate(
        model,
        make_loader(args.train, True, args.batch, args.workers, True, args.seed),
        device,
    )

    print("[audit] batch statistics on ordered augmented training set", flush=True)
    restore()
    model.train()
    report["modes"]["batch_stats_augmented_ordered"] = evaluate(
        model,
        make_loader(args.train, True, args.batch, args.workers, False, args.seed),
        device,
    )

    print("[audit] recalibration on shuffled augmented training set", flush=True)
    restore()
    reset_and_augmented_recalibration(
        model,
        make_loader(args.train, True, args.batch, args.workers, True, args.seed),
        device,
    )
    report["modes"]["augmented_recalibrated_eval"] = {
        "train_unaugmented": evaluate(
            model,
            make_loader(args.train, False, args.batch, args.workers, False, args.seed),
            device,
        ),
        "validation_unaugmented": evaluate(
            model,
            make_loader(
                args.validation, False, args.batch, args.workers, False, args.seed
            ),
            device,
        ),
    }
    report["diagnosis"] = {
        "saved_eval_augmented_to_logged_train_mse_ratio": (
            report["modes"]["saved_eval_augmented_ordered"]["voxel_mse"]
            / metadata["train_loss"]
        ),
        "batch_stats_shuffled_to_logged_train_mse_ratio": (
            report["modes"]["batch_stats_augmented_shuffled"]["voxel_mse"]
            / metadata["train_loss"]
        ),
        "batch_stats_ordered_to_logged_train_mse_ratio": (
            report["modes"]["batch_stats_augmented_ordered"]["voxel_mse"]
            / metadata["train_loss"]
        ),
        "augmented_recalibration_fixed_train_inference": bool(
            abs(
                report["modes"]["augmented_recalibrated_eval"][
                    "train_unaugmented"
                ]["voxel_mse"]
                / metadata["train_loss"]
                - 1.0
            )
            <= 0.1
            and report["modes"]["augmented_recalibrated_eval"][
                "train_unaugmented"
            ]["sample_mse_above_0.05"]
            == 0
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
