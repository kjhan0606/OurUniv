#!/usr/bin/env python
"""Train the published Hong et al. (2021) TNG100 network.

Defaults reproduce the paper: 432 unaugmented training cubes, 93 validation
cubes, 3 cyclic axis permutations x 8 flips, batch 6, 200 epochs, MSE, and Adam
(lr=1e-3, beta=(0.9,0.999), eps=1e-7).  The input is deliberately *not*
standardized: the paper feeds galaxy counts and mean radial velocity in km/s.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from hong2021_data import inspect_training_file
from hong2021_model import Hong2021Net, PAPER_CHANNELS, parameter_count


INPUT_MODES = {
    "faithful": (2, "Ngal,mean_radial_vpec_kms"),
    "kinematic3": (
        3,
        "Ngal,mean_radial_vpec_kms,"
        "intrinsic_within_voxel_radial_velocity_dispersion_kms",
    ),
    "uncertainty3": (
        3,
        "Ngal,weighted_mean_radial_vpec_kms,sigma_mean_radial_vpec_kms",
    ),
    "uncertainty4": (
        4,
        "Ngal,weighted_mean_radial_vpec_kms,sigma_mean_radial_vpec_kms,"
        "within_voxel_velocity_dispersion_kms",
    ),
}


def input_preprocessing_spec(path: str | Path, mode: str) -> dict[str, object]:
    if mode == "faithful":
        return {"mode": "faithful", "schema": "hong2021-input-preprocessing-v1"}
    if mode != "occupied_standardized":
        raise ValueError(f"unknown input preprocessing: {mode}")
    count_sum = count_square_sum = velocity_sum = velocity_square_sum = 0.0
    occupied_cells = 0
    with h5py.File(path, "r") as handle:
        data = handle["input"]
        if data.shape[1] != 2:
            raise ValueError("occupied_standardized currently requires two channels")
        for start in range(0, len(data), 8):
            value = np.asarray(data[start : start + 8], dtype=np.float64)
            count = value[:, 0]
            velocity = value[:, 1]
            occupied = count > 0
            selected_count = count[occupied]
            selected_velocity = velocity[occupied]
            occupied_cells += int(occupied.sum())
            count_sum += float(selected_count.sum())
            count_square_sum += float(np.square(selected_count).sum())
            velocity_sum += float(selected_velocity.sum())
            velocity_square_sum += float(np.square(selected_velocity).sum())
    if occupied_cells < 2:
        raise ValueError("not enough occupied cells to estimate preprocessing")
    count_mean = count_sum / occupied_cells
    velocity_mean = velocity_sum / occupied_cells
    count_std = np.sqrt(count_square_sum / occupied_cells - count_mean**2)
    velocity_std = np.sqrt(
        velocity_square_sum / occupied_cells - velocity_mean**2
    )
    if count_std <= 0 or velocity_std <= 0:
        raise ValueError("invalid zero preprocessing scale")
    return {
        "mode": mode,
        "schema": "hong2021-input-preprocessing-v1",
        "fit_file": str(path),
        "occupied_cells": occupied_cells,
        "count_occupied_mean": count_mean,
        "count_scale": float(count_std),
        "count_centered": False,
        "velocity_occupied_mean_kms": velocity_mean,
        "velocity_occupied_std_kms": float(velocity_std),
        "empty_cells_preserved_as_zero": True,
    }


def apply_input_preprocessing(
    value: np.ndarray, spec: dict[str, object]
) -> np.ndarray:
    if spec["mode"] == "faithful":
        return value
    if spec["mode"] != "occupied_standardized":
        raise ValueError(f"unknown input preprocessing: {spec['mode']}")
    output = np.array(value, dtype=np.float32, copy=True)
    occupied = output[0] > 0
    output[0] /= float(spec["count_scale"])
    output[1, occupied] = (
        output[1, occupied] - float(spec["velocity_occupied_mean_kms"])
    ) / float(spec["velocity_occupied_std_kms"])
    output[1, ~occupied] = 0.0
    return output


class AugmentedH5Dataset(Dataset):
    """Lazy HDF5 dataset expanded by the paper's fixed 24 symmetries."""

    def __init__(
        self,
        path: str | Path,
        augment: bool,
        preprocessing: dict[str, object] | None = None,
    ) -> None:
        self.path = str(path)
        self.augment = augment
        self.factor = 24 if augment else 1
        self.preprocessing = preprocessing or input_preprocessing_spec(
            path, "faithful"
        )
        self._handle: h5py.File | None = None
        with h5py.File(self.path, "r") as handle:
            self.n = int(handle["input"].shape[0])

    def __len__(self) -> int:
        return self.n * self.factor

    def _open(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        base, transform = divmod(index, self.factor)
        handle = self._open()
        x = np.asarray(handle["input"][base], dtype=np.float32)
        y = np.asarray(handle["target"][base], dtype=np.float32)
        x = apply_input_preprocessing(x, self.preprocessing)
        if self.augment:
            permutation_index, flip_bits = divmod(transform, 8)
            spatial_axes = ((0, 1, 2), (1, 2, 0), (2, 0, 1))[permutation_index]
            axes = (0,) + tuple(1 + v for v in spatial_axes)
            x = np.transpose(x, axes)
            y = np.transpose(y, axes)
            for dim in range(3):
                if flip_bits & (1 << dim):
                    x = np.flip(x, axis=1 + dim)
                    y = np.flip(y, axis=1 + dim)
            x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
        return torch.from_numpy(x), torch.from_numpy(y)


def epoch_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = nn.functional.mse_loss(prediction, y)
            if training:
                loss.backward()
                optimizer.step()
            batch = x.shape[0]
            total_loss += float(loss.detach()) * batch
            total_samples += batch
    return total_loss / total_samples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    validation_loss: float,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "paper_channels": tuple(int(value) for value in model.channels),
            "normalization": model.normalization,
            "input_preprocessing": model.input_preprocessing,
            "lr_schedule": model.lr_schedule,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        temporary,
    )
    os.replace(temporary, path)


def replace_with_hardlink(source: Path, destination: Path) -> None:
    """Atomically point a named best checkpoint at the saved epoch.

    A paper-width checkpoint including Adam state is several GiB.  Writing the
    same state three times per epoch is unnecessary and would generate terabytes
    of avoidable I/O over 200 epochs.
    """
    temporary = destination.with_suffix(destination.suffix + ".linktmp")
    if temporary.exists():
        raise RuntimeError(f"stale checkpoint link exists: {temporary}")
    os.link(source, temporary)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--out", default="recon/hong2021/tng100_v1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--normalization",
        choices=("batch", "group"),
        default="batch",
        help="batch reproduces Hong; group is the batch-independent successor pilot",
    )
    parser.add_argument(
        "--input-preprocessing",
        choices=("faithful", "occupied_standardized"),
        default="faithful",
    )
    parser.add_argument(
        "--lr-schedule", choices=("constant", "cosine"), default="constant"
    )
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from OUT/last_epoch.pt and its existing history.json",
    )
    parser.add_argument(
        "--input-mode",
        choices=tuple(INPUT_MODES),
        default="faithful",
        help="faithful is the paper; uncertainty3/4 are CF4 comparison models",
    )
    parser.add_argument(
        "--smoke-widths",
        default=None,
        help="non-science smoke test only, e.g. 4,8,16,32,64",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=None,
        help=(
            "limit each augmented split to this many samples; allowed only "
            "with --smoke-widths"
        ),
    )
    args = parser.parse_args()
    if args.smoke_limit is not None and args.smoke_widths is None:
        raise SystemExit("--smoke-limit requires --smoke-widths")
    if args.smoke_limit is not None and args.smoke_limit <= 0:
        raise SystemExit("--smoke-limit must be positive")

    input_channels, channel_label = INPUT_MODES[args.input_mode]
    reports = [
        inspect_training_file(args.train, input_channels, channel_label),
        inspect_training_file(args.validation, input_channels, channel_label),
    ]
    if not all(report["pass"] for report in reports):
        raise SystemExit("input validation failed:\n" + json.dumps(reports, indent=2))
    if args.smoke_widths is None and args.input_mode == "faithful":
        if reports[0]["n_unaugmented"] != 432 or reports[1]["n_unaugmented"] != 93:
            raise SystemExit("paper run requires exactly 432 train and 93 validation cubes")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    widths = (
        tuple(int(v) for v in args.smoke_widths.split(","))
        if args.smoke_widths
        else PAPER_CHANNELS
    )
    model = Hong2021Net(
        in_channels=input_channels,
        channels=widths,
        normalization=args.normalization,
    ).to(device)
    preprocessing = input_preprocessing_spec(args.train, args.input_preprocessing)
    model.input_preprocessing = preprocessing
    model.lr_schedule = {
        "name": args.lr_schedule,
        "initial_lr": args.lr,
        "minimum_lr": args.min_lr if args.lr_schedule == "cosine" else args.lr,
        "epochs": args.epochs,
    }
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1.0e-7
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
        if args.lr_schedule == "cosine"
        else None
    )
    train_set = AugmentedH5Dataset(
        args.train, augment=True, preprocessing=preprocessing
    )
    validation_set = AugmentedH5Dataset(
        args.validation, augment=True, preprocessing=preprocessing
    )
    full_train_samples = len(train_set)
    full_validation_samples = len(validation_set)
    if args.smoke_limit is not None:
        train_set = Subset(
            train_set, range(min(args.smoke_limit, len(train_set)))
        )
        validation_set = Subset(
            validation_set,
            range(min(args.smoke_limit, len(validation_set))),
        )
    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "batch_size": args.batch,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_set, shuffle=True, generator=generator, drop_last=False, **common
    )
    validation_loader = DataLoader(
        validation_set, shuffle=False, drop_last=False, **common
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "faithful_paper_run": (
            args.smoke_widths is None
            and args.input_mode == "faithful"
            and args.normalization == "batch"
            and args.input_preprocessing == "faithful"
            and args.lr_schedule == "constant"
        ),
        "input_mode": args.input_mode,
        "input_channels": input_channels,
        "channels": widths,
        "normalization": args.normalization,
        "input_preprocessing": preprocessing,
        "parameters": parameter_count(model),
        "train_samples_unaugmented": reports[0]["n_unaugmented"],
        "validation_samples_unaugmented": reports[1]["n_unaugmented"],
        "train_samples_augmented": len(train_set),
        "validation_samples_augmented": len(validation_set),
        "full_train_samples_augmented": full_train_samples,
        "full_validation_samples_augmented": full_validation_samples,
        "batch": args.batch,
        "epochs": args.epochs,
        "optimizer": {
            "name": "Adam",
            "lr": args.lr,
            "betas": [0.9, 0.999],
            "eps": 1e-7,
        },
        "lr_schedule": model.lr_schedule,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "resume_supported": True,
    }
    if not args.resume and (out / "last_epoch.pt").exists():
        raise SystemExit(
            f"{out}/last_epoch.pt already exists; use --resume or a new --out"
        )
    if not args.resume:
        (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)

    best_train = float("inf")
    best_validation = float("inf")
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    if args.resume:
        checkpoint_path = out / "last_epoch.pt"
        history_path = out / "history.json"
        if not checkpoint_path.is_file() or not history_path.is_file():
            raise SystemExit(
                "--resume requires existing last_epoch.pt and history.json"
            )
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        checkpoint_normalization = checkpoint.get("normalization", "batch")
        if checkpoint_normalization != args.normalization:
            raise SystemExit(
                "checkpoint normalization does not match --normalization: "
                f"{checkpoint_normalization} != {args.normalization}"
            )
        checkpoint_preprocessing = checkpoint.get(
            "input_preprocessing",
            {"mode": "faithful", "schema": "hong2021-input-preprocessing-v1"},
        )
        if checkpoint_preprocessing != preprocessing:
            raise SystemExit("checkpoint input preprocessing does not match this run")
        if checkpoint.get("lr_schedule", model.lr_schedule) != model.lr_schedule:
            raise SystemExit("checkpoint learning-rate schedule does not match this run")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            if checkpoint.get("scheduler") is None:
                raise SystemExit("cosine resume checkpoint has no scheduler state")
            scheduler.load_state_dict(checkpoint["scheduler"])
        history = json.loads(history_path.read_text())
        if not history or int(history[-1]["epoch"]) != int(checkpoint["epoch"]):
            raise SystemExit("checkpoint epoch and history.json disagree")
        start_epoch = int(checkpoint["epoch"]) + 1
        best_train = min(float(row["train_loss"]) for row in history)
        best_validation = min(
            float(row["validation_loss"]) for row in history
        )
        print(
            f"[resume] epoch={start_epoch} best_train={best_train:.7f} "
            f"best_validation={best_validation:.7f}",
            flush=True,
        )
    if start_epoch > args.epochs:
        print(
            f"[complete] checkpoint already reached epoch {args.epochs}",
            flush=True,
        )
        return
    start = time.time()
    prior_elapsed = (
        float(history[-1]["elapsed_seconds"]) if history else 0.0
    )
    for epoch in range(start_epoch, args.epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = epoch_loss(model, train_loader, device, optimizer)
        validation_loss = epoch_loss(model, validation_loader, device, None)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": learning_rate,
            "elapsed_seconds": prior_elapsed + time.time() - start,
        }
        history.append(row)
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        new_best_train = train_loss < best_train
        new_best_validation = validation_loss < best_validation
        if new_best_train:
            best_train = train_loss
        if new_best_validation:
            best_validation = validation_loss
        if scheduler is not None:
            scheduler.step()
        # Serialize once, then create atomic hard-link aliases for any best
        # checkpoint attained at this epoch.
        save_checkpoint(
            out / "last_epoch.pt",
            model,
            optimizer,
            epoch,
            train_loss,
            validation_loss,
            scheduler,
        )
        if new_best_train:
            replace_with_hardlink(
                out / "last_epoch.pt", out / "minimum_training_loss.pt"
            )
        if new_best_validation:
            replace_with_hardlink(
                out / "last_epoch.pt", out / "minimum_validation_loss.pt"
            )
        print(
            f"epoch={epoch:03d} train={train_loss:.7f} "
            f"validation={validation_loss:.7f} elapsed={row['elapsed_seconds']:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
