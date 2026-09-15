#!/usr/bin/env python
"""Source-balanced TNG+SIMBA fine-tuning of the accepted V6 EDM model.

The simulation identity is deliberately not supplied to the network because
no such label exists for CF4.  Every optimizer update contains equal numbers
of TNG and SIMBA samples.  The primary checkpoint minimizes the worse of the
two validation losses relative to their frozen step-zero values, preventing
one simulation from being improved by sacrificing the other.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from hong2021_residual_diffusion import ConditionalResidualUNet, worker_seed
from hong2021_residual_v6 import (
    V6ResidualDataset,
    atomic_save,
    cache_scale,
    fixed_validation_loss,
    method_loss,
    seed_everything,
    update_ema,
)


SCHEMA = "hong2021-conditional-laplacian-residual-v7-multidomain"


def cycling(loader: DataLoader) -> Iterator[Any]:
    while True:
        yield from loader


def make_loader(
    dataset: Dataset,
    batch: int,
    workers: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(seed),
    )


def validation_pair(
    model: nn.Module,
    tng_loader: DataLoader,
    simba_loader: DataLoader,
    device: torch.device,
    seed: int,
    sigma_data: float,
    edm_p_mean: float,
    edm_p_std: float,
) -> tuple[float, float]:
    tng = fixed_validation_loss(
        "edm", model, tng_loader, device, seed,
        sigma_data, edm_p_mean, edm_p_std,
    )
    simba = fixed_validation_loss(
        "edm", model, simba_loader, device, seed + 1,
        sigma_data, edm_p_mean, edm_p_std,
    )
    return tng, simba


def train(args: argparse.Namespace) -> None:
    if args.batch % 2:
        raise ValueError("source-balanced training requires an even --batch")
    seed_everything(args.seed)
    device = torch.device(args.device)
    source_checkpoint = torch.load(
        args.initialize, map_location="cpu", weights_only=False
    )
    if source_checkpoint["method"] != "edm":
        raise ValueError("V7 multi-domain training requires an EDM checkpoint")
    residual_scale = cache_scale(args.tng_train_cache)
    datasets: dict[str, Dataset] = {
        "tng_train": V6ResidualDataset(
            args.tng_train_data, args.tng_train_cache, residual_scale, augment=True
        ),
        "simba_train": V6ResidualDataset(
            args.simba_train_data, args.simba_train_cache, residual_scale, augment=True
        ),
        "tng_validation": V6ResidualDataset(
            args.tng_validation_data,
            args.tng_validation_cache,
            residual_scale,
            augment=False,
        ),
        "simba_validation": V6ResidualDataset(
            args.simba_validation_data,
            args.simba_validation_cache,
            residual_scale,
            augment=False,
        ),
    }
    if args.smoke_limit is not None:
        for key, dataset in list(datasets.items()):
            datasets[key] = Subset(
                dataset, range(min(args.smoke_limit, len(dataset)))
            )
    half_batch = args.batch // 2
    train_tng = make_loader(
        datasets["tng_train"], half_batch, args.workers, True, args.seed, device
    )
    train_simba = make_loader(
        datasets["simba_train"],
        half_batch,
        args.workers,
        True,
        args.seed + 1,
        device,
    )
    validation_tng = make_loader(
        datasets["tng_validation"], args.validation_batch, args.workers,
        False, args.seed + 2, device,
    )
    validation_simba = make_loader(
        datasets["simba_validation"], args.validation_batch, args.workers,
        False, args.seed + 3, device,
    )

    base_channels = int(source_checkpoint["base_channels"])
    model = ConditionalResidualUNet(base_channels=base_channels).to(device)
    model.load_state_dict(source_checkpoint["ema_model"])
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
    training_noise = torch.Generator(device=device).manual_seed(args.seed + 100)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists():
        raise SystemExit(f"refusing to overwrite existing run: {output}")

    initial_tng, initial_simba = validation_pair(
        ema_model, validation_tng, validation_simba, device,
        args.validation_seed, args.sigma_data, args.edm_p_mean, args.edm_p_std,
    )
    metadata = {
        "schema": SCHEMA,
        "status": "training",
        "method": "edm",
        "initialize": str(Path(args.initialize).resolve()),
        "initialize_step": int(source_checkpoint["step"]),
        "source_balance_per_batch": {"tng": half_batch, "simba": half_batch},
        "simulation_label_conditioning": False,
        "tng_train_data": args.tng_train_data,
        "tng_train_cache": args.tng_train_cache,
        "simba_train_data": args.simba_train_data,
        "simba_train_cache": args.simba_train_cache,
        "tng_validation_data": args.tng_validation_data,
        "tng_validation_cache": args.tng_validation_cache,
        "simba_validation_data": args.simba_validation_data,
        "simba_validation_cache": args.simba_validation_cache,
        "residual_scale": residual_scale,
        "base_channels": base_channels,
        "parameters": sum(value.numel() for value in model.parameters()),
        "steps": args.steps,
        "batch": args.batch,
        "examples_budget": args.steps * args.batch,
        "examples_per_source": args.steps * half_batch,
        "ema_decay": args.ema_decay,
        "fixed_validation_seed": args.validation_seed,
        "validation_every": args.validation_every,
        "initial_fixed_validation_loss_ema": {
            "tng": initial_tng,
            "simba": initial_simba,
        },
        "checkpoint_selection": (
            "minimum of max(TNG/initial_TNG, SIMBA/initial_SIMBA) fixed EMA "
            "validation loss"
        ),
        "sigma_data": args.sigma_data,
        "edm_p_mean": args.edm_p_mean,
        "edm_p_std": args.edm_p_std,
        "augmentation": "independent uniform random 48 cube isometries",
        "device": str(device),
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)

    tng_iterator = cycling(train_tng)
    simba_iterator = cycling(train_simba)
    history: list[dict[str, float | int]] = []
    best_worst_relative = float("inf")
    best_balanced = float("inf")
    interval_loss = 0.0
    interval_samples = 0
    start = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        tng_condition, tng_residual, _, _ = next(tng_iterator)
        simba_condition, simba_residual, _, _ = next(simba_iterator)
        condition = torch.cat((tng_condition, simba_condition), dim=0).to(
            device, non_blocking=True
        )
        residual = torch.cat((tng_residual, simba_residual), dim=0).to(
            device, non_blocking=True
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = method_loss(
                "edm", model, residual, condition, training_noise,
                args.sigma_data, args.edm_p_mean, args.edm_p_std,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        update_ema(ema_model, model, args.ema_decay)
        interval_loss += float(loss.detach()) * len(residual)
        interval_samples += len(residual)

        if step % args.validation_every == 0 or step == args.steps:
            validation_tng_loss, validation_simba_loss = validation_pair(
                ema_model, validation_tng, validation_simba, device,
                args.validation_seed, args.sigma_data,
                args.edm_p_mean, args.edm_p_std,
            )
            balanced = 0.5 * (validation_tng_loss + validation_simba_loss)
            worst_relative = max(
                validation_tng_loss / initial_tng,
                validation_simba_loss / initial_simba,
            )
            row = {
                "step": step,
                "examples_seen": step * args.batch,
                "examples_seen_per_source": step * half_batch,
                "train_loss_interval": interval_loss / interval_samples,
                "fixed_validation_loss_ema_tng": validation_tng_loss,
                "fixed_validation_loss_ema_simba": validation_simba_loss,
                "balanced_validation_loss_ema": balanced,
                "worst_relative_validation_loss_ema": worst_relative,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - start,
            }
            history.append(row)
            (output / "history.json").write_text(
                json.dumps(history, indent=2) + "\n"
            )
            checkpoint = {
                **metadata,
                "step": step,
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                **row,
            }
            atomic_save(checkpoint, output / "last.pt")
            if worst_relative < best_worst_relative:
                best_worst_relative = worst_relative
                atomic_save(checkpoint, output / "minimum_validation.pt")
            if balanced < best_balanced:
                best_balanced = balanced
                atomic_save(checkpoint, output / "minimum_balanced_validation.pt")
            print(
                f"step={step:06d} train={row['train_loss_interval']:.6f} "
                f"val_tng={validation_tng_loss:.6f} "
                f"val_simba={validation_simba_loss:.6f} "
                f"worst_rel={worst_relative:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"elapsed={row['elapsed_seconds']:.0f}s",
                flush=True,
            )
            interval_loss = 0.0
            interval_samples = 0
            model.train()
    metadata["status"] = "complete"
    metadata["best_worst_relative_validation_loss_ema"] = best_worst_relative
    metadata["best_balanced_validation_loss_ema"] = best_balanced
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", required=True)
    parser.add_argument("--tng-train-data", required=True)
    parser.add_argument("--tng-train-cache", required=True)
    parser.add_argument("--simba-train-data", required=True)
    parser.add_argument("--simba-train-cache", required=True)
    parser.add_argument("--tng-validation-data", required=True)
    parser.add_argument("--tng-validation-cache", required=True)
    parser.add_argument("--simba-validation-data", required=True)
    parser.add_argument("--simba-validation-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--validation-batch", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--min-lr", type=float, default=5.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--validation-every", type=int, default=500)
    parser.add_argument("--validation-seed", type=int, default=88173)
    parser.add_argument("--sigma-data", type=float, default=1.0)
    parser.add_argument("--edm-p-mean", type=float, default=-0.8)
    parser.add_argument("--edm-p-std", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=3021)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-limit", type=int, default=None)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
