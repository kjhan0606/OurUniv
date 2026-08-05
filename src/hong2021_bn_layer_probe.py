#!/usr/bin/env python
"""Localize late-epoch BatchNorm and batch-composition dependence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from hong2021_model import Hong2021Net, PAPER_CHANNELS
from hong2021_train import AugmentedH5Dataset


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    percentiles = np.percentile(values, [0, 50, 84, 95, 99, 100])
    return dict(
        zip(("min", "p50", "p84", "p95", "p99", "max"), percentiles.tolist())
    )


def per_sample_mse(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    values: list[np.ndarray] = []
    with torch.inference_mode():
        for x, truth in loader:
            prediction = model(x.to(device, non_blocking=True)).cpu()
            mse = torch.mean(torch.square(prediction - truth), dim=(1, 2, 3, 4))
            values.append(mse.numpy())
    return np.concatenate(values)


def one_target_mse(
    model: nn.Module,
    dataset: AugmentedH5Dataset,
    indices: list[int],
    target_position: int,
    device: torch.device,
) -> float:
    batch = [dataset[index] for index in indices]
    x = torch.stack([item[0] for item in batch]).to(device)
    truth = torch.stack([item[1] for item in batch])
    model.train()
    with torch.inference_mode():
        prediction = model(x).cpu()
    return float(
        torch.mean(torch.square(prediction[target_position] - truth[target_position]))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--companion-repeats", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7321)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    channels = tuple(int(value) for value in checkpoint.get("paper_channels", PAPER_CHANNELS))
    normalization = checkpoint.get("normalization", "batch")
    if normalization != "batch":
        raise SystemExit("BN layer probe requires a BatchNorm checkpoint")
    model = Hong2021Net(channels=channels)
    model.load_state_dict(checkpoint["model"])
    checkpoint_metadata = {
        "path": str(args.checkpoint),
        "epoch": int(checkpoint["epoch"]),
        "train_loss": float(checkpoint["train_loss"]),
        "validation_loss": float(checkpoint["validation_loss"]),
    }
    del checkpoint
    device = torch.device(args.device)
    model.to(device)
    dataset = AugmentedH5Dataset(args.train, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    model.eval()
    saved_eval_mse = per_sample_mse(model, loader, device)
    bn_layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm3d)
    }
    saved_statistics = {
        name: {
            "mean": module.running_mean.detach().cpu().numpy().copy(),
            "variance": module.running_var.detach().cpu().numpy().copy(),
            "epsilon": float(module.eps),
        }
        for name, module in bn_layers.items()
    }
    observed: dict[str, dict[str, Any]] = {
        name: {"mean": [], "variance": [], "effective_values": []}
        for name in bn_layers
    }
    hooks = []
    for name, module in bn_layers.items():
        def record(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            _name: str = name,
        ) -> None:
            value = inputs[0].detach().float()
            variance, mean = torch.var_mean(
                value, dim=(0, 2, 3, 4), unbiased=False
            )
            observed[_name]["mean"].append(mean.cpu().numpy())
            observed[_name]["variance"].append(variance.cpu().numpy())
            observed[_name]["effective_values"].append(
                int(value.shape[0] * np.prod(value.shape[2:]))
            )

        hooks.append(module.register_forward_pre_hook(record))

    print("[probe] collecting train-mode per-batch BN moments", flush=True)
    model.train()
    batch_stats_mse = per_sample_mse(model, loader, device)
    for hook in hooks:
        hook.remove()
    layer_report: dict[str, Any] = {}
    for name in bn_layers:
        batch_mean = np.stack(observed[name]["mean"]).astype(np.float64)
        batch_variance = np.stack(observed[name]["variance"]).astype(np.float64)
        saved_mean = saved_statistics[name]["mean"].astype(np.float64)
        saved_variance = saved_statistics[name]["variance"].astype(np.float64)
        epsilon = saved_statistics[name]["epsilon"]
        standardized_mean_offset = (
            batch_mean - saved_mean[None, :]
        ) / np.sqrt(saved_variance[None, :] + epsilon)
        variance_ratio = (batch_variance + epsilon) / (
            saved_variance[None, :] + epsilon
        )
        channel_batch_mean_dispersion = np.std(batch_mean, axis=0) / np.sqrt(
            saved_variance + epsilon
        )
        layer_report[name] = {
            "channels": int(batch_mean.shape[1]),
            "batches": int(batch_mean.shape[0]),
            "effective_values_per_channel": sorted(
                set(observed[name]["effective_values"])
            ),
            "absolute_standardized_batch_mean_offset": percentile_summary(
                np.abs(standardized_mean_offset).ravel()
            ),
            "batch_variance_over_saved_variance": percentile_summary(
                variance_ratio.ravel()
            ),
            "per_channel_batch_mean_dispersion_over_saved_sigma": percentile_summary(
                channel_batch_mean_dispersion
            ),
        }

    failing = np.flatnonzero(saved_eval_mse > 0.05)
    if not len(failing):
        failing = np.argsort(saved_eval_mse)[-min(11, len(dataset)) :]
        failing_reason = "no MSE>0.05 cubes; probing the 11 largest saved-eval MSEs"
    else:
        failing_reason = "saved-eval MSE>0.05"
    galaxy_count = np.empty(len(dataset), dtype=np.float64)
    for index in range(len(dataset)):
        x, _ = dataset[index]
        galaxy_count[index] = float(x[0].sum())
    order_by_count = np.argsort(galaxy_count)
    generator = np.random.default_rng(args.seed)
    companion_report: list[dict[str, Any]] = []
    all_indices = np.arange(len(dataset))
    for target_index in failing:
        available = all_indices[all_indices != target_index]
        random_mse = []
        reorder_difference = []
        for _ in range(args.companion_repeats):
            companions = generator.choice(available, args.batch - 1, replace=False)
            group = [int(target_index), *companions.astype(int).tolist()]
            first = one_target_mse(model, dataset, group, 0, device)
            reordered = [*group[1:], group[0]]
            last = one_target_mse(
                model, dataset, reordered, len(reordered) - 1, device
            )
            random_mse.append(first)
            reorder_difference.append(abs(first - last))
        nearest = available[np.argsort(np.abs(galaxy_count[available] - galaxy_count[target_index]))[: args.batch - 1]]
        lowest = [int(value) for value in order_by_count if value != target_index][
            : args.batch - 1
        ]
        highest = [int(value) for value in order_by_count[::-1] if value != target_index][
            : args.batch - 1
        ]
        companion_report.append(
            {
                "sample_index": int(target_index),
                "galaxy_count": galaxy_count[target_index],
                "saved_eval_mse": float(saved_eval_mse[target_index]),
                "sequential_batch_stats_mse": float(batch_stats_mse[target_index]),
                "batch1_mse": one_target_mse(
                    model, dataset, [int(target_index)], 0, device
                ),
                "random_companion_mse": percentile_summary(np.asarray(random_mse)),
                "same_batch_reordering_abs_mse_difference": percentile_summary(
                    np.asarray(reorder_difference)
                ),
                "nearest_ngal_companion_mse": one_target_mse(
                    model,
                    dataset,
                    [int(target_index), *nearest.astype(int).tolist()],
                    0,
                    device,
                ),
                "lowest_ngal_companion_mse": one_target_mse(
                    model, dataset, [int(target_index), *lowest], 0, device
                ),
                "highest_ngal_companion_mse": one_target_mse(
                    model, dataset, [int(target_index), *highest], 0, device
                ),
            }
        )
        print(f"[probe] companion sample={target_index}", flush=True)

    report = {
        "schema": "hong2021-bn-layer-composition-probe-v1",
        "checkpoint": checkpoint_metadata,
        "train_file": str(args.train),
        "unaugmented_samples": len(dataset),
        "batch": args.batch,
        "saved_eval": {
            "mse_mean": float(saved_eval_mse.mean()),
            "mse_min_median_max": np.percentile(saved_eval_mse, [0, 50, 100]).tolist(),
            "mse_above_0.05": int(np.sum(saved_eval_mse > 0.05)),
        },
        "batch_stats_sequential": {
            "mse_mean": float(batch_stats_mse.mean()),
            "mse_min_median_max": np.percentile(batch_stats_mse, [0, 50, 100]).tolist(),
            "mse_above_0.05": int(np.sum(batch_stats_mse > 0.05)),
        },
        "layers": layer_report,
        "companion_probe_selection": failing_reason,
        "companion_probe": companion_report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
