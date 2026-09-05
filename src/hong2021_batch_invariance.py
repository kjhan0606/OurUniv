#!/usr/bin/env python
"""Separate normalization coupling from CUDA batch-shape roundoff."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hong2021_evaluate import load_checkpoint_model
from hong2021_train import AugmentedH5Dataset


def predict(
    model: torch.nn.Module,
    dataset: AugmentedH5Dataset,
    order: list[int],
    batch: int,
    device: torch.device,
) -> np.ndarray:
    by_index: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for offset in range(0, len(order), batch):
            indices = order[offset : offset + batch]
            x = torch.stack([dataset[index][0] for index in indices]).to(device)
            values = model(x).cpu().numpy()
            by_index.update(zip(indices, values, strict=True))
    return np.stack([by_index[index] for index in sorted(order)])


def difference(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    absolute = np.abs(delta)
    reference_rms = float(np.sqrt(np.mean(np.square(first, dtype=np.float64))))
    rms = float(np.sqrt(np.mean(np.square(delta))))
    return {
        "max_abs": float(absolute.max()),
        "p99_abs": float(np.percentile(absolute, 99)),
        "rms": rms,
        "rms_over_reference_rms": rms / reference_rms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.samples <= 0 or args.batch <= 1 or args.samples % args.batch:
        raise SystemExit("samples must be positive and divisible by batch > 1")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    dataset = AugmentedH5Dataset(args.data, augment=False)
    if args.samples > len(dataset):
        raise SystemExit("samples exceeds dataset length")
    natural = list(range(args.samples))
    reversed_within_batch = [
        value
        for offset in range(0, args.samples, args.batch)
        for value in reversed(natural[offset : offset + args.batch])
    ]

    model.eval()
    batch_1 = predict(model, dataset, natural, 1, device)
    batch_n = predict(model, dataset, natural, args.batch, device)
    batch_n_repeat = predict(model, dataset, natural, args.batch, device)
    batch_n_reordered = predict(
        model, dataset, reversed_within_batch, args.batch, device
    )
    model.train()
    train_mode = predict(model, dataset, natural, args.batch, device)

    report: dict[str, Any] = {
        "schema": "hong2021-batch-invariance-v1",
        "checkpoint": checkpoint,
        "data": str(args.data),
        "samples": args.samples,
        "batch": args.batch,
        "device": str(device),
        "torch": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "comparisons": {
            "batch_1_vs_batch_n": difference(batch_1, batch_n),
            "batch_n_repeat": difference(batch_n, batch_n_repeat),
            "batch_n_reordered": difference(batch_n, batch_n_reordered),
            "eval_vs_train_mode_batch_n": difference(batch_n, train_mode),
        },
        "interpretation": (
            "A batch-1/N difference with zero same-shape reorder and repeat "
            "differences is CUDA batch-shape numerical variation, not "
            "cross-sample normalization coupling."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
