#!/usr/bin/env python
"""Evaluate Hong et al. (2021) checkpoints on held-out TNG100 cubes.

The paper selects among the minimum-validation-loss, minimum-training-loss,
and last-epoch checkpoints by comparing the *distribution* of the density
two-point correlation function (2pCF) over validation cubes.  This script
implements that gate on the 93 unaugmented, spatially independent validation
cubes and also reports the density residual used in Table 2 of the paper.

The paper does not specify how pairs crossing a sub-cube boundary are treated.
We use an open-boundary FFT estimator: fields are zero padded, every separation
is divided by its exact number of in-cube pairs, and no opposite faces wrap.
This convention is written into the output JSON so results remain reproducible.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
import torch
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from hong2021_data import inspect_training_file
from hong2021_model import Hong2021Net, PAPER_CHANNELS, parameter_count
from hong2021_train import AugmentedH5Dataset


DENSITY_SCALE = 4.5
PAPER_REFERENCE = {
    "log10_rho_pred_over_truth": {"mean": -0.014, "std": 0.543},
    "ks_2pcf": {
        "0-1_mpc_h": {"mean": 0.263, "std": 0.035},
        "1-3_mpc_h": {"mean": 0.175, "std": 0.087},
        "3-10_mpc_h": {"mean": 0.130, "std": 0.042},
    },
}
SCALE_RANGES = ((0.0, 1.0), (1.0, 3.0), (3.0, 10.0))


def safe_label(label: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_.")
    if not value:
        raise ValueError(f"invalid empty checkpoint label from {label!r}")
    return value


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be LABEL=PATH")
    label, path = value.split("=", 1)
    return safe_label(label), Path(path)


@dataclass
class OpenBoundaryTwoPoint:
    """Radially average ``<delta(x) delta(x+r)>`` without face wrapping."""

    grid: int
    voxel_mpc_h: float
    rmax_mpc_h: float

    def __post_init__(self) -> None:
        if self.grid <= 1 or self.voxel_mpc_h <= 0 or self.rmax_mpc_h <= 0:
            raise ValueError("grid, voxel size, and rmax must be positive")
        self.pad = 2 * self.grid
        index = np.arange(self.pad, dtype=np.int32)
        offset = np.where(index < self.grid, index, index - self.pad)
        overlap_1d = np.maximum(self.grid - np.abs(offset), 0).astype(np.float64)
        dx2 = np.square(offset, dtype=np.int64)
        radius = np.sqrt(
            dx2[:, None, None] + dx2[None, :, None] + dx2[None, None, :]
        ) * self.voxel_mpc_h
        overlap = (
            overlap_1d[:, None, None]
            * overlap_1d[None, :, None]
            * overlap_1d[None, None, :]
        )
        self.nbins = int(math.ceil(self.rmax_mpc_h / self.voxel_mpc_h))
        radial_bin = np.floor(radius / self.voxel_mpc_h).astype(np.int32)
        mask = (radius < self.rmax_mpc_h) & (overlap > 0)
        self.flat_index = np.flatnonzero(mask.ravel())
        self.radial_bin = radial_bin.ravel()[self.flat_index]
        self.overlap = overlap.ravel()[self.flat_index]
        self.pair_count = np.bincount(
            self.radial_bin, weights=self.overlap, minlength=self.nbins
        )
        self.radius_mpc_h = (
            np.arange(self.nbins, dtype=np.float64) + 0.5
        ) * self.voxel_mpc_h

    def __call__(self, delta: np.ndarray) -> np.ndarray:
        field = np.asarray(delta, dtype=np.float64)
        expected = (self.grid,) * 3
        if field.shape != expected:
            raise ValueError(f"expected field shape {expected}, got {field.shape}")
        axes = (0, 1, 2)
        fourier = np.fft.rfftn(field, s=(self.pad,) * 3, axes=axes)
        correlation = np.fft.irfftn(
            fourier * fourier.conjugate(), s=(self.pad,) * 3, axes=axes
        ).real
        pair_sum = np.bincount(
            self.radial_bin,
            weights=correlation.ravel()[self.flat_index],
            minlength=self.nbins,
        )
        return np.divide(
            pair_sum,
            self.pair_count,
            out=np.full(self.nbins, np.nan, dtype=np.float64),
            where=self.pair_count > 0,
        )


def ks_statistic(first: np.ndarray, second: np.ndarray) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or not len(a) or not len(b):
        raise ValueError("KS samples must be non-empty one-dimensional arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("KS samples must be finite")
    return float(ks_2samp(a, b, method="exact").statistic)


def scale_key(low: float, high: float) -> str:
    return f"{low:g}-{high:g}_mpc_h"


def summarize_ks(
    truth: np.ndarray, prediction: np.ndarray, radius_mpc_h: np.ndarray
) -> dict[str, Any]:
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("2pCF arrays must have matching (sample, radius) shape")
    ks_radius = np.array(
        [ks_statistic(prediction[:, i], truth[:, i]) for i in range(truth.shape[1])]
    )
    scale_summary: dict[str, dict[str, float | int]] = {}
    for low, high in SCALE_RANGES:
        selected = (radius_mpc_h >= low) & (radius_mpc_h < high)
        values = ks_radius[selected]
        scale_summary[scale_key(low, high)] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "n_radial_bins": int(values.size),
        }
    return {
        "radius_mpc_h": radius_mpc_h.tolist(),
        "ks_at_radius": ks_radius.tolist(),
        "by_scale": scale_summary,
        "mean_0_10_mpc_h": float(ks_radius.mean()),
    }


def density_two_point(
    normalized_log_density: np.ndarray, estimator: OpenBoundaryTwoPoint
) -> np.ndarray:
    density_ratio = np.power(10.0, DENSITY_SCALE * normalized_log_density)
    return estimator(density_ratio - 1.0)


def load_checkpoint_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[Hong2021Net, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    channels = tuple(int(v) for v in checkpoint.get("paper_channels", PAPER_CHANNELS))
    normalization = str(checkpoint.get("normalization", "batch"))
    input_preprocessing = checkpoint.get(
        "input_preprocessing",
        {"mode": "faithful", "schema": "hong2021-input-preprocessing-v1"},
    )
    model = Hong2021Net(
        in_channels=2,
        channels=channels,
        normalization=normalization,
    )
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    metadata = {
        "path": str(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "train_loss": float(checkpoint["train_loss"]),
        "validation_loss": float(checkpoint["validation_loss"]),
        "channels": list(channels),
        "normalization": normalization,
        "input_preprocessing": input_preprocessing,
        "parameters": parameter_count(model),
    }
    del checkpoint
    return model, metadata


def plot_report(
    path: Path,
    history_path: Path | None,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    truth_2pcf: np.ndarray,
    prediction_2pcf: dict[str, np.ndarray],
    radius: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    labels = list(predictions)
    figure, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)

    if history_path and history_path.is_file():
        history = json.loads(history_path.read_text())
        epoch = np.array([row["epoch"] for row in history])
        axes[0, 0].plot(epoch, [row["train_loss"] for row in history], label="train")
        axes[0, 0].plot(
            epoch, [row["validation_loss"] for row in history], label="validation"
        )
        for label in labels:
            axes[0, 0].axvline(
                metrics["candidates"][label]["checkpoint"]["epoch"],
                alpha=0.5,
                linestyle="--",
                label=label,
            )
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_xlabel("epoch")
        axes[0, 0].set_ylabel("MSE")
        axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Learning curve and candidates")

    bins = np.linspace(-2.5, 2.5, 151)
    for label, prediction in predictions.items():
        residual = DENSITY_SCALE * (prediction - target)
        axes[0, 1].hist(
            residual.ravel(), bins=bins, density=True, histtype="step", label=label
        )
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel(r"$\log_{10}(\rho_{pred}/\rho_{truth})$")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title("All validation voxels")

    truth_median = np.median(truth_2pcf, axis=0)
    axes[0, 2].plot(radius, truth_median, color="black", label="truth")
    for label, profiles in prediction_2pcf.items():
        axes[0, 2].plot(radius, np.median(profiles, axis=0), label=label)
    axes[0, 2].set_yscale("symlog", linthresh=1e-2)
    axes[0, 2].set_xlabel(r"$r\ [h^{-1}{\rm Mpc}]$")
    axes[0, 2].set_ylabel(r"$\xi(r)$")
    axes[0, 2].legend(fontsize=8)
    axes[0, 2].set_title("Median density 2pCF")

    for label in labels:
        ks = metrics["candidates"][label]["two_point"]["ks_at_radius"]
        axes[1, 0].plot(radius, ks, label=label)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xlabel(r"$r\ [h^{-1}{\rm Mpc}]$")
    axes[1, 0].set_ylabel("KS statistic")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("2pCF distribution KS")

    selected_label = metrics["selection"]["selected_label"]
    mse_per_sample = np.mean(
        np.square(predictions[selected_label] - target), axis=(1, 2, 3)
    )
    sample = int(np.argsort(mse_per_sample)[len(mse_per_sample) // 2])
    plane = target.shape[-1] // 2
    vmin, vmax = np.percentile(target[sample, :, :, plane], [1, 99])
    image = axes[1, 1].imshow(
        target[sample, :, :, plane].T,
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    figure.colorbar(image, ax=axes[1, 1], fraction=0.046)
    axes[1, 1].set_title(f"Truth y, sample {sample}")
    image = axes[1, 2].imshow(
        predictions[selected_label][sample, :, :, plane].T,
        origin="lower",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )
    figure.colorbar(image, ax=axes[1, 2], fraction=0.046)
    axes[1, 2].set_title(f"{selected_label} prediction y")
    figure.suptitle("Hong et al. (2021) TNG100 held-out validation", fontsize=15)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=parse_checkpoint,
        action="append",
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--inference-normalization",
        choices=("saved", "live_batch"),
        default="saved",
        help=(
            "saved uses the deployable fixed function; live_batch is a "
            "diagnostic that uses each inference mini-batch's BatchNorm moments"
        ),
    )
    parser.add_argument("--voxel-mpc-h", type=float, default=0.3125)
    parser.add_argument("--rmax-mpc-h", type=float, default=10.0)
    args = parser.parse_args()
    if args.batch <= 0 or args.workers < 0:
        raise SystemExit("batch must be positive and workers nonnegative")
    labels = [label for label, _ in args.checkpoint]
    if len(labels) != len(set(labels)):
        raise SystemExit("checkpoint labels must be unique")
    for _, path in args.checkpoint:
        if not path.is_file():
            raise SystemExit(f"checkpoint missing: {path}")

    report = inspect_training_file(args.validation, deep=False)
    if not report["pass"]:
        raise SystemExit("validation file failed inspection: " + json.dumps(report))
    grid = int(report["target_shape"][-1])
    estimator = OpenBoundaryTwoPoint(grid, args.voxel_mpc_h, args.rmax_mpc_h)
    args.out.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.validation, "r") as handle:
        target = np.asarray(handle["target"][:, 0], dtype=np.float32)
    n_samples = target.shape[0]
    print(f"[truth] samples={n_samples} grid={grid} computing 2pCF", flush=True)
    truth_2pcf = np.stack(
        [density_two_point(target[i], estimator) for i in range(n_samples)]
    )

    device = torch.device(args.device)
    predictions: dict[str, np.ndarray] = {}
    prediction_2pcf: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {
        "schema": "hong2021-tng100-heldout-evaluation-v1",
        "validation_file": str(args.validation),
        "samples": n_samples,
        "augmented": False,
        "inference_normalization": args.inference_normalization,
        "inference_batch": args.batch,
        "inference_shuffle": False,
        "grid": grid,
        "voxel_mpc_h": args.voxel_mpc_h,
        "density_definition": "rho/rho_mean = 10**(4.5*y); delta = rho/rho_mean - 1",
        "two_point_estimator": {
            "boundary": "open_no_face_wrapping",
            "implementation": "zero_padded_fft_with_exact_pair_count_normalization",
            "radial_bin_width_mpc_h": args.voxel_mpc_h,
            "rmax_mpc_h": args.rmax_mpc_h,
            "paper_boundary_convention_published": False,
        },
        "paper_table_2_reference": PAPER_REFERENCE,
        "candidates": {},
    }

    prediction_path = args.out / "predictions.h5"
    with h5py.File(prediction_path, "w") as output:
        output.attrs["validation_file"] = str(args.validation)
        output.attrs["density_scale"] = DENSITY_SCALE
        for label, checkpoint_path in args.checkpoint:
            print(f"[candidate:{label}] loading {checkpoint_path}", flush=True)
            model, checkpoint_metadata = load_checkpoint_model(checkpoint_path, device)
            if args.inference_normalization == "live_batch":
                if checkpoint_metadata["normalization"] != "batch":
                    raise SystemExit(
                        "live_batch inference requires a BatchNorm checkpoint"
                    )
                model.train()
            else:
                model.eval()
            checkpoint_metadata["inference_normalization"] = (
                args.inference_normalization
            )
            dataset = AugmentedH5Dataset(
                args.validation,
                augment=False,
                preprocessing=checkpoint_metadata["input_preprocessing"],
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.workers > 0,
            )
            prediction = np.empty_like(target)
            offset = 0
            with torch.inference_mode():
                for x, _ in loader:
                    x = x.to(device, non_blocking=True)
                    value = model(x).cpu().numpy()[:, 0]
                    prediction[offset : offset + len(value)] = value
                    offset += len(value)
                    print(
                        f"[candidate:{label}] inference {offset}/{n_samples}",
                        flush=True,
                    )
            if offset != n_samples or not np.isfinite(prediction).all():
                raise RuntimeError(f"invalid predictions for {label}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            print(f"[candidate:{label}] computing 2pCF", flush=True)
            profiles = np.stack(
                [density_two_point(prediction[i], estimator) for i in range(n_samples)]
            )
            residual = DENSITY_SCALE * (
                prediction.astype(np.float64) - target.astype(np.float64)
            )
            flat_prediction = prediction.astype(np.float64).ravel()
            flat_target = target.astype(np.float64).ravel()
            candidate_metrics = {
                "checkpoint": checkpoint_metadata,
                "voxel_mse_y": float(np.mean(np.square(flat_prediction - flat_target))),
                "voxel_pearson_y": float(np.corrcoef(flat_prediction, flat_target)[0, 1]),
                "log10_rho_pred_over_truth": {
                    "mean": float(residual.mean()),
                    "std": float(residual.std(ddof=0)),
                    "median": float(np.median(residual)),
                    "p16_p84": np.percentile(residual, [16, 84]).tolist(),
                },
                "two_point": summarize_ks(truth_2pcf, profiles, estimator.radius_mpc_h),
            }
            metrics["candidates"][label] = candidate_metrics
            predictions[label] = prediction
            prediction_2pcf[label] = profiles
            dataset_out = output.create_dataset(
                label,
                data=prediction,
                chunks=(1, grid, grid, grid),
                compression="lzf",
            )
            for key, value in checkpoint_metadata.items():
                if isinstance(value, (str, int, float)):
                    dataset_out.attrs[key] = value
            print(
                f"[candidate:{label}] MSE={candidate_metrics['voxel_mse_y']:.7f} "
                f"KSmean={candidate_metrics['two_point']['mean_0_10_mpc_h']:.4f}",
                flush=True,
            )

    selected = min(
        metrics["candidates"],
        key=lambda label: metrics["candidates"][label]["two_point"][
            "mean_0_10_mpc_h"
        ],
    )
    metrics["selection"] = {
        "criterion": "minimum mean KS across radial bins from 0 to 10 Mpc/h",
        "selected_label": selected,
        "selected_epoch": metrics["candidates"][selected]["checkpoint"]["epoch"],
    }
    metrics_path = args.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        args.out / "two_point_profiles.npz",
        radius_mpc_h=estimator.radius_mpc_h,
        truth=truth_2pcf,
        **prediction_2pcf,
    )
    plot_report(
        args.out / "validation_summary.png",
        args.history,
        target,
        predictions,
        truth_2pcf,
        prediction_2pcf,
        estimator.radius_mpc_h,
        metrics,
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
