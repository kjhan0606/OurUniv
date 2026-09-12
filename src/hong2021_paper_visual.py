#!/usr/bin/env python
"""Make the Hong-paper-style 5-Mpc/h projection comparison for TNG cubes."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


DENSITY_SCALE = 4.5


def central_slab(array: np.ndarray, axis: int, cells: int) -> np.ndarray:
    if array.ndim != 3 or axis not in (0, 1, 2) or not 1 <= cells <= array.shape[axis]:
        raise ValueError("invalid cube, axis, or slab thickness")
    start = (array.shape[axis] - cells) // 2
    selected = np.take(array, range(start, start + cells), axis=axis)
    return selected


def slab_sum(array: np.ndarray, axis: int, cells: int) -> np.ndarray:
    return central_slab(array, axis, cells).sum(axis=axis)


def slab_mean(array: np.ndarray, axis: int, cells: int) -> np.ndarray:
    return central_slab(array, axis, cells).mean(axis=axis)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--thickness-mpc-h", type=float, default=5.0)
    parser.add_argument("--voxel-mpc-h", type=float, default=0.3125)
    args = parser.parse_args()
    with h5py.File(args.data, "r") as handle:
        inputs = np.asarray(handle["input"])
        truth_y = np.asarray(handle["target"][:, 0])
    with h5py.File(args.predictions, "r") as handle:
        if args.label not in handle:
            raise SystemExit(f"prediction label missing: {args.label}")
        prediction_y = np.asarray(handle[args.label])
    if prediction_y.shape != truth_y.shape:
        raise SystemExit("prediction and truth shapes differ")
    mse = np.mean(np.square(prediction_y - truth_y), axis=(1, 2, 3))
    sample = (
        int(np.argsort(mse)[len(mse) // 2])
        if args.sample is None
        else int(args.sample)
    )
    if not 0 <= sample < len(truth_y):
        raise SystemExit("sample index outside data")
    cells = int(round(args.thickness_mpc_h / args.voxel_mpc_h))
    grid = truth_y.shape[-1]
    half_box = grid * args.voxel_mpc_h / 2.0
    extent = (-half_box, half_box, -half_box, half_box)
    truth_rho = np.power(10.0, DENSITY_SCALE * truth_y[sample])
    prediction_rho = np.power(10.0, DENSITY_SCALE * prediction_y[sample])

    figure, axes = plt.subplots(3, 5, figsize=(17, 10), constrained_layout=True)
    plane_names = ("YZ", "XZ", "XY")
    for axis, plane in enumerate(plane_names):
        count = slab_sum(inputs[sample, 0], axis, cells)
        velocity_count = slab_sum(inputs[sample, 1] * inputs[sample, 0], axis, cells)
        velocity = np.divide(
            velocity_count,
            count,
            out=np.full_like(velocity_count, np.nan),
            where=count > 0,
        )
        truth = np.log10(slab_mean(truth_rho, axis, cells))
        prediction = np.log10(slab_mean(prediction_rho, axis, cells))
        residual = prediction - truth
        density_limits = np.percentile(np.concatenate((truth.ravel(), prediction.ravel())), [1, 99])
        velocity_limit = np.nanpercentile(np.abs(velocity), 98)
        panels = (
            (np.log1p(count), "gray_r", None, None, r"$\log(1+N_{gal})$"),
            (velocity, "coolwarm", -velocity_limit, velocity_limit, r"$V_{pec}$"),
            (truth, "turbo", *density_limits, r"truth $\log_{10}(\Sigma/\Sigma_0)$"),
            (prediction, "turbo", *density_limits, f"{args.label} prediction"),
            (residual, "RdBu_r", -1.0, 1.0, "prediction - truth"),
        )
        for column, (value, cmap, vmin, vmax, title) in enumerate(panels):
            image = axes[axis, column].imshow(
                value.T,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axes[axis, column].set_title(f"{plane}: {title}", fontsize=10)
            axes[axis, column].set_xlabel(r"$h^{-1}{\rm Mpc}$")
            if column == 0:
                axes[axis, column].set_ylabel(r"$h^{-1}{\rm Mpc}$")
            figure.colorbar(image, ax=axes[axis, column], fraction=0.045)
    figure.suptitle(
        f"Hong-style {args.thickness_mpc_h:g} Mpc/h slab; sample={sample}; "
        f"MSE={mse[sample]:.5f}",
        fontsize=15,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=170)
    plt.close(figure)
    print(f"sample={sample} cells={cells} mse={mse[sample]:.7f} out={args.out}")


if __name__ == "__main__":
    main()
