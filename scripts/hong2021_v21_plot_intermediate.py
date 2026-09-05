#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/gpfs/kjhan/IllustrisTNG/TNG100-1/derived")
PROFILE = ROOT / "hong2021_v21/model/conditional_affine_profile.json"
SOURCE = ROOT / "hong2021_v14/model/tng_validation_standardized.h5"
V21 = ROOT / "hong2021_v21/model/tng100_validation_conditional_affine.h5"


def main() -> None:
    profile = json.loads(PROFILE.read_text())
    with h5py.File(SOURCE, "r") as source, h5py.File(V21, "r") as v21:
        mean = np.asarray(source["conditional_mean"][0, 0], dtype=np.float32)
        residual = np.asarray(source["standardized_residual"][0, 0], dtype=np.float32)
        latent = np.asarray(v21["standardized_residual"][0, 0], dtype=np.float32)
        raw_sample = np.concatenate([
            np.asarray(source["standardized_residual"][i, 0], dtype=np.float32).reshape(-1)[::32]
            for i in range(min(16, len(source["standardized_residual"])))
        ])
        latent_sample = np.concatenate([
            np.asarray(v21["standardized_residual"][i, 0], dtype=np.float32).reshape(-1)[::32]
            for i in range(min(16, len(v21["standardized_residual"])))
        ])
    middle = mean.shape[0] // 2
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    images = (
        (mean[middle], "conditional mean m", "viridis"),
        (residual[middle], "V14 residual r", "coolwarm"),
        (latent[middle], "V21 training target", "coolwarm"),
    )
    for axis, (value, title, cmap) in zip(axes[0], images, strict=True):
        limit = np.quantile(np.abs(value), 0.995) if cmap == "coolwarm" else None
        image = axis.imshow(value, origin="lower", cmap=cmap, vmin=-limit if limit else None, vmax=limit)
        axis.set_title(title); axis.set_xticks([]); axis.set_yticks([])
        figure.colorbar(image, ax=axis, shrink=0.8)
    centers = np.asarray(profile["centers"])
    axes[1, 0].plot(centers, profile["mu"], "o-", label="mu(m)")
    axes[1, 0].set_xlabel("conditional mean m"); axes[1, 0].set_ylabel("residual location")
    axes[1, 0].grid(alpha=0.3); axes[1, 0].legend()
    axes[1, 1].plot(centers, profile["sigma"], "o-", color="tab:red", label="sigma(m)")
    axes[1, 1].set_xlabel("conditional mean m"); axes[1, 1].set_ylabel("residual scale")
    axes[1, 1].grid(alpha=0.3); axes[1, 1].legend()
    bins = np.linspace(-5, 5, 161)
    axes[1, 2].hist(raw_sample, bins=bins, density=True, histtype="step", lw=1.5, label="V14 r")
    axes[1, 2].hist(latent_sample, bins=bins, density=True, histtype="step", lw=1.5, label="V21 target")
    x = np.linspace(-5, 5, 501)
    axes[1, 2].plot(x, np.exp(-0.5*x*x)/np.sqrt(2*np.pi), "k--", lw=1, label="N(0,1)")
    axes[1, 2].set_yscale("log"); axes[1, 2].set_ylim(1e-5, 1)
    axes[1, 2].set_xlabel("value"); axes[1, 2].set_ylabel("PDF"); axes[1, 2].legend()
    figure.suptitle("V21 pre-training diagnostic (not a generated density field)", fontsize=14)
    output = Path("/home/kjhan/BACKUP/CF4/artifacts/hong2021_v21_intermediate.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
