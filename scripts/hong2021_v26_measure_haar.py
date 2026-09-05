#!/usr/bin/env python
"""Measure train-only V21 Haar-detail moments for the V26 flow design audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from hong2021_v18_init import sha256_file
from hong2021_v26_haar import detail_dimensions, haar_pyramid


SCHEMA = "hong2021-v26-train-only-haar-detail-moments-v1"
DOMAIN_KEYS = ("TNG100", "SIMBA", "Swift")
CACHE_KEYS = {
    "TNG100": "TNG100_train",
    "SIMBA": "SIMBA_train",
    "Swift": "Swift_train",
}


def measure_cache(path: Path, *, device: torch.device) -> dict:
    levels = 6
    sums = np.zeros((levels, 7), dtype=np.float64)
    squares = np.zeros_like(sums)
    cubes = 0
    maximum_dc = 0.0
    with h5py.File(path, "r") as handle:
        dataset = handle["standardized_residual"]
        if tuple(dataset.shape[1:]) != (1, 64, 64, 64):
            raise ValueError(f"unexpected V21 cache shape: {path}")
        for index in range(len(dataset)):
            value = torch.from_numpy(
                np.asarray(dataset[index : index + 1], dtype=np.float32)
            ).to(device)
            lowpass, details = haar_pyramid(value, levels=levels)
            maximum_dc = max(maximum_dc, float(lowpass.abs().max().cpu()))
            for level, detail in enumerate(details):
                flat = detail.double().flatten(2)
                sums[level] += flat.sum(dim=(0, 2)).cpu().numpy()
                squares[level] += flat.square().sum(dim=(0, 2)).cpu().numpy()
            cubes += 1
            if (index + 1) % 50 == 0 or index + 1 == len(dataset):
                print(f"[haar] {path.name} {index + 1}/{len(dataset)}", flush=True)
    counts = np.asarray(detail_dimensions(), dtype=np.int64)[:, None] // 7 * cubes
    mean = sums / counts
    second = squares / counts
    return {
        "objects": cubes,
        "mean": mean.tolist(),
        "second_raw_moment": second.tolist(),
        "maximum_absolute_coarsest_dc": maximum_dc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    repo = args.repo.resolve()
    artifacts_path = repo / "config/hong2021_v21_derived_artifacts.json"
    artifacts_sha = sha256_file(artifacts_path)
    artifacts = json.loads(artifacts_path.read_text())
    sources = {}
    per_domain = {}
    for domain in DOMAIN_KEYS:
        row = artifacts["caches"][CACHE_KEYS[domain]]
        path = Path(row["path"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"V21 train-cache hash mismatch: {domain}")
        sources[domain] = row
        per_domain[domain] = measure_cache(path, device=torch.device(args.device))
    source_means = np.asarray(
        [per_domain[domain]["mean"] for domain in DOMAIN_KEYS], dtype=np.float64
    )
    source_seconds = np.asarray(
        [per_domain[domain]["second_raw_moment"] for domain in DOMAIN_KEYS],
        dtype=np.float64,
    )
    balanced_mean = source_means.mean(axis=0)
    balanced_variance = source_seconds.mean(axis=0) - balanced_mean**2
    if not np.isfinite(balanced_variance).all() or np.any(balanced_variance <= 0.0):
        raise RuntimeError("Haar detail variance is not finite and positive")
    report = {
        "schema": SCHEMA,
        "purpose": "Fit only the fixed affine coordinate scale for a later exact conditional flow.",
        "v21_artifacts": str(artifacts_path),
        "v21_artifacts_sha256": artifacts_sha,
        "sources": sources,
        "source_weights": {domain: 1.0 / 3.0 for domain in DOMAIN_KEYS},
        "grid": 64,
        "levels": 6,
        "detail_channels_per_level": 7,
        "detail_dimensions_fine_to_coarse": detail_dimensions(),
        "non_dc_dimensions": sum(detail_dimensions()),
        "per_domain": per_domain,
        "source_balanced_mean": balanced_mean.tolist(),
        "source_balanced_variance": balanced_variance.tolist(),
        "source_balanced_standard_deviation": np.sqrt(balanced_variance).tolist(),
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    output = args.out.resolve()
    partial = output.with_suffix(output.suffix + ".partial")
    if output.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite V26 Haar measurement: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
