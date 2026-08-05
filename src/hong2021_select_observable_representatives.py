#!/usr/bin/env python
"""Select development representatives using observables and model mean only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from hong2021_residual_diffusion import radial_geometry
from hong2021_residual_v8_context import (
    FEATURE_NAMES,
    observable_context_features,
)
from hong2021_train import apply_input_preprocessing


SCHEMA = "hong2021-observable-context-representatives-v1"


def farthest_feature_subset(features: np.ndarray, count: int) -> np.ndarray:
    value = np.asarray(features, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("features must be a finite matrix")
    if count <= 0 or count > len(value):
        raise ValueError("invalid representative count")
    # Index zero is an explicit deterministic anchor.  np.argmax supplies a
    # stable smallest-index tie break.
    chosen = [0]
    minimum_distance2 = np.square(value - value[0]).sum(axis=1)
    minimum_distance2[0] = -1.0
    while len(chosen) < count:
        selected = int(np.argmax(minimum_distance2))
        chosen.append(selected)
        distance2 = np.square(value - value[selected]).sum(axis=1)
        minimum_distance2 = np.minimum(minimum_distance2, distance2)
        minimum_distance2[chosen] = -1.0
    return np.asarray(chosen, dtype=np.int64)


def extract_features(
    data_path: Path, cache_path: Path, feature_mean: np.ndarray, feature_std: np.ndarray
) -> tuple[np.ndarray, str]:
    with h5py.File(data_path, "r") as data, h5py.File(cache_path, "r") as cache:
        if len(data["input"]) != len(cache["conditional_mean"]):
            raise ValueError("data/cache sample counts differ")
        preprocessing = json.loads(cache.attrs["input_preprocessing"])
        grid = int(data["input"].shape[-1])
        radial = radial_geometry(grid)[None]
        rows = []
        for index in range(len(data["input"])):
            observable = apply_input_preprocessing(
                np.asarray(data["input"][index], dtype=np.float32), preprocessing
            )
            mean = np.asarray(
                cache["conditional_mean"][index], dtype=np.float32
            )
            condition = np.concatenate((observable, mean, radial), axis=0)
            rows.append(
                observable_context_features(torch.from_numpy(condition[None]))[
                    0
                ].numpy()
            )
        data_schema = str(data.attrs.get("schema", ""))
    raw = np.asarray(rows, dtype=np.float64)
    return (raw - feature_mean) / feature_std, data_schema


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--feature-fit-checkpoint", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(
        args.feature_fit_checkpoint, map_location="cpu", weights_only=False
    )
    fit = checkpoint["observable_context_features"]
    if fit["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("checkpoint feature order differs")
    mean = np.asarray(fit["mean"], dtype=np.float64)
    std = np.asarray(fit["std"], dtype=np.float64)
    features, data_schema = extract_features(args.data, args.cache, mean, std)
    indices = farthest_feature_subset(features, args.count)
    report = {
        "schema": SCHEMA,
        "selection_uses_density_truth": False,
        "selection_uses_simulation_label": False,
        "algorithm": (
            "deterministic farthest-point selection in standardized observable "
            "context space; source index zero anchor; smallest-index tie break"
        ),
        "source_data": str(args.data.resolve()),
        "source_data_schema": data_schema,
        "source_cache": str(args.cache.resolve()),
        "feature_fit_checkpoint": str(args.feature_fit_checkpoint.resolve()),
        "feature_names": list(FEATURE_NAMES),
        "samples": len(features),
        "representatives": len(indices),
        "indices": indices.tolist(),
        "standardized_features": features[indices].tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "indices": indices.tolist()}, indent=2))


if __name__ == "__main__":
    main()
