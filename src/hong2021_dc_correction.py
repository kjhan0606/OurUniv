#!/usr/bin/env python
"""Train-only observable correction for the conditional cube-mean (DC) mode.

All V6--V12 stochastic residuals deliberately have zero cube mean.  That is
only calibrated if the deterministic conditional mean has the correct DC.
SIMBA development shows a train/validation-stable positive DC residual, so this
module fits a source-balanced ridge model from the same eight observable-only
V8 context features and adds its scalar prediction to both the conditional
mean and every generated member.  It never uses target values while applying
the correction.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from hong2021_residual_diffusion import radial_geometry
from hong2021_residual_v8_context import FEATURE_NAMES, observable_context_features
from hong2021_residual_v11_recentered import V11ResidualDataset
from hong2021_residual_v12_gaussianized import SCHEMA as V12_ENSEMBLE_SCHEMA
from hong2021_train import apply_input_preprocessing


SCHEMA = "hong2021-observable-ridge-dc-correction-v13"
ENSEMBLE_SCHEMA = "hong2021-v13-dc-corrected-v12-ensemble"


def load_training_rows(data: str, cache: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = V11ResidualDataset(data, cache, 1.0, augment=False)
    features = []
    target = []
    for index in range(len(dataset)):
        condition, _, mean, truth = dataset[index]
        features.append(observable_context_features(condition[None])[0].numpy())
        target.append(float((truth - mean).mean()))
    return np.asarray(features, dtype=np.float64), np.asarray(target, dtype=np.float64)


def balanced_moments(tng: np.ndarray, simba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = 0.5 * (tng.mean(axis=0) + simba.mean(axis=0))
    second = 0.5 * ((tng * tng).mean(axis=0) + (simba * simba).mean(axis=0))
    std = np.sqrt(np.maximum(second - mean * mean, 1.0e-12))
    return mean, std


def fit_ridge(
    tng_x: np.ndarray,
    tng_y: np.ndarray,
    simba_x: np.ndarray,
    simba_y: np.ndarray,
    regularization: float,
) -> np.ndarray:
    x = np.concatenate((tng_x, simba_x))
    y = np.concatenate((tng_y, simba_y))
    weights = np.concatenate(
        (
            np.full(len(tng_y), 0.5 / len(tng_y)),
            np.full(len(simba_y), 0.5 / len(simba_y)),
        )
    )
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0.0
    # Solve the weighted ridge problem as an augmented least-squares system.
    # This remains well defined for the predeclared lambda=0 candidate even
    # when two observable summaries happen to be collinear.
    weighted_design = np.sqrt(weights)[:, None] * design
    weighted_target = np.sqrt(weights) * y
    augmented_design = np.vstack((weighted_design, np.sqrt(penalty)))
    augmented_target = np.concatenate((weighted_target, np.zeros(design.shape[1])))
    return np.linalg.lstsq(augmented_design, augmented_target, rcond=None)[0]


def predict(features: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    beta = np.asarray(model["coefficients"], dtype=np.float64)
    standardized = (np.asarray(features, dtype=np.float64) - mean) / std
    return np.column_stack((np.ones(len(standardized)), standardized)) @ beta


def fit(args: argparse.Namespace) -> None:
    output = Path(args.out)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    tng_x, tng_y = load_training_rows(args.tng_train_data, args.tng_train_cache)
    simba_x, simba_y = load_training_rows(
        args.simba_train_data, args.simba_train_cache
    )
    if args.folds < 2 or min(len(tng_y), len(simba_y)) < args.folds:
        raise ValueError("folds must be >=2 and no larger than either source")
    if not np.all(np.isfinite(tng_x)) or not np.all(np.isfinite(simba_x)):
        raise ValueError("non-finite observable feature")
    if not np.all(np.isfinite(tng_y)) or not np.all(np.isfinite(simba_y)):
        raise ValueError("non-finite DC target")
    mean, std = balanced_moments(tng_x, simba_x)
    tng_z = (tng_x - mean) / std
    simba_z = (simba_x - mean) / std
    generator = np.random.default_rng(args.seed)
    tng_folds = generator.permutation(len(tng_y)) % args.folds
    simba_folds = generator.permutation(len(simba_y)) % args.folds
    regularizations = [float(value) for value in args.regularizations.split(",")]
    cv = []
    for regularization in regularizations:
        if not np.isfinite(regularization) or regularization < 0:
            raise ValueError("regularizations must be finite and nonnegative")
        scores = []
        for fold in range(args.folds):
            # Feature scaling is refit inside each training fold.  The final
            # full-training moments above are not allowed to influence the
            # train-only cross-validation score used to select lambda.
            fold_mean, fold_std = balanced_moments(
                tng_x[tng_folds != fold], simba_x[simba_folds != fold]
            )
            tng_fold_z = (tng_x - fold_mean) / fold_std
            simba_fold_z = (simba_x - fold_mean) / fold_std
            beta = fit_ridge(
                tng_fold_z[tng_folds != fold],
                tng_y[tng_folds != fold],
                simba_fold_z[simba_folds != fold],
                simba_y[simba_folds != fold],
                regularization,
            )
            tng_prediction = np.column_stack(
                (
                    np.ones(np.count_nonzero(tng_folds == fold)),
                    tng_fold_z[tng_folds == fold],
                )
            ) @ beta
            simba_prediction = np.column_stack(
                (
                    np.ones(np.count_nonzero(simba_folds == fold)),
                    simba_fold_z[simba_folds == fold],
                )
            ) @ beta
            scores.append(
                0.5 * np.mean((tng_prediction - tng_y[tng_folds == fold]) ** 2)
                + 0.5
                * np.mean((simba_prediction - simba_y[simba_folds == fold]) ** 2)
            )
        cv.append(
            {
                "regularization": regularization,
                "fold_balanced_mse": scores,
                "mean_balanced_mse": float(np.mean(scores)),
            }
        )
    selected = min(cv, key=lambda row: (row["mean_balanced_mse"], row["regularization"]))
    beta = fit_ridge(tng_z, tng_y, simba_z, simba_y, selected["regularization"])
    report = {
        "schema": SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "coefficients": beta.tolist(),
        "regularization_selection": {
            "method": "source-balanced train-only cross-validation with fold-local scaling",
            "folds": args.folds,
            "seed": args.seed,
            "candidates": cv,
            "selected": selected["regularization"],
        },
        "training": {
            "tng_data": str(Path(args.tng_train_data).resolve()),
            "tng_cache": str(Path(args.tng_train_cache).resolve()),
            "tng_samples": len(tng_y),
            "tng_target_mean": float(tng_y.mean()),
            "tng_target_std": float(tng_y.std()),
            "simba_data": str(Path(args.simba_train_data).resolve()),
            "simba_cache": str(Path(args.simba_train_cache).resolve()),
            "simba_samples": len(simba_y),
            "simba_target_mean": float(simba_y.mean()),
            "simba_target_std": float(simba_y.std()),
            "source_weighting": {"tng": 0.5, "simba_development_train": 0.5},
        },
        "nontraining_data_used": False,
        "inference_target_used": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2), flush=True)


def load_model(path: str | Path) -> dict[str, Any]:
    model = json.loads(Path(path).read_text())
    if model.get("schema") != SCHEMA:
        raise ValueError(f"not a V13 DC model: {path}")
    return model


def condition_features(
    data: h5py.File,
    preprocessing: dict[str, Any],
    mean: np.ndarray,
    source_indices: np.ndarray,
) -> np.ndarray:
    grid = int(mean.shape[-1])
    radial = radial_geometry(grid)[None]
    rows = []
    for output_index, source_index in enumerate(source_indices):
        observable = apply_input_preprocessing(
            np.asarray(data["input"][int(source_index)], dtype=np.float32),
            preprocessing,
        )
        condition = np.concatenate((observable, mean[output_index], radial), axis=0)
        rows.append(
            observable_context_features(
                torch.from_numpy(condition[None].copy())
            )[0].numpy()
        )
    return np.asarray(rows, dtype=np.float64)


def apply(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    output = Path(args.out)
    temporary = output.with_suffix(output.suffix + ".tmp")
    if output.exists() or temporary.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(args.ensemble, "r") as source, h5py.File(
            args.data, "r"
        ) as data, h5py.File(args.preprocessing_cache, "r") as preprocessing_source:
            if str(source.attrs.get("schema")) != V12_ENSEMBLE_SCHEMA:
                raise ValueError(f"V13 requires an unmodified V12 ensemble: {args.ensemble}")
            source_indices = np.asarray(source["source_index"], dtype=np.int64)
            mean = np.asarray(source["conditional_mean"], dtype=np.float32)
            if len(source_indices) != len(mean) or len(source_indices) != len(source["sample"]):
                raise ValueError("ensemble arrays have inconsistent object counts")
            if len(source_indices) == 0 or source_indices.min() < 0 or source_indices.max() >= len(data["input"]):
                raise ValueError("ensemble source_index is outside the supplied data")
            preprocessing = json.loads(preprocessing_source.attrs["input_preprocessing"])
            features = condition_features(data, preprocessing, mean, source_indices)
            correction = predict(features, model).astype(np.float32)
            with h5py.File(temporary, "w") as handle:
                sample_shape = source["sample"].shape
                sample_ds = handle.create_dataset(
                    "sample",
                    shape=sample_shape,
                    dtype="f4",
                    chunks=source["sample"].chunks,
                    compression="lzf",
                )
                mean_ds = handle.create_dataset(
                    "conditional_mean",
                    shape=mean.shape,
                    dtype="f4",
                    compression="lzf",
                )
                truth_ds = handle.create_dataset(
                    "truth",
                    data=np.asarray(source["truth"], dtype=np.float32),
                    compression="lzf",
                )
                handle.create_dataset("source_index", data=source_indices)
                for index, dc in enumerate(correction):
                    sample_ds[index] = np.asarray(source["sample"][index], dtype=np.float32) + dc
                    mean_ds[index] = mean[index] + dc
                for key, value in source.attrs.items():
                    handle.attrs[key] = value
                handle.attrs.update(
                    {
                        "schema": ENSEMBLE_SCHEMA,
                        "parent_ensemble": str(Path(args.ensemble).resolve()),
                        "dc_model": str(Path(args.model).resolve()),
                        "dc_prediction_mean": float(correction.mean()),
                        "dc_prediction_std": float(correction.std()),
                        "dc_prediction_min": float(correction.min()),
                        "dc_prediction_max": float(correction.max()),
                        "dc_prediction_uses_target": False,
                        "complete": True,
                    }
                )
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    print(
        json.dumps(
            {
                "out": str(output),
                "objects": len(correction),
                "dc_mean": float(correction.mean()),
                "dc_std": float(correction.std()),
            },
            indent=2,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    fitting = sub.add_parser("fit")
    fitting.add_argument("--tng-train-data", required=True)
    fitting.add_argument("--tng-train-cache", required=True)
    fitting.add_argument("--simba-train-data", required=True)
    fitting.add_argument("--simba-train-cache", required=True)
    fitting.add_argument("--out", required=True)
    fitting.add_argument("--folds", type=int, default=5)
    fitting.add_argument(
        "--regularizations", default="0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1"
    )
    fitting.add_argument("--seed", type=int, default=13021)
    applying = sub.add_parser("apply")
    applying.add_argument("--ensemble", required=True)
    applying.add_argument("--data", required=True)
    applying.add_argument("--preprocessing-cache", required=True)
    applying.add_argument("--model", required=True)
    applying.add_argument("--out", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    {"fit": fit, "apply": apply}[args.mode](args)


if __name__ == "__main__":
    main()
