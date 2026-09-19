#!/usr/bin/env python
"""Fit the frozen V14 target-free conditional residual location and scales."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_residual_v8_context import FEATURE_NAMES
from hong2021_v14_location_scale_audit import (
    fit_source_balanced_ridge,
    ridge_predict,
)
from hong2021_v14_mean_correction import DOMAINS
from hong2021_v14_residual_cache import CORRECTED_SCHEMA


SCHEMA = "hong2021-v14-three-domain-location-scale-model-v1"
RIDGE_TARGETS = ("residual_dc", "log_band_2_rms", "log_band_3_rms")


def load_rows(path: str | Path, domain: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("schema")) != CORRECTED_SCHEMA:
            raise ValueError(f"not a corrected V14 cache: {path}")
        if str(handle.attrs.get("domain")) != domain:
            raise ValueError("corrected cache domain mismatch")
        if bool(handle.attrs.get("feature_uses_target", True)):
            raise ValueError("location-scale features are not target-free")
        names = tuple(json.loads(handle.attrs["feature_names"]))
        if names != FEATURE_NAMES:
            raise ValueError("location-scale feature order differs")
        features = np.asarray(handle["observable_context_features"], dtype=np.float64)
        dc = np.asarray(handle["residual_dc"], dtype=np.float64)
        band = np.asarray(handle["residual_band_rms"], dtype=np.float64)
    if band.shape != (len(features), 4) or np.any(band <= 0):
        raise ValueError("invalid residual band scales")
    targets = np.column_stack((dc, np.log(band[:, 2]), np.log(band[:, 3])))
    return features, targets


def _fit_cv(
    features: list[np.ndarray],
    targets: list[np.ndarray],
    regularizations: list[float],
    folds: int,
    seed: int,
) -> tuple[list[float], list[dict[str, Any]]]:
    if folds < 2 or any(len(value) < folds for value in features):
        raise ValueError("invalid location-scale fold count")
    generator = np.random.default_rng(seed)
    fold_ids = [generator.permutation(len(value)) % folds for value in features]
    rows = []
    for regularization in regularizations:
        fold_scores = []
        for fold in range(folds):
            train_x = [x[f != fold] for x, f in zip(features, fold_ids, strict=True)]
            train_y = [y[f != fold] for y, f in zip(targets, fold_ids, strict=True)]
            mean, std, coefficients = fit_source_balanced_ridge(
                train_x, train_y, regularization
            )
            per_source = []
            for x, y, f in zip(features, targets, fold_ids, strict=True):
                prediction = ridge_predict(x[f == fold], mean, std, coefficients)
                per_source.append(np.mean((prediction - y[f == fold]) ** 2, axis=0))
            fold_scores.append(np.mean(per_source, axis=0))
        mean_score = np.mean(fold_scores, axis=0)
        rows.append(
            {
                "regularization": regularization,
                "fold_source_balanced_mse": np.asarray(fold_scores).tolist(),
                "mean_source_balanced_mse": mean_score.tolist(),
            }
        )
    selected = [
        float(
            min(
                rows,
                key=lambda row: (
                    row["mean_source_balanced_mse"][target],
                    row["regularization"],
                ),
            )["regularization"]
        )
        for target in range(len(RIDGE_TARGETS))
    ]
    return selected, rows


def predict_location_scales(
    features: np.ndarray, model: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    std = np.asarray(model["feature_std"], dtype=np.float64)
    design = np.column_stack((np.ones(len(value)), (value - mean) / std))
    location = design @ np.asarray(model["location"]["coefficients"], dtype=np.float64)
    scales = np.empty((len(value), 4), dtype=np.float64)
    for band, definition in enumerate(model["bands"]):
        if definition["kind"] == "constant_log_rms":
            log_scale = np.full(len(value), definition["log_scale"])
        elif definition["kind"] == "ridge_log_rms":
            log_scale = design @ np.asarray(definition["coefficients"], dtype=np.float64)
        else:
            raise ValueError(f"unknown V14 scale definition: {definition['kind']}")
        scales[:, band] = np.exp(log_scale)
    if not np.isfinite(location).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise RuntimeError("location-scale model produced an invalid prediction")
    return location, scales


def fit(args: argparse.Namespace) -> dict[str, Any]:
    train_paths = {
        "TNG100": args.tng_train,
        "SIMBA": args.simba_train,
        "Swift-EAGLE": args.swift_train,
    }
    validation_paths = {
        "TNG100": args.tng_validation,
        "SIMBA": args.simba_validation,
        "Swift-EAGLE": args.swift_validation,
    }
    train = [load_rows(train_paths[name], name) for name in DOMAINS]
    validation = [load_rows(validation_paths[name], name) for name in DOMAINS]
    features = [value[0] for value in train]
    targets = [value[1] for value in train]
    regularizations = [float(value) for value in args.regularizations.split(",")]
    selected, cv = _fit_cv(features, targets, regularizations, args.folds, args.seed)
    # All ridge targets share the exact same full-training feature moments.
    feature_mean, feature_std, _ = fit_source_balanced_ridge(
        features, targets, selected[0]
    )
    coefficients = []
    for index, regularization in enumerate(selected):
        mean, std, beta = fit_source_balanced_ridge(
            features, targets, regularization
        )
        if not np.array_equal(mean, feature_mean) or not np.array_equal(std, feature_std):
            raise RuntimeError("ridge targets produced inconsistent feature moments")
        coefficients.append(beta[:, index])
    # Bands 0 and 1 deliberately use source-balanced constants after the
    # baseline audit showed observable regression harms TNG validation.
    train_band_logs = []
    for path, domain in zip(train_paths.values(), DOMAINS, strict=True):
        with h5py.File(path, "r") as handle:
            band = np.asarray(handle["residual_band_rms"], dtype=np.float64)
            train_band_logs.append(np.log(band).mean(axis=0))
    constant_log_scale = np.mean(train_band_logs, axis=0)
    model = {
        "schema": SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
        "source_weight": 1.0 / len(DOMAINS),
        "training_sources": {
            name: {"path": str(Path(train_paths[name]).resolve()), "samples": len(features[index])}
            for index, name in enumerate(DOMAINS)
        },
        "location": {
            "kind": "ridge",
            "target": "residual_dc",
            "regularization": selected[0],
            "coefficients": coefficients[0].tolist(),
        },
        "bands": [
            {"band": 0, "kind": "constant_log_rms", "log_scale": float(constant_log_scale[0])},
            {"band": 1, "kind": "constant_log_rms", "log_scale": float(constant_log_scale[1])},
            {"band": 2, "kind": "ridge_log_rms", "regularization": selected[1], "coefficients": coefficients[1].tolist()},
            {"band": 3, "kind": "ridge_log_rms", "regularization": selected[2], "coefficients": coefficients[2].tolist()},
        ],
        "regularization_selection": {
            "method": "per target five-fold equal-source train-only CV with fold-local scaling",
            "folds": args.folds,
            "seed": args.seed,
            "candidates": cv,
            "selected": dict(zip(RIDGE_TARGETS, selected, strict=True)),
            "validation_used": False,
        },
        "simulation_identity_feature": False,
        "inference_target_feature": False,
        "scale_prediction_clipping": False,
    }
    validation_report = {}
    for domain, (x, y) in zip(DOMAINS, validation, strict=True):
        location, scales = predict_location_scales(x, model)
        prediction = np.column_stack((location, np.log(scales[:, 2]), np.log(scales[:, 3])))
        validation_report[domain] = {
            target: {
                "rmse": float(np.sqrt(np.mean((prediction[:, index] - y[:, index]) ** 2))),
                "truth_mean": float(y[:, index].mean()),
                "prediction_mean": float(prediction[:, index].mean()),
            }
            for index, target in enumerate(RIDGE_TARGETS)
        }
    model["development_validation_audit_after_train_only_fit"] = validation_report
    output = Path(args.out)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite location-scale model: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(model, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(model, indent=2), flush=True)
    return model


def load_model(path: str | Path) -> dict[str, Any]:
    model = json.loads(Path(path).read_text())
    if model.get("schema") != SCHEMA:
        raise ValueError(f"not a V14 location-scale model: {path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for prefix in (
        "tng-train", "tng-validation", "simba-train", "simba-validation",
        "swift-train", "swift-validation",
    ):
        parser.add_argument(f"--{prefix}", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--regularizations", default="0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1"
    )
    parser.add_argument("--seed", type=int, default=143021)
    fit(parser.parse_args())


if __name__ == "__main__":
    main()
