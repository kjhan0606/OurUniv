#!/usr/bin/env python
"""Audit target-free V14 location/scale predictability across development codes.

The audit consumes only the baseline caches made by
``hong2021_v14_baseline_audit.py``.  It fits source-balanced ridge regressions
for the residual cube DC and log Fourier-band RMS using the eight observable
context features.  Regularization is selected by train-only cross-validation;
held-out development splits are reported only against a source-balanced
constant baseline.  This is diagnostic evidence, not a frozen V14 model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_residual_v8_context import FEATURE_NAMES
from hong2021_v14_baseline_audit import BAND_EDGES_H_MPC, SCHEMA as CACHE_SCHEMA


SCHEMA = "hong2021-v14-location-scale-predictability-audit-v1"
ALLOWED_DOMAINS = ("TNG100", "CAMELS-SIMBA", "CAMELS-Swift-EAGLE")
TARGET_NAMES = ("residual_dc",) + tuple(
    f"log_residual_band_rms_{index}" for index in range(len(BAND_EDGES_H_MPC) - 1)
)


def balanced_moments(sources: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return moments giving every source equal weight."""
    if not sources or any(value.ndim != 2 or len(value) == 0 for value in sources):
        raise ValueError("balanced moments require nonempty two-dimensional sources")
    if len({value.shape[1] for value in sources}) != 1:
        raise ValueError("source feature dimensions differ")
    mean = np.mean([value.mean(axis=0) for value in sources], axis=0)
    second = np.mean([(value * value).mean(axis=0) for value in sources], axis=0)
    std = np.sqrt(np.maximum(second - mean * mean, 1.0e-12))
    return mean, std


def fit_source_balanced_ridge(
    features: list[np.ndarray],
    targets: list[np.ndarray],
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit an equal-source-weight multi-output ridge with free intercept."""
    if len(features) != len(targets) or not features:
        raise ValueError("features and targets must contain the same sources")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative")
    if any(len(x) != len(y) or y.ndim != 2 for x, y in zip(features, targets, strict=True)):
        raise ValueError("source feature/target shapes differ")
    mean, std = balanced_moments(features)
    standardized = [(value - mean) / std for value in features]
    x = np.concatenate(standardized)
    y = np.concatenate(targets)
    weights = np.concatenate(
        [np.full(len(value), 1.0 / len(features) / len(value)) for value in features]
    )
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0.0
    augmented_design = np.vstack(
        (np.sqrt(weights)[:, None] * design, np.sqrt(penalty))
    )
    augmented_target = np.vstack(
        (np.sqrt(weights)[:, None] * y, np.zeros((design.shape[1], y.shape[1])))
    )
    coefficients = np.linalg.lstsq(
        augmented_design, augmented_target, rcond=None
    )[0]
    return mean, std, coefficients


def ridge_predict(
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    standardized = (np.asarray(features, dtype=np.float64) - mean) / std
    return np.column_stack((np.ones(len(standardized)), standardized)) @ coefficients


def load_cache(path: str | Path, expected_domain: str) -> tuple[np.ndarray, np.ndarray]:
    if expected_domain not in ALLOWED_DOMAINS:
        raise ValueError(f"not a V14 development domain: {expected_domain}")
    value = Path(path)
    forbidden = ("astrid", "refl0100n1504")
    if any(token in str(value).lower() for token in forbidden):
        raise ValueError("sealed independent or historical EAGLE cache is forbidden")
    with h5py.File(value, "r") as handle:
        if str(handle.attrs.get("schema")) != CACHE_SCHEMA:
            raise ValueError(f"not a V14 baseline audit cache: {value}")
        if str(handle.attrs.get("domain")) != expected_domain:
            raise ValueError("cache domain does not match its declared source")
        if bool(handle.attrs.get("feature_uses_target", True)):
            raise ValueError("location-scale features must be target-free")
        names = tuple(json.loads(handle.attrs["feature_names"]))
        if names != FEATURE_NAMES:
            raise ValueError("observable feature definitions differ")
        edges = tuple(json.loads(handle.attrs["band_edges_h_mpc"]))
        if edges != BAND_EDGES_H_MPC:
            raise ValueError("Fourier band definitions differ")
        features = np.asarray(handle["observable_context_features"], dtype=np.float64)
        dc = np.asarray(handle["residual_dc"], dtype=np.float64)
        band = np.asarray(handle["residual_band_rms"], dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("invalid observable feature array")
    if dc.shape != (len(features),) or band.shape != (
        len(features), len(BAND_EDGES_H_MPC) - 1
    ):
        raise ValueError("invalid location/scale target arrays")
    if not np.isfinite(features).all() or not np.isfinite(dc).all():
        raise ValueError("non-finite audit value")
    if not np.isfinite(band).all() or np.any(band <= 0):
        raise ValueError("band RMS values must be finite and positive")
    targets = np.column_stack((dc, np.log(band)))
    return features, targets


def _parse_sources(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("source must be DOMAIN=PATH")
        domain, path = value.split("=", 1)
        if domain not in ALLOWED_DOMAINS:
            raise ValueError(f"not a V14 development domain: {domain}")
        if domain in result:
            raise ValueError(f"duplicate source: {domain}")
        result[domain] = Path(path)
    if set(result) != set(ALLOWED_DOMAINS):
        raise ValueError("all three V14 development domains are required")
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    train_paths = _parse_sources(args.train)
    validation_paths = _parse_sources(args.validation)
    domains = list(ALLOWED_DOMAINS)
    train_rows = [load_cache(train_paths[name], name) for name in domains]
    validation_rows = [load_cache(validation_paths[name], name) for name in domains]
    features = [row[0] for row in train_rows]
    targets = [row[1] for row in train_rows]
    if args.folds < 2 or any(len(value) < args.folds for value in features):
        raise ValueError("fold count exceeds a training source")
    regularizations = [float(value) for value in args.regularizations.split(",")]
    if not regularizations or len(set(regularizations)) != len(regularizations):
        raise ValueError("regularization candidates must be unique")
    generator = np.random.default_rng(args.seed)
    fold_ids = [generator.permutation(len(value)) % args.folds for value in features]
    cv_rows = []
    for regularization in regularizations:
        fold_scores = []
        for fold in range(args.folds):
            train_x = [x[f != fold] for x, f in zip(features, fold_ids, strict=True)]
            train_y = [y[f != fold] for y, f in zip(targets, fold_ids, strict=True)]
            mean, std, coefficients = fit_source_balanced_ridge(
                train_x, train_y, regularization
            )
            source_scores = []
            for x, y, f in zip(features, targets, fold_ids, strict=True):
                prediction = ridge_predict(x[f == fold], mean, std, coefficients)
                source_scores.append(np.mean((prediction - y[f == fold]) ** 2, axis=0))
            fold_scores.append(np.mean(source_scores, axis=0))
        score = np.mean(fold_scores, axis=0)
        cv_rows.append(
            {
                "regularization": regularization,
                "fold_source_balanced_mse": np.asarray(fold_scores).tolist(),
                "mean_source_balanced_mse": score.tolist(),
            }
        )
    selected = []
    for target_index in range(len(TARGET_NAMES)):
        row = min(
            cv_rows,
            key=lambda item: (
                item["mean_source_balanced_mse"][target_index],
                item["regularization"],
            ),
        )
        selected.append(float(row["regularization"]))
    constant = np.mean([value.mean(axis=0) for value in targets], axis=0)
    target_reports = []
    for target_index, (target_name, regularization) in enumerate(
        zip(TARGET_NAMES, selected, strict=True)
    ):
        mean, std, coefficients = fit_source_balanced_ridge(
            features, targets, regularization
        )
        validation = {}
        ratios = []
        for domain, (x, y) in zip(domains, validation_rows, strict=True):
            prediction = ridge_predict(x, mean, std, coefficients)[:, target_index]
            truth = y[:, target_index]
            conditional_rmse = float(np.sqrt(np.mean((prediction - truth) ** 2)))
            constant_rmse = float(np.sqrt(np.mean((constant[target_index] - truth) ** 2)))
            ratio = conditional_rmse / constant_rmse
            correlation = (
                float(np.corrcoef(prediction, truth)[0, 1])
                if np.std(prediction) > 0 and np.std(truth) > 0
                else None
            )
            ratios.append(ratio)
            validation[domain] = {
                "samples": len(x),
                "truth_mean": float(truth.mean()),
                "prediction_mean": float(prediction.mean()),
                "conditional_rmse": conditional_rmse,
                "source_balanced_constant_rmse": constant_rmse,
                "conditional_to_constant_rmse_ratio": ratio,
                "correlation": correlation,
            }
        target_reports.append(
            {
                "target": target_name,
                "selected_regularization_train_only": regularization,
                "source_balanced_training_constant": float(constant[target_index]),
                "conditional_strictly_dominates_constant_on_all_validation_domains": bool(
                    all(value < 1.0 for value in ratios)
                ),
                "validation": validation,
            }
        )
    report = {
        "schema": SCHEMA,
        "status": "diagnostic_only_not_a_frozen_v14_model",
        "feature_names": list(FEATURE_NAMES),
        "features_use_target": False,
        "simulation_identity_feature": False,
        "targets": list(TARGET_NAMES),
        "fourier_band_edges_h_mpc": list(BAND_EDGES_H_MPC),
        "training_sources": {
            domain: {"path": str(train_paths[domain].resolve()), "samples": len(features[index])}
            for index, domain in enumerate(domains)
        },
        "validation_sources": {
            domain: {
                "path": str(validation_paths[domain].resolve()),
                "samples": len(validation_rows[index][0]),
            }
            for index, domain in enumerate(domains)
        },
        "fit_policy": {
            "source_weight": 1.0 / len(domains),
            "folds": args.folds,
            "seed": args.seed,
            "regularization_candidates": regularizations,
            "regularization_selection": "per target, source-balanced train-only CV",
            "validation_used_to_fit_or_select_regularization": False,
        },
        "cross_validation": cv_rows,
        "target_reports": target_reports,
    }
    output = Path(args.out)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="append", required=True)
    parser.add_argument("--validation", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--regularizations", default="0,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1"
    )
    parser.add_argument("--seed", type=int, default=14021)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
