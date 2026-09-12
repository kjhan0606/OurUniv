#!/usr/bin/env python
"""Frozen V32 local-context and recoverable velocity-dispersion audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
from scipy.ndimage import uniform_filter

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v30_backbone_audit import block_mean, block_sum
from hong2021_v31_copula import DOMAIN_ORDER


PROGRAM_SCHEMA = "hong2021-v32-local-context-velocity-dispersion-audit-program-v1"
PROGRAM_SHA256 = "c4c88d0719cf4a806c0058045c03e7fbf26d4a401c8d8bc10c751aa84d2f3314"
SCHEMA = "hong2021-v32-local-context-velocity-dispersion-audit-v1"
FACTORS = (4, 8)
MODEL_COLUMNS = {
    "base": tuple(range(5)),
    "plus_sig_v": tuple(range(6)),
    "plus_local_context": tuple(range(12)),
}
FEATURE_NAMES = (
    "log1p_block_count",
    "block_mean_velocity_kms",
    "backbone_mean_y",
    "backbone_std_y",
    "radius_over_half_width",
    "block_velocity_dispersion_lower_bound_kms",
    "local_logcount_mean",
    "local_logcount_std",
    "local_velocity_mean",
    "local_velocity_std",
    "local_backbone_mean",
    "local_backbone_std",
)


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    if sha256_file(path.resolve()) != PROGRAM_SHA256:
        raise ValueError("V32 program hash differs")
    program = json.loads(path.read_text())
    if program.get("schema") != PROGRAM_SCHEMA or tuple(program["development_domains"]) != DOMAIN_ORDER:
        raise ValueError("V32 program schema or domain order differs")
    parent = program["parent_evidence"]
    record_path = (repo / parent["v31_record"]).resolve()
    if sha256_file(record_path) != parent["v31_record_sha256"]:
        raise ValueError("V32 V31 result record hash differs")
    record = json.loads(record_path.read_text())
    if (
        record.get("decision", {}).get("classification") != parent["required_classification"]
        or record.get("decision", {}).get("next") != parent["required_next"]
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V32 V31 parent conclusion or firewall differs")
    for domain in DOMAIN_ORDER:
        row = program["development_domains"][domain]
        for split in ("train", "validation"):
            for kind in ("data", "cache"):
                artifact = Path(row[f"{split}_{kind}"])
                if sha256_file(artifact) != row[f"{split}_{kind}_sha256"]:
                    raise ValueError(f"V32 {domain} {split} {kind} hash differs")
                lower = str(artifact).lower()
                if "astrid" in lower or "refl0100n1504" in lower:
                    raise ValueError("V32 firewall path violation")
    return program


def native_multiplicity(count: np.ndarray) -> dict[str, int]:
    values = np.asarray(count, dtype=np.float64)
    if np.any(values < 0) or not np.allclose(values, np.rint(values), atol=1.0e-6):
        raise ValueError("V32 galaxy count field is not nonnegative integer-valued")
    occupied = values > 0
    multiple = values >= 2
    return {
        "cells": int(values.size),
        "occupied_cells": int(occupied.sum()),
        "multi_galaxy_cells": int(multiple.sum()),
        "galaxies": int(np.rint(values.sum())),
        "galaxies_in_multi_galaxy_cells": int(np.rint(values[multiple].sum())),
    }


def _periodic_local_mean_std(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.float64)
    mean = uniform_filter(array, size=3, mode="wrap")
    second = uniform_filter(np.square(array), size=3, mode="wrap")
    std = np.sqrt(np.maximum(second - np.square(mean), 0.0))
    return mean, std


def block_context_rows(
    count: np.ndarray,
    velocity: np.ndarray,
    backbone: np.ndarray,
    truth: np.ndarray,
    factor: int,
    *,
    voxel_mpc_h: float = 0.3125,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return fixed target-free block/context features and block residual."""
    fields = [np.asarray(value, dtype=np.float64) for value in (count, velocity, backbone, truth)]
    if any(value.shape != (64, 64, 64) or not np.isfinite(value).all() for value in fields):
        raise ValueError("V32 block fields must be finite 64-cubes")
    count, velocity, backbone, truth = fields
    count_sum = block_sum(count, factor)
    velocity_sum = block_sum(count * velocity, factor)
    velocity_square_sum = block_sum(count * np.square(velocity), factor)
    occupied = count_sum > 0
    velocity_mean = np.divide(
        velocity_sum, count_sum, out=np.zeros_like(count_sum), where=occupied
    )
    velocity_variance = np.divide(
        velocity_square_sum, count_sum, out=np.zeros_like(count_sum), where=occupied
    ) - np.square(velocity_mean)
    velocity_dispersion = np.sqrt(np.maximum(velocity_variance, 0.0))
    backbone_mean = block_mean(backbone, factor)
    backbone_second = block_mean(np.square(backbone), factor)
    backbone_std = np.sqrt(np.maximum(backbone_second - np.square(backbone_mean), 0.0))
    logcount = np.log1p(count_sum)
    local_rows = []
    for value in (logcount, velocity_mean, backbone_mean):
        local_rows.extend(_periodic_local_mean_std(value))
    grid = 64 // factor
    coordinate = ((np.arange(grid, dtype=np.float64) + 0.5) * factor * voxel_mpc_h - 10.0) / 10.0
    radius = np.sqrt(
        coordinate[:, None, None] ** 2
        + coordinate[None, :, None] ** 2
        + coordinate[None, None, :] ** 2
    )
    feature = np.stack(
        (
            logcount,
            velocity_mean,
            backbone_mean,
            backbone_std,
            radius,
            velocity_dispersion,
            *local_rows,
        ),
        axis=-1,
    )
    target = block_mean(truth - backbone, factor)
    diagnostics = {
        "blocks": int(count_sum.size),
        "occupied_block_fraction": float(occupied.mean()),
        "multi_galaxy_block_fraction": float((count_sum >= 2).mean()),
        "velocity_dispersion_nonzero_fraction": float((velocity_dispersion > 0).mean()),
        "velocity_dispersion_occupied_mean_kms": float(
            velocity_dispersion[occupied].mean() if occupied.any() else 0.0
        ),
    }
    return feature, target, diagnostics


def parity_subsample(value: np.ndarray, cube_index: int) -> np.ndarray:
    array = np.asarray(value)
    offsets = (cube_index % 2, (cube_index // 2) % 2, (cube_index // 4) % 2)
    return array[
        offsets[0] :: 2,
        offsets[1] :: 2,
        offsets[2] :: 2,
    ]


def source_balanced_standardization(rows: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if tuple(rows) != DOMAIN_ORDER:
        raise ValueError("V32 source order differs")
    arrays = [np.asarray(rows[source], dtype=np.float64) for source in DOMAIN_ORDER]
    if any(value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES) for value in arrays):
        raise ValueError("V32 training feature shape differs")
    mean = np.mean([value.mean(axis=0) for value in arrays], axis=0)
    second = np.mean([np.square(value).mean(axis=0) for value in arrays], axis=0)
    std = np.sqrt(np.maximum(second - np.square(mean), 1.0e-12))
    return mean, std


def fit_source_balanced_ridge(
    rows: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    columns: tuple[int, ...],
    mean: np.ndarray,
    std: np.ndarray,
    ridge_lambda: float = 1.0e-3,
) -> np.ndarray:
    dimensions = len(columns) + 1
    gram = np.zeros((dimensions, dimensions), dtype=np.float64)
    response = np.zeros(dimensions, dtype=np.float64)
    for source in DOMAIN_ORDER:
        x = (np.asarray(rows[source], dtype=np.float64)[:, columns] - mean[list(columns)]) / std[list(columns)]
        x = np.column_stack((np.ones(len(x)), x))
        y = np.asarray(targets[source], dtype=np.float64).reshape(-1)
        if len(x) != len(y):
            raise ValueError("V32 ridge rows and targets differ")
        gram += x.T @ x / (len(x) * len(DOMAIN_ORDER))
        response += x.T @ y / (len(x) * len(DOMAIN_ORDER))
    penalty = np.eye(dimensions, dtype=np.float64) * ridge_lambda
    penalty[0, 0] = 0.0
    return np.linalg.solve(gram + penalty, response)


def ridge_metrics(
    rows: np.ndarray,
    target: np.ndarray,
    columns: tuple[int, ...],
    mean: np.ndarray,
    std: np.ndarray,
    coefficient: np.ndarray,
) -> dict[str, float | int]:
    x = (np.asarray(rows, dtype=np.float64)[:, columns] - mean[list(columns)]) / std[list(columns)]
    prediction = coefficient[0] + x @ coefficient[1:]
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    residual = target - prediction
    return {
        "rows": int(len(target)),
        "target_rms": float(np.sqrt(np.mean(np.square(target)))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "pearson_prediction_target": float(np.corrcoef(prediction, target)[0, 1]),
        "prediction_std": float(prediction.std()),
    }


def _collect_split(
    row: Mapping[str, Any], split: str, factor: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    features = []
    targets = []
    multiplicity = {
        "cells": 0,
        "occupied_cells": 0,
        "multi_galaxy_cells": 0,
        "galaxies": 0,
        "galaxies_in_multi_galaxy_cells": 0,
    }
    context_sums: dict[str, float] = {}
    objects = int(row[f"{split}_objects"])
    with h5py.File(row[f"{split}_data"], "r") as data, h5py.File(
        row[f"{split}_cache"], "r"
    ) as cache:
        if len(data["target"]) != objects or len(cache["conditional_mean"]) != objects:
            raise ValueError("V32 split object count differs")
        voxel = float(data.attrs["voxel_mpc_h"])
        for index in range(objects):
            count = np.asarray(data["input"][index, 0], dtype=np.float32)
            velocity = np.asarray(data["input"][index, 1], dtype=np.float32)
            backbone = np.asarray(cache["conditional_mean"][index, 0], dtype=np.float32)
            backbone += np.float32(cache["predicted_residual_dc"][index])
            truth = np.asarray(data["target"][index, 0], dtype=np.float32)
            native = native_multiplicity(count)
            for key, value in native.items():
                multiplicity[key] += value
            feature, target, diagnostic = block_context_rows(
                count, velocity, backbone, truth, factor, voxel_mpc_h=voxel
            )
            if split == "train" and factor == 4:
                feature = parity_subsample(feature, index)
                target = parity_subsample(target, index)
            features.append(feature.reshape(-1, len(FEATURE_NAMES)).astype(np.float32))
            targets.append(target.reshape(-1).astype(np.float32))
            for key, value in diagnostic.items():
                context_sums[key] = context_sums.get(key, 0.0) + float(value)
            if (index + 1) % 64 == 0 or index + 1 == objects:
                print(f"[v32] {split} factor={factor} {index + 1}/{objects}", flush=True)
    summary = {
        **multiplicity,
        "occupied_cell_fraction": multiplicity["occupied_cells"] / multiplicity["cells"],
        "multi_galaxy_cell_fraction_among_occupied": (
            multiplicity["multi_galaxy_cells"] / multiplicity["occupied_cells"]
        ),
        "galaxy_fraction_in_multi_galaxy_cells": (
            multiplicity["galaxies_in_multi_galaxy_cells"] / multiplicity["galaxies"]
        ),
        "mean_context_diagnostics_per_cube": {
            key: value / objects for key, value in context_sums.items()
        },
    }
    return np.concatenate(features), np.concatenate(targets), summary


def audit_factor(program: dict[str, Any], factor: int) -> dict[str, Any]:
    train_features: dict[str, np.ndarray] = {}
    train_targets: dict[str, np.ndarray] = {}
    validation_features: dict[str, np.ndarray] = {}
    validation_targets: dict[str, np.ndarray] = {}
    split_summary: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        row = program["development_domains"][domain]
        train_features[domain], train_targets[domain], train_summary = _collect_split(
            row, "train", factor
        )
        validation_features[domain], validation_targets[domain], validation_summary = _collect_split(
            row, "validation", factor
        )
        split_summary[domain] = {"train": train_summary, "validation": validation_summary}
    mean, std = source_balanced_standardization(train_features)
    models = {}
    for model, columns in MODEL_COLUMNS.items():
        coefficient = fit_source_balanced_ridge(
            train_features, train_targets, columns, mean, std
        )
        domains = {
            source: ridge_metrics(
                validation_features[source], validation_targets[source],
                columns, mean, std, coefficient,
            )
            for source in DOMAIN_ORDER
        }
        models[model] = {
            "features": [FEATURE_NAMES[index] for index in columns],
            "coefficient_standardized": coefficient.tolist(),
            "validation": domains,
        }
    for source in DOMAIN_ORDER:
        base = float(models["base"]["validation"][source]["rmse"])
        for model in ("plus_sig_v", "plus_local_context"):
            models[model]["validation"][source]["rmse_over_base"] = float(
                models[model]["validation"][source]["rmse"] / base
            )
    return {
        "factor": factor,
        "scale_mpc_h": factor * 0.3125,
        "feature_names": list(FEATURE_NAMES),
        "train_feature_equal_source_mean": mean.tolist(),
        "train_feature_equal_source_std": std.tolist(),
        "split_summary": split_summary,
        "models": models,
    }


def evaluate(program_path: Path, repo: Path) -> dict[str, Any]:
    program = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V32 audit requires a clean committed worktree")
    factors = {str(factor): audit_factor(program, factor) for factor in FACTORS}
    sig_supported_factors = []
    local_supported_factors = []
    native_material = False
    for factor, row in factors.items():
        sig_ratios = [
            row["models"]["plus_sig_v"]["validation"][domain]["rmse_over_base"]
            for domain in DOMAIN_ORDER
        ]
        local_ratios = [
            row["models"]["plus_local_context"]["validation"][domain]["rmse_over_base"]
            for domain in DOMAIN_ORDER
        ]
        if all(value <= 0.99 for value in sig_ratios):
            sig_supported_factors.append(int(factor))
        if all(value <= 0.98 for value in local_ratios):
            local_supported_factors.append(int(factor))
        for domain in DOMAIN_ORDER:
            for split in ("train", "validation"):
                fraction = row["split_summary"][domain][split][
                    "galaxy_fraction_in_multi_galaxy_cells"
                ]
                native_material = native_material or fraction >= 0.10
    sig_supported = bool(sig_supported_factors)
    local_supported = bool(local_supported_factors)
    if local_supported and (sig_supported or native_material):
        classification = "local_multiscale_context_and_velocity_second_moment_supported"
        next_step = "freeze_multiscale_locally_conditioned_physical_residual_likelihood_with_uncertainty_channel_comparison"
    elif local_supported:
        classification = "local_multiscale_context_supported_but_velocity_second_moment_not_common"
        next_step = "freeze_local_multiscale_model_and_keep_CF4_observational_sigma_as_a_separate_uncertainty_channel"
    elif sig_supported or native_material:
        classification = "velocity_second_moment_supported_without_linear_local_patch_gain"
        next_step = "rebuild_development_inputs_with_explicit_velocity_second_moments_before_generative_training"
    else:
        classification = "fixed_linear_context_controls_do_not_explain_V31_residual_failure"
        next_step = "audit_nonlinear_conditional_sufficiency_and_residual_spatial_copula_before_expanding_channels"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_development_only_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "factors": factors,
        "native_within_cell_velocity_dispersion_recoverable": False,
        "native_multiplicity_material": native_material,
        "recoverable_sig_v_supported_factors": sig_supported_factors,
        "local_patch_supported_factors": local_supported_factors,
        "classification": classification,
        "next": next_step,
        "posthoc_Ak_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("V32 refuses to overwrite its audit")
    report = evaluate(args.program.resolve(), args.repo.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
