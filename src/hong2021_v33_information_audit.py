#!/usr/bin/env python
"""Audit the incremental information in the frozen V33 intrinsic sig_v grid."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v30_backbone_audit import block_mean, block_sum
from hong2021_v31_copula import DOMAIN_ORDER
from hong2021_v32_context_audit import (
    PROGRAM_SHA256 as V32_PROGRAM_SHA256,
    _periodic_local_mean_std,
    fit_source_balanced_ridge,
    load_program as load_v32_program,
    parity_subsample,
    ridge_metrics,
)
from hong2021_v33_kinematic_data import (
    CHANNELS,
    OUTPUT_SCHEMA,
    PROGRAM_SCHEMA,
    PROGRAM_SHA256,
    load_program as load_v33_program,
)


SCHEMA = "hong2021-v33-intrinsic-velocity-information-audit-v1"
FACTORS = (4, 8)
FEATURE_NAMES = (
    "log1p_block_count",
    "block_mean_velocity_kms",
    "backbone_mean_y",
    "backbone_std_y",
    "radius_over_half_width",
    "recoverable_between_native_cell_velocity_dispersion_kms",
    "exact_individual_galaxy_velocity_dispersion_kms",
    "local_logcount_mean",
    "local_logcount_std",
    "local_velocity_mean",
    "local_velocity_std",
    "local_backbone_mean",
    "local_backbone_std",
)
MODEL_COLUMNS = {
    "base": tuple(range(5)),
    "plus_recoverable_sig_v": tuple(range(6)),
    "plus_exact_sig_v": (0, 1, 2, 3, 4, 6),
    "plus_exact_sig_v_and_local_context": (0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12),
}


def _load_inputs(program_path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    v33 = load_v33_program(program_path, repo)
    if v33.get("schema") != PROGRAM_SCHEMA:
        raise ValueError("V33 information-audit program schema differs")
    v32_path = repo / "config/hong2021_v32_context_audit_program.json"
    if sha256_file(v32_path) != V32_PROGRAM_SHA256:
        raise ValueError("V33 frozen V32 program hash differs")
    v32 = load_v32_program(v32_path, repo)
    for domain in DOMAIN_ORDER:
        if domain not in v33["datasets"]:
            raise ValueError("V33 domain order or membership differs")
        for split in ("train", "validation"):
            new = v33["datasets"][domain][split]
            old = v32["development_domains"][domain]
            if Path(new["source"]).resolve() != Path(old[f"{split}_data"]).resolve():
                raise ValueError("V33 source does not match frozen V32 data")
            report_path = Path(new["output"]).with_suffix(".json")
            if not report_path.is_file():
                raise FileNotFoundError(f"V33 build report absent: {report_path}")
            report = json.loads(report_path.read_text())
            if (
                report.get("schema") != OUTPUT_SCHEMA
                or report.get("status") != "complete"
                or report.get("program_sha256") != PROGRAM_SHA256
                or report.get("source_sha256") != new["source_sha256"]
                or report.get("count_max_abs_difference_from_v14") != 0.0
                or report.get("mean_velocity_max_abs_difference_from_v14_kms") != 0.0
                or report.get("observational_sigma_mean_included") is not False
                or report.get("Astrid_accessed") is not False
                or report.get("historical_EAGLE_accessed") is not False
            ):
                raise ValueError("V33 build report failed hard acceptance")
            if sha256_file(Path(new["output"])) != report["output_sha256"]:
                raise ValueError("V33 output hash differs from build report")
    return v33, v32


def block_information_rows(
    count: np.ndarray,
    velocity: np.ndarray,
    native_dispersion: np.ndarray,
    backbone: np.ndarray,
    truth: np.ndarray,
    factor: int,
    *,
    voxel_mpc_h: float = 0.3125,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    fields = [
        np.asarray(value, dtype=np.float64)
        for value in (count, velocity, native_dispersion, backbone, truth)
    ]
    if any(value.shape != (64, 64, 64) or not np.isfinite(value).all() for value in fields):
        raise ValueError("V33 information fields must be finite 64-cubes")
    count, velocity, native_dispersion, backbone, truth = fields
    if np.any(count < 0) or np.any(native_dispersion < 0):
        raise ValueError("V33 count/dispersion is negative")
    if np.any(native_dispersion[count < 2] != 0):
        raise ValueError("V33 dispersion is nonzero below two galaxies")

    count_sum = block_sum(count, factor)
    velocity_sum = block_sum(count * velocity, factor)
    between_second_sum = block_sum(count * np.square(velocity), factor)
    within_centered_second_sum = block_sum(
        np.maximum(count - 1.0, 0.0) * np.square(native_dispersion), factor
    )
    occupied = count_sum > 0
    velocity_mean = np.divide(
        velocity_sum, count_sum, out=np.zeros_like(count_sum), where=occupied
    )
    recoverable_variance = np.divide(
        between_second_sum,
        count_sum,
        out=np.zeros_like(count_sum),
        where=occupied,
    ) - np.square(velocity_mean)
    exact_variance = np.divide(
        between_second_sum + within_centered_second_sum,
        count_sum,
        out=np.zeros_like(count_sum),
        where=occupied,
    ) - np.square(velocity_mean)
    recoverable_dispersion = np.sqrt(np.maximum(recoverable_variance, 0.0))
    exact_dispersion = np.sqrt(np.maximum(exact_variance, 0.0))

    backbone_mean = block_mean(backbone, factor)
    backbone_second = block_mean(np.square(backbone), factor)
    backbone_std = np.sqrt(np.maximum(backbone_second - np.square(backbone_mean), 0.0))
    logcount = np.log1p(count_sum)
    local_rows = []
    for value in (logcount, velocity_mean, backbone_mean):
        local_rows.extend(_periodic_local_mean_std(value))
    grid = 64 // factor
    coordinate = (
        (np.arange(grid, dtype=np.float64) + 0.5) * factor * voxel_mpc_h - 10.0
    ) / 10.0
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
            recoverable_dispersion,
            exact_dispersion,
            *local_rows,
        ),
        axis=-1,
    )
    target = block_mean(truth - backbone, factor)
    exact_increment = np.maximum(exact_dispersion - recoverable_dispersion, 0.0)
    diagnostic = {
        "blocks": int(count_sum.size),
        "occupied_block_fraction": float(occupied.mean()),
        "recoverable_dispersion_occupied_mean_kms": float(
            recoverable_dispersion[occupied].mean() if occupied.any() else 0.0
        ),
        "exact_dispersion_occupied_mean_kms": float(
            exact_dispersion[occupied].mean() if occupied.any() else 0.0
        ),
        "exact_minus_recoverable_occupied_mean_kms": float(
            exact_increment[occupied].mean() if occupied.any() else 0.0
        ),
        "blocks_with_strict_intrinsic_increment_fraction": float(
            (exact_increment > 1.0e-6).mean()
        ),
    }
    return feature, target, diagnostic


def source_balanced_standardization(
    rows: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    if tuple(rows) != DOMAIN_ORDER:
        raise ValueError("V33 source order differs")
    arrays = [np.asarray(rows[source], dtype=np.float64) for source in DOMAIN_ORDER]
    if any(value.ndim != 2 or value.shape[1] != len(FEATURE_NAMES) for value in arrays):
        raise ValueError("V33 training feature shape differs")
    mean = np.mean([value.mean(axis=0) for value in arrays], axis=0)
    second = np.mean([np.square(value).mean(axis=0) for value in arrays], axis=0)
    std = np.sqrt(np.maximum(second - np.square(mean), 1.0e-12))
    return mean, std


def _collect_split(
    data_path: Path,
    cache_path: Path,
    objects: int,
    split: str,
    factor: int,
    domain: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    features = []
    targets = []
    diagnostic_sums: dict[str, float] = {}
    with h5py.File(data_path, "r") as data, h5py.File(cache_path, "r") as cache:
        if (
            tuple(data["input"].shape) != (objects, 3, 64, 64, 64)
            or str(data.attrs.get("schema", "")) != OUTPUT_SCHEMA
            or str(data.attrs.get("channels", "")) != CHANNELS
            or data.attrs.get("complete") != np.bool_(True)
            or len(cache["conditional_mean"]) != objects
        ):
            raise ValueError("V33 input/cache shape or metadata differs")
        voxel = float(data.attrs["voxel_mpc_h"])
        for index in range(objects):
            count = np.asarray(data["input"][index, 0], dtype=np.float32)
            velocity = np.asarray(data["input"][index, 1], dtype=np.float32)
            dispersion = np.asarray(data["input"][index, 2], dtype=np.float32)
            backbone = np.asarray(cache["conditional_mean"][index, 0], dtype=np.float32)
            backbone += np.float32(cache["predicted_residual_dc"][index])
            truth = np.asarray(data["target"][index, 0], dtype=np.float32)
            feature, target, diagnostic = block_information_rows(
                count,
                velocity,
                dispersion,
                backbone,
                truth,
                factor,
                voxel_mpc_h=voxel,
            )
            if split == "train" and factor == 4:
                feature = parity_subsample(feature, index)
                target = parity_subsample(target, index)
            features.append(feature.reshape(-1, len(FEATURE_NAMES)).astype(np.float32))
            targets.append(target.reshape(-1).astype(np.float32))
            for key, value in diagnostic.items():
                diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(value)
            if (index + 1) % 64 == 0 or index + 1 == objects:
                print(
                    f"[v33-audit] {domain} {split} factor={factor} "
                    f"{index + 1}/{objects}",
                    flush=True,
                )
    return (
        np.concatenate(features),
        np.concatenate(targets),
        {key: value / objects for key, value in diagnostic_sums.items()},
    )


def audit_factor(v33: dict[str, Any], v32: dict[str, Any], factor: int) -> dict[str, Any]:
    train_features: dict[str, np.ndarray] = {}
    train_targets: dict[str, np.ndarray] = {}
    validation_features: dict[str, np.ndarray] = {}
    validation_targets: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        diagnostics[domain] = {}
        for split, feature_store, target_store in (
            ("train", train_features, train_targets),
            ("validation", validation_features, validation_targets),
        ):
            artifact = v33["datasets"][domain][split]
            old = v32["development_domains"][domain]
            feature_store[domain], target_store[domain], diagnostics[domain][split] = _collect_split(
                Path(artifact["output"]),
                Path(old[f"{split}_cache"]),
                int(artifact["objects"]),
                split,
                factor,
                domain,
            )
    mean, std = source_balanced_standardization(train_features)
    models: dict[str, Any] = {}
    for model, columns in MODEL_COLUMNS.items():
        coefficient = fit_source_balanced_ridge(
            train_features, train_targets, columns, mean, std, ridge_lambda=0.001
        )
        models[model] = {
            "features": [FEATURE_NAMES[index] for index in columns],
            "coefficient_standardized": coefficient.tolist(),
            "validation": {
                domain: ridge_metrics(
                    validation_features[domain],
                    validation_targets[domain],
                    columns,
                    mean,
                    std,
                    coefficient,
                )
                for domain in DOMAIN_ORDER
            },
        }
    for domain in DOMAIN_ORDER:
        base = float(models["base"]["validation"][domain]["rmse"])
        lower = float(models["plus_recoverable_sig_v"]["validation"][domain]["rmse"])
        for model in models:
            models[model]["validation"][domain]["rmse_over_base"] = float(
                models[model]["validation"][domain]["rmse"] / base
            )
        models["plus_exact_sig_v"]["validation"][domain][
            "rmse_over_recoverable_sig_v"
        ] = float(models["plus_exact_sig_v"]["validation"][domain]["rmse"] / lower)
    return {
        "factor": factor,
        "scale_mpc_h": factor * 0.3125,
        "feature_names": list(FEATURE_NAMES),
        "train_feature_equal_source_mean": mean.tolist(),
        "train_feature_equal_source_std": std.tolist(),
        "mean_diagnostics_per_cube": diagnostics,
        "models": models,
    }


def evaluate(program_path: Path, repo: Path) -> dict[str, Any]:
    v33, v32 = _load_inputs(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V33 information audit requires a clean committed worktree")
    factors = {str(factor): audit_factor(v33, v32, factor) for factor in FACTORS}
    intrinsic_supported = []
    unique_supported = []
    for factor, row in factors.items():
        if all(
            row["models"]["plus_exact_sig_v"]["validation"][domain]["rmse_over_base"]
            <= 0.99
            for domain in DOMAIN_ORDER
        ):
            intrinsic_supported.append(int(factor))
        if all(
            row["models"]["plus_exact_sig_v"]["validation"][domain][
                "rmse_over_recoverable_sig_v"
            ]
            <= 0.995
            for domain in DOMAIN_ORDER
        ):
            unique_supported.append(int(factor))
    if intrinsic_supported and unique_supported:
        classification = "intrinsic_sig_v_and_unique_second_moment_information_supported"
        next_step = "freeze_matched_two_channel_versus_kinematic_three_channel_nonlinear_backbone_ablation"
    elif intrinsic_supported:
        classification = "intrinsic_sig_v_supported_but_unique_increment_over_retained_means_is_weak"
        next_step = "freeze_matched_nonlinear_backbone_ablation_and_record_weak_unique_linear_information"
    else:
        classification = "intrinsic_sig_v_not_supported_by_fixed_linear_common_domain_gate"
        next_step = "retain_physical_CF4_channel_boundary_but_audit_nonlinear_residual_spatial_sufficiency_before_full_generative_run"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_development_only_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "v32_program_sha256": V32_PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "pooled_dispersion_denominator": "N; population dispersion for direct comparison to the V32 between-cell lower bound",
        "factors": factors,
        "intrinsic_sig_v_supported_factors": intrinsic_supported,
        "unique_second_moment_supported_factors": unique_supported,
        "classification": classification,
        "next": next_step,
        "observational_sigma_mean_included": False,
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
        raise FileExistsError("V33 refuses to overwrite its information audit")
    report = evaluate(args.program.resolve(), args.repo.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
