#!/usr/bin/env python
"""Frozen V34 nonlinear oriented/multiscale residual-sufficiency audit."""
from __future__ import annotations

import argparse
import json
import os
from itertools import product
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v30_backbone_audit import block_mean, block_sum
from hong2021_v31_copula import DOMAIN_ORDER
from hong2021_v33_kinematic_data import CHANNELS, OUTPUT_SCHEMA


PROGRAM_SCHEMA = "hong2021-v34-oriented-multiscale-nonlinear-sufficiency-program-v1"
PROGRAM_SHA256 = "82f7027589484e711a4956c56128073590374aea0e8a7484323ba95e6f7b8314"
SCHEMA = "hong2021-v34-oriented-multiscale-nonlinear-sufficiency-audit-v1"
FACTORS = (4, 8)
PATCH_OFFSETS = tuple(product((-1, 0, 1), repeat=3))
SCALAR_FEATURES = (
    "central_log1p_block_count",
    "central_block_mean_velocity_kms",
    "central_exact_population_velocity_dispersion_kms",
    "central_backbone_mean_y",
    "central_backbone_std_y",
    "radius_over_half_width",
)
PATCH_FIELDS = (
    "log1p_block_count",
    "block_mean_velocity_kms",
    "exact_population_velocity_dispersion_kms",
    "backbone_mean_y",
)
MODEL_FEATURE_COUNTS = {
    "nonlinear_scalar": len(SCALAR_FEATURES),
    "nonlinear_oriented_single_scale": len(SCALAR_FEATURES)
    + len(PATCH_FIELDS) * len(PATCH_OFFSETS),
    "nonlinear_oriented_multiscale": len(SCALAR_FEATURES)
    + 2 * len(PATCH_FIELDS) * len(PATCH_OFFSETS),
}


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(path.resolve()) != PROGRAM_SHA256:
        raise ValueError("V34 program hash differs")
    program = json.loads(path.read_text())
    if program.get("schema") != PROGRAM_SCHEMA or tuple(program["development_domains"]) != DOMAIN_ORDER:
        raise ValueError("V34 program schema or domain order differs")
    parent = program["parent_evidence"]
    record_path = (repo / parent["v33_record"]).resolve()
    if sha256_file(record_path) != parent["v33_record_sha256"]:
        raise ValueError("V34 V33 result record hash differs")
    record = json.loads(record_path.read_text())
    information = record.get("information_audit", {})
    if (
        information.get("classification") != parent["required_classification"]
        or information.get("next") != parent["required_next"]
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V34 V33 parent conclusion or firewall differs")
    reference_path = Path(parent["v33_information_audit"])
    if sha256_file(reference_path) != parent["v33_information_audit_sha256"]:
        raise ValueError("V34 frozen V33 information audit hash differs")
    reference = json.loads(reference_path.read_text())
    for domain in DOMAIN_ORDER:
        row = program["development_domains"][domain]
        for split in ("train", "validation"):
            for kind in ("data", "cache"):
                artifact = Path(row[f"{split}_{kind}"])
                if sha256_file(artifact) != row[f"{split}_{kind}_sha256"]:
                    raise ValueError(f"V34 {domain} {split} {kind} hash differs")
                lower = str(artifact).lower()
                if "astrid" in lower or "refl0100n1504" in lower:
                    raise ValueError("V34 firewall path violation")
    return program, reference


def pooled_fields(
    count: np.ndarray,
    velocity: np.ndarray,
    native_dispersion: np.ndarray,
    backbone: np.ndarray,
    factor: int,
) -> dict[str, np.ndarray]:
    values = [
        np.asarray(field, dtype=np.float64)
        for field in (count, velocity, native_dispersion, backbone)
    ]
    if any(field.shape != (64, 64, 64) or not np.isfinite(field).all() for field in values):
        raise ValueError("V34 input fields must be finite 64-cubes")
    count, velocity, native_dispersion, backbone = values
    if np.any(count < 0) or np.any(native_dispersion < 0) or np.any(native_dispersion[count < 2] != 0):
        raise ValueError("V34 count or intrinsic dispersion is invalid")
    count_sum = block_sum(count, factor)
    occupied = count_sum > 0
    velocity_sum = block_sum(count * velocity, factor)
    velocity_mean = np.divide(
        velocity_sum, count_sum, out=np.zeros_like(count_sum), where=occupied
    )
    individual_velocity_second_sum = block_sum(
        count * np.square(velocity)
        + np.maximum(count - 1.0, 0.0) * np.square(native_dispersion),
        factor,
    )
    velocity_variance = np.divide(
        individual_velocity_second_sum,
        count_sum,
        out=np.zeros_like(count_sum),
        where=occupied,
    ) - np.square(velocity_mean)
    backbone_mean = block_mean(backbone, factor)
    backbone_second = block_mean(np.square(backbone), factor)
    return {
        "log1p_block_count": np.log1p(count_sum),
        "block_mean_velocity_kms": velocity_mean,
        "exact_population_velocity_dispersion_kms": np.sqrt(
            np.maximum(velocity_variance, 0.0)
        ),
        "backbone_mean_y": backbone_mean,
        "backbone_std_y": np.sqrt(
            np.maximum(backbone_second - np.square(backbone_mean), 0.0)
        ),
    }


def periodic_oriented_patch(value: np.ndarray) -> np.ndarray:
    """Return value[i+dx,j+dy,k+dz] in frozen lexicographic offset order."""
    field = np.asarray(value)
    if field.ndim != 3 or len(set(field.shape)) != 1:
        raise ValueError("V34 patch field must be cubic")
    return np.stack(
        [np.roll(field, shift=(-dx, -dy, -dz), axis=(0, 1, 2)) for dx, dy, dz in PATCH_OFFSETS],
        axis=-1,
    )


def repeat_parent_patch(parent_patch: np.ndarray) -> np.ndarray:
    value = np.asarray(parent_patch)
    if value.ndim != 4 or len(set(value.shape[:3])) != 1:
        raise ValueError("V34 parent patch shape differs")
    return np.repeat(np.repeat(np.repeat(value, 2, axis=0), 2, axis=1), 2, axis=2)


def multiscale_features(
    count: np.ndarray,
    velocity: np.ndarray,
    native_dispersion: np.ndarray,
    backbone: np.ndarray,
    truth: np.ndarray,
    factor: int,
    *,
    voxel_mpc_h: float = 0.3125,
) -> tuple[np.ndarray, np.ndarray]:
    current = pooled_fields(count, velocity, native_dispersion, backbone, factor)
    parent = pooled_fields(count, velocity, native_dispersion, backbone, factor * 2)
    grid = 64 // factor
    coordinate = (
        (np.arange(grid, dtype=np.float64) + 0.5) * factor * voxel_mpc_h - 10.0
    ) / 10.0
    radius = np.sqrt(
        coordinate[:, None, None] ** 2
        + coordinate[None, :, None] ** 2
        + coordinate[None, None, :] ** 2
    )
    scalar = np.stack(
        (
            current["log1p_block_count"],
            current["block_mean_velocity_kms"],
            current["exact_population_velocity_dispersion_kms"],
            current["backbone_mean_y"],
            current["backbone_std_y"],
            radius,
        ),
        axis=-1,
    )
    current_patches = np.concatenate(
        [periodic_oriented_patch(current[name]) for name in PATCH_FIELDS], axis=-1
    )
    parent_patches = np.concatenate(
        [repeat_parent_patch(periodic_oriented_patch(parent[name])) for name in PATCH_FIELDS],
        axis=-1,
    )
    feature = np.concatenate((scalar, current_patches, parent_patches), axis=-1)
    if feature.shape != (grid, grid, grid, MODEL_FEATURE_COUNTS["nonlinear_oriented_multiscale"]):
        raise RuntimeError("V34 multiscale feature shape differs")
    target = block_mean(np.asarray(truth, dtype=np.float64) - np.asarray(backbone, dtype=np.float64), factor)
    return feature.astype(np.float32), target.astype(np.float32)


def _open_validated(row: dict[str, Any], split: str) -> tuple[h5py.File, h5py.File]:
    data = h5py.File(row[f"{split}_data"], "r")
    cache = h5py.File(row[f"{split}_cache"], "r")
    objects = int(row[f"{split}_objects"])
    valid = (
        tuple(data["input"].shape) == (objects, 3, 64, 64, 64)
        and str(data.attrs.get("schema", "")) == OUTPUT_SCHEMA
        and str(data.attrs.get("channels", "")) == CHANNELS
        and bool(data.attrs.get("complete", False))
        and tuple(cache["conditional_mean"].shape) == (objects, 1, 64, 64, 64)
        and tuple(cache["predicted_residual_dc"].shape) == (objects,)
    )
    if not valid:
        data.close()
        cache.close()
        raise ValueError("V34 data/cache metadata or shape differs")
    return data, cache


def _cube_features(
    data: h5py.File, cache: h5py.File, index: int, factor: int
) -> tuple[np.ndarray, np.ndarray]:
    count = np.asarray(data["input"][index, 0], dtype=np.float32)
    velocity = np.asarray(data["input"][index, 1], dtype=np.float32)
    dispersion = np.asarray(data["input"][index, 2], dtype=np.float32)
    backbone = np.asarray(cache["conditional_mean"][index, 0], dtype=np.float32)
    backbone += np.float32(cache["predicted_residual_dc"][index])
    truth = np.asarray(data["target"][index, 0], dtype=np.float32)
    return multiscale_features(
        count,
        velocity,
        dispersion,
        backbone,
        truth,
        factor,
        voxel_mpc_h=float(data.attrs["voxel_mpc_h"]),
    )


def _parity_rows(value: np.ndarray, cube_index: int) -> np.ndarray:
    offsets = (cube_index % 2, (cube_index // 2) % 2, (cube_index // 4) % 2)
    return value[offsets[0] :: 2, offsets[1] :: 2, offsets[2] :: 2]


def select_train_rows(
    row: dict[str, Any],
    domain: str,
    factor: int,
    selected_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    objects = int(row["train_objects"])
    rows_per_cube = 512
    expected_total = objects * rows_per_cube
    selected = np.asarray(selected_indices, dtype=np.int64)
    if (
        selected.shape != (65536,)
        or np.any(np.diff(selected) <= 0)
        or selected[0] < 0
        or selected[-1] >= expected_total
    ):
        raise ValueError("V34 selected train row indices differ")
    feature_parts = []
    target_parts = []
    data, cache = _open_validated(row, "train")
    try:
        for index in range(objects):
            feature, target = _cube_features(data, cache, index, factor)
            if factor == 4:
                feature = _parity_rows(feature, index)
                target = _parity_rows(target, index)
            feature = feature.reshape(rows_per_cube, -1)
            target = target.reshape(rows_per_cube)
            lower = index * rows_per_cube
            left = np.searchsorted(selected, lower, side="left")
            right = np.searchsorted(selected, lower + rows_per_cube, side="left")
            local = selected[left:right] - lower
            if len(local):
                feature_parts.append(feature[local])
                target_parts.append(target[local])
            if (index + 1) % 64 == 0 or index + 1 == objects:
                print(
                    f"[v34] collect {domain} factor={factor} {index + 1}/{objects}",
                    flush=True,
                )
    finally:
        data.close()
        cache.close()
    features = np.concatenate(feature_parts)
    targets = np.concatenate(target_parts)
    if features.shape != (65536, MODEL_FEATURE_COUNTS["nonlinear_oriented_multiscale"]):
        raise RuntimeError("V34 selected train feature shape differs")
    return features, targets


class StreamingMetrics:
    def __init__(self) -> None:
        self.n = 0
        self.sum_prediction = 0.0
        self.sum_target = 0.0
        self.sum_prediction2 = 0.0
        self.sum_target2 = 0.0
        self.sum_cross = 0.0
        self.sum_error2 = 0.0

    def add(self, prediction: np.ndarray, target: np.ndarray) -> None:
        prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        if prediction.shape != target.shape or not np.isfinite(prediction).all():
            raise ValueError("V34 prediction shape or finiteness differs")
        self.n += len(target)
        self.sum_prediction += float(prediction.sum())
        self.sum_target += float(target.sum())
        self.sum_prediction2 += float(np.square(prediction).sum())
        self.sum_target2 += float(np.square(target).sum())
        self.sum_cross += float((prediction * target).sum())
        self.sum_error2 += float(np.square(prediction - target).sum())

    def result(self) -> dict[str, float | int]:
        if self.n == 0:
            raise RuntimeError("V34 metric accumulator is empty")
        mean_prediction = self.sum_prediction / self.n
        mean_target = self.sum_target / self.n
        variance_prediction = max(self.sum_prediction2 / self.n - mean_prediction**2, 0.0)
        variance_target = max(self.sum_target2 / self.n - mean_target**2, 0.0)
        covariance = self.sum_cross / self.n - mean_prediction * mean_target
        denominator = np.sqrt(variance_prediction * variance_target)
        return {
            "rows": self.n,
            "target_rms": float(np.sqrt(self.sum_target2 / self.n)),
            "rmse": float(np.sqrt(self.sum_error2 / self.n)),
            "pearson_prediction_target": float(covariance / denominator if denominator > 0 else 0.0),
            "prediction_std": float(np.sqrt(variance_prediction)),
        }


def _new_model(specification: dict[str, Any]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=specification["loss"],
        learning_rate=float(specification["learning_rate"]),
        max_iter=int(specification["max_iter"]),
        max_leaf_nodes=int(specification["max_leaf_nodes"]),
        max_depth=specification["max_depth"],
        min_samples_leaf=int(specification["min_samples_leaf"]),
        l2_regularization=float(specification["l2_regularization"]),
        max_bins=int(specification["max_bins"]),
        early_stopping=bool(specification["early_stopping"]),
        random_state=int(specification["random_state"]),
    )


def _array_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    metric = StreamingMetrics()
    metric.add(prediction, target)
    return metric.result()


def audit_factor(
    program: dict[str, Any], reference: dict[str, Any], factor: int
) -> dict[str, Any]:
    row_rule = program["fixed_training_rows"]
    generator = np.random.default_rng(int(row_rule["seeds"][f"factor_{factor}"]))
    train_features = []
    train_targets = []
    selected_digests: dict[str, str] = {}
    import hashlib

    for domain in DOMAIN_ORDER:
        objects = int(program["development_domains"][domain]["train_objects"])
        selected = np.sort(generator.permutation(objects * 512)[: int(row_rule["rows_per_source_per_factor"])]).astype(np.int64)
        selected_digests[domain] = hashlib.sha256(selected.tobytes()).hexdigest()
        features, target = select_train_rows(
            program["development_domains"][domain], domain, factor, selected
        )
        train_features.append(features)
        train_targets.append(target)
    x_train = np.concatenate(train_features)
    y_train = np.concatenate(train_targets)
    models: dict[str, HistGradientBoostingRegressor] = {}
    report_models: dict[str, Any] = {}
    learner = program["fixed_nonlinear_learner"]
    for name in learner["models"]:
        columns = MODEL_FEATURE_COUNTS[name]
        model = _new_model(learner)
        print(f"[v34] fit factor={factor} model={name} rows={len(y_train)} features={columns}", flush=True)
        model.fit(x_train[:, :columns], y_train)
        prediction = model.predict(x_train[:, :columns])
        models[name] = model
        report_models[name] = {
            "features": columns,
            "iterations": int(model.n_iter_),
            "train": _array_metrics(prediction, y_train),
            "validation": {},
        }

    for domain in DOMAIN_ORDER:
        row = program["development_domains"][domain]
        objects = int(row["validation_objects"])
        metrics = {name: StreamingMetrics() for name in models}
        data, cache = _open_validated(row, "validation")
        try:
            for index in range(objects):
                feature, target = _cube_features(data, cache, index, factor)
                feature = feature.reshape(-1, feature.shape[-1])
                target = target.reshape(-1)
                for name, model in models.items():
                    columns = MODEL_FEATURE_COUNTS[name]
                    metrics[name].add(model.predict(feature[:, :columns]), target)
                if (index + 1) % 32 == 0 or index + 1 == objects:
                    print(
                        f"[v34] validate {domain} factor={factor} {index + 1}/{objects}",
                        flush=True,
                    )
        finally:
            data.close()
            cache.close()
        linear_rmse = float(reference["factors"][str(factor)]["models"]["base"]["validation"][domain]["rmse"])
        for name in models:
            result = metrics[name].result()
            result["rmse_over_v33_linear_base"] = float(result["rmse"] / linear_rmse)
            report_models[name]["validation"][domain] = result
        scalar_rmse = float(report_models["nonlinear_scalar"]["validation"][domain]["rmse"])
        single_rmse = float(
            report_models["nonlinear_oriented_single_scale"]["validation"][domain]["rmse"]
        )
        report_models["nonlinear_oriented_single_scale"]["validation"][domain][
            "rmse_over_nonlinear_scalar"
        ] = float(single_rmse / scalar_rmse)
        report_models["nonlinear_oriented_multiscale"]["validation"][domain][
            "rmse_over_nonlinear_scalar"
        ] = float(
            report_models["nonlinear_oriented_multiscale"]["validation"][domain]["rmse"]
            / scalar_rmse
        )
        report_models["nonlinear_oriented_multiscale"]["validation"][domain][
            "rmse_over_oriented_single_scale"
        ] = float(
            report_models["nonlinear_oriented_multiscale"]["validation"][domain]["rmse"]
            / single_rmse
        )
    return {
        "factor": factor,
        "scale_mpc_h": factor * 0.3125,
        "coarser_context_scale_mpc_h": factor * 2 * 0.3125,
        "patch_offset_order": [list(offset) for offset in PATCH_OFFSETS],
        "scalar_features": list(SCALAR_FEATURES),
        "patch_fields": list(PATCH_FIELDS),
        "selected_train_row_sha256": selected_digests,
        "train_rows": int(len(y_train)),
        "models": report_models,
    }


def evaluate(program_path: Path, repo: Path) -> dict[str, Any]:
    program, reference = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V34 audit requires a clean committed worktree")
    factors = {str(factor): audit_factor(program, reference, factor) for factor in FACTORS}
    scalar_supported = []
    oriented_supported = []
    multiscale_supported = []
    material_supported = []
    for factor, row in factors.items():
        models = row["models"]
        if all(
            models["nonlinear_scalar"]["validation"][domain]["rmse_over_v33_linear_base"]
            <= 0.98
            for domain in DOMAIN_ORDER
        ):
            scalar_supported.append(int(factor))
        if all(
            models["nonlinear_oriented_single_scale"]["validation"][domain][
                "rmse_over_nonlinear_scalar"
            ]
            <= 0.98
            for domain in DOMAIN_ORDER
        ):
            oriented_supported.append(int(factor))
        if all(
            models["nonlinear_oriented_multiscale"]["validation"][domain][
                "rmse_over_oriented_single_scale"
            ]
            <= 0.99
            for domain in DOMAIN_ORDER
        ):
            multiscale_supported.append(int(factor))
        if all(
            models["nonlinear_oriented_multiscale"]["validation"][domain][
                "rmse_over_v33_linear_base"
            ]
            <= 0.95
            for domain in DOMAIN_ORDER
        ):
            material_supported.append(int(factor))
    if oriented_supported:
        classification = "nonlinear_oriented_local_residual_mean_is_supported"
        next_step = "freeze_matched_local_conditional_mean_backbone_ablation_before_conditional_residual_likelihood"
    elif scalar_supported:
        classification = "nonlinear_scalar_conditioning_supported_without_oriented_patch_gain"
        next_step = "replace_scalar_binned_conditioning_with_nonlinear_scalar_conditional_mean"
    else:
        classification = "local_residual_mean_not_predictable_by_frozen_nonlinear_audit"
        next_step = "freeze_conditional_multiscale_residual_spectrum_and_phase_coupling_audit"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_development_only_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "sklearn_version": sklearn.__version__,
        "factors": factors,
        "nonlinear_scalar_supported_factors": scalar_supported,
        "oriented_patch_supported_factors": oriented_supported,
        "multiscale_increment_supported_factors": multiscale_supported,
        "material_full_model_gain_factors": material_supported,
        "classification": classification,
        "next": next_step,
        "simulation_identity_feature_used": False,
        "validation_used_for_fit_or_early_stopping": False,
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
        raise FileExistsError("V34 refuses to overwrite its audit")
    report = evaluate(args.program.resolve(), args.repo.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
