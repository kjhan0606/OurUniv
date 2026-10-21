#!/usr/bin/env python
"""Frozen V41 supervised structure-seeding and object-amplitude calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import h5py
import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment

from hong2021_augmentation import CUBE_ISOMETRIES, apply_cube_isometry
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_KEYS, DOMAIN_ORDER
from hong2021_v31_copula import conditional_forward, conditional_inverse, load_model
from hong2021_v34_nonlinear_sufficiency import multiscale_features, pooled_fields
from hong2021_v35_spectrum_phase import _backbone, _open_split, load_program as load_v35_program
from hong2021_v37_query_alignment import _selection_arrays
from hong2021_v40_object_structure_sufficiency import (
    BACKBONE_COLUMNS,
    FULL_COLUMNS,
    collect_train,
    exact_train_thresholds,
    object_features,
)


PROGRAM_SCHEMA = "hong2021-v41-two-stage-structure-amplitude-development-program-v1"
PROGRAM_SHA256 = "6b189e3aeb8ad7d8a69d84fd029c5656a9b62ee154cd317008d2d82147cfaccd"
FIT_SCHEMA = "hong2021-v41-train-only-two-stage-model-v1"
FIT_REPORT_SCHEMA = "hong2021-v41-train-only-two-stage-fit-report-v1"
PREFLIGHT_SCHEMA = "hong2021-v41-two-stage-hard-preflight-v1"
ENSEMBLE_SCHEMA = "hong2021-v41-two-stage-structure-amplitude-ensemble-v1"
ARMS = (
    "two_stage",
    "backbone_risk_ablation",
    "rolled_risk_control",
    "shuffled_amplitude_control",
)
FACTOR = 4
GRID = 16
BLOCK = 4
BLOCKS = GRID**3
RISK_SHIFT = (3, 5, 7)


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"{label} hash differs")
    return json.loads(path.read_text())


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "V41 program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_implementation_fit_sampling_or_development_evaluation"
    ):
        raise ValueError("V41 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json((repo / parent["v40_record"]).resolve(), parent["v40_record_sha256"], "V41 V40 record")
    audit = _verified_json(Path(parent["v40_audit"]), parent["v40_audit_sha256"], "V41 V40 audit")
    if (
        audit.get("classification") != parent["required_classification"]
        or audit.get("next") != parent["required_next"]
        or audit.get("structure_location_supported") is not True
        or audit.get("object_amplitude_supported") is not True
        or audit.get("Astrid_accessed") is not False
        or audit.get("historical_EAGLE_accessed") is not False
        or record.get("audit", {}).get("sha256") != parent["v40_audit_sha256"]
    ):
        raise ValueError("V41 V40 conclusion or firewall differs")
    v40_path = (repo / record["program"]).resolve()
    if sha256_file(v40_path) != record["program_sha256"]:
        raise ValueError("V41 V40 program hash differs")
    v40 = json.loads(v40_path.read_text())
    inherited = program["inherited_inputs"]
    v35_path = (repo / inherited["v35_program"]).resolve()
    if sha256_file(v35_path) != inherited["v35_program_sha256"]:
        raise ValueError("V41 V35 program hash differs")
    v35, _ = load_v35_program(v35_path, repo)
    if sha256_file((repo / inherited["v31_record"]).resolve()) != inherited["v31_record_sha256"]:
        raise ValueError("V41 V31 record hash differs")
    if sha256_file(Path(inherited["conditional_copula_artifact"])) != inherited["conditional_copula_artifact_sha256"]:
        raise ValueError("V41 V31 copula hash differs")
    return program, v35, v40


def _classifier(spec: Mapping[str, Any]):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        loss=spec["loss"], learning_rate=float(spec["learning_rate"]),
        max_iter=int(spec["max_iter"]), max_leaf_nodes=int(spec["max_leaf_nodes"]),
        min_samples_leaf=int(spec["min_samples_leaf"]), l2_regularization=float(spec["l2_regularization"]),
        early_stopping=bool(spec["early_stopping"]), random_state=int(spec["random_state"]),
    )


def _regressor(spec: Mapping[str, Any]):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        loss=spec["loss"], learning_rate=float(spec["learning_rate"]),
        max_iter=int(spec["max_iter"]), max_leaf_nodes=int(spec["max_leaf_nodes"]),
        min_samples_leaf=int(spec["min_samples_leaf"]), l2_regularization=float(spec["l2_regularization"]),
        early_stopping=bool(spec["early_stopping"]), random_state=int(spec["random_state"]),
    )


def fit_model(program_path: Path, repo: Path, artifact_path: Path, report_path: Path) -> dict[str, Any]:
    _, v35, v40 = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V41 fit requires a clean committed worktree")
    if artifact_path.exists() or report_path.exists():
        raise FileExistsError("V41 refuses existing fit artifacts")
    thresholds, blocks, object_x, object_y, fit_rows = {}, {}, {}, {}, {}
    for domain_index, domain in enumerate(DOMAIN_ORDER):
        row = v35["development_domains"][domain]
        thresholds[domain] = exact_train_thresholds(row, domain)
        block, feature, target = collect_train(row, domain, thresholds[domain], domain_index)
        blocks[domain], object_x[domain], object_y[domain] = block, feature, target["log10_mean_delta_squared"]
        fit_rows[domain] = {
            "train_objects": int(len(feature)),
            "extreme_threshold_log10rho": float(thresholds[domain]["extreme"]),
            "extreme_positive_blocks": int(block["extreme"]["positive_seen"]),
            "extreme_negative_blocks": int(block["extreme"]["negative_seen"]),
        }
    per_class = min(
        min(len(blocks[d]["extreme"]["positive"]), len(blocks[d]["extreme"]["negative"]))
        for d in DOMAIN_ORDER
    )
    x_parts, y_parts = [], []
    for domain in DOMAIN_ORDER:
        x_parts.extend((blocks[domain]["extreme"]["positive"][:per_class], blocks[domain]["extreme"]["negative"][:per_class]))
        y_parts.extend((np.ones(per_class, dtype=np.uint8), np.zeros(per_class, dtype=np.uint8)))
    x_block = np.concatenate(x_parts); y_block = np.concatenate(y_parts)
    classifier_spec = v40["fixed_training"]["block_classifier"]
    classifiers = {}
    for name, columns in (("full", FULL_COLUMNS), ("backbone_only", BACKBONE_COLUMNS)):
        print(f"[v41-fit] classifier {name} rows={len(y_block)} features={len(columns)}", flush=True)
        classifiers[name] = _classifier(classifier_spec).fit(x_block[:, columns], y_block)
    x_object = np.concatenate([object_x[d] for d in DOMAIN_ORDER])
    y_object = np.concatenate([object_y[d] for d in DOMAIN_ORDER])
    weights = np.concatenate([np.full(len(object_x[d]), 1.0 / len(object_x[d])) for d in DOMAIN_ORDER])
    weights *= len(weights) / weights.sum()
    print(f"[v41-fit] object amplitude rows={len(y_object)} features={x_object.shape[1]}", flush=True)
    regressor = _regressor(v40["fixed_training"]["object_regressor"]).fit(x_object, y_object, sample_weight=weights)
    mean_blocks = np.mean([
        blocks[d]["extreme"]["positive_seen"] / len(object_x[d]) for d in DOMAIN_ORDER
    ])
    seed_block_count = int(np.floor(mean_blocks + 0.5))
    artifact = {
        "schema": FIT_SCHEMA,
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "thresholds_log10rho": thresholds,
        "seed_block_count": seed_block_count,
        "full_columns": FULL_COLUMNS,
        "backbone_columns": BACKBONE_COLUMNS,
        "classifiers": classifiers,
        "object_amplitude_regressor": regressor,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    partial_artifact = artifact_path.with_suffix(artifact_path.suffix + ".partial")
    joblib.dump(artifact, partial_artifact, compress=3)
    os.replace(partial_artifact, artifact_path)
    artifact_sha = sha256_file(artifact_path)
    report: dict[str, Any] = {
        "schema": FIT_REPORT_SCHEMA,
        "status": "complete_train_only_source_balanced_fit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": artifact_sha,
        "fit_rows": fit_rows,
        "balanced_extreme_rows_per_source_per_class": int(per_class),
        "balanced_extreme_total_rows": int(len(y_block)),
        "equal_source_mean_extreme_positive_blocks_per_cube": float(mean_blocks),
        "seed_block_count": seed_block_count,
        "structure_models": {
            name: {"features": int(len(columns)), "iterations": int(classifiers[name].n_iter_)}
            for name, columns in (("full", FULL_COLUMNS), ("backbone_only", BACKBONE_COLUMNS))
        },
        "object_model": {"features": int(x_object.shape[1]), "iterations": int(regressor.n_iter_)},
        "target_density_role": "train targets only",
        "validation_opened": False,
        "simulation_identity_feature_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial_report = report_path.with_suffix(report_path.suffix + ".partial")
    partial_report.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial_report, report_path)
    print(json.dumps(report, indent=2), flush=True)
    return report


def load_fit(artifact_path: Path, artifact_sha: str, report_path: Path, report_sha: str, commit: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _verified_json(report_path, report_sha, "V41 fit report")
    if (
        report.get("schema") != FIT_REPORT_SCHEMA
        or report.get("status") != "complete_train_only_source_balanced_fit"
        or report.get("program_sha256") != PROGRAM_SHA256
        or report.get("code_commit") != commit
        or report.get("artifact_sha256") != artifact_sha
        or report.get("validation_opened") is not False
    ):
        raise ValueError("V41 fit report binding differs")
    if sha256_file(artifact_path) != artifact_sha:
        raise ValueError("V41 fit artifact hash differs")
    artifact = joblib.load(artifact_path)
    if (
        artifact.get("schema") != FIT_SCHEMA
        or artifact.get("program_sha256") != PROGRAM_SHA256
        or artifact.get("code_commit") != commit
        or int(artifact.get("seed_block_count", 0)) != int(report["seed_block_count"])
    ):
        raise ValueError("V41 fit artifact metadata differs")
    return artifact, report


def target_free_features(data: h5py.File, cache: h5py.File, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = np.asarray(data["input"][index, 0], dtype=np.float32)
    velocity = np.asarray(data["input"][index, 1], dtype=np.float32)
    dispersion = np.asarray(data["input"][index, 2], dtype=np.float32)
    backbone = _backbone(cache, index).astype(np.float32)
    feature, _ = multiscale_features(
        count, velocity, dispersion, backbone, np.zeros_like(backbone), FACTOR,
        voxel_mpc_h=float(data.attrs["voxel_mpc_h"]),
    )
    pooled = pooled_fields(count, velocity, dispersion, backbone, FACTOR)
    return feature.reshape(BLOCKS, -1), object_features(pooled)[None], backbone[None]


def cube_to_blocks(field: np.ndarray) -> np.ndarray:
    value = np.asarray(field)
    if value.shape != (64, 64, 64):
        raise ValueError("V41 block payload requires a 64-cube")
    return value.reshape(GRID, BLOCK, GRID, BLOCK, GRID, BLOCK).transpose(0, 2, 4, 1, 3, 5).reshape(BLOCKS, BLOCK, BLOCK, BLOCK)


def blocks_to_cube(blocks: np.ndarray) -> np.ndarray:
    value = np.asarray(blocks)
    if value.shape != (BLOCKS, BLOCK, BLOCK, BLOCK):
        raise ValueError("V41 block array shape differs")
    return value.reshape(GRID, GRID, GRID, BLOCK, BLOCK, BLOCK).transpose(0, 3, 1, 4, 2, 5).reshape(64, 64, 64)


def _top_indices(value: np.ndarray, count: int) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(flat) != BLOCKS or not np.isfinite(flat).all() or count <= 0 or count > BLOCKS:
        raise ValueError("V41 top-index payload differs")
    return np.lexsort((np.arange(BLOCKS), -flat))[:count].astype(np.int64)


def _periodic_cost(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = np.column_stack(np.unravel_index(np.asarray(first, dtype=np.int64), (GRID,) * 3))
    b = np.column_stack(np.unravel_index(np.asarray(second, dtype=np.int64), (GRID,) * 3))
    difference = np.abs(a[:, None] - b[None])
    difference = np.minimum(difference, GRID - difference)
    return np.square(difference, dtype=np.float64).sum(axis=-1) / (3.0 * (GRID / 2.0) ** 2)


def seed_permutation(risk: np.ndarray, carrier_score: np.ndarray, count: int) -> tuple[np.ndarray, dict[str, Any]]:
    targets = _top_indices(risk, count)
    carriers = _top_indices(carrier_score, count)
    spatial = _periodic_cost(targets, carriers)
    rank = np.arange(count, dtype=np.float64)
    mismatch = np.square(rank[:, None] - rank[None, :]) / max((count - 1) ** 2, 1)
    rows, columns = linear_sum_assignment(spatial + mismatch)
    permutation = np.arange(BLOCKS, dtype=np.int64)
    permutation[targets[rows]] = carriers[columns]
    leftover_query = np.asarray(sorted(set(carriers) - set(targets)), dtype=np.int64)
    leftover_donor = np.asarray(sorted(set(targets) - set(carriers)), dtype=np.int64)
    if len(leftover_query) != len(leftover_donor):
        raise RuntimeError("V41 displaced block cardinality differs")
    if len(leftover_query):
        displaced_rows, displaced_columns = linear_sum_assignment(_periodic_cost(leftover_query, leftover_donor))
        permutation[leftover_query[displaced_rows]] = leftover_donor[displaced_columns]
    if not np.array_equal(np.sort(permutation), np.arange(BLOCKS)):
        raise RuntimeError("V41 seed map is not a complete permutation")
    nonidentity = int(np.count_nonzero(permutation != np.arange(BLOCKS)))
    if nonidentity > 2 * count:
        raise RuntimeError("V41 seed map changed too many blocks")
    return permutation.astype(np.int16), {
        "seed_block_count": int(count),
        "nonidentity_blocks": nonidentity,
        "nonidentity_fraction": float(nonidentity / BLOCKS),
        "mean_seed_spatial_cost": float(spatial[rows, columns].mean()),
        "mean_seed_rank_mismatch_cost": float(mismatch[rows, columns].mean()),
    }


def transport_blocks(field: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    source = np.asarray(field)
    mapping = np.asarray(permutation, dtype=np.int64)
    if mapping.shape != (BLOCKS,) or not np.array_equal(np.sort(mapping), np.arange(BLOCKS)):
        raise ValueError("V41 transport requires a block permutation")
    output = blocks_to_cube(cube_to_blocks(source)[mapping])
    if not np.array_equal(np.sort(output.reshape(-1)), np.sort(source.reshape(-1))):
        raise RuntimeError("V41 transport changed conditional-rank values")
    return output


def log10_mean_delta_squared(field: np.ndarray) -> float:
    log10rho = 4.5 * np.asarray(field, dtype=np.float64)
    density = np.power(10.0, log10rho)
    value = float(np.square(density - 1.0).mean(dtype=np.float64))
    if not np.isfinite(value):
        raise FloatingPointError("V41 density moment is nonfinite")
    return float(np.log10(max(value, np.finfo(np.float64).tiny)))


def amplitude_scale(backbone: np.ndarray, residual: np.ndarray, target: float) -> tuple[float, float, list[float]]:
    mean = np.asarray(backbone, dtype=np.float64)
    innovation = np.asarray(residual, dtype=np.float64)
    levels = [log10_mean_delta_squared(mean + scale * innovation) for scale in (0.0, 1.0, 2.0)]
    brackets = [index for index in (0, 1) if min(levels[index], levels[index + 1]) <= target <= max(levels[index], levels[index + 1])]
    if brackets:
        index = brackets[0]
        denominator = levels[index + 1] - levels[index]
        fraction = 0.0 if denominator == 0 else (target - levels[index]) / denominator
        scale = float(index + np.clip(fraction, 0.0, 1.0))
    else:
        scale = float(np.argmin(np.abs(np.asarray(levels) - target)))
    achieved = log10_mean_delta_squared(mean + scale * innovation)
    return scale, achieved, levels


def make_sample(
    oriented_rank: np.ndarray,
    risk: np.ndarray,
    query_backbone: np.ndarray,
    copula: Mapping[str, Any],
    target_amplitude: float,
    seed_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rank = np.asarray(oriented_rank)[0]
    carrier = cube_to_blocks(rank).max(axis=(1, 2, 3))
    permutation, diagnostics = seed_permutation(risk, carrier, seed_count)
    transported = transport_blocks(rank, permutation)[None]
    residual = conditional_inverse(transported, query_backbone, copula).astype(np.float64)
    residual -= residual.mean(axis=(-3, -2, -1), keepdims=True)
    scale, achieved, levels = amplitude_scale(query_backbone, residual, float(target_amplitude))
    residual *= scale
    residual -= residual.mean(axis=(-3, -2, -1), keepdims=True)
    sample = np.asarray(query_backbone, dtype=np.float64) + residual
    diagnostics.update(
        {
            "amplitude_scale": scale,
            "predicted_log10_mean_delta_squared": float(target_amplitude),
            "achieved_log10_mean_delta_squared": achieved,
            "absolute_amplitude_error": float(abs(achieved - target_amplitude)),
            "amplitude_levels_s0_s1_s2": levels,
            "maximum_absolute_residual_dc": float(np.max(np.abs(residual.mean(axis=(-3, -2, -1))))),
        }
    )
    if not np.isfinite(sample).all() or not 0.0 <= scale <= 2.0:
        raise RuntimeError("V41 sample or amplitude scale differs")
    return sample.astype(np.float32), permutation, diagnostics


def _predictions(feature: np.ndarray, object_feature: np.ndarray, artifact: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    full = artifact["classifiers"]["full"].predict_proba(feature[:, FULL_COLUMNS])[:, 1].reshape(GRID, GRID, GRID)
    backbone = artifact["classifiers"]["backbone_only"].predict_proba(feature[:, BACKBONE_COLUMNS])[:, 1].reshape(GRID, GRID, GRID)
    amplitude = float(artifact["object_amplitude_regressor"].predict(object_feature)[0])
    return full, backbone, amplitude


def preflight(
    program_path: Path, repo: Path, artifact_path: Path, artifact_sha: str,
    report_path: Path, report_sha: str, output: Path,
) -> dict[str, Any]:
    program, v35, _ = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V41 preflight requires a clean committed worktree")
    artifact, report = load_fit(artifact_path, artifact_sha, report_path, report_sha, commit)
    seed_count = int(artifact["seed_block_count"])
    synthetic = np.arange(64**3, dtype=np.float32).reshape(64, 64, 64)
    synthetic_risk = np.arange(BLOCKS, dtype=np.float64).reshape(GRID, GRID, GRID)
    synthetic_carrier = cube_to_blocks(synthetic).max(axis=(1, 2, 3))
    mapping, synthetic_diagnostics = seed_permutation(synthetic_risk, synthetic_carrier, seed_count)
    transport_blocks(synthetic, mapping)
    selections = _selection_arrays(v35)
    domain = DOMAIN_ORDER[0]
    query_index = int(selections[domain]["source_index"][0])
    next_query_index = int(selections[domain]["source_index"][1])
    donor_source = DOMAIN_ORDER[int(selections[domain]["donor_source"][0, 0])]
    donor_index = int(selections[domain]["donor_index"][0, 0])
    isometry = int(selections[domain]["donor_isometry"][0, 0])
    query_data, query_cache = _open_split(v35["development_domains"][domain], "validation")
    donor_data, donor_cache = _open_split(v35["development_domains"][donor_source], "train")
    copula = load_model(Path(program["inherited_inputs"]["conditional_copula_artifact"]), program["inherited_inputs"]["conditional_copula_artifact_sha256"])
    arm_report = {}
    try:
        feature, object_feature, query_backbone = target_free_features(query_data, query_cache, query_index)
        next_feature, next_object_feature, _ = target_free_features(query_data, query_cache, next_query_index)
        del next_feature
        full_risk, backbone_risk, amplitude = _predictions(feature, object_feature, artifact)
        next_amplitude = float(artifact["object_amplitude_regressor"].predict(next_object_feature)[0])
        donor_backbone = _backbone(donor_cache, donor_index)[None]
        donor_truth = np.asarray(donor_data["target"][donor_index], dtype=np.float32)
        rank = conditional_forward(donor_truth - donor_backbone, donor_backbone, copula)
        permutation_axes, reflections = CUBE_ISOMETRIES[isometry]
        rank = apply_cube_isometry(rank, permutation_axes, reflections)
        arms = {
            "two_stage": (full_risk, amplitude),
            "backbone_risk_ablation": (backbone_risk, amplitude),
            "rolled_risk_control": (np.roll(full_risk, RISK_SHIFT, axis=(0, 1, 2)), amplitude),
            "shuffled_amplitude_control": (full_risk, next_amplitude),
        }
        for arm, (risk, target) in arms.items():
            sample, permutation, diagnostics = make_sample(rank, risk, query_backbone, copula, target, seed_count)
            if not np.isfinite(sample).all() or diagnostics["maximum_absolute_residual_dc"] > 1e-7:
                raise RuntimeError(f"V41 {arm} real preflight failed")
            arm_report[arm] = {**diagnostics, "permutation_bijective": bool(np.array_equal(np.sort(permutation), np.arange(BLOCKS)))}
    finally:
        query_data.close(); query_cache.close(); donor_data.close(); donor_cache.close()
    result: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "pass",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "artifact": str(artifact_path.resolve()), "artifact_sha256": artifact_sha,
        "fit_report": str(report_path.resolve()), "fit_report_sha256": report_sha,
        "fit_report_decision_digest_sha256": report["decision_digest_sha256"],
        "seed_block_count": seed_count,
        "synthetic": synthetic_diagnostics,
        "real_arms": arm_report,
        "validation_truth_used_for_risk_or_amplitude": False,
        "conditional_rank_multiset_preserved_before_inverse": True,
        "donor_translation": False, "donor_reselection": False,
        "density_field_clipping": False, "posthoc_Ak_used": False,
        "Astrid_accessed": False, "historical_EAGLE_accessed": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    if output.exists():
        raise FileExistsError("V41 refuses existing preflight")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, output)
    print(json.dumps(result, indent=2), flush=True)
    return result


def _new_ensemble(handle: h5py.File) -> dict[str, h5py.Dataset]:
    return {
        "sample": handle.create_dataset("sample", shape=(16, 16, 1, 64, 64, 64), dtype="f4", chunks=(1, 1, 1, 64, 64, 64), compression="lzf"),
        "conditional_mean": handle.create_dataset("conditional_mean", shape=(16, 1, 64, 64, 64), dtype="f4", compression="lzf"),
        "truth": handle.create_dataset("truth", shape=(16, 1, 64, 64, 64), dtype="f4", compression="lzf"),
        "block_permutation": handle.create_dataset("block_permutation", shape=(16, 16, BLOCKS), dtype="i2", compression="lzf"),
        "nonidentity_blocks": handle.create_dataset("nonidentity_blocks", shape=(16, 16), dtype="i2"),
        "mean_seed_spatial_cost": handle.create_dataset("mean_seed_spatial_cost", shape=(16, 16), dtype="f4"),
        "mean_seed_rank_mismatch_cost": handle.create_dataset("mean_seed_rank_mismatch_cost", shape=(16, 16), dtype="f4"),
        "amplitude_scale": handle.create_dataset("amplitude_scale", shape=(16, 16), dtype="f4"),
        "predicted_log10_mean_delta_squared": handle.create_dataset("predicted_log10_mean_delta_squared", shape=(16, 16), dtype="f4"),
        "achieved_log10_mean_delta_squared": handle.create_dataset("achieved_log10_mean_delta_squared", shape=(16, 16), dtype="f4"),
        "absolute_amplitude_error": handle.create_dataset("absolute_amplitude_error", shape=(16, 16), dtype="f4"),
        "risk_reference_query_index": handle.create_dataset("risk_reference_query_index", shape=(16,), dtype="i4"),
        "amplitude_reference_query_index": handle.create_dataset("amplitude_reference_query_index", shape=(16,), dtype="i4"),
        "conditional_rank_multiset_sha256": handle.create_dataset("conditional_rank_multiset_sha256", shape=(16, 16, 32), dtype="u1"),
    }


def sample_all(
    program_path: Path, repo: Path, artifact_path: Path, artifact_sha: str,
    report_path: Path, report_sha: str, preflight_path: Path, preflight_sha: str,
    output_root: Path,
) -> None:
    program, v35, _ = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V41 sampling requires a clean committed worktree")
    artifact, _ = load_fit(artifact_path, artifact_sha, report_path, report_sha, commit)
    checked = _verified_json(preflight_path, preflight_sha, "V41 preflight")
    if checked.get("schema") != PREFLIGHT_SCHEMA or checked.get("status") != "pass" or checked.get("code_commit") != commit:
        raise ValueError("V41 preflight binding differs")
    if output_root.exists():
        raise FileExistsError("V41 refuses a pre-existing output root")
    seed_count = int(artifact["seed_block_count"])
    copula = load_model(Path(program["inherited_inputs"]["conditional_copula_artifact"]), program["inherited_inputs"]["conditional_copula_artifact_sha256"])
    selections = _selection_arrays(v35)
    train = {domain: _open_split(v35["development_domains"][domain], "train") for domain in DOMAIN_ORDER}
    try:
        for domain in DOMAIN_ORDER:
            row = v35["development_domains"][domain]
            indices = np.asarray(selections[domain]["source_index"], dtype=np.int64)
            query_data, query_cache = _open_split(row, "validation")
            handles, datasets, partials = {}, {}, {}
            maximum_dc = {arm: 0.0 for arm in ARMS}
            maximum_error = {arm: 0.0 for arm in ARMS}
            boundary = {arm: 0 for arm in ARMS}
            try:
                full_risks, backbone_risks, amplitudes, backbones = [], [], [], []
                for query_index in indices:
                    feature, object_feature, backbone = target_free_features(query_data, query_cache, int(query_index))
                    full, backbone_risk, amplitude = _predictions(feature, object_feature, artifact)
                    full_risks.append(full); backbone_risks.append(backbone_risk); amplitudes.append(amplitude); backbones.append(backbone)
                for arm in ARMS:
                    path = output_root / arm / "development_candidate" / DOMAIN_KEYS[domain] / "ensemble16.h5"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partials[arm] = path.with_suffix(path.suffix + ".partial")
                    handles[arm] = h5py.File(partials[arm], "w")
                    datasets[arm] = _new_ensemble(handles[arm])
                    for name, value in selections[domain].items():
                        handles[arm].create_dataset(name, data=value)
                for object_index, query_index in enumerate(indices):
                    risk_by_arm = {
                        "two_stage": full_risks[object_index],
                        "backbone_risk_ablation": backbone_risks[object_index],
                        "rolled_risk_control": np.roll(full_risks[object_index], RISK_SHIFT, axis=(0, 1, 2)),
                        "shuffled_amplitude_control": full_risks[object_index],
                    }
                    amplitude_by_arm = {arm: amplitudes[object_index] for arm in ARMS}
                    amplitude_by_arm["shuffled_amplitude_control"] = amplitudes[(object_index + 1) % len(indices)]
                    for arm in ARMS:
                        datasets[arm]["risk_reference_query_index"][object_index] = int(query_index)
                        amplitude_reference = indices[(object_index + 1) % len(indices)] if arm == "shuffled_amplitude_control" else query_index
                        datasets[arm]["amplitude_reference_query_index"][object_index] = int(amplitude_reference)
                    for member in range(16):
                        donor_source = DOMAIN_ORDER[int(selections[domain]["donor_source"][object_index, member])]
                        donor_index = int(selections[domain]["donor_index"][object_index, member])
                        isometry = int(selections[domain]["donor_isometry"][object_index, member])
                        donor_data, donor_cache = train[donor_source]
                        donor_backbone = _backbone(donor_cache, donor_index)[None]
                        donor_truth = np.asarray(donor_data["target"][donor_index], dtype=np.float32)
                        rank = conditional_forward(donor_truth - donor_backbone, donor_backbone, copula)
                        axes, reflections = CUBE_ISOMETRIES[isometry]
                        rank = apply_cube_isometry(rank, axes, reflections)
                        rank_digest = np.frombuffer(hashlib.sha256(np.sort(rank.reshape(-1)).tobytes()).digest(), dtype=np.uint8)
                        for arm in ARMS:
                            sample, permutation, diagnostics = make_sample(
                                rank, risk_by_arm[arm], backbones[object_index], copula,
                                amplitude_by_arm[arm], seed_count,
                            )
                            datasets[arm]["sample"][object_index, member] = sample
                            datasets[arm]["block_permutation"][object_index, member] = permutation
                            datasets[arm]["conditional_rank_multiset_sha256"][object_index, member] = rank_digest
                            for name in (
                                "nonidentity_blocks", "mean_seed_spatial_cost", "mean_seed_rank_mismatch_cost",
                                "amplitude_scale", "predicted_log10_mean_delta_squared",
                                "achieved_log10_mean_delta_squared", "absolute_amplitude_error",
                            ):
                                datasets[arm][name][object_index, member] = diagnostics[name]
                            maximum_dc[arm] = max(maximum_dc[arm], diagnostics["maximum_absolute_residual_dc"])
                            maximum_error[arm] = max(maximum_error[arm], diagnostics["absolute_amplitude_error"])
                            boundary[arm] += int(diagnostics["amplitude_scale"] in (0.0, 2.0))
                    for arm in ARMS:
                        datasets[arm]["conditional_mean"][object_index] = backbones[object_index]
                        datasets[arm]["truth"][object_index] = np.asarray(query_data["target"][int(query_index)], dtype=np.float32)
                    print(f"[v41-sample] {domain} {object_index + 1}/16", flush=True)
                for arm in ARMS:
                    handles[arm].attrs.update(
                        {
                            "schema": ENSEMBLE_SCHEMA, "method": "train_only_two_stage_structure_amplitude",
                            "arm": arm, "v41_program_sha256": PROGRAM_SHA256,
                            "fit_artifact": str(artifact_path.resolve()), "fit_artifact_sha256": artifact_sha,
                            "fit_report": str(report_path.resolve()), "fit_report_sha256": report_sha,
                            "preflight": str(preflight_path.resolve()), "preflight_sha256": preflight_sha,
                            "parent_selection": str(Path(row["phase_object_selection"]).resolve()),
                            "parent_selection_sha256": row["phase_object_selection_sha256"],
                            "conditional_copula_model": program["inherited_inputs"]["conditional_copula_artifact"],
                            "conditional_copula_model_sha256": program["inherited_inputs"]["conditional_copula_artifact_sha256"],
                            "block_factor": FACTOR, "block_grid": GRID, "seed_block_count": seed_count,
                            "diagnostic_k_h_mpc": 1.0, "maximum_absolute_residual_dc": maximum_dc[arm],
                            "maximum_absolute_amplitude_calibration_error": maximum_error[arm],
                            "amplitude_scale_boundary_fraction": boundary[arm] / 256.0,
                            "ensemble_members": 16, "conditional_rank_multiset_preserved_before_inverse": True,
                            "validation_truth_used_for_risk_or_amplitude": False,
                            "donor_translation": False, "donor_reselection": False,
                            "density_field_clipping": False, "posthoc_Ak_used": False,
                            "worktree_clean_at_sampling": clean, "sampling_code_commit": commit,
                            "Astrid_accessed": False, "historical_EAGLE_accessed": False, "complete": True,
                        }
                    )
            finally:
                for handle in handles.values(): handle.close()
                query_data.close(); query_cache.close()
            for arm in ARMS:
                os.replace(partials[arm], partials[arm].with_suffix(""))
    finally:
        for data, cache in train.values(): data.close(); cache.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--program", type=Path, required=True); fit.add_argument("--repo", type=Path, required=True)
    fit.add_argument("--artifact", type=Path, required=True); fit.add_argument("--report", type=Path, required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--program", type=Path, required=True); check.add_argument("--repo", type=Path, required=True)
    check.add_argument("--artifact", type=Path, required=True); check.add_argument("--artifact-sha256", required=True)
    check.add_argument("--report", type=Path, required=True); check.add_argument("--report-sha256", required=True)
    check.add_argument("--out", type=Path, required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--program", type=Path, required=True); sample.add_argument("--repo", type=Path, required=True)
    sample.add_argument("--artifact", type=Path, required=True); sample.add_argument("--artifact-sha256", required=True)
    sample.add_argument("--report", type=Path, required=True); sample.add_argument("--report-sha256", required=True)
    sample.add_argument("--preflight", type=Path, required=True); sample.add_argument("--preflight-sha256", required=True)
    sample.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        fit_model(args.program, args.repo, args.artifact, args.report)
    elif args.command == "preflight":
        preflight(args.program, args.repo, args.artifact, args.artifact_sha256, args.report, args.report_sha256, args.out)
    else:
        sample_all(
            args.program, args.repo, args.artifact, args.artifact_sha256,
            args.report, args.report_sha256, args.preflight, args.preflight_sha256, args.out,
        )


if __name__ == "__main__":
    main()
