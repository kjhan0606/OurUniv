#!/usr/bin/env python
"""Frozen V42 native within-block extreme placement and tail-only calibration."""
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
from hong2021_v34_nonlinear_sufficiency import pooled_fields
from hong2021_v35_spectrum_phase import _backbone, _open_split, load_program as load_v35_program
from hong2021_v36_local_tail import PATCH_OFFSETS, v31_quantile_prediction
from hong2021_v37_query_alignment import _selection_arrays
from hong2021_v41_two_stage import (
    BLOCK,
    BLOCKS,
    GRID,
    FIT_REPORT_SCHEMA as V41_FIT_REPORT_SCHEMA,
    FIT_SCHEMA as V41_FIT_SCHEMA,
    _classifier,
    _predictions,
    _top_indices,
    blocks_to_cube,
    cube_to_blocks,
    log10_mean_delta_squared,
    seed_permutation,
    target_free_features,
    transport_blocks,
)


PROGRAM_SCHEMA = "hong2021-v42-within-block-tail-body-development-program-v1"
PROGRAM_SHA256 = "186fa2738995e30e068fcd903ff0e69c121996ab3ca8655a05b8f2e1fb8e5b1e"
FIT_SCHEMA = "hong2021-v42-train-only-native-extreme-model-v1"
FIT_REPORT_SCHEMA = "hong2021-v42-train-only-native-extreme-fit-report-v1"
PREFLIGHT_SCHEMA = "hong2021-v42-within-block-tail-body-hard-preflight-v1"
ENSEMBLE_SCHEMA = "hong2021-v42-within-block-tail-body-ensemble-v1"
ARMS = (
    "within_block_tail_body",
    "block_only_tail_control",
    "rolled_native_risk_control",
    "tail_calibration_disabled_control",
)
NATIVE_FEATURES = 221
MAX_NATIVE_ROWS = 32768
TAIL_QUANTILE = 0.999
LAMBDA_KNOTS = (1.0 / 64.0, 0.25, 1.0)
NATIVE_RISK_SHIFT = (1, 2, 3)


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"{label} hash differs")
    return json.loads(path.read_text())


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "V42 program")
    if program.get("schema") != PROGRAM_SCHEMA or program.get("status") != "frozen_before_implementation_fit_sampling_or_development_evaluation":
        raise ValueError("V42 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json((repo / parent["v41_record"]).resolve(), parent["v41_record_sha256"], "V42 V41 record")
    decision = record.get("decision", {})
    if (
        decision.get("classification") != parent["required_classification"]
        or decision.get("next") != parent["required_next"]
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V42 V41 conclusion or firewall differs")
    inherited = program["inherited_inputs"]
    v35_path = (repo / inherited["v35_program"]).resolve()
    if sha256_file(v35_path) != inherited["v35_program_sha256"]:
        raise ValueError("V42 V35 program hash differs")
    v35, _ = load_v35_program(v35_path, repo)
    if sha256_file((repo / inherited["v31_record"]).resolve()) != inherited["v31_record_sha256"]:
        raise ValueError("V42 V31 record hash differs")
    if sha256_file(Path(inherited["conditional_copula_artifact"])) != inherited["conditional_copula_artifact_sha256"]:
        raise ValueError("V42 V31 copula hash differs")
    v41_report = _verified_json(Path(inherited["v41_report"]), inherited["v41_report_sha256"], "V42 V41 report")
    if (
        v41_report.get("schema") != V41_FIT_REPORT_SCHEMA
        or v41_report.get("artifact_sha256") != inherited["v41_artifact_sha256"]
        or v41_report.get("decision_digest_sha256") != inherited["v41_report_decision_digest_sha256"]
        or v41_report.get("validation_opened") is not False
        or int(v41_report.get("seed_block_count", 0)) != inherited["v41_seed_block_count"]
    ):
        raise ValueError("V42 V41 report binding differs")
    if sha256_file(Path(inherited["v41_artifact"])) != inherited["v41_artifact_sha256"]:
        raise ValueError("V42 V41 artifact hash differs")
    v41 = joblib.load(inherited["v41_artifact"])
    if v41.get("schema") != V41_FIT_SCHEMA or int(v41.get("seed_block_count", 0)) != inherited["v41_seed_block_count"]:
        raise ValueError("V42 V41 artifact metadata differs")
    v41_program_path = (repo / record["program"]).resolve()
    if sha256_file(v41_program_path) != record["program_sha256"]:
        raise ValueError("V42 V41 program hash differs")
    v41_program = json.loads(v41_program_path.read_text())
    v40_record = _verified_json((repo / v41_program["parent_evidence"]["v40_record"]).resolve(), v41_program["parent_evidence"]["v40_record_sha256"], "V42 V40 record")
    v40_path = (repo / v40_record["program"]).resolve()
    if sha256_file(v40_path) != v40_record["program_sha256"]:
        raise ValueError("V42 V40 program hash differs")
    return program, v35, v41, json.loads(v40_path.read_text())


class IndexPrioritySampler:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity); self.rng = np.random.default_rng(seed)
        self.priority = np.empty(0, dtype=np.float64); self.index = np.empty(0, dtype=np.int64); self.seen = 0

    def add(self, indices: np.ndarray) -> None:
        value = np.asarray(indices, dtype=np.int64).reshape(-1)
        if not len(value): return
        self.seen += len(value); priority = self.rng.random(len(value))
        if len(value) > self.capacity:
            keep = np.argpartition(priority, self.capacity - 1)[: self.capacity]
            value, priority = value[keep], priority[keep]
        p = np.concatenate((self.priority, priority)); i = np.concatenate((self.index, value))
        if len(p) > self.capacity:
            keep = np.argpartition(p, self.capacity - 1)[: self.capacity]
            p, i = p[keep], i[keep]
        self.priority, self.index = p, i

    def result(self) -> np.ndarray:
        return self.index[np.argsort(self.priority, kind="stable")]


def _sample(field: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    return np.asarray(field)[coordinate[:, 0], coordinate[:, 1], coordinate[:, 2]]


def native_features(data: h5py.File, cache: h5py.File, index: int, coordinate: np.ndarray) -> np.ndarray:
    coordinate = np.asarray(coordinate, dtype=np.int64)
    if coordinate.ndim != 2 or coordinate.shape[1] != 3:
        raise ValueError("V42 native coordinates differ")
    count = np.asarray(data["input"][index, 0], dtype=np.float32)
    velocity = np.asarray(data["input"][index, 1], dtype=np.float32)
    dispersion = np.asarray(data["input"][index, 2], dtype=np.float32)
    backbone = _backbone(cache, index).astype(np.float32)
    native = {"logcount": np.log1p(count), "velocity": velocity, "dispersion": dispersion, "backbone": backbone}
    parent_raw = pooled_fields(count, velocity, dispersion, backbone, 4)
    parent = {
        "logcount": parent_raw["log1p_block_count"], "velocity": parent_raw["block_mean_velocity_kms"],
        "dispersion": parent_raw["exact_population_velocity_dispersion_kms"], "backbone": parent_raw["backbone_mean_y"],
    }
    radius = np.sqrt(np.square((coordinate.astype(np.float64) + 0.5) * float(data.attrs["voxel_mpc_h"]) - 10.0).sum(axis=1)) / 10.0
    pieces = [_sample(backbone, coordinate), _sample(native["logcount"], coordinate), _sample(velocity, coordinate), _sample(dispersion, coordinate), radius]
    for field in native.values():
        for offset in PATCH_OFFSETS:
            pieces.append(_sample(field, (coordinate + np.asarray(offset)) % 64))
    parent_coordinate = coordinate // 4
    for field in parent.values():
        for offset in PATCH_OFFSETS:
            pieces.append(_sample(field, (parent_coordinate + np.asarray(offset)) % 16))
    result = np.column_stack(pieces).astype(np.float32)
    if result.shape != (len(coordinate), NATIVE_FEATURES) or not np.isfinite(result).all():
        raise RuntimeError("V42 native feature shape or values differ")
    return result


def _coordinates(selected: np.ndarray, cube_index: int) -> np.ndarray:
    selected = np.asarray(selected, dtype=np.int64)
    lower = cube_index * 64**3
    left = np.searchsorted(selected, lower); right = np.searchsorted(selected, lower + 64**3)
    local = selected[left:right] - lower
    return np.column_stack(np.unravel_index(local, (64, 64, 64))).astype(np.int64) if len(local) else np.empty((0, 3), dtype=np.int64)


def collect_native_train(row: dict[str, Any], domain: str, threshold: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    objects = int(row["train_objects"])
    positive_sampler = IndexPrioritySampler(MAX_NATIVE_ROWS, seed)
    negative_sampler = IndexPrioritySampler(MAX_NATIVE_ROWS, seed + 1)
    data, cache = _open_split(row, "train"); cache.close()
    try:
        for index in range(objects):
            log10rho = np.asarray(data["target"][index, 0], dtype=np.float32).reshape(-1) * np.float32(4.5)
            mask = log10rho > threshold
            base = index * 64**3
            positive_sampler.add(base + np.flatnonzero(mask)); negative_sampler.add(base + np.flatnonzero(~mask))
            if (index + 1) % 64 == 0 or index + 1 == objects:
                print(f"[v42-index] {domain} {index + 1}/{objects}", flush=True)
    finally:
        data.close()
    positive = np.sort(positive_sampler.result()); negative = np.sort(negative_sampler.result())
    x_positive, x_negative = [], []
    data, cache = _open_split(row, "train")
    try:
        for index in range(objects):
            pos = _coordinates(positive, index); neg = _coordinates(negative, index)
            if len(pos): x_positive.append(native_features(data, cache, index, pos))
            if len(neg): x_negative.append(native_features(data, cache, index, neg))
            if (index + 1) % 64 == 0 or index + 1 == objects:
                print(f"[v42-feature] {domain} {index + 1}/{objects}", flush=True)
    finally:
        data.close(); cache.close()
    return np.concatenate(x_positive), np.concatenate(x_negative), {
        "positive_seen": positive_sampler.seen, "negative_seen": negative_sampler.seen,
        "positive_retained": len(positive), "negative_retained": len(negative),
    }


def fit_model(program_path: Path, repo: Path, artifact_path: Path, report_path: Path) -> dict[str, Any]:
    program, v35, v41, v40 = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean: raise RuntimeError("V42 fit requires a clean committed worktree")
    if artifact_path.exists() or report_path.exists(): raise FileExistsError("V42 refuses existing fit outputs")
    train, rows = {}, {}
    seeds = program["train_only_native_extreme_model"]["seeds"]
    for domain in DOMAIN_ORDER:
        threshold = float(v41["thresholds_log10rho"][domain]["extreme"])
        positive, negative, summary = collect_native_train(v35["development_domains"][domain], domain, threshold, int(seeds[domain]))
        train[domain] = (positive, negative); rows[domain] = {"threshold_log10rho": threshold, **summary}
    per_class = min(min(len(train[d][0]), len(train[d][1])) for d in DOMAIN_ORDER)
    x_parts, y_parts = [], []
    for domain in DOMAIN_ORDER:
        x_parts.extend((train[domain][0][:per_class], train[domain][1][:per_class]))
        y_parts.extend((np.ones(per_class, dtype=np.uint8), np.zeros(per_class, dtype=np.uint8)))
    x = np.concatenate(x_parts); y = np.concatenate(y_parts)
    classifier = _classifier(v40["fixed_training"]["block_classifier"])
    print(f"[v42-fit] rows={len(y)} features={x.shape[1]}", flush=True); classifier.fit(x, y)
    mean_native = np.mean([rows[d]["positive_seen"] / int(v35["development_domains"][d]["train_objects"]) for d in DOMAIN_ORDER])
    native_per_block = int(np.floor(mean_native / int(v41["seed_block_count"]) + 0.5))
    artifact = {
        "schema": FIT_SCHEMA, "program_sha256": PROGRAM_SHA256, "code_commit": commit,
        "native_classifier": classifier, "native_features": NATIVE_FEATURES,
        "native_seed_voxels_per_seed_block": native_per_block,
        "v41_artifact_sha256": program["inherited_inputs"]["v41_artifact_sha256"],
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    partial_artifact = artifact_path.with_suffix(artifact_path.suffix + ".partial")
    joblib.dump(artifact, partial_artifact, compress=3); os.replace(partial_artifact, artifact_path)
    report: dict[str, Any] = {
        "schema": FIT_REPORT_SCHEMA, "status": "complete_train_only_source_class_balanced_fit",
        "program_sha256": PROGRAM_SHA256, "code_commit": commit, "worktree_clean": clean,
        "artifact": str(artifact_path.resolve()), "artifact_sha256": sha256_file(artifact_path),
        "rows": rows, "balanced_rows_per_source_per_class": int(per_class), "balanced_total_rows": int(len(y)),
        "equal_source_mean_native_extreme_voxels_per_cube": float(mean_native),
        "v41_seed_block_count": int(v41["seed_block_count"]),
        "native_seed_voxels_per_seed_block": native_per_block,
        "model_features": NATIVE_FEATURES, "model_iterations": int(classifier.n_iter_),
        "validation_opened": False, "simulation_identity_feature_used": False,
        "Astrid_accessed": False, "historical_EAGLE_accessed": False,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    partial_report = report_path.with_suffix(report_path.suffix + ".partial")
    partial_report.write_text(json.dumps(report, indent=2) + "\n"); os.replace(partial_report, report_path)
    print(json.dumps(report, indent=2), flush=True); return report


def load_fit(artifact_path: Path, artifact_sha: str, report_path: Path, report_sha: str, commit: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _verified_json(report_path, report_sha, "V42 fit report")
    if report.get("schema") != FIT_REPORT_SCHEMA or report.get("code_commit") != commit or report.get("artifact_sha256") != artifact_sha or report.get("validation_opened") is not False:
        raise ValueError("V42 fit report binding differs")
    if sha256_file(artifact_path) != artifact_sha: raise ValueError("V42 fit artifact hash differs")
    artifact = joblib.load(artifact_path)
    if artifact.get("schema") != FIT_SCHEMA or artifact.get("program_sha256") != PROGRAM_SHA256 or artifact.get("code_commit") != commit:
        raise ValueError("V42 fit artifact metadata differs")
    return artifact, report


def _local_periodic_cost(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    a = np.column_stack(np.unravel_index(first, (BLOCK,) * 3)); b = np.column_stack(np.unravel_index(second, (BLOCK,) * 3))
    difference = np.abs(a[:, None] - b[None]); difference = np.minimum(difference, BLOCK - difference)
    return np.square(difference, dtype=np.float64).sum(axis=-1)


def local_permutation(risk: np.ndarray, carrier: np.ndarray, count: int) -> np.ndarray:
    flat_risk = np.asarray(risk, dtype=np.float64).reshape(-1); flat_carrier = np.asarray(carrier, dtype=np.float64).reshape(-1)
    target = np.lexsort((np.arange(BLOCK**3), -flat_risk))[:count]
    source = np.lexsort((np.arange(BLOCK**3), -flat_carrier))[:count]
    mapping = np.arange(BLOCK**3, dtype=np.int64); mapping[target] = source
    leftover_query = np.asarray(sorted(set(source) - set(target)), dtype=np.int64)
    leftover_source = np.asarray(sorted(set(target) - set(source)), dtype=np.int64)
    if len(leftover_query):
        rows, columns = linear_sum_assignment(_local_periodic_cost(leftover_query, leftover_source))
        mapping[leftover_query[rows]] = leftover_source[columns]
    if not np.array_equal(np.sort(mapping), np.arange(BLOCK**3)):
        raise RuntimeError("V42 local map is not bijective")
    return mapping.astype(np.int16)


def nested_transport(
    rank: np.ndarray, block_risk: np.ndarray, native_risk: np.ndarray,
    seed_blocks: int, native_count: int, *, native_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    carrier = cube_to_blocks(rank).max(axis=(1, 2, 3))
    block_map, block_diagnostics = seed_permutation(block_risk, carrier, seed_blocks)
    transported_blocks = cube_to_blocks(rank)[np.asarray(block_map, dtype=np.int64)]
    target_blocks = _top_indices(block_risk, seed_blocks)
    local_maps = np.broadcast_to(np.arange(BLOCK**3, dtype=np.int16), (seed_blocks, BLOCK**3)).copy()
    for position, block_index in enumerate(target_blocks):
        if native_mode == "identity": continue
        risk = np.asarray(native_risk[block_index], dtype=np.float64)
        if native_mode == "rolled": risk = np.roll(risk, NATIVE_RISK_SHIFT, axis=(0, 1, 2))
        mapping = local_permutation(risk, transported_blocks[block_index], native_count)
        transported_blocks[block_index] = transported_blocks[block_index].reshape(-1)[mapping].reshape(BLOCK, BLOCK, BLOCK)
        local_maps[position] = mapping
    output = blocks_to_cube(transported_blocks)
    if not np.array_equal(np.sort(output.reshape(-1)), np.sort(np.asarray(rank).reshape(-1))):
        raise RuntimeError("V42 nested transport changed rank multiset")
    diagnostics = {
        **block_diagnostics,
        "native_seed_voxels_per_block": int(native_count),
        "native_modified_blocks": int(0 if native_mode == "identity" else seed_blocks),
        "native_nonidentity_voxels": int(np.count_nonzero(local_maps != np.arange(BLOCK**3))),
    }
    return output, block_map, local_maps, diagnostics


def tail_calibrate(
    backbone: np.ndarray, residual: np.ndarray, threshold: np.ndarray,
    target: float, *, enabled: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    uncalibrated = np.asarray(residual, dtype=np.float64)
    limit = np.asarray(threshold, dtype=np.float64)
    excess = np.maximum(uncalibrated - limit, 0.0)
    body = uncalibrated - excess

    def evaluate(value: float) -> tuple[np.ndarray, float, float]:
        raw = body + value * excess
        dc = float(raw.mean(dtype=np.float64)); centered = raw - dc
        return centered, log10_mean_delta_squared(np.asarray(backbone, dtype=np.float64) + centered), dc

    levels = [evaluate(value)[1] for value in LAMBDA_KNOTS]
    if enabled:
        brackets = [i for i in (0, 1) if min(levels[i], levels[i + 1]) <= target <= max(levels[i], levels[i + 1])]
        if brackets:
            i = brackets[0]; denominator = levels[i + 1] - levels[i]
            fraction = 0.0 if denominator == 0 else (target - levels[i]) / denominator
            value = float(LAMBDA_KNOTS[i] + np.clip(fraction, 0.0, 1.0) * (LAMBDA_KNOTS[i + 1] - LAMBDA_KNOTS[i]))
        else:
            value = float(LAMBDA_KNOTS[int(np.argmin(np.abs(np.asarray(levels) - target)))])
    else:
        value = 1.0
    calibrated, achieved, dc = evaluate(value)
    non_tail = excess == 0
    body_error = float(np.max(np.abs((calibrated + dc - uncalibrated)[non_tail]))) if np.any(non_tail) else 0.0
    diagnostics = {
        "tail_lambda": value, "tail_fraction": float(np.mean(~non_tail)),
        "predicted_log10_mean_delta_squared": float(target), "achieved_log10_mean_delta_squared": achieved,
        "absolute_amplitude_error": float(abs(achieved - target)), "amplitude_levels_lambda_knots": levels,
        "tail_DC_projection": dc, "maximum_non_tail_error_after_undoing_DC": body_error,
        "maximum_absolute_residual_dc": float(abs(calibrated.mean(dtype=np.float64))),
    }
    return calibrated, diagnostics


def query_native_risk(data: h5py.File, cache: h5py.File, index: int, target_blocks: np.ndarray, classifier: Any) -> np.ndarray:
    risk = np.full((BLOCKS, BLOCK, BLOCK, BLOCK), -np.inf, dtype=np.float32)
    for block_index in np.asarray(target_blocks, dtype=np.int64):
        block_coordinate = np.asarray(np.unravel_index(block_index, (GRID,) * 3), dtype=np.int64)
        local = np.column_stack(np.unravel_index(np.arange(BLOCK**3), (BLOCK,) * 3)).astype(np.int64)
        coordinate = block_coordinate[None] * BLOCK + local
        risk[block_index] = classifier.predict_proba(native_features(data, cache, index, coordinate))[:, 1].reshape(BLOCK, BLOCK, BLOCK)
    return risk


def make_sample(
    rank: np.ndarray, block_risk: np.ndarray, native_risk: np.ndarray,
    backbone: np.ndarray, copula: Mapping[str, Any], amplitude: float,
    seed_blocks: int, native_count: int, *, native_mode: str, tail_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    transported, block_map, local_maps, diagnostics = nested_transport(rank[0], block_risk, native_risk, seed_blocks, native_count, native_mode=native_mode)
    residual = conditional_inverse(transported[None], backbone, copula).astype(np.float64)
    residual -= residual.mean(axis=(-3, -2, -1), keepdims=True)
    threshold = v31_quantile_prediction(backbone, TAIL_QUANTILE, copula)
    calibrated, tail = tail_calibrate(backbone, residual, threshold, amplitude, enabled=tail_enabled)
    diagnostics.update(tail)
    sample = np.asarray(backbone, dtype=np.float64) + calibrated
    if not np.isfinite(sample).all() or diagnostics["maximum_non_tail_error_after_undoing_DC"] > 1e-7:
        raise RuntimeError("V42 sample or body invariant differs")
    return sample.astype(np.float32), block_map, local_maps, diagnostics


def preflight(
    program_path: Path, repo: Path, artifact_path: Path, artifact_sha: str,
    report_path: Path, report_sha: str, output: Path,
) -> dict[str, Any]:
    program, v35, v41, _ = load_program(program_path, repo); commit, clean = git_state(repo.resolve())
    if not clean: raise RuntimeError("V42 preflight requires a clean worktree")
    artifact, report = load_fit(artifact_path, artifact_sha, report_path, report_sha, commit)
    synthetic = np.arange(64**3, dtype=np.float32).reshape(64, 64, 64)
    risk = np.arange(BLOCKS, dtype=np.float64).reshape(GRID, GRID, GRID)
    native = np.broadcast_to(np.arange(BLOCK**3).reshape(BLOCK, BLOCK, BLOCK), (BLOCKS, BLOCK, BLOCK, BLOCK))
    nested_transport(synthetic, risk, native, int(v41["seed_block_count"]), int(artifact["native_seed_voxels_per_seed_block"]), native_mode="full")
    selections = _selection_arrays(v35); domain = DOMAIN_ORDER[0]
    query_index = int(selections[domain]["source_index"][0]); donor_source = DOMAIN_ORDER[int(selections[domain]["donor_source"][0, 0])]
    donor_index = int(selections[domain]["donor_index"][0, 0]); isometry = int(selections[domain]["donor_isometry"][0, 0])
    query_data, query_cache = _open_split(v35["development_domains"][domain], "validation")
    donor_data, donor_cache = _open_split(v35["development_domains"][donor_source], "train")
    copula = load_model(Path(program["inherited_inputs"]["conditional_copula_artifact"]), program["inherited_inputs"]["conditional_copula_artifact_sha256"])
    try:
        feature, object_feature, backbone = target_free_features(query_data, query_cache, query_index)
        block_risk, _, amplitude = _predictions(feature, object_feature, v41)
        target_blocks = _top_indices(block_risk, int(v41["seed_block_count"]))
        native_risk = query_native_risk(query_data, query_cache, query_index, target_blocks, artifact["native_classifier"])
        donor_backbone = _backbone(donor_cache, donor_index)[None]; donor_truth = np.asarray(donor_data["target"][donor_index], dtype=np.float32)
        rank = conditional_forward(donor_truth - donor_backbone, donor_backbone, copula)
        axes, reflections = CUBE_ISOMETRIES[isometry]; rank = apply_cube_isometry(rank, axes, reflections)
        arm_report = {}
        modes = {
            "within_block_tail_body": ("full", True), "block_only_tail_control": ("identity", True),
            "rolled_native_risk_control": ("rolled", True), "tail_calibration_disabled_control": ("full", False),
        }
        for arm, (mode, enabled) in modes.items():
            sample, block_map, local_maps, diagnostics = make_sample(
                rank, block_risk, native_risk, backbone, copula, amplitude,
                int(v41["seed_block_count"]), int(artifact["native_seed_voxels_per_seed_block"]),
                native_mode=mode, tail_enabled=enabled,
            )
            arm_report[arm] = {**diagnostics, "block_permutation_bijective": bool(np.array_equal(np.sort(block_map), np.arange(BLOCKS))), "all_local_permutations_bijective": bool(np.all(np.sort(local_maps, axis=-1) == np.arange(BLOCK**3)))}
    finally:
        query_data.close(); query_cache.close(); donor_data.close(); donor_cache.close()
    result: dict[str, Any] = {
        "schema": PREFLIGHT_SCHEMA, "status": "pass", "program_sha256": PROGRAM_SHA256,
        "code_commit": commit, "worktree_clean": clean,
        "artifact": str(artifact_path.resolve()), "artifact_sha256": artifact_sha,
        "fit_report": str(report_path.resolve()), "fit_report_sha256": report_sha,
        "fit_report_decision_digest_sha256": report["decision_digest_sha256"],
        "real_arms": arm_report, "global_residual_scale": 1.0,
        "validation_truth_used_for_risk_or_amplitude": False,
        "conditional_rank_multiset_preserved_before_inverse": True,
        "hard_density_or_residual_clipping": False, "donor_translation": False, "donor_reselection": False,
        "posthoc_Ak_used": False, "Astrid_accessed": False, "historical_EAGLE_accessed": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    if output.exists(): raise FileExistsError("V42 refuses existing preflight")
    output.parent.mkdir(parents=True, exist_ok=True); partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n"); os.replace(partial, output); print(json.dumps(result, indent=2), flush=True)
    return result


def _new_ensemble(handle: h5py.File, seed_blocks: int) -> dict[str, h5py.Dataset]:
    return {
        "sample": handle.create_dataset("sample", shape=(16, 16, 1, 64, 64, 64), dtype="f4", chunks=(1, 1, 1, 64, 64, 64), compression="lzf"),
        "conditional_mean": handle.create_dataset("conditional_mean", shape=(16, 1, 64, 64, 64), dtype="f4", compression="lzf"),
        "truth": handle.create_dataset("truth", shape=(16, 1, 64, 64, 64), dtype="f4", compression="lzf"),
        "block_permutation": handle.create_dataset("block_permutation", shape=(16, 16, BLOCKS), dtype="i2", compression="lzf"),
        "local_permutation": handle.create_dataset("local_permutation", shape=(16, 16, seed_blocks, BLOCK**3), dtype="i1", compression="lzf"),
        "tail_lambda": handle.create_dataset("tail_lambda", shape=(16, 16), dtype="f4"),
        "tail_fraction": handle.create_dataset("tail_fraction", shape=(16, 16), dtype="f4"),
        "predicted_log10_mean_delta_squared": handle.create_dataset("predicted_log10_mean_delta_squared", shape=(16, 16), dtype="f4"),
        "achieved_log10_mean_delta_squared": handle.create_dataset("achieved_log10_mean_delta_squared", shape=(16, 16), dtype="f4"),
        "absolute_amplitude_error": handle.create_dataset("absolute_amplitude_error", shape=(16, 16), dtype="f4"),
        "maximum_non_tail_error_after_undoing_DC": handle.create_dataset("maximum_non_tail_error_after_undoing_DC", shape=(16, 16), dtype="f4"),
        "conditional_rank_multiset_sha256": handle.create_dataset("conditional_rank_multiset_sha256", shape=(16, 16, 32), dtype="u1"),
    }


def sample_all(
    program_path: Path, repo: Path, artifact_path: Path, artifact_sha: str, report_path: Path,
    report_sha: str, preflight_path: Path, preflight_sha: str, output_root: Path,
) -> None:
    program, v35, v41, _ = load_program(program_path, repo); commit, clean = git_state(repo.resolve())
    if not clean: raise RuntimeError("V42 sampling requires a clean worktree")
    artifact, _ = load_fit(artifact_path, artifact_sha, report_path, report_sha, commit)
    checked = _verified_json(preflight_path, preflight_sha, "V42 preflight")
    if checked.get("schema") != PREFLIGHT_SCHEMA or checked.get("code_commit") != commit: raise ValueError("V42 preflight binding differs")
    if output_root.exists(): raise FileExistsError("V42 refuses existing output root")
    seed_blocks = int(v41["seed_block_count"]); native_count = int(artifact["native_seed_voxels_per_seed_block"])
    copula = load_model(Path(program["inherited_inputs"]["conditional_copula_artifact"]), program["inherited_inputs"]["conditional_copula_artifact_sha256"])
    selections = _selection_arrays(v35); train = {d: _open_split(v35["development_domains"][d], "train") for d in DOMAIN_ORDER}
    modes = {
        "within_block_tail_body": ("full", True), "block_only_tail_control": ("identity", True),
        "rolled_native_risk_control": ("rolled", True), "tail_calibration_disabled_control": ("full", False),
    }
    try:
        for domain in DOMAIN_ORDER:
            row = v35["development_domains"][domain]; indices = np.asarray(selections[domain]["source_index"], dtype=np.int64)
            query_data, query_cache = _open_split(row, "validation"); handles = {}; datasets = {}; partials = {}
            try:
                query_payload = []
                for query_index in indices:
                    feature, object_feature, backbone = target_free_features(query_data, query_cache, int(query_index))
                    block_risk, _, amplitude = _predictions(feature, object_feature, v41)
                    targets = _top_indices(block_risk, seed_blocks)
                    native_risk = query_native_risk(query_data, query_cache, int(query_index), targets, artifact["native_classifier"])
                    query_payload.append((backbone, block_risk, native_risk, amplitude))
                for arm in ARMS:
                    path = output_root / arm / "development_candidate" / DOMAIN_KEYS[domain] / "ensemble16.h5"
                    path.parent.mkdir(parents=True, exist_ok=True); partials[arm] = path.with_suffix(path.suffix + ".partial")
                    handles[arm] = h5py.File(partials[arm], "w"); datasets[arm] = _new_ensemble(handles[arm], seed_blocks)
                    for name, value in selections[domain].items(): handles[arm].create_dataset(name, data=value)
                maxima = {arm: {"dc": 0.0, "body": 0.0, "error": 0.0} for arm in ARMS}; boundaries = {arm: 0 for arm in ARMS}
                for object_index, query_index in enumerate(indices):
                    backbone, block_risk, native_risk, amplitude = query_payload[object_index]
                    for member in range(16):
                        donor_source = DOMAIN_ORDER[int(selections[domain]["donor_source"][object_index, member])]
                        donor_index = int(selections[domain]["donor_index"][object_index, member]); isometry = int(selections[domain]["donor_isometry"][object_index, member])
                        donor_data, donor_cache = train[donor_source]; donor_backbone = _backbone(donor_cache, donor_index)[None]
                        donor_truth = np.asarray(donor_data["target"][donor_index], dtype=np.float32)
                        rank = conditional_forward(donor_truth - donor_backbone, donor_backbone, copula)
                        axes, reflections = CUBE_ISOMETRIES[isometry]; rank = apply_cube_isometry(rank, axes, reflections)
                        digest = np.frombuffer(hashlib.sha256(np.sort(rank.reshape(-1)).tobytes()).digest(), dtype=np.uint8)
                        for arm, (mode, enabled) in modes.items():
                            sample, block_map, local_maps, diagnostic = make_sample(
                                rank, block_risk, native_risk, backbone, copula, amplitude,
                                seed_blocks, native_count, native_mode=mode, tail_enabled=enabled,
                            )
                            datasets[arm]["sample"][object_index, member] = sample; datasets[arm]["block_permutation"][object_index, member] = block_map
                            datasets[arm]["local_permutation"][object_index, member] = local_maps; datasets[arm]["conditional_rank_multiset_sha256"][object_index, member] = digest
                            for name in ("tail_lambda", "tail_fraction", "predicted_log10_mean_delta_squared", "achieved_log10_mean_delta_squared", "absolute_amplitude_error", "maximum_non_tail_error_after_undoing_DC"):
                                datasets[arm][name][object_index, member] = diagnostic[name]
                            maxima[arm]["dc"] = max(maxima[arm]["dc"], diagnostic["maximum_absolute_residual_dc"])
                            maxima[arm]["body"] = max(maxima[arm]["body"], diagnostic["maximum_non_tail_error_after_undoing_DC"])
                            maxima[arm]["error"] = max(maxima[arm]["error"], diagnostic["absolute_amplitude_error"])
                            boundaries[arm] += int(diagnostic["tail_lambda"] in (LAMBDA_KNOTS[0], LAMBDA_KNOTS[-1]))
                    for arm in ARMS:
                        datasets[arm]["conditional_mean"][object_index] = backbone
                        datasets[arm]["truth"][object_index] = np.asarray(query_data["target"][int(query_index)], dtype=np.float32)
                    print(f"[v42-sample] {domain} {object_index + 1}/16", flush=True)
                for arm in ARMS:
                    handles[arm].attrs.update({
                        "schema": ENSEMBLE_SCHEMA, "method": "train_only_within_block_tail_body", "arm": arm,
                        "v42_program_sha256": PROGRAM_SHA256, "fit_artifact": str(artifact_path.resolve()), "fit_artifact_sha256": artifact_sha,
                        "fit_report": str(report_path.resolve()), "fit_report_sha256": report_sha,
                        "preflight": str(preflight_path.resolve()), "preflight_sha256": preflight_sha,
                        "parent_selection": str(Path(row["phase_object_selection"]).resolve()), "parent_selection_sha256": row["phase_object_selection_sha256"],
                        "v41_artifact": program["inherited_inputs"]["v41_artifact"], "v41_artifact_sha256": program["inherited_inputs"]["v41_artifact_sha256"],
                        "conditional_copula_model": program["inherited_inputs"]["conditional_copula_artifact"], "conditional_copula_model_sha256": program["inherited_inputs"]["conditional_copula_artifact_sha256"],
                        "block_factor": 4, "block_grid": 16, "seed_block_count": seed_blocks, "native_seed_voxels_per_seed_block": native_count,
                        "tail_quantile": TAIL_QUANTILE, "tail_lambda_minimum": LAMBDA_KNOTS[0], "global_residual_scale": 1.0,
                        "maximum_absolute_residual_dc": maxima[arm]["dc"], "maximum_non_tail_error_after_undoing_DC": maxima[arm]["body"],
                        "maximum_absolute_amplitude_error": maxima[arm]["error"], "tail_lambda_boundary_fraction": boundaries[arm] / 256.0,
                        "diagnostic_k_h_mpc": 1.0, "ensemble_members": 16, "conditional_rank_multiset_preserved_before_inverse": True,
                        "validation_truth_used_for_risk_or_amplitude": False, "hard_density_or_residual_clipping": False,
                        "donor_translation": False, "donor_reselection": False, "posthoc_Ak_used": False,
                        "worktree_clean_at_sampling": clean, "sampling_code_commit": commit,
                        "Astrid_accessed": False, "historical_EAGLE_accessed": False, "complete": True,
                    })
            finally:
                for handle in handles.values(): handle.close()
                query_data.close(); query_cache.close()
            for arm in ARMS: os.replace(partials[arm], partials[arm].with_suffix(""))
    finally:
        for data, cache in train.values(): data.close(); cache.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit"); fit.add_argument("--program", type=Path, required=True); fit.add_argument("--repo", type=Path, required=True); fit.add_argument("--artifact", type=Path, required=True); fit.add_argument("--report", type=Path, required=True)
    check = commands.add_parser("preflight"); check.add_argument("--program", type=Path, required=True); check.add_argument("--repo", type=Path, required=True); check.add_argument("--artifact", type=Path, required=True); check.add_argument("--artifact-sha256", required=True); check.add_argument("--report", type=Path, required=True); check.add_argument("--report-sha256", required=True); check.add_argument("--out", type=Path, required=True)
    sample = commands.add_parser("sample"); sample.add_argument("--program", type=Path, required=True); sample.add_argument("--repo", type=Path, required=True); sample.add_argument("--artifact", type=Path, required=True); sample.add_argument("--artifact-sha256", required=True); sample.add_argument("--report", type=Path, required=True); sample.add_argument("--report-sha256", required=True); sample.add_argument("--preflight", type=Path, required=True); sample.add_argument("--preflight-sha256", required=True); sample.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit": fit_model(args.program, args.repo, args.artifact, args.report)
    elif args.command == "preflight": preflight(args.program, args.repo, args.artifact, args.artifact_sha256, args.report, args.report_sha256, args.out)
    else: sample_all(args.program, args.repo, args.artifact, args.artifact_sha256, args.report, args.report_sha256, args.preflight, args.preflight_sha256, args.out)


if __name__ == "__main__": main()
