#!/usr/bin/env python
"""Frozen V40 object- and connected-structure-level observable sufficiency audit."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import sklearn
from scipy import ndimage
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v30_backbone_audit import block_mean
from hong2021_v31_copula import DOMAIN_ORDER
from hong2021_v34_nonlinear_sufficiency import MODEL_FEATURE_COUNTS, multiscale_features, pooled_fields
from hong2021_v35_spectrum_phase import _backbone, _open_split, load_program as load_v35_program


PROGRAM_SCHEMA = "hong2021-v40-object-structure-observable-sufficiency-program-v1"
PROGRAM_SHA256 = "707f62ccb1ea6d7aee5ef6fa77355e28412119fffdd694924f2d7280f97d3162"
SCHEMA = "hong2021-v40-object-structure-observable-sufficiency-audit-v1"
FACTOR = 4
GRID = 16
THRESHOLDS = {"dense": 0.999, "extreme": 0.9999}
MAX_ROWS = 65536
SPATIAL_CONTROL_SHIFT = (3, 5, 7)
FULL_COLUMNS = np.arange(MODEL_FEATURE_COUNTS["nonlinear_oriented_multiscale"])
BACKBONE_COLUMNS = np.asarray(
    [3, 4, 5, *range(87, 114), *range(195, 222)], dtype=np.int64
)
OBJECT_STATISTICS = ("mean", "std", "q10", "q50", "q90", "q99", "maximum")
OBJECT_FIELDS = (
    "log1p_block_count",
    "block_mean_velocity_kms",
    "exact_population_velocity_dispersion_kms",
    "backbone_mean_y",
)


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(path.resolve()) != PROGRAM_SHA256:
        raise ValueError("V40 program hash differs")
    program = json.loads(path.read_text())
    if program.get("schema") != PROGRAM_SCHEMA:
        raise ValueError("V40 program schema differs")
    parent = program["parent_evidence"]
    record_path = (repo / parent["v39_record"]).resolve()
    if sha256_file(record_path) != parent["v39_record_sha256"]:
        raise ValueError("V40 V39 record hash differs")
    record = json.loads(record_path.read_text())
    decision = record.get("decision", {})
    if (
        decision.get("classification") != parent["required_classification"]
        or decision.get("next") != parent["required_next"]
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V40 V39 parent conclusion or firewall differs")
    inherited = program["inherited_inputs"]
    v35_path = (repo / inherited["v35_program"]).resolve()
    if sha256_file(v35_path) != inherited["v35_program_sha256"]:
        raise ValueError("V40 V35 program hash differs")
    v35, _ = load_v35_program(v35_path, repo)
    return program, v35


def block_max(value: np.ndarray, factor: int = FACTOR) -> np.ndarray:
    field = np.asarray(value)
    if field.shape != (64, 64, 64) or 64 % factor:
        raise ValueError("V40 block maximum requires a native 64-cube")
    grid = 64 // factor
    return field.reshape(grid, factor, grid, factor, grid, factor).max(axis=(1, 3, 5))


def top_count_mask(score: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    if values.shape != (GRID, GRID, GRID) or not np.isfinite(values).all():
        raise ValueError("V40 top-count score must be a finite factor-4 grid")
    if count < 0 or count > values.size:
        raise ValueError("V40 top-count is outside the grid")
    selected = np.zeros(values.size, dtype=bool)
    if count:
        order = np.lexsort((np.arange(values.size), -values.reshape(-1)))
        selected[order[:count]] = True
    return selected.reshape(values.shape)


def component_recall(
    native_positive: np.ndarray,
    delta_squared: np.ndarray,
    selected_blocks: np.ndarray,
    factor: int = FACTOR,
) -> dict[str, float | int]:
    positive = np.asarray(native_positive, dtype=bool)
    mass = np.asarray(delta_squared, dtype=np.float64)
    selected = np.asarray(selected_blocks, dtype=bool)
    if positive.shape != (64, 64, 64) or mass.shape != positive.shape:
        raise ValueError("V40 component fields require native 64-cubes")
    if selected.shape != (64 // factor,) * 3 or np.any(mass < 0):
        raise ValueError("V40 selected block grid or component mass differs")
    labels, components = ndimage.label(positive, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if components == 0:
        return {"components": 0, "hit_components": 0, "mass": 0.0, "hit_mass": 0.0}
    native_selected = np.repeat(np.repeat(np.repeat(selected, factor, axis=0), factor, axis=1), factor, axis=2)
    hit = np.unique(labels[native_selected & positive])
    hit = hit[hit > 0]
    component_mass = np.bincount(
        labels.reshape(-1), weights=np.where(positive, mass, 0.0).reshape(-1), minlength=components + 1
    )[1:]
    return {
        "components": int(components),
        "hit_components": int(len(hit)),
        "mass": float(component_mass.sum()),
        "hit_mass": float(component_mass[hit - 1].sum()) if len(hit) else 0.0,
    }


def object_features(fields: dict[str, np.ndarray]) -> np.ndarray:
    pieces = []
    for name in OBJECT_FIELDS:
        value = np.asarray(fields[name], dtype=np.float64).reshape(-1)
        if value.size != GRID**3 or not np.isfinite(value).all():
            raise ValueError("V40 object summary field differs")
        pieces.extend(
            (
                float(value.mean()),
                float(value.std()),
                *np.quantile(value, (0.1, 0.5, 0.9, 0.99)).tolist(),
                float(value.max()),
            )
        )
    result = np.asarray(pieces, dtype=np.float32)
    if result.shape != (len(OBJECT_FIELDS) * len(OBJECT_STATISTICS),):
        raise RuntimeError("V40 object feature count differs")
    return result


def classify(structure_supported: bool, amplitude_supported: bool) -> tuple[str, str]:
    if structure_supported and amplitude_supported:
        return (
            "object_amplitude_and_structure_location_are_transferably_observable",
            "freeze_two_stage_structure_seeded_amplitude_calibrated_stochastic_residual",
        )
    if structure_supported:
        return (
            "structure_location_is_observable_but_object_amplitude_is_not",
            "freeze_structure_seeded_rank_preserving_copula_with_train_marginal_amplitude",
        )
    if amplitude_supported:
        return (
            "object_amplitude_is_observable_but_structure_location_is_not",
            "freeze_object_level_conditional_scale_with_spatially_exchangeable_LCDM_innovation",
        )
    return (
        "backbone_observables_are_insufficient_for_rare_structure_reconstruction",
        "stop_learning_high_resolution_z0_density_from_CF4_and_treat_unresolved_modes_as_a_conditional_LCDM_prior",
    )


class PrioritySampler:
    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.priority = np.empty(0, dtype=np.float64)
        self.value = np.empty((0, len(FULL_COLUMNS)), dtype=np.float32)
        self.seen = 0

    def add(self, value: np.ndarray) -> None:
        rows = np.asarray(value, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != len(FULL_COLUMNS):
            raise ValueError("V40 reservoir row shape differs")
        if not len(rows):
            return
        self.seen += len(rows)
        priority = self.rng.random(len(rows))
        if len(rows) > self.capacity:
            keep = np.argpartition(priority, self.capacity - 1)[: self.capacity]
            rows, priority = rows[keep], priority[keep]
        combined_priority = np.concatenate((self.priority, priority))
        combined_value = np.concatenate((self.value, rows))
        if len(combined_priority) > self.capacity:
            keep = np.argpartition(combined_priority, self.capacity - 1)[: self.capacity]
            combined_priority, combined_value = combined_priority[keep], combined_value[keep]
        self.priority, self.value = combined_priority, combined_value

    def result(self) -> np.ndarray:
        order = np.argsort(self.priority, kind="stable")
        return self.value[order]


def exact_train_thresholds(row: dict[str, Any], domain: str) -> dict[str, float]:
    objects = int(row["train_objects"])
    values = np.empty(objects * 64**3, dtype=np.float32)
    data, cache = _open_split(row, "train")
    cache.close()
    try:
        for index in range(objects):
            lower = index * 64**3
            values[lower : lower + 64**3] = np.asarray(data["target"][index, 0], dtype=np.float32).reshape(-1) * np.float32(4.5)
            if (index + 1) % 64 == 0 or index + 1 == objects:
                print(f"[v40-threshold] {domain} {index + 1}/{objects}", flush=True)
    finally:
        data.close()
    quantile = np.quantile(values, tuple(THRESHOLDS.values()), overwrite_input=True)
    return {name: float(value) for name, value in zip(THRESHOLDS, quantile)}


def _cube_fields(data: h5py.File, cache: h5py.File, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = np.asarray(data["input"][index, 0], dtype=np.float32)
    velocity = np.asarray(data["input"][index, 1], dtype=np.float32)
    dispersion = np.asarray(data["input"][index, 2], dtype=np.float32)
    backbone = _backbone(cache, index).astype(np.float32)
    truth = np.asarray(data["target"][index, 0], dtype=np.float32)
    return count, velocity, dispersion, backbone, truth


def collect_train(
    row: dict[str, Any], domain: str, thresholds: dict[str, float], domain_index: int
) -> tuple[dict[str, dict[str, Any]], np.ndarray, dict[str, np.ndarray]]:
    samplers = {}
    for name in THRESHOLDS:
        base = int({"dense": 40031, "extreme": 40041}[name]) + 101 * domain_index
        samplers[name] = {
            "positive": PrioritySampler(MAX_ROWS, base + 1),
            "negative": PrioritySampler(MAX_ROWS, base + 2),
        }
    object_x, amplitude_y, extreme_y = [], [], []
    objects = int(row["train_objects"])
    data, cache = _open_split(row, "train")
    try:
        voxel = float(data.attrs["voxel_mpc_h"])
        for index in range(objects):
            count, velocity, dispersion, backbone, truth = _cube_fields(data, cache, index)
            feature, _ = multiscale_features(count, velocity, dispersion, backbone, truth, FACTOR, voxel_mpc_h=voxel)
            flat_feature = feature.reshape(GRID**3, -1)
            log10rho = np.asarray(truth, dtype=np.float64) * 4.5
            maximum = block_max(log10rho).reshape(-1)
            for name in THRESHOLDS:
                positive = maximum > thresholds[name]
                samplers[name]["positive"].add(flat_feature[positive])
                samplers[name]["negative"].add(flat_feature[~positive])
            pooled = pooled_fields(count, velocity, dispersion, backbone, FACTOR)
            object_x.append(object_features(pooled))
            delta = np.power(10.0, log10rho) - 1.0
            amplitude_y.append(float(np.log10(np.square(delta).mean(dtype=np.float64))))
            extreme_y.append(float(np.quantile(log10rho, 0.99999)))
            if (index + 1) % 32 == 0 or index + 1 == objects:
                print(f"[v40-train] {domain} {index + 1}/{objects}", flush=True)
    finally:
        data.close(); cache.close()
    blocks = {}
    for name in THRESHOLDS:
        positive = samplers[name]["positive"].result()
        negative = samplers[name]["negative"].result()
        blocks[name] = {
            "positive": positive,
            "negative": negative,
            "positive_seen": samplers[name]["positive"].seen,
            "negative_seen": samplers[name]["negative"].seen,
        }
    targets = {
        "log10_mean_delta_squared": np.asarray(amplitude_y, dtype=np.float64),
        "q99_999_log10rho": np.asarray(extreme_y, dtype=np.float64),
    }
    return blocks, np.asarray(object_x, dtype=np.float32), targets


def _classifier(spec: dict[str, Any]) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss=spec["loss"], learning_rate=float(spec["learning_rate"]),
        max_iter=int(spec["max_iter"]), max_leaf_nodes=int(spec["max_leaf_nodes"]),
        min_samples_leaf=int(spec["min_samples_leaf"]), l2_regularization=float(spec["l2_regularization"]),
        early_stopping=bool(spec["early_stopping"]), random_state=int(spec["random_state"]),
    )


def _regressor(spec: dict[str, Any]) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=spec["loss"], learning_rate=float(spec["learning_rate"]),
        max_iter=int(spec["max_iter"]), max_leaf_nodes=int(spec["max_leaf_nodes"]),
        min_samples_leaf=int(spec["min_samples_leaf"]), l2_regularization=float(spec["l2_regularization"]),
        early_stopping=bool(spec["early_stopping"]), random_state=int(spec["random_state"]),
    )


def fit_classifiers(
    program: dict[str, Any], train: dict[str, dict[str, dict[str, Any]]]
) -> tuple[dict[str, dict[str, dict[str, HistGradientBoostingClassifier]]], dict[str, Any]]:
    models: dict[str, dict[str, dict[str, HistGradientBoostingClassifier]]] = {}
    report = {}
    families = {"pooled": tuple(DOMAIN_ORDER)}
    families.update({f"leave_{domain}_out": tuple(d for d in DOMAIN_ORDER if d != domain) for domain in DOMAIN_ORDER})
    spec = program["fixed_training"]["block_classifier"]
    for threshold in THRESHOLDS:
        models[threshold], report[threshold] = {}, {}
        for family, included in families.items():
            per_class = min(
                min(len(train[domain][threshold]["positive"]), len(train[domain][threshold]["negative"]))
                for domain in included
            )
            x_parts, y_parts = [], []
            for domain in included:
                positive = train[domain][threshold]["positive"][:per_class]
                negative = train[domain][threshold]["negative"][:per_class]
                x_parts.extend((positive, negative))
                y_parts.extend((np.ones(per_class, dtype=np.uint8), np.zeros(per_class, dtype=np.uint8)))
            x = np.concatenate(x_parts); y = np.concatenate(y_parts)
            models[threshold][family] = {}
            report[threshold][family] = {"included_sources": list(included), "rows_per_source_per_class": per_class, "models": {}}
            for name, columns in (("full", FULL_COLUMNS), ("backbone_only", BACKBONE_COLUMNS)):
                model = _classifier(spec)
                print(f"[v40-fit-block] {threshold} {family} {name} rows={len(y)} features={len(columns)}", flush=True)
                model.fit(x[:, columns], y)
                models[threshold][family][name] = model
                report[threshold][family]["models"][name] = {
                    "features": int(len(columns)), "iterations": int(model.n_iter_),
                    "train_average_precision": float(average_precision_score(y, model.predict_proba(x[:, columns])[:, 1])),
                }
    return models, report


def fit_object_models(
    program: dict[str, Any], object_x: dict[str, np.ndarray], targets: dict[str, dict[str, np.ndarray]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    families = {"pooled": tuple(DOMAIN_ORDER)}
    families.update({f"leave_{domain}_out": tuple(d for d in DOMAIN_ORDER if d != domain) for domain in DOMAIN_ORDER})
    spec = program["fixed_training"]["object_regressor"]
    shuffled: dict[str, dict[str, np.ndarray]] = {target: {} for target in targets[DOMAIN_ORDER[0]]}
    rng = np.random.default_rng(40071)
    for target in shuffled:
        for domain in DOMAIN_ORDER:
            value = targets[domain][target]
            shuffled[target][domain] = value[rng.permutation(len(value))]
    models, report = {}, {}
    for target in shuffled:
        models[target], report[target] = {}, {}
        for family, included in families.items():
            x = np.concatenate([object_x[d] for d in included])
            y = np.concatenate([targets[d][target] for d in included])
            y_shuffled = np.concatenate([shuffled[target][d] for d in included])
            weights = np.concatenate([np.full(len(object_x[d]), 1.0 / len(object_x[d])) for d in included])
            weights *= len(weights) / weights.sum()
            models[target][family] = {}
            report[target][family] = {
                "included_sources": list(included),
                "equal_source_constant": float(np.mean([targets[d][target].mean() for d in included])),
                "models": {},
            }
            for name, columns, fit_y in (
                ("full", np.arange(object_x[DOMAIN_ORDER[0]].shape[1]), y),
                ("backbone_only", np.arange(21, 28), y),
                ("permuted_target_control", np.arange(object_x[DOMAIN_ORDER[0]].shape[1]), y_shuffled),
            ):
                model = _regressor(spec)
                print(f"[v40-fit-object] {target} {family} {name} rows={len(y)} features={len(columns)}", flush=True)
                model.fit(x[:, columns], fit_y, sample_weight=weights)
                models[target][family][name] = model
                report[target][family]["models"][name] = {"features": int(len(columns)), "iterations": int(model.n_iter_)}
    return models, report


@dataclass
class LocationAccumulator:
    labels: list[np.ndarray] = field(default_factory=list)
    scores: list[np.ndarray] = field(default_factory=list)
    positive: int = 0
    selected: int = 0
    intersected: int = 0
    components: int = 0
    hit_components: int = 0
    mass: float = 0.0
    hit_mass: float = 0.0

    def add(self, label: np.ndarray, score: np.ndarray, native_positive: np.ndarray, delta_squared: np.ndarray) -> None:
        truth = np.asarray(label, dtype=bool)
        values = np.asarray(score, dtype=np.float64)
        selected = top_count_mask(values, int(truth.sum()))
        self.labels.append(truth.reshape(-1)); self.scores.append(values.reshape(-1))
        self.positive += int(truth.sum()); self.selected += int(selected.sum())
        self.intersected += int(np.count_nonzero(selected & truth))
        components = component_recall(native_positive, delta_squared, selected)
        for key in ("components", "hit_components", "mass", "hit_mass"):
            setattr(self, key, getattr(self, key) + components[key])

    def result(self) -> dict[str, float | int]:
        labels = np.concatenate(self.labels); scores = np.concatenate(self.scores)
        prevalence = float(labels.mean())
        average_precision = float(average_precision_score(labels, scores))
        return {
            "blocks": int(len(labels)), "positive_blocks": self.positive,
            "positive_prevalence": prevalence, "average_precision": average_precision,
            "average_precision_over_prevalence": float(average_precision / prevalence),
            "top_count_precision": float(self.intersected / self.selected),
            "top_count_recall": float(self.intersected / self.positive),
            "components": self.components, "hit_components": self.hit_components,
            "component_number_recall": float(self.hit_components / self.components),
            "component_delta_squared_mass_recall": float(self.hit_mass / self.mass),
        }


def evaluate_locations(
    v35: dict[str, Any], thresholds: dict[str, dict[str, float]], models: dict[str, Any]
) -> dict[str, Any]:
    result = {}
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        families = ("pooled", f"leave_{domain}_out")
        accumulators = {
            threshold: {
                family: {name: LocationAccumulator() for name in ("full", "backbone_only", "spatial_control_full")}
                for family in families
            }
            for threshold in THRESHOLDS
        }
        objects = int(row["validation_objects"])
        data, cache = _open_split(row, "validation")
        try:
            voxel = float(data.attrs["voxel_mpc_h"])
            for index in range(objects):
                count, velocity, dispersion, backbone, truth = _cube_fields(data, cache, index)
                feature, _ = multiscale_features(count, velocity, dispersion, backbone, truth, FACTOR, voxel_mpc_h=voxel)
                flat = feature.reshape(GRID**3, -1)
                log10rho = np.asarray(truth, dtype=np.float64) * 4.5
                delta_squared = np.square(np.power(10.0, log10rho) - 1.0)
                maximum = block_max(log10rho)
                for threshold in THRESHOLDS:
                    native_positive = log10rho > thresholds[domain][threshold]
                    block_label = maximum > thresholds[domain][threshold]
                    for family in families:
                        full_score = models[threshold][family]["full"].predict_proba(flat[:, FULL_COLUMNS])[:, 1].reshape(GRID, GRID, GRID)
                        backbone_score = models[threshold][family]["backbone_only"].predict_proba(flat[:, BACKBONE_COLUMNS])[:, 1].reshape(GRID, GRID, GRID)
                        accumulators[threshold][family]["full"].add(block_label, full_score, native_positive, delta_squared)
                        accumulators[threshold][family]["backbone_only"].add(block_label, backbone_score, native_positive, delta_squared)
                        accumulators[threshold][family]["spatial_control_full"].add(
                            block_label, np.roll(full_score, SPATIAL_CONTROL_SHIFT, axis=(0, 1, 2)), native_positive, delta_squared
                        )
                if (index + 1) % 16 == 0 or index + 1 == objects:
                    print(f"[v40-location] {domain} {index + 1}/{objects}", flush=True)
        finally:
            data.close(); cache.close()
        result[domain] = {}
        for threshold in THRESHOLDS:
            result[domain][threshold] = {}
            for family in families:
                metrics = {name: accumulator.result() for name, accumulator in accumulators[threshold][family].items()}
                metrics["ratios"] = {
                    "full_over_backbone_average_precision": float(metrics["full"]["average_precision"] / metrics["backbone_only"]["average_precision"]),
                    "full_over_spatial_control_average_precision": float(metrics["full"]["average_precision"] / metrics["spatial_control_full"]["average_precision"]),
                }
                result[domain][threshold][family] = metrics
    return result


def _regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=np.float64); target = np.asarray(target, dtype=np.float64)
    correlation = spearmanr(prediction, target).statistic
    return {"objects": int(len(target)), "rmse": float(np.sqrt(np.square(prediction - target).mean())), "spearman": float(correlation if np.isfinite(correlation) else 0.0)}


def evaluate_objects(
    v35: dict[str, Any], models: dict[str, Any], fit_report: dict[str, Any]
) -> dict[str, Any]:
    validation_x, validation_y = {}, {}
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        x, amplitude, extreme = [], [], []
        objects = int(row["validation_objects"])
        data, cache = _open_split(row, "validation")
        try:
            for index in range(objects):
                count, velocity, dispersion, backbone, truth = _cube_fields(data, cache, index)
                pooled = pooled_fields(count, velocity, dispersion, backbone, FACTOR)
                x.append(object_features(pooled))
                log10rho = np.asarray(truth, dtype=np.float64) * 4.5
                delta = np.power(10.0, log10rho) - 1.0
                amplitude.append(float(np.log10(np.square(delta).mean(dtype=np.float64))))
                extreme.append(float(np.quantile(log10rho, 0.99999)))
        finally:
            data.close(); cache.close()
        validation_x[domain] = np.asarray(x, dtype=np.float32)
        validation_y[domain] = {
            "log10_mean_delta_squared": np.asarray(amplitude),
            "q99_999_log10rho": np.asarray(extreme),
        }
    report = {}
    for target in models:
        report[target] = {}
        for domain in DOMAIN_ORDER:
            report[target][domain] = {}
            for family in ("pooled", f"leave_{domain}_out"):
                x = validation_x[domain]; y = validation_y[domain][target]
                constant_value = float(fit_report[target][family]["equal_source_constant"])
                constant = _regression_metrics(np.full_like(y, constant_value), y)
                full = _regression_metrics(models[target][family]["full"].predict(x), y)
                backbone = _regression_metrics(models[target][family]["backbone_only"].predict(x[:, 21:28]), y)
                control = _regression_metrics(models[target][family]["permuted_target_control"].predict(x), y)
                report[target][domain][family] = {
                    "constant": constant, "full": full, "backbone_only": backbone,
                    "permuted_target_control": control,
                    "ratios": {
                        "full_rmse_over_constant": float(full["rmse"] / constant["rmse"]),
                        "full_rmse_over_backbone": float(full["rmse"] / backbone["rmse"]),
                        "full_rmse_over_permuted_control": float(full["rmse"] / control["rmse"]),
                    },
                }
    return report


def support_decision(location: dict[str, Any], objects: dict[str, Any]) -> tuple[list[str], bool, bool, bool]:
    supported_thresholds = []
    increment_thresholds = []
    for threshold in THRESHOLDS:
        pooled_ok = True; leave_ok = True; increment_ok = True
        for domain in DOMAIN_ORDER:
            pooled = location[domain][threshold]["pooled"]
            leave = location[domain][threshold][f"leave_{domain}_out"]
            pooled_ok &= (
                pooled["full"]["average_precision_over_prevalence"] >= 2.0
                and pooled["full"]["top_count_recall"] >= 0.20
                and pooled["full"]["component_delta_squared_mass_recall"] >= 0.50
                and pooled["ratios"]["full_over_spatial_control_average_precision"] >= 1.25
            )
            leave_ok &= (
                leave["full"]["average_precision_over_prevalence"] >= 1.5
                and leave["full"]["top_count_recall"] >= 0.15
                and leave["full"]["component_delta_squared_mass_recall"] >= 0.35
                and leave["ratios"]["full_over_spatial_control_average_precision"] >= 1.25
            )
            increment_ok &= (
                pooled["ratios"]["full_over_backbone_average_precision"] >= 1.10
                and leave["ratios"]["full_over_backbone_average_precision"] >= 1.10
            )
        if pooled_ok and leave_ok:
            supported_thresholds.append(threshold)
        if increment_ok:
            increment_thresholds.append(threshold)
    amplitude = objects["log10_mean_delta_squared"]
    amplitude_supported = all(
        all(
            amplitude[domain][family]["ratios"]["full_rmse_over_constant"] <= 0.90
            and amplitude[domain][family]["full"]["spearman"] >= 0.30
            and amplitude[domain][family]["ratios"]["full_rmse_over_permuted_control"] <= 0.90
            for family in ("pooled", f"leave_{domain}_out")
        )
        for domain in DOMAIN_ORDER
    )
    return supported_thresholds, bool(supported_thresholds), bool(amplitude_supported), bool(increment_thresholds)


def evaluate(program_path: Path, repo: Path) -> dict[str, Any]:
    program, v35 = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V40 audit requires a clean committed worktree")
    thresholds = {domain: exact_train_thresholds(v35["development_domains"][domain], domain) for domain in DOMAIN_ORDER}
    block_train, object_x, object_y, train_summary = {}, {}, {}, {}
    for index, domain in enumerate(DOMAIN_ORDER):
        blocks, features, targets = collect_train(v35["development_domains"][domain], domain, thresholds[domain], index)
        block_train[domain], object_x[domain], object_y[domain] = blocks, features, targets
        train_summary[domain] = {
            "objects": int(len(features)),
            "thresholds_log10rho": thresholds[domain],
            "block_rows": {
                name: {
                    "positive_seen": int(blocks[name]["positive_seen"]),
                    "negative_seen": int(blocks[name]["negative_seen"]),
                    "positive_retained": int(len(blocks[name]["positive"])),
                    "negative_retained": int(len(blocks[name]["negative"])),
                }
                for name in THRESHOLDS
            },
        }
    classifiers, classifier_fit = fit_classifiers(program, block_train)
    object_models, object_fit = fit_object_models(program, object_x, object_y)
    location = evaluate_locations(v35, thresholds, classifiers)
    object_report = evaluate_objects(v35, object_models, object_fit)
    supported_thresholds, structure_supported, amplitude_supported, increment_supported = support_decision(location, object_report)
    classification, next_step = classify(structure_supported, amplitude_supported)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_development_only_observable_sufficiency_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "sklearn_version": sklearn.__version__,
        "train_summary": train_summary,
        "classifier_fit": classifier_fit,
        "object_fit": object_fit,
        "block_and_connected_structure_validation": location,
        "object_validation": object_report,
        "structure_supported_thresholds": supported_thresholds,
        "structure_location_supported": structure_supported,
        "object_amplitude_supported": amplitude_supported,
        "observable_increment_supported": increment_supported,
        "classification": classification,
        "next": next_step,
        "new_generator_fit_or_sampled": False,
        "validation_used_for_fit_threshold_or_hyperparameter_choice": False,
        "simulation_identity_feature_used": False,
        "density_field_clipping": False,
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
        raise FileExistsError("V40 refuses to overwrite its audit")
    report = evaluate(args.program.resolve(), args.repo.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
