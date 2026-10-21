#!/usr/bin/env python
"""Frozen V36 local conditional residual-tail sufficiency audit."""
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
from hong2021_v30_backbone_audit import tail_diagnostics
from hong2021_v31_copula import (
    DOMAIN_ORDER,
    conditional_forward,
    conditional_inverse,
    load_model,
)
from hong2021_v34_nonlinear_sufficiency import pooled_fields
from hong2021_v35_spectrum_phase import _backbone, _open_split, load_program as load_v35_program


PROGRAM_SCHEMA = "hong2021-v36-local-conditional-scale-tail-sufficiency-program-v1"
PROGRAM_SHA256 = "6813b6ec10a042f20baf13b7243cc86e70a71f6e9171bcb7521b200a1cf19deb"
SCHEMA = "hong2021-v36-local-conditional-scale-tail-sufficiency-audit-v1"
QUANTILES = (0.001, 0.01, 0.5, 0.99, 0.999)
PATCH_OFFSETS = tuple(product((-1, 0, 1), repeat=3))
SCALAR_FEATURES = 5
FULL_FEATURES = SCALAR_FEATURES + 2 * 4 * len(PATCH_OFFSETS)
TRAIN_ROWS = 262144
VALIDATION_ROWS = 262144


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(path.resolve()) != PROGRAM_SHA256:
        raise ValueError("V36 program hash differs")
    program = json.loads(path.read_text())
    if program.get("schema") != PROGRAM_SCHEMA:
        raise ValueError("V36 program schema differs")
    parent = program["parent_evidence"]
    record_path = (repo / parent["v35_record"]).resolve()
    if sha256_file(record_path) != parent["v35_record_sha256"]:
        raise ValueError("V36 V35 record hash differs")
    record = json.loads(record_path.read_text())
    audit = record.get("audit", {})
    if (
        audit.get("classification") != parent["required_classification"]
        or audit.get("next") != parent["required_next"]
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V36 V35 parent conclusion or firewall differs")
    inherited = program["inherited_inputs"]
    v35_path = (repo / inherited["v35_program"]).resolve()
    if sha256_file(v35_path) != inherited["v35_program_sha256"]:
        raise ValueError("V36 inherited V35 program hash differs")
    v35, _ = load_v35_program(v35_path, repo)
    return program, v35


def selected_coordinates(
    selected_global: np.ndarray,
    cube_index: int,
    rows_per_cube: int,
    *,
    lattice: bool,
) -> np.ndarray:
    lower = cube_index * rows_per_cube
    left = np.searchsorted(selected_global, lower, side="left")
    right = np.searchsorted(selected_global, lower + rows_per_cube, side="left")
    local = selected_global[left:right] - lower
    if not len(local):
        return np.empty((0, 3), dtype=np.int64)
    if lattice:
        offset = np.asarray(
            (cube_index % 4, (cube_index // 4) % 4, (cube_index // 16) % 4),
            dtype=np.int64,
        )
        coordinate = np.column_stack(np.unravel_index(local, (16, 16, 16)))
        return coordinate * 4 + offset
    return np.column_stack(np.unravel_index(local, (64, 64, 64))).astype(np.int64)


def _sample_field(field: np.ndarray, coordinate: np.ndarray) -> np.ndarray:
    return np.asarray(field)[coordinate[:, 0], coordinate[:, 1], coordinate[:, 2]]


def local_features_targets(
    data: h5py.File,
    cache: h5py.File,
    index: int,
    coordinate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if coordinate.ndim != 2 or coordinate.shape[1] != 3:
        raise ValueError("V36 selected coordinate shape differs")
    count = np.asarray(data["input"][index, 0], dtype=np.float32)
    velocity = np.asarray(data["input"][index, 1], dtype=np.float32)
    dispersion = np.asarray(data["input"][index, 2], dtype=np.float32)
    backbone = _backbone(cache, index).astype(np.float32)
    truth = np.asarray(data["target"][index, 0], dtype=np.float32)
    native = {
        "logcount": np.log1p(count),
        "velocity": velocity,
        "dispersion": dispersion,
        "backbone": backbone,
    }
    parent_raw = pooled_fields(count, velocity, dispersion, backbone, 4)
    parent = {
        "logcount": parent_raw["log1p_block_count"],
        "velocity": parent_raw["block_mean_velocity_kms"],
        "dispersion": parent_raw["exact_population_velocity_dispersion_kms"],
        "backbone": parent_raw["backbone_mean_y"],
    }
    normalized_radius = np.sqrt(
        np.square((coordinate.astype(np.float64) + 0.5) * 0.3125 - 10.0).sum(axis=1)
    ) / 10.0
    pieces = [
        _sample_field(backbone, coordinate),
        _sample_field(native["logcount"], coordinate),
        _sample_field(velocity, coordinate),
        _sample_field(dispersion, coordinate),
        normalized_radius,
    ]
    for field in native.values():
        for offset in PATCH_OFFSETS:
            neighbor = (coordinate + np.asarray(offset)) % 64
            pieces.append(_sample_field(field, neighbor))
    parent_coordinate = coordinate // 4
    for field in parent.values():
        for offset in PATCH_OFFSETS:
            neighbor = (parent_coordinate + np.asarray(offset)) % 16
            pieces.append(_sample_field(field, neighbor))
    feature = np.column_stack(pieces).astype(np.float32)
    target = _sample_field(truth - backbone, coordinate).astype(np.float32)
    if feature.shape != (len(coordinate), FULL_FEATURES):
        raise RuntimeError("V36 feature shape differs")
    return feature, target


def collect_selected_rows(
    row: dict[str, Any],
    domain: str,
    split: str,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lattice = split == "train"
    rows_per_cube = 4096 if lattice else 64**3
    objects = int(row[f"{split}_objects"])
    features = []
    targets = []
    data, cache = _open_split(row, split)
    try:
        for index in range(objects):
            coordinate = selected_coordinates(
                selected, index, rows_per_cube, lattice=lattice
            )
            if len(coordinate):
                feature, target = local_features_targets(
                    data, cache, index, coordinate
                )
                features.append(feature)
                targets.append(target)
            if (index + 1) % 32 == 0 or index + 1 == objects:
                print(
                    f"[v36] collect {domain} {split} {index + 1}/{objects}",
                    flush=True,
                )
    finally:
        data.close()
        cache.close()
    result_x = np.concatenate(features)
    result_y = np.concatenate(targets)
    if len(result_y) != len(selected):
        raise RuntimeError("V36 selected row coverage differs")
    return result_x, result_y


def pinball_loss(prediction: np.ndarray, target: np.ndarray, quantile: float) -> float:
    difference = np.asarray(target, dtype=np.float64) - np.asarray(
        prediction, dtype=np.float64
    )
    return float(np.mean(np.maximum(quantile * difference, (quantile - 1.0) * difference)))


def quantile_metrics(
    prediction: np.ndarray, target: np.ndarray, quantile: float
) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    coverage = float(np.mean(target <= prediction))
    return {
        "rows": int(len(target)),
        "pinball_loss": pinball_loss(prediction, target, quantile),
        "empirical_fraction_residual_le_prediction": coverage,
        "coverage_error": coverage - quantile,
    }


def v31_quantile_prediction(
    backbone: np.ndarray, quantile: float, model: dict[str, Any]
) -> np.ndarray:
    edges = np.asarray(model["backbone_edges"], dtype=np.float64)
    levels = np.asarray(model["quantile_levels"], dtype=np.float64)
    table = np.asarray(model["residual_quantiles"], dtype=np.float64)
    per_bin = np.asarray([np.interp(quantile, levels, row) for row in table])
    bins = np.searchsorted(edges[1:-1], np.asarray(backbone), side="right")
    return per_bin[bins]


def _new_quantile_model(spec: dict[str, Any], quantile: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        learning_rate=float(spec["learning_rate"]),
        max_iter=int(spec["max_iter"]),
        max_leaf_nodes=int(spec["max_leaf_nodes"]),
        min_samples_leaf=int(spec["min_samples_leaf"]),
        l2_regularization=float(spec["l2_regularization"]),
        early_stopping=bool(spec["early_stopping"]),
        random_state=int(spec["random_state"]),
    )


def quantile_audit(
    program: dict[str, Any], v35: dict[str, Any], copula_model: dict[str, Any]
) -> dict[str, Any]:
    train_generator = np.random.default_rng(36031)
    train_x_parts = []
    train_y_parts = []
    train_selection_sha = {}
    import hashlib

    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        total = int(row["train_objects"]) * 4096
        selected = np.sort(train_generator.permutation(total)[:TRAIN_ROWS]).astype(np.int64)
        train_selection_sha[domain] = hashlib.sha256(selected.tobytes()).hexdigest()
        x, y = collect_selected_rows(row, domain, "train", selected)
        train_x_parts.append(x)
        train_y_parts.append(y)
    x_train = np.concatenate(train_x_parts)
    y_train = np.concatenate(train_y_parts)
    learner = program["native_voxel_quantile_audit"]["fixed_learner"]
    models: dict[float, dict[str, HistGradientBoostingRegressor]] = {}
    train_report = {}
    for quantile in QUANTILES:
        models[quantile] = {}
        train_report[str(quantile)] = {}
        for name, columns in (
            ("nonlinear_scalar_quantile", SCALAR_FEATURES),
            ("nonlinear_oriented_multiscale_quantile", FULL_FEATURES),
        ):
            print(
                f"[v36] fit q={quantile} model={name} rows={len(y_train)} features={columns}",
                flush=True,
            )
            model = _new_quantile_model(learner, quantile)
            model.fit(x_train[:, :columns], y_train)
            models[quantile][name] = model
            train_report[str(quantile)][name] = {
                "features": columns,
                "iterations": int(model.n_iter_),
                **quantile_metrics(
                    model.predict(x_train[:, :columns]), y_train, quantile
                ),
            }

    validation_generator = np.random.default_rng(36041)
    validation_report: dict[str, Any] = {}
    validation_selection_sha = {}
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        total = int(row["validation_objects"]) * 64**3
        selected = np.sort(
            validation_generator.permutation(total)[:VALIDATION_ROWS]
        ).astype(np.int64)
        validation_selection_sha[domain] = hashlib.sha256(selected.tobytes()).hexdigest()
        x, y = collect_selected_rows(row, domain, "validation", selected)
        validation_report[domain] = {}
        for quantile in QUANTILES:
            v31_prediction = v31_quantile_prediction(x[:, 0], quantile, copula_model)
            v31_metrics = quantile_metrics(v31_prediction, y, quantile)
            scalar_prediction = models[quantile]["nonlinear_scalar_quantile"].predict(
                x[:, :SCALAR_FEATURES]
            )
            scalar_metrics = quantile_metrics(scalar_prediction, y, quantile)
            local_prediction = models[quantile][
                "nonlinear_oriented_multiscale_quantile"
            ].predict(x)
            local_metrics = quantile_metrics(local_prediction, y, quantile)
            scalar_metrics["pinball_over_V31"] = float(
                scalar_metrics["pinball_loss"] / v31_metrics["pinball_loss"]
            )
            local_metrics["pinball_over_scalar"] = float(
                local_metrics["pinball_loss"] / scalar_metrics["pinball_loss"]
            )
            local_metrics["pinball_over_V31"] = float(
                local_metrics["pinball_loss"] / v31_metrics["pinball_loss"]
            )
            validation_report[domain][str(quantile)] = {
                "V31_scalar_backbone_bin_reference": v31_metrics,
                "nonlinear_scalar_quantile": scalar_metrics,
                "nonlinear_oriented_multiscale_quantile": local_metrics,
            }
    return {
        "train_rows": int(len(y_train)),
        "train_rows_per_source": TRAIN_ROWS,
        "validation_rows_per_source": VALIDATION_ROWS,
        "train_selection_sha256": train_selection_sha,
        "validation_selection_sha256": validation_selection_sha,
        "train": train_report,
        "validation": validation_report,
    }


def translated_tail_control(
    v35: dict[str, Any], copula_model: dict[str, Any]
) -> dict[str, Any]:
    result = {}
    directions = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        with h5py.File(row["phase_object_selection"], "r") as selection:
            indices = np.asarray(selection["source_index"], dtype=np.int64)
        reference = []
        candidate = []
        data, cache = _open_split(row, "validation")
        try:
            for position, index in enumerate(indices):
                backbone = _backbone(cache, int(index)).astype(np.float64)
                truth = np.asarray(data["target"][index, 0], dtype=np.float64)
                residual = truth - backbone
                reference_residual = residual - residual.mean(dtype=np.float64)
                uniform = conditional_forward(residual, backbone, copula_model)
                reference_field = (backbone + reference_residual).astype(np.float32)
                for direction in directions:
                    translated = np.roll(
                        uniform,
                        shift=tuple(16 * value for value in direction),
                        axis=(0, 1, 2),
                    )
                    candidate_residual = conditional_inverse(
                        translated, backbone, copula_model
                    ).astype(np.float64)
                    candidate_residual -= candidate_residual.mean(dtype=np.float64)
                    reference.append(reference_field)
                    candidate.append((backbone + candidate_residual).astype(np.float32))
                print(f"[v36-tail] {domain} {position + 1}/16", flush=True)
        finally:
            data.close()
            cache.close()
        metrics = tail_diagnostics(np.asarray(reference), np.asarray(candidate))
        result[domain] = metrics
    return result


def _tail_rule(
    ratios: dict[str, dict[str, float]],
    key: str,
    validation: dict[str, Any] | None = None,
    model_key: str | None = None,
) -> bool:
    def passes(quantile: float) -> bool:
        if not all(
            ratios[domain][str(quantile)][key] <= 0.90
            for domain in DOMAIN_ORDER
        ):
            return False
        if validation is None and model_key is None:
            return True
        if validation is None or model_key is None:
            raise ValueError("validation and model_key must be provided together")
        tolerance = 0.002 if quantile in (0.001, 0.999) else 0.005
        return all(
            abs(
                float(
                    validation[domain][str(quantile)][model_key]["coverage_error"]
                )
            )
            <= tolerance
            for domain in DOMAIN_ORDER
        )

    extreme = any(passes(q) for q in (0.001, 0.999))
    paired = all(passes(q) for q in (0.01, 0.99))
    return extreme or paired


def evaluate(program_path: Path, repo: Path) -> dict[str, Any]:
    program, v35 = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V36 audit requires a clean committed worktree")
    copula = v35["v31_conditional_copula"]
    copula_model = load_model(Path(copula["artifact"]), copula["artifact_sha256"])
    quantiles = quantile_audit(program, v35, copula_model)
    translated = translated_tail_control(v35, copula_model)
    ratios = {
        domain: {
            q: {
                "scalar_over_V31": row["nonlinear_scalar_quantile"]["pinball_over_V31"],
                "local_over_scalar": row["nonlinear_oriented_multiscale_quantile"][
                    "pinball_over_scalar"
                ],
            }
            for q, row in quantiles["validation"][domain].items()
        }
        for domain in DOMAIN_ORDER
    }
    validation = quantiles["validation"]
    scalar_ratio_gate = _tail_rule(ratios, "scalar_over_V31")
    local_ratio_gate = _tail_rule(ratios, "local_over_scalar")
    scalar_supported = _tail_rule(
        ratios,
        "scalar_over_V31",
        validation,
        "nonlinear_scalar_quantile",
    )
    local_supported = _tail_rule(
        ratios,
        "local_over_scalar",
        validation,
        "nonlinear_oriented_multiscale_quantile",
    )
    transport_material = any(
        not bool(row["Q3_pass"]) or not bool(row["Q4_pass"])
        for row in translated.values()
    )
    if local_supported:
        classification = "local_multiscale_conditional_tail_is_supported"
        next_step = "fit_train_only_local_conditional_body_plus_extreme_value_tail_continuation"
    elif scalar_supported:
        classification = "smooth_nonlinear_scalar_tail_is_supported_without_local_gain"
        next_step = "replace_V31_bins_with_smooth_scalar_body_plus_extreme_value_tail_continuation"
    elif transport_material:
        classification = "tail_failure_is_joint_donor_query_coupling_not_marginal_tail_prediction"
        next_step = "replace_donor_transport_with_joint_query_aligned_conditional_stochastic_field"
    else:
        classification = "registered_local_tail_and_transport_mechanisms_not_material"
        next_step = "audit_Q3_Q4_implementation_and_donor_provenance"
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_development_only_tail_sufficiency_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "sklearn_version": sklearn.__version__,
        "conditional_quantiles": quantiles,
        "translated_tail_causal_control": translated,
        "nonlinear_scalar_tail_ratio_gate_passed": scalar_ratio_gate,
        "local_tail_ratio_gate_passed": local_ratio_gate,
        "nonlinear_scalar_tail_supported": scalar_supported,
        "local_tail_supported": local_supported,
        "translated_transport_tail_material": transport_material,
        "classification": classification,
        "next": next_step,
        "validation_used_for_fit_or_early_stopping": False,
        "validation_extrema_used_for_tail_fit": False,
        "field_clipping": False,
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
        raise FileExistsError("V36 refuses to overwrite its audit")
    report = evaluate(args.program.resolve(), args.repo.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
