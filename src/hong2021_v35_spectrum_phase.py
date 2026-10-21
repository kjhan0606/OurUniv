#!/usr/bin/env python
"""Frozen V35 conditional spectrum, copula support, and phase-coupling audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v30_backbone_audit import fourier_masks, tail_diagnostics
from hong2021_v31_copula import (
    DOMAIN_ORDER,
    conditional_forward,
    conditional_inverse,
    load_model,
)
from hong2021_v33_kinematic_data import CHANNELS, OUTPUT_SCHEMA


PROGRAM_SCHEMA = "hong2021-v35-conditional-residual-spectrum-phase-coupling-program-v1"
PROGRAM_SHA256 = "161b5b9c7345c6777e39ebc342243ee12226b75d8e461c0db69f410ea2193e4a"
SCHEMA = "hong2021-v35-conditional-residual-spectrum-phase-coupling-audit-v1"
EDGES = np.asarray((0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, np.inf))
TRANSLATIONS = (1, 2, 4, 8, 16)
DIRECTIONS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(path.resolve()) != PROGRAM_SHA256:
        raise ValueError("V35 program hash differs")
    program = json.loads(path.read_text())
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_revision_before_implementation_or_execution"
        or tuple(program["development_domains"]) != DOMAIN_ORDER
    ):
        raise ValueError("V35 program schema, status, or domain order differs")
    parent = program["parent_evidence"]
    record_path = (repo / parent["v34_record"]).resolve()
    if sha256_file(record_path) != parent["v34_record_sha256"]:
        raise ValueError("V35 V34 record hash differs")
    record = json.loads(record_path.read_text())
    audit = record.get("audit", {})
    if (
        audit.get("classification") != parent["required_classification"]
        or audit.get("next") != parent["required_next"]
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V35 V34 parent conclusion or firewall differs")
    for domain in DOMAIN_ORDER:
        row = program["development_domains"][domain]
        for split in ("train", "validation"):
            for kind in ("data", "cache"):
                artifact = Path(row[f"{split}_{kind}"])
                if sha256_file(artifact) != row[f"{split}_{kind}_sha256"]:
                    raise ValueError(f"V35 {domain} {split} {kind} hash differs")
        selection = Path(row["phase_object_selection"])
        if sha256_file(selection) != row["phase_object_selection_sha256"]:
            raise ValueError(f"V35 {domain} phase selection hash differs")
    copula = program["v31_conditional_copula"]
    if (
        sha256_file(Path(copula["artifact"])) != copula["artifact_sha256"]
        or sha256_file(Path(copula["fit_report"])) != copula["fit_report_sha256"]
    ):
        raise ValueError("V35 V31 copula artifact/report hash differs")
    fit_report = json.loads(Path(copula["fit_report"]).read_text())
    if fit_report.get("artifact_sha256") != copula["artifact_sha256"]:
        raise ValueError("V35 V31 copula artifact/report binding differs")
    return program, fit_report


def transforms_and_spectra(
    *fields: np.ndarray, voxel_mpc_h: float = 0.3125
) -> tuple[list[np.ndarray], np.ndarray]:
    masks = fourier_masks(64, voxel_mpc_h, EDGES)
    transforms = []
    spectra = []
    for field in fields:
        value = np.asarray(field, dtype=np.float64)
        if value.shape != (64, 64, 64) or not np.isfinite(value).all():
            raise ValueError("V35 Fourier field must be a finite 64-cube")
        transform = np.fft.fftn(value - value.mean(dtype=np.float64))
        transforms.append(transform)
        spectra.append(
            [float(np.square(np.abs(transform[mask])).mean()) for mask in masks]
        )
    return transforms, np.asarray(spectra, dtype=np.float64)


def band_cross(first: np.ndarray, second: np.ndarray, voxel_mpc_h: float = 0.3125) -> np.ndarray:
    masks = fourier_masks(64, voxel_mpc_h, EDGES)
    return np.asarray(
        [float(np.real(first[mask] * np.conj(second[mask])).mean()) for mask in masks],
        dtype=np.float64,
    )


def decomposition_summary(
    backbone_power: np.ndarray, residual_power: np.ndarray, cross: np.ndarray
) -> dict[str, list[float]]:
    backbone_power = np.asarray(backbone_power, dtype=np.float64)
    residual_power = np.asarray(residual_power, dtype=np.float64)
    cross = np.asarray(cross, dtype=np.float64)
    total = backbone_power + residual_power + 2.0 * cross
    if np.any(backbone_power <= 0) or np.any(residual_power <= 0) or np.any(total <= 0):
        raise ValueError("V35 Fourier decomposition is not positive")
    return {
        "backbone_power": backbone_power.tolist(),
        "residual_power": residual_power.tolist(),
        "backbone_residual_cross_power": cross.tolist(),
        "total_power": total.tolist(),
        "backbone_residual_cross_correlation": (
            cross / np.sqrt(backbone_power * residual_power)
        ).tolist(),
        "cross_term_fraction_of_total_power": (2.0 * cross / total).tolist(),
    }


def _open_split(row: dict[str, Any], split: str) -> tuple[h5py.File, h5py.File]:
    data = h5py.File(row[f"{split}_data"], "r")
    cache = h5py.File(row[f"{split}_cache"], "r")
    objects = int(row[f"{split}_objects"])
    valid = (
        tuple(data["input"].shape) == (objects, 3, 64, 64, 64)
        and str(data.attrs.get("schema", "")) == OUTPUT_SCHEMA
        and str(data.attrs.get("channels", "")) == CHANNELS
        and bool(data.attrs.get("complete", False))
        and tuple(cache["conditional_mean"].shape) == (objects, 1, 64, 64, 64)
        and tuple(cache["observable_context_features"].shape) == (objects, 8)
    )
    if not valid:
        data.close()
        cache.close()
        raise ValueError("V35 data/cache shape or metadata differs")
    return data, cache


def _backbone(cache: h5py.File, index: int) -> np.ndarray:
    value = np.asarray(cache["conditional_mean"][index, 0], dtype=np.float32)
    return value + np.float32(cache["predicted_residual_dc"][index])


def collect_spectrum_rows(
    row: dict[str, Any], domain: str, split: str
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    features = []
    targets = []
    backbone_sum = np.zeros(8, dtype=np.float64)
    residual_sum = np.zeros(8, dtype=np.float64)
    cross_sum = np.zeros(8, dtype=np.float64)
    objects = int(row[f"{split}_objects"])
    data, cache = _open_split(row, split)
    try:
        for index in range(objects):
            count = np.asarray(data["input"][index, 0], dtype=np.float32)
            velocity = np.asarray(data["input"][index, 1], dtype=np.float32)
            backbone = _backbone(cache, index)
            truth = np.asarray(data["target"][index, 0], dtype=np.float32)
            residual = truth - backbone
            transforms, power = transforms_and_spectra(
                count, velocity, backbone, residual,
                voxel_mpc_h=float(data.attrs["voxel_mpc_h"]),
            )
            count_power, velocity_power, backbone_power, residual_power = power
            cross = band_cross(
                transforms[2], transforms[3], float(data.attrs["voxel_mpc_h"])
            )
            observable = np.asarray(cache["observable_context_features"][index], dtype=np.float64)
            feature = np.concatenate(
                (
                    observable,
                    np.log(np.maximum(backbone_power, np.finfo(np.float64).tiny)),
                    np.log(np.maximum(count_power, np.finfo(np.float64).tiny)),
                    np.log(np.maximum(velocity_power, np.finfo(np.float64).tiny)),
                )
            )
            features.append(feature)
            targets.append(np.log(np.maximum(residual_power, np.finfo(np.float64).tiny)))
            backbone_sum += backbone_power
            residual_sum += residual_power
            cross_sum += cross
            if (index + 1) % 64 == 0 or index + 1 == objects:
                print(
                    f"[v35-spectrum] {domain} {split} {index + 1}/{objects}",
                    flush=True,
                )
    finally:
        data.close()
        cache.close()
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        decomposition_summary(backbone_sum, residual_sum, cross_sum),
    )


def _regression_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    error = prediction - truth
    return {
        "objects": int(len(truth)),
        "rmse_natural_log_power": float(np.sqrt(np.mean(np.square(error)))),
        "bias_natural_log_power": float(error.mean()),
        "pearson_prediction_truth": float(
            np.corrcoef(prediction, truth)[0, 1]
            if prediction.std() > 0 and truth.std() > 0
            else 0.0
        ),
    }


def spectrum_audit(program: dict[str, Any]) -> dict[str, Any]:
    train_x: dict[str, np.ndarray] = {}
    train_y: dict[str, np.ndarray] = {}
    validation_x: dict[str, np.ndarray] = {}
    validation_y: dict[str, np.ndarray] = {}
    decomposition: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        row = program["development_domains"][domain]
        train_x[domain], train_y[domain], train_decomposition = collect_spectrum_rows(
            row, domain, "train"
        )
        validation_x[domain], validation_y[domain], validation_decomposition = collect_spectrum_rows(
            row, domain, "validation"
        )
        decomposition[domain] = {
            "train": train_decomposition,
            "validation": validation_decomposition,
        }
    x = np.concatenate([train_x[domain] for domain in DOMAIN_ORDER])
    y = np.concatenate([train_y[domain] for domain in DOMAIN_ORDER])
    weights = np.concatenate(
        [
            np.full(len(train_x[domain]), 1.0 / (len(DOMAIN_ORDER) * len(train_x[domain])))
            for domain in DOMAIN_ORDER
        ]
    )
    constant = np.mean([train_y[domain].mean(axis=0) for domain in DOMAIN_ORDER], axis=0)
    learner = program["conditional_spectrum_audit"]["learner"]
    bands = []
    supported_bands = []
    for band in range(8):
        model = HistGradientBoostingRegressor(
            loss=learner["loss"],
            learning_rate=float(learner["learning_rate"]),
            max_iter=int(learner["max_iter"]),
            max_leaf_nodes=int(learner["max_leaf_nodes"]),
            min_samples_leaf=int(learner["min_samples_leaf"]),
            l2_regularization=float(learner["l2_regularization"]),
            early_stopping=bool(learner["early_stopping"]),
            random_state=int(learner["random_state"]),
        )
        print(f"[v35-spectrum] fit band={band} rows={len(x)}", flush=True)
        model.fit(x, y[:, band], sample_weight=weights)
        validation = {}
        ratios = []
        for domain in DOMAIN_ORDER:
            truth = validation_y[domain][:, band]
            nonlinear = _regression_metrics(model.predict(validation_x[domain]), truth)
            reference = _regression_metrics(np.full(len(truth), constant[band]), truth)
            ratio = float(
                nonlinear["rmse_natural_log_power"]
                / reference["rmse_natural_log_power"]
            )
            nonlinear["rmse_over_constant"] = ratio
            ratios.append(ratio)
            validation[domain] = {"nonlinear": nonlinear, "constant": reference}
        if all(value <= 0.90 for value in ratios):
            supported_bands.append(band)
        bands.append(
            {
                "band": band,
                "edges_h_mpc": [
                    float(EDGES[band]),
                    "inf" if np.isinf(EDGES[band + 1]) else float(EDGES[band + 1]),
                ],
                "iterations": int(model.n_iter_),
                "constant_equal_source_train_mean_log_power": float(constant[band]),
                "validation": validation,
            }
        )
    return {
        "feature_count": int(x.shape[1]),
        "train_objects": {domain: int(len(train_x[domain])) for domain in DOMAIN_ORDER},
        "bands": bands,
        "supported_bands": supported_bands,
        "supported": len(supported_bands) >= 2,
        "fourier_decomposition": decomposition,
    }


def _phase_accumulator() -> dict[str, np.ndarray]:
    return {
        "candidate_backbone_power": np.zeros(8),
        "candidate_residual_power": np.zeros(8),
        "candidate_cross": np.zeros(8),
        "reference_backbone_power": np.zeros(8),
        "reference_residual_power": np.zeros(8),
        "reference_cross": np.zeros(8),
    }


def _add_phase_fields(
    accumulator: dict[str, np.ndarray],
    backbone: np.ndarray,
    candidate_residual: np.ndarray,
    reference_residual: np.ndarray,
    voxel_mpc_h: float,
) -> None:
    transforms, power = transforms_and_spectra(
        backbone, candidate_residual, reference_residual, voxel_mpc_h=voxel_mpc_h
    )
    backbone_power, candidate_power, reference_power = power
    accumulator["candidate_backbone_power"] += backbone_power
    accumulator["candidate_residual_power"] += candidate_power
    accumulator["candidate_cross"] += band_cross(transforms[0], transforms[1], voxel_mpc_h)
    accumulator["reference_backbone_power"] += backbone_power
    accumulator["reference_residual_power"] += reference_power
    accumulator["reference_cross"] += band_cross(transforms[0], transforms[2], voxel_mpc_h)


def _phase_result(accumulator: dict[str, np.ndarray]) -> dict[str, Any]:
    candidate = decomposition_summary(
        accumulator["candidate_backbone_power"],
        accumulator["candidate_residual_power"],
        accumulator["candidate_cross"],
    )
    reference = decomposition_summary(
        accumulator["reference_backbone_power"],
        accumulator["reference_residual_power"],
        accumulator["reference_cross"],
    )
    candidate_total = np.asarray(candidate["total_power"])
    reference_total = np.asarray(reference["total_power"])
    return {
        "candidate": candidate,
        "reference": reference,
        "candidate_over_reference_total_power": (candidate_total / reference_total).tolist(),
        "absolute_log10_total_power_error": np.abs(
            np.log10(candidate_total / reference_total)
        ).tolist(),
    }


def phase_domain(
    row: dict[str, Any], domain: str, model: dict[str, Any]
) -> dict[str, Any]:
    with h5py.File(row["phase_object_selection"], "r") as selection:
        indices = np.asarray(selection["source_index"], dtype=np.int64)
    if indices.shape != (16,) or len(np.unique(indices)) != 16:
        raise ValueError("V35 phase source indices differ")
    endpoint_voxels = 0
    total_voxels = 0
    maximum_control_error_y = 0.0
    maximum_control_dc = 0.0
    control_accumulator = _phase_accumulator()
    shift_accumulators = {scale: _phase_accumulator() for scale in TRANSLATIONS}
    reference_tail = []
    control_tail = []
    data, cache = _open_split(row, "validation")
    try:
        voxel = float(data.attrs["voxel_mpc_h"])
        for position, index in enumerate(indices):
            backbone = _backbone(cache, int(index)).astype(np.float64)
            truth = np.asarray(data["target"][index, 0], dtype=np.float64)
            residual = truth - backbone
            reference_residual = residual - residual.mean(dtype=np.float64)
            uniform = conditional_forward(residual, backbone, model)
            endpoint_voxels += int(np.count_nonzero((uniform == 0.0) | (uniform == 1.0)))
            total_voxels += uniform.size
            control_residual = conditional_inverse(uniform, backbone, model).astype(np.float64)
            control_residual -= control_residual.mean(dtype=np.float64)
            maximum_control_error_y = max(
                maximum_control_error_y,
                float(np.max(np.abs(control_residual - reference_residual))),
            )
            maximum_control_dc = max(
                maximum_control_dc, abs(float(control_residual.mean(dtype=np.float64)))
            )
            _add_phase_fields(
                control_accumulator,
                backbone,
                control_residual,
                reference_residual,
                voxel,
            )
            reference_tail.append((backbone + reference_residual).astype(np.float32))
            control_tail.append((backbone + control_residual).astype(np.float32))
            for scale in TRANSLATIONS:
                for direction in DIRECTIONS:
                    shift = tuple(scale * value for value in direction)
                    translated_uniform = np.roll(uniform, shift=shift, axis=(0, 1, 2))
                    candidate_residual = conditional_inverse(
                        translated_uniform, backbone, model
                    ).astype(np.float64)
                    candidate_residual -= candidate_residual.mean(dtype=np.float64)
                    _add_phase_fields(
                        shift_accumulators[scale],
                        backbone,
                        candidate_residual,
                        reference_residual,
                        voxel,
                    )
            print(f"[v35-phase] {domain} {position + 1}/16", flush=True)
    finally:
        data.close()
        cache.close()
    return {
        "source_indices": indices.tolist(),
        "endpoint_voxels": endpoint_voxels,
        "total_voxels": total_voxels,
        "endpoint_fraction": endpoint_voxels / total_voxels,
        "maximum_coherent_control_reconstruction_error_y": maximum_control_error_y,
        "maximum_coherent_control_reconstruction_error_log10rho_dex": 4.5
        * maximum_control_error_y,
        "maximum_coherent_control_residual_dc": maximum_control_dc,
        "coherent_control": _phase_result(control_accumulator),
        "coherent_control_tail": tail_diagnostics(
            np.asarray(reference_tail), np.asarray(control_tail)
        ),
        "translations": {
            str(scale): {
                "native_cells": scale,
                "mpc_h": scale * 0.3125,
                **_phase_result(shift_accumulators[scale]),
            }
            for scale in TRANSLATIONS
        },
    }


def phase_audit(program: dict[str, Any]) -> dict[str, Any]:
    copula = program["v31_conditional_copula"]
    model = load_model(Path(copula["artifact"]), copula["artifact_sha256"])
    domains = {
        domain: phase_domain(program["development_domains"][domain], domain, model)
        for domain in DOMAIN_ORDER
    }
    support_material = any(
        row["endpoint_fraction"] >= 1.0e-5
        or row["maximum_coherent_control_reconstruction_error_log10rho_dex"] > 0.1
        for row in domains.values()
    )
    common_evidence = []
    for scale in (8, 16):
        for band in range(6):
            cross_material = all(
                abs(
                    domains[domain]["translations"][str(scale)]["reference"][
                        "backbone_residual_cross_correlation"
                    ][band]
                )
                >= 0.05
                for domain in DOMAIN_ORDER
            )
            error_material = all(
                domains[domain]["translations"][str(scale)][
                    "absolute_log10_total_power_error"
                ][band]
                >= 0.02
                for domain in DOMAIN_ORDER
            )
            if cross_material and error_material:
                common_evidence.append({"translation_native_cells": scale, "band": band})
    return {
        "domains": domains,
        "conditional_support_material": support_material,
        "phase_coupling_common_evidence": common_evidence,
        "phase_coupling_material": bool(common_evidence),
    }


def evaluate(program_path: Path, repo: Path) -> dict[str, Any]:
    program, _ = load_program(program_path, repo)
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V35 audit requires a clean committed worktree")
    spectrum = spectrum_audit(program)
    phase = phase_audit(program)
    support_material = bool(phase["conditional_support_material"])
    spectrum_supported = bool(spectrum["supported"])
    phase_supported = bool(phase["phase_coupling_material"])
    mechanisms = []
    repairs = []
    if support_material:
        mechanisms.append("finite_conditional_copula_support_is_material")
        repairs.append("replace_endpoint_saturation_with_train_only_conditional_tail_likelihood")
    if spectrum_supported:
        mechanisms.append("target_free_context_predicts_residual_spectrum")
        repairs.append("condition_residual_band_amplitudes_on_target_free_context")
    if phase_supported:
        mechanisms.append("translated_donor_innovation_breaks_query_aligned_phase_coupling")
        repairs.append("learn_query_aligned_joint_backbone_residual_cross_covariance")
    if not mechanisms:
        classification = "v35_registered_spectrum_phase_and_support_mechanisms_not_material"
        next_step = "audit_downstream_two_point_estimator_and_finite_ensemble_gate_implementation"
    else:
        classification = "v35_material_mechanisms:" + "+".join(mechanisms)
        next_step = "freeze_joint_conditional_residual_model_with_repairs:" + "+".join(repairs)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_development_only_mechanism_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "sklearn_version": sklearn.__version__,
        "fourier_edges_h_mpc": [
            "inf" if np.isinf(value) else float(value) for value in EDGES
        ],
        "conditional_spectrum": spectrum,
        "conditional_copula_support_and_phase": phase,
        "material_mechanisms": mechanisms,
        "required_repairs": repairs,
        "classification": classification,
        "next": next_step,
        "posthoc_Ak_used": False,
        "validation_used_for_spectrum_fit_or_early_stopping": False,
        "phase_translations_used_as_generated_candidates": False,
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
        raise FileExistsError("V35 refuses to overwrite its audit")
    report = evaluate(args.program.resolve(), args.repo.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
