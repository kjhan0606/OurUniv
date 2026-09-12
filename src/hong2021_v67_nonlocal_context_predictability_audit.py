#!/usr/bin/env python
"""Train-only target-free nonlocal-context predictability audit for V67."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import sklearn
from sklearn.linear_model import Ridge

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import (
    EDGES,
    PROGRAM_SCHEMA as V35_PROGRAM_SCHEMA,
    _backbone,
    _open_split,
    transforms_and_spectra,
)
from hong2021_v63_preflight import _path
from hong2021_v63_train import _is_ancestor


PROGRAM_SHA256 = "e200bb88c3e820c0350067db926f724b28b3cdef22c58bf6dd3ad07e6070933a"
PROGRAM_SCHEMA = "hong2021-v67-train-only-target-free-nonlocal-context-predictability-audit-program-v1"
SCHEMA = "hong2021-v67-train-only-target-free-nonlocal-context-predictability-audit-v1"
PROGRAM_FREEZE_COMMIT = "93d38878c654a92dbeb6a07db7a37086075efdfc"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V67 {label} hash differs")
    return _json(path)


def load_program(
    path: Path, repo: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    repo = repo.resolve()
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_audit_implementation_or_execution"
    ):
        raise ValueError("V67 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        _path(repo, parent["v66_result_record"]),
        parent["v66_result_record_sha256"],
        "V66 result record",
    )
    routing = record.get("conditional_gradient_routing", {})
    firewall = record.get("firewall", {})
    if (
        record.get("status") != parent["required_status"]
        or record.get("audit", {}).get("classification")
        != parent["required_classification"]
        or record.get("audit", {}).get("candidate_selected")
        is not parent["required_candidate_selected"]
        or routing.get("conditional_routing_supported")
        is not parent["required_conditional_routing_supported"]
        or routing.get("first_singular_direction_squared_norm_fraction", 0.0)
        < parent["required_first_singular_direction_squared_norm_fraction_minimum"]
        or firewall.get("training_or_refit_performed")
        is not parent["required_training_or_refit_performed"]
        or firewall.get("new_development_accessed")
        is not parent["required_new_development_accessed"]
        or firewall.get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
    ):
        raise ValueError("V67 parent result or firewall differs")
    frozen = program["frozen_inputs"]
    for version in ("v65", "v66"):
        audit = _verified_json(
            _path(repo, frozen[f"{version}_audit"]),
            frozen[f"{version}_audit_sha256"],
            f"{version.upper()} audit",
        )
        if (
            audit.get("decision_digest_sha256")
            != frozen[f"{version}_audit_decision_digest_sha256"]
            or canonical_digest(audit)
            != frozen[f"{version}_audit_decision_digest_sha256"]
            or audit.get("training_or_refit_performed") is not False
            or audit.get("new_development_accessed") is not False
            or audit.get("independent_gate_locked") is not True
        ):
            raise ValueError(f"V67 sealed {version.upper()} audit differs")
        if version == "v65":
            v65_audit = audit
        else:
            v66_audit = audit
    v35 = _verified_json(
        _path(repo, frozen["v35_program"]),
        frozen["v35_program_sha256"],
        "V35 program",
    )
    if (
        v35.get("schema") != V35_PROGRAM_SCHEMA
        or v35.get("status") != "frozen_revision_before_implementation_or_execution"
        or tuple(v35.get("development_domains", {})) != DOMAIN_ORDER
    ):
        raise ValueError("V67 V35 schema or domain order differs")
    for domain in DOMAIN_ORDER:
        row = v35["development_domains"][domain]
        for kind in ("data", "cache"):
            artifact = Path(row[f"train_{kind}"])
            if sha256_file(artifact) != row[f"train_{kind}_sha256"]:
                raise ValueError(f"V67 {domain} train {kind} differs")
    if sha256_file(_path(repo, frozen["conditioning_cache"])) != frozen[
        "conditioning_cache_sha256"
    ]:
        raise ValueError("V67 conditioning cache differs")
    return program, v35, v65_audit, v66_audit


def response_rows(
    program: dict[str, Any], v65_audit: dict[str, Any]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    values = []
    rows = []
    queries = program["immutable_train_queries"]
    for domain in DOMAIN_ORDER:
        sealed = v65_audit["objects"][domain]
        if [int(row["query_object_index"]) for row in sealed] != queries[domain]:
            raise ValueError("V67 immutable V65 response query binding differs")
        for position, row in enumerate(sealed):
            signed = [
                float(item["signed_log_ratio"])
                for item in row["controls"]["source_balanced"]["separations"]
            ]
            response = float(np.mean(signed))
            values.append(response)
            rows.append(
                {
                    "domain": domain,
                    "query_position": position,
                    "query_object_index": int(row["query_object_index"]),
                    "separation_signed_log_ratios": signed,
                    "response_mean_signed_log_ratio": response,
                }
            )
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (48,) or not np.isfinite(result).all():
        raise RuntimeError("V67 response differs")
    return result, rows


def target_free_features(
    program: dict[str, Any],
    v35: dict[str, Any],
    conditioning_cache_path: Path,
) -> tuple[np.ndarray, list[dict[str, Any]], str]:
    queries = program["immutable_train_queries"]
    feature_rows = []
    provenance_rows = []
    hasher = hashlib.sha256()
    with h5py.File(conditioning_cache_path, "r") as prepared:
        if (
            bool(prepared.attrs.get("validation_truth_opened", True))
            or not bool(prepared.attrs.get("complete", False))
        ):
            raise ValueError("V67 conditioning cache provenance differs")
        for domain in DOMAIN_ORDER:
            data, cache = _open_split(v35["development_domains"][domain], "train")
            try:
                if (
                    bool(cache.attrs.get("feature_uses_target", True))
                    or not bool(cache.attrs.get("complete", False))
                    or tuple(cache["observable_context_features"].shape)
                    != (
                        int(v35["development_domains"][domain]["train_objects"]),
                        8,
                    )
                ):
                    raise ValueError("V67 target-free context cache differs")
                voxel = float(data.attrs["voxel_mpc_h"])
                for position, query_index in enumerate(queries[domain]):
                    observable = np.asarray(
                        cache["observable_context_features"][query_index],
                        dtype=np.float64,
                    )
                    count = np.asarray(data["input"][query_index, 0], dtype=np.float64)
                    velocity = np.asarray(
                        data["input"][query_index, 1], dtype=np.float64
                    )
                    backbone = np.asarray(_backbone(cache, query_index), dtype=np.float64)
                    _, power = transforms_and_spectra(
                        backbone, count, velocity, voxel_mpc_h=voxel
                    )
                    log_power = np.log(
                        np.maximum(power, np.finfo(np.float64).tiny)
                    ).reshape(-1)
                    amplitude = float(
                        prepared[f"{domain}/train/object_amplitude"][query_index]
                    )
                    feature = np.concatenate((observable, log_power, [amplitude]))
                    if feature.shape != (33,) or not np.isfinite(feature).all():
                        raise RuntimeError("V67 target-free feature differs")
                    hasher.update(feature.tobytes())
                    feature_rows.append(feature)
                    provenance_rows.append(
                        {
                            "domain": domain,
                            "query_position": position,
                            "query_object_index": int(query_index),
                            "features": feature.tolist(),
                        }
                    )
            finally:
                data.close()
                cache.close()
    result = np.asarray(feature_rows, dtype=np.float64)
    if result.shape != (48, 33):
        raise RuntimeError("V67 feature matrix differs")
    return result, provenance_rows, hasher.hexdigest()


def _pearson(prediction: np.ndarray, truth: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if prediction.std() <= 0.0 or truth.std() <= 0.0:
        return float("nan")
    return float(np.corrcoef(prediction, truth)[0, 1])


def _metrics(
    prediction: np.ndarray, truth: np.ndarray, reference: np.ndarray
) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    error = prediction - truth
    reference_error = reference - truth
    return {
        "objects": int(len(truth)),
        "Ridge_RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "constant_reference_RMSE": float(
            np.sqrt(np.mean(np.square(reference_error)))
        ),
        "Ridge_MAE": float(np.mean(np.abs(error))),
        "constant_reference_MAE": float(np.mean(np.abs(reference_error))),
        "Pearson_prediction_response": _pearson(prediction, truth),
        "zero_threshold_sign_accuracy": float(
            np.mean((prediction >= 0.0) == (truth >= 0.0))
        ),
    }


def leave_one_domain_out(
    features: np.ndarray,
    response: np.ndarray,
    alpha: float,
    *,
    record_fits: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if features.shape != (48, 33) or response.shape != (48,):
        raise ValueError("V67 fixed probe shape differs")
    prediction = np.empty_like(response)
    reference = np.empty_like(response)
    folds = {}
    domain_index = np.repeat(np.arange(3), 16)
    for held_index, held_domain in enumerate(DOMAIN_ORDER):
        train = domain_index != held_index
        held = ~train
        mean = features[train].mean(axis=0)
        std = features[train].std(axis=0)
        std = np.where(std < 1.0e-12, 1.0, std)
        model = Ridge(alpha=alpha, fit_intercept=True, solver="svd")
        model.fit((features[train] - mean) / std, response[train])
        prediction[held] = model.predict((features[held] - mean) / std)
        reference[held] = response[train].mean()
        if record_fits:
            folds[held_domain] = {
                "training_domains": [
                    domain for domain in DOMAIN_ORDER if domain != held_domain
                ],
                "training_objects": int(train.sum()),
                "held_objects": int(held.sum()),
                "feature_mean": mean.tolist(),
                "feature_std": std.tolist(),
                "ridge_intercept": float(model.intercept_),
                "ridge_coefficients": np.asarray(model.coef_).tolist(),
                "constant_reference": float(response[train].mean()),
                "metrics": _metrics(prediction[held], response[held], reference[held]),
            }
    return prediction, reference, folds


def pooled_metrics(
    prediction: np.ndarray, response: np.ndarray, reference: np.ndarray
) -> dict[str, float | int]:
    result = _metrics(prediction, response, reference)
    result["relative_RMSE_improvement_over_constant"] = float(
        (result["constant_reference_RMSE"] - result["Ridge_RMSE"])
        / result["constant_reference_RMSE"]
    )
    return result


def classify(
    integrity_pass: bool, predictive: bool, significant: bool
) -> tuple[str, str, bool]:
    if not integrity_pass:
        return (
            "nonlocal_context_predictability_audit_failed_integrity",
            "stop_before_refit_and_preserve_the_failed_train_only_audit",
            False,
        )
    if predictive and significant:
        return (
            "target_free_global_context_predicts_the_object_specific_pair_correction_cross_domain",
            "freeze_one_nonlocal_context_scalar_head_model_before_refit",
            True,
        )
    return (
        "target_free_global_context_does_not_predict_the_pair_correction_cross_domain",
        "stop_before_refit_and_do_not_add_a_nonlocal_context_head",
        False,
    )


def audit(program_path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program, v35, v65_audit, v66_audit = load_program(program_path, repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, PROGRAM_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V67 audit requires clean Lageunha with frozen ancestry")
    frozen = program["frozen_inputs"]
    response, response_provenance = response_rows(program, v65_audit)
    features, feature_provenance, feature_digest = target_free_features(
        program, v35, _path(repo, frozen["conditioning_cache"])
    )
    alpha = float(program["fixed_probe"]["alpha"])
    prediction, reference, folds = leave_one_domain_out(
        features, response, alpha, record_fits=True
    )
    pooled = pooled_metrics(prediction, response, reference)
    generator = np.random.default_rng(int(program["permutation_control"]["seed"]))
    replicates = int(program["permutation_control"]["replicates"])
    permutation_rows = []
    for replicate in range(replicates):
        permuted = response.copy()
        for domain_index in range(3):
            selected = slice(16 * domain_index, 16 * (domain_index + 1))
            permuted[selected] = generator.permutation(permuted[selected])
        permuted_prediction, permuted_reference, _ = leave_one_domain_out(
            features, permuted, alpha, record_fits=False
        )
        metrics = pooled_metrics(
            permuted_prediction, permuted, permuted_reference
        )
        permutation_rows.append(
            {
                "replicate": replicate,
                "pooled_Pearson": metrics["Pearson_prediction_response"],
                "pooled_relative_RMSE_improvement": metrics[
                    "relative_RMSE_improvement_over_constant"
                ],
            }
        )
    permuted_pearson = np.asarray(
        [row["pooled_Pearson"] for row in permutation_rows], dtype=np.float64
    )
    permuted_improvement = np.asarray(
        [row["pooled_relative_RMSE_improvement"] for row in permutation_rows],
        dtype=np.float64,
    )
    pearson_p = float(
        (1 + np.count_nonzero(permuted_pearson >= pooled["Pearson_prediction_response"]))
        / (replicates + 1)
    )
    improvement_p = float(
        (
            1
            + np.count_nonzero(
                permuted_improvement
                >= pooled["relative_RMSE_improvement_over_constant"]
            )
        )
        / (replicates + 1)
    )
    predictive = bool(
        all(
            folds[domain]["metrics"]["Ridge_RMSE"]
            < folds[domain]["metrics"]["constant_reference_RMSE"]
            and folds[domain]["metrics"]["Pearson_prediction_response"] > 0.0
            and folds[domain]["metrics"]["zero_threshold_sign_accuracy"] >= 0.625
            for domain in DOMAIN_ORDER
        )
        and pooled["Pearson_prediction_response"] >= 0.5
        and pooled["zero_threshold_sign_accuracy"] >= 2.0 / 3.0
    )
    significant = bool(pearson_p <= 0.05 and improvement_p <= 0.05)
    finite_values = np.concatenate(
        (
            features.reshape(-1),
            response,
            prediction,
            reference,
            permuted_pearson,
            permuted_improvement,
            np.asarray([pearson_p, improvement_p]),
        )
    )
    provenance_pass = bool(
        v66_audit.get("candidate_selected") is False
        and v66_audit.get("conditional_routing_supported") is False
        and features.shape == (48, 33)
        and response.shape == (48,)
    )
    numerical_pass = bool(np.isfinite(finite_values).all())
    integrity_pass = provenance_pass and numerical_pass
    classification, next_step, selected = classify(
        integrity_pass, predictive, significant
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_train_only_target_free_nonlocal_context_predictability_audit",
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "host": socket.gethostname(),
        "sklearn_version": sklearn.__version__,
        "immutable_train_queries": {
            domain: program["immutable_train_queries"][domain]
            for domain in DOMAIN_ORDER
        },
        "response_provenance": response_provenance,
        "target_free_feature_provenance": feature_provenance,
        "target_free_feature_stream_sha256": feature_digest,
        "feature_components": int(features.shape[1]),
        "ridge_alpha": alpha,
        "folds": folds,
        "pooled": pooled,
        "permutation_control": {
            "seed": int(program["permutation_control"]["seed"]),
            "replicates": replicates,
            "rows": permutation_rows,
            "pooled_Pearson_one_sided_p_value": pearson_p,
            "pooled_relative_RMSE_improvement_one_sided_p_value": improvement_p,
        },
        "cross_domain_predictive": predictive,
        "permutation_significant": significant,
        "feature_query_response_provenance_pass": provenance_pass,
        "numerical_pass": numerical_pass,
        "candidate_selected": selected,
        "classification": classification,
        "next": next_step,
        "audit_only_fixed_probe_fit_performed": True,
        "probe_artifact_saved_or_reused": False,
        "training_or_refit_performed": False,
        "optimizer_step_performed_on_V63_or_density_model": False,
        "validation_accessed": False,
        "new_development_accessed": False,
        "development_rank_or_selection_accessed": False,
        "simulation_identity_used_as_predictor": False,
        "truth_derived_predictor_used": False,
        "posthoc_feature_alpha_threshold_or_metric_tuning_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("V67 refuses existing audit output")
    result = audit(args.program.resolve(), args.repo.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
