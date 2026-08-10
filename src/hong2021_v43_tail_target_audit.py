#!/usr/bin/env python
"""Frozen V43 audit of V42 tail support and object-target compatibility."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import spearmanr

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v31_copula import load_model
from hong2021_v36_local_tail import v31_quantile_prediction
from hong2021_v42_tail_body import LAMBDA_KNOTS


PROGRAM_SCHEMA = "hong2021-v43-tail-threshold-target-compatibility-audit-program-v1"
PROGRAM_SHA256 = "bd3e300cbf1206051366c2731ede4f658d25aa893f8c250543cb83f728de97a1"
SCHEMA = "hong2021-v43-tail-threshold-target-compatibility-audit-v1"
DOMAIN_ORDER = ("TNG100", "SIMBA", "Swift")
QUANTILES = (0.9, 0.95, 0.975, 0.99, 0.995, 0.999)
OBJECT_RMSE_MAX = 0.30
OBJECT_BIAS_MAX = 0.15
OBJECT_SPEARMAN_MIN = 0.50
BRACKET_FRACTION_MIN = 0.90
RECONSTRUCTION_TOLERANCE_DEX = 2.0e-6


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"{label} hash differs")
    return json.loads(path.read_text())


def load_program(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "V43 program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_audit_implementation_or_execution"
    ):
        raise ValueError("V43 program schema or status differs")
    parent = program["parent_evidence"]
    record = _verified_json(
        (repo / parent["v42_record"]).resolve(),
        parent["v42_record_sha256"],
        "V43 V42 record",
    )
    decision_row = record.get("decision", {})
    if (
        decision_row.get("classification") != parent["required_classification"]
        or decision_row.get("next") != parent["required_next"]
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V43 V42 conclusion or firewall differs")
    decision = _verified_json(
        Path(parent["v42_decision"]), parent["v42_decision_sha256"], "V43 V42 decision"
    )
    if (
        decision.get("decision_digest_sha256")
        != parent["v42_decision_digest_sha256"]
        or decision.get("classification") != parent["required_classification"]
        or decision.get("next") != parent["required_next"]
    ):
        raise ValueError("V43 V42 decision binding differs")
    inherited = program["inherited_inputs"]
    if (
        sha256_file(Path(inherited["conditional_copula_artifact"]))
        != inherited["conditional_copula_artifact_sha256"]
    ):
        raise ValueError("V43 copula hash differs")
    return program, decision


def batch_log10_mean_delta_squared(field: np.ndarray) -> np.ndarray:
    value = np.asarray(field, dtype=np.float64)
    if value.ndim < 2:
        raise ValueError("V43 amplitude batch requires sample and spatial axes")
    density = np.power(10.0, 4.5 * value)
    axes = tuple(range(1, value.ndim))
    moment = np.square(density - 1.0).mean(axis=axes, dtype=np.float64)
    if not np.isfinite(moment).all():
        raise FloatingPointError("V43 density moment is nonfinite")
    return np.log10(np.maximum(moment, np.finfo(np.float64).tiny))


def endpoint_diagnostics(
    backbone: np.ndarray,
    residual: np.ndarray,
    threshold: np.ndarray,
    target: np.ndarray,
    top_counts: tuple[int, int] = (26, 3),
) -> dict[str, np.ndarray]:
    mean = np.asarray(backbone, dtype=np.float64)
    innovation = np.asarray(residual, dtype=np.float64)
    limit = np.asarray(threshold, dtype=np.float64)
    prediction = np.asarray(target, dtype=np.float64).reshape(-1)
    if innovation.ndim != 5 or mean.shape != (1, 64, 64, 64):
        raise ValueError("V43 field shape differs")
    if len(innovation) != len(prediction):
        raise ValueError("V43 target batch differs")
    excess = np.maximum(innovation - limit[None], 0.0)
    body = innovation - excess
    body -= body.mean(axis=(-3, -2, -1), keepdims=True)
    full = innovation - innovation.mean(axis=(-3, -2, -1), keepdims=True)
    zero_amplitude = batch_log10_mean_delta_squared(mean[None] + body)
    unit_amplitude = batch_log10_mean_delta_squared(mean[None] + full)
    lower = np.minimum(zero_amplitude, unit_amplitude)
    upper = np.maximum(zero_amplitude, unit_amplitude)
    bracket = (lower <= prediction) & (prediction <= upper)
    moment_zero = np.power(10.0, zero_amplitude)
    moment_unit = np.power(10.0, unit_amplitude)
    removable = np.divide(
        moment_unit - moment_zero,
        moment_unit,
        out=np.zeros_like(moment_unit),
        where=moment_unit != 0,
    )

    flattened_field = (mean[None] + full).reshape(len(full), -1)
    flattened_mask = (excess > 0).reshape(len(full), -1)
    recalls = []
    for count in top_counts:
        selected = np.argpartition(flattened_field, -count, axis=1)[:, -count:]
        recalls.append(
            np.take_along_axis(flattened_mask, selected, axis=1).mean(axis=1)
        )
    return {
        "zero_amplitude": zero_amplitude,
        "unit_amplitude": unit_amplitude,
        "target_bracketed": bracket,
        "tail_fraction": (excess > 0).mean(axis=(-3, -2, -1)),
        "removable_moment_fraction": removable,
        "top26_recall": recalls[0],
        "top3_recall": recalls[1],
        "body": body,
        "excess": excess,
    }


def classify(object_supported: bool, q999_supported: bool, q99_supported: bool) -> tuple[str, str]:
    if not object_supported:
        return (
            "v41_object_amplitude_target_is_not_cross_domain_supported",
            "audit_train_only_domain_calibration_of_the_object_target_before_any_generator_change",
        )
    if q999_supported:
        return (
            "v42_fixed_lambda_solver_or_implementation_is_inconsistent_with_attainable_support",
            "audit_v42_tail_solver_reconstruction_before_any_model_change",
        )
    if q99_supported:
        return (
            "object_target_is_supported_but_q99_9_tail_support_is_too_narrow",
            "freeze_train_only_continuous_conditional_tail_likelihood_above_q99_without_spatial_rank_rearrangement",
        )
    return (
        "object_target_is_supported_but_transported_body_and_backbone_are_incompatible",
        "retire_block_and_native_rank_transport_and_fit_a_train_only_local_conditional_residual_likelihood",
    )


def _summarize(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _domain_audit(
    candidate_path: Path,
    uncalibrated_path: Path,
    candidate_sha: str,
    uncalibrated_sha: str,
    copula: dict[str, Any],
) -> dict[str, Any]:
    if sha256_file(candidate_path) != candidate_sha or sha256_file(uncalibrated_path) != uncalibrated_sha:
        raise ValueError("V43 ensemble hash differs")
    with h5py.File(candidate_path, "r") as candidate, h5py.File(
        uncalibrated_path, "r"
    ) as uncalibrated:
        reused = (
            "source_index",
            "donor_source",
            "donor_index",
            "donor_isometry",
            "donor_distance",
            "block_permutation",
            "local_permutation",
            "conditional_rank_multiset_sha256",
            "predicted_log10_mean_delta_squared",
            "conditional_mean",
            "truth",
        )
        if any(
            not np.array_equal(candidate[name][:], uncalibrated[name][:])
            for name in reused
        ):
            raise ValueError("V43 candidate and uncalibrated provenance differs")
        if (
            str(candidate.attrs["arm"]) != "within_block_tail_body"
            or str(uncalibrated.attrs["arm"]) != "tail_calibration_disabled_control"
            or float(candidate.attrs["global_residual_scale"]) != 1.0
            or float(uncalibrated.attrs["global_residual_scale"]) != 1.0
            or not bool(candidate.attrs["complete"])
            or not bool(uncalibrated.attrs["complete"])
        ):
            raise ValueError("V43 ensemble metadata differs")
        candidate_lambda = np.asarray(candidate["tail_lambda"], dtype=np.float64)
        if not np.allclose(candidate_lambda, LAMBDA_KNOTS[0]):
            raise ValueError("V43 candidate lambda is not at its frozen lower boundary")
        backbone = np.asarray(uncalibrated["conditional_mean"], dtype=np.float64)
        truth = np.asarray(uncalibrated["truth"], dtype=np.float64)
        target_members = np.asarray(
            uncalibrated["predicted_log10_mean_delta_squared"], dtype=np.float64
        )
        if np.max(np.ptp(target_members, axis=1)) > 1.0e-7:
            raise ValueError("V43 object target differs across donor members")
        prediction = target_members[:, 0]
        truth_amplitude = batch_log10_mean_delta_squared(truth)
        difference = prediction - truth_amplitude
        correlation = float(spearmanr(prediction, truth_amplitude).statistic)
        object_metrics = {
            "objects": 16,
            "prediction_log10_mean_delta_squared": prediction.tolist(),
            "truth_log10_mean_delta_squared": truth_amplitude.tolist(),
            "mean_bias_dex": float(difference.mean()),
            "MAE_dex": float(np.abs(difference).mean()),
            "RMSE_dex": float(np.sqrt(np.square(difference).mean())),
            "Spearman": correlation,
        }
        object_metrics["pass"] = bool(
            object_metrics["RMSE_dex"] <= OBJECT_RMSE_MAX
            and abs(object_metrics["mean_bias_dex"]) <= OBJECT_BIAS_MAX
            and correlation >= OBJECT_SPEARMAN_MIN
        )

        accumulators = {
            quantile: {
                key: []
                for key in (
                    "zero_amplitude",
                    "unit_amplitude",
                    "target",
                    "target_bracketed",
                    "tail_fraction",
                    "removable_moment_fraction",
                    "top26_recall",
                    "top3_recall",
                    "zero_minus_target",
                )
            }
            for quantile in QUANTILES
        }
        maximum_reconstruction_error = 0.0
        maximum_residual_dc = 0.0
        for object_index in range(16):
            residual = np.asarray(
                uncalibrated["sample"][object_index], dtype=np.float64
            ) - backbone[object_index][None]
            maximum_residual_dc = max(
                maximum_residual_dc,
                float(np.max(np.abs(residual.mean(axis=(-3, -2, -1))))),
            )
            targets = target_members[object_index]
            for quantile in QUANTILES:
                threshold = v31_quantile_prediction(
                    backbone[object_index], quantile, copula
                )
                diagnostics = endpoint_diagnostics(
                    backbone[object_index], residual, threshold, targets
                )
                accumulator = accumulators[quantile]
                for key in (
                    "zero_amplitude",
                    "unit_amplitude",
                    "target_bracketed",
                    "tail_fraction",
                    "removable_moment_fraction",
                    "top26_recall",
                    "top3_recall",
                ):
                    accumulator[key].extend(np.asarray(diagnostics[key]).tolist())
                accumulator["target"].extend(targets.tolist())
                accumulator["zero_minus_target"].extend(
                    (diagnostics["zero_amplitude"] - targets).tolist()
                )
                if quantile == 0.999:
                    raw = diagnostics["body"] + LAMBDA_KNOTS[0] * diagnostics["excess"]
                    raw -= raw.mean(axis=(-3, -2, -1), keepdims=True)
                    reconstructed = batch_log10_mean_delta_squared(
                        backbone[object_index][None] + raw
                    )
                    stored = np.asarray(
                        candidate["achieved_log10_mean_delta_squared"][object_index],
                        dtype=np.float64,
                    )
                    maximum_reconstruction_error = max(
                        maximum_reconstruction_error,
                        float(np.max(np.abs(reconstructed - stored))),
                    )
            print(f"[v43] object {object_index + 1}/16", flush=True)

    thresholds: dict[str, Any] = {}
    for quantile in QUANTILES:
        row = accumulators[quantile]
        bracket = np.asarray(row["target_bracketed"], dtype=bool)
        thresholds[f"q{quantile:g}"] = {
            "member_object_pairs": int(len(bracket)),
            "target_bracket_fraction": float(bracket.mean()),
            "target_bracket_pass": bool(bracket.mean() >= BRACKET_FRACTION_MIN),
            "tail_fraction": _summarize(row["tail_fraction"]),
            "zero_excess_amplitude": _summarize(row["zero_amplitude"]),
            "unit_excess_amplitude": _summarize(row["unit_amplitude"]),
            "target_amplitude": _summarize(row["target"]),
            "zero_excess_minus_target_dex": _summarize(row["zero_minus_target"]),
            "removable_density_moment_fraction": _summarize(
                row["removable_moment_fraction"]
            ),
            "generated_top26_voxel_tail_recall": _summarize(row["top26_recall"]),
            "generated_top3_voxel_tail_recall": _summarize(row["top3_recall"]),
        }
    if maximum_reconstruction_error > RECONSTRUCTION_TOLERANCE_DEX:
        raise ValueError("V43 q99.9 reconstruction differs from V42")
    return {
        "candidate": str(candidate_path.resolve()),
        "candidate_sha256": candidate_sha,
        "uncalibrated": str(uncalibrated_path.resolve()),
        "uncalibrated_sha256": uncalibrated_sha,
        "object_target": object_metrics,
        "conditional_tail_thresholds": thresholds,
        "maximum_q99_9_candidate_reconstruction_error_dex": maximum_reconstruction_error,
        "maximum_uncalibrated_residual_DC": maximum_residual_dc,
    }


def audit(program_path: Path, repo: Path) -> dict[str, Any]:
    program, decision = load_program(program_path, repo)
    copula = load_model(
        Path(program["inherited_inputs"]["conditional_copula_artifact"]),
        program["inherited_inputs"]["conditional_copula_artifact_sha256"],
    )
    domains: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        candidate = decision["arms"]["within_block_tail_body"]["domains"][domain]
        uncalibrated = decision["arms"]["tail_calibration_disabled_control"][
            "domains"
        ][domain]
        print(f"[v43] domain {domain}", flush=True)
        domains[domain] = _domain_audit(
            Path(candidate["ensemble"]),
            Path(uncalibrated["ensemble"]),
            candidate["ensemble_sha256"],
            uncalibrated["ensemble_sha256"],
            copula,
        )
    object_supported = all(row["object_target"]["pass"] for row in domains.values())
    q999_supported = all(
        row["conditional_tail_thresholds"]["q0.999"]["target_bracket_pass"]
        for row in domains.values()
    )
    q99_supported = all(
        row["conditional_tail_thresholds"]["q0.99"]["target_bracket_pass"]
        for row in domains.values()
    )
    classification, next_step = classify(
        object_supported, q999_supported, q99_supported
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "complete_diagnostic_only",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "domains": domains,
        "cross_domain": {
            "object_target_supported_all_domains": object_supported,
            "q99_9_target_bracket_supported_all_domains": q999_supported,
            "q99_target_bracket_supported_all_domains": q99_supported,
        },
        "classification": classification,
        "next": next_step,
        "fit_or_sampling_performed": False,
        "validation_truth_role": "diagnostic_metrics_only_after_v42_was_frozen",
        "new_field_generated": False,
        "donor_reselection": False,
        "posthoc_Ak_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    commit, clean = git_state(args.repo.resolve())
    if not clean:
        raise RuntimeError("V43 audit requires a clean committed worktree")
    result = audit(args.program.resolve(), args.repo.resolve())
    result["audit_code_commit"] = commit
    result["worktree_clean_at_audit"] = clean
    result["decision_digest_sha256"] = canonical_digest(result)
    if args.out.exists():
        raise FileExistsError("V43 refuses existing audit output")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
