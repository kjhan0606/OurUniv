#!/usr/bin/env python
"""V53 sealed V50/V52 domainwise, two-point, and backbone audit."""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from hong2021_evaluate import OpenBoundaryTwoPoint
from hong2021_v6_gate import field_gate
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v20_development_gate import marginal_diagnostics
from hong2021_v28_empirical import DOMAIN_ORDER
from hong2021_v35_spectrum_phase import _open_split
from hong2021_v46_tail_occupancy_audit import (
    EXPECTED_OBJECTS,
    PROBE_VOXELS,
    _probe_indices,
)
from hong2021_v50_network import (
    LOWER_SUPPORT,
    UPPER_SUPPORT,
    LocalMixtureUNet,
    mixture_parameters,
    parameter_count,
)
from hong2021_v51_bounded_support_audit import (
    _physical_delta_squared,
    _quadrature_object,
    _relative_difference,
    _truth_probe,
)
from hong2021_v52_train import (
    CHECKPOINT_SCHEMA,
    PARAMETERS,
    PROGRAM_SHA256 as V52_PROGRAM_SHA256,
    load_cache,
    load_program as load_v52_program,
    no_risk_condition_cube,
)


PROGRAM_SCHEMA = "hong2021-v53-v50-v52-domainwise-extreme-backbone-audit-program-v1"
PROGRAM_SHA256 = "d93dff7a1fff8ad49e9841b62e99d647bf2220b39ea9c165b0168e2ae5cae004"
RESULT_SCHEMA = "hong2021-v53-v50-v52-domainwise-extreme-backbone-audit-v1"
BOOTSTRAP_RESAMPLES = 50_000
BOOTSTRAP_SEED = 153_053
STRATUM_QUANTILES = (0.9, 0.99, 0.999)
STRATUM_LABELS = (
    "below_q90",
    "q90_to_q99",
    "q99_to_q99_9",
    "q99_9_and_above",
)
PRIMARY_QUADRATURE_ORDER = 64
CONTROL_QUADRATURE_ORDER = 32
MOMENT_RATIO_MINIMUM = 2.0 / 3.0
MOMENT_RATIO_MAXIMUM = 1.5
QUADRATURE_RELATIVE_DIFFERENCE_MAXIMUM = 0.005
COMMON_ARRAYS = (
    "truth",
    "conditional_mean",
    "source_index",
    "donor_source",
    "donor_index",
    "donor_isometry",
    "donor_distance",
    "predicted_residual_dc",
    "predicted_band_scales",
    "conditional_rank_multiset_sha256",
    "object_amplitude_prediction",
)


def _verified_json(path: Path, digest: str, label: str) -> dict[str, Any]:
    if sha256_file(path) != digest:
        raise ValueError(f"V53 {label} hash differs")
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def load_program(
    path: Path, repo: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    program = _verified_json(path.resolve(), PROGRAM_SHA256, "program")
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != "frozen_before_audit_implementation_or_execution"
    ):
        raise ValueError("V53 program schema or status differs")
    parent = program["parent_evidence"]
    v52_record = _verified_json(
        _resolve(repo, parent["v52_record"]),
        parent["v52_record_sha256"],
        "V52 result record",
    )
    decision_record = v52_record.get("development_decision", {})
    if (
        v52_record.get("status") != parent["required_status"]
        or decision_record.get("classification") != parent["required_classification"]
        or decision_record.get("next") != parent["required_next"]
        or decision_record.get("development_pass") is not parent["required_development_pass"]
        or v52_record.get("firewall", {}).get("independent_gate_locked")
        is not parent["required_independent_gate_locked"]
        or v52_record.get("firewall", {}).get("Astrid_accessed") is not False
        or v52_record.get("firewall", {}).get("historical_EAGLE_accessed")
        is not False
    ):
        raise ValueError("V53 V52 parent conclusion or firewall differs")

    records: dict[str, dict[str, Any]] = {"v52": v52_record}
    for key in ("v50_record", "v51_record", "v52_program"):
        value = program["frozen_records"][key]
        candidate = _resolve(repo, value)
        digest = program["frozen_records"][f"{key}_sha256"]
        if key.endswith("_record"):
            records[key[:3]] = _verified_json(candidate, digest, key)
        elif sha256_file(candidate) != digest:
            raise ValueError(f"V53 {key} hash differs")

    train = program["frozen_train_inputs"]
    for key in (
        "v51_audit",
        "v52_checkpoint",
        "v52_training_report",
        "v52_preflight",
        "conditioning_cache",
        "support_selection",
    ):
        if sha256_file(Path(train[key])) != train[f"{key}_sha256"]:
            raise ValueError(f"V53 train input differs: {key}")
    v51_audit = _verified_json(
        Path(train["v51_audit"]), train["v51_audit_sha256"], "V51 audit"
    )
    v52_report = _verified_json(
        Path(train["v52_training_report"]),
        train["v52_training_report_sha256"],
        "V52 training report",
    )
    if (
        canonical_digest(v51_audit) != train["v51_audit_decision_digest_sha256"]
        or canonical_digest(v52_report)
        != train["v52_training_report_decision_digest_sha256"]
        or v52_report.get("AMP_overflow_count") != 4
        or v52_report.get("risk_channel_exact_standardized_zero") is not True
        or v52_report.get(
            "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection"
        )
        is not False
        or v52_report.get("Astrid_accessed") is not False
        or v52_report.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V53 V51 audit or V52 report binding differs")

    development = program["frozen_development_inputs"]
    decisions: dict[str, dict[str, Any]] = {}
    for version in ("v50", "v52"):
        key = f"{version}_decision"
        decision = _verified_json(
            Path(development[key]), development[f"{key}_sha256"], key
        )
        if canonical_digest(decision) != development[f"{key}_digest_sha256"]:
            raise ValueError(f"V53 {key} digest differs")
        if (
            decision.get("independent_gate_locked") is not True
            or decision.get("Astrid_accessed") is not False
            or decision.get("historical_EAGLE_accessed") is not False
        ):
            raise ValueError(f"V53 {key} firewall differs")
        decisions[version] = decision
    for domain in DOMAIN_ORDER:
        row = development["domains"][domain]
        for version in ("v50", "v52"):
            for kind in ("ensemble", "metrics"):
                key = f"{version}_{kind}"
                if sha256_file(Path(row[key])) != row[f"{key}_sha256"]:
                    raise ValueError(f"V53 {domain} {key} hash differs")

    _, v35, _ = load_v52_program(
        _resolve(repo, program["frozen_records"]["v52_program"]), repo
    )
    return program, records, v51_audit, v52_report, decisions, v35


def _bootstrap_indices(objects: int, domain_index: int) -> np.ndarray:
    generator = np.random.default_rng(BOOTSTRAP_SEED + 10_000_000 * domain_index)
    return generator.integers(
        0, objects, size=(BOOTSTRAP_RESAMPLES, objects), dtype=np.int16
    )


def _bootstrap_mean_difference(
    first: np.ndarray, second: np.ndarray, indices: np.ndarray
) -> dict[str, Any]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("V53 paired bootstrap vector differs")
    difference = first - second
    sampled = difference[indices].mean(axis=1)
    interval = np.quantile(sampled, [0.025, 0.975])
    return {
        "paired_mean_difference": float(difference.mean()),
        "paired_object_bootstrap_95": interval.tolist(),
        "interval_excludes_zero": bool(interval[0] > 0.0 or interval[1] < 0.0),
    }


def _bootstrap_ratio(
    numerator: np.ndarray, denominator: np.ndarray, indices: np.ndarray
) -> dict[str, Any]:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if (
        numerator.shape != denominator.shape
        or numerator.ndim != 1
        or np.any(numerator < 0.0)
        or np.any(denominator <= 0.0)
    ):
        raise ValueError("V53 paired bootstrap ratio input differs")
    sampled_denominator = denominator[indices].sum(axis=1)
    if np.any(sampled_denominator <= 0.0):
        raise RuntimeError("V53 paired bootstrap denominator differs")
    sampled = numerator[indices].sum(axis=1) / sampled_denominator
    interval = np.quantile(sampled, [0.025, 0.975])
    return {
        "ratio": float(numerator.sum() / denominator.sum()),
        "paired_object_bootstrap_95": interval.tolist(),
        "interval_excludes_one": bool(interval[0] > 1.0 or interval[1] < 1.0),
    }


def _physical_delta_squared_y(value: np.ndarray) -> np.ndarray:
    return _physical_delta_squared(np.asarray(value, dtype=np.float64))


def _stratum_masks(value: np.ndarray, boundaries: np.ndarray) -> tuple[np.ndarray, ...]:
    return (
        value < boundaries[0],
        (value >= boundaries[0]) & (value < boundaries[1]),
        (value >= boundaries[1]) & (value < boundaries[2]),
        value >= boundaries[2],
    )


def _metric_payload(path: Path) -> dict[str, Any]:
    payload = _verified_json(path, sha256_file(path), "already bound metrics")
    candidates = payload.get("candidates", {})
    if tuple(candidates) != ("edm",):
        raise ValueError("V53 metrics candidate differs")
    return candidates["edm"]


def _two_point_summary(
    metrics: dict[str, Any], voxel_mpc_h: float
) -> dict[str, Any]:
    two_point = metrics["two_point_cosmic_mean"]
    generated = two_point["generated_vs_truth_ks"]
    deterministic = two_point["deterministic_vs_truth_ks"]
    generated_radius = np.asarray(generated["ks_at_radius"], dtype=np.float64)
    deterministic_radius = np.asarray(
        deterministic["ks_at_radius"], dtype=np.float64
    )
    estimator = OpenBoundaryTwoPoint(64, voxel_mpc_h, 10.0)
    radius = estimator.radius_mpc_h
    if generated_radius.shape != (32,) or not np.array_equal(
        radius, (np.arange(32, dtype=np.float64) + 0.5) * voxel_mpc_h
    ):
        raise ValueError("V53 two-point radius grid differs")
    radius_margin = deterministic_radius - generated_radius
    scales: dict[str, Any] = {}
    for scale in generated["by_scale"]:
        generated_mean = float(generated["by_scale"][scale]["mean"])
        deterministic_mean = float(deterministic["by_scale"][scale]["mean"])
        scales[scale] = {
            "generated_KS_mean": generated_mean,
            "deterministic_KS_mean": deterministic_mean,
            "deterministic_minus_generated_KS_mean": deterministic_mean
            - generated_mean,
            "passes_strict_improvement": generated_mean < deterministic_mean,
        }
    return {
        "radius_mpc_h": radius.tolist(),
        "deterministic_minus_generated_KS_at_radius": radius_margin.tolist(),
        "failed_radius_mpc_h": radius[radius_margin <= 0.0].tolist(),
        "failed_radius_count": int(np.count_nonzero(radius_margin <= 0.0)),
        "scales": scales,
        "all_scales_strictly_improve": all(
            row["passes_strict_improvement"] for row in scales.values()
        ),
    }


def _object_metrics(samples: np.ndarray) -> dict[str, np.ndarray]:
    if samples.shape != (16, 16, 64, 64, 64):
        raise ValueError("V53 sample array shape differs")
    result = {name: np.empty(16, dtype=np.float64) for name in (
        "q99_99_log10rho",
        "q99_999_log10rho",
        "maximum_log10rho",
        "mean_delta_squared",
    )}
    for object_index in range(16):
        value = samples[object_index].astype(np.float64, copy=False)
        log_density = 4.5 * value
        result["q99_99_log10rho"][object_index] = np.quantile(log_density, 0.9999)
        result["q99_999_log10rho"][object_index] = np.quantile(
            log_density, 0.99999
        )
        result["maximum_log10rho"][object_index] = np.max(log_density)
        result["mean_delta_squared"][object_index] = np.mean(
            _physical_delta_squared_y(value), dtype=np.float64
        )
    return result


def _development_strata(
    truth: np.ndarray,
    backbone: np.ndarray,
    v50: np.ndarray,
    v52: np.ndarray,
    boundaries: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    masks = _stratum_masks(backbone, boundaries)
    truth_total = float(_physical_delta_squared_y(truth).sum(dtype=np.float64))
    model_total = {
        "V50": float(_physical_delta_squared_y(v50).sum(dtype=np.float64)),
        "V52": float(_physical_delta_squared_y(v52).sum(dtype=np.float64)),
    }
    rows: dict[str, Any] = {}
    for label, mask in zip(STRATUM_LABELS, masks, strict=True):
        truth_sums = np.empty(16, dtype=np.float64)
        v50_sums = np.empty(16, dtype=np.float64)
        v52_sums = np.empty(16, dtype=np.float64)
        truth_count = np.empty(16, dtype=np.int64)
        truth_values: list[np.ndarray] = []
        v50_values: list[np.ndarray] = []
        v52_values: list[np.ndarray] = []
        for object_index in range(16):
            selected = mask[object_index].reshape(-1)
            count = int(selected.sum())
            if count <= 0:
                raise RuntimeError("V53 empty development backbone stratum")
            truth_selected = truth[object_index].reshape(-1)[selected]
            v50_selected = v50[object_index].reshape(16, -1)[:, selected]
            v52_selected = v52[object_index].reshape(16, -1)[:, selected]
            truth_sums[object_index] = _physical_delta_squared_y(
                truth_selected
            ).sum(dtype=np.float64)
            v50_sums[object_index] = _physical_delta_squared_y(v50_selected).sum(
                dtype=np.float64
            )
            v52_sums[object_index] = _physical_delta_squared_y(v52_selected).sum(
                dtype=np.float64
            )
            truth_count[object_index] = count
            truth_values.append(truth_selected.astype(np.float32, copy=False))
            v50_values.append(v50_selected.reshape(-1).astype(np.float32, copy=False))
            v52_values.append(v52_selected.reshape(-1).astype(np.float32, copy=False))
        count = int(truth_count.sum())
        truth_sum = float(truth_sums.sum())
        v50_sum = float(v50_sums.sum())
        v52_sum = float(v52_sums.sum())
        v50_over_truth = _bootstrap_ratio(
            v50_sums, 16.0 * truth_sums, bootstrap_indices
        )
        v52_over_truth = _bootstrap_ratio(
            v52_sums, 16.0 * truth_sums, bootstrap_indices
        )
        v52_over_v50 = _bootstrap_ratio(v52_sums, v50_sums, bootstrap_indices)
        rows[label] = {
            "truth_voxels": count,
            "generated_voxels_per_model": 16 * count,
            "truth_mean_delta_squared": truth_sum / count,
            "V50_mean_delta_squared": v50_sum / (16 * count),
            "V52_mean_delta_squared": v52_sum / (16 * count),
            "V50_over_truth_mean_delta_squared": v50_over_truth,
            "V52_over_truth_mean_delta_squared": v52_over_truth,
            "V52_over_V50_mean_delta_squared": v52_over_v50,
            "q99_9_log10rho": {
                "truth": float(
                    4.5 * np.quantile(np.concatenate(truth_values), 0.999)
                ),
                "V50": float(4.5 * np.quantile(np.concatenate(v50_values), 0.999)),
                "V52": float(4.5 * np.quantile(np.concatenate(v52_values), 0.999)),
            },
            "delta_squared_fraction_of_domain_total": {
                "truth": truth_sum / truth_total,
                "V50": v50_sum / model_total["V50"],
                "V52": v52_sum / model_total["V52"],
            },
        }
    return {"boundaries_physical_y": boundaries.tolist(), "strata": rows}


def _assert_close_mechanism(
    actual: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    keys = (
        "truth_q99_999_log10rho",
        "generated_q99_999_log10rho",
        "delta_q99_999_dex",
        "truth_max_log10rho",
        "generated_max_log10rho",
        "generated_max_above_truth_max_dex",
        "truth_mean_delta_squared",
        "generated_mean_delta_squared",
        "generated_over_truth_mean_delta_squared",
    )
    for key in keys:
        if not math.isclose(
            float(actual[key]), float(expected[key]), rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError(f"V53 sealed mechanism differs: {label} {key}")


def _audit_development_domain(
    domain: str,
    domain_index: int,
    specification: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    v50_path = Path(specification["v50_ensemble"])
    v52_path = Path(specification["v52_ensemble"])
    with h5py.File(v50_path, "r") as first, h5py.File(v52_path, "r") as second:
        for name in COMMON_ARRAYS:
            if name not in first or name not in second or not np.array_equal(
                first[name][:], second[name][:]
            ):
                raise ValueError(f"V53 {domain} paired array differs: {name}")
        if tuple(first["sample"].shape) != (16, 16, 1, 64, 64, 64) or tuple(
            second["sample"].shape
        ) != (16, 16, 1, 64, 64, 64):
            raise ValueError(f"V53 {domain} sample shape differs")
        truth = np.asarray(first["truth"][:, 0], dtype=np.float32)
        backbone = np.asarray(first["conditional_mean"][:, 0], dtype=np.float32)
        v50 = np.asarray(first["sample"][:, :, 0], dtype=np.float32)
        v52 = np.asarray(second["sample"][:, :, 0], dtype=np.float32)
    if not all(np.isfinite(value).all() for value in (truth, backbone, v50, v52)):
        raise RuntimeError(f"V53 {domain} development value is nonfinite")
    boundaries = np.quantile(backbone.astype(np.float64), STRATUM_QUANTILES)
    bootstrap_indices = _bootstrap_indices(16, domain_index)
    v50_objects = _object_metrics(v50)
    v52_objects = _object_metrics(v52)
    paired_metrics = {
        key: _bootstrap_mean_difference(
            v52_objects[key], v50_objects[key], bootstrap_indices
        )
        for key in ("q99_99_log10rho", "q99_999_log10rho", "maximum_log10rho")
    }
    paired_metrics["mean_delta_squared_V52_over_V50"] = _bootstrap_ratio(
        v52_objects["mean_delta_squared"],
        v50_objects["mean_delta_squared"],
        bootstrap_indices,
    )

    v50_metrics = _metric_payload(Path(specification["v50_metrics"]))
    v52_metrics = _metric_payload(Path(specification["v52_metrics"]))
    v50_field = field_gate(v50_metrics)
    v52_field = field_gate(v52_metrics)
    expected_v50 = decisions["v50"]["arms"][
        "bounded_query_local_mixture_copula"
    ]["domains"][domain]
    expected_v52 = decisions["v52"]["arms"][
        "no_risk_query_local_mixture_copula"
    ]["domains"][domain]
    v50_marginal = marginal_diagnostics(v50_path)
    v52_marginal = marginal_diagnostics(v52_path)
    _assert_close_mechanism(
        v50_marginal, expected_v50["mechanism_Q3_Q4"], f"{domain} V50"
    )
    _assert_close_mechanism(
        v52_marginal, expected_v52["mechanism_Q3_Q4"], f"{domain} V52"
    )
    if v50_field != expected_v50["field_gate"] or v52_field != expected_v52[
        "field_gate"
    ]:
        raise ValueError(f"V53 {domain} sealed field gate differs")
    return {
        "paired_arrays_exact": True,
        "objects": 16,
        "members_per_object": 16,
        "sealed_extremes": {"V50": v50_marginal, "V52": v52_marginal},
        "paired_object_changes_V52_minus_V50": paired_metrics,
        "backbone": _development_strata(
            truth, backbone, v50, v52, boundaries, bootstrap_indices
        ),
        "field_gate": {"V50": v50_field, "V52": v52_field},
        "two_point": {
            "V50": _two_point_summary(v50_metrics, 0.3125),
            "V52": _two_point_summary(v52_metrics, 0.3125),
        },
        "artifact_sha256": {
            key: specification[f"{key}_sha256"]
            for key in ("v50_ensemble", "v50_metrics", "v52_ensemble", "v52_metrics")
        },
    }


def _load_v52_model(
    program: dict[str, Any], repo: Path, commit: str
) -> tuple[LocalMixtureUNet, dict[str, Any]]:
    frozen = program["frozen_train_inputs"]
    checkpoint = torch.load(
        frozen["v52_checkpoint"], map_location="cpu", weights_only=False
    )
    source_commit = str(checkpoint.get("code_commit"))
    if (
        checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("program_sha256") != V52_PROGRAM_SHA256
        or checkpoint.get("step") != 12_000
        or checkpoint.get("parameters") != PARAMETERS
        or checkpoint.get("conditioning_cache_sha256")
        != frozen["conditioning_cache_sha256"]
        or checkpoint.get("support_selection_sha256")
        != frozen["support_selection_sha256"]
        or checkpoint.get("open_standardized_support")
        != [LOWER_SUPPORT, UPPER_SUPPORT]
        or checkpoint.get("risk_channel_exact_standardized_zero") is not True
        or checkpoint.get("sample_clipping") is not False
        or checkpoint.get("component_scale_cap") is not False
        or checkpoint.get(
            "validation_used_for_training_stopping_checkpoint_or_hyperparameter_selection"
        )
        is not False
        or checkpoint.get("Astrid_accessed") is not False
        or checkpoint.get("historical_EAGLE_accessed") is not False
        or subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, commit],
            cwd=repo,
            capture_output=True,
        ).returncode
    ):
        raise ValueError("V53 V52 checkpoint binding differs")
    model = LocalMixtureUNet()
    if parameter_count(model) != PARAMETERS:
        raise RuntimeError("V53 V52 parameter count differs")
    model.load_state_dict(checkpoint["ema_state_dict"])
    return model, checkpoint


def _train_strata_rows(
    variable: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
    boundaries: np.ndarray,
) -> dict[str, Any]:
    masks = _stratum_masks(variable, boundaries)
    rows: dict[str, Any] = {}
    for label, mask in zip(STRATUM_LABELS, masks, strict=True):
        count = int(mask.sum())
        truth_mean = float(np.mean(truth[mask], dtype=np.float64))
        predicted_mean = float(np.mean(predicted[mask], dtype=np.float64))
        if count <= 0 or truth_mean <= 0.0:
            raise RuntimeError("V53 empty or invalid train backbone stratum")
        rows[label] = {
            "count": count,
            "truth_mean_delta_squared": truth_mean,
            "quadrature_mean_delta_squared": predicted_mean,
            "quadrature_over_truth_mean_delta_squared": predicted_mean / truth_mean,
        }
    return {"boundaries": boundaries.tolist(), "strata": rows}


@torch.inference_mode()
def _audit_train_domain(
    model: LocalMixtureUNet,
    device: torch.device,
    v35: dict[str, Any],
    prepared: h5py.File,
    support: dict[str, Any],
    v51_audit: dict[str, Any],
    domain: str,
    domain_index: int,
) -> dict[str, Any]:
    objects = int(v35["development_domains"][domain]["train_objects"])
    if objects != EXPECTED_OBJECTS[domain]:
        raise RuntimeError("V53 train object count differs")
    truth_probe = _truth_probe(v35, prepared, domain, domain_index)
    target_mean = float(prepared["target_mean"][()])
    target_std = float(prepared["target_std"][()])
    domain_maximum = float(
        support["domains"][domain]["maximum_standardized_residual"]
    )
    nodes64_np, weights64_np = np.polynomial.hermite.hermgauss(
        PRIMARY_QUADRATURE_ORDER
    )
    nodes32_np, weights32_np = np.polynomial.hermite.hermgauss(
        CONTROL_QUADRATURE_ORDER
    )
    nodes64 = torch.from_numpy(nodes64_np).to(device)
    weights64 = torch.from_numpy(weights64_np).to(device)
    nodes32 = torch.from_numpy(nodes32_np).to(device)
    weights32 = torch.from_numpy(weights32_np).to(device)
    primary_parts: list[np.ndarray] = []
    primary_sum = control_sum = 0.0
    data, cache = _open_split(v35["development_domains"][domain], "train")
    try:
        for object_index in range(objects):
            condition, _, backbone = no_risk_condition_cube(
                data, cache, prepared, domain, "train", object_index
            )
            parameter = model(torch.from_numpy(condition[None]).to(device))
            indices = _probe_indices(domain_index, object_index)
            index_tensor = torch.from_numpy(indices).to(device)
            flat = (
                parameter.reshape(1, 15, -1)
                .index_select(2, index_tensor)
                .reshape(1, 15, 1, 1, -1)
            )
            logits, locations, scales = mixture_parameters(flat)
            mixture_weights = torch.softmax(logits, dim=1)[0, :, 0, 0]
            locations = locations[0, :, 0, 0]
            scales = scales[0, :, 0, 0]
            base = torch.from_numpy(
                backbone.reshape(-1)[indices].astype(np.float64) + target_mean
            ).to(device)
            primary = _quadrature_object(
                mixture_weights,
                locations,
                scales,
                base,
                target_std,
                nodes64,
                weights64,
                domain_maximum,
            )
            control = _quadrature_object(
                mixture_weights,
                locations,
                scales,
                base,
                target_std,
                nodes32,
                weights32,
                domain_maximum,
            )
            primary_value = primary["delta_squared"].sum(axis=0)
            control_value = control["delta_squared"].sum(axis=0)
            primary_parts.append(primary_value)
            primary_sum += float(primary_value.sum(dtype=np.float64))
            control_sum += float(control_value.sum(dtype=np.float64))
            if (object_index + 1) % 16 == 0 or object_index + 1 == objects:
                print(
                    f"[v53-audit] train {domain} {object_index + 1}/{objects}",
                    flush=True,
                )
    finally:
        data.close()
        cache.close()
    predicted = np.concatenate(primary_parts)
    truth = _physical_delta_squared_y(truth_probe["physical_y"])
    variable = truth_probe["backbone_base"].astype(np.float64)
    if predicted.shape != truth.shape or predicted.shape != variable.shape:
        raise RuntimeError("V53 train probe alignment differs")
    boundaries = np.quantile(variable, STRATUM_QUANTILES)
    v50 = v51_audit["domains"][domain]["conditional_strata"]["backbone"]
    if not np.allclose(
        boundaries,
        np.asarray(v50["boundaries"], dtype=np.float64),
        rtol=0.0,
        atol=1.0e-7,
    ):
        raise ValueError("V53 V51 backbone boundaries differ")
    relative = _relative_difference(primary_sum, control_sum)
    if relative > QUADRATURE_RELATIVE_DIFFERENCE_MAXIMUM:
        raise RuntimeError("V53 V52 quadrature convergence differs")
    v52 = _train_strata_rows(variable, truth, predicted, boundaries)
    for label in STRATUM_LABELS:
        v52["strata"][label]["V52_over_V50_quadrature_mean_delta_squared"] = (
            v52["strata"][label]["quadrature_mean_delta_squared"]
            / float(v50["strata"][label]["quadrature_mean_delta_squared"])
        )
    return {
        "train_objects": objects,
        "probe_voxels": objects * PROBE_VOXELS,
        "V50_sealed": v50,
        "V52": v52,
        "V52_32_to_64_mean_delta_squared_relative_difference": relative,
        "V52_primary_mean_delta_squared": primary_sum / len(predicted),
        "V52_control_mean_delta_squared": control_sum / len(predicted),
    }


def classify(
    common_high_backbone: bool,
    domain_dependent_risk: bool,
    train_calibrated: bool,
    any_two_point_failure: bool,
    point_direction_matches: bool,
    direction_uncertain: bool,
) -> tuple[str, str]:
    if common_high_backbone:
        return (
            "risk_choice_does_not_remove_common_train_high_backbone_physical_tail_miscalibration",
            "freeze_a_matched_V50_risk_model_with_a_train_only_multi_threshold_Brier_tail_score_added_to_the_unchanged_bounded_NLL",
        )
    if domain_dependent_risk:
        return (
            "structure_risk_has_domain_dependent_high_backbone_utility",
            "freeze_a_matched_V50_model_with_train_only_Bernoulli_structure_risk_dropout_and_all_other_factors_unchanged",
        )
    if train_calibrated and any_two_point_failure:
        return (
            "calibrated_train_marginals_leave_empirical_rank_copula_two_point_failure",
            "audit_only_conditional_rank_tail_dependence_without_changing_the_bounded_likelihood_or_training",
        )
    if point_direction_matches and direction_uncertain:
        return (
            "sixteen_object_development_sample_does_not_resolve_domain_dependent_risk_utility",
            "seal_without_model_selection_and_freeze_a_larger_paired_development_diagnostic_that_cannot_change_the_V52_gate",
        )
    return (
        "V50_V52_mixed_domain_failure_not_explained_by_risk_backbone_or_two_point_audits",
        "stop_new_model_selection_and_reassess_the_observable_information_ceiling",
    )


def _assert_finite_tree(value: Any, path: str = "result") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"V53 nonfinite result at {path}")


def audit(program_path: Path, repo: Path, output: Path) -> dict[str, Any]:
    program, records, v51_audit, _, decisions, v35 = load_program(
        program_path, repo.resolve()
    )
    commit, clean = git_state(repo.resolve())
    if not clean:
        raise RuntimeError("V53 audit requires a clean committed worktree")
    if socket.gethostname().split(".")[0].lower() != "lageunha":
        raise RuntimeError("V53 audit requires Lageunha")
    if not torch.cuda.is_available() or "ada" not in torch.cuda.get_device_name(0).lower():
        raise RuntimeError("V53 audit requires the Lageunha Ada GPU")
    if output.exists():
        raise FileExistsError("V53 refuses existing output")
    development = {
        domain: _audit_development_domain(
            domain,
            domain_index,
            program["frozen_development_inputs"]["domains"][domain],
            decisions,
        )
        for domain_index, domain in enumerate(DOMAIN_ORDER)
    }

    device = torch.device("cuda")
    model, checkpoint = _load_v52_model(program, repo.resolve(), commit)
    model = model.to(device).eval()
    frozen = program["frozen_train_inputs"]
    prepared = load_cache(
        Path(frozen["conditioning_cache"]),
        frozen["conditioning_cache_sha256"],
        str(checkpoint["code_commit"]),
    )
    support = _verified_json(
        Path(frozen["support_selection"]),
        frozen["support_selection_sha256"],
        "support selection",
    )
    train_probe: dict[str, Any] = {}
    try:
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            train_probe[domain] = _audit_train_domain(
                model,
                device,
                v35,
                prepared,
                support,
                v51_audit,
                domain,
                domain_index,
            )
    finally:
        prepared.close()

    top = "q99_9_and_above"
    common_high_backbone = all(
        float(train_probe[domain]["V50_sealed"]["strata"][top][
            "quadrature_over_truth_mean_delta_squared"
        ])
        > MOMENT_RATIO_MAXIMUM
        and float(train_probe[domain]["V52"]["strata"][top][
            "quadrature_over_truth_mean_delta_squared"
        ])
        > MOMENT_RATIO_MAXIMUM
        and float(train_probe[domain][
            "V52_32_to_64_mean_delta_squared_relative_difference"
        ])
        <= QUADRATURE_RELATIVE_DIFFERENCE_MAXIMUM
        for domain in DOMAIN_ORDER
    )
    top_effect = {
        domain: development[domain]["backbone"]["strata"][top][
            "V52_over_V50_mean_delta_squared"
        ]
        for domain in DOMAIN_ORDER
    }
    point_direction_matches = bool(
        float(top_effect["TNG100"]["ratio"]) > 1.0
        and float(top_effect["SIMBA"]["ratio"]) < 1.0
        and float(top_effect["Swift"]["ratio"]) < 1.0
        and decisions["v52"]["V52_vs_V50_extreme_comparison"]["TNG100"][
            "candidate_strictly_improves_all_three"
        ]
        is False
        and all(
            decisions["v52"]["V52_vs_V50_extreme_comparison"][domain][
                "candidate_strictly_improves_all_three"
            ]
            is True
            for domain in ("SIMBA", "Swift")
        )
    )
    intervals = {
        domain: top_effect[domain]["paired_object_bootstrap_95"]
        for domain in DOMAIN_ORDER
    }
    domain_dependent_risk = bool(
        point_direction_matches
        and float(intervals["TNG100"][0]) > 1.0
        and float(intervals["SIMBA"][1]) < 1.0
        and float(intervals["Swift"][1]) < 1.0
    )
    direction_uncertain = bool(
        point_direction_matches
        and any(float(interval[0]) <= 1.0 <= float(interval[1]) for interval in intervals.values())
    )
    train_calibrated = all(
        MOMENT_RATIO_MINIMUM
        <= float(train_probe[domain][model_name]["strata"][top][
            "quadrature_over_truth_mean_delta_squared"
        ])
        <= MOMENT_RATIO_MAXIMUM
        for domain in DOMAIN_ORDER
        for model_name in ("V50_sealed", "V52")
    )
    any_two_point_failure = any(
        not development[domain]["two_point"][model_name][
            "all_scales_strictly_improve"
        ]
        for domain in DOMAIN_ORDER
        for model_name in ("V50", "V52")
    )
    classification, next_step = classify(
        common_high_backbone,
        domain_dependent_risk,
        train_calibrated,
        any_two_point_failure,
        point_direction_matches,
        direction_uncertain,
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete_read_only_paired_development_and_train_probe_audit",
        "program": str(program_path.resolve()),
        "program_sha256": PROGRAM_SHA256,
        "code_commit": commit,
        "worktree_clean": clean,
        "development": development,
        "train_only_high_backbone_probe": train_probe,
        "branch_evidence": {
            "both_V50_and_V52_common_train_high_backbone_miscalibration": common_high_backbone,
            "paired_domain_dependent_structure_risk_utility": domain_dependent_risk,
            "both_train_top_backbone_marginals_calibrated": train_calibrated,
            "any_V50_or_V52_two_point_field_failure": any_two_point_failure,
            "top_backbone_point_direction_matches_sealed_mixed_result": point_direction_matches,
            "top_backbone_direction_unresolved_by_paired_bootstrap": direction_uncertain,
            "top_backbone_V52_over_V50_bootstrap_95": intervals,
        },
        "classification": classification,
        "next": next_step,
        "integrity": {
            "v50_record_sha256": program["frozen_records"]["v50_record_sha256"],
            "v51_record_sha256": program["frozen_records"]["v51_record_sha256"],
            "v52_record_sha256": program["parent_evidence"]["v52_record_sha256"],
            "v51_audit_sha256": frozen["v51_audit_sha256"],
            "v52_checkpoint_sha256": frozen["v52_checkpoint_sha256"],
            "v52_training_report_sha256": frozen["v52_training_report_sha256"],
            "v50_decision_digest_sha256": program["frozen_development_inputs"][
                "v50_decision_digest_sha256"
            ],
            "v52_decision_digest_sha256": program["frozen_development_inputs"][
                "v52_decision_digest_sha256"
            ],
            "paired_common_arrays_exact_all_domains": all(
                row["paired_arrays_exact"] for row in development.values()
            ),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "V52_AMP_overflow_count_acknowledged": 4,
        },
        "training_or_refit_performed": False,
        "new_development_sample_generated": False,
        "support_changed": False,
        "development_threshold_changed": False,
        "posthoc_scale_or_clipping_used": False,
        "posthoc_DC_or_Ak_used": False,
        "validation_used_for_selection": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    _assert_finite_tree(result)
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.program.resolve(), args.repo.resolve(), args.out.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
