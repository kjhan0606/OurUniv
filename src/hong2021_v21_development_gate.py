#!/usr/bin/env python
"""Integrity-bound three-domain development gate for V21-E9."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_v6_gate import field_gate
from hong2021_v14_edm import V21_E9_SCHEMA
from hong2021_v14_multiscale import standardize_residual
from hong2021_v15_development_gate import (
    DOMAINS, _load_metrics, _validate_ensemble, canonical_digest,
    git_state, select_candidate_rows,
)
from hong2021_v18_edm import _indices
from hong2021_v18_init import (
    SCHEMA as INIT_SCHEMA, measure_band_mode_variances, sha256_file,
)
from hong2021_v20_development_gate import (
    Q5_CHECKS, _sampling_commit_is_ancestor, _source_indices, marginal_diagnostics,
)
from hong2021_v21_edm import (
    ARTIFACT_SHA256, P_MEAN, P_STD, REGISTRY_SHA256,
    _validate_checkpoint, load_frozen_program,
)


SCHEMA = "hong2021-v21-integrity-bound-three-domain-decision-v1"
CACHE_SCHEMA = "hong2021-v21-conditional-affine-standardized-residual-cache-v1"
REGISTRY_DOMAINS = {"tng": "TNG100", "simba_dev": "SIMBA", "swift_dev": "Swift"}
CACHE_KEYS = {
    "TNG100": {"train": "TNG100_train", "validation": "TNG100_validation"},
    "SIMBA": {"train": "SIMBA_train", "validation": "SIMBA_validation"},
    "Swift": {"train": "Swift_train", "validation": "Swift_validation"},
}


def _validate_v21_ensemble(
    path: Path, *, artifacts: dict[str, Any], v20: dict[str, Any],
    domain: str, step: int, checkpoint_path: Path, checkpoint_sha: str,
    expected_indices: list[int], gate_commit: str,
) -> dict[str, Any]:
    experiment = v20["e8_gaussianized_marginal_retrain"]
    registry_domain = REGISTRY_DOMAINS[domain]
    seed = int(experiment["sampler"]["sampling_seeds"][registry_domain])
    _validate_ensemble(
        path, checkpoint=checkpoint_path, checkpoint_schema=V21_E9_SCHEMA,
        step=step, seed=seed,
    )
    if _source_indices(path) != expected_indices:
        raise ValueError("V21 ensemble source indices differ from frozen subset")
    data = experiment["data"][registry_domain]["validation_data"]
    cache = artifacts["caches"][CACHE_KEYS[registry_domain]["validation"]]
    initialization = artifacts["initialization"]
    with h5py.File(path, "r") as handle:
        if tuple(handle["sample"].shape) != (16, 16, 1, 64, 64, 64):
            raise ValueError("V21 development ensemble is not 16x16x1x64^3")
        sampling_commit = str(handle.attrs.get("sampling_code_commit", ""))
        if not _sampling_commit_is_ancestor(sampling_commit, gate_commit):
            raise ValueError("V21 sampling commit is not an ancestor of gate commit")
        exact = {
            "checkpoint_sha256": checkpoint_sha,
            "source_cache_sha256": cache["sha256"],
            "source_data_sha256": data["sha256"],
            "init_schema": INIT_SCHEMA,
            "v21_registry_sha256": REGISTRY_SHA256,
            "v21_artifact_attestation_sha256": ARTIFACT_SHA256,
            "v21_profile_sha256": artifacts["profile"]["sha256"],
            "v21_gaussianization_sha256": artifacts["gaussianization"]["sha256"],
            "init_measurement_report_sha256": initialization["measurement_sha256"],
            "init_band_mode_variances_json": json.dumps(initialization["source_balanced_band_mode_variance"]),
            "conditional_inverse_additional_rng_draws": 0,
            "training_noise_p_mean": P_MEAN,
            "training_noise_p_std": P_STD,
            "worktree_clean_at_sampling": True,
            "sampling_code_commit": sampling_commit,
            "init_rng_pairing_self_check": True,
            "init_additional_rng_draws": 0,
        }
        for key, expected in exact.items():
            actual = handle.attrs.get(key)
            if isinstance(actual, np.generic):
                actual = actual.item()
            if actual != expected:
                raise ValueError(f"V21 ensemble metadata differs: {key}")
        effective_sigma = float(handle.attrs.get("init_sigma_effective_first_step", -1))
        imaginary_ratio = float(handle.attrs.get("init_maximum_imaginary_over_real_rms", np.inf))
        if abs(effective_sigma - 40.0) > 1e-4 or imaginary_ratio > 1e-12:
            raise ValueError("V21 initialization metadata exceeds frozen bounds")
    return {
        "seed": seed, "effective_sigma_first_step": effective_sigma,
        "maximum_imaginary_over_real_rms": imaginary_ratio,
        "sampling_code_commit": sampling_commit,
    }


def _remeasure_variance(artifacts: dict[str, Any]) -> dict[str, Any]:
    caches = artifacts["caches"]
    specifications = {}
    for domain, expected_domain in (("TNG100", "TNG100"), ("SIMBA", "SIMBA"), ("Swift", "Swift-EAGLE")):
        row = caches[CACHE_KEYS[domain]["train"]]
        with h5py.File(row["path"], "r") as handle:
            objects = len(handle["standardized_residual"])
        specifications[domain] = {**row, "objects": objects, "domain_attribute": expected_domain}
    measured = measure_band_mode_variances(
        specifications, parseval_relative_tolerance=1e-12,
        maximum_absolute_ortho_dc=1e-9, allowed_schemas=(CACHE_SCHEMA,),
    )
    expected = np.asarray(artifacts["initialization"]["source_balanced_band_mode_variance"])
    actual = np.asarray(measured["source_balanced"])
    relative = float(np.max(np.abs(actual - expected) / expected))
    if relative > 1e-9:
        raise ValueError("V21 variance remeasurement differs from attestation")
    return {**measured, "maximum_relative_difference_from_attestation": relative}


def conditional_diagnostics(path: Path, edges: np.ndarray) -> dict[str, Any]:
    """Q6 in original V14 standardized-residual units by frozen m bin."""
    bins = len(edges) - 1
    truth_sum = np.zeros(bins); truth_square = np.zeros(bins); truth_count = np.zeros(bins, dtype=np.int64)
    generated_sum = np.zeros(bins); generated_square = np.zeros(bins); generated_count = np.zeros(bins, dtype=np.int64)
    with h5py.File(path, "r") as handle:
        for object_index in range(handle["sample"].shape[0]):
            location = float(handle["predicted_residual_dc"][object_index])
            scales = np.asarray(handle["predicted_band_scales"][object_index], dtype=np.float64)
            mean = np.asarray(handle["conditional_mean"][object_index, 0], dtype=np.float64) - location
            assignment = np.clip(np.searchsorted(edges, mean, side="right") - 1, 0, bins - 1)
            truth_y = np.asarray(handle["truth"][object_index, 0], dtype=np.float64)
            _, truth_r = standardize_residual(truth_y - mean, predicted_scales=scales, voxel_mpc_h=0.3125)
            for index in range(bins):
                selected = assignment == index
                values = truth_r[selected]
                truth_sum[index] += values.sum(); truth_square[index] += np.square(values).sum(); truth_count[index] += values.size
            for member in range(handle["sample"].shape[1]):
                generated_y = np.asarray(handle["sample"][object_index, member, 0], dtype=np.float64)
                _, generated_r = standardize_residual(generated_y - mean, predicted_scales=scales, voxel_mpc_h=0.3125)
                for index in range(bins):
                    selected = assignment == index
                    values = generated_r[selected]
                    generated_sum[index] += values.sum(); generated_square[index] += np.square(values).sum(); generated_count[index] += values.size
    truth_mean = truth_sum / truth_count
    generated_mean = generated_sum / generated_count
    truth_std = np.sqrt(np.maximum(truth_square / truth_count - truth_mean**2, 0.0))
    generated_std = np.sqrt(np.maximum(generated_square / generated_count - generated_mean**2, 0.0))
    return {
        "selection_role": "none", "edges": edges.tolist(),
        "truth_mean": truth_mean.tolist(), "generated_mean": generated_mean.tolist(),
        "generated_minus_truth_mean": (generated_mean - truth_mean).tolist(),
        "truth_std": truth_std.tolist(), "generated_std": generated_std.tolist(),
        "generated_over_truth_std": (generated_std / truth_std).tolist(),
        "truth_voxels": truth_count.tolist(), "generated_voxels": generated_count.tolist(),
    }


def evaluate(*, root: Path, training: Path, registry_path: Path, artifacts_path: Path, repo: Path, gate_commit: str) -> dict[str, Any]:
    registry, artifacts, v20 = load_frozen_program(registry_path, artifacts_path, repo)
    experiment = v20["e8_gaussianized_marginal_retrain"]
    run_path = training / "run.json"
    run = json.loads(run_path.read_text())
    if run.get("status") != "complete" or run.get("schema") != V21_E9_SCHEMA:
        raise ValueError("V21 training run is incomplete or has wrong schema")
    if run.get("experiment_registry_sha256") != REGISTRY_SHA256:
        raise ValueError("V21 training registry provenance mismatch")
    initialization = artifacts["initialization"]
    if run.get("sigma_data") != initialization["sigma_data"] or run.get("edm_p_mean") != P_MEAN or run.get("edm_p_std") != P_STD:
        raise ValueError("V21 training normalization/noise mismatch")
    variance = _remeasure_variance(artifacts)
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    edges = np.asarray(profile["edges"], dtype=np.float64)
    expected_indices = {
        domain: _indices(experiment["development_objects"][REGISTRY_DOMAINS[domain]], repo)
        for domain in DOMAINS
    }
    candidates = []
    for step in (5000, 10000):
        checkpoint_path = training / "validation_checkpoints" / f"step_{step:06d}.pt"
        checkpoint, checkpoint_sha = _validate_checkpoint(checkpoint_path, step, artifacts)
        training_commit = str(checkpoint.get("code_commit_at_launch", ""))
        if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", training_commit, gate_commit], capture_output=True).returncode:
            raise ValueError("V21 training commit is not an ancestor of gate commit")
        domains = {}
        for domain in DOMAINS:
            domain_root = root / f"step_{step:06d}" / domain
            ensemble_path = domain_root / "ensemble16_steps40.h5"
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            init_metadata = _validate_v21_ensemble(
                ensemble_path, artifacts=artifacts, v20=v20, domain=domain, step=step,
                checkpoint_path=checkpoint_path, checkpoint_sha=checkpoint_sha,
                expected_indices=expected_indices[domain], gate_commit=gate_commit,
            )
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble_path.resolve():
                raise ValueError("V21 metrics refer to a different ensemble")
            domains[domain] = {
                "ensemble": str(ensemble_path.resolve()), "ensemble_sha256": sha256_file(ensemble_path),
                "metrics": str(metrics_path.resolve()), "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics), "initialization_metadata": init_metadata,
                "mechanism_Q3_Q4": marginal_diagnostics(ensemble_path) if step == 10000 else None,
                "conditional_Q6": conditional_diagnostics(ensemble_path, edges) if step == 10000 else None,
            }
        candidates.append({
            "step": step, "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha, "checkpoint_training_code_commit": training_commit,
            "gradient_diagnostic": checkpoint.get("gradient_diagnostic"), "domains": domains,
            "all_three_pass": all(row["field_gate"]["pass"] for row in domains.values()),
        })
    selected = select_candidate_rows(candidates)
    final = candidates[-1]
    q3_pass = all(abs(row["mechanism_Q3_Q4"]["delta_q99_999_dex"]) <= 0.10 and row["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"] <= 0.30 for row in final["domains"].values())
    q4_pass = all(row["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"] <= 1.5 for row in final["domains"].values())
    q5_pass = all(all(row["field_gate"]["checks"].get(check, False) for check in Q5_CHECKS) for row in final["domains"].values())
    classification = None if selected is not None else {
        "class": "conditional_affine_spatial_structure_limited" if q3_pass and q4_pass else "conditional_affine_falsified",
        "Q3_all_domains": q3_pass, "Q4_all_domains": q4_pass, "Q5_all_domains": q5_pass,
        "next": "stop_unopened_and_audit_joint_spatial_structure" if q3_pass and q4_pass else "stop_unopened_and_audit_conditional_affine_failure",
    }
    next_step = "await_user_approval_before_v21_seal_or_astrid" if selected is not None else classification["next"]
    report = {
        "schema": SCHEMA, "experiment": "e9_voxel_conditional_affine_retrain",
        "registry": str(registry_path.resolve()), "registry_sha256": REGISTRY_SHA256,
        "artifacts": str(artifacts_path.resolve()), "artifacts_sha256": ARTIFACT_SHA256,
        "training": str(training.resolve()), "training_run_sha256": sha256_file(run_path),
        "gate_code_commit": gate_commit, "worktree_clean_at_gate": True,
        "EAGLE_RefL0100N1504_used": False, "Astrid_used": False,
        "independent_data_paths_accessed_by_gate": False,
        "initialization_variance_remeasurement": variance,
        "predeclared_steps": [5000, 10000], "mechanism_diagnostics_selection_role": "none",
        "candidates": candidates, "selected_step": None if selected is None else selected["step"],
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "development_pass": selected is not None,
        "mechanism_10k": {"Q3_all_domains": q3_pass, "Q4_all_domains": q4_pass, "Q5_all_domains": q5_pass},
        "failure_classification_10k": classification, "next": next_step,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    commit, clean = git_state(args.repo.resolve())
    if not clean:
        raise RuntimeError("V21 gate requires a clean committed worktree")
    report = evaluate(root=args.root, training=args.training, registry_path=args.registry, artifacts_path=args.artifacts, repo=args.repo.resolve(), gate_commit=commit)
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if existing != report:
            raise RuntimeError("existing V21 decision differs from recomputed decision")
        print(json.dumps(existing, indent=2)); return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n"); os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
