#!/usr/bin/env python
"""Frozen three-domain gate for the V22 long-horizon experiment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_residual_v12_gaussianized import gaussianize_numpy
from hong2021_v6_gate import field_gate
from hong2021_v14_edm import V22_E10_SCHEMA
from hong2021_v14_multiscale import standardize_residual
from hong2021_v15_development_gate import (
    DOMAINS, _load_metrics, _validate_ensemble, canonical_digest,
    git_state, select_candidate_rows,
)
from hong2021_v18_edm import _indices
from hong2021_v18_init import SCHEMA as INIT_SCHEMA, sha256_file
from hong2021_v20_development_gate import (
    Q5_CHECKS, _sampling_commit_is_ancestor, _source_indices, marginal_diagnostics,
)
from hong2021_v21_conditional_affine import apply_profile, transform_cube
from hong2021_v21_development_gate import _remeasure_variance, conditional_diagnostics
from hong2021_v21_edm import ARTIFACT_SHA256, P_MEAN, P_STD
from hong2021_v22_edm import (
    REGISTRY_SHA256, _validate_checkpoint, load_frozen_program,
)


SCHEMA = "hong2021-v22-long-horizon-three-domain-decision-v1"
REGISTRY_DOMAINS = {"tng": "TNG100", "simba_dev": "SIMBA", "swift_dev": "Swift"}
CACHE_KEYS = {"TNG100": "TNG100_validation", "SIMBA": "SIMBA_validation", "Swift": "Swift_validation"}
CANDIDATES = (10000, 20000, 30000)


def _validate_ensemble_v22(
    path: Path, *, artifacts: dict[str, Any], v20: dict[str, Any],
    domain: str, step: int, checkpoint_path: Path, checkpoint_sha: str,
    expected_indices: list[int], gate_commit: str,
) -> dict[str, Any]:
    experiment = v20["e8_gaussianized_marginal_retrain"]
    source = REGISTRY_DOMAINS[domain]
    seed = int(experiment["sampler"]["sampling_seeds"][source])
    _validate_ensemble(path, checkpoint=checkpoint_path, checkpoint_schema=V22_E10_SCHEMA, step=step, seed=seed)
    if _source_indices(path) != expected_indices:
        raise ValueError("V22 ensemble source indices differ from frozen subset")
    data = experiment["data"][source]["validation_data"]
    cache = artifacts["caches"][CACHE_KEYS[source]]
    init = artifacts["initialization"]
    with h5py.File(path, "r") as handle:
        if tuple(handle["sample"].shape) != (16, 16, 1, 64, 64, 64):
            raise ValueError("V22 ensemble shape differs from 16x16x1x64^3")
        sampling_commit = str(handle.attrs.get("sampling_code_commit", ""))
        if not _sampling_commit_is_ancestor(sampling_commit, gate_commit):
            raise ValueError("V22 sampling commit is not an ancestor of gate commit")
        exact = {
            "checkpoint_sha256": checkpoint_sha,
            "source_cache_sha256": cache["sha256"], "source_data_sha256": data["sha256"],
            "init_schema": INIT_SCHEMA, "v22_registry_sha256": REGISTRY_SHA256,
            "v21_artifact_attestation_sha256": ARTIFACT_SHA256,
            "v21_profile_sha256": artifacts["profile"]["sha256"],
            "v21_gaussianization_sha256": artifacts["gaussianization"]["sha256"],
            "init_measurement_report_sha256": init["measurement_sha256"],
            "init_band_mode_variances_json": json.dumps(init["source_balanced_band_mode_variance"]),
            "conditional_inverse_additional_rng_draws": 0,
            "training_noise_p_mean": P_MEAN, "training_noise_p_std": P_STD,
            "worktree_clean_at_sampling": True, "sampling_code_commit": sampling_commit,
            "init_rng_pairing_self_check": True, "init_additional_rng_draws": 0,
        }
        for key, expected in exact.items():
            actual = handle.attrs.get(key)
            if isinstance(actual, np.generic): actual = actual.item()
            if actual != expected:
                raise ValueError(f"V22 ensemble metadata differs: {key}")
        effective_sigma = float(handle.attrs.get("init_sigma_effective_first_step", -1))
        imaginary = float(handle.attrs.get("init_maximum_imaginary_over_real_rms", np.inf))
        if abs(effective_sigma - 40.0) > 1e-4 or imaginary > 1e-12:
            raise ValueError("V22 initialization metadata exceeds frozen bounds")
    return {"seed": seed, "effective_sigma_first_step": effective_sigma, "maximum_imaginary_over_real_rms": imaginary, "sampling_code_commit": sampling_commit}


def _moments(sums: np.ndarray, squares: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = sums / counts
    std = np.sqrt(np.maximum(squares / counts - mean**2, 0.0))
    return mean, std


def latent_conditional_diagnostics(
    ensemble_path: Path, source_cache_path: Path,
    profile: dict[str, Any], transform: dict[str, Any],
) -> dict[str, Any]:
    """Compare exact truth-cache and forward-mapped generated latents by m bin."""
    edges = np.asarray(profile["edges"], dtype=np.float64); bins = len(edges) - 1
    accum = {key: [np.zeros(bins), np.zeros(bins), np.zeros(bins, dtype=np.int64)] for key in ("truth", "generated")}
    with h5py.File(ensemble_path, "r") as ensemble, h5py.File(source_cache_path, "r") as cache:
        for object_index, source_index in enumerate(ensemble["source_index"][:]):
            location = float(ensemble["predicted_residual_dc"][object_index])
            scales = np.asarray(ensemble["predicted_band_scales"][object_index], dtype=np.float64)
            mean = np.asarray(ensemble["conditional_mean"][object_index, 0], dtype=np.float64) - location
            assignment = np.clip(np.searchsorted(edges, mean, side="right") - 1, 0, bins - 1)
            truth_latent = np.asarray(cache["standardized_residual"][int(source_index), 0], dtype=np.float64)
            for index in range(bins):
                values = truth_latent[assignment == index]
                accum["truth"][0][index] += values.sum(); accum["truth"][1][index] += np.square(values).sum(); accum["truth"][2][index] += values.size
            for member in range(ensemble["sample"].shape[1]):
                generated_y = np.asarray(ensemble["sample"][object_index, member, 0], dtype=np.float64)
                _, residual = standardize_residual(generated_y - mean, predicted_scales=scales, voxel_mpc_h=0.3125)
                latent, _, _ = transform_cube(residual, mean, profile, transform)
                latent = latent.astype(np.float64)
                for index in range(bins):
                    values = latent[assignment == index]
                    accum["generated"][0][index] += values.sum(); accum["generated"][1][index] += np.square(values).sum(); accum["generated"][2][index] += values.size
    truth_mean, truth_std = _moments(*accum["truth"])
    generated_mean, generated_std = _moments(*accum["generated"])
    delta = generated_mean - truth_mean
    return {
        "selection_role": "none", "edges": edges.tolist(),
        "truth_mean": truth_mean.tolist(), "generated_mean": generated_mean.tolist(),
        "generated_minus_truth_mean": delta.tolist(),
        "maximum_absolute_generated_minus_truth_mean": float(np.max(np.abs(delta))),
        "truth_std": truth_std.tolist(), "generated_std": generated_std.tolist(),
        "generated_over_truth_std": (generated_std / truth_std).tolist(),
    }


def _candidate_mechanism_pass(domains: dict[str, Any]) -> tuple[bool, bool, bool]:
    q3 = all(abs(row["mechanism_Q3_Q4"]["delta_q99_999_dex"]) <= 0.10 and row["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"] <= 0.30 for row in domains.values())
    q4 = all(row["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"] <= 1.5 for row in domains.values())
    q5 = all(all(row["field_gate"]["checks"].get(check, False) for check in Q5_CHECKS) for row in domains.values())
    return q3, q4, q5


def evaluate(*, root: Path, training: Path, registry_path: Path, repo: Path, gate_commit: str) -> dict[str, Any]:
    registry, artifacts, v20, _ = load_frozen_program(registry_path, repo)
    experiment = v20["e8_gaussianized_marginal_retrain"]
    run_path = training / "run.json"; run = json.loads(run_path.read_text())
    if run.get("status") != "complete" or run.get("schema") != V22_E10_SCHEMA or run.get("experiment_registry_sha256") != REGISTRY_SHA256:
        raise ValueError("V22 training run status/schema/provenance mismatch")
    if run.get("steps") != 30000 or run.get("candidate_steps") != list(CANDIDATES):
        raise ValueError("V22 run horizon differs from registry")
    if run.get("sigma_data") != artifacts["initialization"]["sigma_data"] or run.get("edm_p_mean") != P_MEAN or run.get("edm_p_std") != P_STD:
        raise ValueError("V22 run normalization/noise mismatch")
    variance = _remeasure_variance(artifacts)
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    transform = json.loads(Path(artifacts["gaussianization"]["path"]).read_text())
    edges = np.asarray(profile["edges"], dtype=np.float64)
    expected_indices = {domain: _indices(experiment["development_objects"][REGISTRY_DOMAINS[domain]], repo) for domain in DOMAINS}
    candidates = []
    for step in CANDIDATES:
        checkpoint_path = training / "validation_checkpoints" / f"step_{step:06d}.pt"
        checkpoint, checkpoint_sha = _validate_checkpoint(checkpoint_path, step=step, artifacts=artifacts)
        training_commit = str(checkpoint.get("code_commit_at_launch", ""))
        if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", training_commit, gate_commit], capture_output=True).returncode:
            raise ValueError("V22 training commit is not an ancestor of gate commit")
        domains = {}
        for domain in DOMAINS:
            source = REGISTRY_DOMAINS[domain]
            domain_root = root / f"step_{step:06d}" / domain
            ensemble_path = domain_root / "ensemble16_steps40.h5"
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            init = _validate_ensemble_v22(
                ensemble_path, artifacts=artifacts, v20=v20, domain=domain, step=step,
                checkpoint_path=checkpoint_path, checkpoint_sha=checkpoint_sha,
                expected_indices=expected_indices[domain], gate_commit=gate_commit,
            )
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble_path.resolve():
                raise ValueError("V22 metrics refer to another ensemble")
            domains[domain] = {
                "ensemble": str(ensemble_path.resolve()), "ensemble_sha256": sha256_file(ensemble_path),
                "metrics": str(metrics_path.resolve()), "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics), "initialization_metadata": init,
                "mechanism_Q3_Q4": marginal_diagnostics(ensemble_path),
                "conditional_Q6_residual": conditional_diagnostics(ensemble_path, edges),
                "conditional_Q6_latent": latent_conditional_diagnostics(
                    ensemble_path, Path(artifacts["caches"][CACHE_KEYS[source]]["path"]), profile, transform,
                ),
            }
        q3, q4, q5 = _candidate_mechanism_pass(domains)
        field_pass = all(row["field_gate"]["pass"] for row in domains.values())
        candidates.append({
            "step": step, "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_sha,
            "checkpoint_training_code_commit": training_commit, "gradient_diagnostic": checkpoint.get("gradient_diagnostic"),
            "domains": domains, "Q3_all_domains": q3, "Q4_all_domains": q4, "Q5_all_domains": q5,
            "all_three_field_pass": field_pass, "all_three_pass": field_pass and q3 and q4,
        })
    selected = select_candidate_rows(candidates)
    history = json.loads((training / "history.json").read_text())
    validation = {int(row["step"]): float(row["balanced_validation"]) for row in history}
    final_improvement = (validation[25000] - validation[30000]) / validation[25000]
    amp20 = max(row["conditional_Q6_latent"]["maximum_absolute_generated_minus_truth_mean"] for row in candidates[1]["domains"].values())
    amp30 = max(row["conditional_Q6_latent"]["maximum_absolute_generated_minus_truth_mean"] for row in candidates[2]["domains"].values())
    plateau = final_improvement < 0.01 and candidates[2]["domains"]["tng"]["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"] > 1.5 and amp30 >= amp20
    if selected is not None:
        classification = None; next_step = "await_user_approval_before_v22_seal_or_astrid"
    else:
        classification = {
            "class": "conditional_representation_capacity_limitation" if plateau else "long_horizon_failed_no_further_extension",
            "plateau": plateau, "validation_relative_improvement_25000_to_30000": final_improvement,
            "maximum_latent_Q6_mean_error_20000": amp20, "maximum_latent_Q6_mean_error_30000": amp30,
            "next": "stop_unopened_and_audit_conditional_capacity_or_objective",
        }
        next_step = classification["next"]
    report = {
        "schema": SCHEMA, "experiment": "v22_long_horizon_from_scratch",
        "registry": str(registry_path.resolve()), "registry_sha256": REGISTRY_SHA256,
        "v21_artifacts_sha256": ARTIFACT_SHA256, "training": str(training.resolve()),
        "training_run_sha256": sha256_file(run_path), "gate_code_commit": gate_commit,
        "worktree_clean_at_gate": True, "Astrid_used": False,
        "EAGLE_RefL0100N1504_used": False, "independent_data_paths_accessed_by_gate": False,
        "initialization_variance_remeasurement": variance, "predeclared_steps": list(CANDIDATES),
        "selection_rule": registry["diagnostics"]["selection"], "candidates": candidates,
        "selected_step": None if selected is None else selected["step"],
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "development_pass": selected is not None, "plateau_diagnostic": {
            "relative_validation_improvement_25000_to_30000": final_improvement,
            "maximum_latent_Q6_mean_error_20000": amp20, "maximum_latent_Q6_mean_error_30000": amp30,
            "plateau_falsification_triggered": plateau,
        },
        "failure_classification": classification, "next": next_step,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True); parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    commit, clean = git_state(args.repo.resolve())
    if not clean: raise RuntimeError("V22 gate requires a clean committed worktree")
    report = evaluate(root=args.root, training=args.training, registry_path=args.registry, repo=args.repo.resolve(), gate_commit=commit)
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if existing != report: raise RuntimeError("existing V22 decision differs from recomputed decision")
        print(json.dumps(existing, indent=2)); return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n"); os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
