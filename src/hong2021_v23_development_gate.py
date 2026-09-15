#!/usr/bin/env python
"""Frozen three-domain gate for the V23 conditional-mean experiment."""
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
from hong2021_v14_edm import V23_E11_SCHEMA
from hong2021_v15_development_gate import (
    DOMAINS,
    _load_metrics,
    _validate_ensemble,
    canonical_digest,
    git_state,
    select_candidate_rows,
)
from hong2021_v18_edm import _indices
from hong2021_v18_init import SCHEMA as INIT_SCHEMA, sha256_file
from hong2021_v20_development_gate import (
    Q5_CHECKS,
    _sampling_commit_is_ancestor,
    _source_indices,
    marginal_diagnostics,
)
from hong2021_v21_development_gate import _remeasure_variance, conditional_diagnostics
from hong2021_v21_edm import ARTIFACT_SHA256, P_MEAN, P_STD
from hong2021_v22_development_gate import latent_conditional_diagnostics
from hong2021_v23_edm import (
    REGISTRY_SHA256,
    _validate_checkpoint,
    load_frozen_program,
)


SCHEMA = "hong2021-v23-conditional-mean-three-domain-decision-v1"
REGISTRY_DOMAINS = {"tng": "TNG100", "simba_dev": "SIMBA", "swift_dev": "Swift"}
Q6_THRESHOLD_KEYS = {
    "tng": "tng100_dev",
    "simba_dev": "simba_dev",
    "swift_dev": "swift_dev",
}
CACHE_KEYS = {
    "TNG100": "TNG100_validation",
    "SIMBA": "SIMBA_validation",
    "Swift": "Swift_validation",
}
CANDIDATES = (10000, 20000, 30000)
TRAINING_DOMAINS = {"TNG100", "SIMBA", "Swift-EAGLE"}


def _validate_ensemble_v23(
    path: Path,
    *,
    artifacts: dict[str, Any],
    v20: dict[str, Any],
    domain: str,
    step: int,
    checkpoint_path: Path,
    checkpoint_sha: str,
    expected_indices: list[int],
    gate_commit: str,
) -> dict[str, Any]:
    experiment = v20["e8_gaussianized_marginal_retrain"]
    source = REGISTRY_DOMAINS[domain]
    seed = int(experiment["sampler"]["sampling_seeds"][source])
    _validate_ensemble(
        path,
        checkpoint=checkpoint_path,
        checkpoint_schema=V23_E11_SCHEMA,
        step=step,
        seed=seed,
    )
    if _source_indices(path) != expected_indices:
        raise ValueError("V23 ensemble source indices differ from frozen subset")
    data = experiment["data"][source]["validation_data"]
    cache = artifacts["caches"][CACHE_KEYS[source]]
    initialization = artifacts["initialization"]
    with h5py.File(path, "r") as handle:
        if tuple(handle["sample"].shape) != (16, 16, 1, 64, 64, 64):
            raise ValueError("V23 ensemble shape differs from 16x16x1x64^3")
        sampling_commit = str(handle.attrs.get("sampling_code_commit", ""))
        if not _sampling_commit_is_ancestor(sampling_commit, gate_commit):
            raise ValueError("V23 sampling commit is not an ancestor of gate commit")
        exact = {
            "checkpoint_sha256": checkpoint_sha,
            "source_cache_sha256": cache["sha256"],
            "source_data_sha256": data["sha256"],
            "init_schema": INIT_SCHEMA,
            "v23_registry_sha256": REGISTRY_SHA256,
            "v21_artifact_attestation_sha256": ARTIFACT_SHA256,
            "v21_profile_sha256": artifacts["profile"]["sha256"],
            "v21_gaussianization_sha256": artifacts["gaussianization"]["sha256"],
            "init_measurement_report_sha256": initialization["measurement_sha256"],
            "init_band_mode_variances_json": json.dumps(
                initialization["source_balanced_band_mode_variance"]
            ),
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
                raise ValueError(f"V23 ensemble metadata differs: {key}")
        effective_sigma = float(
            handle.attrs.get("init_sigma_effective_first_step", -1)
        )
        imaginary = float(
            handle.attrs.get("init_maximum_imaginary_over_real_rms", np.inf)
        )
        if abs(effective_sigma - 40.0) > 1e-4 or imaginary > 1e-12:
            raise ValueError("V23 initialization metadata exceeds frozen bounds")
    return {
        "seed": seed,
        "effective_sigma_first_step": effective_sigma,
        "maximum_imaginary_over_real_rms": imaginary,
        "sampling_code_commit": sampling_commit,
    }


def _mechanism_pass(domains: dict[str, Any]) -> tuple[bool, bool, bool, bool]:
    q3 = all(
        abs(row["mechanism_Q3_Q4"]["delta_q99_999_dex"]) <= 0.10
        and row["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"] <= 0.30
        for row in domains.values()
    )
    q4 = all(
        row["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"] <= 1.5
        for row in domains.values()
    )
    q5 = all(
        all(row["field_gate"]["checks"].get(check, False) for check in Q5_CHECKS)
        for row in domains.values()
    )
    q6_std = all(row["conditional_Q6_std_no_harm"]["pass"] for row in domains.values())
    return q3, q4, q5, q6_std


def _failure_classification(final: dict[str, Any]) -> dict[str, Any]:
    maxima = {
        domain: row["conditional_Q6_latent"][
            "maximum_absolute_generated_minus_truth_mean"
        ]
        for domain, row in final["domains"].items()
    }
    all_controlled = all(value <= 0.05 for value in maxima.values())
    any_uncontrolled = any(value >= 0.10 for value in maxima.values())
    if all_controlled and (not final["Q3_all_domains"] or not final["Q4_all_domains"]):
        classification = "conditional_mean_controlled_but_insufficient"
        next_step = "audit_capacity_or_frozen_second_moment_objective"
    elif any_uncontrolled:
        classification = "conditional_mean_penalty_failed_to_control_mean"
        next_step = "audit_formula_binning_gradients_and_optimization"
    elif not all_controlled:
        classification = "intermediate_conditional_mean_response"
        next_step = "classify_by_frozen_Q3_Q4_without_tuning"
    else:
        classification = "conditional_mean_controlled_but_other_frozen_gate_failed"
        next_step = "audit_field_or_Q6_std_no_harm_failure_without_tuning"
    return {
        "class": classification,
        "maximum_absolute_latent_Q6_mean_error": maxima,
        "Q3_all_domains": final["Q3_all_domains"],
        "Q4_all_domains": final["Q4_all_domains"],
        "Q6_std_no_harm_all_domains": final["Q6_std_no_harm_all_domains"],
        "next": next_step,
    }


def evaluate(
    *, root: Path, training: Path, registry_path: Path, repo: Path,
    gate_commit: str,
) -> dict[str, Any]:
    registry, artifacts, v20, _ = load_frozen_program(registry_path, repo)
    experiment = v20["e8_gaussianized_marginal_retrain"]
    run_path = training / "run.json"
    run = json.loads(run_path.read_text())
    if (
        run.get("status") != "complete"
        or run.get("schema") != V23_E11_SCHEMA
        or run.get("experiment_registry_sha256") != REGISTRY_SHA256
    ):
        raise ValueError("V23 training run status/schema/provenance mismatch")
    if run.get("steps") != 30000 or run.get("candidate_steps") != list(CANDIDATES):
        raise ValueError("V23 run horizon differs from registry")
    if (
        run.get("sigma_data") != artifacts["initialization"]["sigma_data"]
        or run.get("edm_p_mean") != P_MEAN
        or run.get("edm_p_std") != P_STD
    ):
        raise ValueError("V23 run normalization/noise mismatch")
    preflight_path = Path(str(run.get("hard_preflight", "")))
    if not preflight_path.is_file() or sha256_file(preflight_path) != run.get(
        "hard_preflight_sha256"
    ):
        raise ValueError("V23 run hard-preflight seal mismatch")
    preflight = json.loads(preflight_path.read_text())
    if (
        preflight.get("status") != "pass"
        or preflight.get("code_commit") != run.get("code_commit_at_launch")
        or preflight.get("registry_sha256") != REGISTRY_SHA256
        or preflight.get("host") != run.get("execution_host")
        or preflight.get("gpu") != run.get("execution_gpu")
        or str(run.get("execution_host", "")).lower() != "lageunha"
        or "ada" not in str(run.get("execution_gpu", "")).lower()
    ):
        raise ValueError("V23 run execution environment is not sealed")
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    if run["denoising_loss"].get("conditional_mean_edges") != profile["edges"]:
        raise ValueError("V23 run conditional edges differ from V21 profile")
    history = json.loads((training / "history.json").read_text())
    if not history or any(
        row.get("conditional_validation_selection_role") != "none"
        or set(row.get("fixed_conditional_validation", {})) != TRAINING_DOMAINS
        for row in history
    ):
        raise ValueError("V23 fixed conditional validation history is incomplete")
    variance = _remeasure_variance(artifacts)
    transform = json.loads(Path(artifacts["gaussianization"]["path"]).read_text())
    edges = np.asarray(profile["edges"], dtype=np.float64)
    expected_indices = {
        domain: _indices(
            experiment["development_objects"][REGISTRY_DOMAINS[domain]], repo
        )
        for domain in DOMAINS
    }
    thresholds = registry["diagnostics_and_selection"]["Q6_std_no_harm"][
        "thresholds"
    ]
    candidates = []
    for step in CANDIDATES:
        checkpoint_path = (
            training / "validation_checkpoints" / f"step_{step:06d}.pt"
        )
        checkpoint, checkpoint_sha = _validate_checkpoint(
            checkpoint_path, step=step, artifacts=artifacts
        )
        training_commit = str(checkpoint.get("code_commit_at_launch", ""))
        if subprocess.run(
            [
                "git", "-C", str(repo), "merge-base", "--is-ancestor",
                training_commit, gate_commit,
            ],
            capture_output=True,
        ).returncode:
            raise ValueError("V23 training commit is not an ancestor of gate commit")
        domains = {}
        for domain in DOMAINS:
            source = REGISTRY_DOMAINS[domain]
            domain_root = root / f"step_{step:06d}" / domain
            ensemble_path = domain_root / "ensemble16_steps40.h5"
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            initialization = _validate_ensemble_v23(
                ensemble_path,
                artifacts=artifacts,
                v20=v20,
                domain=domain,
                step=step,
                checkpoint_path=checkpoint_path,
                checkpoint_sha=checkpoint_sha,
                expected_indices=expected_indices[domain],
                gate_commit=gate_commit,
            )
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble_path.resolve():
                raise ValueError("V23 metrics refer to another ensemble")
            latent = latent_conditional_diagnostics(
                ensemble_path,
                Path(artifacts["caches"][CACHE_KEYS[source]]["path"]),
                profile,
                transform,
            )
            std_maximum = float(
                np.max(
                    np.abs(
                        np.asarray(latent["generated_over_truth_std"], dtype=np.float64)
                        - 1.0
                    )
                )
            )
            threshold = float(thresholds[Q6_THRESHOLD_KEYS[domain]])
            domains[domain] = {
                "ensemble": str(ensemble_path.resolve()),
                "ensemble_sha256": sha256_file(ensemble_path),
                "metrics": str(metrics_path.resolve()),
                "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics),
                "initialization_metadata": initialization,
                "mechanism_Q3_Q4": marginal_diagnostics(ensemble_path),
                "conditional_Q6_residual": conditional_diagnostics(
                    ensemble_path, edges
                ),
                "conditional_Q6_latent": latent,
                "conditional_Q6_std_no_harm": {
                    "selection_role": "blocking",
                    "maximum_absolute_generated_over_truth_std_minus_one": std_maximum,
                    "threshold": threshold,
                    "pass": std_maximum <= threshold,
                },
            }
        q3, q4, q5, q6_std = _mechanism_pass(domains)
        field_pass = all(row["field_gate"]["pass"] for row in domains.values())
        candidates.append(
            {
                "step": step,
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_training_code_commit": training_commit,
                "gradient_diagnostic": checkpoint.get("gradient_diagnostic"),
                "fixed_conditional_validation": checkpoint.get(
                    "fixed_conditional_validation"
                ),
                "domains": domains,
                "Q3_all_domains": q3,
                "Q4_all_domains": q4,
                "Q5_all_domains": q5,
                "Q6_std_no_harm_all_domains": q6_std,
                "all_three_field_pass": field_pass,
                "all_three_pass": field_pass and q3 and q4 and q6_std,
            }
        )
    selected = select_candidate_rows(candidates)
    failure = None if selected is not None else _failure_classification(candidates[-1])
    next_step = (
        "await_user_approval_before_v23_seal_or_astrid"
        if selected is not None
        else failure["next"]
    )
    report = {
        "schema": SCHEMA,
        "experiment": "v23_conditional_mean_penalty",
        "registry": str(registry_path.resolve()),
        "registry_sha256": REGISTRY_SHA256,
        "v21_artifacts_sha256": ARTIFACT_SHA256,
        "training": str(training.resolve()),
        "training_run_sha256": sha256_file(run_path),
        "gate_code_commit": gate_commit,
        "worktree_clean_at_gate": True,
        "Astrid_used": False,
        "EAGLE_RefL0100N1504_used": False,
        "independent_data_paths_accessed_by_gate": False,
        "initialization_variance_remeasurement": variance,
        "predeclared_steps": list(CANDIDATES),
        "selection_rule": registry["diagnostics_and_selection"]["selection"],
        "candidates": candidates,
        "selected_step": None if selected is None else selected["step"],
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "development_pass": selected is not None,
        "failure_classification": failure,
        "next": next_step,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    commit, clean = git_state(args.repo.resolve())
    if not clean:
        raise RuntimeError("V23 gate requires a clean committed worktree")
    report = evaluate(
        root=args.root,
        training=args.training,
        registry_path=args.registry,
        repo=args.repo.resolve(),
        gate_commit=commit,
    )
    if args.out.exists():
        existing = json.loads(args.out.read_text())
        if existing != report:
            raise RuntimeError("existing V23 decision differs from recomputed decision")
        print(json.dumps(existing, indent=2))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
