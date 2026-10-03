#!/usr/bin/env python
"""Frozen three-domain development gate for the V27 parent-aligned flow."""
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
from hong2021_v15_development_gate import (
    DOMAINS as GATE_DOMAINS,
    _load_metrics,
    canonical_digest,
    git_state,
    select_candidate_rows,
)
from hong2021_v18_edm import _indices
from hong2021_v18_init import sha256_file
from hong2021_v20_development_gate import marginal_diagnostics
from hong2021_v21_development_gate import conditional_diagnostics
from hong2021_v22_development_gate import (
    CACHE_KEYS,
    REGISTRY_DOMAINS,
    _candidate_mechanism_pass,
    latent_conditional_diagnostics,
)
from hong2021_v26_development_gate import _comparison, _nll_plateau
from hong2021_v27 import (
    CANDIDATE_STEPS,
    DESIGN_AUDIT_SHA256,
    ENSEMBLE_METHOD,
    HAAR_ARTIFACT_SHA256,
    MODEL_SCHEMA,
    NON_DC_DIMENSIONS,
    PARAMETERS,
    PREFLIGHT_SCHEMA,
    REGISTRY_ATTRIBUTE,
    REGISTRY_SHA256,
    _validate_checkpoint,
    load_frozen_program,
)


SCHEMA = "hong2021-v27-parent-aligned-flow-three-domain-decision-v1"
V26_DECISION = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/"
    "tng100_simba_swift_v26_e14_conditional_haar_flow/development_decision.json"
)


def _validate_ensemble(
    path: Path,
    *,
    domain: str,
    step: int,
    checkpoint_path: Path,
    checkpoint_sha: str,
    expected_indices: list[int],
    artifacts: dict[str, Any],
    v20: dict[str, Any],
    gate_commit: str,
) -> dict[str, Any]:
    source = REGISTRY_DOMAINS[domain]
    experiment = v20["e8_gaussianized_marginal_retrain"]
    expected_seed = int(experiment["sampler"]["sampling_seeds"][source])
    data = experiment["data"][source]["validation_data"]
    cache = artifacts["caches"][CACHE_KEYS[source]]
    with h5py.File(path, "r") as handle:
        if tuple(handle["sample"].shape) != (16, 16, 1, 64, 64, 64):
            raise ValueError("V27 ensemble shape differs from 16x16x1x64^3")
        if [int(value) for value in handle["source_index"][:]] != expected_indices:
            raise ValueError("V27 ensemble source indices differ from frozen subset")
        exact = {
            "schema": "hong2021-v14-multiscale-location-scale-edm-ensemble-v1",
            "method": ENSEMBLE_METHOD,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_step": step,
            "checkpoint_schema": MODEL_SCHEMA,
            "source_cache_sha256": cache["sha256"],
            "source_data_sha256": data["sha256"],
            REGISTRY_ATTRIBUTE: REGISTRY_SHA256,
            "v21_artifact_attestation_sha256": "5622fc5a22b7502eac433f50e6cc2b51e6253c6e89b179335b1f8eef4e6d5852",
            "v21_profile_sha256": artifacts["profile"]["sha256"],
            "v21_gaussianization_sha256": artifacts["gaussianization"]["sha256"],
            "haar_artifact_sha256": HAAR_ARTIFACT_SHA256,
            "ensemble_members": 16,
            "seed": expected_seed,
            "location_scale_uses_target": False,
            "direct_sampling": True,
            "modeled_non_dc_dimensions": NON_DC_DIMENSIONS,
            "worktree_clean_at_sampling": True,
            "Astrid_accessed": False,
            "historical_EAGLE_accessed": False,
            "complete": True,
        }
        for key, expected in exact.items():
            actual = handle.attrs.get(key)
            if isinstance(actual, np.generic):
                actual = actual.item()
            if actual != expected:
                raise ValueError(f"V27 ensemble metadata differs: {key}")
        commit = str(handle.attrs.get("sampling_code_commit", ""))
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, gate_commit],
            capture_output=True,
        ).returncode:
            raise ValueError("V27 sampling commit is not an ancestor of gate commit")
        pre_dc = float(handle.attrs.get("maximum_pre_center_latent_dc", np.inf))
        post_dc = float(handle.attrs.get("maximum_post_center_latent_dc", np.inf))
        if not np.isfinite(pre_dc) or post_dc > 1.0e-7:
            raise ValueError("V27 sample DC audit failed")
    return {
        "seed": expected_seed,
        "maximum_pre_center_latent_dc": pre_dc,
        "maximum_post_center_latent_dc": post_dc,
        "sampling_code_commit": commit,
    }


def _physical_improvement_over_v26(
    candidate: dict[str, Any], v26: dict[str, Any]
) -> dict[str, Any]:
    old = v26["candidates"][-1]
    rows = {}
    for domain, current in candidate["domains"].items():
        prior = old["domains"][domain]
        old_mechanism = prior["mechanism_Q3_Q4"]
        new_mechanism = current["mechanism_Q3_Q4"]
        rows[domain] = {
            "Q3_absolute_q99_999_error_improved": abs(
                new_mechanism["delta_q99_999_dex"]
            )
            < abs(old_mechanism["delta_q99_999_dex"]),
            "Q3_maximum_excess_improved": new_mechanism[
                "generated_max_above_truth_max_dex"
            ]
            < old_mechanism["generated_max_above_truth_max_dex"],
            "Q4_improved": new_mechanism[
                "generated_over_truth_mean_delta_squared"
            ]
            < old_mechanism["generated_over_truth_mean_delta_squared"],
        }
        rows[domain]["all_three_improve"] = all(rows[domain].values())
    return {
        "domains": rows,
        "all_domains_all_three_improve": all(
            row["all_three_improve"] for row in rows.values()
        ),
        "selection_role": "failure_classification_proxy_only",
    }


def evaluate(
    *, root: Path, training: Path, registry_path: Path, repo: Path, gate_commit: str
) -> dict[str, Any]:
    registry, artifacts, v20, _, _ = load_frozen_program(registry_path, repo)
    run_path = training / "run.json"
    run = json.loads(run_path.read_text())
    if (
        run.get("status") != "complete"
        or run.get("schema") != MODEL_SCHEMA
        or run.get("experiment_registry_sha256") != REGISTRY_SHA256
        or run.get("design_audit_sha256") != DESIGN_AUDIT_SHA256
        or run.get("parameters") != PARAMETERS
        or run.get("non_dc_dimensions") != NON_DC_DIMENSIONS
        or run.get("steps") != 30_000
        or run.get("candidate_steps") != list(CANDIDATE_STEPS)
        or run.get("target_or_density_dependent_weights") is not False
    ):
        raise ValueError("V27 training run protocol/provenance mismatch")
    preflight_path = Path(run["hard_preflight"])
    if sha256_file(preflight_path) != run["hard_preflight_sha256"]:
        raise ValueError("V27 hard-preflight hash mismatch")
    preflight = json.loads(preflight_path.read_text())
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("status") != "pass"
        or preflight.get("code_commit") != run.get("code_commit_at_launch")
        or preflight.get("registry_sha256") != REGISTRY_SHA256
        or preflight.get("Astrid_accessed") is not False
        or preflight.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V27 hard-preflight payload mismatch")
    experiment = v20["e8_gaussianized_marginal_retrain"]
    expected_indices = {
        domain: _indices(
            experiment["development_objects"][REGISTRY_DOMAINS[domain]], repo
        )
        for domain in GATE_DOMAINS
    }
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    transform = json.loads(Path(artifacts["gaussianization"]["path"]).read_text())
    edges = np.asarray(profile["edges"], dtype=np.float64)
    v24 = json.loads(Path(
        "/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/"
        "tng100_simba_swift_v24_e12_base48/development_decision.json"
    ).read_text())
    v25 = json.loads(Path(
        "/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/"
        "tng100_simba_swift_v25_e13_unweighted/development_decision.json"
    ).read_text())
    v26 = json.loads(V26_DECISION.read_text())
    candidates = []
    for step in CANDIDATE_STEPS:
        checkpoint_path = training / "validation_checkpoints" / f"step_{step:06d}.pt"
        checkpoint, checkpoint_sha = _validate_checkpoint(
            checkpoint_path, step=step, artifacts=artifacts
        )
        training_commit = str(checkpoint["code_commit_at_launch"])
        if subprocess.run(
            ["git", "-C", str(repo), "merge-base", "--is-ancestor", training_commit, gate_commit],
            capture_output=True,
        ).returncode:
            raise ValueError("V27 training commit is not an ancestor of gate commit")
        domains = {}
        for domain in GATE_DOMAINS:
            source = REGISTRY_DOMAINS[domain]
            domain_root = root / f"step_{step:06d}" / domain
            ensemble_path = domain_root / "ensemble16.h5"
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            initialization = _validate_ensemble(
                ensemble_path,
                domain=domain,
                step=step,
                checkpoint_path=checkpoint_path,
                checkpoint_sha=checkpoint_sha,
                expected_indices=expected_indices[domain],
                artifacts=artifacts,
                v20=v20,
                gate_commit=gate_commit,
            )
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble_path.resolve():
                raise ValueError("V27 metrics refer to another ensemble")
            domains[domain] = {
                "ensemble": str(ensemble_path),
                "ensemble_sha256": sha256_file(ensemble_path),
                "metrics": str(metrics_path),
                "metrics_sha256": sha256_file(metrics_path),
                "field_gate": field_gate(metrics),
                "sampling_diagnostics": initialization,
                "mechanism_Q3_Q4": marginal_diagnostics(ensemble_path),
                "conditional_Q6_residual": conditional_diagnostics(ensemble_path, edges),
                "conditional_Q6_latent": latent_conditional_diagnostics(
                    ensemble_path,
                    Path(artifacts["caches"][CACHE_KEYS[source]]["path"]),
                    profile,
                    transform,
                ),
            }
        q3, q4, q5 = _candidate_mechanism_pass(domains)
        field_pass = all(row["field_gate"]["pass"] for row in domains.values())
        candidate = {
            "step": step,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_training_code_commit": training_commit,
            "balanced_validation_nll": checkpoint["balanced_validation_nll"],
            "fixed_validation": checkpoint["fixed_validation"],
            "gradient_diagnostic": checkpoint["gradient_diagnostic"],
            "domains": domains,
            "Q3_all_domains": q3,
            "Q4_all_domains": q4,
            "Q5_all_domains": q5,
            "all_three_field_pass": field_pass,
            "all_three_pass": field_pass and q3 and q4,
        }
        candidate["comparison_to_v24"] = _comparison(candidate, v24, "v24")
        candidate["comparison_to_v25"] = _comparison(candidate, v25, "v25")
        candidate["comparison_to_v26"] = _comparison(candidate, v26, "v26")
        candidate["physical_tail_improvement_over_v26"] = _physical_improvement_over_v26(
            candidate, v26
        )
        candidates.append(candidate)
    selected = select_candidate_rows(candidates)
    plateau = _nll_plateau(json.loads((training / "history.json").read_text()))
    final = candidates[-1]
    if selected is not None:
        classification = {
            "class": "parent_aligned_condition_interface_sufficient",
            "next": "seal_v27_and_await_explicit_user_approval_before_independent_data",
        }
    elif final["physical_tail_improvement_over_v26"]["all_domains_all_three_improve"]:
        classification = {
            "class": "condition_phase_repair_improves_tails_but_field_morphology_remains_insufficient",
            "next": "run_frozen_v27_latent_audit_then_audit_deterministic_current_density_backbone",
        }
    else:
        classification = {
            "class": "explicit_conditional_haar_flow_insufficient_after_information_preserving_context",
            "next": "run_frozen_v27_latent_audit_then_test_train_only_empirical_joint_residual_control",
        }
    report = {
        "schema": SCHEMA,
        "experiment": "v27_parent_aligned_conditional_haar_spline_flow",
        "registry": str(registry_path),
        "registry_sha256": REGISTRY_SHA256,
        "training": str(training),
        "training_run_sha256": sha256_file(run_path),
        "hard_preflight": {
            "path": str(preflight_path),
            "sha256": sha256_file(preflight_path),
            "payload": preflight,
        },
        "gate_code_commit": gate_commit,
        "worktree_clean_at_gate": True,
        "Astrid_used": False,
        "EAGLE_RefL0100N1504_used": False,
        "independent_data_paths_accessed_by_gate": False,
        "predeclared_steps": list(CANDIDATE_STEPS),
        "selection_rule": registry["diagnostics_and_selection"]["selection"],
        "candidates": candidates,
        "nll_plateau_diagnostic": plateau,
        "selected_step": None if selected is None else selected["step"],
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "development_pass": selected is not None,
        "failure_classification": None if selected is not None else classification,
        "success_classification": classification if selected is not None else None,
        "next": classification["next"],
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
        raise RuntimeError("V27 gate requires a clean committed worktree")
    report = evaluate(
        root=args.root,
        training=args.training,
        registry_path=args.registry,
        repo=args.repo.resolve(),
        gate_commit=commit,
    )
    if args.out.exists():
        if json.loads(args.out.read_text()) != report:
            raise RuntimeError("existing V27 decision differs from recomputed decision")
        print(json.dumps(report, indent=2))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
