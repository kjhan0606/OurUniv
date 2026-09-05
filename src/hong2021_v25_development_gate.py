#!/usr/bin/env python
"""Frozen three-domain gate for the V25 proper-score experiment."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from hong2021_v6_gate import field_gate
from hong2021_v14_edm import V25_E13_SCHEMA
from hong2021_v15_development_gate import (
    DOMAINS,
    _load_metrics,
    canonical_digest,
    git_state,
    select_candidate_rows,
)
from hong2021_v18_edm import _indices
from hong2021_v18_init import sha256_file
from hong2021_v20_development_gate import marginal_diagnostics
from hong2021_v21_development_gate import _remeasure_variance, conditional_diagnostics
from hong2021_v21_edm import ARTIFACT_SHA256, P_MEAN, P_STD
from hong2021_v22_development_gate import (
    CACHE_KEYS,
    REGISTRY_DOMAINS,
    _candidate_mechanism_pass,
    _validate_ensemble_v22,
    latent_conditional_diagnostics,
)
from hong2021_v24_edm import PARAMETERS
from hong2021_v25_edm import (
    REGISTRY_SHA256,
    _validate_checkpoint,
    load_frozen_program,
)


SCHEMA = "hong2021-v25-proper-unweighted-score-three-domain-decision-v1"
CANDIDATES = (10000, 20000, 30000)


def _preflight(run: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(run.get("hard_preflight", "")))
    if not path.is_file() or sha256_file(path) != run.get("hard_preflight_sha256"):
        raise ValueError("V25 run hard-preflight seal mismatch")
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != "hong2021-v25-hard-preflight-v1"
        or payload.get("status") != "pass"
        or payload.get("code_commit") != run.get("code_commit_at_launch")
        or payload.get("registry_sha256") != REGISTRY_SHA256
        or payload.get("host") != run.get("execution_host")
        or payload.get("gpu") != run.get("execution_gpu")
        or payload.get("base_channels") != 48
        or payload.get("parameters") != PARAMETERS
        or payload.get("optimized_loss_coefficients")
        != {"unweighted": 1.0, "tail_weighted": 0.0}
        or str(run.get("execution_host", "")).lower() != "lageunha"
        or "ada" not in str(run.get("execution_gpu", "")).lower()
        or payload.get("Astrid_accessed") is not False
        or payload.get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V25 run execution environment is not sealed")
    return {"path": str(path), "sha256": sha256_file(path), "payload": payload}


def _comparison(
    candidate: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    prior = {row["step"]: row for row in baseline["candidates"]}[
        candidate["step"]
    ]
    domains = {}
    for domain, row in candidate["domains"].items():
        old = prior["domains"][domain]
        domains[domain] = {
            "field_pass_v24_to_v25": [
                old["field_gate"]["pass"],
                row["field_gate"]["pass"],
            ],
            "Q3_delta_q99_999_dex_v24_to_v25": [
                old["mechanism_Q3_Q4"]["delta_q99_999_dex"],
                row["mechanism_Q3_Q4"]["delta_q99_999_dex"],
            ],
            "Q3_generated_max_above_truth_v24_to_v25": [
                old["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"],
                row["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"],
            ],
            "Q4_v24_to_v25": [
                old["mechanism_Q3_Q4"][
                    "generated_over_truth_mean_delta_squared"
                ],
                row["mechanism_Q3_Q4"][
                    "generated_over_truth_mean_delta_squared"
                ],
            ],
            "Q6_latent_mean_v24_to_v25": [
                old["conditional_Q6_latent"][
                    "maximum_absolute_generated_minus_truth_mean"
                ],
                row["conditional_Q6_latent"][
                    "maximum_absolute_generated_minus_truth_mean"
                ],
            ],
        }
    return domains


def _failure_classification(
    final: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    prior = baseline["candidates"][-1]
    improved = {
        domain: final["domains"][domain]["mechanism_Q3_Q4"][
            "generated_max_above_truth_max_dex"
        ]
        < prior["domains"][domain]["mechanism_Q3_Q4"][
            "generated_max_above_truth_max_dex"
        ]
        for domain in ("tng", "simba_dev")
    }
    lost_checks = {}
    for domain in DOMAINS:
        old_checks = prior["domains"][domain]["field_gate"]["checks"]
        new_checks = final["domains"][domain]["field_gate"]["checks"]
        lost = [
            name for name, passed in old_checks.items()
            if passed and not new_checks[name]
        ]
        if lost:
            lost_checks[domain] = lost
    if all(improved.values()) and lost_checks:
        classification = "tail_reweighting_created_a_calibration_morphology_tradeoff"
        next_step = (
            "stop_without_mixture_tuning_and_design_a_proper_conditional_likelihood_or_representation"
        )
    elif not all(improved.values()):
        classification = "target_tail_reweighting_not_the_primary_remaining_mechanism"
        next_step = (
            "stop_edm_loss_variants_and_design_an_explicitly_conditional_alternative_likelihood"
        )
    else:
        classification = "proper_score_near_miss_without_rescue"
        next_step = "stop_and_audit_frozen_trajectories_without_rescue"
    return {
        "class": classification,
        "Q3_maximum_improved_vs_v24": improved,
        "newly_lost_field_checks_vs_v24": lost_checks,
        "next": next_step,
    }


def evaluate(
    *,
    root: Path,
    training: Path,
    registry_path: Path,
    repo: Path,
    gate_commit: str,
) -> dict[str, Any]:
    registry, artifacts, v20, baseline = load_frozen_program(registry_path, repo)
    experiment = v20["e8_gaussianized_marginal_retrain"]
    run_path = training / "run.json"
    run = json.loads(run_path.read_text())
    expected_loss = {
        "coefficients": {"unweighted": 1.0, "tail_weighted": 0.0},
        "tail_weighted_role": "detached_diagnostic_only",
        "proper_score_for_unreweighted_conditional_distribution": True,
        "band_balanced": False,
    }
    if (
        run.get("status") != "complete"
        or run.get("schema") != V25_E13_SCHEMA
        or run.get("experiment_registry_sha256") != REGISTRY_SHA256
        or run.get("steps") != 30000
        or run.get("candidate_steps") != list(CANDIDATES)
        or run.get("base_channels") != 48
        or run.get("parameters") != PARAMETERS
        or run.get("denoising_loss") != expected_loss
        or run.get("sigma_data") != artifacts["initialization"]["sigma_data"]
        or run.get("edm_p_mean") != P_MEAN
        or run.get("edm_p_std") != P_STD
    ):
        raise ValueError("V25 training run protocol/provenance mismatch")
    preflight = _preflight(run)
    variance = _remeasure_variance(artifacts)
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    transform = json.loads(Path(artifacts["gaussianization"]["path"]).read_text())
    edges = np.asarray(profile["edges"], dtype=np.float64)
    expected_indices = {
        domain: _indices(
            experiment["development_objects"][REGISTRY_DOMAINS[domain]], repo
        )
        for domain in DOMAINS
    }
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
            raise ValueError("V25 training commit is not an ancestor of gate commit")
        domains = {}
        for domain in DOMAINS:
            source = REGISTRY_DOMAINS[domain]
            domain_root = root / f"step_{step:06d}" / domain
            ensemble_path = domain_root / "ensemble16_steps40.h5"
            metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
            initialization = _validate_ensemble_v22(
                ensemble_path,
                artifacts=artifacts,
                v20=v20,
                domain=domain,
                step=step,
                checkpoint_path=checkpoint_path,
                checkpoint_sha=checkpoint_sha,
                expected_indices=expected_indices[domain],
                gate_commit=gate_commit,
                checkpoint_schema=V25_E13_SCHEMA,
                registry_sha=REGISTRY_SHA256,
                registry_metadata_key="v25_registry_sha256",
                label="V25",
            )
            metrics = _load_metrics(metrics_path)
            if Path(metrics["path"]).resolve() != ensemble_path.resolve():
                raise ValueError("V25 metrics refer to another ensemble")
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
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_training_code_commit": training_commit,
            "gradient_diagnostic": checkpoint.get("gradient_diagnostic"),
            "balanced_unweighted_validation": checkpoint.get(
                "balanced_validation"
            ),
            "fixed_tail_weighted_validation_diagnostic": checkpoint.get(
                "fixed_tail_weighted_validation_diagnostic"
            ),
            "tail_weighted_validation_selection_role": checkpoint.get(
                "tail_weighted_validation_selection_role"
            ),
            "domains": domains,
            "Q3_all_domains": q3,
            "Q4_all_domains": q4,
            "Q5_all_domains": q5,
            "all_three_field_pass": field_pass,
            "all_three_pass": field_pass and q3 and q4,
        }
        candidate["comparison_to_v24"] = _comparison(candidate, baseline)
        candidates.append(candidate)
    selected = select_candidate_rows(candidates)
    if selected is not None:
        classification = {
            "class": "proper_unweighted_score_objective_sufficient",
            "next": "seal_v25_and_await_explicit_approval_before_independent_data",
        }
    else:
        classification = _failure_classification(candidates[-1], baseline)
    report = {
        "schema": SCHEMA,
        "experiment": "v25_proper_unweighted_score",
        "registry": str(registry_path.resolve()),
        "registry_sha256": REGISTRY_SHA256,
        "v21_artifacts_sha256": ARTIFACT_SHA256,
        "training": str(training.resolve()),
        "training_run_sha256": sha256_file(run_path),
        "hard_preflight": preflight,
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
        raise RuntimeError("V25 gate requires a clean committed worktree")
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
            raise RuntimeError("existing V25 decision differs from recomputed decision")
        print(json.dumps(existing, indent=2))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
