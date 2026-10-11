#!/usr/bin/env python
"""Frozen three-domain gate for the V28 empirical joint residual control."""
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
from hong2021_v15_development_gate import _load_metrics, canonical_digest, git_state
from hong2021_v18_edm import _indices
from hong2021_v18_init import sha256_file
from hong2021_v20_development_gate import marginal_diagnostics
from hong2021_v21_development_gate import conditional_diagnostics
from hong2021_v22_development_gate import (
    _candidate_mechanism_pass,
    latent_conditional_diagnostics,
)
from hong2021_v26 import CACHE_KEYS
from hong2021_v28_empirical import (
    DESIGN_AUDIT_SHA256,
    DOMAIN_KEYS,
    DOMAIN_ORDER,
    DONOR_COUNTS,
    ENSEMBLE_MEMBERS,
    ENSEMBLE_SCHEMA,
    GLOBAL_PREFILTER,
    PARENT_AUDIT_SHA256,
    PREFLIGHT_SCHEMA,
    REGISTRY_SHA256,
    load_frozen_program,
    source_quota,
)


SCHEMA = "hong2021-v28-empirical-joint-control-three-domain-decision-v1"
V27_DECISION = Path(
    "/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/"
    "tng100_simba_swift_v27_e15_parent_aligned_haar_flow/development_decision.json"
)


def _validate_ensemble(
    path: Path,
    *,
    source: str,
    expected_indices: list[int],
    artifacts: dict[str, Any],
    v20: dict[str, Any],
    gate_commit: str,
    global_offset: int,
) -> dict[str, Any]:
    experiment = v20["e8_gaussianized_marginal_retrain"]
    data = experiment["data"][source]["validation_data"]
    cache = artifacts["caches"][CACHE_KEYS[source]["validation"]]
    with h5py.File(path, "r") as handle:
        if (
            tuple(handle["sample"].shape) != (16, 16, 1, 64, 64, 64)
            or [int(value) for value in handle["source_index"][:]]
            != expected_indices
        ):
            raise ValueError("V28 ensemble shape or source indices differ")
        exact = {
            "schema": ENSEMBLE_SCHEMA,
            "method": "source_balanced_observation_matched_full_cube_latent_knn",
            "v28_registry_sha256": REGISTRY_SHA256,
            "design_audit_sha256": DESIGN_AUDIT_SHA256,
            "parent_audit_sha256": PARENT_AUDIT_SHA256,
            "source_cache_sha256": cache["sha256"],
            "source_data_sha256": data["sha256"],
            "ensemble_members": ENSEMBLE_MEMBERS,
            "global_prefilter_per_source": GLOBAL_PREFILTER,
            "selection_uses_validation_truth": False,
            "location_scale_uses_target": False,
            "direct_empirical_sampling": True,
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
                raise ValueError(f"V28 ensemble metadata differs: {key}")
        commit = str(handle.attrs.get("sampling_code_commit", ""))
        if subprocess.run(
            ["git", "-C", str(Path.cwd()), "merge-base", "--is-ancestor", commit, gate_commit],
            capture_output=True,
        ).returncode:
            raise ValueError("V28 sampling commit is not an ancestor of gate commit")
        preflight_path = Path(str(handle.attrs.get("hard_preflight", "")))
        preflight_sha = str(handle.attrs.get("hard_preflight_sha256", ""))
        if sha256_file(preflight_path) != preflight_sha:
            raise ValueError("V28 hard-preflight hash differs")
        preflight = json.loads(preflight_path.read_text())
        if (
            preflight.get("schema") != PREFLIGHT_SCHEMA
            or preflight.get("status") != "pass"
            or preflight.get("code_commit") != commit
        ):
            raise ValueError("V28 hard-preflight payload differs")
        donor_source = np.asarray(handle["donor_source"], dtype=np.int64)
        donor_index = np.asarray(handle["donor_index"], dtype=np.int64)
        donor_isometry = np.asarray(handle["donor_isometry"], dtype=np.int64)
        distance = np.asarray(handle["donor_distance"], dtype=np.float64)
        if (
            donor_source.shape != (16, 16)
            or donor_index.shape != (16, 16)
            or donor_isometry.shape != (16, 16)
            or distance.shape != (16, 16, 3)
            or not np.isfinite(distance).all()
            or not np.allclose(distance[..., 2], distance[..., 0] + distance[..., 1], rtol=2e-6, atol=2e-6)
            or np.any((donor_isometry < 0) | (donor_isometry >= 48))
        ):
            raise ValueError("V28 donor diagnostic arrays differ")
        counts = {domain: 0 for domain in DOMAIN_ORDER}
        for object_index in range(16):
            expected = source_quota(global_offset + object_index)
            for source_code, domain in enumerate(DOMAIN_ORDER):
                selected = donor_index[object_index][donor_source[object_index] == source_code]
                if (
                    len(selected) != expected[domain]
                    or len(np.unique(selected)) != len(selected)
                    or np.any(selected < 0)
                    or np.any(selected >= DONOR_COUNTS[domain])
                ):
                    raise ValueError("V28 donor source quota or uniqueness differs")
                counts[domain] += len(selected)
        maximum_dc = float(
            handle.attrs.get("maximum_absolute_selected_latent_dc", np.inf)
        )
        if maximum_dc > 1.0e-7:
            raise ValueError("V28 selected train latent DC differs")
    return {
        "sampling_code_commit": commit,
        "hard_preflight": str(preflight_path),
        "hard_preflight_sha256": preflight_sha,
        "maximum_absolute_selected_latent_dc": maximum_dc,
        "selected_donor_counts": counts,
    }


def _comparison_to_v27(domains: dict[str, Any], v27: dict[str, Any]) -> dict[str, Any]:
    prior = v27["candidates"][-1]["domains"]
    output = {}
    for domain, row in domains.items():
        old = prior[domain]
        output[domain] = {
            "field_pass_v27_to_v28": [old["field_gate"]["pass"], row["field_gate"]["pass"]],
            "Q3_delta_q99_999_dex_v27_to_v28": [
                old["mechanism_Q3_Q4"]["delta_q99_999_dex"],
                row["mechanism_Q3_Q4"]["delta_q99_999_dex"],
            ],
            "Q3_generated_max_above_truth_v27_to_v28": [
                old["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"],
                row["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"],
            ],
            "Q4_v27_to_v28": [
                old["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"],
                row["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"],
            ],
        }
    return output


def evaluate(*, root: Path, registry_path: Path, repo: Path, gate_commit: str) -> dict[str, Any]:
    registry, artifacts, v20 = load_frozen_program(registry_path, repo)
    experiment = v20["e8_gaussianized_marginal_retrain"]
    profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
    transform = json.loads(Path(artifacts["gaussianization"]["path"]).read_text())
    edges = np.asarray(profile["edges"], dtype=np.float64)
    expected_indices = {
        source: _indices(experiment["development_objects"][source], repo)
        for source in DOMAIN_ORDER
    }
    domains = {}
    total_donors = {source: 0 for source in DOMAIN_ORDER}
    offset = 0
    for source in DOMAIN_ORDER:
        domain = DOMAIN_KEYS[source]
        domain_root = root / domain
        ensemble_path = domain_root / "ensemble16.h5"
        metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
        selection = _validate_ensemble(
            ensemble_path,
            source=source,
            expected_indices=expected_indices[source],
            artifacts=artifacts,
            v20=v20,
            gate_commit=gate_commit,
            global_offset=offset,
        )
        offset += 16
        for donor_source, count in selection["selected_donor_counts"].items():
            total_donors[donor_source] += count
        metrics = _load_metrics(metrics_path)
        if Path(metrics["path"]).resolve() != ensemble_path.resolve():
            raise ValueError("V28 metrics refer to another ensemble")
        domains[domain] = {
            "ensemble": str(ensemble_path.resolve()),
            "ensemble_sha256": sha256_file(ensemble_path),
            "metrics": str(metrics_path.resolve()),
            "metrics_sha256": sha256_file(metrics_path),
            "field_gate": field_gate(metrics),
            "selection_diagnostics": selection,
            "mechanism_Q3_Q4": marginal_diagnostics(ensemble_path),
            "conditional_Q6_residual": conditional_diagnostics(ensemble_path, edges),
            "conditional_Q6_latent": latent_conditional_diagnostics(
                ensemble_path,
                Path(artifacts["caches"][CACHE_KEYS[source]["validation"]]["path"]),
                profile,
                transform,
            ),
        }
    if total_donors != {source: 256 for source in DOMAIN_ORDER}:
        raise ValueError("V28 aggregate donor source balance differs")
    q3, q4, q5 = _candidate_mechanism_pass(domains)
    field_pass = all(row["field_gate"]["pass"] for row in domains.values())
    development_pass = field_pass and q3 and q4
    if development_pass:
        classification = {
            "class": "intact_train_joint_residual_control_sufficient",
            "next": "seal_v28_and_await_explicit_user_approval_before_Astrid",
        }
    elif q3 and q4:
        classification = {
            "class": "joint_residual_support_repaired_but_observation_matching_or_deterministic_backbone_limits_morphology",
            "next": "audit_deterministic_current_density_backbone_and_local_condition_matching",
        }
    else:
        classification = {
            "class": "train_empirical_joint_support_does_not_transfer_through_current_representation",
            "next": "audit_V21_V14_density_representation_and_deterministic_backbone",
        }
    v27 = json.loads(V27_DECISION.read_text())
    candidate = {
        "domains": domains,
        "aggregate_selected_donor_counts": total_donors,
        "Q3_all_domains": q3,
        "Q4_all_domains": q4,
        "Q5_all_domains": q5,
        "all_three_field_pass": field_pass,
        "all_three_pass": development_pass,
    }
    report = {
        "schema": SCHEMA,
        "experiment": "v28_train_only_empirical_joint_residual_control",
        "registry": str(registry_path.resolve()),
        "registry_sha256": REGISTRY_SHA256,
        "gate_code_commit": gate_commit,
        "worktree_clean_at_gate": True,
        "candidate": candidate,
        "comparison_to_v27_step_30000": _comparison_to_v27(domains, v27),
        "development_pass": development_pass,
        "classification": classification,
        "next": classification["next"],
        "validation_truth_used_for_donor_selection": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    commit, clean = git_state(args.repo.resolve())
    if not clean:
        raise RuntimeError("V28 gate requires a clean committed worktree")
    report = evaluate(
        root=args.root.resolve(),
        registry_path=args.registry.resolve(),
        repo=args.repo.resolve(),
        gate_commit=commit,
    )
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite V28 decision: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
