#!/usr/bin/env python
"""Frozen three-domain gate for V29 direct physical residual transport."""
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
from hong2021_v18_init import sha256_file
from hong2021_v20_development_gate import marginal_diagnostics
from hong2021_v21_development_gate import conditional_diagnostics
from hong2021_v22_development_gate import _candidate_mechanism_pass
from hong2021_v28_empirical import DOMAIN_KEYS, DOMAIN_ORDER
from hong2021_v29_physical import (
    DESIGN_AUDIT_SHA256,
    ENSEMBLE_SCHEMA,
    FAILURE_AUDIT_SHA256,
    PREFLIGHT_SCHEMA,
    REGISTRY_SHA256,
    load_frozen_program,
)


SCHEMA = "hong2021-v29-direct-physical-residual-three-domain-decision-v1"


def _validate_ensemble(
    path: Path, parent: Path, *, gate_commit: str
) -> dict[str, Any]:
    with h5py.File(path, "r") as current, h5py.File(parent, "r") as old:
        exact = {
            "schema": ENSEMBLE_SCHEMA,
            "method": "same_donors_direct_centered_physical_y_residual_transport",
            "v29_registry_sha256": REGISTRY_SHA256,
            "design_audit_sha256": DESIGN_AUDIT_SHA256,
            "failure_audit_sha256": FAILURE_AUDIT_SHA256,
            "parent_v28_ensemble_sha256": sha256_file(parent),
            "ensemble_members": 16,
            "donor_reselection": False,
            "selection_uses_validation_truth": False,
            "query_dependent_nonlinear_inverse": False,
            "worktree_clean_at_sampling": True,
            "Astrid_accessed": False,
            "historical_EAGLE_accessed": False,
            "complete": True,
        }
        for key, expected in exact.items():
            actual = current.attrs.get(key)
            if isinstance(actual, np.generic):
                actual = actual.item()
            if actual != expected:
                raise ValueError(f"V29 ensemble metadata differs: {key}")
        reused = (
            "source_index", "donor_source", "donor_index", "donor_isometry",
            "donor_distance", "predicted_residual_dc", "predicted_band_scales",
        )
        if any(not np.array_equal(current[name][:], old[name][:]) for name in reused):
            raise ValueError("V29 did not reuse V28 donor selections exactly")
        residual = np.asarray(current["sample"], dtype=np.float32) - np.asarray(
            current["conditional_mean"][:, None], dtype=np.float32
        )
        maximum_dc = float(
            np.max(np.abs(residual.mean(axis=(-3, -2, -1))))
        )
        if (
            maximum_dc > 1.0e-7
            or float(current.attrs.get("maximum_absolute_centered_donor_residual_dc", np.inf)) > 1.0e-7
        ):
            raise ValueError("V29 transported physical residual DC differs")
        commit = str(current.attrs.get("sampling_code_commit", ""))
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, gate_commit],
            capture_output=True,
        ).returncode:
            raise ValueError("V29 sampling commit is not an ancestor of gate commit")
        preflight_path = Path(str(current.attrs.get("hard_preflight", "")))
        preflight_sha = str(current.attrs.get("hard_preflight_sha256", ""))
        preflight = json.loads(preflight_path.read_text())
        if (
            sha256_file(preflight_path) != preflight_sha
            or preflight.get("schema") != PREFLIGHT_SCHEMA
            or preflight.get("status") != "pass"
            or preflight.get("code_commit") != commit
        ):
            raise ValueError("V29 hard preflight differs")
    return {
        "sampling_code_commit": commit,
        "hard_preflight": str(preflight_path),
        "hard_preflight_sha256": preflight_sha,
        "maximum_absolute_sample_residual_dc": maximum_dc,
        "donor_selection_exactly_reused": True,
    }


def evaluate(*, root: Path, registry_path: Path, repo: Path, gate_commit: str) -> dict[str, Any]:
    registry, artifacts, _ = load_frozen_program(registry_path, repo)
    domains = {}
    for source in DOMAIN_ORDER:
        domain = DOMAIN_KEYS[source]
        domain_root = root / domain
        ensemble_path = domain_root / "ensemble16.h5"
        parent_path = Path(registry["frozen_v28_selections"][domain]["ensemble"])
        selection = _validate_ensemble(
            ensemble_path, parent_path, gate_commit=gate_commit
        )
        metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
        metrics = _load_metrics(metrics_path)
        if Path(metrics["path"]).resolve() != ensemble_path.resolve():
            raise ValueError("V29 metrics refer to another ensemble")
        profile = json.loads(Path(artifacts["profile"]["path"]).read_text())
        domains[domain] = {
            "ensemble": str(ensemble_path.resolve()),
            "ensemble_sha256": sha256_file(ensemble_path),
            "metrics": str(metrics_path.resolve()),
            "metrics_sha256": sha256_file(metrics_path),
            "field_gate": field_gate(metrics),
            "selection_diagnostics": selection,
            "mechanism_Q3_Q4": marginal_diagnostics(ensemble_path),
            "conditional_Q6_residual": conditional_diagnostics(
                ensemble_path, np.asarray(profile["edges"], dtype=np.float64)
            ),
        }
    q3, q4, q5 = _candidate_mechanism_pass(domains)
    field_pass = all(row["field_gate"]["pass"] for row in domains.values())
    passed = field_pass and q3 and q4
    if passed:
        classification = {
            "class": "direct_physical_residual_coordinate_sufficient",
            "next": "seal_v29_and_await_explicit_user_approval_before_Astrid",
        }
    elif q3 and q4:
        classification = {
            "class": "physical_support_repaired_but_deterministic_backbone_or_local_phase_alignment_limits_morphology",
            "next": "audit_and_replace_deterministic_current_density_backbone",
        }
    else:
        classification = {
            "class": "direct_physical_residual_transport_still_mismatches_validation_population",
            "next": "replace_deterministic_current_density_backbone_and_local_condition_representation",
        }
    v28 = json.loads(Path(registry["parent_evidence"]["v28_decision"]).read_text())
    comparison = {}
    for domain, row in domains.items():
        old = v28["candidate"]["domains"][domain]
        comparison[domain] = {
            "field_pass_v28_to_v29": [old["field_gate"]["pass"], row["field_gate"]["pass"]],
            "Q3_delta_q99_999_dex_v28_to_v29": [old["mechanism_Q3_Q4"]["delta_q99_999_dex"], row["mechanism_Q3_Q4"]["delta_q99_999_dex"]],
            "Q3_maximum_excess_v28_to_v29": [old["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"], row["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"]],
            "Q4_v28_to_v29": [old["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"], row["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"]],
        }
    report = {
        "schema": SCHEMA,
        "experiment": "v29_same_donors_direct_physical_residual_transport",
        "registry": str(registry_path.resolve()),
        "registry_sha256": REGISTRY_SHA256,
        "gate_code_commit": gate_commit,
        "worktree_clean_at_gate": True,
        "candidate": {
            "domains": domains,
            "Q3_all_domains": q3,
            "Q4_all_domains": q4,
            "Q5_all_domains": q5,
            "all_three_field_pass": field_pass,
            "all_three_pass": passed,
        },
        "comparison_to_v28_paired_donors": comparison,
        "development_pass": passed,
        "classification": classification,
        "next": classification["next"],
        "donor_reselection": False,
        "validation_truth_used_for_sampling": False,
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
        raise RuntimeError("V29 gate requires a clean worktree")
    report = evaluate(
        root=args.root.resolve(), registry_path=args.registry.resolve(),
        repo=args.repo.resolve(), gate_commit=commit,
    )
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite V29 decision: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
