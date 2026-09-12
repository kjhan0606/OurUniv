#!/usr/bin/env python
"""Frozen three-domain gate for V31 physical conditional-copula transport."""
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
from hong2021_v22_development_gate import _candidate_mechanism_pass
from hong2021_v28_empirical import DOMAIN_KEYS, DOMAIN_ORDER
from hong2021_v31_copula import (
    ENSEMBLE_SCHEMA,
    PREFLIGHT_SCHEMA,
    REGISTRY_SHA256,
    load_program,
)


SCHEMA = "hong2021-v31-physical-conditional-copula-three-domain-decision-v1"


def _validate_ensemble(path: Path, parent: Path, gate_commit: str) -> dict[str, Any]:
    with h5py.File(path, "r") as current, h5py.File(parent, "r") as old:
        exact = {
            "schema": ENSEMBLE_SCHEMA,
            "method": "train_only_physical_residual_conditional_copula",
            "v31_registry_sha256": REGISTRY_SHA256,
            "parent_v28_ensemble_sha256": sha256_file(parent),
            "ensemble_members": 16,
            "diagnostic_k_h_mpc": 1.0,
            "donor_reselection": False,
            "selection_uses_validation_truth": False,
            "copula_fit_uses_validation_truth": False,
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
                raise ValueError(f"V31 ensemble metadata differs: {key}")
        reused = (
            "source_index", "donor_source", "donor_index", "donor_isometry",
            "donor_distance", "predicted_residual_dc", "predicted_band_scales",
        )
        if any(not np.array_equal(current[name][:], old[name][:]) for name in reused):
            raise ValueError("V31 did not reuse V28 selections exactly")
        residual = np.asarray(current["sample"], dtype=np.float32) - np.asarray(
            current["conditional_mean"], dtype=np.float32
        )[:, None]
        maximum_dc = float(np.max(np.abs(residual.mean(axis=(-3, -2, -1)))))
        if maximum_dc > 1.0e-7:
            raise ValueError("V31 transported residual DC differs")
        sampling_commit = str(current.attrs.get("sampling_code_commit", ""))
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", sampling_commit, gate_commit],
            capture_output=True,
        ).returncode:
            raise ValueError("V31 sampling commit is not an ancestor of gate commit")
        preflight_path = Path(str(current.attrs["hard_preflight"]))
        if sha256_file(preflight_path) != str(current.attrs["hard_preflight_sha256"]):
            raise ValueError("V31 preflight hash differs")
        preflight = json.loads(preflight_path.read_text())
        if (
            preflight.get("schema") != PREFLIGHT_SCHEMA
            or preflight.get("status") != "pass"
            or preflight.get("code_commit") != sampling_commit
        ):
            raise ValueError("V31 preflight content differs")
        model_path = Path(str(current.attrs["conditional_copula_model"]))
        report_path = Path(str(current.attrs["conditional_copula_report"]))
        if (
            sha256_file(model_path) != str(current.attrs["conditional_copula_model_sha256"])
            or sha256_file(report_path) != str(current.attrs["conditional_copula_report_sha256"])
        ):
            raise ValueError("V31 conditional-copula artifact binding differs")
    return {
        "sampling_code_commit": sampling_commit,
        "maximum_absolute_sample_residual_dc": maximum_dc,
        "donor_selection_exactly_reused": True,
        "conditional_copula_model": str(model_path),
        "conditional_copula_model_sha256": sha256_file(model_path),
    }


def evaluate(root: Path, registry_path: Path, repo: Path, gate_commit: str) -> dict[str, Any]:
    registry = load_program(registry_path, repo)
    domains = {}
    for source in DOMAIN_ORDER:
        domain = DOMAIN_KEYS[source]
        domain_root = root / domain
        ensemble = domain_root / "ensemble16.h5"
        parent = Path(registry["frozen_v28_selections"][source]["ensemble"])
        selection = _validate_ensemble(ensemble, parent, gate_commit)
        metrics_path = domain_root / "ensemble_evaluation" / "metrics.json"
        metrics = _load_metrics(metrics_path)
        if Path(metrics["path"]).resolve() != ensemble.resolve():
            raise ValueError("V31 metrics refer to another ensemble")
        domains[domain] = {
            "ensemble": str(ensemble.resolve()),
            "ensemble_sha256": sha256_file(ensemble),
            "metrics": str(metrics_path.resolve()),
            "metrics_sha256": sha256_file(metrics_path),
            "field_gate": field_gate(metrics),
            "mechanism_Q3_Q4": marginal_diagnostics(ensemble),
            "selection_diagnostics": selection,
        }
    q3, q4, q5 = _candidate_mechanism_pass(domains)
    field_pass = all(row["field_gate"]["pass"] for row in domains.values())
    passed = field_pass and q3 and q4
    if passed:
        classification = {
            "class": "physical_conditional_copula_control_sufficient",
            "next": "seal_v31_and_await_explicit_user_approval_before_Astrid",
        }
    elif q3 and q4:
        classification = {
            "class": "local_conditional_copula_repairs_tails_but_spatial_innovation_or_descriptor_limits_morphology",
            "next": "replace_empirical_donor_retrieval_with_learned_local_conditional_innovation_model_while_retaining_this_coordinate",
        }
    else:
        classification = {
            "class": "backbone_only_conditional_copula_is_insufficient",
            "next": "include_local_observable_patches_and_multiscale_backbone_context_in_conditional_residual_representation",
        }
    comparison: dict[str, Any] = {}
    v29_path = Path(
        "/gpfs/kjhan/IllustrisTNG/TNG100-1/evaluation/"
        "tng100_simba_swift_v29_e17_direct_physical_residual/development_decision.json"
    )
    if sha256_file(v29_path) != "7e745a3b435d633fb30b10ba3dc636a2dfa1bd203b8dd1850442d56796dfb3d9":
        raise ValueError("V31 paired V29 decision hash differs")
    v29 = json.loads(v29_path.read_text())
    for domain, row in domains.items():
        old = v29["candidate"]["domains"][domain]
        comparison[domain] = {
            "field_pass_v29_to_v31": [old["field_gate"]["pass"], row["field_gate"]["pass"]],
            "Q3_delta_q99_999_dex_v29_to_v31": [
                old["mechanism_Q3_Q4"]["delta_q99_999_dex"],
                row["mechanism_Q3_Q4"]["delta_q99_999_dex"],
            ],
            "Q3_maximum_excess_v29_to_v31": [
                old["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"],
                row["mechanism_Q3_Q4"]["generated_max_above_truth_max_dex"],
            ],
            "Q4_v29_to_v31": [
                old["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"],
                row["mechanism_Q3_Q4"]["generated_over_truth_mean_delta_squared"],
            ],
        }
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": "v31_train_only_physical_conditional_copula",
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
        "comparison_to_v29_paired_donors": comparison,
        "development_pass": passed,
        "classification": classification,
        "next": classification["next"],
        "donor_reselection": False,
        "validation_truth_used_for_fit_or_sampling": False,
        "posthoc_Ak_used": False,
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
        raise RuntimeError("V31 gate requires a clean worktree")
    report = evaluate(args.root.resolve(), args.registry.resolve(), args.repo.resolve(), commit)
    if args.out.exists():
        raise RuntimeError("V31 refuses to overwrite its decision")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
