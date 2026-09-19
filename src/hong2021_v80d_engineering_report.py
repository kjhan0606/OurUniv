#!/usr/bin/env python
"""Measure V80D with V79 formulas without creating a V79 gate result."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

import hong2021_v79_complete_gate as v79
import hong2021_v80_sample as base_sample
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor
from hong2021_v80d_engineering_sample import (
    diagnostic_freeze_commit,
    load_diagnostic_program,
)


REPORT_SCHEMA = "hong2021-v80d-engineering-diagnostic-report-v1"
V79_PROGRAM = Path("config/hong2021_v79_complete_candidate_agnostic_gate_program.json")
DOMAIN_ORDER = base_sample.DOMAIN_ORDER


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path.resolve())}


def run(program_path: Path, repo: Path, output_path: Path) -> dict[str, Any]:
    repo = repo.resolve()
    diagnostic, base = load_diagnostic_program(program_path.resolve(), repo)
    commit, clean = git_state(repo)
    freeze_commit = diagnostic_freeze_commit(program_path, repo)
    if not clean or not _is_ancestor(repo, freeze_commit, commit):
        raise RuntimeError("V80D report requires clean frozen ancestry")
    v79_program, references = v79.load_program((repo / V79_PROGRAM).resolve(), repo)
    root = Path(diagnostic["outputs"]["ensemble_root"])
    domains: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        key = base_sample.DOMAIN_KEYS[domain]
        candidate = root / "candidate" / key / "ensemble16.h5"
        control = root / "control" / key / "ensemble16.h5"
        candidate_metrics_path = root / "candidate" / key / "ensemble_evaluation" / "metrics.json"
        control_metrics_path = root / "control" / key / "ensemble_evaluation" / "metrics.json"
        rows = {
            "candidate_ensemble": artifact(candidate),
            "control_ensemble": artifact(control),
            "candidate_metrics": artifact(candidate_metrics_path),
            "control_metrics": artifact(control_metrics_path),
        }
        contract = base["frozen_domain_execution_contracts"][domain]
        numerical = v79.validate_ensemble_pair(
            candidate,
            control,
            base["single_use_fresh_selection"][domain],
            contract["candidate_expected_attrs"],
            contract["control_expected_attrs"],
            contract["pairing"],
        )
        domains[domain] = {
            "candidate_ensemble": candidate,
            "candidate_ensemble_sha256": rows["candidate_ensemble"]["sha256"],
            "control_ensemble": control,
            "control_ensemble_sha256": rows["control_ensemble"]["sha256"],
            "candidate_metrics_path": candidate_metrics_path,
            "candidate_metrics_sha256": rows["candidate_metrics"]["sha256"],
            "control_metrics_path": control_metrics_path,
            "control_metrics_sha256": rows["control_metrics"]["sha256"],
            "candidate_metrics": v79.load_metrics(
                candidate_metrics_path,
                candidate,
                base["single_use_fresh_selection"][domain],
            ),
            "control_metrics": v79.load_metrics(
                control_metrics_path,
                control,
                base["single_use_fresh_selection"][domain],
            ),
            "numerical_and_pairing": numerical,
        }
        artifacts[domain] = rows
    physical_p, physical = v79.physical_energy_observation(domains, references)
    rank_p, rank = v79.rank_coverage_observation(domains)
    formula_global_p = float(
        v79.v78.global_p_value(
            np.asarray([physical_p]), np.asarray([rank_p])
        )[0]
    )
    deterministic = all(
        row["numerical_and_pairing"]["residual_DC_pass"]
        for row in domains.values()
    )
    formula_pass = bool(formula_global_p > v79.GLOBAL_ALPHA and deterministic)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete_V80D_engineering_diagnostic_not_a_V79_gate",
        "program": str(program_path.resolve()),
        "program_sha256": sha256_file(program_path),
        "program_freeze_commit": freeze_commit,
        "report_code_commit": commit,
        "worktree_clean": clean,
        "parent_terminal_failure_seal_sha256": diagnostic["parent_failure"][
            "terminal_seal_sha256"
        ],
        "engineering_only": True,
        "same_previously_reserved_subset_reused": True,
        "statistically_valid_target_unseen_V79_result": False,
        "V79_gate_or_manifest_executed": False,
        "only_code_change": diagnostic["only_code_change"],
        "frozen_V79_formula_diagnostic": {
            "physical_energy": physical,
            "rank_coverage": rank,
            "formula_global_p": formula_global_p,
            "formula_global_alpha_reference": v79.GLOBAL_ALPHA,
            "deterministic_numerical_and_pairing_pass": deterministic,
            "would_pass_formula_if_prospective": formula_pass,
            "may_be_reported_as_a_V79_pass": False,
        },
        "artifacts": artifacts,
        "V72_stage_B_accessed": False,
        "Astrid_or_EAGLE_accessed": False,
        "RAMSES_modified_or_stopped": False,
        "next": (
            "If scientifically promising, acquire genuinely independent simulation objects and prospectively freeze a redesigned V81 validation."
            if formula_pass
            else "Stop V80 development; do not spend new independent validation data on this candidate."
        ),
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    if output_path.exists():
        raise FileExistsError("V80D report refuses an existing output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(".json.partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.program, args.repo, args.out), indent=2), flush=True)


if __name__ == "__main__":
    main()
