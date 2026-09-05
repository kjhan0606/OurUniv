#!/usr/bin/env python
"""Compute the V80DR engineering formula from sealed existing metrics only."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

import hong2021_v79_complete_gate as v79
import hong2021_v80_sample as base_sample
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor


PROGRAM_SCHEMA = "hong2021-v80dr2-report-only-program-v1"
PROGRAM_STATUS = "frozen_before_formula_report_execution_or_metric_value_inspection"
REPORT_SCHEMA = "hong2021-v80dr2-report-only-engineering-diagnostic-v1"
PARENT_SEAL = Path("config/hong2021_v80dr_terminal_failure_seal.json")
PARENT_SEAL_SHA256 = "daa4cbeba6654f38fe0d2491f6bf7d5d1c3df2981bbf0cf7a583912efe756fb4"
V79_PROGRAM = Path("config/hong2021_v79_complete_candidate_agnostic_gate_program.json")
V80_PROGRAM = Path("config/hong2021_v80_single_candidate_program.json")
DOMAIN_ORDER = base_sample.DOMAIN_ORDER
DOMAIN_ROW_KEYS = frozenset(
    {
        "candidate_ensemble",
        "candidate_ensemble_sha256",
        "control_ensemble",
        "control_ensemble_sha256",
        "candidate_metrics_path",
        "candidate_metrics_sha256",
        "control_metrics_path",
        "control_metrics_sha256",
        "candidate_metrics",
        "control_metrics",
        "numerical_and_pairing",
    }
)


def strict_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def resolve_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def program_freeze_commit(program_path: Path, repo: Path) -> str:
    commit = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(program_path.resolve())],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise ValueError("V80DR2 program freeze commit cannot be resolved")
    return commit


def hash_row(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _bound_artifact(row: dict[str, Any], label: str) -> tuple[Path, str]:
    if set(row) != {"path", "sha256"}:
        raise ValueError(f"V80DR2 {label} artifact keys differ")
    path = Path(str(row["path"])).resolve()
    digest = str(row["sha256"])
    if len(digest) != 64 or sha256_file(path) != digest:
        raise ValueError(f"V80DR2 {label} artifact hash differs")
    return path, digest


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program = strict_json(path.resolve())
    authorization = program.get("authorization", {})
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or program.get("engineering_only") is not True
        or program.get("statistically_valid_V79_reexecution") is not False
        or authorization.get("user_approved_report_only_formula") is not True
        or authorization.get("metadata_recovery_or_mutation") is not False
        or authorization.get("resampling") is not False
        or authorization.get("evaluator_execution") is not False
        or authorization.get("V79_manifest_or_gate") is not False
    ):
        raise ValueError("V80DR2 report-only boundary differs")
    parent = (repo / PARENT_SEAL).resolve()
    if (
        sha256_file(parent) != PARENT_SEAL_SHA256
        or program["parent_failure"]["terminal_seal_sha256"]
        != PARENT_SEAL_SHA256
    ):
        raise ValueError("V80DR2 parent seal differs")
    for label, row in program["implementation_sources"].items():
        source = resolve_path(repo, str(row["path"]))
        if sha256_file(source) != row["sha256"]:
            raise ValueError(f"V80DR2 implementation source differs: {label}")
    for label, row in program["fixed_support_artifacts"].items():
        _bound_artifact(row, f"support/{label}")
    if set(program["input_artifacts"]) != set(DOMAIN_ORDER):
        raise ValueError("V80DR2 input domains differ")
    required = {
        "candidate_ensemble",
        "control_ensemble",
        "candidate_metrics",
        "control_metrics",
    }
    seen_paths: set[Path] = set()
    for domain in DOMAIN_ORDER:
        rows = program["input_artifacts"][domain]
        if set(rows) != required:
            raise ValueError(f"V80DR2 {domain} input artifact set differs")
        for label, row in rows.items():
            artifact, _ = _bound_artifact(row, f"{domain}/{label}")
            if artifact in seen_paths:
                raise ValueError("V80DR2 input artifact path reused")
            seen_paths.add(artifact)
    return program


def build_domain_row(
    candidate: Path,
    candidate_sha: str,
    control: Path,
    control_sha: str,
    candidate_metrics_path: Path,
    candidate_metrics_sha: str,
    control_metrics_path: Path,
    control_metrics_sha: str,
    expected_indices: list[int],
    contract: dict[str, Any],
) -> dict[str, Any]:
    numerical = v79.validate_ensemble_pair(
        candidate,
        control,
        expected_indices,
        contract["candidate_expected_attrs"],
        contract["control_expected_attrs"],
        contract["pairing"],
    )
    row = {
        "candidate_ensemble": candidate,
        "candidate_ensemble_sha256": candidate_sha,
        "control_ensemble": control,
        "control_ensemble_sha256": control_sha,
        "candidate_metrics_path": candidate_metrics_path,
        "candidate_metrics_sha256": candidate_metrics_sha,
        "control_metrics_path": control_metrics_path,
        "control_metrics_sha256": control_metrics_sha,
        "candidate_metrics": v79.load_metrics(
            candidate_metrics_path, candidate, expected_indices
        ),
        "control_metrics": v79.load_metrics(
            control_metrics_path, control, expected_indices
        ),
        "numerical_and_pairing": numerical,
    }
    if set(row) != DOMAIN_ROW_KEYS:
        raise RuntimeError("V80DR2 internal V79 domain-row contract differs")
    return row


def formula_diagnostic(
    domains: dict[str, dict[str, Any]], references: dict[str, Path]
) -> dict[str, Any]:
    if set(domains) != set(DOMAIN_ORDER) or any(
        set(domains[domain]) != DOMAIN_ROW_KEYS for domain in DOMAIN_ORDER
    ):
        raise ValueError("V80DR2 domains do not satisfy the exact V79 helper contract")
    physical_p, physical = v79.physical_energy_observation(domains, references)
    rank_p, rank = v79.rank_coverage_observation(domains)
    formula_global_p = float(
        v79.v78.global_p_value(np.asarray([physical_p]), np.asarray([rank_p]))[0]
    )
    deterministic = all(
        domains[domain]["numerical_and_pairing"]["residual_DC_pass"]
        for domain in DOMAIN_ORDER
    )
    return {
        "physical_energy": physical,
        "rank_coverage": rank,
        "formula_global_p": formula_global_p,
        "formula_global_alpha_reference": v79.GLOBAL_ALPHA,
        "deterministic_numerical_and_pairing_pass": deterministic,
        "would_pass_formula_if_prospective": bool(
            formula_global_p > v79.GLOBAL_ALPHA and deterministic
        ),
        "may_be_reported_as_a_V79_pass": False,
    }


def run(
    program_path: Path, repo: Path, output_path: Path
) -> dict[str, Any]:
    repo = repo.resolve()
    program_path = program_path.resolve()
    program = load_program(program_path, repo)
    commit, clean = git_state(repo)
    freeze_commit = program_freeze_commit(program_path, repo)
    if (
        not clean
        or not _is_ancestor(repo, freeze_commit, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V80DR2 report requires clean frozen Lageunha ancestry")
    if output_path.resolve() != Path(program["outputs"]["report"]).resolve():
        raise ValueError("V80DR2 report output differs")
    if output_path.exists():
        raise FileExistsError("V80DR2 refuses an existing report")
    failure = program["frozen_failure_state"]
    if (
        Path(failure["status_path"]).read_text().strip() != failure["status"]
        or sha256_file(Path(failure["status_path"])) != failure["status_sha256"]
        or sha256_file(Path(failure["sealed_result"]))
        != failure["sealed_result_sha256"]
        or Path(failure["prior_report"]).exists()
    ):
        raise ValueError("V80DR2 frozen parent failure state differs")
    candidate_program = base_sample.load_program((repo / V80_PROGRAM).resolve(), repo)
    _, references = v79.load_program((repo / V79_PROGRAM).resolve(), repo)
    domains: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Any] = {}
    for domain in DOMAIN_ORDER:
        rows = program["input_artifacts"][domain]
        candidate, candidate_sha = _bound_artifact(
            rows["candidate_ensemble"], f"{domain}/candidate_ensemble"
        )
        control, control_sha = _bound_artifact(
            rows["control_ensemble"], f"{domain}/control_ensemble"
        )
        candidate_metrics, candidate_metrics_sha = _bound_artifact(
            rows["candidate_metrics"], f"{domain}/candidate_metrics"
        )
        control_metrics, control_metrics_sha = _bound_artifact(
            rows["control_metrics"], f"{domain}/control_metrics"
        )
        domains[domain] = build_domain_row(
            candidate,
            candidate_sha,
            control,
            control_sha,
            candidate_metrics,
            candidate_metrics_sha,
            control_metrics,
            control_metrics_sha,
            candidate_program["single_use_fresh_selection"][domain],
            candidate_program["frozen_domain_execution_contracts"][domain],
        )
        artifacts[domain] = rows
    diagnostic = formula_diagnostic(domains, references)
    formula_pass = diagnostic["would_pass_formula_if_prospective"]
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete_V80DR2_report_only_engineering_diagnostic_not_a_V79_gate",
        "program": str(program_path),
        "program_sha256": sha256_file(program_path),
        "program_freeze_commit": freeze_commit,
        "report_code_commit": commit,
        "worktree_clean": clean,
        "parent_terminal_failure_seal_sha256": PARENT_SEAL_SHA256,
        "engineering_only": True,
        "same_previously_consumed_subset_reused": True,
        "statistically_valid_target_unseen_V79_result": False,
        "report_only": True,
        "metadata_recovery_or_mutation_repeated": False,
        "sampling_repeated": False,
        "evaluator_repeated": False,
        "V79_gate_or_manifest_executed": False,
        "frozen_V79_formula_diagnostic": diagnostic,
        "artifacts": artifacts,
        "V72_stage_B_accessed": False,
        "Astrid_or_EAGLE_accessed": False,
        "RAMSES_modified_or_stopped": False,
        "next": (
            "Audit availability and cost of genuinely independent simulation objects, freeze a prospective V81 validation design, and await explicit approval before acquiring or opening them."
            if formula_pass
            else "Stop V80 development and do not spend new independent validation objects on this candidate."
        ),
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
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
