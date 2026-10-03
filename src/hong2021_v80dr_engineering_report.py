#!/usr/bin/env python
"""Measure recovered V80D copies with the frozen V79 formulas diagnostically."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import hong2021_v79_complete_gate as v79
import hong2021_v80_sample as base_sample
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor
from hong2021_v80dr_metadata_recovery import (
    ADDED_ATTRIBUTE,
    ADDED_VALUE,
    RECORD_SCHEMA,
    load_program,
    program_freeze_commit,
    strict_json,
)


REPORT_SCHEMA = "hong2021-v80dr-recovered-engineering-diagnostic-report-v1"
V79_PROGRAM = Path("config/hong2021_v79_complete_candidate_agnostic_gate_program.json")
DOMAIN_ORDER = base_sample.DOMAIN_ORDER


def artifact(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def load_recovery_record(
    path: Path, program_path: Path, program: dict[str, Any], repo: Path, commit: str
) -> dict[str, Any]:
    record = strict_json(path.resolve())
    if (
        record.get("schema") != RECORD_SCHEMA
        or record.get("status")
        != "complete_metadata_only_copy_recovery_evaluation_may_run_once"
        or record.get("program_sha256") != sha256_file(program_path)
        or record.get("engineering_only") is not True
        or record.get("statistically_valid_V79_reexecution") is not False
        or record.get("only_added_attribute") != {ADDED_ATTRIBUTE: ADDED_VALUE}
        or record.get("sealed_originals_modified") is not False
        or record.get("sampling_repeated") is not False
        or record.get("metrics_created_before_recovery") != 0
        or canonical_digest(record) != record.get("decision_digest_sha256")
        or not _is_ancestor(repo, str(record.get("recovery_code_commit")), commit)
    ):
        raise ValueError("V80DR recovery record differs")
    expected_keys = {
        f"{arm}/{domain}"
        for arm in base_sample.ARMS
        for domain in base_sample.DOMAIN_KEYS.values()
    }
    if set(record.get("artifacts", {})) != expected_keys:
        raise ValueError("V80DR recovery artifact set differs")
    target_root = Path(program["outputs"]["recovered_ensemble_root"]).resolve()
    for key, row in record["artifacts"].items():
        arm, domain = key.split("/")
        recovered = target_root / arm / domain / "ensemble16.h5"
        source = Path(program["frozen_failure_state"]["original_ensemble_root"]) / arm / domain / "ensemble16.h5"
        expected_source = program["sealed_source_ensembles"][key]["sha256"]
        if (
            Path(row["recovered_path"]).resolve() != recovered
            or Path(row["sealed_source_path"]).resolve() != source.resolve()
            or row["sealed_source_sha256"] != expected_source
            or sha256_file(source) != expected_source
            or sha256_file(recovered) != row["recovered_sha256"]
            or row["only_added_attribute"] != {ADDED_ATTRIBUTE: ADDED_VALUE}
            or row["all_dataset_bytes_identical"] is not True
            or row["sealed_source_unchanged"] is not True
        ):
            raise ValueError(f"V80DR recovered artifact differs: {key}")
        with h5py.File(recovered, "r") as handle:
            if float(handle.attrs.get(ADDED_ATTRIBUTE, np.nan)) != ADDED_VALUE:
                raise ValueError(f"V80DR diagnostic attribute differs: {key}")
    return record


def run(
    program_path: Path,
    recovery_record_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    program_path = program_path.resolve()
    program = load_program(program_path, repo)
    commit, clean = git_state(repo)
    freeze_commit = program_freeze_commit(program_path, repo)
    if not clean or not _is_ancestor(repo, freeze_commit, commit):
        raise RuntimeError("V80DR report requires clean frozen ancestry")
    if output_path.resolve() != Path(program["outputs"]["report"]).resolve():
        raise ValueError("V80DR report output differs")
    recovery_record = load_recovery_record(
        recovery_record_path, program_path, program, repo, commit
    )
    candidate_program_value = Path(
        program["frozen_scientific_contract"]["V80_candidate_program"]
    )
    candidate_program_path = (
        candidate_program_value.resolve()
        if candidate_program_value.is_absolute()
        else (repo / candidate_program_value).resolve()
    )
    candidate_program = base_sample.load_program(candidate_program_path, repo)
    _, references = v79.load_program((repo / V79_PROGRAM).resolve(), repo)
    root = Path(program["outputs"]["recovered_ensemble_root"])
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
        contract = candidate_program["frozen_domain_execution_contracts"][domain]
        numerical = v79.validate_ensemble_pair(
            candidate,
            control,
            candidate_program["single_use_fresh_selection"][domain],
            contract["candidate_expected_attrs"],
            contract["control_expected_attrs"],
            contract["pairing"],
        )
        domains[domain] = {
            "candidate_ensemble": candidate,
            "control_ensemble": control,
            "candidate_metrics": v79.load_metrics(
                candidate_metrics_path,
                candidate,
                candidate_program["single_use_fresh_selection"][domain],
            ),
            "control_metrics": v79.load_metrics(
                control_metrics_path,
                control,
                candidate_program["single_use_fresh_selection"][domain],
            ),
            "numerical_and_pairing": numerical,
        }
        artifacts[domain] = rows
    physical_p, physical = v79.physical_energy_observation(domains, references)
    rank_p, rank = v79.rank_coverage_observation(domains)
    formula_global_p = float(
        v79.v78.global_p_value(np.asarray([physical_p]), np.asarray([rank_p]))[0]
    )
    deterministic = all(
        row["numerical_and_pairing"]["residual_DC_pass"]
        for row in domains.values()
    )
    formula_pass = bool(formula_global_p > v79.GLOBAL_ALPHA and deterministic)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete_V80DR_recovered_engineering_diagnostic_not_a_V79_gate",
        "program": str(program_path),
        "program_sha256": sha256_file(program_path),
        "program_freeze_commit": freeze_commit,
        "report_code_commit": commit,
        "worktree_clean": clean,
        "recovery_record": artifact(recovery_record_path),
        "recovery_record_decision_digest_sha256": recovery_record[
            "decision_digest_sha256"
        ],
        "parent_terminal_failure_seal_sha256": program["parent_failure"][
            "terminal_seal_sha256"
        ],
        "engineering_only": True,
        "same_previously_consumed_subset_reused": True,
        "statistically_valid_target_unseen_V79_result": False,
        "V79_gate_or_manifest_executed": False,
        "metadata_recovery_only": True,
        "only_added_attribute": {ADDED_ATTRIBUTE: ADDED_VALUE},
        "sealed_original_ensembles_modified": False,
        "sampling_repeated": False,
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
            "Audit availability and cost of genuinely independent simulation objects, freeze a prospective V81 validation design, and await explicit approval before acquiring or opening them."
            if formula_pass
            else "Stop V80 development and do not spend new independent validation objects on this candidate."
        ),
    }
    report["decision_digest_sha256"] = canonical_digest(report)
    if output_path.exists():
        raise FileExistsError("V80DR report refuses an existing output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_suffix(output_path.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(partial, output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--recovery-record", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.program, args.recovery_record, args.repo, args.out), indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
