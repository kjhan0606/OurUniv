#!/usr/bin/env python
"""Authorize the frozen V72 stage-A gate label-only recovery."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from typing import Any

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor
from hong2021_v72_gate import EVALUATION_SCHEMA, _load_sqt_metrics
from hong2021_v72_sqt import PROGRAM_SHA256, load_program, validate_preflight


RECOVERY_SCHEMA = "hong2021-v72-stage-A-gate-label-only-recovery-program-v1"
RECOVERY_STATUS = "frozen_after_all_stage_A_metrics_before_any_decision_or_stage_B_access"
RECOVERY_SHA256 = "cbbcc1d73e6f01e2e66772e55af31181b6e4355aed5974e8f9aa837f4ff34352"
RECOVERY_FREEZE_COMMIT = "7aca17acd0406c7ded9b293ef6e8adcc7273f486"
RECORD_SCHEMA = "hong2021-v72-stage-A-gate-label-only-recovery-record-v1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def authorize(
    recovery_program_path: Path,
    v72_program_path: Path,
    repo: Path,
    out: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    recovery_program_path = recovery_program_path.resolve()
    if sha256_file(recovery_program_path) != RECOVERY_SHA256:
        raise ValueError("V72 gate-label recovery program hash differs")
    recovery = _json(recovery_program_path)
    if (
        recovery.get("schema") != RECOVERY_SCHEMA
        or recovery.get("status") != RECOVERY_STATUS
    ):
        raise ValueError("V72 gate-label recovery schema or status differs")
    program = load_program(v72_program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, RECOVERY_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V72 gate-label recovery requires clean frozen Lageunha")
    if out.exists():
        raise FileExistsError("V72 gate-label recovery refuses an existing record")

    failure = recovery["failure_state"]
    sequence = Path(program["output_roots"]["sequence"])
    stage_A = Path(program["output_roots"]["stage_A"])
    stage_B = Path(program["output_roots"]["stage_B"])
    seal = Path(program["output_roots"]["terminal_seal"])
    status_path = sequence / "status"
    if (
        status_path.read_text().strip() != failure["sequence_status"]
        or sha256_file(status_path) != failure["sequence_status_sha256"]
        or sha256_file(Path(failure["preflight"])) != failure["preflight_sha256"]
        or sha256_file(Path(failure["metadata_recovery_record"]))
        != failure["metadata_recovery_record_sha256"]
        or sha256_file(Path(failure["decision_log"]))
        != failure["decision_log_sha256"]
        or (stage_A / "decision.json").exists()
        or stage_B.exists()
        or seal.exists()
    ):
        raise ValueError("V72 gate-label failure-state precondition differs")
    validate_preflight(
        Path(failure["preflight"]), failure["preflight_sha256"], repo, commit
    )

    verified: dict[str, Any] = {}
    for key, expected in recovery["stage_A_artifacts"].items():
        arm, domain = key.split("/")
        base = stage_A / arm / "fresh_candidate" / domain
        ensemble = base / "ensemble16.h5"
        metrics = base / "ensemble_evaluation" / "metrics.json"
        if (
            sha256_file(ensemble) != expected["ensemble_sha256"]
            or sha256_file(metrics) != expected["metrics_sha256"]
        ):
            raise ValueError(f"V72 gate-label artifact hash differs: {key}")
        row = _load_sqt_metrics(metrics)
        if Path(row["path"]).resolve() != ensemble.resolve():
            raise ValueError(f"V72 gate-label metrics path differs: {key}")
        verified[key] = {
            "ensemble": str(ensemble.resolve()),
            "ensemble_sha256": expected["ensemble_sha256"],
            "metrics": str(metrics.resolve()),
            "metrics_sha256": expected["metrics_sha256"],
            "evaluation_schema": EVALUATION_SCHEMA,
            "sole_candidate_key": "sqt",
        }
    if len(verified) != 9:
        raise ValueError("V72 gate-label recovery requires exactly nine artifacts")

    result: dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "status": "complete_gate_label_only_recovery_gate_may_run_once",
        "recovery_program": str(recovery_program_path),
        "recovery_program_sha256": RECOVERY_SHA256,
        "v72_program": str(v72_program_path.resolve()),
        "v72_program_sha256": PROGRAM_SHA256,
        "recovery_code_commit": commit,
        "worktree_clean": clean,
        "preflight": failure["preflight"],
        "preflight_sha256": failure["preflight_sha256"],
        "metadata_recovery_record": failure["metadata_recovery_record"],
        "metadata_recovery_record_sha256": failure[
            "metadata_recovery_record_sha256"
        ],
        "artifacts": verified,
        "all_nine_ensembles_and_metrics_hash_verified": True,
        "metrics_recomputed": False,
        "stage_A_resampled_or_reevaluated": False,
        "stage_A_decision_created_before_authorization": False,
        "stage_B_accessed": False,
        "scientific_candidate_statistic_threshold_or_decision_rule_changed": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
        "independent_EAGLE_accessed": False,
        "independent_gate_locked": True,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-program", type=Path, required=True)
    parser.add_argument("--v72-program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = authorize(
        args.recovery_program, args.v72_program, args.repo, args.out
    )
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
