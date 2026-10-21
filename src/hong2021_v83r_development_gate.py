#!/usr/bin/env python
"""Run the unchanged V83 consumed gate on dataset-identical recovered copies."""
from __future__ import annotations

import argparse
import copy
import json
import os
import socket
from pathlib import Path
from typing import Any

import hong2021_v83_development_gate as frozen
from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor
from hong2021_v83_contract import load_program as load_v83_program
from hong2021_v83r_metadata_recovery import (
    RECORD_SCHEMA,
    load_program,
    program_freeze_commit,
    strict_json,
)


SCHEMA = "hong2021-v83r-recovered-consumed-development-engineering-gate-v1"


def run(
    recovery_program_path: Path,
    v83_program_path: Path,
    repo: Path,
    recovery_record_path: Path,
    recovery_record_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    train_gate_path: Path,
    train_gate_sha256: str,
    ensemble_root: Path,
    metrics_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    recovery_program_path = recovery_program_path.resolve()
    recovery_program = load_program(recovery_program_path, repo)
    commit, clean = git_state(repo)
    freeze_commit = program_freeze_commit(recovery_program_path, repo)
    if (
        not clean
        or not _is_ancestor(repo, freeze_commit, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V83R gate requires clean frozen Lageunha ancestry")
    if (
        sha256_file(recovery_record_path) != recovery_record_sha256
        or recovery_record_path.resolve()
        != Path(recovery_program["outputs"]["recovery_record"]).resolve()
    ):
        raise ValueError("V83R recovery record hash or path differs")
    recovery = strict_json(recovery_record_path)
    if (
        recovery.get("schema") != RECORD_SCHEMA
        or recovery.get("status")
        != "complete_metadata_only_copy_recovery_evaluation_may_resume_once"
        or recovery.get("all_six_dataset_manifests_unchanged") is not True
        or recovery.get("sealed_originals_modified") is not False
        or recovery.get("sampling_repeated") is not False
        or canonical_digest(recovery) != recovery.get("decision_digest_sha256")
    ):
        raise ValueError("V83R recovery evidence differs")
    expected_root = Path(recovery_program["outputs"]["recovered_ensemble_root"])
    expected_metrics = Path(recovery_program["outputs"]["metrics_root"])
    expected_gate = Path(recovery_program["outputs"]["development_gate"])
    if (
        ensemble_root.resolve() != expected_root.resolve()
        or metrics_root.resolve() != expected_metrics.resolve()
        or output_path.resolve() != expected_gate.resolve()
    ):
        raise ValueError("V83R recovered output binding differs")
    for key, row in recovery["artifacts"].items():
        expected = expected_root / key / "ensemble16.h5"
        if (
            Path(row["recovered_path"]).resolve() != expected.resolve()
            or sha256_file(expected) != row["recovered_sha256"]
            or row["all_dataset_bytes_identical"] is not True
        ):
            raise ValueError(f"V83R recovered artifact differs: {key}")
    v83_program, v35, partition = load_v83_program(
        v83_program_path.resolve(), repo, commit
    )
    effective = copy.deepcopy(v83_program)
    effective["output_roots"]["development"] = str(expected_root.resolve())
    effective["output_roots"]["development_metrics"] = str(expected_metrics.resolve())
    effective["output_roots"]["development_gate"] = str(expected_gate.resolve())
    inherited_load = frozen.load_program
    frozen.load_program = lambda path, root, revision: (effective, v35, partition)
    try:
        result = frozen.gate(
            v83_program_path,
            repo,
            checkpoint_path,
            checkpoint_sha256,
            train_gate_path,
            train_gate_sha256,
            ensemble_root,
            metrics_root,
            output_path,
        )
    finally:
        frozen.load_program = inherited_load
    result.update(
        {
            "schema": SCHEMA,
            "status": (
                "pass" if result["consumed_development_engineering_pass"] else "fail"
            ),
            "recovery_program": str(recovery_program_path),
            "recovery_program_sha256": sha256_file(recovery_program_path),
            "recovery_program_freeze_commit": freeze_commit,
            "recovery_record": str(recovery_record_path.resolve()),
            "recovery_record_sha256": recovery_record_sha256,
            "all_scientific_dataset_bytes_identical_to_sealed_V83": True,
            "only_recovery_metadata_added": {"diagnostic_k_h_mpc": 1.0},
            "sampling_repeated": False,
            "statistically_independent_validation_result": False,
            "may_be_reported_as_independent_validation_pass": False,
        }
    )
    result["decision_digest_sha256"] = canonical_digest(result)
    partial = output_path.with_suffix(output_path.suffix + ".recovery.partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, output_path)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-program", type=Path, required=True)
    parser.add_argument("--v83-program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--recovery-record", type=Path, required=True)
    parser.add_argument("--recovery-record-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--train-gate", type=Path, required=True)
    parser.add_argument("--train-gate-sha256", required=True)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.recovery_program,
        args.v83_program,
        args.repo,
        args.recovery_record,
        args.recovery_record_sha256,
        args.checkpoint,
        args.checkpoint_sha256,
        args.train_gate,
        args.train_gate_sha256,
        args.ensemble_root,
        args.metrics_root,
        args.out,
    )


if __name__ == "__main__":
    main()
