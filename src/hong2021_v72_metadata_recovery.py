#!/usr/bin/env python
"""Apply the frozen metadata-only V72 stage-A evaluator recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor
from hong2021_v72_sqt import (
    ENSEMBLE_SCHEMA,
    PROGRAM_SHA256,
    load_program,
    validate_preflight,
)


RECOVERY_SCHEMA = "hong2021-v72-stage-A-evaluator-metadata-only-recovery-program-v1"
RECOVERY_STATUS = "frozen_after_sampling_before_any_metric_gate_or_stage_B_access"
RECOVERY_SHA256 = "8ad681eb58075701ed680dada751b7e2f659d9850bdede9eab4d209fbf86a0e9"
RECOVERY_FREEZE_COMMIT = "8a72cb2e18b7204b88f1fd3de9c1d7b1b3e8d3ee"
RECORD_SCHEMA = "hong2021-v72-stage-A-evaluator-metadata-only-recovery-record-v1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def _normal(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode()
    return value


def attribute_manifest(handle: h5py.File) -> dict[str, Any]:
    return {str(key): _normal(handle.attrs[key]) for key in sorted(handle.attrs)}


def dataset_manifest(handle: h5py.File) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}

    def visit(name: str, value: h5py.Dataset | h5py.Group) -> None:
        if not isinstance(value, h5py.Dataset):
            return
        digest = hashlib.sha256()
        digest.update(name.encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(str(value.dtype).encode())
        if value.shape == ():
            digest.update(np.ascontiguousarray(value[()]).tobytes())
        else:
            for index in range(value.shape[0]):
                digest.update(np.ascontiguousarray(value[index]).tobytes())
        manifest[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": digest.hexdigest(),
        }

    handle.visititems(visit)
    return manifest


def repair_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    path = path.resolve()
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"V72 recovery pre-repair hash differs: {path}")
    with h5py.File(path, "r") as handle:
        before_datasets = dataset_manifest(handle)
        before_attributes = attribute_manifest(handle)
        if (
            before_attributes.get("schema") != ENSEMBLE_SCHEMA
            or before_attributes.get("v72_program_sha256") != PROGRAM_SHA256
            or before_attributes.get("stage") != "A"
            or before_attributes.get("complete") is not True
            or "diagnostic_k_h_mpc" in before_attributes
        ):
            raise ValueError(f"V72 recovery precondition differs: {path}")
    with h5py.File(path, "r+") as handle:
        handle.attrs["diagnostic_k_h_mpc"] = np.float64(1.0)
        handle.flush()
    with h5py.File(path, "r") as handle:
        after_datasets = dataset_manifest(handle)
        after_attributes = attribute_manifest(handle)
    if before_datasets != after_datasets:
        raise RuntimeError(f"V72 recovery changed a dataset: {path}")
    expected_attributes = dict(before_attributes)
    expected_attributes["diagnostic_k_h_mpc"] = 1.0
    if after_attributes != expected_attributes:
        raise RuntimeError(f"V72 recovery changed another attribute: {path}")
    return {
        "path": str(path),
        "pre_repair_sha256": expected_sha256,
        "post_repair_sha256": sha256_file(path),
        "dataset_manifest": before_datasets,
        "pre_repair_attributes_sha256": hashlib.sha256(
            json.dumps(before_attributes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "post_repair_attributes_sha256": hashlib.sha256(
            json.dumps(after_attributes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "only_added_attribute": {"diagnostic_k_h_mpc": 1.0},
        "all_datasets_byte_identical": True,
    }


def recover(
    recovery_program_path: Path,
    v72_program_path: Path,
    repo: Path,
    out: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    if sha256_file(recovery_program_path.resolve()) != RECOVERY_SHA256:
        raise ValueError("V72 recovery program hash differs")
    recovery = _json(recovery_program_path.resolve())
    if (
        recovery.get("schema") != RECOVERY_SCHEMA
        or recovery.get("status") != RECOVERY_STATUS
    ):
        raise ValueError("V72 recovery program schema or status differs")
    program = load_program(v72_program_path.resolve(), repo)
    commit, clean = git_state(repo)
    if (
        not clean
        or not _is_ancestor(repo, RECOVERY_FREEZE_COMMIT, commit)
        or socket.gethostname().split(".")[0].lower() != "lageunha"
    ):
        raise RuntimeError("V72 recovery requires clean frozen Lageunha")
    if out.exists():
        raise FileExistsError("V72 recovery refuses an existing record")
    sequence = Path(program["output_roots"]["sequence"])
    stage_A = Path(program["output_roots"]["stage_A"])
    stage_B = Path(program["output_roots"]["stage_B"])
    status = (sequence / "status").read_text().strip()
    failure = recovery["failure_state"]
    if (
        status != failure["sequence_status"]
        or sha256_file(Path(failure["sampling_log"]))
        != failure["sampling_log_sha256"]
        or sha256_file(Path(failure["preflight"])) != failure["preflight_sha256"]
        or list(stage_A.glob("**/metrics.json"))
        or (stage_A / "decision.json").exists()
        or stage_B.exists()
    ):
        raise ValueError("V72 recovery failure-state precondition differs")
    validate_preflight(
        Path(failure["preflight"]), failure["preflight_sha256"], repo, commit
    )
    artifacts = recovery["pre_repair_artifacts"]
    repaired = {}
    for key in sorted(artifacts):
        arm, domain = key.split("/")
        path = stage_A / arm / "fresh_candidate" / domain / "ensemble16.h5"
        repaired[key] = repair_file(path, str(artifacts[key]))
    result: dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "status": "complete_metadata_only_recovery_evaluation_may_resume",
        "recovery_program": str(recovery_program_path.resolve()),
        "recovery_program_sha256": RECOVERY_SHA256,
        "v72_program": str(v72_program_path.resolve()),
        "v72_program_sha256": PROGRAM_SHA256,
        "recovery_code_commit": commit,
        "worktree_clean": clean,
        "preflight": failure["preflight"],
        "preflight_sha256": failure["preflight_sha256"],
        "artifacts": repaired,
        "all_nine_dataset_manifests_unchanged": True,
        "only_added_attribute": {"diagnostic_k_h_mpc": 1.0},
        "sampling_repeated": False,
        "metrics_created_before_recovery": 0,
        "stage_A_decision_created_before_recovery": False,
        "stage_B_accessed": False,
        "candidate_strata_seed_member_sampler_metric_or_threshold_changed": False,
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
    result = recover(
        args.recovery_program, args.v72_program, args.repo, args.out
    )
    partial = args.out.with_suffix(args.out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
