#!/usr/bin/env python
"""Copy and metadata-repair the sealed V80D ensembles without resampling."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_v15_development_gate import canonical_digest, git_state
from hong2021_v18_init import sha256_file
from hong2021_v63_train import _is_ancestor
from hong2021_v80_sample import ENSEMBLE_SCHEMA


PROGRAM_SCHEMA = "hong2021-v80dr-metadata-only-recovery-program-v1"
PROGRAM_STATUS = "frozen_before_copy_or_metadata_mutation_or_evaluator_recovery"
RECORD_SCHEMA = "hong2021-v80dr-metadata-only-recovery-record-v1"
PARENT_SEAL = Path("config/hong2021_v80d_terminal_failure_seal.json")
PARENT_SEAL_SHA256 = "8d205293101a5c49d00527fef16cdf7208c6ddd5baae4254da15ff6296088e3c"
V80D_PROGRAM_SHA256 = "318a01d4b28e2950624af0835836feaf9884db78a6a098b3571b114312587fc6"
DOMAIN_KEYS = ("tng", "simba_dev", "swift_dev")
ARMS = ("candidate", "control")
ADDED_ATTRIBUTE = "diagnostic_k_h_mpc"
ADDED_VALUE = 1.0


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
        raise ValueError("V80DR program freeze commit cannot be resolved")
    return commit


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    program = strict_json(path.resolve())
    authorization = program.get("authorization", {})
    mutation = program.get("only_authorized_mutation", {})
    if (
        program.get("schema") != PROGRAM_SCHEMA
        or program.get("status") != PROGRAM_STATUS
        or program.get("engineering_only") is not True
        or program.get("statistically_valid_V79_reexecution") is not False
        or authorization.get("user_approved_metadata_only_recovery") is not True
        or authorization.get("resampling") is not False
        or authorization.get("V79_gate_or_manifest") is not False
        or mutation.get("attribute") != {ADDED_ATTRIBUTE: ADDED_VALUE}
        or mutation.get("modify_original_ensembles") is not False
    ):
        raise ValueError("V80DR recovery boundary differs")
    parent = (repo / PARENT_SEAL).resolve()
    if (
        sha256_file(parent) != PARENT_SEAL_SHA256
        or program["parent_failure"]["terminal_seal_sha256"]
        != PARENT_SEAL_SHA256
    ):
        raise ValueError("V80DR parent terminal seal differs")
    for label, row in program["implementation_sources"].items():
        source = resolve_path(repo, str(row["path"]))
        if sha256_file(source) != row["sha256"]:
            raise ValueError(f"V80DR implementation source differs: {label}")
    return program


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
        if value.shape == ():
            digest.update(np.ascontiguousarray(value[()]).tobytes())
        else:
            for index in range(value.shape[0]):
                digest.update(np.ascontiguousarray(value[index]).tobytes(order="C"))
        manifest[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "raw_C_order_bytes_sha256": digest.hexdigest(),
        }

    handle.visititems(visit)
    return manifest


def manifest_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def copy_and_repair(
    source: Path,
    temporary_target: Path,
    final_target: Path,
    expected_source_sha256: str,
) -> dict[str, Any]:
    source = source.resolve()
    temporary_target = temporary_target.resolve()
    final_target = final_target.resolve()
    if sha256_file(source) != expected_source_sha256:
        raise ValueError(f"V80DR source ensemble hash differs: {source}")
    with h5py.File(source, "r") as handle:
        before_datasets = dataset_manifest(handle)
        before_attributes = attribute_manifest(handle)
    if (
        before_attributes.get("schema") != ENSEMBLE_SCHEMA
        or before_attributes.get("candidate_program_sha256") != V80D_PROGRAM_SHA256
        or before_attributes.get("complete") is not True
        or ADDED_ATTRIBUTE in before_attributes
    ):
        raise ValueError(f"V80DR source precondition differs: {source}")
    if temporary_target.exists() or final_target.exists():
        raise FileExistsError(f"V80DR target exists: {final_target}")
    temporary_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, temporary_target)
    if sha256_file(temporary_target) != expected_source_sha256:
        raise RuntimeError(f"V80DR exact pre-repair copy differs: {source}")
    with h5py.File(temporary_target, "r+") as handle:
        handle.attrs[ADDED_ATTRIBUTE] = np.float64(ADDED_VALUE)
        handle.flush()
    with h5py.File(temporary_target, "r") as handle:
        after_datasets = dataset_manifest(handle)
        after_attributes = attribute_manifest(handle)
    if before_datasets != after_datasets:
        raise RuntimeError(f"V80DR metadata repair changed a dataset: {source}")
    expected_attributes = dict(before_attributes)
    expected_attributes[ADDED_ATTRIBUTE] = ADDED_VALUE
    if after_attributes != expected_attributes:
        raise RuntimeError(f"V80DR metadata repair changed another attribute: {source}")
    if sha256_file(source) != expected_source_sha256:
        raise RuntimeError(f"V80DR metadata repair changed the sealed source: {source}")
    return {
        "sealed_source_path": str(source),
        "sealed_source_sha256": expected_source_sha256,
        "recovered_path": str(final_target),
        "recovered_sha256": sha256_file(temporary_target),
        "dataset_manifest": before_datasets,
        "dataset_manifest_sha256": manifest_sha256(before_datasets),
        "pre_recovery_attributes": before_attributes,
        "pre_recovery_attributes_sha256": manifest_sha256(before_attributes),
        "post_recovery_attributes": after_attributes,
        "post_recovery_attributes_sha256": manifest_sha256(after_attributes),
        "only_added_attribute": {ADDED_ATTRIBUTE: ADDED_VALUE},
        "all_dataset_bytes_identical": True,
        "sealed_source_unchanged": True,
    }


def recover(program_path: Path, repo: Path, out: Path) -> dict[str, Any]:
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
        raise RuntimeError("V80DR recovery requires clean frozen Lageunha ancestry")
    if out.exists():
        raise FileExistsError("V80DR recovery refuses an existing record")
    evidence = program["frozen_failure_state"]
    sequence = Path(evidence["sequence_root"])
    if (
        (sequence / "status").read_text().strip() != evidence["status"]
        or sha256_file(sequence / "status") != evidence["status_sha256"]
        or sha256_file(sequence / "sealed_result.json")
        != evidence["sealed_result_sha256"]
        or list(Path(evidence["original_ensemble_root"]).glob("**/metrics.json"))
        or Path(evidence["original_report"]).exists()
    ):
        raise ValueError("V80DR frozen failure-state precondition differs")
    target_root = Path(program["outputs"]["recovered_ensemble_root"])
    partial_root = target_root.with_name(target_root.name + ".partial")
    if target_root.exists() or partial_root.exists():
        raise FileExistsError("V80DR recovered ensemble output exists")
    source_root = Path(evidence["original_ensemble_root"])
    artifacts: dict[str, Any] = {}
    try:
        for arm in ARMS:
            for domain in DOMAIN_KEYS:
                key = f"{arm}/{domain}"
                source = source_root / arm / domain / "ensemble16.h5"
                temporary = partial_root / arm / domain / "ensemble16.h5"
                final = target_root / arm / domain / "ensemble16.h5"
                artifacts[key] = copy_and_repair(
                    source,
                    temporary,
                    final,
                    str(program["sealed_source_ensembles"][key]["sha256"]),
                )
        os.replace(partial_root, target_root)
    except Exception:
        # Preserve any partial recovery evidence for audit; never delete or retry it.
        raise
    for key, row in artifacts.items():
        final = Path(row["recovered_path"])
        if sha256_file(final) != row["recovered_sha256"]:
            raise RuntimeError(f"V80DR recovered file changed after rename: {key}")
    result: dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "status": "complete_metadata_only_copy_recovery_evaluation_may_run_once",
        "program": str(program_path),
        "program_sha256": sha256_file(program_path),
        "program_freeze_commit": freeze_commit,
        "recovery_code_commit": commit,
        "worktree_clean": clean,
        "parent_terminal_seal_sha256": PARENT_SEAL_SHA256,
        "engineering_only": True,
        "statistically_valid_V79_reexecution": False,
        "artifacts": artifacts,
        "all_six_dataset_manifests_unchanged": True,
        "only_added_attribute": {ADDED_ATTRIBUTE: ADDED_VALUE},
        "sealed_originals_modified": False,
        "sampling_repeated": False,
        "metrics_created_before_recovery": 0,
        "V79_manifest_or_gate_executed": False,
        "V72_stage_B_accessed": False,
        "Astrid_or_EAGLE_accessed": False,
        "RAMSES_modified_or_stopped": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(partial, out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.program, args.repo, args.out), indent=2), flush=True)


if __name__ == "__main__":
    main()
