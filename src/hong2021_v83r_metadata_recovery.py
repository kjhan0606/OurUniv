#!/usr/bin/env python
"""Copy-repair V83 ensembles while proving all scientific datasets unchanged."""
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
from hong2021_v83_sample import SCHEMA as ENSEMBLE_SCHEMA


PROGRAM_SCHEMA = "hong2021-v83r-metadata-only-copy-recovery-program-v1"
PROGRAM_STATUS = "frozen_before_copy_recovery_evaluation_or_gate"
RECORD_SCHEMA = "hong2021-v83r-metadata-only-copy-recovery-record-v1"
V83_PROGRAM = Path("config/hong2021_v83_conditional_marginal_spline_program.json")
V83_PROGRAM_SHA256 = "035e52b3d7059816b61dbf2b23e0cca9f5c5592903f704f0103a913556cea174"
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
        raise ValueError("V83R program freeze commit cannot be resolved")
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
        or program.get("statistically_independent") is not False
        or authorization.get("user_approved_next_metadata_recovery_action") is not True
        or authorization.get("resampling") is not False
        or authorization.get("independent_validation") is not False
        or mutation.get("attribute") != {ADDED_ATTRIBUTE: ADDED_VALUE}
        or mutation.get("modify_original_ensembles") is not False
    ):
        raise ValueError("V83R recovery boundary differs")
    if (
        sha256_file(repo / V83_PROGRAM) != V83_PROGRAM_SHA256
        or program["parent_failure"]["v83_program_sha256"] != V83_PROGRAM_SHA256
    ):
        raise ValueError("V83R frozen V83 program differs")
    for label, row in program["implementation_sources"].items():
        if sha256_file(resolve_path(repo, row["path"])) != row["sha256"]:
            raise ValueError(f"V83R implementation differs: {label}")
    failure = program["frozen_failure_state"]
    status = Path(failure["status_path"])
    if (
        status.read_text().strip() != failure["status"]
        or sha256_file(status) != failure["status_sha256"]
    ):
        raise ValueError("V83R failure status differs")
    for label, row in failure["artifacts"].items():
        artifact = Path(row["path"])
        if sha256_file(artifact) != row["sha256"]:
            raise ValueError(f"V83R frozen failure artifact differs: {label}")
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
    expected_source_bytes: int,
) -> dict[str, Any]:
    source = source.resolve()
    temporary_target = temporary_target.resolve()
    final_target = final_target.resolve()
    if (
        source.stat().st_size != expected_source_bytes
        or sha256_file(source) != expected_source_sha256
    ):
        raise ValueError(f"V83R source ensemble bytes or hash differs: {source}")
    with h5py.File(source, "r") as handle:
        before_datasets = dataset_manifest(handle)
        before_attributes = attribute_manifest(handle)
    if (
        before_attributes.get("schema") != ENSEMBLE_SCHEMA
        or before_attributes.get("program_sha256") != V83_PROGRAM_SHA256
        or before_attributes.get("complete") is not True
        or ADDED_ATTRIBUTE in before_attributes
    ):
        raise ValueError(f"V83R source precondition differs: {source}")
    if temporary_target.exists() or final_target.exists():
        raise FileExistsError(f"V83R target exists: {final_target}")
    temporary_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, temporary_target)
    if sha256_file(temporary_target) != expected_source_sha256:
        raise RuntimeError(f"V83R exact pre-repair copy differs: {source}")
    with h5py.File(temporary_target, "r+") as handle:
        handle.attrs[ADDED_ATTRIBUTE] = np.float64(ADDED_VALUE)
        handle.flush()
    with h5py.File(temporary_target, "r") as handle:
        after_datasets = dataset_manifest(handle)
        after_attributes = attribute_manifest(handle)
    if before_datasets != after_datasets:
        raise RuntimeError(f"V83R recovery changed a dataset: {source}")
    expected_attributes = dict(before_attributes)
    expected_attributes[ADDED_ATTRIBUTE] = ADDED_VALUE
    if after_attributes != expected_attributes:
        raise RuntimeError(f"V83R recovery changed another attribute: {source}")
    if (
        source.stat().st_size != expected_source_bytes
        or sha256_file(source) != expected_source_sha256
    ):
        raise RuntimeError(f"V83R recovery changed the sealed source: {source}")
    return {
        "sealed_source_path": str(source),
        "sealed_source_bytes": expected_source_bytes,
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
        raise RuntimeError("V83R recovery requires clean frozen Lageunha ancestry")
    if out.resolve() != Path(program["outputs"]["recovery_record"]).resolve():
        raise ValueError("V83R recovery record output differs")
    if out.exists():
        raise FileExistsError("V83R recovery record exists")
    source_root = Path(program["frozen_failure_state"]["original_ensemble_root"])
    target_root = Path(program["outputs"]["recovered_ensemble_root"])
    partial_root = target_root.with_name(target_root.name + ".partial")
    if target_root.exists() or partial_root.exists():
        raise FileExistsError("V83R recovered ensemble root exists")
    expected_keys = {f"{arm}/{domain}" for arm in ARMS for domain in DOMAIN_KEYS}
    if set(program["sealed_source_ensembles"]) != expected_keys:
        raise ValueError("V83R sealed source set differs")
    artifacts: dict[str, Any] = {}
    for arm in ARMS:
        for domain in DOMAIN_KEYS:
            key = f"{arm}/{domain}"
            source = source_root / arm / domain / "ensemble16.h5"
            temporary = partial_root / arm / domain / "ensemble16.h5"
            final = target_root / arm / domain / "ensemble16.h5"
            binding = program["sealed_source_ensembles"][key]
            if Path(binding["path"]).resolve() != source.resolve():
                raise ValueError(f"V83R source path differs: {key}")
            artifacts[key] = copy_and_repair(
                source,
                temporary,
                final,
                binding["sha256"],
                int(binding["bytes"]),
            )
    os.replace(partial_root, target_root)
    for row in artifacts.values():
        if sha256_file(Path(row["recovered_path"])) != row["recovered_sha256"]:
            raise RuntimeError("V83R recovered artifact changed after rename")
    result: dict[str, Any] = {
        "schema": RECORD_SCHEMA,
        "status": "complete_metadata_only_copy_recovery_evaluation_may_resume_once",
        "program": str(program_path),
        "program_sha256": sha256_file(program_path),
        "program_freeze_commit": freeze_commit,
        "recovery_code_commit": commit,
        "worktree_clean": clean,
        "engineering_only": True,
        "statistically_independent": False,
        "artifacts": artifacts,
        "all_six_dataset_manifests_unchanged": True,
        "only_added_attribute": {ADDED_ATTRIBUTE: ADDED_VALUE},
        "sealed_originals_modified": False,
        "sampling_repeated": False,
        "independent_validation_accessed": False,
        "RAMSES_modified_or_stopped": False,
    }
    result["decision_digest_sha256"] = canonical_digest(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    os.replace(partial, out)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    recover(args.program, args.repo, args.out)


if __name__ == "__main__":
    main()
