"""Fail-closed loader for the sealed grammar-debug v2 overlap diagnostics.

The public entry point remains deliberately disabled.  The private implementation
exists so that its filesystem and provenance contracts can be tested without
reading the canonical GPFS artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_read_only_analysis import (
    _analyze_arrays,
)


REPOSITORY_ROOT = Path("/home/kjhan/BACKUP/CF4")
DESIGN_PATH = REPOSITORY_ROOT / (
    "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_"
    "grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_wrapper_"
    "design_v1.json"
)
DESIGN_COMMIT = "c6c45c69565c48dffbcd0f425466cffb48ac5a4e"
DESIGN_SHA256 = "6bec76f28dfbbf7eb295eb3e62716a47d925f021d5bb71974a9521c0b08ac73c"
EXPECTED_SCIENCE_STATUS = "complete_scientific_fail_production_smc"
EXPECTED_FAILURE_CLASS = "paired_incoherence"
EXPECTED_MASTER_SEEDS = (2026082301, 2026082302, 2026082303, 2026082304)
GIT_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}
GIT_COMMANDS = (
    ("/usr/bin/git", "rev-parse", "HEAD"),
    ("/usr/bin/git", "rev-parse", "@{upstream}"),
    (
        "/usr/bin/git",
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)scripts/tripwire/**",
    ),
)
ROOT_NAMES = ("data", "cache", "state", "receipt")


class CanonicalReadContractError(RuntimeError):
    """Raised when immutable input or provenance validation fails."""


@dataclass(frozen=True)
class FileIdentity:
    dev: int
    ino: int
    size: int
    mode: int


@dataclass(frozen=True)
class LoadedArtifact:
    identity: FileIdentity
    sha256: str
    value: Any


def run_canonical_parent_key_overlap_read_only_analysis_wrapper_v1() -> None:
    """Refuse before Git, canonical paths, or sealed artifacts are accessed."""

    raise PermissionError(
        "canonical parent/key overlap artifact reads and analysis are not authorized"
    )


def _identity(value: os.stat_result) -> FileIdentity:
    return FileIdentity(
        dev=int(value.st_dev),
        ino=int(value.st_ino),
        size=int(value.st_size),
        mode=int(value.st_mode),
    )


def _permission_mode(value: os.stat_result) -> str:
    return f"{stat.S_IMODE(value.st_mode):04o}"


def _reject_symlink_components(path: Path) -> None:
    path = Path(path)
    if not path.is_absolute():
        raise CanonicalReadContractError("validated path is not absolute")
    current = Path(path.anchor)
    try:
        root_stat = os.lstat(current)
        if stat.S_ISLNK(root_stat.st_mode):
            raise CanonicalReadContractError("path root is a symlink")
        for part in path.parts[1:]:
            current /= part
            value = os.lstat(current)
            if stat.S_ISLNK(value.st_mode):
                raise CanonicalReadContractError(
                    f"symlink path component is forbidden: {current}"
                )
    except FileNotFoundError as error:
        raise CanonicalReadContractError(f"validated path is absent: {current}") from error


def _object_without_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalReadContractError("JSON contains a duplicate object key")
        result[key] = value
    return result


def _parse_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload, object_pairs_hook=_object_without_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanonicalReadContractError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise CanonicalReadContractError(f"{label} JSON root is not an object")
    return value


def _read_fd_and_hash(descriptor: int, *, retain_bytes: bool) -> tuple[str, bytes | None]:
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if retain_bytes else None
    while True:
        block = os.read(descriptor, 8 << 20)
        if not block:
            break
        digest.update(block)
        if chunks is not None:
            chunks.append(block)
    return digest.hexdigest(), b"".join(chunks) if chunks is not None else None


def _required_npz_arrays(role: str) -> tuple[str, ...]:
    if role == "immutable_evidence_cache_shard":
        return ("keys", "log_Z", "log_Z_bar")
    if role.startswith("replicate_"):
        return ("master_seed", "keys", "weights", "log_Z_bar")
    if role == "terminal_parent_posterior":
        return ("master_seed", "parent_seed", "log_I_bar", "P_rep", "P_pool")
    return ()


def _materialize_npz_from_snapshot_bytes(
    payload: bytes,
    *,
    role: str,
) -> Mapping[str, np.ndarray]:
    required = _required_npz_arrays(role)
    stream = io.BytesIO(payload)
    archive: Any = None
    try:
        archive = np.load(stream, allow_pickle=False)
        available = tuple(archive.files)
        if any(name not in available for name in required):
            raise CanonicalReadContractError(f"{role} lacks a required NPZ array")
        arrays = {name: np.array(archive[name], copy=True) for name in required}
        for value in arrays.values():
            value.setflags(write=False)
        return MappingProxyType(arrays)
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, CanonicalReadContractError):
            raise
        raise CanonicalReadContractError(f"{role} is an invalid NPZ archive") from error
    finally:
        if archive is not None:
            archive.close()
        stream.close()


def _stable_artifact_read(
    row: Mapping[str, Any],
    *,
    materialize: bool,
) -> LoadedArtifact:
    path = Path(str(row["path"]))
    _reject_symlink_components(path)
    try:
        initial_stat = os.lstat(path)
    except OSError as error:
        raise CanonicalReadContractError(f"artifact lstat failed: {path}") from error
    if not stat.S_ISREG(initial_stat.st_mode):
        raise CanonicalReadContractError(f"artifact is not a regular file: {path}")
    if _permission_mode(initial_stat) != row["mode"]:
        raise CanonicalReadContractError(f"artifact mode changed: {path}")
    if initial_stat.st_size != row["size_bytes"]:
        raise CanonicalReadContractError(f"artifact size changed: {path}")

    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if _identity(initial_stat) != _identity(before):
            raise CanonicalReadContractError(f"artifact identity changed before read: {path}")
        retain = row["kind"] in {"JSON", "NPZ"}
        digest, payload = _read_fd_and_hash(descriptor, retain_bytes=retain)
        if digest != row["sha256"]:
            raise CanonicalReadContractError(f"artifact SHA changed: {path}")
        value: Any = None
        if row["kind"] == "JSON":
            if payload is None:
                raise CanonicalReadContractError("JSON payload was not retained")
            value = _parse_json_bytes(payload, str(path))
        elif row["kind"] == "NPZ":
            if payload is None:
                raise CanonicalReadContractError("NPZ payload was not retained")
            value = _materialize_npz_from_snapshot_bytes(
                payload,
                role=str(row["role"]) if materialize else "postvalidation_no_arrays",
            )
        elif row["kind"] != "LOG":
            raise CanonicalReadContractError(f"artifact kind changed: {path}")
        after = os.fstat(descriptor)
        terminal = os.lstat(path)
    except OSError as error:
        raise CanonicalReadContractError(f"stable artifact read failed: {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if _identity(before) != _identity(after) or _identity(after) != _identity(terminal):
        raise CanonicalReadContractError(f"artifact identity was unstable: {path}")
    return LoadedArtifact(identity=_identity(before), sha256=digest, value=value)


def _root_paths(contract: Mapping[str, Any]) -> Mapping[str, Path]:
    paths = contract["canonical_paths"]
    return MappingProxyType(
        {
            "data": Path(paths["data_root"]),
            "cache": Path(paths["cache_root"]),
            "state": Path(paths["state_root"]),
            "receipt": Path(paths["receipt_root"]),
        }
    )


def _validate_root_inventory(contract: Mapping[str, Any]) -> None:
    artifact = contract["artifact_contract"]
    roots = _root_paths(contract)
    for name in ROOT_NAMES:
        root = roots[name]
        _reject_symlink_components(root)
        value = os.lstat(root)
        if not stat.S_ISDIR(value.st_mode):
            raise CanonicalReadContractError(f"{name} root is not a directory")
        if _permission_mode(value) != artifact["terminal_root_modes"][name]:
            raise CanonicalReadContractError(f"{name} root mode changed")
        expected = set(artifact["exact_root_entry_sets"][name])
        with os.scandir(root) as entries:
            actual_entries = list(entries)
        if {entry.name for entry in actual_entries} != expected:
            raise CanonicalReadContractError(f"{name} root entry set changed")
        if any(entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in actual_entries):
            raise CanonicalReadContractError(f"{name} root contains a non-regular entry")


def _run_whitelisted_git(
    argv: Sequence[str],
    *,
    cwd: Path,
    runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> bytes:
    command = tuple(argv)
    if command not in GIT_COMMANDS:
        raise CanonicalReadContractError("non-whitelisted subprocess command refused")
    try:
        completed = runner(
            list(command),
            cwd=str(cwd),
            env=dict(GIT_ENVIRONMENT),
            shell=False,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise CanonicalReadContractError("whitelisted Git command failed") from error
    if not isinstance(completed.stdout, bytes):
        raise CanonicalReadContractError("Git output is not bytes")
    return completed.stdout


def _validate_git(
    contract: Mapping[str, Any],
    *,
    repo_root: Path,
    runner: Callable[..., subprocess.CompletedProcess[bytes]],
) -> None:
    expected = contract["git_subprocess_contract"]["expected_HEAD_and_tracking"].encode()
    head = _run_whitelisted_git(GIT_COMMANDS[0], cwd=repo_root, runner=runner).strip()
    tracking = _run_whitelisted_git(GIT_COMMANDS[1], cwd=repo_root, runner=runner).strip()
    status_value = _run_whitelisted_git(GIT_COMMANDS[2], cwd=repo_root, runner=runner)
    if head != expected:
        raise CanonicalReadContractError("Git HEAD differs from the frozen commit")
    if tracking != expected:
        raise CanonicalReadContractError("Git tracking ref differs from the frozen commit")
    if status_value != b"":
        raise CanonicalReadContractError("Git science scope is dirty")


def _verify_code_pins(contract: Mapping[str, Any], *, repo_root: Path) -> None:
    for label, row in contract["immutable_code_contracts"].items():
        relative = Path(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise CanonicalReadContractError(f"{label} code pin path is noncanonical")
        path = repo_root / relative
        synthetic = {
            "path": str(path),
            "mode": row["mode"],
            "size_bytes": path.lstat().st_size,
            "sha256": row["sha256"],
            "kind": "LOG",
            "role": label,
        }
        _stable_artifact_read(synthetic, materialize=False)


def _semantic_bindings(
    loaded: Mapping[str, LoadedArtifact],
    contract: Mapping[str, Any],
) -> None:
    result = loaded["scientific_result"].value
    manifest = loaded["immutable_top_manifest"].value
    cf4 = loaded["post_terminal_CF4_gates"].value
    state = loaded["state_complete_marker"].value
    receipt = loaded["receipt_complete_marker"].value
    manifest_sha = next(
        row["sha256"]
        for row in contract["artifact_contract"]["exact_17_rows"]
        if row["role"] == "immutable_top_manifest"
    )
    expected_common = {
        "status": EXPECTED_SCIENCE_STATUS,
        "failure_class": EXPECTED_FAILURE_CLASS,
    }
    if (
        result.get("status") != expected_common["status"]
        or result.get("outcome_kind") != "scientific_fail"
        or result.get("failure_class") != expected_common["failure_class"]
        or manifest.get("science_status") != expected_common["status"]
        or manifest.get("outcome_kind") != "scientific_fail"
        or manifest.get("failure_class") != expected_common["failure_class"]
        or cf4.get("status") != expected_common["status"]
        or cf4.get("failure_class") != expected_common["failure_class"]
        or manifest.get("failed_channels") != result.get("failed_channels")
        or cf4.get("pre_CF4_metrics") != result.get("pre_CF4_metrics")
        or cf4.get("CF4_metrics") != result.get("CF4_metrics")
        or cf4.get("gates") != result.get("CF4_gates")
        or state.get("status") != "complete_valid_provenance_and_scientific_postcheck"
        or receipt.get("status") != "complete_valid_provenance_and_scientific_postcheck"
        or state.get("science_status") != EXPECTED_SCIENCE_STATUS
        or state.get("result_manifest_sha256_or_null") != manifest_sha
        or receipt.get("result_manifest_sha256_or_null") != manifest_sha
    ):
        raise CanonicalReadContractError("sealed JSON semantic binding changed")


def _analysis_arguments(loaded: Mapping[str, LoadedArtifact]) -> dict[str, Any]:
    replicates = [loaded[f"replicate_{index}"].value for index in range(4)]
    for expected, value in zip(EXPECTED_MASTER_SEEDS, replicates, strict=True):
        seed = value["master_seed"]
        if seed.dtype != np.dtype("int64") or seed.shape != () or int(seed) != expected:
            raise CanonicalReadContractError("replicate master seed changed")
    terminal = loaded["terminal_parent_posterior"].value
    terminal_seed = terminal["master_seed"]
    if (
        terminal_seed.dtype != np.dtype("int64")
        or terminal_seed.shape != (4,)
        or not np.array_equal(terminal_seed, np.asarray(EXPECTED_MASTER_SEEDS, dtype=np.int64))
    ):
        raise CanonicalReadContractError("terminal master seeds changed")
    cache = loaded["immutable_evidence_cache_shard"].value
    return {
        "replicate_keys": [value["keys"] for value in replicates],
        "replicate_weights": [value["weights"] for value in replicates],
        "cache_keys": cache["keys"],
        "cache_log_z": cache["log_Z"],
        "cache_log_z_bar": cache["log_Z_bar"],
        "replicate_log_z_bar": [value["log_Z_bar"] for value in replicates],
        "stored_p_rep": terminal["P_rep"],
        "log_i_bar": terminal["log_I_bar"],
        "stored_p_pool": terminal["P_pool"],
        "parent_seed": terminal["parent_seed"],
    }


def _run_verified_analysis_for_test(
    contract: Mapping[str, Any],
    *,
    repo_root: Path,
    git_runner: Callable[..., subprocess.CompletedProcess[bytes]],
    analyzer: Callable[..., Mapping[str, Any]] = _analyze_arrays,
) -> Mapping[str, Any]:
    """Exercise the future canonical flow using only injected test fixtures."""

    _validate_git(contract, repo_root=repo_root, runner=git_runner)
    _verify_code_pins(contract, repo_root=repo_root)
    _validate_root_inventory(contract)
    loaded: dict[str, LoadedArtifact] = {}
    before: dict[str, FileIdentity] = {}
    for row in contract["artifact_contract"]["exact_17_rows"]:
        role = str(row["role"])
        if role in loaded:
            raise CanonicalReadContractError("duplicate artifact role")
        item = _stable_artifact_read(row, materialize=True)
        loaded[role] = item
        before[str(row["path"])] = item.identity
    _semantic_bindings(loaded, contract)
    result = analyzer(**_analysis_arguments(loaded))

    _validate_root_inventory(contract)
    for row in contract["artifact_contract"]["exact_17_rows"]:
        item = _stable_artifact_read(row, materialize=False)
        if item.identity != before[str(row["path"])]:
            raise CanonicalReadContractError("artifact inventory changed after analysis")
    return result
