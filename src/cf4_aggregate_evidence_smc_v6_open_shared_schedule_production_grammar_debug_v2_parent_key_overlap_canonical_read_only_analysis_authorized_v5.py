"""One-shot authorization boundary for the canonical overlap analysis.

This module has no caller-selected paths or science arguments.  It reads only
fixed authorization metadata until the program, grant, Git lineage, external
pair, host, and Slurm allocation have all passed.  The sealed science artifacts
are then consumed once, in memory, by the separately audited loader.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


ROOT = Path("/home/kjhan/BACKUP/CF4")
PROGRAM = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_program_v5.json"
GRANT = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_execution_grant_v5.json"
RELEASE = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_execution_release_v5.json")
MANIFEST = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_execution_manifest_v5.json")
RECEIPT_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_receipts_v5")

AUTHORIZATION_DESIGN = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_execution_authorization_design_v1.json"
REMEDIATION_DESIGN = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_wrapper_remediation_design_v4.json"
PAIR_DESIGN = REMEDIATION_DESIGN
WRAPPER_DESIGN = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_wrapper_design_v1.json"
WRAPPER_ERRATUM = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_wrapper_design_erratum_v1.json"
LOADER_RESULT_RECORD = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_wrapper_implementation_result_record_v1.json"

AUTHORIZATION_DESIGN_SHA = "493b0c2654ef115f9e4187a6d3e65cef09b6ce3ef7cb59dacf83cae4e61f9b89"
REMEDIATION_DESIGN_SHA = "5cc6db51353cef7085c0265a9d62b1c2664ff6e40d170a9691321e6bd6503d60"
REMEDIATION_DESIGN_COMMIT = "4622e0eb51e63caa3eaa06ffc026429faeafe2be"
PAIR_DESIGN_SHA = REMEDIATION_DESIGN_SHA
PAIR_DESIGN_COMMIT = REMEDIATION_DESIGN_COMMIT
WRAPPER_DESIGN_SHA = "f6cdf7e6818e20052adfdbacd16b7e4333104fd8de0675ebdd5ba058719d3d80"
WRAPPER_DESIGN_COMMIT = "6f3083dcd9dba9908e7e80304d8d53e751b6b56b"
WRAPPER_ERRATUM_SHA = "be58546a6742320881d179cd51b8943468877c45fe12d060acd61a277b926943"
WRAPPER_ERRATUM_COMMIT = "292f6cfb6f42bdd95a4d44f812d3ad8f222bd9d1"
LOADER_RESULT_RECORD_SHA = "74b076a5b8a2be53e47a15302450c7b4f43eb64a59551369f4339d29c89a4fa3"

LOADER_MODULE = "cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_wrapper_v1"
LOADER_PATH = ROOT / f"src/{LOADER_MODULE}.py"
LOADER_SHA = "fcf1f330f9423e9b9af26c09709259664f9d95fdda5eae9be385638107332d29"
LOADER_TEST_PATH = ROOT / "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_wrapper_v1.py"
LOADER_TEST_SHA = "c8aa9f0519eeced6d8c303a3168e9e03556574ed194cf98858b0095c0ecf0bf0"
ANALYSIS_MODULE = "cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_read_only_analysis"
ANALYSIS_PATH = ROOT / f"src/{ANALYSIS_MODULE}.py"
ANALYSIS_SHA = "076b249cece56c8ea2076aacffff65d029c950a6338e3bc898ca1b794ed98031"

CANONICAL_PYTHON = Path("/home/kjhan/miniconda3/envs/circle/bin/python3.11")
CANONICAL_PYTHON_SHA = "9ee5fb16ef60eb6a53af53ae6bd300a5ac8c01d81a8c961e7cdf1497efee3ccc"
EXPECTED_PYTHON_VERSION = "3.11.15"
EXPECTED_HOST = "grammar-debug"

AUTH_KEYS = (
    "HOP_authorized", "PM_authorized", "RAMSES_authorized",
    "analysis_result_artifact_creation_authorized", "automatic_follow_on_authorized",
    "cache_write_authorized", "candidate_selection_authorized",
    "canonical_analysis_execution_authorized", "canonical_artifact_read_authorized",
    "conditional_bank_authorized", "downstream_execution_authorized",
    "filesystem_scientific_output_authorized", "one_shot_receipt_creation_authorized",
    "production_rerun_authorized", "retry_resume_retune_or_scale_up_authorized",
    "slurm_submission_authorized", "stdout_scientific_payload_authorized",
)
FUTURE_AUTHORIZATION = {key: key in {
    "canonical_analysis_execution_authorized", "canonical_artifact_read_authorized",
    "one_shot_receipt_creation_authorized", "slurm_submission_authorized",
} for key in AUTH_KEYS}
FALSE_AUTHORIZATION = {key: False for key in AUTH_KEYS}

GIT_ENV = {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}
GIT_COMMANDS = (
    ("/usr/bin/git", "symbolic-ref", "--quiet", "--short", "HEAD"),
    ("/usr/bin/git", "rev-parse", "--verify", "HEAD"),
    ("/usr/bin/git", "rev-parse", "--verify", "@{upstream}"),
    ("/usr/bin/git", "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", ".", ":(exclude)scripts/tripwire/**"),
    ("/usr/bin/git", "rev-list", "--parents", "-n", "1", "HEAD"),
)

PROGRAM_KEYS = {"schema", "status", "date", "purpose", "lineage", "canonical_paths", "resource_contract", "implementation_files", "authorization", "next"}
GRANT_KEYS = {
    "schema", "status", "grant_id", "one_shot", "program_path", "program_sha256",
    "remediation_design_path", "remediation_design_commit", "remediation_design_sha256",
    "grant_release_manifest_design_path", "grant_release_manifest_design_commit",
    "grant_release_manifest_design_sha256", "implementation_commit", "implementation_tree",
    "implementation_file_sha256_map", "artifact_contract_digest", "execution_contract_digest",
    "canonical_paths_digest", "resource_contract_digest", "release_path", "release_id",
    "release_payload_sha256", "release_sha256", "manifest_path", "manifest_id",
    "manifest_sha256", "receipt_root", "authorization",
}
MANIFEST_KEYS = {
    "schema", "status", "manifest_id", "release_path", "release_id",
    "release_payload_sha256", "remediation_design_sha256",
    "grant_release_manifest_design_sha256", "implementation_commit", "implementation_tree",
    "program_sha256", "implementation_file_sha256_map", "artifact_contract_digest", "execution_contract_digest",
    "canonical_paths_digest", "resource_contract_digest", "one_shot",
}
RELEASE_KEYS = {"schema", "status", "release_id", "payload", "payload_sha256", "manifest_path", "manifest_id", "manifest_sha256"}
PAYLOAD_KEYS = {
    "schema", "status", "release_id", "remediation_design_commit",
    "remediation_design_sha256", "grant_release_manifest_design_commit",
    "grant_release_manifest_design_sha256", "implementation_commit", "implementation_tree",
    "program_sha256", "implementation_file_sha256_map",
    "artifact_contract_digest", "execution_contract_digest",
    "canonical_paths_digest", "resource_contract_digest", "one_shot", "authorization",
}


class AuthorizationError(RuntimeError):
    """Fail-closed provenance, authorization, or lifecycle error."""


@dataclass(frozen=True)
class SnapshotFile:
    path: Path
    payload: bytes
    sha256: str
    dev: int
    ino: int
    size: int
    nlink: int
    mode: int


@dataclass(frozen=True)
class AuthorizationBundle:
    program: Mapping[str, Any]
    grant: Mapping[str, Any]
    release: Mapping[str, Any]
    manifest: Mapping[str, Any]
    program_file: SnapshotFile
    grant_file: SnapshotFile
    release_file: SnapshotFile
    manifest_file: SnapshotFile
    head: str


@dataclass(frozen=True)
class CreatedDirectory:
    path: Path
    name: str
    parent_fd: int
    fd: int
    dev: int
    ino: int
    mode: int


@dataclass
class ReceiptLease:
    receipt: Path
    snapshot_sha: str
    bundle: AuthorizationBundle
    root_parent_path: Path
    root_parent_fd: int
    root_parent_dev: int
    root_parent_ino: int
    root_parent_mode: int
    root_item: CreatedDirectory
    parent_item: CreatedDirectory
    parent_fd: int
    receipt_fd: int
    item: CreatedDirectory
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            raise AuthorizationError("receipt lease closed more than once")
        self.closed = True
        error: OSError | None = None
        for descriptor in (self.receipt_fd, self.parent_fd, self.root_item.fd, self.root_parent_fd):
            try:
                os.close(descriptor)
            except OSError as caught:
                error = error or caught
        if error is not None:
            raise error


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _canonical_file_bytes(value: Any) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _full_hex(value: Any, size: int, label: str) -> str:
    text = value if isinstance(value, str) else ""
    if len(text) != size or any(ch not in "0123456789abcdef" for ch in text):
        raise AuthorizationError(f"{label} is not lowercase {size}-hex")
    return text


def _git_blob_oid(payload: bytes) -> str:
    framed = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    return hashlib.sha1(framed, usedforsecurity=False).hexdigest()


def _ascii_short_host(value: str) -> str:
    try:
        encoded = value.split(".", 1)[0].encode("ascii")
    except UnicodeEncodeError as error:
        raise AuthorizationError("hostname is not ASCII") from error
    return encoded.decode("ascii").lower()


def _reject_symlink_components(path: Path, *, final_may_be_absent: bool = False) -> None:
    if not path.is_absolute():
        raise AuthorizationError("path is not absolute")
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:]):
        current /= part
        try:
            value = os.lstat(current)
        except FileNotFoundError:
            if final_may_be_absent and index == len(path.parts[1:]) - 1:
                return
            raise AuthorizationError(f"path component is absent: {current}")
        if stat.S_ISLNK(value.st_mode):
            raise AuthorizationError(f"symlink path component is forbidden: {current}")


def _duplicate_free_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuthorizationError("duplicate JSON key")
        result[key] = value
    return result


def _parse_canonical_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload, object_pairs_hook=_duplicate_free_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthorizationError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict) or _canonical_file_bytes(value) != payload:
        raise AuthorizationError(f"{label} is not canonical compact JSON plus newline")
    return value


def _stable_read(path: Path, *, mode: int, label: str) -> SnapshotFile:
    _reject_symlink_components(path)
    first = os.lstat(path)
    if not stat.S_ISREG(first.st_mode) or stat.S_IMODE(first.st_mode) != mode:
        raise AuthorizationError(f"{label} mode or type is wrong")
    fd = -1
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        before = os.fstat(fd)
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        payload = b"".join(chunks)
        after = os.fstat(fd)
        final = os.lstat(path)
    except OSError as error:
        raise AuthorizationError(f"stable read failed for {label}") from error
    finally:
        if fd >= 0:
            os.close(fd)
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mode, value.st_nlink)
    if identity(first) != identity(before) or identity(before) != identity(after) or identity(after) != identity(final):
        raise AuthorizationError(f"{label} identity changed")
    return SnapshotFile(path, payload, _sha256_bytes(payload), int(before.st_dev), int(before.st_ino), int(before.st_size), int(before.st_nlink), stat.S_IMODE(before.st_mode))


def _strict_object(file: SnapshotFile, keys: set[str], label: str) -> dict[str, Any]:
    value = _parse_canonical_object(file.payload, label)
    if set(value) != keys:
        raise AuthorizationError(f"{label} keyset changed")
    return value


def _validate_program(program_file: SnapshotFile) -> dict[str, Any]:
    program = _strict_object(program_file, PROGRAM_KEYS, "program")
    if (
        program.get("schema") != "ouruniv-cf4-v6-open-parent-key-overlap-canonical-read-only-analysis-authorized-program-v5"
        or program.get("status") != "frozen_v5_program_execution_false_until_sealed_v5_pair_grant_and_audits"
        or program.get("date") != "2026-09-03"
        or program.get("purpose") != "Freeze the executable v5 binding for the exact path-aware one-shot memory-only parent/key overlap analysis."
    ):
        raise AuthorizationError("program identity changed")
    if program.get("authorization") != FALSE_AUTHORIZATION:
        raise AuthorizationError("program is not execution-false")
    expected_paths = {"external_manifest": str(MANIFEST), "external_release": str(RELEASE), "local_grant": str(GRANT.relative_to(ROOT)), "local_program": str(PROGRAM.relative_to(ROOT)), "receipt_root": str(RECEIPT_ROOT)}
    if program.get("canonical_paths") != expected_paths:
        raise AuthorizationError("program canonical paths changed")
    expected_resource = {"cpus_per_task": 4, "login_host": "grammar", "node": "grammar-debug", "nodes": 1, "ntasks": 1, "partition": "debug", "pre_registered_maximum_expected_RSS_GiB": 6.5, "requested_memory_GiB": 8, "requested_memory_margin_over_expected_percent": 23.076923076923077, "requeue": False, "submission_mechanism": "Slurm_only", "time_limit": "01:00:00"}
    if program.get("resource_contract") != expected_resource:
        raise AuthorizationError("program resource contract changed")
    lineage = program.get("lineage")
    expected_lineage = {
        "branch": "agent/freeze-zoom-pipeline",
        "remediation_design_commit": REMEDIATION_DESIGN_COMMIT,
        "remediation_design_sha256": REMEDIATION_DESIGN_SHA,
        "implementation_parent": REMEDIATION_DESIGN_COMMIT,
        "analyzer_sha256": ANALYSIS_SHA,
        "loader_sha256": LOADER_SHA,
        "abandoned_v3_remediation_commit": "4b71d8a204468bca2fc92a180af2344b9d96eb3c",
        "abandoned_v3_pair_design_commit": "864e6e0468ab9135e2be47c773c3ea7614196a32",
    }
    if lineage != expected_lineage:
        raise AuthorizationError("program lineage changed")
    rows = program.get("implementation_files")
    if not isinstance(rows, list) or len(rows) != 5:
        raise AuthorizationError("program implementation file rows changed")
    expected_order = (
        ("src/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v5.py", "0644"),
        ("scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v5.sbatch", "0755"),
        ("scripts/status_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v5.sh", "0755"),
        ("tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v5.py", "0644"),
        ("tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v5_runner.py", "0644"),
    )
    if any(not isinstance(row, dict) for row in rows):
        raise AuthorizationError("program implementation row schema changed")
    if tuple((row.get("path"), row.get("mode")) for row in rows) != expected_order:
        raise AuthorizationError("program implementation row order or mode changed")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "mode"}:
            raise AuthorizationError("program implementation row schema changed")
        path = row.get("path")
        if not isinstance(path, str) or path in seen:
            raise AuthorizationError("program implementation path changed")
        seen.add(path)
        _full_hex(row.get("sha256"), 64, "implementation SHA")
        if row.get("mode") not in {"0644", "0755"}:
            raise AuthorizationError("program implementation mode changed")
    expected_names = {
        "src/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v5.py",
        "scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v5.sbatch",
        "scripts/status_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v5.sh",
        "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v5.py",
        "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v5_runner.py",
    }
    if seen != expected_names:
        raise AuthorizationError("program implementation path set changed")
    expected_next = {"canonical_analysis_execution_authorized": False, "canonical_artifact_read_authorized": False, "external_pair_or_local_grant_creation_authorized": False, "immediate": "fixture_only_v5_implementation_complete_pair_issuance_requires_separate_approval", "receipt_creation_authorized": False, "slurm_submission_authorized": False}
    if program.get("next") != expected_next:
        raise AuthorizationError("program next contract changed")
    return program


def _verify_program_implementation_files(program: Mapping[str, Any]) -> None:
    for row in program["implementation_files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise AuthorizationError("implementation path is noncanonical")
        mode = int(row["mode"], 8)
        if _stable_read(ROOT / relative, mode=mode, label=row["path"]).sha256 != row["sha256"]:
            raise AuthorizationError("implementation file SHA changed")


def _run_git(argv: Sequence[str], *, runner: Callable[..., subprocess.CompletedProcess[bytes]], allowed: Sequence[Sequence[str]] = GIT_COMMANDS) -> bytes:
    command = tuple(argv)
    if command not in tuple(tuple(row) for row in allowed):
        raise AuthorizationError("non-whitelisted Git argv")
    try:
        completed = runner(list(command), cwd=str(ROOT), env=GIT_ENV, shell=False, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise AuthorizationError("Git validation command failed") from error
    if not isinstance(completed.stdout, bytes) or completed.stderr != b"":
        raise AuthorizationError("Git output contract changed")
    return completed.stdout


def _validate_git(grant_file: SnapshotFile, grant: Mapping[str, Any], *, runner: Callable[..., subprocess.CompletedProcess[bytes]]) -> str:
    implementation = _full_hex(grant.get("implementation_commit"), 40, "grant implementation commit")
    implementation_tree = _full_hex(grant.get("implementation_tree"), 40, "grant implementation tree")
    grant_rel = str(GRANT.relative_to(ROOT))
    dynamic = (
        ("/usr/bin/git", "diff-tree", "--no-commit-id", "--name-status", "-r", "--no-renames", "HEAD^", "HEAD"),
        ("/usr/bin/git", "ls-tree", "HEAD", "--", grant_rel),
        ("/usr/bin/git", "rev-parse", "--verify", f"{implementation}^{{tree}}"),
    )
    commands = GIT_COMMANDS + dynamic
    outputs = [_run_git(command, runner=runner, allowed=commands) for command in commands]
    if outputs[0] != b"agent/freeze-zoom-pipeline\n":
        raise AuthorizationError("Git symbolic branch changed")
    head = _full_hex(outputs[1].decode("ascii").rstrip("\n"), 40, "HEAD")
    tracking = _full_hex(outputs[2].decode("ascii").rstrip("\n"), 40, "tracking HEAD")
    if head != tracking or outputs[3] != b"":
        raise AuthorizationError("Git HEAD, tracking, or clean-scope binding failed")
    if outputs[4] != f"{head} {implementation}\n".encode("ascii"):
        raise AuthorizationError("HEAD is not the required single-parent commit")
    if outputs[5] != f"A\t{grant_rel}\n".encode("utf-8"):
        raise AuthorizationError("grant is not the exact only HEAD addition")
    blob = _git_blob_oid(grant_file.payload)
    if outputs[6] != f"100644 blob {blob}\t{grant_rel}\n".encode("utf-8"):
        raise AuthorizationError("grant Git mode or framed blob OID changed")
    if outputs[7] != f"{implementation_tree}\n".encode("ascii"):
        raise AuthorizationError("implementation tree binding changed")
    return head


def _validate_pair(program_file: SnapshotFile, grant_file: SnapshotFile, release_file: SnapshotFile, manifest_file: SnapshotFile, program: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    grant = _strict_object(grant_file, GRANT_KEYS, "grant")
    release = _strict_object(release_file, RELEASE_KEYS, "release")
    manifest = _strict_object(manifest_file, MANIFEST_KEYS, "manifest")
    payload = release.get("payload")
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        raise AuthorizationError("release payload schema changed")
    for value, label in (
        (grant.get("grant_id"), "grant id"),
        (release.get("release_id"), "release id"),
        (manifest.get("manifest_id"), "manifest id"),
        (release.get("payload_sha256"), "payload SHA"),
    ):
        _full_hex(value, 64, label)
    _full_hex(grant.get("implementation_tree"), 40, "implementation tree")
    if (
        grant.get("schema") != "ouruniv-cf4-v6-open-parent-key-overlap-canonical-read-only-analysis-execution-grant-v5"
        or grant.get("status") != "sealed_one_shot_parent_key_overlap_read_only_analysis_authorization"
        or grant.get("one_shot") is not True
    ):
        raise AuthorizationError("grant identity changed")
    if (
        release.get("schema") != "ouruniv-cf4-v6-open-parent-key-overlap-canonical-read-only-analysis-execution-release-v5"
        or release.get("status") != "complete_pass_external_postcommit_lineage_audit"
    ):
        raise AuthorizationError("release identity changed")
    if (
        manifest.get("schema") != "ouruniv-cf4-v6-open-parent-key-overlap-canonical-read-only-analysis-execution-manifest-v5"
        or manifest.get("status") != "complete_paired_external_manifest"
        or manifest.get("one_shot") is not True
    ):
        raise AuthorizationError("manifest identity changed")
    if (
        payload.get("schema") != "ouruniv-cf4-v6-open-parent-key-overlap-canonical-read-only-analysis-execution-release-payload-v5"
        or payload.get("status") != "sealed_external_postcommit_lineage_release_payload"
        or payload.get("one_shot") is not True
    ):
        raise AuthorizationError("release payload identity changed")
    if grant.get("authorization") != FUTURE_AUTHORIZATION or payload.get("authorization") != FUTURE_AUTHORIZATION:
        raise AuthorizationError("runtime authorization matrix changed")
    if release.get("payload_sha256") != _sha256_bytes(_canonical_bytes(payload)):
        raise AuthorizationError("release payload digest changed")
    if (
        release.get("manifest_sha256") != manifest_file.sha256
        or grant.get("manifest_sha256") != manifest_file.sha256
        or grant.get("release_sha256") != release_file.sha256
    ):
        raise AuthorizationError("complete pair file SHA binding changed")

    implementation_map = {row["path"]: row["sha256"] for row in program["implementation_files"]}
    equality = (
        grant.get("release_id") == release.get("release_id") == manifest.get("release_id") == payload.get("release_id")
        and grant.get("manifest_id") == release.get("manifest_id") == manifest.get("manifest_id")
        and grant.get("release_payload_sha256") == release.get("payload_sha256") == manifest.get("release_payload_sha256")
        and grant.get("program_sha256") == program_file.sha256 == payload.get("program_sha256") == manifest.get("program_sha256")
        and grant.get("implementation_commit") == payload.get("implementation_commit") == manifest.get("implementation_commit")
        and grant.get("implementation_tree") == payload.get("implementation_tree") == manifest.get("implementation_tree")
        and grant.get("implementation_file_sha256_map") == payload.get("implementation_file_sha256_map") == manifest.get("implementation_file_sha256_map") == implementation_map
        and grant.get("remediation_design_sha256") == payload.get("remediation_design_sha256") == manifest.get("remediation_design_sha256") == REMEDIATION_DESIGN_SHA
        and grant.get("grant_release_manifest_design_sha256") == payload.get("grant_release_manifest_design_sha256") == manifest.get("grant_release_manifest_design_sha256") == PAIR_DESIGN_SHA
    )
    if not equality:
        raise AuthorizationError("grant/release/manifest direct binding changed")
    if (
        grant.get("program_path") != str(PROGRAM.relative_to(ROOT))
        or grant.get("release_path") != str(RELEASE)
        or grant.get("manifest_path") != str(MANIFEST)
        or grant.get("receipt_root") != str(RECEIPT_ROOT)
        or grant.get("remediation_design_path") != str(REMEDIATION_DESIGN.relative_to(Path("/home/kjhan/BACKUP/CF4")))
        or grant.get("grant_release_manifest_design_path") != str(PAIR_DESIGN.relative_to(Path("/home/kjhan/BACKUP/CF4")))
        or release.get("manifest_path") != str(MANIFEST)
        or manifest.get("release_path") != str(RELEASE)
    ):
        raise AuthorizationError("pair canonical path binding changed")
    if (
        grant.get("remediation_design_commit") != REMEDIATION_DESIGN_COMMIT
        or payload.get("remediation_design_commit") != REMEDIATION_DESIGN_COMMIT
        or grant.get("grant_release_manifest_design_commit") != PAIR_DESIGN_COMMIT
        or payload.get("grant_release_manifest_design_commit") != PAIR_DESIGN_COMMIT
    ):
        raise AuthorizationError("pair canonical lineage commit changed")
    digests = ("artifact_contract_digest", "execution_contract_digest", "canonical_paths_digest", "resource_contract_digest")
    if any(grant.get(key) != payload.get(key) or payload.get(key) != manifest.get(key) for key in digests):
        raise AuthorizationError("inherited contract digest binding changed")
    expected_digests = {
        "artifact_contract_digest": "1cab22081d83abc5b09c8dfbab81e37aeed789c09c0004344808c67986c95ff5",
        "execution_contract_digest": "2511effaa8860ea9af22aaf2084a91eb2b0a0b5028836f1b054e1d4b166cc5ea",
        "canonical_paths_digest": "e5acd198245a6eb7f9318367f9a8a72d4f3b193bf3380b008ec95cd7298c5bec",
        "resource_contract_digest": "faa8a358d8aeeb4caed9e6b19b3a979227fd8b1b1e938f6aaa099ef2ecdd9abc",
    }
    if any(grant.get(key) != value for key, value in expected_digests.items()):
        raise AuthorizationError("inherited contract digest value changed")

    release_id = hashlib.sha256(
        ("ouruniv-parent-key-overlap-release-v5\0" + PAIR_DESIGN_COMMIT + "\0" + REMEDIATION_DESIGN_SHA + "\0" + program_file.sha256).encode()
    ).hexdigest()
    manifest_id = hashlib.sha256(
        ("ouruniv-parent-key-overlap-manifest-v5\0" + PAIR_DESIGN_COMMIT + "\0" + release_id).encode()
    ).hexdigest()
    grant_id = hashlib.sha256(
        ("ouruniv-parent-key-overlap-grant-v5\0" + PAIR_DESIGN_COMMIT + "\0" + release_file.sha256 + "\0" + manifest_file.sha256 + "\0" + program_file.sha256).encode()
    ).hexdigest()
    if release.get("release_id") != release_id or manifest.get("manifest_id") != manifest_id or grant.get("grant_id") != grant_id:
        raise AuthorizationError("pair or grant identity derivation changed")
    return grant, release, manifest

def _verify_fixed_local_pins() -> None:
    for path, digest, mode in (
        (AUTHORIZATION_DESIGN, AUTHORIZATION_DESIGN_SHA, 0o644),
        (REMEDIATION_DESIGN, REMEDIATION_DESIGN_SHA, 0o644),
        (PAIR_DESIGN, PAIR_DESIGN_SHA, 0o644),
        (WRAPPER_DESIGN, WRAPPER_DESIGN_SHA, 0o644),
        (WRAPPER_ERRATUM, WRAPPER_ERRATUM_SHA, 0o644),
        (LOADER_RESULT_RECORD, LOADER_RESULT_RECORD_SHA, 0o644),
    ):
        if _stable_read(path, mode=mode, label=str(path)).sha256 != digest:
            raise AuthorizationError("fixed local design or record SHA changed")


def _validate_host_and_allocation(hostname: str, environment: Mapping[str, str]) -> None:
    if _ascii_short_host(hostname) != EXPECTED_HOST:
        raise AuthorizationError("execution host is not grammar-debug")
    expected = {"SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "4", "SLURM_MEM_PER_NODE": "8192", "SLURM_JOB_PARTITION": "debug", "SLURM_JOB_NAME": "cf4-parent-overlap-v1"}
    if any(environment.get(key) != value for key, value in expected.items()):
        raise AuthorizationError("Slurm allocation does not match the frozen resource contract")
    if environment.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise AuthorizationError("GPU visibility must be empty")


def _verify_runtime_imports(program: Mapping[str, Any]) -> Any:
    if Path(sys.executable) != CANONICAL_PYTHON or sys.version.split()[0] != EXPECTED_PYTHON_VERSION:
        raise AuthorizationError("canonical Python identity changed")
    if _stable_read(CANONICAL_PYTHON, mode=0o755, label="canonical Python").sha256 != CANONICAL_PYTHON_SHA:
        raise AuthorizationError("canonical Python SHA changed")
    rows = {row["path"]: row for row in program["implementation_files"]}
    this_path = Path(__file__).resolve()
    row = rows.get(str(this_path.relative_to(ROOT)))
    if row is None or _stable_read(this_path, mode=0o644, label="authorized source").sha256 != row["sha256"]:
        raise AuthorizationError("authorized source pin changed")
    for path, digest in ((LOADER_PATH, LOADER_SHA), (LOADER_TEST_PATH, LOADER_TEST_SHA), (ANALYSIS_PATH, ANALYSIS_SHA)):
        if _stable_read(path, mode=0o644, label="audited import").sha256 != digest:
            raise AuthorizationError("audited import source changed")
    loader = importlib.import_module(LOADER_MODULE)
    analysis = importlib.import_module(ANALYSIS_MODULE)
    if Path(loader.__file__).resolve() != LOADER_PATH or Path(analysis.__file__).resolve() != ANALYSIS_PATH:
        raise AuthorizationError("audited import origin changed")
    import numpy as np
    import scipy
    expected = (
        (np, "2.4.6", Path("/home/kjhan/miniconda3/envs/circle/lib/python3.11/site-packages/numpy/__init__.py"), "92a46f791e453926d3292af2582b89995a289475f0eaaea71a949823200b838a"),
        (scipy, "1.17.1", Path("/home/kjhan/miniconda3/envs/circle/lib/python3.11/site-packages/scipy/__init__.py"), "335bc6e0a9909dc7534f9569a3685a92dc8001cb8c63a6da4c239849ff02d4d0"),
    )
    for module, version, path, digest in expected:
        if module.__version__ != version or Path(module.__file__).resolve() != path or _stable_read(path, mode=0o644, label=module.__name__).sha256 != digest:
            raise AuthorizationError(f"{module.__name__} import identity changed")
    return loader


def _snapshot(bundle: AuthorizationBundle, item: CreatedDirectory) -> dict[str, Any]:
    return {
        "grant_path": str(GRANT), "grant_sha256": bundle.grant_file.sha256,
        "program_path": str(PROGRAM), "program_sha256": bundle.program_file.sha256,
        "release_path": str(RELEASE), "release_sha256": bundle.release_file.sha256,
        "release_dev": bundle.release_file.dev, "release_ino": bundle.release_file.ino,
        "release_size": bundle.release_file.size, "release_nlink": bundle.release_file.nlink,
        "manifest_path": str(MANIFEST), "manifest_sha256": bundle.manifest_file.sha256,
        "remediation_design_sha256": REMEDIATION_DESIGN_SHA,
        "grant_release_manifest_design_sha256": PAIR_DESIGN_SHA,
        "implementation_commit": bundle.grant["implementation_commit"],
        "receipt_dev": item.dev, "receipt_ino": item.ino, "receipt_mode": "0700",
    }


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical_file_bytes(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_exclusive_at(directory_fd: int, name: str, value: Mapping[str, Any]) -> None:
    if "/" in name or name in {"", ".", ".."}:
        raise AuthorizationError("noncanonical receipt child name")
    payload = _canonical_file_bytes(value)
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)


def _stable_read_at(directory_fd: int, name: str, *, mode: int, label: str) -> SnapshotFile:
    try:
        first = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(first.st_mode) or stat.S_IMODE(first.st_mode) != mode:
            raise AuthorizationError(f"{label} mode or type is wrong")
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory_fd)
        try:
            before = os.fstat(fd)
            chunks: list[bytes] = []
            while True:
                block = os.read(fd, 1 << 20)
                if not block:
                    break
                chunks.append(block)
            after = os.fstat(fd)
            final = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        finally:
            os.close(fd)
    except OSError as error:
        raise AuthorizationError(f"stable receipt read failed for {label}") from error
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mode, value.st_nlink)
    if identity(first) != identity(before) or identity(before) != identity(after) or identity(after) != identity(final):
        raise AuthorizationError(f"{label} receipt identity changed")
    payload = b"".join(chunks)
    return SnapshotFile(Path(name), payload, _sha256_bytes(payload), int(before.st_dev), int(before.st_ino), int(before.st_size), int(before.st_nlink), stat.S_IMODE(before.st_mode))


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _receipt_entries_at(receipt_fd: int) -> set[str]:
    names = set(os.listdir(receipt_fd))
    for name in names:
        value = os.stat(name, dir_fd=receipt_fd, follow_symlinks=False)
        if not stat.S_ISREG(value.st_mode):
            raise AuthorizationError("receipt contains a non-regular entry")
    return names


def _assert_named_directory(item: CreatedDirectory, *, expected_mode: int | None = None) -> None:
    opened = os.fstat(item.fd)
    named = os.stat(item.name, dir_fd=item.parent_fd, follow_symlinks=False)
    mode = item.mode if expected_mode is None else stat.S_IFMT(item.mode) | expected_mode
    expected = (item.dev, item.ino, mode)
    if (int(opened.st_dev), int(opened.st_ino), int(opened.st_mode)) != expected or (int(named.st_dev), int(named.st_ino), int(named.st_mode)) != expected:
        raise AuthorizationError("opened receipt directory no longer has its canonical name")


def _assert_lease_chain(lease: ReceiptLease, *, receipt_mode: int = 0o700) -> None:
    opened = os.fstat(lease.root_parent_fd)
    named = os.lstat(lease.root_parent_path)
    expected = (lease.root_parent_dev, lease.root_parent_ino, lease.root_parent_mode)
    if (int(opened.st_dev), int(opened.st_ino), int(opened.st_mode)) != expected or (int(named.st_dev), int(named.st_ino), int(named.st_mode)) != expected:
        raise AuthorizationError("receipt root parent no longer has its canonical absolute path")
    _assert_named_directory(lease.root_item)
    _assert_named_directory(lease.parent_item)
    _assert_named_directory(lease.item, expected_mode=receipt_mode)


def _open_receipt_directory(receipt: Path) -> tuple[int, int, CreatedDirectory]:
    _reject_symlink_components(receipt)
    parent_fd = os.open(receipt.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        receipt_fd = os.open(receipt.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
    except BaseException:
        os.close(parent_fd)
        raise
    value = os.fstat(receipt_fd)
    item = CreatedDirectory(receipt, receipt.name, parent_fd, receipt_fd, int(value.st_dev), int(value.st_ino), int(value.st_mode))
    _assert_named_directory(item)
    return parent_fd, receipt_fd, item


@contextmanager
def _masked_signals() -> Iterator[None]:
    wanted = {signal.SIGHUP, signal.SIGINT, signal.SIGTERM}
    old = signal.pthread_sigmask(signal.SIG_BLOCK, wanted)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


def _receipt_path(grant: Mapping[str, Any]) -> Path:
    return RECEIPT_ROOT / _full_hex(grant.get("grant_id"), 64, "grant id") / "analysis"


def _receipt_entries(path: Path) -> set[str]:
    _reject_symlink_components(path)
    root = os.lstat(path)
    if not stat.S_ISDIR(root.st_mode):
        raise AuthorizationError("receipt is not a directory")
    with os.scandir(path) as iterator:
        entries = list(iterator)
    if any(entry.is_symlink() or not entry.is_file(follow_symlinks=False) for entry in entries):
        raise AuthorizationError("receipt contains a non-regular entry")
    return {entry.name for entry in entries}


def _validate_marker(name: str, value: Mapping[str, Any], *, bundle: AuthorizationBundle | None = None, snapshot_sha: str | None = None) -> None:
    common = {"grant_id", "release_id", "manifest_id", "snapshot_sha256"}
    contracts = {
        "RUNNING": (
            common | {"schema", "status"},
            "ouruniv-cf4-v6-open-parent-key-overlap-read-only-analysis-receipt-running-v5",
            "running_authorized_read_only_in_memory_analysis",
        ),
        "COMPLETE": (
            common | {"schema", "status", "analysis_return_validated_in_process"},
            "ouruniv-cf4-v6-open-parent-key-overlap-read-only-analysis-receipt-complete-v5",
            "complete_valid_provenance_and_in_memory_analysis_postvalidation",
        ),
        "FAILED": (
            common | {"schema", "status", "failure_class", "checkpoint"},
            "ouruniv-cf4-v6-open-parent-key-overlap-read-only-analysis-receipt-failed-v5",
            "failed_invalid_provenance_read_or_analysis_execution",
        ),
    }
    if name not in contracts:
        raise AuthorizationError("unknown receipt marker")
    keys, schema, status_value = contracts[name]
    if set(value) != keys or value.get("schema") != schema or value.get("status") != status_value:
        raise AuthorizationError("receipt marker schema or status changed")
    for key in common:
        _full_hex(value.get(key), 64, f"marker {key}")
    if name == "COMPLETE" and value.get("analysis_return_validated_in_process") is not True:
        raise AuthorizationError("COMPLETE postvalidation flag changed")
    if name == "FAILED" and (
        value.get("failure_class") != "invalid_provenance_read_or_analysis_execution"
        or value.get("checkpoint") not in {"after_snapshot", "after_RUNNING", "during_analysis", "during_postvalidation", "during_terminal_transition"}
    ):
        raise AuthorizationError("FAILED classification changed")
    if bundle is not None:
        expected = _marker_common(bundle, snapshot_sha or "")
        if any(value.get(key) != expected[key] for key in common):
            raise AuthorizationError("receipt marker identity binding changed")


def _revalidate_snapshot(bundle: AuthorizationBundle, item: CreatedDirectory, snapshot_sha: str, *, lease: ReceiptLease | None = None, require_canonical_name: bool = True) -> None:
    if require_canonical_name:
        if lease is None:
            raise AuthorizationError("full receipt lease is required for canonical validation")
        _assert_lease_chain(lease)
    grant = _stable_read(GRANT, mode=0o644, label="grant")
    program = _stable_read(PROGRAM, mode=0o644, label="program")
    release = _stable_read(RELEASE, mode=0o444, label="release")
    manifest = _stable_read(MANIFEST, mode=0o444, label="manifest")
    if (grant.sha256, program.sha256, release.sha256, manifest.sha256) != (bundle.grant_file.sha256, bundle.program_file.sha256, bundle.release_file.sha256, bundle.manifest_file.sha256):
        raise AuthorizationError("authorization metadata changed after receipt")
    for current, sealed, label in (
        (grant, bundle.grant_file, "grant"),
        (program, bundle.program_file, "program"),
        (release, bundle.release_file, "release"),
        (manifest, bundle.manifest_file, "manifest"),
    ):
        if (current.dev, current.ino, current.size, current.nlink, current.mode, current.sha256) != (sealed.dev, sealed.ino, sealed.size, sealed.nlink, sealed.mode, sealed.sha256):
            raise AuthorizationError(f"{label} sealed identity changed")
    anchor = _stable_read_at(item.fd, "release.anchor", mode=0o444, label="release anchor")
    if (anchor.dev, anchor.ino, anchor.size, anchor.nlink, anchor.mode, anchor.sha256) != (bundle.release_file.dev, bundle.release_file.ino, bundle.release_file.size, bundle.release_file.nlink, bundle.release_file.mode, bundle.release_file.sha256):
        raise AuthorizationError("release anchor identity changed")
    snapshot_file = _stable_read_at(item.fd, "snapshot.json", mode=0o444, label="snapshot")
    if snapshot_file.sha256 != snapshot_sha or _parse_canonical_object(snapshot_file.payload, "snapshot") != _snapshot(bundle, item):
        raise AuthorizationError("receipt snapshot changed")
    entries = _receipt_entries_at(item.fd)
    if "RUNNING" in entries:
        running_file = _stable_read_at(item.fd, "RUNNING", mode=0o444, label="RUNNING")
        running = _parse_canonical_object(running_file.payload, "RUNNING")
        _validate_marker("RUNNING", running, bundle=bundle, snapshot_sha=snapshot_sha)
    if require_canonical_name:
        _assert_lease_chain(lease)


def _bootstrap_receipt(bundle: AuthorizationBundle, *, interrupt: Callable[[str], None] | None = None) -> ReceiptLease:
    receipt = _receipt_path(bundle.grant)
    created: list[CreatedDirectory] = []
    created_files: dict[str, tuple[int, int, int]] = {}
    directory_fds: dict[Path, int] = {}
    snapshot_sealed = False
    retained_fds: set[int] = set()
    hook = interrupt or (lambda _: None)
    try:
        with _masked_signals():
          try:
            parent_path = RECEIPT_ROOT.parent
            _reject_symlink_components(parent_path)
            directory_fds[parent_path] = os.open(parent_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            root_parent_value = os.fstat(directory_fds[parent_path])
            for directory in (RECEIPT_ROOT, receipt.parent):
                parent_fd = directory_fds[directory.parent]
                existed = os.path.lexists(directory)
                if existed:
                    _reject_symlink_components(directory)
                    value = os.stat(directory.name, dir_fd=parent_fd, follow_symlinks=False)
                    if not stat.S_ISDIR(value.st_mode):
                        raise AuthorizationError("receipt ancestor is not a directory")
                else:
                    _reject_symlink_components(directory, final_may_be_absent=True)
                    os.mkdir(directory.name, mode=0o700, dir_fd=parent_fd)
                    value = os.stat(directory.name, dir_fd=parent_fd, follow_symlinks=False)
                directory_fd = os.open(directory.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent_fd)
                directory_fds[directory] = directory_fd
                if not existed:
                    live = os.fstat(directory_fd)
                    created.append(CreatedDirectory(directory, directory.name, parent_fd, directory_fd, int(live.st_dev), int(live.st_ino), int(live.st_mode)))
            root_value = os.fstat(directory_fds[RECEIPT_ROOT])
            root_item = CreatedDirectory(RECEIPT_ROOT, RECEIPT_ROOT.name, directory_fds[parent_path], directory_fds[RECEIPT_ROOT], int(root_value.st_dev), int(root_value.st_ino), int(root_value.st_mode))
            receipt_parent_fd = directory_fds[receipt.parent]
            parent_value = os.fstat(receipt_parent_fd)
            parent_item = CreatedDirectory(receipt.parent, receipt.parent.name, directory_fds[RECEIPT_ROOT], receipt_parent_fd, int(parent_value.st_dev), int(parent_value.st_ino), int(parent_value.st_mode))
            try:
                os.stat(receipt.name, dir_fd=receipt_parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise AuthorizationError("one-shot receipt already exists")
            _reject_symlink_components(receipt, final_may_be_absent=True)
            os.mkdir(receipt.name, mode=0o700, dir_fd=receipt_parent_fd)
            receipt_fd = os.open(receipt.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=receipt_parent_fd)
            directory_fds[receipt] = receipt_fd
            value = os.fstat(receipt_fd)
            created.append(CreatedDirectory(receipt, receipt.name, receipt_parent_fd, receipt_fd, int(value.st_dev), int(value.st_ino), int(value.st_mode)))
            receipt_item = created[-1]
            lease = ReceiptLease(
                receipt=receipt,
                snapshot_sha="",
                bundle=bundle,
                root_parent_path=parent_path,
                root_parent_fd=directory_fds[parent_path],
                root_parent_dev=int(root_parent_value.st_dev),
                root_parent_ino=int(root_parent_value.st_ino),
                root_parent_mode=int(root_parent_value.st_mode),
                root_item=root_item,
                parent_item=parent_item,
                parent_fd=receipt_parent_fd,
                receipt_fd=receipt_fd,
                item=receipt_item,
            )
            hook("after_mkdir")
            _assert_lease_chain(lease)
            os.link(RELEASE, "release.anchor", dst_dir_fd=receipt_fd, follow_symlinks=False)
            anchor_stat = os.stat("release.anchor", dir_fd=receipt_fd, follow_symlinks=False)
            created_files["release.anchor"] = (int(anchor_stat.st_dev), int(anchor_stat.st_ino), int(anchor_stat.st_mode))
            hook("after_anchor")
            _assert_lease_chain(lease)
            live_release = _stable_read(RELEASE, mode=0o444, label="post-link release")
            if (
                (live_release.dev, live_release.ino, live_release.size, live_release.sha256, live_release.mode)
                != (bundle.release_file.dev, bundle.release_file.ino, bundle.release_file.size, bundle.release_file.sha256, bundle.release_file.mode)
                or live_release.nlink != bundle.release_file.nlink + 1
            ):
                raise AuthorizationError("release identity changed while creating anchor")
            bundle = AuthorizationBundle(bundle.program, bundle.grant, bundle.release, bundle.manifest, bundle.program_file, bundle.grant_file, live_release, bundle.manifest_file, bundle.head)
            lease.bundle = bundle
            value = _snapshot(bundle, receipt_item)
            _write_exclusive_at(receipt_fd, "snapshot.json", value)
            snapshot_stat = os.stat("snapshot.json", dir_fd=receipt_fd, follow_symlinks=False)
            created_files["snapshot.json"] = (int(snapshot_stat.st_dev), int(snapshot_stat.st_ino), int(snapshot_stat.st_mode))
            snapshot_sha = _sha256_bytes(_canonical_file_bytes(value))
            lease.snapshot_sha = snapshot_sha
            snapshot_sealed = True
            hook("after_snapshot")
            _assert_lease_chain(lease)
            running = {"schema": "ouruniv-cf4-v6-open-parent-key-overlap-read-only-analysis-receipt-running-v5", "status": "running_authorized_read_only_in_memory_analysis", "grant_id": bundle.grant["grant_id"], "release_id": bundle.release["release_id"], "manifest_id": bundle.manifest["manifest_id"], "snapshot_sha256": snapshot_sha}
            _write_exclusive_at(receipt_fd, "RUNNING", running)
            running_stat = os.stat("RUNNING", dir_fd=receipt_fd, follow_symlinks=False)
            created_files["RUNNING"] = (int(running_stat.st_dev), int(running_stat.st_ino), int(running_stat.st_mode))
            os.fsync(receipt_fd)
            hook("after_RUNNING")
            _assert_lease_chain(lease)
            _revalidate_snapshot(bundle, receipt_item, snapshot_sha, lease=lease)
            retained_fds = {directory_fds[parent_path], directory_fds[RECEIPT_ROOT], receipt_parent_fd, receipt_fd}
            return lease
          except BaseException:
            if snapshot_sealed and receipt in directory_fds:
                receipt_fd = directory_fds[receipt]
                snapshot_file = _stable_read_at(receipt_fd, "snapshot.json", mode=0o444, label="snapshot")
                try:
                    _assert_lease_chain(lease)
                    canonical_name = True
                except (AuthorizationError, OSError):
                    canonical_name = False
                _seal_failed_at(bundle, created[-1], snapshot_file.sha256, lease=lease, checkpoint="after_snapshot", require_canonical_name=canonical_name)
            else:
                receipt_fd = directory_fds.get(receipt)
                for name in ("RUNNING", "snapshot.json", "release.anchor"):
                    if receipt_fd is None:
                        break
                    try:
                        expected = created_files[name]
                        current = os.stat(name, dir_fd=receipt_fd, follow_symlinks=False)
                        if (int(current.st_dev), int(current.st_ino), int(current.st_mode)) == expected:
                            os.unlink(name, dir_fd=receipt_fd)
                    except FileNotFoundError:
                        pass
                    except KeyError:
                        pass
                for item in reversed(created):
                    try:
                        opened = os.fstat(item.fd)
                        named = os.stat(item.name, dir_fd=item.parent_fd, follow_symlinks=False)
                        expected = (item.dev, item.ino, item.mode)
                        if (int(opened.st_dev), int(opened.st_ino), int(opened.st_mode)) == expected and (int(named.st_dev), int(named.st_ino), int(named.st_mode)) == expected:
                            os.rmdir(item.name, dir_fd=item.parent_fd)
                    except (FileNotFoundError, OSError):
                        pass
            raise
    finally:
        for descriptor in reversed(tuple(dict.fromkeys(directory_fds.values()))):
            if descriptor in retained_fds:
                continue
            try:
                os.close(descriptor)
            except OSError:
                pass


def _marker_common(bundle: AuthorizationBundle, snapshot_sha: str) -> dict[str, Any]:
    return {"grant_id": bundle.grant["grant_id"], "release_id": bundle.release["release_id"], "manifest_id": bundle.manifest["manifest_id"], "snapshot_sha256": snapshot_sha}


def _seal_failed_at(bundle: AuthorizationBundle, item: CreatedDirectory, snapshot_sha: str, *, lease: ReceiptLease | None = None, checkpoint: str, require_canonical_name: bool) -> None:
    entries = _receipt_entries_at(item.fd)
    if entries not in ({"release.anchor", "snapshot.json"}, {"RUNNING", "release.anchor", "snapshot.json"}):
        raise AuthorizationError("receipt is not eligible for FAILED transition")
    _revalidate_snapshot(bundle, item, snapshot_sha, lease=lease, require_canonical_name=require_canonical_name)
    if _receipt_entries_at(item.fd) != entries:
        raise AuthorizationError("receipt changed before FAILED transition")
    try:
        os.unlink("RUNNING", dir_fd=item.fd)
    except FileNotFoundError:
        pass
    value = {"schema": "ouruniv-cf4-v6-open-parent-key-overlap-read-only-analysis-receipt-failed-v5", "status": "failed_invalid_provenance_read_or_analysis_execution", **_marker_common(bundle, snapshot_sha), "failure_class": "invalid_provenance_read_or_analysis_execution", "checkpoint": checkpoint}
    _write_exclusive_at(item.fd, "FAILED", value)
    os.fchmod(item.fd, 0o555)
    os.fsync(item.fd)
    if _receipt_entries_at(item.fd) != {"FAILED", "release.anchor", "snapshot.json"}:
        raise AuthorizationError("FAILED terminal entry set changed")
    marker = _parse_canonical_object(_stable_read_at(item.fd, "FAILED", mode=0o444, label="FAILED").payload, "FAILED")
    _validate_marker("FAILED", marker, bundle=bundle, snapshot_sha=snapshot_sha)
    if require_canonical_name:
        if lease is None:
            raise AuthorizationError("full receipt lease is required after FAILED transition")
        _assert_lease_chain(lease, receipt_mode=0o555)


def _seal_complete(lease: ReceiptLease) -> None:
    with _masked_signals():
        _assert_lease_chain(lease)
        if _receipt_entries_at(lease.receipt_fd) != {"RUNNING", "release.anchor", "snapshot.json"}:
            raise AuthorizationError("receipt is not eligible for COMPLETE transition")
        _revalidate_snapshot(lease.bundle, lease.item, lease.snapshot_sha, lease=lease)
        _assert_lease_chain(lease)
        os.unlink("RUNNING", dir_fd=lease.receipt_fd)
        value = {"schema": "ouruniv-cf4-v6-open-parent-key-overlap-read-only-analysis-receipt-complete-v5", "status": "complete_valid_provenance_and_in_memory_analysis_postvalidation", **_marker_common(lease.bundle, lease.snapshot_sha), "analysis_return_validated_in_process": True}
        _write_exclusive_at(lease.receipt_fd, "COMPLETE", value)
        os.fchmod(lease.receipt_fd, 0o555)
        os.fsync(lease.receipt_fd)
        if _receipt_entries_at(lease.receipt_fd) != {"COMPLETE", "release.anchor", "snapshot.json"}:
            raise AuthorizationError("COMPLETE terminal entry set changed")
        marker = _parse_canonical_object(_stable_read_at(lease.receipt_fd, "COMPLETE", mode=0o444, label="COMPLETE").payload, "COMPLETE")
        _validate_marker("COMPLETE", marker, bundle=lease.bundle, snapshot_sha=lease.snapshot_sha)
        _assert_lease_chain(lease, receipt_mode=0o555)


PAIR_ORDER = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
TOP_K = (1, 5, 10, 20, 50)
TOP_LEVEL_RESULT_KEYS = {
    "pair_order", "pair_L1_matrix", "pairs", "maximum_pair",
    "replicate_unique_key_count", "key_mass", "parent_given_key",
    "reconstructed_cache_log_Z_bar", "cache_logmeanexp_max_abs_residual",
    "replicate_cache_log_Z_bar_max_abs_residual", "reconstructed_P_rep",
    "factorization_max_abs_residual", "evidence_pooling_weights",
    "reconstructed_P_pool", "pooling_max_abs_residual", "replicate_to_pool_L1",
}
PAIR_RESULT_KEYS = {
    "pair", "L1", "signed_parent_difference", "parent_contribution",
    "parent_fractional_contribution", "parent_rank_order", "parent_seed_ranked",
    "top_k_cumulative_fraction", "key_score", "key_rank_order", "keys_ranked",
    "key_intersection", "key_union", "key_Jaccard", "weighted_key_overlap",
    "coordinate_marginal_TV",
}


def _exact_int_tuple(value: Any, *, length: int) -> bool:
    return type(value) is tuple and len(value) == length and all(type(item) is int for item in value)


def _sealed_array(value: Any, *, dtype: Any, shape: tuple[int, ...]) -> bool:
    return (
        type(value) is np.ndarray
        and value.dtype == np.dtype(dtype)
        and value.shape == shape
        and value.dtype.hasobject is False
        and value.flags.writeable is False
        and value.base is None
    )


def _valid_top_k_mapping(value: Any) -> bool:
    return (
        type(value) is MappingProxyType
        and len(value) == len(TOP_K)
        and set(value.keys()) == set(TOP_K)
        and all(type(key) is int and type(item) is float for key, item in value.items())
    )


def _valid_pair_row(value: Any, *, pair: tuple[int, int], cache_keys: int) -> bool:
    if type(value) is not MappingProxyType or set(value.keys()) != PAIR_RESULT_KEYS:
        return False
    if not _exact_int_tuple(value.get("pair"), length=2) or value["pair"] != pair:
        return False
    if any(type(value.get(key)) is not float for key in ("L1", "key_Jaccard", "weighted_key_overlap")):
        return False
    if any(type(value.get(key)) is not int for key in ("key_intersection", "key_union")):
        return False
    arrays = (
        ("signed_parent_difference", np.float64, (256,)),
        ("parent_contribution", np.float64, (256,)),
        ("parent_fractional_contribution", np.float64, (256,)),
        ("parent_rank_order", np.intp, (256,)),
        ("parent_seed_ranked", np.int32, (256,)),
        ("key_score", np.float64, (cache_keys,)),
        ("key_rank_order", np.intp, (cache_keys,)),
        ("keys_ranked", np.int16, (cache_keys, 6)),
        ("coordinate_marginal_TV", np.float64, (6,)),
    )
    return all(_sealed_array(value.get(name), dtype=dtype, shape=shape) for name, dtype, shape in arrays) and _valid_top_k_mapping(value.get("top_k_cumulative_fraction"))


def _valid_analysis_result(value: Any) -> bool:
    """Validate the exact immutable analyzer return frozen by the v5 design."""
    if type(value) is not MappingProxyType or set(value.keys()) != TOP_LEVEL_RESULT_KEYS:
        return False
    if type(value.get("pair_order")) is not tuple or value["pair_order"] != PAIR_ORDER:
        return False
    if not all(_exact_int_tuple(pair, length=2) for pair in value["pair_order"]):
        return False
    maximum_pair = value.get("maximum_pair")
    if not _exact_int_tuple(maximum_pair, length=2) or maximum_pair not in PAIR_ORDER:
        return False
    parent_given_key = value.get("parent_given_key")
    if type(parent_given_key) is not np.ndarray or parent_given_key.ndim != 2 or parent_given_key.shape[1:] != (256,):
        return False
    cache_keys = int(parent_given_key.shape[0])
    if cache_keys <= 0 or not _sealed_array(parent_given_key, dtype=np.float64, shape=(cache_keys, 256)):
        return False
    arrays = (
        ("pair_L1_matrix", np.float64, (4, 4)),
        ("replicate_unique_key_count", np.intp, (4,)),
        ("key_mass", np.float64, (4, cache_keys)),
        ("reconstructed_cache_log_Z_bar", np.float64, (cache_keys,)),
        ("replicate_cache_log_Z_bar_max_abs_residual", np.float64, (4,)),
        ("reconstructed_P_rep", np.float64, (4, 256)),
        ("evidence_pooling_weights", np.float64, (4,)),
        ("reconstructed_P_pool", np.float64, (256,)),
        ("replicate_to_pool_L1", np.float64, (4,)),
    )
    if not all(_sealed_array(value.get(name), dtype=dtype, shape=shape) for name, dtype, shape in arrays):
        return False
    if any(type(value.get(name)) is not float for name in ("cache_logmeanexp_max_abs_residual", "factorization_max_abs_residual", "pooling_max_abs_residual")):
        return False
    pairs = value.get("pairs")
    return type(pairs) is tuple and len(pairs) == len(PAIR_ORDER) and all(_valid_pair_row(row, pair=pair, cache_keys=cache_keys) for row, pair in zip(pairs, PAIR_ORDER))


def _derive_runtime_contract(head: str) -> dict[str, Any]:
    base_file = _stable_read(ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_wrapper_design_v1.json", mode=0o644, label="loader design")
    if base_file.sha256 != "6bec76f28dfbbf7eb295eb3e62716a47d925f021d5bb71974a9521c0b08ac73c":
        raise AuthorizationError("loader design changed")
    contract = deepcopy(_parse_canonical_object(base_file.payload, "loader design"))
    contract["git_subprocess_contract"]["expected_HEAD_and_tracking"] = head
    return contract


def _metadata_bundle(*, git_runner: Callable[..., subprocess.CompletedProcess[bytes]], hostname: str, environment: Mapping[str, str]) -> AuthorizationBundle:
    _verify_fixed_local_pins()
    program_file = _stable_read(PROGRAM, mode=0o644, label="program")
    program = _validate_program(program_file)
    _verify_program_implementation_files(program)
    grant_file = _stable_read(GRANT, mode=0o644, label="grant")
    # External metadata is intentionally delayed until local program/grant and Git metadata exist.
    grant_pre = _parse_canonical_object(grant_file.payload, "grant")
    if set(grant_pre) != GRANT_KEYS:
        raise AuthorizationError("grant keyset changed")
    head = _validate_git(grant_file, grant_pre, runner=git_runner)
    release_file = _stable_read(RELEASE, mode=0o444, label="release")
    manifest_file = _stable_read(MANIFEST, mode=0o444, label="manifest")
    grant, release, manifest = _validate_pair(program_file, grant_file, release_file, manifest_file, program)
    if release["payload"].get("implementation_commit") != grant.get("implementation_commit"):
        raise AuthorizationError("release implementation commit differs from the Git-validated grant")
    _validate_host_and_allocation(hostname, environment)
    return AuthorizationBundle(MappingProxyType(program), MappingProxyType(grant), MappingProxyType(release), MappingProxyType(manifest), program_file, grant_file, release_file, manifest_file, head)


def _run_authorized_for_test(*, git_runner: Callable[..., subprocess.CompletedProcess[bytes]], hostname: str, environment: Mapping[str, str], loader_factory: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None, interrupt: Callable[[str], None] | None = None) -> Mapping[str, Any]:
    bundle = _metadata_bundle(git_runner=git_runner, hostname=hostname, environment=environment)
    lease = _bootstrap_receipt(bundle, interrupt=interrupt)
    hook = interrupt or (lambda _: None)
    checkpoint = "during_analysis"
    try:
        hook("before_science")
        _revalidate_snapshot(lease.bundle, lease.item, lease.snapshot_sha, lease=lease)
        contract = _derive_runtime_contract(lease.bundle.head)
        if loader_factory is None:
            loader = _verify_runtime_imports(lease.bundle.program)
            result = loader._run_verified_analysis_for_test(contract, repo_root=ROOT, git_runner=subprocess.run)
        else:
            result = loader_factory(contract)
        hook("after_analysis")
        checkpoint = "during_postvalidation"
        if not _valid_analysis_result(result):
            raise AuthorizationError("analysis return violates the exact v5 immutable schema")
        _revalidate_snapshot(lease.bundle, lease.item, lease.snapshot_sha, lease=lease)
        hook("before_terminal")
        checkpoint = "during_terminal_transition"
        _seal_complete(lease)
        return result
    except BaseException:
        try:
            with _masked_signals():
                try:
                    _assert_lease_chain(lease)
                    canonical_name = True
                except (AuthorizationError, OSError):
                    canonical_name = False
                _seal_failed_at(lease.bundle, lease.item, lease.snapshot_sha, lease=lease, checkpoint=checkpoint, require_canonical_name=canonical_name)
        except (AuthorizationError, OSError):
            pass
        finally:
            raise
    finally:
        lease.close()


def run_authorized_canonical_parent_key_overlap_read_only_analysis_v5() -> Mapping[str, Any]:
    """Run the no-argument, one-shot canonical analysis after every fixed gate."""

    return _run_authorized_for_test(git_runner=subprocess.run, hostname=os.uname().nodename, environment=os.environ)


def _read_terminal_marker(receipt: Path, *, bundle: AuthorizationBundle | None = None) -> tuple[str, dict[str, Any] | None]:
    if not receipt.exists():
        return "not_started", None
    parent_fd = receipt_fd = None
    try:
        parent_fd, receipt_fd, item = _open_receipt_directory(receipt)
        entries = _receipt_entries_at(receipt_fd)
        states = entries & {"RUNNING", "COMPLETE", "FAILED"}
        if len(states) != 1:
            return "invalid_fail_closed", None
        marker = next(iter(states))
        if entries != {marker, "release.anchor", "snapshot.json"}:
            return "invalid_fail_closed", None
        file = _stable_read_at(receipt_fd, marker, mode=0o444, label=marker)
        value = _parse_canonical_object(file.payload, marker)
        snapshot_sha = _stable_read_at(receipt_fd, "snapshot.json", mode=0o444, label="snapshot").sha256
        _validate_marker(marker, value, bundle=bundle, snapshot_sha=snapshot_sha)
        if bundle is not None:
            _revalidate_snapshot(bundle, item, snapshot_sha, require_canonical_name=False)
        expected_mode = 0o555 if marker in {"COMPLETE", "FAILED"} else 0o700
        _assert_named_directory(item, expected_mode=expected_mode)
        return marker.lower(), value
    finally:
        for descriptor in (receipt_fd, parent_fd):
            if isinstance(descriptor, int):
                os.close(descriptor)


def receipt_status() -> str:
    """Return an advisory raw lifecycle label; never attest science success."""

    if not GRANT.exists():
        return "not_started"
    try:
        grant = _parse_canonical_object(_stable_read(GRANT, mode=0o644, label="grant").payload, "grant")
        receipt = _receipt_path(grant)
        state, _ = _read_terminal_marker(receipt)
        if state in {"running", "complete", "failed"}:
            return f"advisory_raw_{state}_untrusted"
        return state
    except (AuthorizationError, OSError):
        return "invalid_fail_closed"


def _supervise_receipt(child_class: str) -> int:
    allowed = {"success_0": 65, "timeout_124": 124, "killed_137": 137, "terminated_143": 143, "other_nonzero": 1}
    if child_class not in allowed:
        raise AuthorizationError("unknown child exit class")
    # A process without the continuously held original FD cannot attest success
    # or mutate lifecycle state.  Inspection is advisory and rejection-only.
    if child_class == "success_0":
        return 65
    try:
        bundle = _metadata_bundle(git_runner=subprocess.run, hostname=os.uname().nodename, environment=os.environ)
        _read_terminal_marker(_receipt_path(bundle.grant), bundle=bundle)
    except (AuthorizationError, OSError):
        pass
    return allowed[child_class]


def main() -> int:
    mode = os.environ.get("OURUNIV_LIFECYCLE_MODE")
    child = os.environ.get("OURUNIV_CHILD_EXIT_CLASS")
    if mode is None and child is None:
        run_authorized_canonical_parent_key_overlap_read_only_analysis_v5()
        return 0
    if mode == "receipt_supervisor_only" and child is not None:
        return _supervise_receipt(child)
    if mode == "receipt_status_only" and child is None:
        print(receipt_status())
        return 0
    raise AuthorizationError("unknown or inherited OURUNIV lifecycle environment")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuthorizationError, PermissionError):
        raise SystemExit(65)
