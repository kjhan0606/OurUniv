"""One-shot grammar-debug Slurm authorization boundary for shared-schedule production.

The committed program is intentionally execution-false.  This module performs
all grant, paired external provenance, receipt, and lifecycle checks before it
can call the separately audited private science entry.  Tests use temporary
namespaces and injected science callables; importing this module never writes.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import stat
import subprocess
from typing import Any, Callable, Iterator, Mapping

import cf4_aggregate_evidence_oracle as oracle
import cf4_aggregate_evidence_parallel_oracle as parallel_oracle
import cf4_aggregate_evidence_smc as smc
import cf4_aggregate_evidence_smc_capability as base_capability
import cf4_aggregate_evidence_smc_v6_open_shared_schedule_production as capability
import cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution as execution


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_program_v1.json"
GRANT_RELATIVE = "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_grant_v1.json"
WRAPPER_RESULT_RECORD_RELATIVE = "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_implementation_result_record_v1.json"
WRAPPER_DESIGN = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_design_v1.json"
IMPLEMENTATION_RECORD = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_implementation_result_record.json"
WRAPPER_RESULT_RECORD = ROOT / WRAPPER_RESULT_RECORD_RELATIVE
GRANT = ROOT / GRANT_RELATIVE
RELEASE = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_release_v1.json")
EXTERNAL_MANIFEST = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_manifest_v1.json")
RECEIPT_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v1_receipts")
CACHE_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v1_cache")
DATA_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v1")
STATE_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v1_run")
RUNNER = ROOT / "scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v1.sbatch"
SLURM_LOG_TEMPLATE = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v1_slurm-%j.log")

WRAPPER_DESIGN_SHA256 = "23ab1fb29f5b6779aa096fe631849aeebf89f17ef8ed487792d03851b4c22b36"
IMPLEMENTATION_RECORD_SHA256 = "57ccc0ac99ead2a3903e96e22f0b40182e2a6db3a3b3184a5257d8d8530ba867"
WRAPPER_DESIGN_COMMIT = "8d0dc64d6e9ddbb384900562c9346d34e65f34c9"
BRANCH = "agent/freeze-zoom-pipeline"
REMOTE_REF = f"refs/heads/{BRANCH}"
IMPLEMENTATION_FILES = (
    "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_program_v1.json",
    "src/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_authorized_v1.py",
    "scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v1.sbatch",
    "scripts/status_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v1.sh",
    "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_authorized_v1.py",
    "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v1.py",
)

PROGRAM_KEYS = {
    "schema", "status", "date", "purpose", "lineage", "canonical_paths",
    "fixed_science", "resource_contract", "pair_grant_contract",
    "receipt_contract", "artifact_and_postcheck_contract", "authorization", "next",
}
PROGRAM_AUTHORIZATION_KEYS = {
    "implementation_authorized", "implementation_result_record_authorized", "pair_creation_authorized",
    "grant_creation_authorized", "receipt_creation_authorized",
    "cache_population_authorized", "production_execution_authorized", "Slurm_submission_authorized",
    "retry_resume_requeue_retune_or_scale_up_authorized", "conditional_bank_authorized",
    "candidate_selection_authorized", "PM_authorized", "HOP_authorized",
    "RAMSES_authorized", "downstream_execution_authorized",
    "automatic_follow_on_authorized",
}
SNAPSHOT_KEYS = {
    "schema", "status", "grant_id", "release_id", "manifest_id", "grant_path",
    "grant_sha256", "release_path", "release_sha256", "release_dev", "release_ino",
    "release_size", "release_nlink", "manifest_path", "manifest_sha256",
    "wrapper_program_sha256", "wrapper_source_sha256", "runner_sha256",
    "wrapper_design_sha256", "implementation_result_record_sha256",
    "canonical_paths_digest", "SLURM_JOB_ID", "partition", "node", "nodes",
    "ntasks", "cpus_per_task", "memory_MiB", "submit_time", "start_time",
    "combined_log_path", "combined_log_pre_receipt_dev",
    "combined_log_pre_receipt_ino", "combined_log_pre_receipt_size_bytes",
    "combined_log_pre_receipt_sha256_of_exact_prefix_size_bytes",
    "command", "workdir", "stdout_path", "stderr_path", "time_limit",
    "requeue", "minimum_memory_node", "tres",
}
RUNNING_SCHEDULER_KEYS = {
    "SLURM_JOB_ID", "combined_log_path", "combined_log_pre_receipt_dev",
    "combined_log_pre_receipt_ino", "combined_log_pre_receipt_size_bytes",
    "combined_log_pre_receipt_sha256_of_exact_prefix_size_bytes",
}
TERMINAL_SCHEDULER_KEYS = {
    "SLURM_JOB_ID", "combined_log_path", "combined_log_terminal_dev",
    "combined_log_terminal_ino", "combined_log_terminal_size_bytes",
    "combined_log_terminal_sha256_of_exact_prefix_size_bytes",
}
SCIENTIFIC_STATUSES = {
    "complete_pass_production_smc", "complete_scientific_fail_production_smc",
}


def _receipt_checkpoint(_name: str) -> None:
    """Test injection seam; production behavior is deliberately a no-op."""


class AuthorizationError(PermissionError):
    """Fail-closed provenance or authorization failure."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _full_sha(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise AuthorizationError(f"{label} is not a lowercase full SHA256")
    return text


def _object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuthorizationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path: Path, label: str, *, canonical: bool = False) -> dict[str, Any]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise AuthorizationError(f"{label} is absent or not a regular file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_object_no_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuthorizationError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise AuthorizationError(f"{label} root is not an object")
    if canonical and raw != canonical_json(value) + b"\n":
        raise AuthorizationError(f"{label} is not canonical JSON")
    return value


def _mode(path: Path) -> str:
    return f"{stat.S_IMODE(Path(path).stat().st_mode):04o}"


def _require_file(path: Path, digest: str, mode: str, label: str) -> None:
    if (
        Path(path).is_symlink() or not Path(path).is_file()
        or sha256_file(path) != digest or _mode(path) != mode
    ):
        raise AuthorizationError(f"{label} hash, mode, or type changed")


def _require_import_origins(program: Mapping[str, Any]) -> None:
    pins = program["lineage"]["protected_science_hard_pins"]
    record = _read_json(IMPLEMENTATION_RECORD, "base implementation record")
    execution_rows = {
        row.get("path"): row for row in record.get("committed_files", [])
        if isinstance(row, dict)
    }
    execution_relative = "src/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution.py"
    if execution_relative not in execution_rows:
        raise AuthorizationError("base execution import pin is absent")
    expected = (
        (capability, pins["shared_schedule_capability_source"]),
        (smc, pins["smc_source"]),
        (base_capability, pins["production_capability_source"]),
        (oracle, pins["oracle_source"]),
        (parallel_oracle, pins["parallel_oracle_source"]),
        (execution, {
            "path": execution_relative,
            "sha256": execution_rows[execution_relative]["sha256"],
            "mode": "0644",
        }),
    )
    for module, row in expected:
        wanted = _resolved(row["path"])
        origin = Path(str(getattr(module, "__file__", "")))
        if origin.is_symlink() or origin.resolve() != wanted.resolve():
            raise AuthorizationError(f"import origin changed: {module.__name__}")
        _require_file(wanted, row["sha256"], row["mode"], f"imported {module.__name__}")


def _require_science_worktree_clean() -> None:
    rows = _git(
        "status", "--porcelain=v1", "-z", "--untracked-files=all", "--",
        "config", "src", "scripts", "tests",
    ).split("\0")
    unexpected = [
        row for row in rows if row and not row.startswith("?? scripts/tripwire/")
    ]
    if unexpected:
        raise AuthorizationError("science scope contains a tracked or untracked shadow change")


def _resolved(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(value))).hexdigest()


def load_program() -> dict[str, Any]:
    """Load the unique execution-false grammar-debug program and all science pins."""
    program = _read_json(PROGRAM, "grammar-debug program")
    if set(program) != PROGRAM_KEYS \
            or program.get("schema") != "ouruniv-cf4-v6-open-shared-schedule-production-grammar-debug-execution-program-v1" \
            or program.get("status") != "frozen_grammar_debug_slurm_program_execution_disabled_until_sealed_pair_grant_and_audits":
        raise AuthorizationError("grammar-debug program identity changed")
    if set(program.get("authorization", {})) != PROGRAM_AUTHORIZATION_KEYS or any(
        value is not False for value in program["authorization"].values()
    ):
        raise AuthorizationError("grammar-debug program authorization is not closed")
    _require_file(WRAPPER_DESIGN, WRAPPER_DESIGN_SHA256, "0644", "grammar-debug design")
    _require_file(IMPLEMENTATION_RECORD, IMPLEMENTATION_RECORD_SHA256, "0644", "base implementation record")
    design = _read_json(WRAPPER_DESIGN, "grammar-debug design")
    lineage = program.get("lineage", {})
    if lineage.get("design") != {
        "path": str(WRAPPER_DESIGN.relative_to(ROOT)), "commit": WRAPPER_DESIGN_COMMIT,
        "sha256": WRAPPER_DESIGN_SHA256, "mode": "0644",
    } or lineage.get("base_implementation_commit") != "7eb25554abec278a3710b99aed90e73c39f37b9b":
        raise AuthorizationError("grammar-debug program lineage changed")
    if program.get("fixed_science") != design.get("fixed_science"):
        raise AuthorizationError("fixed science differs from design")
    expected_paths = {
        "future_grant": str(GRANT.relative_to(ROOT)),
        "future_external_release": str(RELEASE),
        "future_external_manifest": str(EXTERNAL_MANIFEST),
        "receipt_root": str(RECEIPT_ROOT), "cache_root": str(CACHE_ROOT),
        "data_root": str(DATA_ROOT), "state_root": str(STATE_ROOT),
        "slurm_combined_log_template": str(SLURM_LOG_TEMPLATE),
    }
    if program.get("canonical_paths") != expected_paths:
        raise AuthorizationError("grammar-debug canonical paths changed")
    pins = design["protected_existing_artifacts"]["science_hard_pins"]
    if lineage.get("protected_science_hard_pins") != pins:
        raise AuthorizationError("protected science hard pins changed")
    for row in pins.values():
        _require_file(_resolved(row["path"]), row["sha256"], row["mode"], row["path"])
    if lineage.get("implementation_files_exact") != list(IMPLEMENTATION_FILES):
        raise AuthorizationError("implementation file surface changed")
    resource = program.get("resource_contract", {})
    wanted_resource = {
        "backend": "Slurm_one_shot_CPU_batch", "partition": "debug",
        "nodelist": "grammar-debug", "hostname_casefold_exact": "grammar-debug",
        "nodes": 1, "ntasks": 1, "cpus_per_task": 12, "memory": "96G",
        "memory_MiB": 98304, "time": "12:00:00", "signal": "B:TERM@300",
        "no_requeue": True, "batch_shebang": "/bin/bash", "slurm_export": "NONE",
        "command": str(RUNNER), "workdir": str(ROOT),
        "stdout_stderr_template": str(SLURM_LOG_TEMPLATE),
        "scontrol_memory_exact": ["96G", "98304M"],
        "minimum_CPU_affinity_count": 12,
        "minimum_MemAvailable_GiB": 80, "minimum_free_GPFS_GiB": 40,
        "worker_processes": 8, "threads_per_worker": 1,
        "replicates_sequential": True,
        "timeout_argv": ["/usr/bin/timeout", "--foreground", "--signal=TERM", "--kill-after=240s", "12h"],
        "CUDA_VISIBLE_DEVICES": "", "MALLOC_ARENA_MAX": "2",
        "Slurm_submission": True, "syn101_execution": False,
        "manual_node_execution": False, "process_table_polling": False,
    }
    if resource != wanted_resource:
        raise AuthorizationError("grammar-debug resource contract changed")
    pair = program.get("pair_grant_contract", {})
    if _canonical_digest(program["fixed_science"]) != pair.get("fixed_science_digest") \
            or _canonical_digest(program["canonical_paths"]) != pair.get("canonical_paths_digest") \
            or _canonical_digest(resource) != pair.get("resource_contract_digest"):
        raise AuthorizationError("program digests changed")
    _require_import_origins(program)
    return program


def _read_pair() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    manifest = _read_json(EXTERNAL_MANIFEST, "external manifest", canonical=True)
    release = _read_json(RELEASE, "external release", canonical=True)
    if _mode(EXTERNAL_MANIFEST) != "0444" or _mode(RELEASE) != "0444":
        raise AuthorizationError("external pair is not mode 0444")
    program = load_program()
    contract = program["pair_grant_contract"]
    if set(manifest) != set(contract["manifest_exact_keys"]) \
            or set(release) != set(contract["release_exact_keys"]):
        raise AuthorizationError("external pair keyset changed")
    if manifest.get("schema") != contract["manifest_schema"] \
            or manifest.get("status") != contract["manifest_status"] \
            or release.get("schema") != contract["release_schema"] \
            or release.get("status") != contract["release_status"]:
        raise AuthorizationError("external pair schema or status changed")
    for value, label in (
        (manifest.get("manifest_id"), "manifest id"),
        (manifest.get("release_id"), "release id"),
        (manifest.get("release_payload_sha256"), "manifest payload SHA"),
        (release.get("release_id"), "release id"),
        (release.get("payload_sha256"), "release payload SHA"),
        (release.get("manifest_id"), "release manifest id"),
        (release.get("manifest_sha256"), "release manifest SHA"),
    ):
        _full_sha(value, label)
    payload = release.get("payload")
    if not isinstance(payload, dict) or set(payload) != set(contract["payload_exact_keys"]):
        raise AuthorizationError("release payload keyset changed")
    payload_sha = _canonical_digest(payload)
    manifest_sha = sha256_file(EXTERNAL_MANIFEST)
    release_sha = sha256_file(RELEASE)
    if (
        payload.get("schema") != contract["payload_schema"]
        or payload.get("status") != contract["payload_status"]
        or release.get("payload_sha256") != payload_sha
        or manifest.get("release_payload_sha256") != payload_sha
        or manifest.get("design_sha256") != WRAPPER_DESIGN_SHA256
        or manifest.get("canonical_paths_digest") != contract["canonical_paths_digest"]
        or manifest.get("resource_contract_digest") != contract["resource_contract_digest"]
        or manifest.get("release_path") != str(RELEASE)
        or release.get("manifest_path") != str(EXTERNAL_MANIFEST)
        or release.get("manifest_sha256") != manifest_sha
        or release.get("manifest_id") != manifest.get("manifest_id")
        or release.get("release_id") != manifest.get("release_id")
        or payload.get("release_id") != release.get("release_id")
        or manifest.get("one_shot") is not True
    ):
        raise AuthorizationError("external pair binding changed")
    if len({release["release_id"], manifest["manifest_id"]}) != 2:
        raise AuthorizationError("release and manifest IDs are not distinct")
    return release, manifest, release_sha, manifest_sha


def _validate_payload(payload: Mapping[str, Any], program: Mapping[str, Any]) -> None:
    contract = program["pair_grant_contract"]
    checks = {
        "design_commit": WRAPPER_DESIGN_COMMIT,
        "design_sha256": WRAPPER_DESIGN_SHA256,
        "fixed_science_digest": contract["fixed_science_digest"],
        "canonical_paths_digest": contract["canonical_paths_digest"],
        "resource_contract_digest": contract["resource_contract_digest"],
        "one_shot": True,
        "authorization": contract["future_runtime_authorization_exact"],
    }
    if any(payload.get(key) != value for key, value in checks.items()):
        raise AuthorizationError("release payload lineage changed")
    _full_sha(payload.get("release_id"), "payload release id")
    _full_sha(payload.get("implementation_result_record_sha256"), "payload result-record SHA")
    if payload.get("release_id") is None or not isinstance(payload.get("implementation_file_sha256_map"), dict):
        raise AuthorizationError("release payload implementation binding changed")


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise AuthorizationError("git lineage check failed") from error


def _git_bytes(*args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise AuthorizationError("git binary lineage check failed") from error


def _validate_grant_git_lineage(program: Mapping[str, Any]) -> dict[str, Any]:
    head = _git("rev-parse", "HEAD")
    result_commit = _git("rev-parse", "HEAD^")
    implementation_commit = _git("rev-parse", "HEAD^^")
    design_commit = _git("rev-parse", "HEAD^^^")
    tracking = _git("rev-parse", "@{upstream}")
    remote_rows = _git("ls-remote", "origin", REMOTE_REF).split()
    if len(remote_rows) != 2 or head != tracking or remote_rows[0] != head:
        raise AuthorizationError("grant commit is not synchronized with local tracking and remote")
    if design_commit != WRAPPER_DESIGN_COMMIT:
        raise AuthorizationError("grant chain is not rooted at the audited wrapper design")
    grant_relative = GRANT_RELATIVE
    result_relative = WRAPPER_RESULT_RECORD_RELATIVE
    if _git("diff-tree", "--no-commit-id", "--name-status", "-r", head).splitlines() != [f"A\t{grant_relative}"]:
        raise AuthorizationError("grant commit is not an exact single-file addition")
    if _git("diff-tree", "--no-commit-id", "--name-status", "-r", result_commit).splitlines() != [f"A\t{result_relative}"]:
        raise AuthorizationError("wrapper result-record commit is not an exact single-file addition")
    implementation_changes = _git(
        "diff-tree", "--no-commit-id", "--name-status", "-r", implementation_commit,
    ).splitlines()
    if set(implementation_changes) != {f"A\t{path}" for path in IMPLEMENTATION_FILES} \
            or len(implementation_changes) != len(IMPLEMENTATION_FILES):
        raise AuthorizationError("wrapper implementation commit is not the exact six additions")
    if _git_bytes("show", f"{head}:{grant_relative}") != GRANT.read_bytes():
        raise AuthorizationError("grant worktree bytes differ from committed grant")
    if _git_bytes("show", f"{result_commit}:{result_relative}") != WRAPPER_RESULT_RECORD.read_bytes():
        raise AuthorizationError("wrapper result-record worktree bytes differ from its commit")

    record_contract = program["lineage"]["future_implementation_result_record"]
    if WRAPPER_RESULT_RECORD.is_symlink() or not WRAPPER_RESULT_RECORD.is_file() \
            or _mode(WRAPPER_RESULT_RECORD) != "0644":
        raise AuthorizationError("wrapper implementation result record changed")
    record = _read_json(
        WRAPPER_RESULT_RECORD, "wrapper implementation result record", canonical=True,
    )
    if set(record) != set(record_contract["exact_keys"]) \
            or record.get("schema") != record_contract["schema"] \
            or record.get("status") != record_contract["status"] \
            or record.get("implementation_commit") != implementation_commit \
            or record.get("implementation_parent_commit") != WRAPPER_DESIGN_COMMIT \
            or record.get("design_commit") != WRAPPER_DESIGN_COMMIT:
        raise AuthorizationError("wrapper implementation result-record identity changed")
    audits = record.get("independent_audits")
    if audits != record_contract["independent_audits_exact"]:
        raise AuthorizationError("wrapper implementation audits are not sealed GO")
    record_authorization = record.get("authorization")
    if record_authorization != record_contract["authorization_exact"]:
        raise AuthorizationError("wrapper implementation record is not runtime-closed")
    result_tree = _git("ls-tree", result_commit, "--", result_relative).split()
    result_raw = WRAPPER_RESULT_RECORD.read_bytes()
    expected_result_blob = hashlib.sha1(
        f"blob {len(result_raw)}\0".encode("ascii") + result_raw,
    ).hexdigest()
    if len(result_tree) < 3 or result_tree[0] != "100644" \
            or result_tree[1] != "blob" or result_tree[2] != expected_result_blob:
        raise AuthorizationError("wrapper result-record tree mode or blob changed")
    rows = record.get("implementation_files")
    if not isinstance(rows, list) or len(rows) != len(IMPLEMENTATION_FILES):
        raise AuthorizationError("wrapper implementation result-record rows changed")
    by_path = {row.get("path"): row for row in rows if isinstance(row, dict)}
    if set(by_path) != set(IMPLEMENTATION_FILES):
        raise AuthorizationError("wrapper implementation result-record path set changed")
    for relative in IMPLEMENTATION_FILES:
        row = by_path[relative]
        if set(row) != {"path", "sha256", "git_blob_oid", "mode"}:
            raise AuthorizationError("wrapper implementation row schema changed")
        raw = _git_bytes("show", f"{implementation_commit}:{relative}")
        tree = _git("ls-tree", implementation_commit, "--", relative).split()
        wanted_mode = "100755" if relative.startswith("scripts/") else "100644"
        if len(tree) < 3 or tree[0] != wanted_mode or tree[1] != "blob" \
                or row["mode"] != wanted_mode or row["git_blob_oid"] != tree[2] \
                or row["sha256"] != hashlib.sha256(raw).hexdigest() \
                or (ROOT / relative).read_bytes() != raw:
            raise AuthorizationError(f"wrapper implementation row changed: {relative}")
    _require_science_worktree_clean()
    return {
        "grant_commit": head,
        "implementation_commit": implementation_commit,
        "implementation_result_record_sha256": sha256_file(WRAPPER_RESULT_RECORD),
        "implementation_file_sha256_map": {
            relative: by_path[relative]["sha256"] for relative in IMPLEMENTATION_FILES
        },
    }


def validate_authorization(program: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the future grant/pair completely without creating anything."""
    if dict(program) != load_program():
        raise AuthorizationError("caller did not supply the canonical wrapper program")
    grant = _read_json(GRANT, "one-shot grant", canonical=True)
    if _mode(GRANT) != "0644":
        raise AuthorizationError("grant mode changed")
    contract = program["pair_grant_contract"]
    if set(grant) != set(contract["grant_exact_keys"]) \
            or grant.get("schema") != contract["grant_schema"] \
            or grant.get("status") != contract["grant_status"] \
            or grant.get("one_shot") is not True:
        raise AuthorizationError("grant schema, keyset, status, or one-shot flag changed")
    for key in ("grant_id", "release_id", "release_payload_sha256", "release_sha256", "manifest_id", "manifest_sha256", "fixed_science_digest", "canonical_paths_digest"):
        _full_sha(grant.get(key), f"grant {key}")
    release, manifest, release_sha, manifest_sha = _read_pair()
    _validate_payload(release["payload"], program)
    grant_id = grant["grant_id"]
    if len({grant_id, release["release_id"], manifest["manifest_id"]}) != 3:
        raise AuthorizationError("grant, release, and manifest IDs are not distinct")
    lineage = _validate_grant_git_lineage(program)
    expected_map = lineage["implementation_file_sha256_map"]
    wanted_auth = contract["future_runtime_authorization_exact"]
    checks = {
        "design_commit": WRAPPER_DESIGN_COMMIT,
        "design_sha256": WRAPPER_DESIGN_SHA256,
        "implementation_commit": lineage["implementation_commit"],
        "implementation_result_record_path": WRAPPER_RESULT_RECORD_RELATIVE,
        "implementation_result_record_sha256": lineage["implementation_result_record_sha256"],
        "implementation_file_sha256_map": expected_map,
        "fixed_science_digest": contract["fixed_science_digest"],
        "canonical_paths_digest": contract["canonical_paths_digest"],
        "resource_contract_digest": contract["resource_contract_digest"],
        "release_path": str(RELEASE), "release_id": release["release_id"],
        "release_payload_sha256": release["payload_sha256"], "release_sha256": release_sha,
        "manifest_path": str(EXTERNAL_MANIFEST), "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_sha, "receipt_root": str(RECEIPT_ROOT),
        "cache_root": str(CACHE_ROOT), "data_root": str(DATA_ROOT),
        "state_root": str(STATE_ROOT),
        "slurm_combined_log_template": str(SLURM_LOG_TEMPLATE),
        "authorization": wanted_auth,
    }
    if any(grant.get(key) != value for key, value in checks.items()):
        raise AuthorizationError("grant lineage, pair binding, paths, or authorization changed")
    payload = release["payload"]
    if payload.get("release_id") != release["release_id"] \
            or payload.get("implementation_commit") != lineage["implementation_commit"] \
            or payload.get("implementation_result_record_sha256") != lineage["implementation_result_record_sha256"] \
            or payload.get("implementation_file_sha256_map") != expected_map \
            or manifest.get("implementation_result_record_sha256") != lineage["implementation_result_record_sha256"]:
        raise AuthorizationError("pair implementation lineage changed")
    return {
        "grant": grant, "release": release, "manifest": manifest,
        "grant_sha256": sha256_file(GRANT), "release_sha256": release_sha,
        "manifest_sha256": manifest_sha, "grant_commit": lineage["grant_commit"],
    }


def _required_decimal_environment(name: str, minimum: int | None = None) -> int:
    value = os.environ.get(name, "")
    if not value.isdecimal():
        raise AuthorizationError(f"{name} is absent or not decimal")
    number = int(value)
    if minimum is not None and number < minimum:
        raise AuthorizationError(f"{name} is below its frozen minimum")
    return number


def _scheduler_log_path(job_id: int) -> Path:
    return Path(str(SLURM_LOG_TEMPLATE).replace("%j", str(job_id)))


def _scheduler_log_checkpoint(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        initial = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(initial.st_mode):
            raise AuthorizationError("scheduler log is not a regular non-symlink")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            size = before.st_size
            digest = hashlib.sha256()
            remaining = size
            while remaining:
                block = os.read(descriptor, min(8 << 20, remaining))
                if not block:
                    raise AuthorizationError("scheduler log prefix was truncated")
                digest.update(block)
                remaining -= len(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise AuthorizationError("scheduler log checkpoint failed") from error
    if (initial.st_dev, initial.st_ino, initial.st_size) != (before.st_dev, before.st_ino, before.st_size) \
            or (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size):
        raise AuthorizationError("scheduler log identity or size was unstable")
    return {
        "path": str(path), "dev": before.st_dev, "ino": before.st_ino,
        "size_bytes": size, "sha256_of_exact_prefix_size_bytes": digest.hexdigest(),
    }


def _revalidate_scheduler_log(record: Mapping[str, Any]) -> None:
    if set(record) != {"path", "dev", "ino", "size_bytes", "sha256_of_exact_prefix_size_bytes"}:
        raise AuthorizationError("scheduler log checkpoint keyset changed")
    path = Path(str(record["path"]))
    current = _scheduler_log_checkpoint(path)
    if current["dev"] != record["dev"] or current["ino"] != record["ino"] \
            or current["size_bytes"] < record["size_bytes"]:
        raise AuthorizationError("scheduler log was replaced or truncated")
    size = int(record["size_bytes"])
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            payload = bytearray()
            remaining = size
            while remaining:
                block = os.read(descriptor, min(8 << 20, remaining))
                if not block:
                    raise AuthorizationError("scheduler log recorded prefix is absent")
                payload.extend(block)
                remaining -= len(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise AuthorizationError("scheduler log prefix revalidation failed") from error
    if (before.st_dev, before.st_ino, before.st_size) != (after.st_dev, after.st_ino, after.st_size) \
            or before.st_dev != record["dev"] or before.st_ino != record["ino"] \
            or before.st_size < size \
            or hashlib.sha256(payload).hexdigest() != record["sha256_of_exact_prefix_size_bytes"]:
        raise AuthorizationError("scheduler log recorded prefix changed")


def _scontrol_job(job_id: int) -> dict[str, str]:
    try:
        output = subprocess.run(
            ["/usr/bin/scontrol", "show", "job", "-o", str(job_id)],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise AuthorizationError("Slurm job provenance is unavailable") from error
    result: dict[str, str] = {}
    for token in output.split():
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def _slurm_memory_mib(value: str) -> int:
    if value.endswith("G") and value[:-1].isdecimal():
        return int(value[:-1]) * 1024
    if value.endswith("M") and value[:-1].isdecimal():
        return int(value[:-1])
    raise AuthorizationError("scontrol memory is not an exact G or M allocation")


def _scheduler_allocation_identity(scheduler: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: scheduler[key] for key in (
        "SLURM_JOB_ID", "partition", "node", "nodes", "ntasks",
        "cpus_per_task", "memory_MiB", "submit_time", "start_time",
        "command", "workdir", "stdout_path", "stderr_path", "time_limit",
        "requeue", "minimum_memory_node", "tres",
    )}
    log = scheduler["combined_log"]
    result.update({
        "combined_log_path": log["path"], "combined_log_dev": log["dev"],
        "combined_log_ino": log["ino"],
    })
    return result


def _read_scheduler_context() -> dict[str, Any]:
    job_id = _required_decimal_environment("SLURM_JOB_ID", 1)
    if os.environ.get("SLURM_JOB_PARTITION") != "debug" \
            or os.environ.get("SLURMD_NODENAME") != "grammar-debug" \
            or os.environ.get("SLURM_JOB_NODELIST") != "grammar-debug" \
            or _required_decimal_environment("SLURM_NNODES") != 1 \
            or _required_decimal_environment("SLURM_NTASKS") != 1 \
            or _required_decimal_environment("SLURM_CPUS_PER_TASK") != 12 \
            or _required_decimal_environment("SLURM_MEM_PER_NODE", 98304) < 98304:
        raise AuthorizationError("Slurm allocation differs from the frozen contract")
    details = _scontrol_job(job_id)
    log_path = _scheduler_log_path(job_id)
    minimum_memory = str(details.get("MinMemoryNode", ""))
    tres = str(details.get("TRES", ""))
    expected = {
        "Partition": "debug", "NodeList": "grammar-debug", "BatchHost": "grammar-debug",
        "NumNodes": "1", "NumCPUs": "12", "NumTasks": "1", "CPUs/Task": "12",
        "Command": str(RUNNER), "WorkDir": str(ROOT),
        "StdOut": str(log_path), "StdErr": str(log_path),
        "TimeLimit": "12:00:00", "Requeue": "0", "BatchFlag": "1",
    }
    if any(details.get(key) != value for key, value in expected.items()) \
            or not details.get("SubmitTime") or not details.get("StartTime") \
            or minimum_memory not in {"96G", "98304M"} \
            or _slurm_memory_mib(minimum_memory) != 98304 \
            or "mem=96G" not in tres.split(","):
        raise AuthorizationError("scontrol allocation provenance changed")
    log = _scheduler_log_checkpoint(log_path)
    return {
        "SLURM_JOB_ID": str(job_id), "partition": "debug", "node": "grammar-debug",
        "nodes": 1, "ntasks": 1, "cpus_per_task": 12,
        "memory_MiB": _required_decimal_environment("SLURM_MEM_PER_NODE", 98304),
        "submit_time": details["SubmitTime"], "start_time": details["StartTime"],
        "command": details["Command"], "workdir": details["WorkDir"],
        "stdout_path": details["StdOut"], "stderr_path": details["StdErr"],
        "time_limit": details["TimeLimit"], "requeue": details["Requeue"],
        "minimum_memory_node": minimum_memory, "tres": tres,
        "combined_log": log,
    }


def _release_identity(path: Path) -> dict[str, int]:
    try:
        value = Path(path).stat()
    except OSError as error:
        raise AuthorizationError("release identity is unavailable") from error
    if not stat.S_ISREG(value.st_mode) or _mode(path) != "0444":
        raise AuthorizationError("release is not a read-only regular file")
    return {
        "dev": value.st_dev, "ino": value.st_ino, "size": value.st_size,
        "nlink": value.st_nlink,
    }


def _snapshot(
    authorization: Mapping[str, Any], scheduler: Mapping[str, Any],
) -> dict[str, Any]:
    identity = _release_identity(RELEASE)
    log = dict(scheduler["combined_log"])
    snapshot = {
        "schema": "ouruniv-cf4-v6-open-shared-schedule-production-grammar-debug-receipt-snapshot-v1",
        "status": "sealed_preexecution_identity_snapshot",
        "grant_id": authorization["grant"]["grant_id"],
        "release_id": authorization["release"]["release_id"],
        "manifest_id": authorization["manifest"]["manifest_id"],
        "grant_path": str(GRANT), "grant_sha256": authorization["grant_sha256"],
        "release_path": str(RELEASE), "release_sha256": authorization["release_sha256"],
        "release_dev": identity["dev"], "release_ino": identity["ino"],
        "release_size": identity["size"], "release_nlink": identity["nlink"],
        "manifest_path": str(EXTERNAL_MANIFEST),
        "manifest_sha256": authorization["manifest_sha256"],
        "wrapper_program_sha256": sha256_file(PROGRAM),
        "wrapper_source_sha256": sha256_file(Path(__file__)),
        "runner_sha256": sha256_file(RUNNER),
        "wrapper_design_sha256": WRAPPER_DESIGN_SHA256,
        "implementation_result_record_sha256": authorization["grant"]["implementation_result_record_sha256"],
        "canonical_paths_digest": authorization["grant"]["canonical_paths_digest"],
        "SLURM_JOB_ID": scheduler["SLURM_JOB_ID"], "partition": scheduler["partition"],
        "node": scheduler["node"], "nodes": scheduler["nodes"],
        "ntasks": scheduler["ntasks"], "cpus_per_task": scheduler["cpus_per_task"],
        "memory_MiB": scheduler["memory_MiB"], "submit_time": scheduler["submit_time"],
        "start_time": scheduler["start_time"], "command": scheduler["command"],
        "workdir": scheduler["workdir"], "stdout_path": scheduler["stdout_path"],
        "stderr_path": scheduler["stderr_path"], "time_limit": scheduler["time_limit"],
        "requeue": scheduler["requeue"],
        "minimum_memory_node": scheduler["minimum_memory_node"], "tres": scheduler["tres"],
        "combined_log_path": log["path"],
        "combined_log_pre_receipt_dev": log["dev"],
        "combined_log_pre_receipt_ino": log["ino"],
        "combined_log_pre_receipt_size_bytes": log["size_bytes"],
        "combined_log_pre_receipt_sha256_of_exact_prefix_size_bytes": log["sha256_of_exact_prefix_size_bytes"],
    }
    if set(snapshot) != SNAPSHOT_KEYS:
        raise AuthorizationError("receipt snapshot keyset changed")
    return snapshot


def canonical_receipt_path(grant_id: str) -> Path:
    return RECEIPT_ROOT / _full_sha(grant_id, "grant id") / "production"


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json(dict(value)) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            Path(path).unlink()
        except OSError:
            pass
        raise
    os.chmod(path, 0o444)


@contextmanager
def _blocked_lifecycle_signals() -> Iterator[None]:
    signals = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    if hasattr(signal, "pthread_sigmask"):
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
    else:
        yield


def _write_marker(receipt: Path, name: str, value: Mapping[str, Any]) -> None:
    for other in {"RUNNING", "COMPLETE", "FAILED"} - {name}:
        candidate = receipt / other
        if candidate.exists():
            candidate.unlink()
    _exclusive_json(receipt / name, value)


def _receipt_failed(
    receipt: Path, authorization: Mapping[str, Any], checkpoint: str,
    snapshot_sha: str | None, scheduler: Mapping[str, Any] | None = None,
) -> None:
    if receipt.is_symlink() or not receipt.is_dir():
        return
    with _blocked_lifecycle_signals():
        os.chmod(receipt, 0o700)
        for marker in ("RUNNING", "COMPLETE", "FAILED"):
            candidate = receipt / marker
            if os.path.lexists(candidate):
                candidate.unlink()
        snapshot_path = receipt / "snapshot.json"
        if snapshot_path.is_file() and not snapshot_path.is_symlink():
            terminal_scheduler = _terminal_scheduler_fields(
                _read_json(snapshot_path, "receipt snapshot", canonical=True)
            )
        elif scheduler is not None:
            terminal_scheduler = _terminal_scheduler_fields_from_context(scheduler)
        else:
            terminal_scheduler = {}
        failed = {
            "schema": "ouruniv-cf4-v6-open-shared-schedule-production-grammar-debug-receipt-marker-v1",
            "status": "failed_invalid_lifecycle_provenance_execution_or_postcheck",
            "grant_id": authorization["grant"]["grant_id"],
            "release_id": authorization["release"]["release_id"],
            "manifest_id": authorization["manifest"]["manifest_id"],
            "snapshot_sha256": snapshot_sha,
            "failure_class": "invalid_lifecycle_provenance_execution_or_postcheck",
            "failed_at_checkpoint": checkpoint,
            "result_manifest_sha256_or_null": None,
            **terminal_scheduler,
        }
        _exclusive_json(receipt / "FAILED", failed)
        os.chmod(receipt, 0o555)


def _scheduler_from_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "SLURM_JOB_ID": snapshot["SLURM_JOB_ID"], "partition": snapshot["partition"],
        "node": snapshot["node"], "nodes": snapshot["nodes"], "ntasks": snapshot["ntasks"],
        "cpus_per_task": snapshot["cpus_per_task"], "memory_MiB": snapshot["memory_MiB"],
        "submit_time": snapshot["submit_time"], "start_time": snapshot["start_time"],
        "command": snapshot["command"], "workdir": snapshot["workdir"],
        "stdout_path": snapshot["stdout_path"], "stderr_path": snapshot["stderr_path"],
        "time_limit": snapshot["time_limit"], "requeue": snapshot["requeue"],
        "minimum_memory_node": snapshot["minimum_memory_node"], "tres": snapshot["tres"],
        "combined_log": {
            "path": snapshot["combined_log_path"],
            "dev": snapshot["combined_log_pre_receipt_dev"],
            "ino": snapshot["combined_log_pre_receipt_ino"],
            "size_bytes": snapshot["combined_log_pre_receipt_size_bytes"],
            "sha256_of_exact_prefix_size_bytes": snapshot["combined_log_pre_receipt_sha256_of_exact_prefix_size_bytes"],
        },
    }


def _scheduler_running_fields(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "SLURM_JOB_ID": snapshot["SLURM_JOB_ID"],
        "combined_log_path": snapshot["combined_log_path"],
        "combined_log_pre_receipt_dev": snapshot["combined_log_pre_receipt_dev"],
        "combined_log_pre_receipt_ino": snapshot["combined_log_pre_receipt_ino"],
        "combined_log_pre_receipt_size_bytes": snapshot["combined_log_pre_receipt_size_bytes"],
        "combined_log_pre_receipt_sha256_of_exact_prefix_size_bytes": snapshot["combined_log_pre_receipt_sha256_of_exact_prefix_size_bytes"],
    }


def _terminal_scheduler_fields(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    initial = _scheduler_from_snapshot(snapshot)["combined_log"]
    try:
        _revalidate_scheduler_log(initial)
        terminal = _scheduler_log_checkpoint(Path(str(initial["path"])))
    except AuthorizationError:
        # Preserve the last sealed scheduler identity in FAILED markers even
        # when the log itself is the provenance failure being reported.
        terminal = initial
    return {
        "SLURM_JOB_ID": snapshot["SLURM_JOB_ID"],
        "combined_log_path": terminal["path"],
        "combined_log_terminal_dev": terminal["dev"],
        "combined_log_terminal_ino": terminal["ino"],
        "combined_log_terminal_size_bytes": terminal["size_bytes"],
        "combined_log_terminal_sha256_of_exact_prefix_size_bytes": terminal["sha256_of_exact_prefix_size_bytes"],
    }


def _terminal_scheduler_fields_from_context(scheduler: Mapping[str, Any]) -> dict[str, Any]:
    log = dict(scheduler["combined_log"])
    try:
        log = _scheduler_log_checkpoint(Path(str(log["path"])))
    except AuthorizationError:
        pass
    return {
        "SLURM_JOB_ID": scheduler["SLURM_JOB_ID"], "combined_log_path": log["path"],
        "combined_log_terminal_dev": log["dev"], "combined_log_terminal_ino": log["ino"],
        "combined_log_terminal_size_bytes": log["size_bytes"],
        "combined_log_terminal_sha256_of_exact_prefix_size_bytes": log["sha256_of_exact_prefix_size_bytes"],
    }


def create_receipt(
    authorization: Mapping[str, Any], scheduler: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], str]:
    """Create the receipt in the audited post-hardlink snapshot order."""
    grant_id = authorization["grant"]["grant_id"]
    receipt = canonical_receipt_path(grant_id)
    created_root = False
    created_parent = False
    snapshot_sha: str | None = None
    checkpoint = "before_receipt_mkdir"
    try:
        with _blocked_lifecycle_signals():
            RECEIPT_ROOT.mkdir(mode=0o700)
            created_root = True
            receipt.parent.mkdir(mode=0o700)
            created_parent = True
            receipt.mkdir(mode=0o700)
            checkpoint = "after_receipt_mkdir"
            _receipt_checkpoint(checkpoint)
            os.link(RELEASE, receipt / "release.anchor")
            checkpoint = "after_release_anchor_link"
            _receipt_checkpoint(checkpoint)
            release_identity = _release_identity(RELEASE)
            anchor_identity = _release_identity(receipt / "release.anchor")
            if release_identity != anchor_identity:
                raise AuthorizationError("release anchor identity differs after hardlink")
            current = validate_authorization(load_program())
            snapshot = _snapshot(current, scheduler)
            _exclusive_json(receipt / "snapshot.json", snapshot)
            snapshot_sha = sha256_file(receipt / "snapshot.json")
            checkpoint = "after_snapshot_seal"
            _receipt_checkpoint(checkpoint)
            running = {
                "schema": "ouruniv-cf4-v6-open-shared-schedule-production-grammar-debug-receipt-marker-v1",
                "status": "running_authorized_shared_schedule_production",
                "grant_id": grant_id, "release_id": current["release"]["release_id"],
                "manifest_id": current["manifest"]["manifest_id"],
                "snapshot_sha256": snapshot_sha,
                **_scheduler_running_fields(snapshot),
            }
            _exclusive_json(receipt / "RUNNING", running)
            checkpoint = "after_RUNNING_seal"
            _receipt_checkpoint(checkpoint)
            revalidate_receipt(receipt, snapshot_sha)
            return receipt, snapshot, snapshot_sha
    except BaseException:
        if receipt.exists():
            _receipt_failed(receipt, authorization, checkpoint, snapshot_sha, scheduler)
        else:
            if created_parent:
                receipt.parent.rmdir()
            if created_root:
                RECEIPT_ROOT.rmdir()
        raise


def revalidate_receipt(
    receipt: Path, expected_snapshot_sha: str,
    scheduler: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    receipt = Path(receipt)
    if receipt != canonical_receipt_path(receipt.parent.name):
        raise AuthorizationError("receipt path is not canonical")
    if receipt.is_symlink() or not receipt.is_dir() or _mode(receipt) != "0700":
        raise AuthorizationError("RUNNING receipt type or mode changed")
    snapshot_path = receipt / "snapshot.json"
    anchor = receipt / "release.anchor"
    if anchor.is_symlink() or snapshot_path.is_symlink() \
            or not anchor.is_file() or not snapshot_path.is_file() \
            or _mode(snapshot_path) != "0444" or _mode(anchor) != "0444":
        raise AuthorizationError("receipt anchor or snapshot changed")
    _full_sha(expected_snapshot_sha, "snapshot SHA")
    if sha256_file(snapshot_path) != expected_snapshot_sha:
        raise AuthorizationError("sealed receipt snapshot hash changed")
    stored = _read_json(snapshot_path, "receipt snapshot", canonical=True)
    if set(stored) != SNAPSHOT_KEYS:
        raise AuthorizationError("receipt snapshot keyset changed")
    stored_scheduler = _scheduler_from_snapshot(stored)
    _revalidate_scheduler_log(stored_scheduler["combined_log"])
    fresh_scheduler = _read_scheduler_context()
    if _scheduler_allocation_identity(fresh_scheduler) != _scheduler_allocation_identity(stored_scheduler):
        raise AuthorizationError("live Slurm allocation differs from sealed receipt")
    if scheduler is not None and _scheduler_allocation_identity(scheduler) != _scheduler_allocation_identity(stored_scheduler):
        raise AuthorizationError("Slurm scheduler provenance differs from receipt")
    authorization = validate_authorization(load_program())
    current = _snapshot(authorization, stored_scheduler)
    if current != stored or _release_identity(anchor) != _release_identity(RELEASE):
        raise AuthorizationError("receipt identity snapshot no longer revalidates")
    allowed = {"release.anchor", "snapshot.json"} | (
        {"RUNNING"} if (receipt / "RUNNING").exists() else set()
    )
    if {entry.name for entry in receipt.iterdir()} != allowed:
        raise AuthorizationError("receipt contains an unlisted entry")
    return authorization


def _state_failed(
    checkpoint: str, authorization: Mapping[str, Any], snapshot_sha: str | None,
    snapshot: Mapping[str, Any] | None = None,
    scheduler: Mapping[str, Any] | None = None,
) -> None:
    if STATE_ROOT.is_symlink() or not STATE_ROOT.is_dir():
        return
    os.chmod(STATE_ROOT, 0o700)
    for marker in ("RUNNING", "COMPLETE", "FAILED"):
        path = STATE_ROOT / marker
        if os.path.lexists(path):
            path.unlink()
    terminal_scheduler = (
        _terminal_scheduler_fields(snapshot) if snapshot is not None
        else _terminal_scheduler_fields_from_context(scheduler) if scheduler is not None
        else {}
    )
    _exclusive_json(STATE_ROOT / "FAILED", {
        "schema": "ouruniv-cf4-v6-open-shared-schedule-production-grammar-debug-state-marker-v1",
        "status": "failed_invalid_lifecycle_provenance_execution_or_postcheck",
        "grant_id": authorization["grant"]["grant_id"],
        "release_id": authorization["release"]["release_id"],
        "manifest_id": authorization["manifest"]["manifest_id"],
        "snapshot_sha256": snapshot_sha,
        "failure_class": "invalid_lifecycle_provenance_execution_or_postcheck",
        "failed_at_checkpoint": checkpoint,
        "result_manifest_sha256_or_null": None,
        **terminal_scheduler,
    })
    os.chmod(STATE_ROOT, 0o555)


def read_only_postcheck(data_root: Path) -> dict[str, Any]:
    data_root = Path(data_root)
    result = data_root / "result.json"
    manifest = data_root / "manifest.json"
    if not result.is_file() or not manifest.is_file() or _mode(manifest) != "0444":
        raise AuthorizationError("published result or read-only manifest is absent")
    checked = execution.validate_published_bundle(data_root)
    if checked.get("status") not in SCIENTIFIC_STATUSES \
            or checked.get("valid_scientific_complete") is not True:
        raise AuthorizationError("published science bundle is not a valid completion")
    return checked


def _terminal_log_record(marker: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": marker["combined_log_path"], "dev": marker["combined_log_terminal_dev"],
        "ino": marker["combined_log_terminal_ino"],
        "size_bytes": marker["combined_log_terminal_size_bytes"],
        "sha256_of_exact_prefix_size_bytes": marker["combined_log_terminal_sha256_of_exact_prefix_size_bytes"],
    }


def _read_only_complete_status() -> dict[str, Any]:
    """Revalidate a COMPLETE marker, terminal receipt, and full science bundle."""
    authorization = validate_authorization(load_program())
    state_marker = STATE_ROOT / "COMPLETE"
    if STATE_ROOT.is_symlink() or not STATE_ROOT.is_dir() or _mode(STATE_ROOT) != "0555" \
            or DATA_ROOT.is_symlink() or not DATA_ROOT.is_dir() or _mode(DATA_ROOT) != "0555" \
            or CACHE_ROOT.is_symlink() or not CACHE_ROOT.is_dir() or _mode(CACHE_ROOT) != "0555" \
            or _mode(state_marker) != "0444":
        raise AuthorizationError("terminal state type or mode changed")
    state_value = _read_json(state_marker, "state COMPLETE", canonical=True)
    if set(state_value) != {
        "schema", "status", "grant_id", "snapshot_sha256",
        "release_id", "manifest_id", "result_manifest_sha256_or_null", "science_status",
    } | TERMINAL_SCHEDULER_KEYS \
            or state_value.get("schema") != "ouruniv-cf4-v6-open-shared-schedule-production-grammar-debug-state-marker-v1" \
            or state_value.get("status") != "complete_valid_provenance_and_scientific_postcheck" \
            or state_value.get("science_status") not in SCIENTIFIC_STATUSES:
        raise AuthorizationError("state COMPLETE contract changed")
    receipt = canonical_receipt_path(authorization["grant"]["grant_id"])
    if receipt.is_symlink() or not receipt.is_dir() or _mode(receipt) != "0555" \
            or {item.name for item in receipt.iterdir()} != {
                "release.anchor", "snapshot.json", "COMPLETE",
            }:
        raise AuthorizationError("terminal receipt entry set changed")
    for name in ("release.anchor", "snapshot.json", "COMPLETE"):
        item = receipt / name
        if item.is_symlink() or not item.is_file() or _mode(item) != "0444":
            raise AuthorizationError("terminal receipt type or mode changed")
    receipt_value = _read_json(receipt / "COMPLETE", "receipt COMPLETE", canonical=True)
    if set(receipt_value) != {
        "schema", "status", "grant_id", "release_id", "manifest_id",
        "snapshot_sha256", "result_manifest_sha256_or_null",
    } | TERMINAL_SCHEDULER_KEYS \
            or receipt_value.get("schema") != "ouruniv-cf4-v6-open-shared-schedule-production-grammar-debug-receipt-marker-v1" \
            or receipt_value.get("status") != "complete_valid_provenance_and_scientific_postcheck":
        raise AuthorizationError("receipt COMPLETE contract changed")
    snapshot_sha = sha256_file(receipt / "snapshot.json")
    manifest_sha = sha256_file(DATA_ROOT / "manifest.json")
    if (
        state_value["grant_id"] != authorization["grant"]["grant_id"]
        or receipt_value["grant_id"] != authorization["grant"]["grant_id"]
        or receipt_value["release_id"] != authorization["release"]["release_id"]
        or receipt_value["manifest_id"] != authorization["manifest"]["manifest_id"]
        or state_value["snapshot_sha256"] != snapshot_sha
        or receipt_value["snapshot_sha256"] != snapshot_sha
        or state_value["result_manifest_sha256_or_null"] != manifest_sha
        or receipt_value["result_manifest_sha256_or_null"] != manifest_sha
        or _release_identity(receipt / "release.anchor") != _release_identity(RELEASE)
    ):
        raise AuthorizationError("terminal marker, receipt, or manifest binding changed")
    if any(state_value[key] != receipt_value[key] for key in TERMINAL_SCHEDULER_KEYS) \
            or state_value.get("release_id") != receipt_value.get("release_id") \
            or state_value.get("manifest_id") != receipt_value.get("manifest_id"):
        raise AuthorizationError("terminal scheduler or identity markers differ")
    _revalidate_scheduler_log(_terminal_log_record(receipt_value))
    checked = read_only_postcheck(DATA_ROOT)
    if checked["status"] != state_value["science_status"]:
        raise AuthorizationError("state science status differs from full postcheck")
    return {
        "status": "complete", "science_status": checked["status"],
        "manifest_sha256": manifest_sha, "snapshot_sha256": snapshot_sha,
    }


def _failure_identity(grant_id: str) -> tuple[Path, dict[str, Any], str | None]:
    receipt = canonical_receipt_path(grant_id)
    snapshot_path = receipt / "snapshot.json"
    if snapshot_path.is_file() and not snapshot_path.is_symlink():
        snapshot = _read_json(snapshot_path, "receipt snapshot", canonical=True)
        if set(snapshot) != SNAPSHOT_KEYS or snapshot.get("grant_id") != grant_id:
            raise AuthorizationError("failure receipt snapshot changed")
        snapshot_sha: str | None = sha256_file(snapshot_path)
        release_id = snapshot["release_id"]
        manifest_id = snapshot["manifest_id"]
    else:
        grant = _read_json(GRANT, "grant for failed lifecycle", canonical=True)
        if grant.get("grant_id") != grant_id:
            raise AuthorizationError("failed lifecycle grant identity changed")
        snapshot_sha = None
        release_id = grant.get("release_id")
        manifest_id = grant.get("manifest_id")
        _full_sha(release_id, "failed lifecycle release id")
        _full_sha(manifest_id, "failed lifecycle manifest id")
    authorization = {
        "grant": {"grant_id": grant_id},
        "release": {"release_id": release_id},
        "manifest": {"manifest_id": manifest_id},
    }
    return receipt, authorization, snapshot_sha


def _supervisor_force_failed(grant_id: str, checkpoint: str) -> None:
    """Seal an orphaned timeout/KILL lifecycle without starting science."""
    grant_id = _full_sha(grant_id, "supervisor grant id")
    receipt = canonical_receipt_path(grant_id)
    if not os.path.lexists(receipt):
        return
    if receipt.is_symlink() or not receipt.is_dir():
        raise AuthorizationError("supervisor found an invalid receipt type")
    receipt, authorization, snapshot_sha = _failure_identity(grant_id)
    snapshot_path = receipt / "snapshot.json"
    snapshot = (
        _read_json(snapshot_path, "supervisor receipt snapshot", canonical=True)
        if snapshot_path.is_file() and not snapshot_path.is_symlink() else None
    )
    try:
        scheduler = _read_scheduler_context()
    except AuthorizationError:
        scheduler = None
    receipt_error: BaseException | None = None
    try:
        _receipt_failed(receipt, authorization, checkpoint, snapshot_sha, scheduler)
    except BaseException as error:
        receipt_error = error
    try:
        _state_failed(checkpoint, authorization, snapshot_sha, snapshot, scheduler)
    finally:
        if receipt_error is not None:
            raise receipt_error


def _read_only_failed_status(*, allow_state_absent: bool = False) -> dict[str, Any]:
    grant = _read_json(GRANT, "grant for FAILED status", canonical=True)
    grant_id = _full_sha(grant.get("grant_id"), "FAILED grant id")
    receipt = canonical_receipt_path(grant_id)
    receipt_marker = receipt / "FAILED"
    if receipt.is_symlink() or not receipt.is_dir() or _mode(receipt) != "0555" \
            or receipt_marker.is_symlink() or not receipt_marker.is_file() \
            or _mode(receipt_marker) != "0444":
        raise AuthorizationError("FAILED receipt type or mode changed")
    receipt_value = _read_json(receipt_marker, "receipt FAILED", canonical=True)
    failed_keys = {
        "schema", "status", "grant_id", "release_id", "manifest_id",
        "snapshot_sha256", "failure_class", "failed_at_checkpoint",
        "result_manifest_sha256_or_null",
    } | TERMINAL_SCHEDULER_KEYS
    if set(receipt_value) != failed_keys \
            or receipt_value.get("schema") != "ouruniv-cf4-v6-open-shared-schedule-production-grammar-debug-receipt-marker-v1" \
            or receipt_value.get("status") != "failed_invalid_lifecycle_provenance_execution_or_postcheck" \
            or receipt_value.get("grant_id") != grant_id \
            or receipt_value.get("release_id") != grant.get("release_id") \
            or receipt_value.get("manifest_id") != grant.get("manifest_id"):
        raise AuthorizationError("receipt FAILED schema or identity changed")
    allowed = {"FAILED"}
    anchor = receipt / "release.anchor"
    snapshot_path = receipt / "snapshot.json"
    if os.path.lexists(anchor):
        if anchor.is_symlink() or not anchor.is_file() or _mode(anchor) != "0444" \
                or _release_identity(anchor) != _release_identity(RELEASE):
            raise AuthorizationError("FAILED release anchor changed")
        allowed.add("release.anchor")
    if os.path.lexists(snapshot_path):
        if snapshot_path.is_symlink() or not snapshot_path.is_file() \
                or _mode(snapshot_path) != "0444" or "release.anchor" not in allowed:
            raise AuthorizationError("FAILED snapshot type, mode, or order changed")
        allowed.add("snapshot.json")
        snapshot = _read_json(snapshot_path, "FAILED receipt snapshot", canonical=True)
        if set(snapshot) != SNAPSHOT_KEYS \
                or snapshot.get("grant_id") != grant_id \
                or snapshot.get("release_id") != receipt_value.get("release_id") \
                or snapshot.get("manifest_id") != receipt_value.get("manifest_id") \
                or snapshot.get("grant_sha256") != sha256_file(GRANT) \
                or snapshot.get("release_sha256") != sha256_file(RELEASE) \
                or snapshot.get("manifest_sha256") != sha256_file(EXTERNAL_MANIFEST):
            raise AuthorizationError("FAILED snapshot identity or hash binding changed")
        _revalidate_scheduler_log(_scheduler_from_snapshot(snapshot)["combined_log"])
        snapshot_sha = sha256_file(snapshot_path)
        if receipt_value.get("snapshot_sha256") != snapshot_sha:
            raise AuthorizationError("FAILED snapshot hash binding changed")
    elif receipt_value.get("snapshot_sha256") is not None:
        raise AuthorizationError("pre-snapshot FAILED has a snapshot hash")
    if {item.name for item in receipt.iterdir()} != allowed:
        raise AuthorizationError("FAILED receipt contains an unlisted entry")
    _revalidate_scheduler_log(_terminal_log_record(receipt_value))
    if allow_state_absent and not os.path.lexists(STATE_ROOT):
        return {"status": "failed", "failure_class": receipt_value["failure_class"]}
    state_marker = STATE_ROOT / "FAILED"
    if STATE_ROOT.is_symlink() or not STATE_ROOT.is_dir() or _mode(STATE_ROOT) != "0555" \
            or state_marker.is_symlink() or not state_marker.is_file() \
            or _mode(state_marker) != "0444":
        raise AuthorizationError("FAILED state type or mode changed")
    state_value = _read_json(state_marker, "state FAILED", canonical=True)
    if set(state_value) != failed_keys \
            or state_value.get("schema") != "ouruniv-cf4-v6-open-shared-schedule-production-grammar-debug-state-marker-v1" \
            or any(
        state_value.get(key) != receipt_value.get(key) for key in failed_keys - {"schema"}
    ):
        raise AuthorizationError("state and receipt FAILED markers are not bound")
    return {"status": "failed", "failure_class": receipt_value["failure_class"]}


def _read_only_running_status() -> dict[str, Any]:
    authorization = validate_authorization(load_program())
    marker = STATE_ROOT / "RUNNING"
    if STATE_ROOT.is_symlink() or not STATE_ROOT.is_dir() or _mode(STATE_ROOT) != "0700" \
            or DATA_ROOT.is_symlink() or not DATA_ROOT.is_dir() or _mode(DATA_ROOT) != "0700" \
            or CACHE_ROOT.is_symlink() or not CACHE_ROOT.is_dir() or _mode(CACHE_ROOT) != "0700" \
            or marker.is_symlink() or not marker.is_file() or _mode(marker) != "0444":
        raise AuthorizationError("RUNNING state or runtime namespace changed")
    value = _read_json(marker, "state RUNNING", canonical=True)
    if set(value) != {
        "schema", "status", "grant_id", "release_id", "manifest_id", "snapshot_sha256",
    } | RUNNING_SCHEDULER_KEYS \
            or value.get("schema") != "ouruniv-cf4-v6-open-shared-schedule-production-grammar-debug-state-marker-v1" \
            or value.get("status") != "running_authorized_shared_schedule_production" \
            or value.get("grant_id") != authorization["grant"]["grant_id"]:
        raise AuthorizationError("state RUNNING schema or identity changed")
    receipt = canonical_receipt_path(value["grant_id"])
    revalidate_receipt(receipt, value["snapshot_sha256"])
    receipt_running = _read_json(receipt / "RUNNING", "receipt RUNNING", canonical=True)
    if set(receipt_running) != {
        "schema", "status", "grant_id", "release_id", "manifest_id", "snapshot_sha256",
    } | RUNNING_SCHEDULER_KEYS \
            or receipt_running.get("schema") != "ouruniv-cf4-v6-open-shared-schedule-production-grammar-debug-receipt-marker-v1" \
            or any(
        receipt_running.get(key) != value.get(key)
        for key in ({"status", "grant_id", "release_id", "manifest_id", "snapshot_sha256"} | RUNNING_SCHEDULER_KEYS)
    ):
        raise AuthorizationError("state and receipt RUNNING identities differ")
    return {"status": "running", "snapshot_sha256": value["snapshot_sha256"]}


def _reserve_runtime() -> None:
    for path in (STATE_ROOT, CACHE_ROOT, DATA_ROOT):
        try:
            path.mkdir(mode=0o700)
        except BaseException:
            raise AuthorizationError(f"runtime reservation failed: {path}")


def _require_resources() -> None:
    try:
        available_kib = next(
            int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
    except (OSError, StopIteration, ValueError) as error:
        raise AuthorizationError("MemAvailable cannot be verified") from error
    if available_kib < 80 * 1024 * 1024:
        raise AuthorizationError("MemAvailable is below 80 GiB")
    probe = DATA_ROOT.parent
    if shutil.disk_usage(probe).free < 40 * 1024**3:
        raise AuthorizationError("free disk is below 40 GiB")
    try:
        affinity = os.sched_getaffinity(0)
    except (AttributeError, OSError) as error:
        raise AuthorizationError("CPU affinity cannot be verified") from error
    if len(affinity) < 12 or (os.cpu_count() or 0) < 12:
        raise AuthorizationError("fewer than twelve allocated CPUs are available")


def _require_runtime_environment() -> None:
    wanted = {
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2", "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1", "PYTHONDONTWRITEBYTECODE": "1",
    }
    if any(os.environ.get(key) != value for key, value in wanted.items()):
        raise AuthorizationError("fixed thread, allocator, or CUDA environment changed")
    if any(os.environ.get(key) not in (None, "") for key in (
        "BASH_ENV", "ENV", "PYTHONSTARTUP", "LD_PRELOAD",
    )):
        raise AuthorizationError("inherited shell or loader environment is not empty")


def _host_short_ascii_lower() -> str:
    raw = os.uname().nodename.split(".", 1)[0]
    try:
        raw.encode("ascii")
    except UnicodeEncodeError as error:
        raise AuthorizationError("hostname is not ASCII") from error
    return raw.translate(str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"))


@contextmanager
def _termination_as_exception() -> Iterator[None]:
    previous: dict[signal.Signals, Any] = {}

    def interrupted(signum: int, _frame: Any) -> None:
        raise InterruptedError(f"received termination signal {signum}")

    for item in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[item] = signal.getsignal(item)
        signal.signal(item, interrupted)
    try:
        yield
    finally:
        for item, handler in previous.items():
            signal.signal(item, handler)


def run_authorized_production() -> dict[str, Any]:
    """Sole public entry; no runtime parameter or path override is accepted."""
    program = load_program()
    authorization = validate_authorization(program)
    scheduler = _read_scheduler_context()
    if _host_short_ascii_lower() != "grammar-debug":
        raise AuthorizationError("authorized production requires grammar-debug")
    _require_resources()
    _require_runtime_environment()
    for path in (DATA_ROOT, STATE_ROOT, RECEIPT_ROOT, CACHE_ROOT):
        if os.path.lexists(path):
            raise AuthorizationError("one-shot runtime namespace is not absent")
    receipt: Path | None = None
    snapshot: dict[str, Any] | None = None
    snapshot_sha: str | None = None
    receipt_descriptor: int | None = None
    checkpoint = "pre_receipt"
    try:
        with _termination_as_exception():
            receipt, snapshot, snapshot_sha = create_receipt(authorization, scheduler)
            receipt_descriptor = os.open(receipt, os.O_RDONLY | os.O_DIRECTORY)
            fcntl.flock(receipt_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            checkpoint = "before_runtime_reservation"
            revalidate_receipt(receipt, snapshot_sha, scheduler)
            _reserve_runtime()
            checkpoint = "after_state_reservation"
            _exclusive_json(STATE_ROOT / "RUNNING", {
                "schema": "ouruniv-cf4-v6-open-shared-schedule-production-grammar-debug-state-marker-v1",
                "status": "running_authorized_shared_schedule_production",
                "grant_id": authorization["grant"]["grant_id"],
                "release_id": authorization["release"]["release_id"],
                "manifest_id": authorization["manifest"]["manifest_id"],
                "snapshot_sha256": snapshot_sha,
                **_scheduler_running_fields(snapshot),
            })
            checkpoint = "before_science_core"
            _receipt_checkpoint("after_state_reservation")
            revalidate_receipt(receipt, snapshot_sha, scheduler)
            _require_resources()
            _require_runtime_environment()
            base_program = execution.load_canonical_program(verify_file_hashes=True)
            contract = capability.load_frozen_contract()
            result = execution._execute_reserved_canonical_private(
                base_program, contract, DATA_ROOT, CACHE_ROOT,
            )
            checkpoint = "after_science_core"
            revalidate_receipt(receipt, snapshot_sha, scheduler)
            checked = read_only_postcheck(DATA_ROOT)
            result_manifest_sha = sha256_file(DATA_ROOT / "manifest.json")
            checkpoint = "before_terminal_marker"
            revalidate_receipt(receipt, snapshot_sha, scheduler)
            terminal_scheduler = _terminal_scheduler_fields(snapshot)
            with _blocked_lifecycle_signals():
                for marker in (STATE_ROOT / "RUNNING", receipt / "RUNNING"):
                    marker.unlink()
                state_complete = {
                    "schema": "ouruniv-cf4-v6-open-shared-schedule-production-grammar-debug-state-marker-v1",
                    "status": "complete_valid_provenance_and_scientific_postcheck",
                    "grant_id": authorization["grant"]["grant_id"],
                    "release_id": authorization["release"]["release_id"],
                    "manifest_id": authorization["manifest"]["manifest_id"],
                    "snapshot_sha256": snapshot_sha,
                    "result_manifest_sha256_or_null": result_manifest_sha,
                    "science_status": checked["status"],
                    **terminal_scheduler,
                }
                receipt_complete = {
                    "schema": "ouruniv-cf4-v6-open-shared-schedule-production-grammar-debug-receipt-marker-v1",
                    "status": "complete_valid_provenance_and_scientific_postcheck",
                    "grant_id": authorization["grant"]["grant_id"],
                    "release_id": authorization["release"]["release_id"],
                    "manifest_id": authorization["manifest"]["manifest_id"],
                    "snapshot_sha256": snapshot_sha,
                    "result_manifest_sha256_or_null": result_manifest_sha,
                    **terminal_scheduler,
                }
                _exclusive_json(STATE_ROOT / "COMPLETE", state_complete)
                _exclusive_json(receipt / "COMPLETE", receipt_complete)
                for path in (DATA_ROOT, STATE_ROOT, CACHE_ROOT, receipt):
                    os.chmod(path, 0o555)
        return dict(result)
    except BaseException:
        if receipt is not None:
            receipt_failure: BaseException | None = None
            try:
                _receipt_failed(receipt, authorization, checkpoint, snapshot_sha, scheduler)
            except BaseException as error:
                receipt_failure = error
            try:
                _state_failed(checkpoint, authorization, snapshot_sha, snapshot, scheduler)
            finally:
                if receipt_failure is not None:
                    raise receipt_failure
        raise
    finally:
        if receipt_descriptor is not None:
            os.close(receipt_descriptor)


def main() -> None:
    import sys
    if sys.argv[1:]:
        raise AuthorizationError("grammar-debug public entry accepts no arguments")
    run_authorized_production()


if __name__ == "__main__":
    main()


__all__ = ["main", "run_authorized_production"]
