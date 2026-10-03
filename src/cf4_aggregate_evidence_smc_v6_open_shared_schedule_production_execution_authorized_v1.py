"""One-shot authorization boundary for v6-open shared-schedule production.

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

import cf4_aggregate_evidence_smc_v6_open_shared_schedule_production as capability
import cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution as execution


ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_program_v1.json"
GRANT_RELATIVE = "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_grant.json"
WRAPPER_RESULT_RECORD_RELATIVE = "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_wrapper_implementation_result_record_v1.json"
WRAPPER_DESIGN = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_wrapper_design_v1.json"
PAIR_DESIGN = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_grant_release_manifest_design_v1.json"
AUTHORIZATION_DESIGN = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorization_design_v1.json"
IMPLEMENTATION_RECORD = ROOT / "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_implementation_result_record.json"
WRAPPER_RESULT_RECORD = ROOT / WRAPPER_RESULT_RECORD_RELATIVE
GRANT = ROOT / GRANT_RELATIVE
RELEASE = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_execution_release.json")
EXTERNAL_MANIFEST = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_execution_manifest.json")
RECEIPT_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_receipts")
CACHE_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_cache")
DATA_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1")
STATE_ROOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_shared_schedule_production_v1_run")
RUNNER = ROOT / "scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_authorized_v1_lageunha.sh"

WRAPPER_DESIGN_SHA256 = "3769ee888de4e01f267355bd1aff6f32e9dc43099bf459f0f03d40a3b55729d8"
PAIR_DESIGN_SHA256 = "ca9011f2135ed621ccd2f1bc9336aa924e6db1c82ed672bedff1b73270d48d9b"
AUTHORIZATION_DESIGN_SHA256 = "eed1f831b629166eff68a51f87b4cb7fc796abf2a5e5e7a676451681ebc78cc1"
IMPLEMENTATION_RECORD_SHA256 = "57ccc0ac99ead2a3903e96e22f0b40182e2a6db3a3b3184a5257d8d8530ba867"
WRAPPER_DESIGN_COMMIT = "12f38110a98417501da9da2810296dfb64029451"
BRANCH = "agent/freeze-zoom-pipeline"
REMOTE_REF = f"refs/heads/{BRANCH}"
IMPLEMENTATION_FILES = (
    "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_program_v1.json",
    "src/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_v1.py",
    "scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_authorized_v1_lageunha.sh",
    "scripts/launch_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_authorized_v1_lageunha.sh",
    "scripts/status_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_authorized_v1.sh",
    "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_v1.py",
    "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_authorized_v1_runner.py",
)

PROGRAM_KEYS = {
    "schema", "status", "date", "purpose", "lineage", "canonical_paths",
    "fixed_science", "resource_contract", "grant_release_manifest_contract",
    "receipt_contract", "artifact_and_postcheck_contract", "authorization", "next",
}
PROGRAM_AUTHORIZATION_KEYS = {
    "wrapper_implementation_authorized", "pair_creation_authorized",
    "grant_creation_authorized", "receipt_creation_authorized",
    "cache_population_authorized", "production_execution_authorized",
    "retry_resume_retune_or_scale_up_authorized", "conditional_bank_authorized",
    "candidate_selection_authorized", "PM_authorized", "HOP_authorized",
    "RAMSES_authorized", "downstream_execution_authorized",
    "automatic_follow_on_authorized",
}
SNAPSHOT_KEYS = {
    "schema", "status", "grant_id", "release_id", "manifest_id", "grant_path",
    "grant_sha256", "release_path", "release_sha256", "release_dev", "release_ino",
    "release_size", "release_nlink", "manifest_path", "manifest_sha256",
    "wrapper_program_sha256", "wrapper_source_sha256", "runner_sha256",
    "wrapper_design_sha256", "grant_pair_design_sha256",
    "implementation_result_record_sha256", "canonical_paths_digest",
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


def _resolved(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(value))).hexdigest()


def load_program() -> dict[str, Any]:
    """Load the unique execution-false wrapper program and all inherited pins."""
    program = _read_json(PROGRAM, "authorized wrapper program")
    if (
        set(program) != PROGRAM_KEYS
        or program.get("schema")
        != "ouruniv-cf4-v6-open-shared-schedule-production-execution-authorized-program-v1"
        or program.get("status")
        != "frozen_authorized_wrapper_program_execution_disabled_until_separate_grant_pair_and_release_audits"
    ):
        raise AuthorizationError("authorized wrapper program identity changed")
    if set(program.get("authorization", {})) != PROGRAM_AUTHORIZATION_KEYS or any(
        value is not False for value in program["authorization"].values()
    ):
        raise AuthorizationError("wrapper program authorization is not exactly closed")

    pinned = (
        (WRAPPER_DESIGN, WRAPPER_DESIGN_SHA256, "wrapper design"),
        (PAIR_DESIGN, PAIR_DESIGN_SHA256, "grant/pair design"),
        (AUTHORIZATION_DESIGN, AUTHORIZATION_DESIGN_SHA256, "authorization design"),
        (IMPLEMENTATION_RECORD, IMPLEMENTATION_RECORD_SHA256, "implementation record"),
    )
    for path, digest, label in pinned:
        _require_file(path, digest, "0644", label)

    authorization_design = _read_json(AUTHORIZATION_DESIGN, "authorization design")
    if program.get("fixed_science") != authorization_design.get("fixed_science"):
        raise AuthorizationError("fixed science differs from authorization design")
    if program.get("canonical_paths") != authorization_design.get("canonical_paths"):
        raise AuthorizationError("canonical paths differ from authorization design")
    lineage = program.get("lineage", {})
    if not isinstance(lineage, dict) or lineage.get("implementation_commit") != "7eb25554abec278a3710b99aed90e73c39f37b9b":
        raise AuthorizationError("wrapper program implementation lineage changed")
    expected_rows = {
        row["path"]: row["sha256"]
        for row in authorization_design.get("implementation_hard_pins", [])
    }
    if lineage.get("protected_file_sha256") != expected_rows:
        raise AuthorizationError("wrapper program protected-file map changed")
    for relative, digest in expected_rows.items():
        _require_file(ROOT / relative, digest, "0755" if relative.startswith("scripts/") else "0644", relative)

    expected_paths = {
        "future_grant": str(GRANT.relative_to(ROOT)),
        "future_external_release": str(RELEASE),
        "future_external_manifest": str(EXTERNAL_MANIFEST),
        "receipt_root": str(RECEIPT_ROOT), "cache_root": str(CACHE_ROOT),
        "data_root": str(DATA_ROOT), "state_root": str(STATE_ROOT),
    }
    if program.get("canonical_paths") != expected_paths:
        raise AuthorizationError("canonical storage paths changed")
    digest_contract = program.get("grant_release_manifest_contract", {})
    if _canonical_digest(program["fixed_science"]) != digest_contract.get("fixed_science_digest") \
            or _canonical_digest(program["canonical_paths"]) != digest_contract.get("canonical_paths_digest"):
        raise AuthorizationError("fixed-science or canonical-path digest changed")
    resource = program.get("resource_contract", {})
    if resource.get("worker_processes") != 8 or resource.get("threads_per_worker") != 1 \
            or resource.get("cpus_required") != 8 \
            or resource.get("MemAvailable_minimum_GiB") != 80 \
            or resource.get("free_disk_minimum_GiB") != 40 \
            or resource.get("MALLOC_ARENA_MAX") != "2" \
            or resource.get("Slurm_submission") is not False \
            or resource.get("syn101_execution") is not False:
        raise AuthorizationError("wrapper resource contract changed")
    return program


def _read_pair() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    manifest = _read_json(EXTERNAL_MANIFEST, "external manifest", canonical=True)
    release = _read_json(RELEASE, "external release", canonical=True)
    if _mode(EXTERNAL_MANIFEST) != "0444" or _mode(RELEASE) != "0444":
        raise AuthorizationError("external pair is not mode 0444")
    pair_design = _read_json(PAIR_DESIGN, "grant/pair design")
    manifest_contract = pair_design["external_manifest_contract"]
    release_contract = pair_design["external_release_contract"]
    if set(manifest) != set(manifest_contract["exact_keys"]) \
            or set(release) != set(release_contract["exact_keys"]):
        raise AuthorizationError("external pair keyset changed")
    if manifest.get("schema") != manifest_contract["schema"] \
            or manifest.get("status") != manifest_contract["required_status"] \
            or release.get("schema") != release_contract["schema"] \
            or release.get("status") != release_contract["required_status"]:
        raise AuthorizationError("external pair schema or status changed")
    for value, label in (
        (manifest.get("manifest_id"), "manifest id"),
        (manifest.get("release_id"), "manifest release id"),
        (manifest.get("release_payload_sha256"), "manifest payload SHA"),
        (release.get("release_id"), "release id"),
        (release.get("payload_sha256"), "release payload SHA"),
        (release.get("manifest_id"), "release manifest id"),
        (release.get("manifest_sha256"), "release manifest SHA"),
    ):
        _full_sha(value, label)
    payload = release.get("payload")
    payload_contract = pair_design["release_payload_contract"]
    if not isinstance(payload, dict) or set(payload) != set(payload_contract["exact_keys"]):
        raise AuthorizationError("release payload keyset changed")
    payload_sha = _canonical_digest(payload)
    manifest_sha = sha256_file(EXTERNAL_MANIFEST)
    release_sha = sha256_file(RELEASE)
    if (
        payload.get("schema") != payload_contract["schema"]
        or payload.get("status") != payload_contract["required_status"]
        or release.get("payload_sha256") != payload_sha
        or manifest.get("release_payload_sha256") != payload_sha
        or manifest.get("authorization_design_sha256") != AUTHORIZATION_DESIGN_SHA256
        or manifest.get("grant_release_manifest_design_sha256") != PAIR_DESIGN_SHA256
        or manifest.get("implementation_result_record_sha256") != IMPLEMENTATION_RECORD_SHA256
        or manifest.get("canonical_paths_digest")
        != pair_design["derived_digest_contract"]["canonical_paths_digest"]["expected_sha256"]
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
    pair_design = _read_json(PAIR_DESIGN, "grant/pair design")
    protected = program["lineage"]["protected_file_sha256"]
    names = {
        "program_sha256": "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_program.json",
        "source_sha256": "src/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution.py",
        "runner_sha256": "scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_lageunha.sh",
        "launcher_sha256": "scripts/launch_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_lageunha.sh",
        "status_script_sha256": "scripts/status_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production.sh",
        "execution_test_sha256": "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution.py",
        "runner_test_sha256": "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_runner.py",
    }
    for key, relative in names.items():
        if payload.get(key) != protected[relative]:
            raise AuthorizationError(f"release payload {key} changed")
    wanted = pair_design["release_payload_contract"]["future_authorization_matrix"]
    if any(payload.get(key) is not value for key, value in wanted.items()):
        raise AuthorizationError("release payload authorization matrix changed")
    checks = {
        "authorization_design_commit": "e6ba2c3482855a6c8c16aa8068df83bbfb9c62e8",
        "authorization_design_sha256": AUTHORIZATION_DESIGN_SHA256,
        "grant_release_manifest_design_commit": "3d0a2d2eed3dd9fbfc0cc88d4a705586c156021f",
        "grant_release_manifest_design_sha256": PAIR_DESIGN_SHA256,
        "implementation_commit": "7eb25554abec278a3710b99aed90e73c39f37b9b",
        "implementation_result_record_sha256": IMPLEMENTATION_RECORD_SHA256,
        "execution_design_sha256": "08d99219b88a232dc809b3a2c945381cbbcda1fac0c7202c1c2681a09be609aa",
        "fixed_science_digest": program["grant_release_manifest_contract"]["fixed_science_digest"],
        "canonical_paths_digest": program["grant_release_manifest_contract"]["canonical_paths_digest"],
        "one_shot": True,
    }
    if any(payload.get(key) != value for key, value in checks.items()):
        raise AuthorizationError("release payload lineage changed")


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


def _validate_grant_git_lineage(program: Mapping[str, Any]) -> str:
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
        raise AuthorizationError("wrapper implementation commit is not the exact seven additions")
    if _git_bytes("show", f"{head}:{grant_relative}") != GRANT.read_bytes():
        raise AuthorizationError("grant worktree bytes differ from committed grant")
    if _git_bytes("show", f"{result_commit}:{result_relative}") != WRAPPER_RESULT_RECORD.read_bytes():
        raise AuthorizationError("wrapper result-record worktree bytes differ from its commit")

    record_contract = program["lineage"]["future_wrapper_implementation_result_record"]
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
            or record.get("wrapper_design_commit") != WRAPPER_DESIGN_COMMIT:
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
    if _git("status", "--porcelain", "--untracked-files=no", "--", "config", "src", "scripts", "tests"):
        raise AuthorizationError("science-scoped worktree is dirty")
    return head


def validate_authorization(program: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the future grant/pair completely without creating anything."""
    if dict(program) != load_program():
        raise AuthorizationError("caller did not supply the canonical wrapper program")
    grant = _read_json(GRANT, "one-shot grant", canonical=True)
    if _mode(GRANT) != "0644":
        raise AuthorizationError("grant mode changed")
    pair_design = _read_json(PAIR_DESIGN, "grant/pair design")
    contract = pair_design["local_grant_contract"]
    if set(grant) != set(contract["exact_keys"]) \
            or grant.get("schema") != contract["schema"] \
            or grant.get("status") != contract["required_status"] \
            or grant.get("one_shot") is not True:
        raise AuthorizationError("grant schema, keyset, status, or one-shot flag changed")
    for key in ("grant_id", "release_id", "release_payload_sha256", "release_sha256", "manifest_id", "manifest_sha256", "fixed_science_digest", "canonical_paths_digest"):
        _full_sha(grant.get(key), f"grant {key}")
    release, manifest, release_sha, manifest_sha = _read_pair()
    _validate_payload(release["payload"], program)
    grant_id = grant["grant_id"]
    if len({grant_id, release["release_id"], manifest["manifest_id"]}) != 3:
        raise AuthorizationError("grant, release, and manifest IDs are not distinct")
    expected_map = program["lineage"]["protected_file_sha256"]
    wanted_auth = contract["authorization_exact"]
    checks = {
        "authorization_design_path": str(AUTHORIZATION_DESIGN.relative_to(ROOT)),
        "authorization_design_commit": "e6ba2c3482855a6c8c16aa8068df83bbfb9c62e8",
        "authorization_design_sha256": AUTHORIZATION_DESIGN_SHA256,
        "grant_release_manifest_design_path": str(PAIR_DESIGN.relative_to(ROOT)),
        "grant_release_manifest_design_commit": "3d0a2d2eed3dd9fbfc0cc88d4a705586c156021f",
        "grant_release_manifest_design_sha256": PAIR_DESIGN_SHA256,
        "implementation_commit": "7eb25554abec278a3710b99aed90e73c39f37b9b",
        "implementation_result_record_path": str(IMPLEMENTATION_RECORD.relative_to(ROOT)),
        "implementation_result_record_sha256": IMPLEMENTATION_RECORD_SHA256,
        "implementation_file_sha256_map": expected_map,
        "fixed_science_digest": program["grant_release_manifest_contract"]["fixed_science_digest"],
        "canonical_paths_digest": program["grant_release_manifest_contract"]["canonical_paths_digest"],
        "release_path": str(RELEASE), "release_id": release["release_id"],
        "release_payload_sha256": release["payload_sha256"], "release_sha256": release_sha,
        "manifest_path": str(EXTERNAL_MANIFEST), "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest_sha, "receipt_root": str(RECEIPT_ROOT),
        "cache_root": str(CACHE_ROOT), "data_root": str(DATA_ROOT),
        "state_root": str(STATE_ROOT), "authorization": wanted_auth,
    }
    if any(grant.get(key) != value for key, value in checks.items()):
        raise AuthorizationError("grant lineage, pair binding, paths, or authorization changed")
    grant_commit = _validate_grant_git_lineage(program)
    return {
        "grant": grant, "release": release, "manifest": manifest,
        "grant_sha256": sha256_file(GRANT), "release_sha256": release_sha,
        "manifest_sha256": manifest_sha, "grant_commit": grant_commit,
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


def _snapshot(authorization: Mapping[str, Any]) -> dict[str, Any]:
    identity = _release_identity(RELEASE)
    snapshot = {
        "schema": "ouruniv-cf4-v6-open-shared-schedule-production-receipt-snapshot-v1",
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
        "grant_pair_design_sha256": PAIR_DESIGN_SHA256,
        "implementation_result_record_sha256": IMPLEMENTATION_RECORD_SHA256,
        "canonical_paths_digest": authorization["grant"]["canonical_paths_digest"],
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
    snapshot_sha: str | None,
) -> None:
    if receipt.is_symlink() or not receipt.is_dir():
        return
    with _blocked_lifecycle_signals():
        os.chmod(receipt, 0o700)
        for marker in ("RUNNING", "COMPLETE", "FAILED"):
            candidate = receipt / marker
            if os.path.lexists(candidate):
                candidate.unlink()
        failed = {
            "schema": "ouruniv-cf4-v6-open-shared-schedule-production-receipt-marker-v1",
            "status": "failed_invalid_lifecycle_provenance_execution_or_postcheck",
            "grant_id": authorization["grant"]["grant_id"],
            "release_id": authorization["release"]["release_id"],
            "manifest_id": authorization["manifest"]["manifest_id"],
            "snapshot_sha256": snapshot_sha,
            "failure_class": "invalid_lifecycle_provenance_execution_or_postcheck",
            "failed_at_checkpoint": checkpoint,
        }
        _exclusive_json(receipt / "FAILED", failed)
        os.chmod(receipt, 0o555)


def create_receipt(authorization: Mapping[str, Any]) -> tuple[Path, dict[str, Any], str]:
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
            snapshot = _snapshot(current)
            _exclusive_json(receipt / "snapshot.json", snapshot)
            snapshot_sha = sha256_file(receipt / "snapshot.json")
            checkpoint = "after_snapshot_seal"
            _receipt_checkpoint(checkpoint)
            running = {
                "schema": "ouruniv-cf4-v6-open-shared-schedule-production-receipt-marker-v1",
                "status": "running_authorized_shared_schedule_production",
                "grant_id": grant_id, "release_id": current["release"]["release_id"],
                "manifest_id": current["manifest"]["manifest_id"],
                "snapshot_sha256": snapshot_sha,
            }
            _exclusive_json(receipt / "RUNNING", running)
            checkpoint = "after_RUNNING_seal"
            _receipt_checkpoint(checkpoint)
            revalidate_receipt(receipt, snapshot_sha)
            return receipt, snapshot, snapshot_sha
    except BaseException:
        if receipt.exists():
            _receipt_failed(receipt, authorization, checkpoint, snapshot_sha)
        else:
            if created_parent:
                receipt.parent.rmdir()
            if created_root:
                RECEIPT_ROOT.rmdir()
        raise


def revalidate_receipt(receipt: Path, expected_snapshot_sha: str) -> dict[str, Any]:
    receipt = Path(receipt)
    if receipt != canonical_receipt_path(receipt.parent.name):
        raise AuthorizationError("receipt path is not canonical")
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
    authorization = validate_authorization(load_program())
    current = _snapshot(authorization)
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
) -> None:
    if STATE_ROOT.is_symlink() or not STATE_ROOT.is_dir():
        return
    os.chmod(STATE_ROOT, 0o700)
    for marker in ("RUNNING", "COMPLETE", "FAILED"):
        path = STATE_ROOT / marker
        if os.path.lexists(path):
            path.unlink()
    _exclusive_json(STATE_ROOT / "FAILED", {
        "schema": "ouruniv-cf4-v6-open-shared-schedule-production-state-marker-v1",
        "status": "failed_invalid_lifecycle_provenance_execution_or_postcheck",
        "grant_id": authorization["grant"]["grant_id"],
        "release_id": authorization["release"]["release_id"],
        "manifest_id": authorization["manifest"]["manifest_id"],
        "snapshot_sha256": snapshot_sha,
        "failure_class": "invalid_lifecycle_provenance_execution_or_postcheck",
        "failed_at_checkpoint": checkpoint,
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
        "result_manifest_sha256", "science_status",
    } or state_value.get("status") != "complete_valid_provenance_and_scientific_postcheck" \
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
        "snapshot_sha256", "result_manifest_sha256",
    } or receipt_value.get("status") != "complete_valid_provenance_and_scientific_postcheck":
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
        or state_value["result_manifest_sha256"] != manifest_sha
        or receipt_value["result_manifest_sha256"] != manifest_sha
        or _release_identity(receipt / "release.anchor") != _release_identity(RELEASE)
    ):
        raise AuthorizationError("terminal marker, receipt, or manifest binding changed")
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
    receipt_error: BaseException | None = None
    try:
        _receipt_failed(receipt, authorization, checkpoint, snapshot_sha)
    except BaseException as error:
        receipt_error = error
    try:
        _state_failed(checkpoint, authorization, snapshot_sha)
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
    }
    if set(receipt_value) != failed_keys \
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
        snapshot_sha = sha256_file(snapshot_path)
        if receipt_value.get("snapshot_sha256") != snapshot_sha:
            raise AuthorizationError("FAILED snapshot hash binding changed")
    elif receipt_value.get("snapshot_sha256") is not None:
        raise AuthorizationError("pre-snapshot FAILED has a snapshot hash")
    if {item.name for item in receipt.iterdir()} != allowed:
        raise AuthorizationError("FAILED receipt contains an unlisted entry")
    if allow_state_absent and not os.path.lexists(STATE_ROOT):
        return {"status": "failed", "failure_class": receipt_value["failure_class"]}
    state_marker = STATE_ROOT / "FAILED"
    if STATE_ROOT.is_symlink() or not STATE_ROOT.is_dir() or _mode(STATE_ROOT) != "0555" \
            or state_marker.is_symlink() or not state_marker.is_file() \
            or _mode(state_marker) != "0444":
        raise AuthorizationError("FAILED state type or mode changed")
    state_value = _read_json(state_marker, "state FAILED", canonical=True)
    if set(state_value) != failed_keys or any(
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
    if set(value) != {"schema", "status", "grant_id", "snapshot_sha256"} \
            or value.get("status") != "running_authorized_shared_schedule_production" \
            or value.get("grant_id") != authorization["grant"]["grant_id"]:
        raise AuthorizationError("state RUNNING schema or identity changed")
    receipt = canonical_receipt_path(value["grant_id"])
    revalidate_receipt(receipt, value["snapshot_sha256"])
    receipt_running = _read_json(receipt / "RUNNING", "receipt RUNNING", canonical=True)
    if receipt_running.get("snapshot_sha256") != value["snapshot_sha256"]:
        raise AuthorizationError("state and receipt RUNNING snapshots differ")
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
    if len(affinity) < 8 or (os.cpu_count() or 0) < 8:
        raise AuthorizationError("fewer than eight CPUs are available")


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


def run_authorized_production(program_path: Path) -> dict[str, Any]:
    """Sole public entry; no runtime parameter or path override is accepted."""
    if Path(program_path).resolve() != PROGRAM.resolve():
        raise AuthorizationError("authorized wrapper accepts only its canonical program")
    program = load_program()
    authorization = validate_authorization(program)
    if _host_short_ascii_lower() != "lageunha":
        raise AuthorizationError("authorized production requires Lageunha")
    _require_resources()
    _require_runtime_environment()
    for path in (DATA_ROOT, STATE_ROOT, RECEIPT_ROOT, CACHE_ROOT):
        if os.path.lexists(path):
            raise AuthorizationError("one-shot runtime namespace is not absent")
    receipt: Path | None = None
    snapshot_sha: str | None = None
    receipt_descriptor: int | None = None
    checkpoint = "pre_receipt"
    try:
        with _termination_as_exception():
            receipt, _, snapshot_sha = create_receipt(authorization)
            receipt_descriptor = os.open(receipt, os.O_RDONLY | os.O_DIRECTORY)
            fcntl.flock(receipt_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            checkpoint = "before_runtime_reservation"
            revalidate_receipt(receipt, snapshot_sha)
            _reserve_runtime()
            checkpoint = "after_state_reservation"
            _exclusive_json(STATE_ROOT / "RUNNING", {
                "schema": "ouruniv-cf4-v6-open-shared-schedule-production-state-marker-v1",
                "status": "running_authorized_shared_schedule_production",
                "grant_id": authorization["grant"]["grant_id"],
                "snapshot_sha256": snapshot_sha,
            })
            checkpoint = "before_science_core"
            _receipt_checkpoint("after_state_reservation")
            revalidate_receipt(receipt, snapshot_sha)
            _require_resources()
            _require_runtime_environment()
            base_program = execution.load_canonical_program(verify_file_hashes=True)
            contract = capability.load_frozen_contract()
            result = execution._execute_reserved_canonical_private(
                base_program, contract, DATA_ROOT, CACHE_ROOT,
            )
            checkpoint = "after_science_core"
            revalidate_receipt(receipt, snapshot_sha)
            checked = read_only_postcheck(DATA_ROOT)
            result_manifest_sha = sha256_file(DATA_ROOT / "manifest.json")
            checkpoint = "before_terminal_marker"
            revalidate_receipt(receipt, snapshot_sha)
            with _blocked_lifecycle_signals():
                for marker in (STATE_ROOT / "RUNNING", receipt / "RUNNING"):
                    marker.unlink()
                state_complete = {
                    "schema": "ouruniv-cf4-v6-open-shared-schedule-production-state-marker-v1",
                    "status": "complete_valid_provenance_and_scientific_postcheck",
                    "grant_id": authorization["grant"]["grant_id"],
                    "snapshot_sha256": snapshot_sha,
                    "result_manifest_sha256": result_manifest_sha,
                    "science_status": checked["status"],
                }
                receipt_complete = {
                    "schema": "ouruniv-cf4-v6-open-shared-schedule-production-receipt-marker-v1",
                    "status": "complete_valid_provenance_and_scientific_postcheck",
                    "grant_id": authorization["grant"]["grant_id"],
                    "release_id": authorization["release"]["release_id"],
                    "manifest_id": authorization["manifest"]["manifest_id"],
                    "snapshot_sha256": snapshot_sha,
                    "result_manifest_sha256": result_manifest_sha,
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
                _receipt_failed(receipt, authorization, checkpoint, snapshot_sha)
            except BaseException as error:
                receipt_failure = error
            try:
                _state_failed(checkpoint, authorization, snapshot_sha)
            finally:
                if receipt_failure is not None:
                    raise receipt_failure
        raise
    finally:
        if receipt_descriptor is not None:
            os.close(receipt_descriptor)


__all__ = ["run_authorized_production"]
