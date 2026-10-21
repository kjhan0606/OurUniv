"""Fail-closed v5 provenance boundary for aggregate-evidence SMC.

v5 deliberately has no grant, release, manifest, receipt, data, or state in
this change.  The functions below are also used by the runner so that the
same canonical preflight receipt is checked at every lifecycle boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

import cf4_aggregate_evidence_smc_execution as base_execution


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PROGRAM = ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_program_v5.json"
CANONICAL_GRANT = ROOT / "config/cf4_aggregate_evidence_smc_execution_grant_v5.json"
EXTERNAL_RELEASE = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_execution_authorization_v5_release.json")
EXTERNAL_MANIFEST = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_execution_authorization_v5_manifest.json")
DATA_DIRECTORY = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v5")
STATE_DIRECTORY = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v5_run")
RECEIPTS_DIRECTORY = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v5_receipts")
POLICY = 5

AUTHORIZATION_KEYS = (
    "production_SMC_execution_authorized", "oracle_cache_population_authorized",
    "conditional_field_bank_authorized", "candidate_generation_authorized",
    "parent_or_seed_selection_authorized", "PM_authorized", "HOP_authorized",
    "RAMSES_authorized", "downstream_execution_authorized", "automatic_retry_authorized",
    "automatic_retune_authorized", "automatic_scale_up_authorized",
    "automatic_follow_on_authorized",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _full_sha(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise PermissionError(f"{label} is not a full lowercase SHA256")
    return text


def _path(value: Any, label: str) -> Path:
    result = Path(str(value))
    if not result.is_absolute() or "v4" in str(result).lower():
        raise PermissionError(f"{label} is not a new absolute v5 path")
    return result.resolve()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PermissionError(f"{label} is absent")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PermissionError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise PermissionError(f"{label} is not an object")
    return value


def _release_stat(path: Path, *, require_read_only: bool) -> dict[str, int]:
    try:
        value = path.stat()
    except OSError as error:
        raise PermissionError("external release is absent") from error
    if not stat.S_ISREG(value.st_mode):
        raise PermissionError("external release is not a regular file")
    if require_read_only and value.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise PermissionError("external release must be read-only")
    return {"dev": value.st_dev, "ino": value.st_ino, "size": value.st_size, "nlink": value.st_nlink}


def validate_program(program: dict[str, Any]) -> None:
    if program.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-program-v5" \
            or program.get("status") != "frozen_versioned_one_shot_program_execution_unauthorized":
        raise PermissionError("v5 authorization program identity changed")
    if program.get("authorization") != {key: False for key in AUTHORIZATION_KEYS}:
        raise PermissionError("v5 authorization program is not exactly execution-false")
    storage = program.get("storage", {})
    if storage != {
        "data_directory": str(DATA_DIRECTORY), "state_directory": str(STATE_DIRECTORY),
        "receipts_directory": str(RECEIPTS_DIRECTORY), "exclusive_reservation": True,
        "restart_or_checkpoint_import": False,
    }:
        raise PermissionError("v5 storage contract changed")
    base = program.get("base_production_program", {})
    if base.get("path") != "config/cf4_aggregate_evidence_smc_production_program.json" \
            or _full_sha(base.get("sha256"), "base production program") != sha256_file(ROOT / base["path"]):
        raise PermissionError("frozen science program changed")
    if program.get("frozen_science_parameters") != {
        "parent_seed_range_inclusive": [3193, 3448], "parent_count": 256,
        "source": "base_production_program",
    }:
        raise PermissionError("frozen science parameters changed")
    interface = program.get("future_grant_interface", {})
    if interface.get("canonical_path") != str(CANONICAL_GRANT.relative_to(ROOT)) \
            or interface.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-execution-grant-v5" \
            or interface.get("current_grant_present") is not False \
            or interface.get("runtime_grant_path_override_allowed") is not False \
            or interface.get("must_pin_release_path_release_id_payload_and_manifest_hashes") is not True:
        raise PermissionError("v5 grant interface changed")
    external = program.get("external_pre_execution_release", {})
    if external != {
        "canonical_path": str(EXTERNAL_RELEASE), "manifest_path": str(EXTERNAL_MANIFEST),
        "current_release_present": False, "current_manifest_present": False,
        "runtime_path_override_allowed": False,
    }:
        raise PermissionError("v5 external provenance interface changed")


def load_canonical_authorization_program() -> dict[str, Any]:
    if not CANONICAL_PROGRAM.is_file():
        raise PermissionError("canonical v5 authorization program is absent")
    program = _read_object(CANONICAL_PROGRAM, "canonical v5 authorization program")
    validate_program(program)
    return program


def _paired_external_objects() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    release = _read_object(EXTERNAL_RELEASE, "external v5 release")
    manifest = _read_object(EXTERNAL_MANIFEST, "external v5 manifest")
    release_sha, manifest_sha = sha256_file(EXTERNAL_RELEASE), sha256_file(EXTERNAL_MANIFEST)
    expected_release = {
        "schema", "status", "verdict", "release_id", "payload", "payload_sha256",
        "manifest_path", "manifest_sha256", "manifest_id",
    }
    expected_manifest = {
        "schema", "status", "manifest_id", "release_path", "release_id",
        "release_payload_sha256",
    }
    if set(release) != expected_release or set(manifest) != expected_manifest \
            or release.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-release-v5" \
            or manifest.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-manifest-v5" \
            or release.get("status") != "complete_pass_external_postcommit_lineage_audit" \
            or release.get("verdict") != "LINEAGE GO" \
            or manifest.get("status") != "complete_paired_external_manifest":
        raise PermissionError("external v5 release or manifest is unsealed")
    for value, label in ((release.get("release_id"), "release id"),
                         (release.get("payload_sha256"), "release payload hash"),
                         (release.get("manifest_sha256"), "release manifest hash"),
                         (release.get("manifest_id"), "release manifest id"),
                         (manifest.get("manifest_id"), "manifest id"),
                         (manifest.get("release_id"), "manifest release id"),
                         (manifest.get("release_payload_sha256"), "manifest release payload hash")):
        _full_sha(value, label)
    if not isinstance(release.get("payload"), dict) \
            or release["payload_sha256"] != _sha256(release["payload"]) \
            or _path(release.get("manifest_path"), "release manifest path") != EXTERNAL_MANIFEST.resolve() \
            or _path(manifest.get("release_path"), "manifest release path") != EXTERNAL_RELEASE.resolve() \
            or release["manifest_sha256"] != manifest_sha \
            or release["manifest_id"] != manifest["manifest_id"] \
            or release["release_id"] != manifest["release_id"] \
            or manifest["release_payload_sha256"] != release["payload_sha256"]:
        raise PermissionError("external v5 release/manifest pairing changed")
    return release, manifest, release_sha, manifest_sha


def validate_future_grant(grant_path: Path, program: dict[str, Any]) -> dict[str, Any]:
    validate_program(program)
    if Path(grant_path).resolve() != CANONICAL_GRANT.resolve():
        raise PermissionError("v5 grant path is not canonical")
    grant = _read_object(CANONICAL_GRANT, "sealed v5 one-shot grant")
    required = {
        "schema", "status", "one_shot", "authorization_program_sha256", "grant_id",
        "authorization", "data_directory", "state_directory", "receipts_directory",
        "external_release", "precommit_audit_verdict",
    }
    wanted_auth = {key: False for key in AUTHORIZATION_KEYS}
    wanted_auth["production_SMC_execution_authorized"] = True
    wanted_auth["oracle_cache_population_authorized"] = True
    if set(grant) != required or grant.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-execution-grant-v5" \
            or grant.get("status") != "sealed_one_shot_execution_authorization" \
            or grant.get("one_shot") is not True or grant.get("authorization") != wanted_auth \
            or grant.get("precommit_audit_verdict") != "EXECUTION GO" \
            or grant.get("authorization_program_sha256") != sha256_file(CANONICAL_PROGRAM) \
            or grant.get("data_directory") != str(DATA_DIRECTORY) \
            or grant.get("state_directory") != str(STATE_DIRECTORY) \
            or grant.get("receipts_directory") != str(RECEIPTS_DIRECTORY):
        raise PermissionError("sealed v5 one-shot grant is wrong")
    _full_sha(grant.get("grant_id"), "grant id")
    release, manifest, release_sha, manifest_sha = _paired_external_objects()
    pin = grant.get("external_release")
    if not isinstance(pin, dict) or set(pin) != {
        "path", "release_id", "payload_sha256", "manifest_path", "manifest_sha256", "manifest_id"
    } or _path(pin.get("path"), "grant release path") != EXTERNAL_RELEASE.resolve() \
            or _path(pin.get("manifest_path"), "grant manifest path") != EXTERNAL_MANIFEST.resolve() \
            or pin.get("release_id") != release["release_id"] or pin.get("payload_sha256") != release["payload_sha256"] \
            or pin.get("manifest_sha256") != manifest_sha or pin.get("manifest_id") != manifest["manifest_id"]:
        raise PermissionError("grant/release/manifest binding changed")
    return grant


def require_execution_authorization(program: dict[str, Any]) -> dict[str, Any]:
    return validate_future_grant(CANONICAL_GRANT, program)


def _static_snapshot(program: dict[str, Any], grant: dict[str, Any], release: dict[str, Any], manifest: dict[str, Any], release_sha: str, manifest_sha: str, release_stat: dict[str, int]) -> dict[str, Any]:
    return {
        "schema": "ouruniv-cf4-aggregate-evidence-smc-execution-preflight-snapshot-v5",
        "policy": POLICY,
        "release": {"path": str(EXTERNAL_RELEASE), "sha256": release_sha, "release_id": release["release_id"], "stat": release_stat},
        "external_manifest": {"path": str(EXTERNAL_MANIFEST), "sha256": manifest_sha, "manifest_id": manifest["manifest_id"]},
        "grant": {"path": str(CANONICAL_GRANT), "sha256": sha256_file(CANONICAL_GRANT), "payload_sha256": _sha256(grant), "grant_id": grant["grant_id"]},
        "program": {"path": str(CANONICAL_PROGRAM), "sha256": sha256_file(CANONICAL_PROGRAM)},
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        "runner": {"path": str(ROOT / "scripts/run_cf4_aggregate_evidence_smc_authorized_v5_lageunha.sh"), "sha256": sha256_file(ROOT / "scripts/run_cf4_aggregate_evidence_smc_authorized_v5_lageunha.sh")},
        "data_directory": str(DATA_DIRECTORY), "state_directory": str(STATE_DIRECTORY), "receipts_directory": str(RECEIPTS_DIRECTORY),
    }


def create_preflight_receipt(receipt: Path, program: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Atomically reserve a new receipt, hard-link the read-only release, then snapshot."""
    grant = require_execution_authorization(program)
    receipt = Path(receipt).resolve()
    if receipt.parent != RECEIPTS_DIRECTORY.resolve() or receipt.name != "one-shot-receipt":
        raise PermissionError("v5 receipt path is not canonical")
    try:
        receipt.mkdir(mode=0o700)
    except FileExistsError as error:
        raise PermissionError("v5 receipt already exists") from error
    try:
        anchor = receipt / "release.anchor"
        os.link(EXTERNAL_RELEASE, anchor)
        release, manifest, release_sha, manifest_sha = _paired_external_objects()
        release_stat = _release_stat(EXTERNAL_RELEASE, require_read_only=True)
        anchor_stat = _release_stat(anchor, require_read_only=True)
        if anchor_stat != release_stat:
            raise PermissionError("release anchor inode differs from canonical release")
        snapshot = _static_snapshot(program, grant, release, manifest, release_sha, manifest_sha, release_stat)
        snapshot_sha = _sha256(snapshot)
        (receipt / "preflight-snapshot.json").write_bytes(canonical_json(snapshot) + b"\n")
        (receipt / "preflight-snapshot.sha256").write_text(snapshot_sha + "\n")
        return snapshot, snapshot_sha
    except BaseException:
        # A failed receipt remains as an exclusive forensic marker; it is never reused.
        raise


def revalidate_preflight_receipt(receipt: Path, expected_snapshot_sha: str, program: dict[str, Any]) -> dict[str, Any]:
    receipt = Path(receipt).resolve()
    if receipt.parent != RECEIPTS_DIRECTORY.resolve() or receipt.name != "one-shot-receipt":
        raise PermissionError("v5 receipt path is not canonical")
    anchor = receipt / "release.anchor"
    snapshot_path = receipt / "preflight-snapshot.json"
    if not anchor.is_file() or not snapshot_path.is_file() or not (receipt / "preflight-snapshot.sha256").is_file():
        raise PermissionError("v5 preflight receipt is incomplete")
    stored = _read_object(snapshot_path, "v5 preflight snapshot")
    if _sha256(stored) != expected_snapshot_sha or (receipt / "preflight-snapshot.sha256").read_text().strip() != expected_snapshot_sha:
        raise PermissionError("v5 preflight snapshot hash changed")
    grant = require_execution_authorization(program)
    release, manifest, release_sha, manifest_sha = _paired_external_objects()
    release_stat = _release_stat(EXTERNAL_RELEASE, require_read_only=True)
    anchor_stat = _release_stat(anchor, require_read_only=True)
    if release_stat != anchor_stat:
        raise PermissionError("canonical release and hard-link anchor differ")
    current = _static_snapshot(program, grant, release, manifest, release_sha, manifest_sha, release_stat)
    if current != stored:
        raise PermissionError("v5 preflight snapshot does not revalidate exactly")
    return stored


def read_only_science_postcheck(data_directory: Path) -> dict[str, Any]:
    """Validate the published science bundle without mutating it or its manifest.

    A scientific failure is still a valid, sealed scientific outcome.  Every
    other return is an invalid execution/provenance failure and cannot reach a
    COMPLETE marker.
    """
    data_directory = Path(data_directory).resolve()
    result_path = data_directory / "result.json"
    manifest_path = data_directory / "manifest.json"
    if not result_path.is_file() or result_path.stat().st_size == 0 \
            or not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise RuntimeError("published result or manifest is absent or empty")
    manifest_mode = manifest_path.stat().st_mode
    if manifest_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise PermissionError("published manifest must be read-only before COMPLETE")
    checked = base_execution.validate_published_bundle(data_directory)
    status = checked.get("status")
    if status not in {
        "complete_pass_production_smc", "complete_scientific_fail_production_smc",
    } or checked.get("valid_scientific_complete") is not True:
        raise RuntimeError("published bundle is not a valid scientific completion")
    outcome_kind = checked.get("outcome_kind")
    failure_class = checked.get("failure_class")
    if not isinstance(outcome_kind, str) or (status == "complete_pass_production_smc" and failure_class is not None) \
            or (status == "complete_scientific_fail_production_smc" and not isinstance(failure_class, str)):
        raise RuntimeError("published bundle completion classification is invalid")
    return {
        "science_status": status,
        "outcome_kind": outcome_kind,
        "failure_class": "none" if failure_class is None else failure_class,
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def run_authorized_v5(program_path: Path, receipt: Path, snapshot_sha: str) -> dict[str, Any]:
    """Run only after a runner-created receipt has revalidated exactly."""
    if Path(program_path).resolve() != CANONICAL_PROGRAM.resolve():
        raise PermissionError("authorized v5 accepts only the canonical program path")
    program = load_canonical_authorization_program()
    revalidate_preflight_receipt(receipt, snapshot_sha, program)
    base_program = base_execution.load_canonical_program(verify_file_hashes=True)
    result = base_execution._execute_into_reserved_directory(
        base_program, DATA_DIRECTORY,
        validation_runner=base_execution.run_validation,
        evaluator_factory=base_execution._actual_evaluator_factory(base_program),
        control_runner=base_execution.run_sealed_regression_control,
        capability_core=base_execution._run_fixed_capability_core,
    )
    revalidate_preflight_receipt(receipt, snapshot_sha, program)
    return result
