"""Read-only, additive v6-open preflight; it cannot start scientific work."""
from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import stat
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_design_v6_open_preflight.json"
PROGRAM = ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_program_v6_open_preflight.json"
LIFT = ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_preflight_lift_v6_open.json"
GRANT = ROOT / "config/cf4_aggregate_evidence_smc_execution_grant_v6_open.json"
RELEASE = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_execution_authorization_v6_open_release.json")
MANIFEST = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_execution_authorization_v6_open_manifest.json")
RECEIPTS = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_receipts")
PILOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_disposable_pilot")
DATA = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open")
STATE = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_run")

DESIGN_SHA = "fc202c6fcbca7a077221529a99a1195df39473837a8238534a9927bab88a7451"
PROGRAM_SHA = "facd7126b6a2abc312838ccd25f93d87b7077cc07c4c611e3d34003a8def5b17"
LIFT_SHA = "cdd53e7454f8f1512e044a60a2522314820d075129778768cfe557c4ed419a9a"
GRANT_SHA = "384b9c1fc0a9ccfe46cf1ff6b3a7524995e0c12f7017a048bb06fcebef8ea4d8"
GRANT_ID = "8f6689e85bd4e19fe114cda1dd1051f46af259b79c5f52d95ab6d697641b5f7d"
RELEASE_SHA = "32da63e6a27622f1ae57bd4b431ddf9892a69db566a824fbc517a5201c2c5bc3"
RELEASE_ID = "dc66fe4bd14d38ded3acf66d54673321e0a13516d23a53da48798eecd3317c35"
PAYLOAD_SHA = "81cd280301abfe75a29c8fc8fba33a60ed492945b03cf9b14f09de574d9ced2a"
MANIFEST_SHA = "5c946a100dfd3fea1eca4dcec00f5347960252fabc4eddc62a9f0b3eb73ef23d"
MANIFEST_ID = "82fbc16750c505f718f0bebbaf1080f96ebb44a8a39ba305e5f25713f81d7e27"
OLD_DESIGN_SHA = "162ee1122ad0d756420021a6855872dde0643d54a0d21131fd08a82da4cdca1e"
OLD_PROGRAM_SHA = "77f10acd607dfe55d8bced94ad6ae036a3997a291780f569ddf4028db03d1765"
OLD_SOURCE_SHA = "920b3b44a4580e19defd572c83ff53967e2e7bbb26661a75a1b73e2c27400e03"
RUNTIME_ROOTS = (RECEIPTS, PILOT, DATA, STATE)
AUTH = {"preflight_only_authorized": True, "pilot_stage_validation_authorized": True,
        "pilot_execution_authorized": False, "production_stage_authorized": False,
        "receipt_creation_authorized": False, "cache_population_authorized": False,
        "downstream_execution_authorized": False, "automatic_follow_on_authorized": False,
        "runtime_root_mutation_forbidden": True}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PermissionError(f"{label} is absent")
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(f"{label} is invalid") from error
    if not isinstance(value, dict):
        raise PermissionError(f"{label} is invalid")
    return value


def _full_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PermissionError(f"{label} is not a full SHA-256")
    return value


def _canonical_path(raw: Any, expected: Path, label: str) -> None:
    if not isinstance(raw, str):
        raise PermissionError(f"{label} path is invalid")
    try:
        expected_text = str(expected.relative_to(ROOT))
    except ValueError:
        expected_text = str(expected)
    if raw != expected_text or any(old in raw.lower() for old in ("v4", "v5")) or "v6_open" not in raw:
        raise PermissionError(f"{label} path is not canonical v6-open")


def _require_mode(path: Path, expected: int, label: str) -> None:
    if stat.S_IMODE(path.stat().st_mode) != expected:
        raise PermissionError(f"{label} mode is not {expected:04o}")


def ascii_lower_short_hostname(hostname: str) -> str:
    short = hostname.split(".", 1)[0]
    if not short or any(ord(char) > 127 for char in short):
        raise PermissionError("hostname is not ASCII")
    return "".join(chr(ord(char) + 32) if "A" <= char <= "Z" else char for char in short)


def _check_config_hashes() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(DESIGN) != DESIGN_SHA or sha256_file(PROGRAM) != PROGRAM_SHA or sha256_file(LIFT) != LIFT_SHA:
        raise PermissionError("preflight configuration hash mismatch")
    design, program, lift = (_read_json(DESIGN, "preflight design"), _read_json(PROGRAM, "preflight program"), _read_json(LIFT, "preflight lift"))
    if design.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-design-v6-open-preflight" or design.get("authorization") != AUTH:
        raise PermissionError("preflight design changed")
    if program.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-program-v6-open-preflight" or program.get("authorization") != AUTH:
        raise PermissionError("preflight program changed")
    if lift.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-preflight-lift-v6-open" or lift.get("authorization") != AUTH:
        raise PermissionError("preflight lift changed")
    if program.get("design") != {"path": str(DESIGN.relative_to(ROOT)), "sha256": DESIGN_SHA} or program.get("lift") != {"path": str(LIFT.relative_to(ROOT)), "sha256": LIFT_SHA}:
        raise PermissionError("preflight program design or lift binding changed")
    return design, program, lift


def _check_program_pins(design: dict[str, Any], program: dict[str, Any], lift: dict[str, Any]) -> None:
    for document, pins in ((design, document_pins := design.get("hard_pins")), (program, program.get("sealed_v6_open_hard_pins"))):
        if not isinstance(pins, dict):
            raise PermissionError("sealed v6-open hard pins are invalid")
    expected = (("sealed_grant", "sha256", GRANT_SHA), ("external_release", "sha256", RELEASE_SHA), ("external_manifest", "sha256", MANIFEST_SHA),
                ("sealed_open_design", "sha256", OLD_DESIGN_SHA), ("sealed_open_program", "sha256", OLD_PROGRAM_SHA), ("sealed_open_source", "sha256", OLD_SOURCE_SHA))
    for key, field, value in expected:
        if design["hard_pins"].get(key, {}).get(field) != value:
            raise PermissionError("preflight design hard pin changed")
    program_names = {"grant": GRANT_SHA, "release": RELEASE_SHA, "manifest": MANIFEST_SHA, "old_design": OLD_DESIGN_SHA, "old_program": OLD_PROGRAM_SHA, "old_source": OLD_SOURCE_SHA}
    if any(program["sealed_v6_open_hard_pins"].get(name, {}).get("sha256") != value for name, value in program_names.items()):
        raise PermissionError("preflight program hard pin changed")
    lift_pins = lift.get("hard_pins", {})
    for field, value in (("grant_sha256", GRANT_SHA), ("release_sha256", RELEASE_SHA), ("manifest_sha256", MANIFEST_SHA), ("sealed_open_design_sha256", OLD_DESIGN_SHA), ("sealed_open_program_sha256", OLD_PROGRAM_SHA), ("sealed_open_source_sha256", OLD_SOURCE_SHA)):
        if lift_pins.get(field) != value:
            raise PermissionError("preflight lift hard pin changed")
    if design.get("fixed_science") != program.get("fixed_science"):
        raise PermissionError("fixed science changed")
    for raw in program.get("paths", {}).values():
        _canonical_path(raw, {str(GRANT.relative_to(ROOT)): GRANT, str(RELEASE): RELEASE, str(MANIFEST): MANIFEST, str(RECEIPTS): RECEIPTS, str(PILOT): PILOT, str(DATA): DATA, str(STATE): STATE}.get(raw, Path("/noncanonical")), "program")


def _check_pair() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _canonical_path(str(GRANT.relative_to(ROOT)), GRANT, "grant")
    if not GRANT.is_file() or not RELEASE.is_file() or not MANIFEST.is_file():
        raise PermissionError("sealed grant or external pair is absent")
    if sha256_file(GRANT) != GRANT_SHA or sha256_file(RELEASE) != RELEASE_SHA or sha256_file(MANIFEST) != MANIFEST_SHA:
        raise PermissionError("sealed grant or external pair bytes changed")
    _require_mode(GRANT, 0o644, "grant")
    _require_mode(RELEASE, 0o444, "release")
    _require_mode(MANIFEST, 0o444, "manifest")
    grant, release, manifest = _read_json(GRANT, "grant"), _read_json(RELEASE, "release"), _read_json(MANIFEST, "manifest")
    external = grant.get("external_release", {})
    _canonical_path(external.get("path"), RELEASE, "grant release")
    _canonical_path(external.get("manifest_path"), MANIFEST, "grant manifest")
    if grant.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-execution-grant-v6-open" or grant.get("grant_id") != GRANT_ID:
        raise PermissionError("grant identity changed")
    if grant.get("authorization", {}).get("pilot_stage_authorized") is not True or any(grant.get("authorization", {}).get(name) is not False for name in ("production_stage_authorized", "cache_population_authorized", "downstream_execution_authorized", "automatic_retry_retune_scale_up_or_follow_on_authorized")):
        raise PermissionError("grant stage authorization changed")
    pairs = ((external.get("release_file_sha256"), RELEASE_SHA), (external.get("payload_sha256"), PAYLOAD_SHA), (external.get("release_id"), RELEASE_ID), (external.get("manifest_sha256"), MANIFEST_SHA), (external.get("manifest_id"), MANIFEST_ID),
             (release.get("release_id"), RELEASE_ID), (release.get("payload_sha256"), PAYLOAD_SHA), (release.get("manifest_file_sha256"), MANIFEST_SHA), (release.get("manifest_id"), MANIFEST_ID),
             (manifest.get("release_id"), RELEASE_ID), (manifest.get("release_payload_sha256"), PAYLOAD_SHA), (manifest.get("manifest_id"), MANIFEST_ID))
    for value, expected in pairs:
        if _full_hash(value, "pair binding") != expected:
            raise PermissionError("grant and external pair binding changed")
    if release.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-release-v6-open" or release.get("status") != "complete_pass_external_postcommit_lineage_audit":
        raise PermissionError("release schema or status changed")
    if manifest.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-manifest-v6-open" or manifest.get("status") != "complete_paired_external_manifest":
        raise PermissionError("manifest schema or status changed")
    _canonical_path(release.get("manifest_path"), MANIFEST, "release manifest")
    _canonical_path(manifest.get("release_path"), RELEASE, "manifest release")
    return grant, release, manifest


def run_preflight_v6_open(program_path: Path, stage: str, hostname: str | None = None) -> dict[str, str]:
    """Validate only canonical preflight inputs; this function performs no runtime action."""
    if Path(program_path).resolve() != PROGRAM.resolve():
        raise PermissionError("preflight accepts only canonical program")
    if stage != "pilot":
        raise PermissionError("only pilot stage validation is authorized")
    if ascii_lower_short_hostname(socket.gethostname() if hostname is None else hostname) != "lageunha":
        raise PermissionError("Lageunha host gate failed")
    design, program, lift = _check_config_hashes()
    _check_program_pins(design, program, lift)
    if any(root.exists() for root in RUNTIME_ROOTS):
        raise PermissionError("v6-open runtime root is present")
    grant, release, manifest = _check_pair()
    return {"status": "complete_preflight_only_v6_open", "stage": "pilot", "hostname": ascii_lower_short_hostname(socket.gethostname() if hostname is None else hostname), "grant_id": grant["grant_id"], "release_id": release["release_id"], "manifest_id": manifest["manifest_id"]}
