"""Additive v6-open pilot boundary; public execution is fail-closed until separately authorized."""
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
DESIGN = ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_design_v6_open_pilot.json"
PROGRAM = ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_program_v6_open_pilot.json"
FUTURE_AUTH = ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_pilot_execution_result_record_v6_open.json"
OPEN_GRANT = ROOT / "config/cf4_aggregate_evidence_smc_execution_grant_v6_open.json"
OPEN_RELEASE = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_execution_authorization_v6_open_release.json")
OPEN_MANIFEST = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_execution_authorization_v6_open_manifest.json")
CLOSED_GRANT = ROOT / "config/cf4_aggregate_evidence_smc_execution_grant_v6.json"
CLOSED_RELEASE = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_execution_authorization_v6_release.json")
CLOSED_MANIFEST = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_execution_authorization_v6_manifest.json")
PREFLIGHT_IMPLEMENTATION = ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_preflight_implementation_result_record_v6_open.json"
PREFLIGHT_RESULT = ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_preflight_result_record_v6_open.json"
PREFLIGHT_PROGRAM = ROOT / "config/cf4_aggregate_evidence_smc_execution_authorization_program_v6_open_preflight.json"
PREFLIGHT_SOURCE = ROOT / "src/cf4_aggregate_evidence_smc_execution_authorized_v6_open_preflight.py"
RECEIPTS = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_receipts")
PILOT = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_disposable_pilot")
DATA_FORBIDDEN = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open")
STATE_FORBIDDEN = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_run")
DESIGN_SHA = "0af04e41ac903bc81bab70f0f1813eb3f9bd10ea6221f5a9268892acde45d97a"
PROGRAM_SHA = "e4231667c6949df429f4698f57d59c0595113c4c46ebffff6d7640f09841041c"
OPEN_GRANT_SHA = "384b9c1fc0a9ccfe46cf1ff6b3a7524995e0c12f7017a048bb06fcebef8ea4d8"
OPEN_RELEASE_SHA = "32da63e6a27622f1ae57bd4b431ddf9892a69db566a824fbc517a5201c2c5bc3"
OPEN_MANIFEST_SHA = "5c946a100dfd3fea1eca4dcec00f5347960252fabc4eddc62a9f0b3eb73ef23d"
CLOSED_GRANT_SHA = "8c6bf135be677254fae1a8a1d60f75bad813e0969e5216dc7ff89ed9e9e9509a"
CLOSED_RELEASE_SHA = "0b1dcce3e47dc9be292ad1eda599985b00f9f2e880808cdddf069aa48376b4a7"
CLOSED_MANIFEST_SHA = "344bed3d1758589f7489069599e0ec3b326b285be1aec945a49168129f3bed30"
OPEN_GRANT_ID = "8f6689e85bd4e19fe114cda1dd1051f46af259b79c5f52d95ab6d697641b5f7d"
OPEN_RELEASE_ID = "dc66fe4bd14d38ded3acf66d54673321e0a13516d23a53da48798eecd3317c35"
OPEN_MANIFEST_ID = "82fbc16750c505f718f0bebbaf1080f96ebb44a8a39ba305e5f25713f81d7e27"
PREFLIGHT_IMPLEMENTATION_SHA = "caee387ea8cac61c5d39006a1db666d7fa8bbfc93b2ebdf134c66e14081a0b64"
PREFLIGHT_RESULT_SHA = "ac1be7e41809a91bf10d1a9f6174a7588c75a131e2073c27df1478362915be81"
PREFLIGHT_PROGRAM_SHA = "facd7126b6a2abc312838ccd25f93d87b7077cc07c4c611e3d34003a8def5b17"
PREFLIGHT_SOURCE_SHA = "15d3d68675ebb516689f748186fb999c114735594f4bf653c4caa0d12515d3ec"
AUTH = {"implementation_authorized": True, "pilot_stage_authorized": True, "receipt_creation_authorized": True,
        "pilot_execution_authorized_now": False, "production_stage_authorized": False,
        "cache_population_authorized": False, "downstream_execution_authorized": False,
        "automatic_follow_on_authorized": False}
SCIENTIFIC_COMPLETE = {"complete_pass_production_smc", "complete_scientific_fail_production_smc"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PermissionError(f"{label} is absent")
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(f"{label} is invalid") from error
    if not isinstance(value, dict):
        raise PermissionError(f"{label} is invalid")
    return value


def ascii_lower_short_hostname(hostname: str) -> str:
    short = hostname.split(".", 1)[0]
    if not short or any(ord(char) > 127 for char in short):
        raise PermissionError("hostname is not ASCII")
    return "".join(chr(ord(char) + 32) if "A" <= char <= "Z" else char for char in short)


def _canonical(raw: Any, expected: Path, label: str) -> None:
    if not isinstance(raw, str) or "v6_open" not in raw or any(old in raw.lower() for old in ("v4", "v5")):
        raise PermissionError(f"{label} path is not canonical v6-open")
    try:
        wanted = str(expected.relative_to(ROOT))
    except ValueError:
        wanted = str(expected)
    if raw != wanted:
        raise PermissionError(f"{label} path is not canonical v6-open")


def _require_file(path: Path, digest: str, mode: int, label: str) -> None:
    if not path.is_file() or sha256_file(path) != digest:
        raise PermissionError(f"{label} hash mismatch")
    if stat.S_IMODE(path.stat().st_mode) != mode:
        raise PermissionError(f"{label} mode mismatch")


def load_program() -> dict[str, Any]:
    if sha256_file(DESIGN) != DESIGN_SHA or sha256_file(PROGRAM) != PROGRAM_SHA:
        raise PermissionError("pilot boundary configuration hash mismatch")
    design, program = _json(DESIGN, "pilot design"), _json(PROGRAM, "pilot program")
    if design.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-design-v6-open-pilot" or program.get("schema") != "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-program-v6-open-pilot":
        raise PermissionError("pilot boundary schema changed")
    if design.get("authorization") != AUTH or program.get("authorization") != AUTH:
        raise PermissionError("pilot boundary authorization changed")
    if program.get("design") != {"path": str(DESIGN.relative_to(ROOT)), "sha256": DESIGN_SHA} or design.get("fixed_science") != program.get("fixed_science"):
        raise PermissionError("pilot boundary design binding changed")
    for raw, expected, label in ((program["paths"]["grant"], OPEN_GRANT, "grant"), (program["paths"]["release"], OPEN_RELEASE, "release"), (program["paths"]["manifest"], OPEN_MANIFEST, "manifest"), (program["paths"]["receipt_root"], RECEIPTS, "receipt"), (program["paths"]["pilot_root"], PILOT, "pilot"), (program["paths"]["data_root_forbidden"], DATA_FORBIDDEN, "data"), (program["paths"]["state_root_forbidden"], STATE_FORBIDDEN, "state"), (program["paths"]["future_pilot_execution_authorization_record"], FUTURE_AUTH, "future authorization")):
        _canonical(raw, expected, label)
    return program


def verify_pinned_provenance(program: dict[str, Any]) -> None:
    for path, digest, mode, label in ((OPEN_GRANT, OPEN_GRANT_SHA, 0o644, "open grant"), (OPEN_RELEASE, OPEN_RELEASE_SHA, 0o444, "open release"), (OPEN_MANIFEST, OPEN_MANIFEST_SHA, 0o444, "open manifest"), (CLOSED_GRANT, CLOSED_GRANT_SHA, 0o644, "closed grant"), (CLOSED_RELEASE, CLOSED_RELEASE_SHA, 0o444, "closed release"), (CLOSED_MANIFEST, CLOSED_MANIFEST_SHA, 0o444, "closed manifest")):
        _require_file(path, digest, mode, label)
    for path, digest, label in ((PREFLIGHT_IMPLEMENTATION, PREFLIGHT_IMPLEMENTATION_SHA, "preflight implementation record"), (PREFLIGHT_RESULT, PREFLIGHT_RESULT_SHA, "preflight result record"), (PREFLIGHT_PROGRAM, PREFLIGHT_PROGRAM_SHA, "preflight program"), (PREFLIGHT_SOURCE, PREFLIGHT_SOURCE_SHA, "preflight source")):
        if sha256_file(path) != digest:
            raise PermissionError(f"{label} hash mismatch")
    grant, release, manifest = _json(OPEN_GRANT, "open grant"), _json(OPEN_RELEASE, "open release"), _json(OPEN_MANIFEST, "open manifest")
    external = grant.get("external_release", {})
    if grant.get("grant_id") != OPEN_GRANT_ID or external.get("release_id") != OPEN_RELEASE_ID or external.get("manifest_id") != OPEN_MANIFEST_ID:
        raise PermissionError("open grant pair identity mismatch")
    if release.get("release_id") != OPEN_RELEASE_ID or release.get("manifest_id") != OPEN_MANIFEST_ID or manifest.get("release_id") != OPEN_RELEASE_ID or manifest.get("manifest_id") != OPEN_MANIFEST_ID:
        raise PermissionError("open external pair binding mismatch")
    if _json(PREFLIGHT_RESULT, "preflight result record").get("status") != "complete_preflight_only_v6_open":
        raise PermissionError("preflight result is not sealed")


def require_future_pilot_execution_authorization(program: dict[str, Any]) -> None:
    verify_pinned_provenance(program)
    if not FUTURE_AUTH.is_file():
        raise PermissionError("future pilot-execution authorization record is absent; mutation forbidden")
    raise PermissionError("future pilot-execution authorization record format is not installed")


def _canonical_snapshot() -> dict[str, Any]:
    release_stat = OPEN_RELEASE.stat()
    return {"schema": "ouruniv-cf4-v6-open-pilot-receipt-snapshot-v1", "grant_sha256": sha256_file(OPEN_GRANT), "release": {"path": str(OPEN_RELEASE), "sha256": sha256_file(OPEN_RELEASE), "dev": release_stat.st_dev, "ino": release_stat.st_ino, "size": release_stat.st_size, "nlink": release_stat.st_nlink}, "manifest_sha256": sha256_file(OPEN_MANIFEST), "program_sha256": sha256_file(PROGRAM)}


def _snapshot_bytes(snapshot: dict[str, Any]) -> bytes:
    return (json.dumps(snapshot, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def create_pilot_receipt(program: dict[str, Any], run_id: str) -> tuple[Path, dict[str, Any]]:
    """Future-only lifecycle primitive; the authorization check always precedes every write."""
    require_future_pilot_execution_authorization(program)
    if re.fullmatch(r"[0-9a-f]{64}", run_id) is None:
        raise PermissionError("pilot receipt run id is invalid")
    receipt = RECEIPTS / run_id / "pilot"
    if receipt.parent.parent != RECEIPTS or DATA_FORBIDDEN.exists() or STATE_FORBIDDEN.exists():
        raise PermissionError("pilot receipt or namespace path is forbidden")
    os.mkdir(RECEIPTS, 0o700)
    os.mkdir(receipt.parent, 0o700)
    os.mkdir(receipt, 0o700)
    os.link(OPEN_RELEASE, receipt / "release.anchor")
    snapshot = _canonical_snapshot()
    descriptor = os.open(receipt / "snapshot.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_snapshot_bytes(snapshot))
    return receipt, snapshot


def revalidate_pilot_receipt(receipt: Path, snapshot: dict[str, Any]) -> None:
    anchor = Path(receipt) / "release.anchor"
    if not anchor.is_file() or anchor.stat().st_ino != OPEN_RELEASE.stat().st_ino or _canonical_snapshot() != snapshot:
        raise PermissionError("pilot receipt provenance changed")


def pilot_output_path(receipt: Path, name: str) -> Path:
    path = (PILOT / Path(receipt).name / name).resolve()
    if DATA_FORBIDDEN in path.parents or STATE_FORBIDDEN in path.parents or PILOT not in path.parents:
        raise PermissionError("pilot output path is forbidden")
    return path


def write_readonly_schedule_manifest(program: dict[str, Any], receipt: Path, schedule: dict[str, Any]) -> Path:
    require_future_pilot_execution_authorization(program)
    target = Path(receipt) / "schedule_manifest.json"
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_snapshot_bytes(schedule))
    os.chmod(target, 0o444)
    return target


def map_terminal_status(status: str | None) -> dict[str, str]:
    if status in SCIENTIFIC_COMPLETE:
        return {"status": status, "outcome_kind": "scientific", "failure_class": "scientific"}
    return {"status": "FAILED", "outcome_kind": "invalid", "failure_class": "invalid_provenance_or_execution"}


def run_authorized_v6_open_pilot(program_path: Path, hostname: str | None = None) -> None:
    """Public gate: validate only, then fail before receipt, namespace, or science mutation."""
    if Path(program_path).resolve() != PROGRAM.resolve():
        raise PermissionError("pilot boundary accepts only canonical program")
    if ascii_lower_short_hostname(socket.gethostname() if hostname is None else hostname) != "lageunha":
        raise PermissionError("Lageunha host gate failed")
    require_future_pilot_execution_authorization(load_program())
