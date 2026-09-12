#!/usr/bin/env bash
set -Eeuo pipefail
readonly python=/home/kjhan/miniconda3/envs/circle/bin/python
exec "$python" - <<'PY'
import hashlib
import json
import re
import stat
from pathlib import Path

receipt_root = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_receipts")
pilot_root = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_disposable_pilot")
data_root = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open")
state_root = Path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v6_open_run")


def fail(message: str, code: int = 65) -> None:
    print(f"status={message}")
    raise SystemExit(code)


def read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        fail(f"invalid_marker_json path={path}")
    if not isinstance(value, dict) or not value:
        fail(f"invalid_marker_json path={path}")
    return value


if data_root.exists() or state_root.exists():
    fail("forbidden_production_namespace_present")
if not receipt_root.exists() and not pilot_root.exists():
    fail("pilot_not_started_fail_closed", 3)
if not receipt_root.is_dir():
    fail("invalid_receipt_root")

markers = sorted(
    path
    for name in ("RUNNING", "COMPLETE", "FAILED")
    for path in receipt_root.glob(f"*/pilot/{name}")
    if path.is_file()
)
if len(markers) != 1:
    fail(f"invalid_marker_count count={len(markers)}")

marker = markers[0]
if stat.S_IMODE(marker.stat().st_mode) != 0o444:
    fail("invalid_marker_mode")
authorization_id = marker.parent.parent.name
if re.fullmatch(r"[0-9a-f]{64}", authorization_id) is None:
    fail("invalid_authorization_id")
if any(path.is_file() for path in receipt_root.glob("*/pilot/*") if path.name in {"RUNNING", "COMPLETE", "FAILED"} and path != marker):
    fail("invalid_marker_conflict")
payload = read_object(marker)

if marker.name == "RUNNING":
    if payload.get("status") != "reserving_disposable_pilot" or pilot_root.exists():
        fail("invalid_running_state")
    print(f"status=pilot_running marker={marker}")
    raise SystemExit(0)

if marker.name == "FAILED":
    if not isinstance(payload.get("status"), str) or not payload["status"].startswith("failed_"):
        fail("invalid_failed_marker")
    print(f"status=pilot_failed marker={marker} failure_class={payload.get('failure_class', 'unknown')}")
    raise SystemExit(0)

if payload.get("status") != "complete_disposable_pilot_schedule_only":
    fail("invalid_complete_marker")
receipt_manifest = marker.parent / "schedule_manifest.json"
pilot_manifest = pilot_root / authorization_id / "schedule_manifest.json"
for manifest in (receipt_manifest, pilot_manifest):
    if not manifest.is_file() or stat.S_IMODE(manifest.stat().st_mode) != 0o444:
        fail(f"invalid_complete_manifest path={manifest}")
receipt_bytes, pilot_bytes = receipt_manifest.read_bytes(), pilot_manifest.read_bytes()
manifest_sha = hashlib.sha256(receipt_bytes).hexdigest()
if receipt_bytes != pilot_bytes or payload.get("manifest_sha256") != manifest_sha:
    fail("complete_manifest_binding_mismatch")
manifest_payload = read_object(receipt_manifest)
if manifest_payload.get("status") != "complete_disposable_pilot_schedule_only" or manifest_payload.get("authorization_id") != authorization_id:
    fail("invalid_complete_manifest_payload")
if manifest_payload.get("schedule", {}).get("sha256") != payload.get("schedule_sha256"):
    fail("complete_schedule_binding_mismatch")
print(f"status=pilot_complete marker={marker} manifest_sha256={manifest_sha}")
PY
