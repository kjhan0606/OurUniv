#!/usr/bin/env python3
"""Fail-closed audit of a published CF4 ROI-leakage artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np


EXPECTED_FILES = {
    "result.json",
    "mode_counts.json",
    "mixing_matrices.npz",
    "manifest.json",
    "COMPLETE",
}


class AuditError(ValueError):
    """A published artifact violates its frozen execution contract."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json_bytes(path: Path) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AuditError(f"cannot parse {path} as JSON") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{path} must contain a JSON object")
    return value, payload


def audit_directory(
    directory: str | Path,
    design_path: str | Path,
    grant_path: str | Path,
    expected_mode: str,
    implementation_commit: str,
) -> dict[str, object]:
    root = Path(directory)
    if expected_mode not in ("preflight", "production"):
        raise AuditError("expected mode must be preflight or production")
    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise AuditError("implementation commit must be lowercase 40-hex")
    if not root.is_dir():
        raise AuditError("artifact directory does not exist")
    actual_files = {item.name for item in root.iterdir()}
    if actual_files != EXPECTED_FILES:
        raise AuditError(
            f"artifact file set mismatch: {sorted(actual_files)} != {sorted(EXPECTED_FILES)}"
        )

    design_bytes = Path(design_path).read_bytes()
    design_sha = _sha256(design_bytes)
    grant, _ = _load_json_bytes(Path(grant_path))
    if grant.get("schema") != "ouruniv-cf4-kf-roi-leakage-execution-v1":
        raise AuditError("unexpected execution grant schema")
    if grant.get("status") != "user_approved_narrow_execution_grant":
        raise AuditError("execution grant is not active narrow approval")
    if grant["scope"]["design_raw_sha256"] != design_sha:
        raise AuditError("execution grant/design SHA256 mismatch")
    authorization = grant["authorization"]
    if expected_mode == "preflight":
        if not authorization["Slurm_preflight_authorized"]:
            raise AuditError("preflight is not authorized")
        if authorization["maximum_preflight_submissions"] != 1:
            raise AuditError("preflight submission cardinality is not exactly one")
        expected_output = grant["scope"]["preflight_output"]
    else:
        if not authorization["Slurm_production_authorized_conditionally"]:
            raise AuditError("production is not conditionally authorized")
        if authorization["maximum_production_submissions"] != 1:
            raise AuditError("production submission cardinality is not at most one")
        expected_output = grant["scope"]["production_output"]
    if str(root) != expected_output:
        raise AuditError(f"artifact path {root} != grant path {expected_output}")
    for forbidden in (
        "network_access_authorized",
        "final_manifest_materialization_authorized",
        "KF_EXPAND_authorized",
        "all_D_mock_execution_authorized",
        "production_science_inference_authorized",
        "retry_authorized",
        "numeric_retuning_authorized",
        "replacement_run_authorized",
    ):
        if authorization[forbidden]:
            raise AuditError(f"forbidden authority unexpectedly true: {forbidden}")

    result, result_bytes = _load_json_bytes(root / "result.json")
    counts, counts_bytes = _load_json_bytes(root / "mode_counts.json")
    manifest, manifest_bytes = _load_json_bytes(root / "manifest.json")
    complete, _ = _load_json_bytes(root / "COMPLETE")
    expected_status = "PRECHECK" if expected_mode == "preflight" else "COMPLETE"
    if result.get("schema") != "ouruniv-cf4-kf-roi-leakage-result-v1":
        raise AuditError("unexpected result schema")
    if result.get("mode") != expected_mode or result.get("status") != expected_status:
        raise AuditError("result mode/status mismatch")
    if result.get("design_raw_sha256") != design_sha:
        raise AuditError("result design SHA256 mismatch")
    if result.get("implementation_commit") != implementation_commit:
        raise AuditError("result implementation commit mismatch")
    if result.get("truth_or_candidate_data_consumed") is not False:
        raise AuditError("result does not certify truth/candidate independence")
    if result.get("final_manifest_materialized") is not False:
        raise AuditError("leakage run improperly materialized final manifest")
    if result.get("k_boundary_claim_created") is not False:
        raise AuditError("leakage run improperly created a k-boundary claim")
    if not isinstance(result.get("overall_leakage_gate_pass"), bool):
        raise AuditError("overall leakage gate must be an exact boolean")
    if expected_mode == "production" and result.get("numerical_convergence", {}).get(
        "status"
    ) != "PASS":
        raise AuditError("production does not bind numerical preflight PASS")

    if counts.get("schema") != "ouruniv-cf4-kf-roi-leakage-mode-counts-v1":
        raise AuditError("unexpected mode-count schema")
    if counts.get("design_raw_sha256") != design_sha:
        raise AuditError("mode counts design SHA256 mismatch")
    if len(counts.get("native_bins", [])) != 38:
        raise AuditError("mode counts do not contain all 38 native bins")
    if not counts.get("count_audit", {}).get("total_count_assertion_pass"):
        raise AuditError("mode-count N^3 assertion did not pass")

    if manifest.get("schema") != (
        "ouruniv-cf4-kf-roi-leakage-artifact-manifest-v1"
    ):
        raise AuditError("unexpected artifact manifest schema")
    if manifest.get("mode") != expected_mode or manifest.get("status") != expected_status:
        raise AuditError("artifact manifest mode/status mismatch")
    if manifest.get("design_raw_sha256") != design_sha:
        raise AuditError("artifact manifest design SHA256 mismatch")
    if manifest.get("implementation_commit") != implementation_commit:
        raise AuditError("artifact manifest commit mismatch")
    payload_bytes = {
        "result.json": result_bytes,
        "mode_counts.json": counts_bytes,
        "mixing_matrices.npz": (root / "mixing_matrices.npz").read_bytes(),
    }
    if set(manifest.get("payloads", {})) != set(payload_bytes):
        raise AuditError("manifest payload set mismatch")
    for filename, payload in payload_bytes.items():
        record = manifest["payloads"][filename]
        if record.get("sha256") != _sha256(payload) or record.get("bytes") != len(
            payload
        ):
            raise AuditError(f"payload hash/size mismatch for {filename}")

    if complete.get("schema") != "ouruniv-cf4-kf-roi-leakage-complete-v1":
        raise AuditError("unexpected COMPLETE schema")
    if complete.get("mode") != expected_mode or complete.get("status") != expected_status:
        raise AuditError("COMPLETE mode/status mismatch")
    if complete.get("manifest_sha256") != _sha256(manifest_bytes):
        raise AuditError("COMPLETE manifest SHA256 mismatch")
    if complete.get("design_raw_sha256") != design_sha:
        raise AuditError("COMPLETE design SHA256 mismatch")
    if complete.get("implementation_commit") != implementation_commit:
        raise AuditError("COMPLETE implementation commit mismatch")

    npz_payload = payload_bytes["mixing_matrices.npz"]
    try:
        with np.load(io.BytesIO(npz_payload), allow_pickle=False) as archive:
            if not archive.files:
                raise AuditError("mixing NPZ has no arrays")
            for name in archive.files:
                array = archive[name]
                if array.ndim != 2 or array.shape != (38, 38):
                    raise AuditError(f"mixing array {name} has wrong shape")
                if not np.all(np.isfinite(array)) or np.any(array < 0):
                    raise AuditError(f"mixing array {name} is invalid")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise AuditError("cannot validate mixing NPZ") from exc

    implementation_path = Path(__file__).resolve().parents[1] / result[
        "implementation_path"
    ]
    if _sha256(implementation_path.read_bytes()) != result["implementation_sha256"]:
        raise AuditError("implementation file SHA256 mismatch")
    return {
        "status": "PASS",
        "mode": expected_mode,
        "directory": str(root),
        "design_raw_sha256": design_sha,
        "implementation_commit": implementation_commit,
        "result_sha256": _sha256(result_bytes),
        "manifest_sha256": _sha256(manifest_bytes),
        "overall_leakage_gate_pass": result["overall_leakage_gate_pass"],
        "scientific_disposition": result["scientific_disposition"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--execution-grant", required=True, type=Path)
    parser.add_argument("--expected-mode", required=True, choices=("preflight", "production"))
    parser.add_argument("--implementation-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = audit_directory(
            args.directory,
            args.design,
            args.execution_grant,
            args.expected_mode,
            args.implementation_commit,
        )
    except (OSError, AuditError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
