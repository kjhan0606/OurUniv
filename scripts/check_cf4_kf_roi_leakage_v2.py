#!/usr/bin/env python3
"""Fail-closed audit of a published CF4 ROI-leakage v2 PRECHECK_PASS."""

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
    """A published artifact violates its frozen v2 execution contract."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(path: Path) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AuditError(f"cannot parse {path} as JSON") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{path} must contain an object")
    return value, payload


def audit_directory(
    directory: str | Path,
    design_path: str | Path,
    grant_path: str | Path,
    implementation_commit: str,
) -> dict[str, object]:
    root = Path(directory)
    if not re.fullmatch(r"[0-9a-f]{40}", implementation_commit):
        raise AuditError("implementation commit must be lowercase 40-hex")
    if not root.is_dir() or {item.name for item in root.iterdir()} != EXPECTED_FILES:
        raise AuditError("artifact directory is absent or file set is not exact")

    design_sha = _sha256(Path(design_path).read_bytes())
    grant, grant_bytes = _load(Path(grant_path))
    grant_sha = _sha256(grant_bytes)
    if grant.get("schema") != "ouruniv-cf4-kf-roi-leakage-execution-v2":
        raise AuditError("unexpected v2 grant schema")
    if grant.get("status") != "user_approved_single_v2_preflight_only":
        raise AuditError("v2 grant is not active")
    if grant.get("scope", {}).get("design_raw_sha256") != design_sha:
        raise AuditError("grant/design binding mismatch")
    if str(root) != grant["scope"]["preflight_output"]:
        raise AuditError("artifact path does not equal granted v2 preflight output")
    authorization = grant["authorization"]
    if authorization.get("Slurm_preflight_authorized") is not True:
        raise AuditError("grant does not authorize the v2 Slurm preflight")
    if authorization.get("maximum_preflight_submissions") != 1:
        raise AuditError("grant does not authorize exactly one preflight")
    for forbidden in (
        "Slurm_production_authorized",
        "network_access_authorized",
        "retry_authorized",
        "numeric_retuning_authorized",
        "replacement_run_authorized",
        "final_manifest_materialization_authorized",
        "KF_EXPAND_authorized",
        "all_D_mock_execution_authorized",
        "production_science_inference_authorized",
        "scientific_leakage_decision_authorized",
    ):
        if authorization.get(forbidden) is not False:
            raise AuditError(f"forbidden authority is not exact false: {forbidden}")

    result, result_bytes = _load(root / "result.json")
    counts, counts_bytes = _load(root / "mode_counts.json")
    manifest, manifest_bytes = _load(root / "manifest.json")
    complete, _ = _load(root / "COMPLETE")
    if result.get("schema") != "ouruniv-cf4-kf-roi-leakage-result-v2":
        raise AuditError("unexpected result schema")
    if result.get("mode") != "preflight" or result.get("status") != "PRECHECK_PASS":
        raise AuditError("only PRECHECK_PASS may be published")
    if result.get("design_raw_sha256") != design_sha:
        raise AuditError("result/design binding mismatch")
    if result.get("execution_grant_raw_sha256") != grant_sha:
        raise AuditError("result/grant binding mismatch")
    if result.get("implementation_commit") != implementation_commit:
        raise AuditError("result commit mismatch")
    if result.get("numerical_convergence", {}).get("status") != "PASS":
        raise AuditError("published precheck lacks numerical PASS")
    if result["numerical_convergence"].get(
        "all_analysis_column_normalizations_valid"
    ) is not True:
        raise AuditError("published precheck has invalid column normalization")
    for key in (
        "truth_or_candidate_data_consumed",
        "scientific_leakage_decision_authorized",
        "final_manifest_materialized",
        "k_boundary_claim_created",
    ):
        if result.get(key) is not False:
            raise AuditError(f"result field is not exact false: {key}")
    for roi in result.get("ROI_results", []):
        columns = np.asarray(roi.get("analysis_column_sum"), dtype=float)
        residual = np.asarray(roi.get("signed_outside_analysis_residual"), dtype=float)
        valid = np.asarray(roi.get("normalization_valid"))
        if columns.shape != (38,) or residual.shape != (38,) or valid.shape != (38,):
            raise AuditError("ROI diagnostics do not contain every native bin")
        if not np.all(np.isfinite(columns)) or not np.allclose(
            residual, 1.0 - columns, rtol=0.0, atol=2e-15
        ):
            raise AuditError("ROI signed normalization diagnostics are inconsistent")
        if not np.all(valid):
            raise AuditError("ROI contains a failed normalization bin")
        parseval = roi.get("numerical_audit", {}).get("parseval_q_space", {})
        if parseval.get("pass") is not True:
            raise AuditError("ROI q-space Parseval audit did not pass")
    if len(result.get("ROI_results", [])) != 6:
        raise AuditError("result does not contain all six semantic ROIs")

    if counts.get("schema") != "ouruniv-cf4-kf-roi-leakage-mode-counts-v2":
        raise AuditError("unexpected mode-count schema")
    if counts.get("design_raw_sha256") != design_sha:
        raise AuditError("mode-count/design binding mismatch")
    if len(counts.get("native_bins", [])) != 38:
        raise AuditError("mode counts omit native bins")
    if counts.get("count_audit", {}).get("total_count_assertion_pass") is not True:
        raise AuditError("N^3 count audit did not pass")

    if manifest.get("schema") != "ouruniv-cf4-kf-roi-leakage-artifact-manifest-v2":
        raise AuditError("unexpected manifest schema")
    if manifest.get("status") != "PRECHECK_PASS" or manifest.get("mode") != "preflight":
        raise AuditError("manifest status/mode mismatch")
    if manifest.get("design_raw_sha256") != design_sha:
        raise AuditError("manifest/design binding mismatch")
    if manifest.get("execution_grant_raw_sha256") != grant_sha:
        raise AuditError("manifest/grant binding mismatch")
    if manifest.get("implementation_commit") != implementation_commit:
        raise AuditError("manifest commit mismatch")
    payloads = {
        "result.json": result_bytes,
        "mode_counts.json": counts_bytes,
        "mixing_matrices.npz": (root / "mixing_matrices.npz").read_bytes(),
    }
    if set(manifest.get("payloads", {})) != set(payloads):
        raise AuditError("manifest payload set mismatch")
    for name, payload in payloads.items():
        record = manifest["payloads"][name]
        if record.get("sha256") != _sha256(payload) or record.get("bytes") != len(payload):
            raise AuditError(f"manifest hash/size mismatch for {name}")
    if complete.get("schema") != "ouruniv-cf4-kf-roi-leakage-complete-v2":
        raise AuditError("unexpected COMPLETE schema")
    if complete.get("status") != "PRECHECK_PASS" or complete.get("mode") != "preflight":
        raise AuditError("COMPLETE status/mode mismatch")
    if complete.get("manifest_sha256") != _sha256(manifest_bytes):
        raise AuditError("COMPLETE does not bind manifest")
    if complete.get("design_raw_sha256") != design_sha:
        raise AuditError("COMPLETE/design binding mismatch")
    if complete.get("execution_grant_raw_sha256") != grant_sha:
        raise AuditError("COMPLETE/grant binding mismatch")
    if complete.get("implementation_commit") != implementation_commit:
        raise AuditError("COMPLETE commit mismatch")

    try:
        with np.load(io.BytesIO(payloads["mixing_matrices.npz"]), allow_pickle=False) as archive:
            if not archive.files:
                raise AuditError("mixing archive is empty")
            for name in archive.files:
                array = archive[name]
                if array.shape != (38, 38) or not np.all(np.isfinite(array)) or np.any(array < 0):
                    raise AuditError(f"invalid mixing array {name}")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise AuditError("cannot validate mixing archive") from exc

    implementation = Path(__file__).resolve().parents[1] / result["implementation_path"]
    if _sha256(implementation.read_bytes()) != result.get("implementation_sha256"):
        raise AuditError("implementation file SHA mismatch")
    return {
        "status": "PASS",
        "mode": "preflight",
        "precheck_status": "PRECHECK_PASS",
        "directory": str(root),
        "design_raw_sha256": design_sha,
        "execution_grant_raw_sha256": grant_sha,
        "implementation_commit": implementation_commit,
        "result_sha256": _sha256(result_bytes),
        "scientific_leakage_decision_authorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--execution-grant", required=True, type=Path)
    parser.add_argument("--implementation-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = audit_directory(
            args.directory,
            args.design,
            args.execution_grant,
            args.implementation_commit,
        )
    except (OSError, AuditError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
