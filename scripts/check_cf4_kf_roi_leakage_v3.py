#!/usr/bin/env python3
"""Fail-closed audit of a published CF4 ROI-leakage v3 PRECHECK_PASS."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
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
TOLERANCE = 5.0e-4


class AuditError(ValueError):
    """A published artifact violates its frozen v3 execution contract."""


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

    design, design_bytes = _load(Path(design_path))
    design_sha = _sha256(design_bytes)
    if design.get("schema") != "ouruniv-cf4-kf-bin-manifest-design-v2":
        raise AuditError("unexpected design v2 schema")
    if design.get("status") != "user_approved_design_frozen_numerical_preflight_pending":
        raise AuditError("design v2 is not frozen/pending")
    grant, grant_bytes = _load(Path(grant_path))
    grant_sha = _sha256(grant_bytes)
    execution_schema = "ouruniv-cf4-kf-roi-leakage-execution-v3a"
    if grant.get("schema") != execution_schema:
        raise AuditError("unexpected v3a grant schema")
    if grant.get("status") != "original_user_approval_operator_correction_single_v3a_preflight_only":
        raise AuditError("v3a grant is not active")
    if grant.get("authorization_basis") != (
        "the_original_user_approval_remains_authoritative_for_this_operator_error_correction; "
        "this is not new scientific or numerical scope"
    ):
        raise AuditError("v3a authorization basis mismatch")
    if grant.get("scope", {}).get("design_raw_sha256") != design_sha:
        raise AuditError("grant/design binding mismatch")
    for order in ("coarse", "fine"):
        if grant.get("numerical_contract", {}).get(order) != design.get(
            "frozen_numerics", {}
        ).get(order):
            raise AuditError(f"grant/design frozen {order} numerics mismatch")
    if str(root) != grant["scope"]["preflight_output"]:
        raise AuditError("artifact path does not equal granted v3 preflight output")
    expected_authorization = {
        "v3a_operator_correction_implementation_authorized": True,
        "corrective_scheduler_submission_authorized": True,
        "Slurm_preflight_authorized": True,
        "maximum_corrective_scheduler_submissions": 1,
        "maximum_numerical_preflight_executions": 1,
        "prior_v3_scheduler_attempts_recorded": 1,
        "prior_v3_numerical_preflight_executions_recorded": 0,
        "further_operator_correction_authorized": False,
        "Slurm_production_authorized": False,
        "GPFS_read_authorized_only_for_declared_inputs_and_output_root": True,
        "GPFS_write_authorized_only_under_output_root": True,
        "network_access_authorized": False,
        "retry_authorized": False,
        "numeric_retuning_authorized": False,
        "replacement_run_authorized": False,
        "final_manifest_materialization_authorized": False,
        "KF_EXPAND_authorized": False,
        "all_D_mock_execution_authorized": False,
        "production_science_inference_authorized": False,
        "scientific_leakage_decision_authorized": False,
    }
    if grant.get("authorization") != expected_authorization:
        raise AuditError("v3a authorization is not exact")
    numerical_payload = (
        json.dumps(
            grant.get("numerical_contract", {}),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    if _sha256(numerical_payload) != "dd94f416ca7d1d0078faef98cf4a36be9703b154a3b4fac665329aece3d433e9":
        raise AuditError("v3a frozen numerical contract is not exact")

    result, result_bytes = _load(root / "result.json")
    counts, counts_bytes = _load(root / "mode_counts.json")
    manifest, manifest_bytes = _load(root / "manifest.json")
    complete, _ = _load(root / "COMPLETE")
    if result.get("schema") != "ouruniv-cf4-kf-roi-leakage-result-v3":
        raise AuditError("unexpected result v3 schema")
    if result.get("operational_execution_schema") != execution_schema:
        raise AuditError("result v3a operational schema mismatch")
    if result.get("mode") != "preflight" or result.get("status") != "PRECHECK_PASS":
        raise AuditError("only v3 PRECHECK_PASS may be published")
    if result.get("design_raw_sha256") != design_sha:
        raise AuditError("result/design binding mismatch")
    if result.get("execution_grant_raw_sha256") != grant_sha:
        raise AuditError("result/grant binding mismatch")
    if result.get("implementation_commit") != implementation_commit:
        raise AuditError("result commit mismatch")
    if result.get("frozen_coarse_numerics") != grant["numerical_contract"]["coarse"]:
        raise AuditError("result/grant coarse numerics mismatch")
    if result.get("frozen_fine_numerics") != grant["numerical_contract"]["fine"]:
        raise AuditError("result/grant fine numerics mismatch")
    convergence = result.get("numerical_convergence", {})
    if convergence.get("status") != "PASS":
        raise AuditError("published v3 precheck lacks numerical PASS")
    for required_pass in (
        "all_analysis_column_normalizations_valid",
        "native_classification_identical",
        "contiguous_run_proposal_identical",
        "threshold_margin_safety_pass",
    ):
        if convergence.get(required_pass) is not True:
            raise AuditError(f"numerical convergence field is not true: {required_pass}")
    if convergence.get("max_analysis_reciprocity_relative_error", math.inf) > TOLERANCE:
        raise AuditError("analysis reciprocity exceeds tolerance")
    if convergence.get("max_guard_decomposition_abs_residual", math.inf) > TOLERANCE:
        raise AuditError("guard decomposition exceeds tolerance")
    for key in (
        "truth_or_candidate_data_consumed",
        "geometry_window_proposals_are_scientific_claims",
        "scientific_leakage_decision_authorized",
        "final_manifest_materialized",
        "k_boundary_claim_created",
    ):
        if result.get(key) is not False:
            raise AuditError(f"result field is not exact false: {key}")

    roi_results = result.get("ROI_results", [])
    if len(roi_results) != 6:
        raise AuditError("result does not contain six semantic ROIs")
    numeric_keys = set()
    for roi in roi_results:
        numeric_keys.add(roi.get("numeric_product_key"))
        column = np.asarray(roi.get("analysis_column_sum"), dtype=float)
        outside = np.asarray(roi.get("signed_outside_analysis_residual"), dtype=float)
        lower = np.asarray(roi.get("lower_guard"), dtype=float)
        upper = np.asarray(roi.get("upper_guard"), dtype=float)
        far = np.asarray(roi.get("far_tail"), dtype=float)
        total = np.asarray(roi.get("total_through_upper_guard"), dtype=float)
        valid = np.asarray(roi.get("normalization_valid"))
        if any(array.shape != (38,) for array in (column, outside, lower, upper, far, total, valid)):
            raise AuditError("ROI guard diagnostics omit native input bins")
        if not all(np.all(np.isfinite(array)) for array in (column, outside, lower, upper, far, total)):
            raise AuditError("ROI guard diagnostics contain nonfinite values")
        if not np.allclose(outside, 1.0 - column, rtol=0.0, atol=2e-14):
            raise AuditError("outside-analysis definition is inconsistent")
        if not np.allclose(outside, lower + upper + far, rtol=0.0, atol=TOLERANCE):
            raise AuditError("outside-analysis guard decomposition is inconsistent")
        if not np.allclose(total, column + lower + upper, rtol=0.0, atol=2e-14):
            raise AuditError("total-through-guard definition is inconsistent")
        if np.any(total > 1.0 + TOLERANCE) or np.any(far < -TOLERANCE):
            raise AuditError("guard normalization bounds fail")
        if not np.all(valid):
            raise AuditError("ROI contains invalid normalization")
        proposal = roi.get("contiguous_run_geometry_proposal", {})
        if proposal.get("terminal_failure_allowed") is not True:
            raise AuditError("terminal failure is not explicitly allowed")
        if "scientific_frontier" not in proposal.get("proposal_semantics", ""):
            raise AuditError("geometry proposal semantics are missing")
    if numeric_keys != {"sphere_R5", "sphere_R8", "sphere_R31", "union_R6_M4"}:
        raise AuditError("unexpected numeric ROI window set")

    if counts.get("schema") != "ouruniv-cf4-kf-roi-leakage-mode-counts-v3":
        raise AuditError("unexpected mode-count schema")
    if counts.get("design_raw_sha256") != design_sha:
        raise AuditError("mode-count/design binding mismatch")
    if len(counts.get("native_bins", [])) != 38:
        raise AuditError("mode counts omit native bins")
    if counts.get("count_audit", {}).get("total_count_assertion_pass") is not True:
        raise AuditError("N^3 mode-count audit did not pass")

    if manifest.get("schema") != "ouruniv-cf4-kf-roi-leakage-artifact-manifest-v3":
        raise AuditError("unexpected manifest schema")
    if manifest.get("operational_execution_schema") != execution_schema:
        raise AuditError("manifest v3a operational schema mismatch")
    if manifest.get("status") != "PRECHECK_PASS" or manifest.get("mode") != "preflight":
        raise AuditError("manifest status/mode mismatch")
    for record, expected, label in (
        (manifest.get("design_raw_sha256"), design_sha, "manifest/design"),
        (manifest.get("execution_grant_raw_sha256"), grant_sha, "manifest/grant"),
        (manifest.get("implementation_commit"), implementation_commit, "manifest/commit"),
    ):
        if record != expected:
            raise AuditError(f"{label} binding mismatch")
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
    if complete.get("schema") != "ouruniv-cf4-kf-roi-leakage-complete-v3":
        raise AuditError("unexpected COMPLETE schema")
    if complete.get("operational_execution_schema") != execution_schema:
        raise AuditError("COMPLETE v3a operational schema mismatch")
    if complete.get("status") != "PRECHECK_PASS" or complete.get("mode") != "preflight":
        raise AuditError("COMPLETE status/mode mismatch")
    if complete.get("manifest_sha256") != _sha256(manifest_bytes):
        raise AuditError("COMPLETE does not bind manifest")
    for record, expected, label in (
        (complete.get("design_raw_sha256"), design_sha, "COMPLETE/design"),
        (complete.get("execution_grant_raw_sha256"), grant_sha, "COMPLETE/grant"),
        (complete.get("implementation_commit"), implementation_commit, "COMPLETE/commit"),
    ):
        if record != expected:
            raise AuditError(f"{label} binding mismatch")

    expected_arrays = set()
    for key in numeric_keys:
        expected_arrays.update((f"coarse__{key}", f"fine__{key}"))
        for order in ("coarse", "fine"):
            for guard in ("lower_guard", "upper_guard", "far_tail"):
                expected_arrays.add(f"{order}_{guard}__{key}")
    try:
        with np.load(io.BytesIO(payloads["mixing_matrices.npz"]), allow_pickle=False) as archive:
            if set(archive.files) != expected_arrays:
                raise AuditError("mixing/guard array set mismatch")
            for name in archive.files:
                array = archive[name]
                expected_shape = (38, 38) if name.startswith(("coarse__", "fine__")) else (38,)
                if (
                    array.shape != expected_shape
                    or not np.all(np.isfinite(array))
                    or np.any(array < -TOLERANCE)
                ):
                    raise AuditError(f"invalid array {name}")
            shell_norms = np.array(
                [
                    (float(item["upper_h_Mpc"]) ** 3 - float(item["lower_h_Mpc"]) ** 3)
                    / 3.0
                    for item in counts["native_bins"]
                ]
            )
            for key in numeric_keys:
                for order in ("coarse", "fine"):
                    matrix = archive[f"{order}__{key}"]
                    scaled = matrix * shell_norms[None, :]
                    scale = max(float(np.max(np.abs(scaled))), np.finfo(float).tiny)
                    if float(np.max(np.abs(scaled - scaled.T))) / scale > TOLERANCE:
                        raise AuditError(f"array reciprocity fails for {order} {key}")
            for roi in roi_results:
                key = roi["numeric_product_key"]
                matrix = archive[f"fine__{key}"]
                if not np.allclose(
                    matrix.sum(axis=0), roi["analysis_column_sum"], rtol=0.0, atol=2e-14
                ):
                    raise AuditError(f"fine matrix/result column mismatch for {key}")
                for guard in ("lower_guard", "upper_guard", "far_tail"):
                    if not np.allclose(
                        archive[f"fine_{guard}__{key}"],
                        roi[guard],
                        rtol=0.0,
                        atol=2e-14,
                    ):
                        raise AuditError(f"fine guard/result mismatch for {key} {guard}")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise AuditError("cannot validate mixing/guard archive") from exc

    implementation = Path(__file__).resolve().parents[1] / result["implementation_path"]
    if _sha256(implementation.read_bytes()) != result.get("implementation_sha256"):
        raise AuditError("implementation file SHA mismatch")
    return {
        "status": "PASS",
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
