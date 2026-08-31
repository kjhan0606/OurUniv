#!/usr/bin/env python3
"""Build and validate the immutable CF4 KF bin-manifest envelope.

The manifest records numerical lattice bins and ROI-window geometry support.
It does not create an observational-resolution, k-frontier, or science claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence


DESIGN_V1_SCHEMA = "ouruniv-cf4-kf-bin-manifest-design-v1"
DESIGN_V2_SCHEMA = "ouruniv-cf4-kf-bin-manifest-design-v2"
RESULT_SCHEMA = "ouruniv-cf4-kf-roi-leakage-result-v3"
MODE_COUNTS_SCHEMA = "ouruniv-cf4-kf-roi-leakage-mode-counts-v3"
ARTIFACT_MANIFEST_SCHEMA = "ouruniv-cf4-kf-roi-leakage-artifact-manifest-v3"
COMPLETE_SCHEMA = "ouruniv-cf4-kf-roi-leakage-complete-v3"
BODY_SCHEMA = "ouruniv-cf4-kf-bin-manifest-body-v1"
ENVELOPE_SCHEMA = "ouruniv-cf4-kf-bin-manifest-envelope-v1"
EXPECTED_ROI_IDS = (
    "Local_Group",
    "Virgo",
    "Coma",
    "Local_Void",
    "Bootes_Void",
    "observer_environment",
)
EXPECTED_PREFLIGHT_FILES = {
    "COMPLETE",
    "manifest.json",
    "mixing_matrices.npz",
    "mode_counts.json",
    "result.json",
}


class ManifestError(ValueError):
    """An input or manifest violates the frozen fail-closed contract."""


def canonical_json_bytes(value: object) -> bytes:
    """Canonical compact UTF-8 JSON terminated by exactly one newline."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_object(path: Path) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot parse JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"JSON root must be an object: {path}")
    return value, payload


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ManifestError(f"{label} must be lowercase SHA256")
    return value


def _finite_vector(value: object, size: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ManifestError(f"{label} must contain all {size} native bins")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ManifestError(f"{label} must contain real numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ManifestError(f"{label} contains a nonfinite value")
        output.append(number)
    return output


def _boolean_vector(value: object, size: int, label: str) -> list[bool]:
    if (
        not isinstance(value, list)
        or len(value) != size
        or any(type(item) is not bool for item in value)
    ):
        raise ManifestError(f"{label} must contain exact booleans for all {size} bins")
    return list(value)


def _greedy_merges(native_bins: list[dict[str, object]], minimum: int) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    membership: list[int] = []
    count = 0
    for item in native_bins:
        membership.append(int(item["index"]))
        count += int(item["independent_real_mode_count"])
        if count >= minimum:
            groups.append(
                {
                    "merged_bin_index": len(groups),
                    "native_bin_indices": membership,
                    "independent_real_mode_count": count,
                }
            )
            membership = []
            count = 0
    if membership:
        if not groups:
            raise ManifestError("terminal underfill has no preceding merged bin")
        groups[-1]["native_bin_indices"].extend(membership)
        groups[-1]["independent_real_mode_count"] += count
    return groups


def _maximal_runs(
    supported: list[bool], native_bins: list[dict[str, object]]
) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    start: int | None = None
    for index, passed in enumerate(supported + [False]):
        if passed and start is None:
            start = index
        elif not passed and start is not None:
            end = index - 1
            runs.append(
                {
                    "start_native_bin": start,
                    "end_native_bin": end,
                    "native_bin_count": end - start + 1,
                    "lower_h_Mpc": native_bins[start]["lower_h_Mpc"],
                    "upper_h_Mpc": native_bins[end]["upper_h_Mpc"],
                    "summed_independent_real_modes": sum(
                        int(native_bins[j]["independent_real_mode_count"])
                        for j in range(start, end + 1)
                    ),
                    "includes_terminal_bin": end == len(native_bins) - 1,
                }
            )
            start = None
    return runs


def _proposal(supported: list[bool], native_bins: list[dict[str, object]]) -> dict[str, object]:
    runs = _maximal_runs(supported, native_bins)
    chosen = (
        max(
            runs,
            key=lambda item: (
                float(item["upper_h_Mpc"]),
                int(item["native_bin_count"]),
                int(item["summed_independent_real_modes"]),
            ),
        )
        if runs
        else None
    )
    return {
        "all_maximal_contiguous_runs": runs,
        "deterministic_proposal": chosen,
        "proposal_semantics": "geometry_window_only_not_scientific_frontier_or_observational_resolution",
        "selection_order": "highest_upper_k_then_more_bins_then_larger_summed_independent_modes",
        "terminal_failure_allowed": True,
    }


def _validate_preflight(
    directory: Path,
    design_v2_sha: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if not directory.is_dir():
        raise ManifestError("preflight directory is absent")
    if {item.name for item in directory.iterdir()} != EXPECTED_PREFLIGHT_FILES:
        raise ManifestError("preflight file set is not exact")

    result, result_bytes = _load_object(directory / "result.json")
    counts, counts_bytes = _load_object(directory / "mode_counts.json")
    artifact, artifact_bytes = _load_object(directory / "manifest.json")
    complete, _ = _load_object(directory / "COMPLETE")
    mixing_bytes = (directory / "mixing_matrices.npz").read_bytes()

    if result.get("schema") != RESULT_SCHEMA or result.get("status") != "PRECHECK_PASS":
        raise ManifestError("ROI leakage result is not PRECHECK_PASS v3")
    if result.get("mode") != "preflight":
        raise ManifestError("ROI leakage result is not a preflight")
    if result.get("design_raw_sha256") != design_v2_sha:
        raise ManifestError("ROI result/design v2 binding mismatch")
    if result.get("truth_or_candidate_data_consumed") is not False:
        raise ManifestError("ROI leakage consumed truth or candidate data")
    for key in (
        "geometry_window_proposals_are_scientific_claims",
        "scientific_leakage_decision_authorized",
        "final_manifest_materialized",
        "k_boundary_claim_created",
    ):
        if result.get(key) is not False:
            raise ManifestError(f"forbidden science-state flag is not false: {key}")
    convergence = result.get("numerical_convergence")
    if not isinstance(convergence, dict) or convergence.get("status") != "PASS":
        raise ManifestError("ROI numerical convergence did not pass")
    for key in (
        "all_analysis_column_normalizations_valid",
        "native_classification_identical",
        "contiguous_run_proposal_identical",
        "threshold_margin_safety_pass",
    ):
        if convergence.get(key) is not True:
            raise ManifestError(f"ROI numerical convergence flag did not pass: {key}")

    if counts.get("schema") != MODE_COUNTS_SCHEMA:
        raise ManifestError("unexpected mode-count schema")
    if counts.get("design_raw_sha256") != design_v2_sha:
        raise ManifestError("mode-count/design v2 binding mismatch")
    audit = counts.get("count_audit")
    if not isinstance(audit, dict) or audit.get("total_count_assertion_pass") is not True:
        raise ManifestError("mode-count lattice audit did not pass")

    if artifact.get("schema") != ARTIFACT_MANIFEST_SCHEMA or artifact.get("status") != "PRECHECK_PASS":
        raise ManifestError("artifact manifest is not PRECHECK_PASS v3")
    payloads = artifact.get("payloads")
    expected_payloads = {
        "result.json": result_bytes,
        "mode_counts.json": counts_bytes,
        "mixing_matrices.npz": mixing_bytes,
    }
    if not isinstance(payloads, dict) or set(payloads) != set(expected_payloads):
        raise ManifestError("artifact payload set is not exact")
    for name, payload in expected_payloads.items():
        record = payloads.get(name)
        if not isinstance(record, dict) or record != {
            "sha256": _sha256(payload),
            "bytes": len(payload),
        }:
            raise ManifestError(f"artifact payload binding mismatch: {name}")
    if artifact.get("design_raw_sha256") != design_v2_sha:
        raise ManifestError("artifact/design v2 binding mismatch")
    if artifact.get("execution_grant_raw_sha256") != result.get("execution_grant_raw_sha256"):
        raise ManifestError("artifact/result grant binding mismatch")
    if artifact.get("implementation_commit") != result.get("implementation_commit"):
        raise ManifestError("artifact/result commit binding mismatch")

    if complete.get("schema") != COMPLETE_SCHEMA or complete.get("status") != "PRECHECK_PASS":
        raise ManifestError("COMPLETE is not PRECHECK_PASS v3")
    if complete.get("manifest_sha256") != _sha256(artifact_bytes):
        raise ManifestError("COMPLETE/artifact-manifest binding mismatch")
    for key in (
        "design_raw_sha256",
        "execution_grant_raw_sha256",
        "implementation_commit",
        "operational_execution_schema",
    ):
        if complete.get(key) != artifact.get(key):
            raise ManifestError(f"COMPLETE/artifact binding mismatch: {key}")

    source_bindings = {
        "preflight_directory": str(directory),
        "result_json_sha256": _sha256(result_bytes),
        "mode_counts_json_sha256": _sha256(counts_bytes),
        "mixing_matrices_npz_sha256": _sha256(mixing_bytes),
        "artifact_manifest_json_sha256": _sha256(artifact_bytes),
        "COMPLETE_sha256": _sha256((directory / "COMPLETE").read_bytes()),
        "execution_grant_raw_sha256": _require_sha(
            result.get("execution_grant_raw_sha256"), "execution grant binding"
        ),
        "implementation_path": result.get("implementation_path"),
        "implementation_sha256": _require_sha(
            result.get("implementation_sha256"), "implementation binding"
        ),
        "implementation_commit": result.get("implementation_commit"),
        "operational_execution_schema": result.get("operational_execution_schema"),
    }
    if not isinstance(source_bindings["implementation_path"], str):
        raise ManifestError("implementation path is missing")
    if not isinstance(source_bindings["implementation_commit"], str) or re.fullmatch(
        r"[0-9a-f]{40}", source_bindings["implementation_commit"]
    ) is None:
        raise ManifestError("implementation commit must be lowercase 40-hex")
    return result, counts, source_bindings


def build_manifest_body(
    design_v1_path: str | Path,
    design_v2_path: str | Path,
    preflight_directory: str | Path,
) -> dict[str, object]:
    """Validate frozen inputs and return a deterministic manifest body."""

    design_v1_path = Path(design_v1_path)
    design_v2_path = Path(design_v2_path)
    design_v1, design_v1_bytes = _load_object(design_v1_path)
    design_v2, design_v2_bytes = _load_object(design_v2_path)
    design_v1_sha = _sha256(design_v1_bytes)
    design_v2_sha = _sha256(design_v2_bytes)
    if design_v1.get("schema") != DESIGN_V1_SCHEMA:
        raise ManifestError("unexpected design v1 schema")
    if design_v2.get("schema") != DESIGN_V2_SCHEMA:
        raise ManifestError("unexpected design v2 schema")
    predecessor = design_v2.get("predecessor")
    if not isinstance(predecessor, dict) or predecessor.get("design_raw_sha256") != design_v1_sha:
        raise ManifestError("design v2/design v1 binding mismatch")

    result, counts, preflight_bindings = _validate_preflight(
        Path(preflight_directory), design_v2_sha
    )
    native_bins = counts.get("native_bins")
    if not isinstance(native_bins, list) or len(native_bins) != 38:
        raise ManifestError("mode counts must contain exactly 38 native bins")
    if [item.get("index") for item in native_bins if isinstance(item, dict)] != list(range(38)):
        raise ManifestError("native bins are missing, reordered, or malformed")
    previous_upper: float | None = None
    for index, item in enumerate(native_bins):
        if not isinstance(item, dict):
            raise ManifestError("native bin record must be an object")
        lower = float(item["lower_h_Mpc"])
        upper = float(item["upper_h_Mpc"])
        representative = float(item["representative_h_Mpc"])
        if not all(math.isfinite(x) for x in (lower, upper, representative)) or not lower < upper:
            raise ManifestError("native bin edges are invalid")
        if previous_upper is not None and lower != previous_upper:
            raise ManifestError("native bins contain a gap or overlap")
        if representative != math.sqrt(lower * upper):
            raise ManifestError("native bin representative is not geometric")
        if type(item.get("terminal_upper_inclusive")) is not bool:
            raise ManifestError("terminal inclusion flag is not boolean")
        if item["terminal_upper_inclusive"] != (index == 37):
            raise ManifestError("terminal inclusion flag is misplaced")
        for count_key in ("full_vector_count", "independent_real_mode_count"):
            if type(item.get(count_key)) is not int or item[count_key] < 0:
                raise ManifestError(f"invalid native bin {count_key}")
        previous_upper = upper

    lattice = design_v2.get("analysis_lattice")
    if not isinstance(lattice, dict):
        raise ManifestError("design v2 lattice is missing")
    if native_bins[0]["lower_h_Mpc"] != lattice.get("fundamental_h_Mpc"):
        raise ManifestError("native bins do not begin at the fundamental")
    if native_bins[-1]["upper_h_Mpc"] != lattice.get("isotropic_analysis_Nyquist_h_Mpc"):
        raise ManifestError("native bins do not end at the analysis Nyquist")
    expected_edges = [float(lattice["fundamental_h_Mpc"])]
    nyquist = float(lattice["isotropic_analysis_Nyquist_h_Mpc"])
    edge_ratio = 2.0**0.25
    while expected_edges[-1] * edge_ratio <= nyquist:
        expected_edges.append(expected_edges[-1] * edge_ratio)
    if expected_edges[-1] != nyquist:
        expected_edges.append(nyquist)
    if len(expected_edges) != 39:
        raise ManifestError("frozen lattice does not produce exactly 38 native bins")
    for index, item in enumerate(native_bins):
        if (
            item["lower_h_Mpc"] != expected_edges[index]
            or item["upper_h_Mpc"] != expected_edges[index + 1]
        ):
            raise ManifestError("native bin edges violate frozen quarter-octave construction")

    minimum = int(
        design_v2["independent_real_mode_count"][
            "minimum_modes_for_greedy_low_mode_merge"
        ]
    )
    merged_bins = _greedy_merges(native_bins, minimum)
    if counts.get("merged_bins") != merged_bins:
        raise ManifestError("stored merged bins violate the frozen greedy rule")

    roi_rows = result.get("ROI_results")
    if not isinstance(roi_rows, list) or [row.get("ROI_id") for row in roi_rows if isinstance(row, dict)] != list(EXPECTED_ROI_IDS):
        raise ManifestError("ROI results are missing, reordered, or malformed")
    gates = design_v2["native_bin_support_and_proposal"]["raw_support_gates"]
    containment_min = float(
        gates["response_contained_in_b_minus_1_b_b_plus_1_min_inclusive"]
    )
    outside_max = float(gates["outside_analysis_fraction_max_inclusive"])
    neff_min = float(gates["localized_effective_independent_mode_count_min_inclusive"])
    roi_records: list[dict[str, object]] = []
    for row in roi_rows:
        containment = _finite_vector(row.get("containment"), 38, "containment")
        outside = _finite_vector(
            row.get("signed_outside_analysis_residual"), 38, "outside-analysis"
        )
        neff = _finite_vector(
            row.get("localized_effective_independent_mode_count"), 38, "localized mode count"
        )
        normalization = _boolean_vector(
            row.get("normalization_valid"), 38, "normalization_valid"
        )
        supported = _boolean_vector(
            row.get("native_bin_supported"), 38, "native_bin_supported"
        )
        recomputed = [
            containment[i] >= containment_min
            and outside[i] <= outside_max
            and neff[i] >= neff_min
            and normalization[i]
            for i in range(38)
        ]
        if supported != recomputed:
            raise ManifestError(f"ROI support mask does not match frozen gates: {row['ROI_id']}")
        proposal = _proposal(supported, native_bins)
        if row.get("contiguous_run_geometry_proposal") != proposal:
            raise ManifestError(f"ROI geometry proposal is not deterministic: {row['ROI_id']}")
        bin_records: list[dict[str, object]] = []
        for index in range(38):
            failed: list[str] = []
            if containment[index] < containment_min:
                failed.append("containment_below_0.9")
            if outside[index] > outside_max:
                failed.append("outside_analysis_above_0.01")
            if neff[index] < neff_min:
                failed.append("localized_independent_modes_below_32")
            if not normalization[index]:
                failed.append("numerical_normalization_invalid")
            bin_records.append(
                {
                    "native_bin_index": index,
                    "lower_h_Mpc": native_bins[index]["lower_h_Mpc"],
                    "upper_h_Mpc": native_bins[index]["upper_h_Mpc"],
                    "containment": containment[index],
                    "signed_outside_analysis_residual": outside[index],
                    "localized_effective_independent_mode_count": neff[index],
                    "normalization_valid": normalization[index],
                    "geometry_supported": supported[index],
                    "failed_geometry_gates": failed,
                }
            )
        roi_records.append(
            {
                "ROI_id": row["ROI_id"],
                "numeric_product_key": row.get("numeric_product_key"),
                "all_native_bin_geometry_records": bin_records,
                "contiguous_run_geometry_proposal": proposal,
                "semantics": "geometry_window_only_not_CF4_observational_effective_resolution_or_scientific_frontier",
            }
        )

    if result.get("Local_Group_observer_numeric_product_shared") is not True:
        raise ManifestError("Local Group/observer shared numeric product is not recorded")
    if result.get("Local_Group_observer_semantic_results_separate") is not True:
        raise ManifestError("Local Group/observer semantic separation is absent")
    if result.get("Local_Group_observer_scores_summed") is not False:
        raise ManifestError("Local Group/observer scores must not be summed")

    return {
        "schema": BODY_SCHEMA,
        "date": "2026-08-31",
        "status": "MATERIALIZED_GEOMETRY_ONLY_NO_SCIENCE_CLAIM",
        "stage": "KF-DESIGN",
        "source_bindings": {
            "design_v1": {
                "path": str(design_v1_path),
                "raw_sha256": design_v1_sha,
            },
            "design_v2": {
                "path": str(design_v2_path),
                "raw_sha256": design_v2_sha,
            },
            "ROI_leakage_PRECHECK_PASS": preflight_bindings,
        },
        "analysis_lattice": lattice,
        "native_bin_count": 38,
        "native_bins": native_bins,
        "low_mode_greedy_merge_contract": {
            "minimum_independent_real_modes": minimum,
            "terminal_underfill_merges_previous": True,
            "truth_or_candidate_input_used": False,
        },
        "merged_bins": merged_bins,
        "mode_count_audit": counts["count_audit"],
        "ROI_geometry_source": design_v1["ROI_geometry"],
        "ROI_support_records": roi_records,
        "ROI_overlap_contract": {
            "Local_Group_and_observer_environment_windows_are_spatially_identical": True,
            "semantic_scores_are_separate": True,
            "scores_may_be_summed": False,
        },
        "semantics": {
            "geometry_only": True,
            "truth_or_candidate_data_consumed": False,
            "observational_effective_resolution_claim_created": False,
            "scientific_frontier_or_k_boundary_claim_created": False,
            "KF_EXPAND_authorized": False,
            "all_D_mock_or_validation_execution_authorized": False,
            "production_or_science_inference_authorized": False,
        },
    }


def make_envelope(body: Mapping[str, object]) -> dict[str, object]:
    """Place the canonical body SHA outside the body without recursion."""

    body_copy = dict(body)
    return {
        "schema": ENVELOPE_SCHEMA,
        "manifest_body": body_copy,
        "manifest_body_sha256": _sha256(canonical_json_bytes(body_copy)),
    }


def validate_manifest_envelope(value: object) -> dict[str, object]:
    """Validate envelope shape, canonical body hash, and no-claim semantics."""

    if not isinstance(value, dict) or set(value) != {
        "schema",
        "manifest_body",
        "manifest_body_sha256",
    }:
        raise ManifestError("manifest envelope key set is not exact")
    if value.get("schema") != ENVELOPE_SCHEMA:
        raise ManifestError("unexpected manifest envelope schema")
    body = value.get("manifest_body")
    if not isinstance(body, dict) or body.get("schema") != BODY_SCHEMA:
        raise ManifestError("unexpected manifest body schema")
    expected = _sha256(canonical_json_bytes(body))
    if value.get("manifest_body_sha256") != expected:
        raise ManifestError("manifest body SHA256 mismatch")
    native_bins = body.get("native_bins")
    if body.get("native_bin_count") != 38 or not isinstance(native_bins, list) or len(native_bins) != 38:
        raise ManifestError("manifest body does not retain all 38 native bins")
    if [item.get("index") for item in native_bins if isinstance(item, dict)] != list(range(38)):
        raise ManifestError("manifest native bins are missing or reordered")
    merged_bins = body.get("merged_bins")
    if not isinstance(merged_bins, list) or not merged_bins:
        raise ManifestError("manifest merged bins are absent")
    flattened: list[int] = []
    for merged_index, merged in enumerate(merged_bins):
        if not isinstance(merged, dict) or merged.get("merged_bin_index") != merged_index:
            raise ManifestError("manifest merged bins are malformed or reordered")
        membership = merged.get("native_bin_indices")
        if not isinstance(membership, list) or any(type(index) is not int for index in membership):
            raise ManifestError("manifest merged-bin membership is malformed")
        expected_count = sum(
            int(native_bins[index]["independent_real_mode_count"])
            for index in membership
            if 0 <= index < 38
        )
        if (
            any(index < 0 or index >= 38 for index in membership)
            or merged.get("independent_real_mode_count") != expected_count
            or expected_count < 32
        ):
            raise ManifestError("manifest merged-bin count/membership violates frozen rule")
        flattened.extend(membership)
    if flattened != list(range(38)):
        raise ManifestError("manifest merged bins omit, duplicate, or reorder native bins")
    merge_contract = body.get("low_mode_greedy_merge_contract")
    if not isinstance(merge_contract, dict):
        raise ManifestError("manifest low-mode greedy-merge contract is absent")
    minimum = merge_contract.get("minimum_independent_real_modes")
    if type(minimum) is not int or minimum != 32:
        raise ManifestError("manifest greedy-merge minimum is not frozen at 32")
    if merged_bins != _greedy_merges(native_bins, minimum):
        raise ManifestError("manifest merged bins are not the exact frozen greedy grouping")
    roi_records = body.get("ROI_support_records")
    if not isinstance(roi_records, list) or [row.get("ROI_id") for row in roi_records] != list(EXPECTED_ROI_IDS):
        raise ManifestError("manifest body does not retain all six ordered ROIs")
    for row in roi_records:
        records = row.get("all_native_bin_geometry_records", [])
        if len(records) != 38 or [record.get("native_bin_index") for record in records] != list(range(38)):
            raise ManifestError("manifest body omits or reorders ROI support/fail bins")
        supported: list[bool] = []
        for record in records:
            if type(record.get("geometry_supported")) is not bool:
                raise ManifestError("ROI geometry-supported flag is not boolean")
            failed = record.get("failed_geometry_gates")
            if not isinstance(failed, list) or any(not isinstance(item, str) for item in failed):
                raise ManifestError("ROI failed-gate record is malformed")
            if record["geometry_supported"] != (len(failed) == 0):
                raise ManifestError("ROI support flag and failed-gate record disagree")
            supported.append(record["geometry_supported"])
        if row.get("contiguous_run_geometry_proposal") != _proposal(supported, native_bins):
            raise ManifestError("ROI geometry proposal does not match support records")
    semantics = body.get("semantics")
    if not isinstance(semantics, dict) or semantics.get("geometry_only") is not True:
        raise ManifestError("manifest is not explicitly geometry-only")
    for key in (
        "observational_effective_resolution_claim_created",
        "scientific_frontier_or_k_boundary_claim_created",
        "KF_EXPAND_authorized",
        "all_D_mock_or_validation_execution_authorized",
        "production_or_science_inference_authorized",
    ):
        if semantics.get(key) is not False:
            raise ManifestError(f"manifest contains forbidden authority/claim: {key}")
    return body


def write_envelope(path: str | Path, envelope: Mapping[str, object]) -> None:
    """Write once with O_EXCL semantics; existing outputs are never replaced."""

    validate_manifest_envelope(dict(envelope))
    output = Path(path)
    if not output.parent.is_dir():
        raise ManifestError("output parent directory does not exist")
    payload = canonical_json_bytes(envelope)
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--design-v1", required=True, type=Path)
    build.add_argument("--design-v2", required=True, type=Path)
    build.add_argument("--preflight", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            body = build_manifest_body(args.design_v1, args.design_v2, args.preflight)
            envelope = make_envelope(body)
            write_envelope(args.output, envelope)
            report = {
                "status": "PASS",
                "output": str(args.output),
                "manifest_body_sha256": envelope["manifest_body_sha256"],
                "semantics": "geometry_only_no_science_claim",
            }
        else:
            value, payload = _load_object(args.manifest)
            validate_manifest_envelope(value)
            if payload != canonical_json_bytes(value):
                raise ManifestError("manifest envelope file is not canonical JSON")
            report = {
                "status": "PASS",
                "manifest": str(args.manifest),
                "manifest_file_sha256": _sha256(payload),
                "manifest_body_sha256": value["manifest_body_sha256"],
            }
    except (OSError, KeyError, TypeError, ManifestError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
