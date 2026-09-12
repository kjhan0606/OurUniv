import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cf4_kf_bin_manifest as manifest  # noqa: E402


DESIGN_V1 = ROOT / "config" / "cf4_kf_bin_manifest_design_v1.json"
DESIGN_V2 = ROOT / "config" / "cf4_kf_bin_manifest_design_v2.json"
FINAL_MANIFEST = ROOT / "config" / "cf4_kf_bin_manifest_v1.json"


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _write_json(path, value):
    path.write_bytes(manifest.canonical_json_bytes(value))


def _native_bins():
    lattice = json.loads(DESIGN_V2.read_text())["analysis_lattice"]
    kf = lattice["fundamental_h_Mpc"]
    kny = lattice["isotropic_analysis_Nyquist_h_Mpc"]
    ratio = 2.0**0.25
    edges = [kf]
    while edges[-1] * ratio <= kny:
        edges.append(edges[-1] * ratio)
    if edges[-1] != kny:
        edges.append(kny)
    assert len(edges) == 39
    return [
        {
            "index": index,
            "lower_h_Mpc": lower,
            "upper_h_Mpc": upper,
            "representative_h_Mpc": math.sqrt(lower * upper),
            "terminal_upper_inclusive": index == 37,
            "full_vector_count": 64,
            "independent_real_mode_count": 32,
        }
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:]))
    ]


def _preflight(tmp_path):
    directory = tmp_path / "preflight"
    directory.mkdir(parents=True)
    design_sha = _sha(DESIGN_V2.read_bytes())
    native = _native_bins()
    merged = manifest._greedy_merges(native, 32)
    counts = {
        "schema": manifest.MODE_COUNTS_SCHEMA,
        "design_raw_sha256": design_sha,
        "grid_size_N": 1280,
        "native_bins": native,
        "merged_bins": merged,
        "count_audit": {"total_count_assertion_pass": True},
    }
    roi_results = []
    for offset, roi_id in enumerate(manifest.EXPECTED_ROI_IDS):
        supported = [10 + offset <= index <= 34 for index in range(38)]
        containment = [0.95 if value else 0.5 for value in supported]
        neff = [64.0 if value else 1.0 for value in supported]
        roi_results.append(
            {
                "ROI_id": roi_id,
                "numeric_product_key": f"window_{offset}",
                "containment": containment,
                "signed_outside_analysis_residual": [0.005] * 38,
                "localized_effective_independent_mode_count": neff,
                "normalization_valid": [True] * 38,
                "native_bin_supported": supported,
                "contiguous_run_geometry_proposal": manifest._proposal(supported, native),
            }
        )
    result = {
        "schema": manifest.RESULT_SCHEMA,
        "operational_execution_schema": "ouruniv-cf4-kf-roi-leakage-execution-v3a",
        "status": "PRECHECK_PASS",
        "mode": "preflight",
        "design_raw_sha256": design_sha,
        "execution_grant_raw_sha256": "a" * 64,
        "implementation_path": "src/cf4_kf_roi_leakage.py",
        "implementation_sha256": "b" * 64,
        "implementation_commit": "c" * 40,
        "truth_or_candidate_data_consumed": False,
        "geometry_window_proposals_are_scientific_claims": False,
        "scientific_leakage_decision_authorized": False,
        "final_manifest_materialized": False,
        "k_boundary_claim_created": False,
        "numerical_convergence": {
            "status": "PASS",
            "all_analysis_column_normalizations_valid": True,
            "native_classification_identical": True,
            "contiguous_run_proposal_identical": True,
            "threshold_margin_safety_pass": True,
        },
        "ROI_results": roi_results,
        "Local_Group_observer_numeric_product_shared": True,
        "Local_Group_observer_semantic_results_separate": True,
        "Local_Group_observer_scores_summed": False,
    }
    _write_json(directory / "result.json", result)
    _write_json(directory / "mode_counts.json", counts)
    (directory / "mixing_matrices.npz").write_bytes(b"synthetic-mixing")
    payloads = {
        name: (directory / name).read_bytes()
        for name in ("result.json", "mode_counts.json", "mixing_matrices.npz")
    }
    artifact = {
        "schema": manifest.ARTIFACT_MANIFEST_SCHEMA,
        "status": "PRECHECK_PASS",
        "mode": "preflight",
        "design_raw_sha256": design_sha,
        "execution_grant_raw_sha256": "a" * 64,
        "implementation_commit": "c" * 40,
        "operational_execution_schema": "ouruniv-cf4-kf-roi-leakage-execution-v3a",
        "payloads": {
            name: {"sha256": _sha(payload), "bytes": len(payload)}
            for name, payload in payloads.items()
        },
    }
    _write_json(directory / "manifest.json", artifact)
    complete = {
        "schema": manifest.COMPLETE_SCHEMA,
        "status": "PRECHECK_PASS",
        "mode": "preflight",
        "design_raw_sha256": design_sha,
        "execution_grant_raw_sha256": "a" * 64,
        "implementation_commit": "c" * 40,
        "operational_execution_schema": "ouruniv-cf4-kf-roi-leakage-execution-v3a",
        "manifest_sha256": _sha((directory / "manifest.json").read_bytes()),
    }
    _write_json(directory / "COMPLETE", complete)
    return directory


def test_build_is_deterministic_and_retains_every_bin_and_roi(tmp_path):
    preflight = _preflight(tmp_path)
    first = manifest.build_manifest_body(DESIGN_V1, DESIGN_V2, preflight)
    second = manifest.build_manifest_body(DESIGN_V1, DESIGN_V2, preflight)
    assert manifest.canonical_json_bytes(first) == manifest.canonical_json_bytes(second)
    envelope = manifest.make_envelope(first)
    assert envelope["manifest_body_sha256"] == _sha(manifest.canonical_json_bytes(first))
    body = manifest.validate_manifest_envelope(envelope)
    assert len(body["native_bins"]) == 38
    assert len(body["merged_bins"]) == 38
    assert len(body["ROI_support_records"]) == 6
    assert all(len(row["all_native_bin_geometry_records"]) == 38 for row in body["ROI_support_records"])
    assert body["semantics"]["observational_effective_resolution_claim_created"] is False


def test_support_fail_records_and_geometry_proposal_are_explicit(tmp_path):
    body = manifest.build_manifest_body(DESIGN_V1, DESIGN_V2, _preflight(tmp_path))
    row = body["ROI_support_records"][0]
    assert row["all_native_bin_geometry_records"][0]["geometry_supported"] is False
    assert "containment_below_0.9" in row["all_native_bin_geometry_records"][0]["failed_geometry_gates"]
    assert row["all_native_bin_geometry_records"][10]["geometry_supported"] is True
    assert row["all_native_bin_geometry_records"][10]["failed_geometry_gates"] == []
    assert row["contiguous_run_geometry_proposal"]["deterministic_proposal"]["start_native_bin"] == 10


def test_refuse_overwrite_and_cli_validate_canonical(tmp_path, capsys):
    envelope = manifest.make_envelope(
        manifest.build_manifest_body(DESIGN_V1, DESIGN_V2, _preflight(tmp_path))
    )
    output = tmp_path / "manifest.json"
    manifest.write_envelope(output, envelope)
    with pytest.raises(FileExistsError):
        manifest.write_envelope(output, envelope)
    assert manifest.main(["validate", "--manifest", str(output)]) == 0
    assert '"status": "PASS"' in capsys.readouterr().out


def test_tampered_body_hash_and_omitted_roi_bins_fail_closed(tmp_path):
    envelope = manifest.make_envelope(
        manifest.build_manifest_body(DESIGN_V1, DESIGN_V2, _preflight(tmp_path))
    )
    envelope["manifest_body"]["native_bins"][0]["independent_real_mode_count"] = 999
    with pytest.raises(manifest.ManifestError, match="SHA256 mismatch"):
        manifest.validate_manifest_envelope(envelope)
    envelope = manifest.make_envelope(
        manifest.build_manifest_body(DESIGN_V1, DESIGN_V2, _preflight(tmp_path / "again"))
    )
    envelope["manifest_body"]["ROI_support_records"][0]["all_native_bin_geometry_records"].pop()
    envelope["manifest_body_sha256"] = _sha(
        manifest.canonical_json_bytes(envelope["manifest_body"])
    )
    with pytest.raises(manifest.ManifestError, match="omits or reorders ROI"):
        manifest.validate_manifest_envelope(envelope)


def test_rehashed_non_greedy_merged_regrouping_fails_closed(tmp_path):
    envelope = manifest.make_envelope(
        manifest.build_manifest_body(DESIGN_V1, DESIGN_V2, _preflight(tmp_path))
    )
    body = envelope["manifest_body"]
    original = body["merged_bins"]
    regrouped = [
        {
            "merged_bin_index": 0,
            "native_bin_indices": [0, 1],
            "independent_real_mode_count": 64,
        }
    ]
    regrouped.extend(
        {
            "merged_bin_index": index,
            "native_bin_indices": row["native_bin_indices"],
            "independent_real_mode_count": row["independent_real_mode_count"],
        }
        for index, row in enumerate(original[2:], start=1)
    )
    body["merged_bins"] = regrouped
    envelope["manifest_body_sha256"] = _sha(manifest.canonical_json_bytes(body))
    with pytest.raises(manifest.ManifestError, match="exact frozen greedy grouping"):
        manifest.validate_manifest_envelope(envelope)


def test_preflight_payload_tamper_and_nonfinite_support_fail(tmp_path):
    preflight = _preflight(tmp_path)
    result = json.loads((preflight / "result.json").read_text())
    result["ROI_results"][0]["containment"][0] = float("nan")
    (preflight / "result.json").write_text(json.dumps(result))
    with pytest.raises(manifest.ManifestError, match="artifact payload binding mismatch"):
        manifest.build_manifest_body(DESIGN_V1, DESIGN_V2, preflight)


def test_tracked_manifest_is_canonical_and_bound_when_materialized():
    if not FINAL_MANIFEST.exists():
        pytest.skip("tracked manifest is generated after unit tests")
    value = json.loads(FINAL_MANIFEST.read_bytes())
    body = manifest.validate_manifest_envelope(value)
    assert FINAL_MANIFEST.read_bytes() == manifest.canonical_json_bytes(value)
    assert body["source_bindings"]["design_v1"]["raw_sha256"] == _sha(DESIGN_V1.read_bytes())
    assert body["source_bindings"]["design_v2"]["raw_sha256"] == _sha(DESIGN_V2.read_bytes())
