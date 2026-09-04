import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cf4_kf_design_readiness import audit_readiness  # noqa: E402


def test_readiness_is_internal_consistency_pass_but_science_blocked():
    report = audit_readiness(ROOT)

    assert report["status"] == "PASS_DESIGN_INTERNAL_CONSISTENCY_BLOCKED_BY_SCIENCE_INPUTS"
    assert report["read_only_local_audit"] is True
    assert report["geometry_manifest"]["validated"] is True
    assert report["geometry_manifest"]["native_bin_count"] == 38
    assert report["geometry_manifest"]["roi_count"] == 6
    assert report["geometry_manifest"]["science_claim_created"] is False
    assert report["authorization"] == {
        "KF_EXPAND": False,
        "all_D_mock_execution": False,
        "observational_inference": False,
        "Slurm_submission": False,
        "GPFS_read": False,
        "GPFS_write": False,
        "network_access": False,
        "IC_PM_HOP_RAMSES": False,
    }


def test_readiness_reports_missing_baseline_and_likelihood_blockers():
    report = audit_readiness(ROOT)
    ids = {item["id"] for item in report["checks"]}
    assert "required_output:frozen_BGc_WF_and_current_artifact_baseline" not in ids
    assert report["baseline_binding"] == {
        "validated": True,
        "external_artifacts_read": False,
        "parent_bank_seed_range": [3193, 3448],
        "parent_bank_count": 256,
    }
    assert any("angular empty-sky" in blocker for blocker in report["blockers"])
    assert not any("immutable declared k-bin manifest" in blocker for blocker in report["blockers"])
    assert "Complete source-bound selection" in report["next_action"]


def test_readiness_keeps_route_firewall_closed():
    report = audit_readiness(ROOT)
    assert any(item["id"] == "required_output:likelihood_and_ABC_preregistration" and item["status"] == "BLOCKED" for item in report["checks"]) is False
    assert any(item["id"] == "kf_expand_firewall" and item["status"] == "FAIL" for item in report["checks"]) is False
    assert all("execution" not in str(item).lower() or item["status"] != "FAIL" for item in report["checks"])


def test_tracked_result_record_matches_current_readiness_bindings():
    report = audit_readiness(ROOT)
    record = __import__("json").loads(
        (ROOT / "config/cf4_kf_design_readiness_result_v1.json").read_text()
    )

    assert record["schema"] == report["schema"]
    assert record["status"] == report["status"]
    assert record["source_bindings"] == report["source_bindings"]
    assert record["geometry_manifest"]["native_bin_count"] == report["geometry_manifest"]["native_bin_count"]
    assert record["geometry_manifest"]["roi_count"] == report["geometry_manifest"]["roi_count"]
    assert record["authorization"] == report["authorization"]
