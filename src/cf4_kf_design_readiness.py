#!/usr/bin/env python3
"""Fail-closed, local-only readiness audit for the CF4 KF-DESIGN stage.

This audit reads only tracked repository contracts and the geometry-only bin
manifest.  It never reads GPFS/catalog data, submits work, opens mock truth,
or promotes a posterior.  A PASS here means that the design dossier is
internally consistent; it is not a KF-EXPAND or science authorization.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cf4_kf_bin_manifest import ManifestError, validate_manifest_envelope


ROOT = Path(__file__).resolve().parents[1]
ROUTE_PATH = ROOT / "config/cf4_science_route_v3.json"
DESIGN_PATH = ROOT / "config/cf4_kf_design_v1.json"
BIN_MANIFEST_PATH = ROOT / "config/cf4_kf_bin_manifest_v1.json"
LIKELIHOOD_PATH = ROOT / "config/cf4_2mpp_joint_likelihood_v1.json"
LOCAL_CONTRACT_PATH = ROOT / "config/cf4_2mpp_joint_likelihood_local_contract_v1.json"
CROSSMATCH_PATH = ROOT / "config/cf4_2mpp_crossmatch_v1_result.json"
BASELINE_PATH = ROOT / "config/cf4_kf_design_baseline_amendment_v1.json"
PARENT_ARCHITECTURE_PATH = ROOT / "config/cf4_independent_parent_architecture_design.json"


class ReadinessError(ValueError):
    """A repository contract cannot be audited safely."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"cannot load JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ReadinessError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _false_authority_flags(value: dict[str, Any], path: str) -> list[str]:
    """Return unexpected true execution/claim flags under an authority object."""

    authority = value.get(path)
    if not isinstance(authority, dict):
        return [f"missing_or_malformed:{path}"]
    forbidden_fragments = (
        "execution",
        "production",
        "slurm",
        "gpfs",
        "network",
        "kf_expand",
        "science_inference",
        "scientific_leakage",
        "final_manifest",
        "all_d_mock",
        "posterior",
        "catalog",
        "observational",
        "ic_pm",
        "hop",
        "ramses",
        "heldout",
        "external_artifact",
        "mock",
    )
    findings: list[str] = []
    for key, item in authority.items():
        lowered = str(key).lower()
        if any(fragment in lowered for fragment in forbidden_fragments) and item is True:
            findings.append(f"{path}.{key}=true")
    return findings


def audit_readiness(root: Path = ROOT) -> dict[str, Any]:
    """Audit current KF-DESIGN contracts and return a deterministic report."""

    paths = {
        "route": root / ROUTE_PATH.relative_to(ROOT),
        "design": root / DESIGN_PATH.relative_to(ROOT),
        "manifest": root / BIN_MANIFEST_PATH.relative_to(ROOT),
        "likelihood": root / LIKELIHOOD_PATH.relative_to(ROOT),
        "local_contract": root / LOCAL_CONTRACT_PATH.relative_to(ROOT),
        "crossmatch": root / CROSSMATCH_PATH.relative_to(ROOT),
        "baseline": root / BASELINE_PATH.relative_to(ROOT),
        "parent_architecture": root / PARENT_ARCHITECTURE_PATH.relative_to(ROOT),
    }
    values = {name: _load(path) for name, path in paths.items()}
    route = values["route"]
    design = values["design"]
    design_authority = design.get("authority")
    likelihood = values["likelihood"]
    local_contract = values["local_contract"]
    crossmatch = values["crossmatch"]
    baseline = values["baseline"]
    parent_architecture = values["parent_architecture"]

    findings: list[dict[str, str]] = []

    if route.get("schema") != "ouruniv-cf4-science-route-v3":
        findings.append({"id": "route_schema", "status": "FAIL", "detail": "unexpected route schema"})
    if route.get("status") != "active_constraint_frontier_hybrid_route_KF_DESIGN":
        findings.append({"id": "route_stage", "status": "FAIL", "detail": "route is not active KF-DESIGN"})
    stages = {item.get("id"): item for item in route.get("mandatory_stage_order", []) if isinstance(item, dict)}
    if stages.get("KF-DESIGN", {}).get("status") != "active":
        findings.append({"id": "kf_design_active", "status": "FAIL", "detail": "KF-DESIGN is not active"})
    if stages.get("KF-EXPAND", {}).get("status") != "blocked_by_KF_DESIGN":
        findings.append({"id": "kf_expand_firewall", "status": "FAIL", "detail": "KF-EXPAND firewall changed"})

    if design.get("schema") != "ouruniv-cf4-kf-design-v1":
        findings.append({"id": "design_schema", "status": "FAIL", "detail": "unexpected KF-DESIGN schema"})
    if design.get("status") != "user_approved_design_frozen_crossmatch_complete_blocked_by_likelihood_and_bin_manifest":
        findings.append({"id": "design_status", "status": "FAIL", "detail": "unexpected KF-DESIGN status"})

    for name, value, authority_path in (
        ("design", design, "authority"),
        ("likelihood", likelihood, "authority"),
        ("local_contract", local_contract, "authority"),
    ):
        for detail in _false_authority_flags(value, authority_path):
            findings.append({"id": f"{name}_authority", "status": "FAIL", "detail": detail})

    try:
        manifest = validate_manifest_envelope(values["manifest"])
    except (ManifestError, KeyError, TypeError, ValueError) as exc:
        findings.append({"id": "bin_manifest", "status": "FAIL", "detail": str(exc)})
        manifest = {}
    else:
        if manifest.get("semantics", {}).get("geometry_only") is not True:
            findings.append({"id": "bin_manifest_semantics", "status": "FAIL", "detail": "manifest is not geometry-only"})
        if manifest.get("semantics", {}).get("KF_EXPAND_authorized") is not False:
            findings.append({"id": "bin_manifest_authority", "status": "FAIL", "detail": "manifest authorizes KF-EXPAND"})

    if crossmatch.get("status") != "COMPLETE":
        findings.append({"id": "crossmatch", "status": "FAIL", "detail": "crossmatch result is not COMPLETE"})
    if baseline.get("schema") != "ouruniv-cf4-kf-design-baseline-amendment-v1":
        findings.append({"id": "baseline_schema", "status": "FAIL", "detail": "unexpected baseline amendment schema"})
    if baseline.get("status") != "FROZEN_REPOSITORY_BASELINE_EXTERNAL_ARTIFACTS_NOT_READ":
        findings.append({"id": "baseline_status", "status": "FAIL", "detail": "baseline is not record-only frozen"})
    baseline_authority = baseline.get("authorization")
    if not isinstance(baseline_authority, dict):
        findings.append({"id": "baseline_authority", "status": "FAIL", "detail": "baseline authorization is missing"})
    else:
        for key in ("external_artifact_read", "catalog_read", "GPFS_read", "GPFS_write", "Slurm_submission", "KF_EXPAND", "all_D_mock_execution", "observational_inference", "IC_PM_HOP_RAMSES", "network_access"):
            if baseline_authority.get(key) is not False:
                findings.append({"id": "baseline_authority", "status": "FAIL", "detail": f"baseline authorization {key} is not false"})
        for detail in _false_authority_flags(baseline, "authorization"):
            findings.append({"id": "baseline_authority", "status": "FAIL", "detail": detail})
    parent_bank = baseline.get("frozen_baseline", {}).get("current_parent_bank", {})
    if parent_bank.get("seed_range_inclusive") != [3193, 3448] or parent_bank.get("count") != 256:
        findings.append({"id": "baseline_parent_bank", "status": "FAIL", "detail": "sealed parent-bank range/count changed"})
    if parent_bank.get("external_manifest_sha256") != "dcf5d24104b178a74371455497f4228eb03188189937915672219f10d4c11687":
        findings.append({"id": "baseline_parent_manifest", "status": "FAIL", "detail": "sealed parent-bank manifest binding changed"})
    source_parent_hash = parent_architecture.get("sealed_CF4_parent_bank", {}).get("manifest_sha256")
    if source_parent_hash != parent_bank.get("external_manifest_sha256"):
        findings.append({"id": "baseline_parent_manifest", "status": "FAIL", "detail": "sealed parent-bank hash does not match tracked source contract"})
    hash_provenance = baseline.get("hash_provenance", {})
    if hash_provenance != {
        "path": "config/cf4_independent_parent_architecture_design.json",
        "json_pointer": "/sealed_CF4_parent_bank/manifest_sha256",
        "semantics": "transcribed historical binding; not recomputed because external GPFS manifest was not read",
    }:
        findings.append({"id": "baseline_hash_provenance", "status": "FAIL", "detail": "external manifest hash provenance is absent or changed"})
    if likelihood.get("status") != "crossmatch_complete_likelihood_blocked":
        findings.append({"id": "likelihood_status", "status": "FAIL", "detail": "unexpected 2M++ likelihood status"})

    # These are design requirements, not claims that the current dossier has
    # already met them.  Keep missing/partial records explicit and auditable.
    required_sections = {
        "data_inventory_provenance_selection_uncertainty_and_overlap": ("data_inventory", "overlap_policy"),
        "independent_selection_and_noise_truth_mock_contract": ("independent_mock_firewall",),
        "all_modeled_mode_and_declared_region_scope": ("objective", "domains", "evaluation_fields"),
        "global_and_ROI_definitions": ("domains",),
        "frontier_metrics": ("material_frontier_extension", "route_level_KF_EXPAND_GO"),
        "material_improvement_threshold_frozen_before_truth": ("material_frontier_extension",),
        "likelihood_and_ABC_preregistration": ("likelihood_subsets", "overlap_policy"),
    }
    missing_sections = [
        section
        for section, keys in required_sections.items()
        if any(key not in design for key in keys)
    ]
    # The route-level frozen_BGc/WF baseline is deliberately reported as a
    # missing design deliverable instead of being inferred from old artifacts.
    if baseline.get("scientific_disposition", {}).get("baseline_binding") != "complete_record_only":
        missing_sections.append("frozen_BGc_WF_and_current_artifact_baseline")
    for section in missing_sections:
        findings.append({"id": f"required_output:{section}", "status": "BLOCKED", "detail": "required design output is absent or incomplete"})

    blockers = []
    if isinstance(design_authority, dict):
        blockers.extend(str(item) for item in design_authority.get("all_D_mock_blockers", []))
    blockers.extend(str(item) for item in likelihood.get("blockers", []))
    if manifest:
        blockers = [item for item in blockers if "immutable declared k-bin manifest" not in item]
    blockers.extend([
        "KF-DESIGN readiness is not a science inference or observational-resolution claim",
        "no KF-EXPAND, all-D mock, Slurm, GPFS, catalog, IC, PM, HOP, or RAMSES execution is authorized by this audit",
    ])
    # Preserve order while removing duplicate blocker text.
    blockers = list(dict.fromkeys(blockers))

    hard_failures = [item for item in findings if item["status"] == "FAIL"]
    status = "PASS_DESIGN_INTERNAL_CONSISTENCY_BLOCKED_BY_SCIENCE_INPUTS" if not hard_failures else "FAIL_CONTRACT_INCONSISTENCY"
    source_bindings = {
        key: {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
        for key, path in paths.items()
    }
    if isinstance(values["manifest"].get("manifest_body_sha256"), str):
        source_bindings["manifest"]["body_sha256"] = values["manifest"]["manifest_body_sha256"]

    return {
        "schema": "ouruniv-cf4-kf-design-readiness-result-v1",
        "stage": "KF-DESIGN",
        "status": status,
        "read_only_local_audit": True,
        "source_bindings": source_bindings,
        "checks": findings,
        "geometry_manifest": {
            "validated": not any(item["id"] == "bin_manifest" and item["status"] == "FAIL" for item in findings),
            "body_sha256": values["manifest"].get("manifest_body_sha256"),
            "native_bin_count": manifest.get("native_bin_count"),
            "roi_count": len(manifest.get("ROI_support_records", [])) if manifest else 0,
            "science_claim_created": False,
        },
        "baseline_binding": {
            "validated": not any(item["id"].startswith("baseline") and item["status"] == "FAIL" for item in findings),
            "external_artifacts_read": False,
            "parent_bank_seed_range": parent_bank.get("seed_range_inclusive"),
            "parent_bank_count": parent_bank.get("count"),
        },
        "blockers": blockers,
        "next_action": "Complete source-bound selection and likelihood design contracts, then request bundle-closure audit; keep KF-EXPAND and all execution closed.",
        "authorization": {
            "KF_EXPAND": False,
            "all_D_mock_execution": False,
            "observational_inference": False,
            "Slurm_submission": False,
            "GPFS_read": False,
            "GPFS_write": False,
            "network_access": False,
            "IC_PM_HOP_RAMSES": False,
        },
    }


def main(root: Path = ROOT) -> int:
    try:
        report = audit_readiness(root)
    except (ReadinessError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"schema": "ouruniv-cf4-kf-design-readiness-result-v1", "status": "FAIL_CONTRACT_INCONSISTENCY", "error": str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["status"] != "FAIL_CONTRACT_INCONSISTENCY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
