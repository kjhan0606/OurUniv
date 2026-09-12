from __future__ import annotations

import json
from pathlib import Path

import pytest

from cf4_q4_preflight import MemoryCeilingError, M, build_seed_manifest, canonical_sha256, enforce_memory_ceiling, geometry_preflight


def test_seed_firewall_is_disjoint_and_hashed() -> None:
    manifest = build_seed_manifest()
    assert manifest["development_count"] == 192
    assert manifest["heldout_count"] == 256
    assert manifest["cross_namespace_disjoint"] is True
    assert len(manifest["manifest_sha256"]) == 64
    payload = dict(manifest)
    digest = payload.pop("manifest_sha256")
    assert digest == canonical_sha256(payload)


def test_procedural_geometry_is_full_count_and_injective() -> None:
    result = geometry_preflight()
    assert result["source_count"] == M == 7_077_888
    assert result["unique_positions_asserted_by_construction"] == M
    assert result["streamed_injective_entries"] == M
    assert result["actual_composite_key_entries_hashed"] == M
    assert len(result["actual_composite_key_payload_sha256"]) == 64
    assert result["group_count_implied_by_injective_lattice"] == M
    assert result["compression_ratio_implied_M_over_G"] == 1.0
    assert result["full_pm_geometry_allocated"] is False


def test_terminal_branch_skips_expensive_gradient() -> None:
    result = geometry_preflight()
    assert result["terminal_exact_grouping_decision"] == "NO_GO_M_OVER_G_LE_1.01"
    assert result["q4_3_gradient_timing"] == "SKIPPED_BY_TERMINAL_BRANCH"


def test_metadata_and_memory_are_precomputed() -> None:
    result = geometry_preflight()
    assert len(result["metadata_sha256"]) == 64
    assert result["dictionary_allocation_performed"] is False
    assert result["projected_persistent_plus_dictionary_bytes"] > result["projected_persistent_array_total_bytes"]
    assert result["host_memory_gate"] == "PASS_PROJECTED_BELOW_CEILING"
    assert result["device_memory_gate"] == "PASS_PROJECTED_BELOW_CEILING"
    assert result["composite_key_injectivity"].startswith("ANALYTIC_ARGUMENT_BY_POSITION_REFINEMENT")


def test_memory_gate_fails_before_forced_breach_allocation() -> None:
    with pytest.raises(MemoryCeilingError) as error:
        enforce_memory_ceiling(101, host_ceiling_bytes=100, device_ceiling_bytes=1000)
    assert error.value.details["host_status"] == "FAIL_PROJECTED_ABOVE_HOST_CEILING"
    assert error.value.details["device_status"] == "PASS_PROJECTED_BELOW_CEILING"
    assert error.value.details["status"] == "FAIL_PROJECTED_ABOVE_MEMORY_CEILING"


def test_memory_gate_reports_device_only_breach_and_equality_boundary() -> None:
    with pytest.raises(MemoryCeilingError) as error:
        enforce_memory_ceiling(101, host_ceiling_bytes=1000, device_ceiling_bytes=100)
    assert error.value.details["host_status"] == "PASS_PROJECTED_BELOW_CEILING"
    assert error.value.details["device_status"] == "FAIL_PROJECTED_ABOVE_DEVICE_CEILING"
    result = enforce_memory_ceiling(100, host_ceiling_bytes=100, device_ceiling_bytes=100)
    assert result["host_status"] == "PASS_PROJECTED_BELOW_CEILING"
    assert result["device_status"] == "PASS_PROJECTED_BELOW_CEILING"


def test_on_disk_seed_manifest_digest_is_reproducible() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "config/cf4_q4_seed_manifest_v1.json").read_text(encoding="utf-8"))
    digest = manifest.pop("manifest_sha256")
    assert digest == canonical_sha256(manifest)
