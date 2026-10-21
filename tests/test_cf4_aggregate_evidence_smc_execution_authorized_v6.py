import json
from pathlib import Path

import pytest

import cf4_aggregate_evidence_smc_execution_authorized_v6 as authorized


def test_v6_program_hard_pins_shared_design_source_and_frozen_science():
    program = authorized.load_canonical_authorization_program()
    assert program["hard_pins"] == authorized._expected_hard_pins()
    assert program["fixed_science"] == authorized._expected_fixed_science()
    assert program["authorization"] == authorized.AUTHORIZATION
    assert program["future_grant_interface"]["current_grant_present"] is False
    assert not authorized.CANONICAL_GRANT.exists()


def test_v6_public_gate_rejects_noncanonical_and_absent_grant_before_execution(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(authorized, "require_execution_authorization", lambda program: calls.append(program) or (_ for _ in ()).throw(PermissionError("grant absent")))
    with pytest.raises(PermissionError, match="canonical"):
        authorized.run_authorized_v6(tmp_path / "wrong.json")
    assert calls == []
    with pytest.raises(PermissionError, match="grant absent"):
        authorized.run_authorized_v6(authorized.CANONICAL_PROGRAM)
    assert len(calls) == 1


def test_v6_rejects_v5_paths_and_read_only_postcheck_requires_allowed_status(tmp_path, monkeypatch):
    with pytest.raises(PermissionError, match="v6-only"):
        authorized._v6_path("/gpfs/kjhan/CF4/recon/linear_cr/aggregate_evidence_smc_v5", authorized.DATA_ROOT, "data")
    monkeypatch.setattr(authorized, "DATA_ROOT", tmp_path)
    result, manifest = tmp_path / "result.json", tmp_path / "manifest.json"
    result.write_text("result\n")
    manifest.write_text("manifest\n")
    with pytest.raises(PermissionError, match="read-only"):
        authorized.read_only_v6_postcheck(tmp_path)
    manifest.chmod(0o444)
    monkeypatch.setattr(authorized.base_execution, "validate_published_bundle", lambda _: {
        "status": "complete_scientific_fail_production_smc", "valid_scientific_complete": True,
    })
    assert authorized.read_only_v6_postcheck(tmp_path)["status"] == "complete_scientific_fail_production_smc"
    monkeypatch.setattr(authorized.base_execution, "validate_published_bundle", lambda _: {
        "status": "running", "valid_scientific_complete": True,
    })
    with pytest.raises(RuntimeError, match="allowed scientific"):
        authorized.read_only_v6_postcheck(tmp_path)


def test_v6_program_contains_only_v6_storage_paths_and_unreachable_shared_contract():
    program = json.loads(authorized.CANONICAL_PROGRAM.read_text())
    assert all("v5" not in str(value).lower() for value in program["storage"].values())
    contract = program["future_shared_pilot_and_production_contract"]
    assert contract["pilot_cache_posterior_and_scientific_result_reuse"] is False
    assert contract["all_four_masters_including_pilot_seed_rerun_under_immutable_schedule"] is True
    assert contract["unreachable_without_future_grant_release_and_manifest"] is True
