import json
import os
from pathlib import Path

import pytest

import cf4_aggregate_evidence_smc_execution_authorized_v5 as authorized


def _write(path: Path, value, *, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    path.chmod(mode)


def _install_fixture(tmp_path, monkeypatch):
    original_root = authorized.ROOT
    root = tmp_path / "repo"
    config = root / "config"
    source_base = original_root / "config/cf4_aggregate_evidence_smc_production_program.json"
    (config / source_base.name).parent.mkdir(parents=True)
    (config / source_base.name).write_bytes(source_base.read_bytes())
    runner = root / "scripts/run_cf4_aggregate_evidence_smc_authorized_v5_lageunha.sh"
    runner.parent.mkdir(parents=True)
    runner.write_bytes((original_root / "scripts/run_cf4_aggregate_evidence_smc_authorized_v5_lageunha.sh").read_bytes())

    external = tmp_path / "external"
    release = external / "release.json"
    manifest = external / "manifest.json"
    data, state, receipts = tmp_path / "data", tmp_path / "state", tmp_path / "receipts"
    grant = config / "cf4_aggregate_evidence_smc_execution_grant_v5.json"
    program_path = config / "cf4_aggregate_evidence_smc_execution_authorization_program_v5.json"
    program = json.loads((original_root / "config/cf4_aggregate_evidence_smc_execution_authorization_program_v5.json").read_text())
    program["storage"] = {
        "data_directory": str(data), "state_directory": str(state),
        "receipts_directory": str(receipts), "exclusive_reservation": True,
        "restart_or_checkpoint_import": False,
    }
    program["external_pre_execution_release"] = {
        "canonical_path": str(release), "manifest_path": str(manifest),
        "current_release_present": False, "current_manifest_present": False,
        "runtime_path_override_allowed": False,
    }
    _write(program_path, program)
    payload = {"audited_commit": "a" * 40, "scope": "v5-one-shot"}
    payload_sha = authorized._sha256(payload)
    manifest_value = {
        "schema": "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-manifest-v5",
        "status": "complete_paired_external_manifest", "manifest_id": "2" * 64,
        "release_path": str(release), "release_id": "1" * 64,
        "release_payload_sha256": payload_sha,
    }
    _write(manifest, manifest_value)
    release_value = {
        "schema": "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-release-v5",
        "status": "complete_pass_external_postcommit_lineage_audit", "verdict": "LINEAGE GO",
        "release_id": "1" * 64, "payload": payload, "payload_sha256": payload_sha,
        "manifest_path": str(manifest), "manifest_sha256": authorized.sha256_file(manifest),
        "manifest_id": "2" * 64,
    }
    _write(release, release_value, mode=0o444)
    grant_value = {
        "schema": "ouruniv-cf4-aggregate-evidence-smc-execution-grant-v5",
        "status": "sealed_one_shot_execution_authorization", "one_shot": True,
        "authorization_program_sha256": authorized.sha256_file(program_path), "grant_id": "3" * 64,
        "authorization": {key: key in {
            "production_SMC_execution_authorized", "oracle_cache_population_authorized"
        } for key in authorized.AUTHORIZATION_KEYS},
        "data_directory": str(data), "state_directory": str(state), "receipts_directory": str(receipts),
        "external_release": {
            "path": str(release), "release_id": "1" * 64, "payload_sha256": payload_sha,
            "manifest_path": str(manifest), "manifest_sha256": authorized.sha256_file(manifest),
            "manifest_id": "2" * 64,
        },
        "precommit_audit_verdict": "EXECUTION GO",
    }
    _write(grant, grant_value)
    monkeypatch.setattr(authorized, "ROOT", root)
    monkeypatch.setattr(authorized, "CANONICAL_PROGRAM", program_path)
    monkeypatch.setattr(authorized, "CANONICAL_GRANT", grant)
    monkeypatch.setattr(authorized, "EXTERNAL_RELEASE", release)
    monkeypatch.setattr(authorized, "EXTERNAL_MANIFEST", manifest)
    monkeypatch.setattr(authorized, "DATA_DIRECTORY", data)
    monkeypatch.setattr(authorized, "STATE_DIRECTORY", state)
    monkeypatch.setattr(authorized, "RECEIPTS_DIRECTORY", receipts)
    receipts.mkdir()
    return {"program": program_path, "grant": grant, "release": release, "manifest": manifest, "receipts": receipts}


def test_valid_flow_binds_canonical_snapshot_and_hard_link_receipt(tmp_path, monkeypatch):
    fixture = _install_fixture(tmp_path, monkeypatch)
    program = authorized.load_canonical_authorization_program()
    receipt = fixture["receipts"] / "one-shot-receipt"
    snapshot, snapshot_sha = authorized.create_preflight_receipt(receipt, program)
    assert snapshot["policy"] == 5
    assert snapshot["release"]["stat"]["nlink"] >= 2
    assert snapshot["release"]["stat"] == {
        "dev": fixture["release"].stat().st_dev, "ino": fixture["release"].stat().st_ino,
        "size": fixture["release"].stat().st_size, "nlink": fixture["release"].stat().st_nlink,
    }
    assert authorized.revalidate_preflight_receipt(receipt, snapshot_sha, program) == snapshot


def test_grant_release_manifest_and_snapshot_mismatches_fail_closed(tmp_path, monkeypatch):
    fixture = _install_fixture(tmp_path, monkeypatch)
    program = authorized.load_canonical_authorization_program()
    receipt = fixture["receipts"] / "one-shot-receipt"
    _, snapshot_sha = authorized.create_preflight_receipt(receipt, program)

    grant = json.loads(fixture["grant"].read_text())
    grant["external_release"]["payload_sha256"] = "0" * 64
    _write(fixture["grant"], grant)
    with pytest.raises(PermissionError, match="grant/release/manifest"):
        authorized.revalidate_preflight_receipt(receipt, snapshot_sha, program)

    fixture = _install_fixture(tmp_path / "release", monkeypatch)
    program = authorized.load_canonical_authorization_program()
    release = json.loads(fixture["release"].read_text())
    release["release_id"] = "5" * 64
    fixture["release"].chmod(0o644)
    _write(fixture["release"], release, mode=0o444)
    with pytest.raises(PermissionError, match="pairing"):
        authorized.require_execution_authorization(program)

    fixture = _install_fixture(tmp_path / "pair", monkeypatch)
    program = authorized.load_canonical_authorization_program()
    manifest = json.loads(fixture["manifest"].read_text())
    manifest["manifest_id"] = "4" * 64
    _write(fixture["manifest"], manifest)
    with pytest.raises(PermissionError, match="pairing"):
        authorized.require_execution_authorization(program)

    fixture = _install_fixture(tmp_path / "snapshot", monkeypatch)
    program = authorized.load_canonical_authorization_program()
    receipt = fixture["receipts"] / "one-shot-receipt"
    _, snapshot_sha = authorized.create_preflight_receipt(receipt, program)
    snapshot = json.loads((receipt / "preflight-snapshot.json").read_text())
    snapshot["policy"] = 4
    _write(receipt / "preflight-snapshot.json", snapshot)
    with pytest.raises(PermissionError, match="snapshot hash"):
        authorized.revalidate_preflight_receipt(receipt, snapshot_sha, program)


def test_delete_recreate_identical_release_is_detected_by_anchor_inode(tmp_path, monkeypatch):
    fixture = _install_fixture(tmp_path, monkeypatch)
    program = authorized.load_canonical_authorization_program()
    receipt = fixture["receipts"] / "one-shot-receipt"
    _, snapshot_sha = authorized.create_preflight_receipt(receipt, program)
    original = fixture["release"].read_bytes()
    # The parent directory permits replacement even though the sealed release is read-only.
    fixture["release"].unlink()
    fixture["release"].write_bytes(original)
    fixture["release"].chmod(0o444)
    assert authorized.sha256_file(fixture["release"]) == authorized.sha256_file(receipt / "release.anchor")
    assert fixture["release"].stat().st_ino != (receipt / "release.anchor").stat().st_ino
    with pytest.raises(PermissionError, match="hard-link anchor"):
        authorized.revalidate_preflight_receipt(receipt, snapshot_sha, program)


def test_receipt_is_exclusive(tmp_path, monkeypatch):
    fixture = _install_fixture(tmp_path, monkeypatch)
    program = authorized.load_canonical_authorization_program()
    receipt = fixture["receipts"] / "one-shot-receipt"
    authorized.create_preflight_receipt(receipt, program)
    with pytest.raises(PermissionError, match="receipt already exists"):
        authorized.create_preflight_receipt(receipt, program)


@pytest.mark.parametrize(
    ("checked", "expected_status", "expected_failure"),
    [
        ({"status": "complete_pass_production_smc", "outcome_kind": "terminal",
          "failure_class": None, "valid_scientific_complete": True},
         "complete_pass_production_smc", "none"),
        ({"status": "complete_scientific_fail_production_smc", "outcome_kind": "architecture_stop",
          "failure_class": "SMC_temperature_stagnation", "valid_scientific_complete": True},
         "complete_scientific_fail_production_smc", "SMC_temperature_stagnation"),
    ],
)
def test_read_only_postcheck_allows_only_valid_science_pass_or_fail(
    tmp_path, monkeypatch, checked, expected_status, expected_failure
):
    (tmp_path / "result.json").write_text("result\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("manifest\n")
    manifest.chmod(0o444)
    monkeypatch.setattr(authorized.base_execution, "validate_published_bundle", lambda _: checked)
    value = authorized.read_only_science_postcheck(tmp_path)
    assert value["science_status"] == expected_status
    assert value["failure_class"] == expected_failure
    assert value["result_sha256"] == authorized.sha256_file(tmp_path / "result.json")
    assert value["manifest_sha256"] == authorized.sha256_file(manifest)


@pytest.mark.parametrize("checked", [
    {"status": "running", "outcome_kind": "terminal", "failure_class": None,
     "valid_scientific_complete": True},
    {"status": "complete_pass_production_smc", "outcome_kind": "terminal", "failure_class": None,
     "valid_scientific_complete": False},
])
def test_read_only_postcheck_rejects_missing_writable_or_invalid_bundle(tmp_path, monkeypatch, checked):
    with pytest.raises(RuntimeError, match="absent or empty"):
        authorized.read_only_science_postcheck(tmp_path)
    (tmp_path / "result.json").write_text("result\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("manifest\n")
    monkeypatch.setattr(authorized.base_execution, "validate_published_bundle", lambda _: checked)
    with pytest.raises(PermissionError, match="read-only"):
        authorized.read_only_science_postcheck(tmp_path)
    manifest.chmod(0o444)
    with pytest.raises(RuntimeError, match="not a valid scientific completion"):
        authorized.read_only_science_postcheck(tmp_path)
