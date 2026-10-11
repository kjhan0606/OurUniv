from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution_authorized_v1 as authorized


def _write(path: Path, value: dict, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(authorized.canonical_json(value) + b"\n")
    path.chmod(mode)


def _fake_authorization(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    program = authorized.load_program()
    grant = tmp_path / "grant.json"
    release_path = tmp_path / "release.json"
    manifest_path = tmp_path / "manifest.json"
    receipts = tmp_path / "receipts"
    data = tmp_path / "data"
    state = tmp_path / "state"
    cache = tmp_path / "cache"
    for name, value in (
        ("GRANT", grant), ("RELEASE", release_path),
        ("EXTERNAL_MANIFEST", manifest_path), ("RECEIPT_ROOT", receipts),
        ("DATA_ROOT", data), ("STATE_ROOT", state), ("CACHE_ROOT", cache),
    ):
        monkeypatch.setattr(authorized, name, value)

    design = json.loads(authorized.PAIR_DESIGN.read_text())
    payload_contract = design["release_payload_contract"]
    protected = program["lineage"]["protected_file_sha256"]
    release_id, manifest_id, grant_id = "1" * 64, "2" * 64, "3" * 64
    payload = {
        "schema": payload_contract["schema"],
        "status": payload_contract["required_status"],
        "release_id": release_id,
        "authorization_design_commit": "e6ba2c3482855a6c8c16aa8068df83bbfb9c62e8",
        "authorization_design_sha256": authorized.AUTHORIZATION_DESIGN_SHA256,
        "grant_release_manifest_design_commit": "3d0a2d2eed3dd9fbfc0cc88d4a705586c156021f",
        "grant_release_manifest_design_sha256": authorized.PAIR_DESIGN_SHA256,
        "implementation_commit": "7eb25554abec278a3710b99aed90e73c39f37b9b",
        "implementation_result_record_sha256": authorized.IMPLEMENTATION_RECORD_SHA256,
        "program_sha256": protected["config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_program.json"],
        "source_sha256": protected["src/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution.py"],
        "runner_sha256": protected["scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_lageunha.sh"],
        "launcher_sha256": protected["scripts/launch_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_lageunha.sh"],
        "status_script_sha256": protected["scripts/status_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production.sh"],
        "execution_test_sha256": protected["tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_execution.py"],
        "runner_test_sha256": protected["tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_runner.py"],
        "execution_design_sha256": "08d99219b88a232dc809b3a2c945381cbbcda1fac0c7202c1c2681a09be609aa",
        "fixed_science_digest": program["grant_release_manifest_contract"]["fixed_science_digest"],
        "canonical_paths_digest": program["grant_release_manifest_contract"]["canonical_paths_digest"],
        "one_shot": True,
        **payload_contract["future_authorization_matrix"],
    }
    assert set(payload) == set(payload_contract["exact_keys"])
    payload_sha = hashlib.sha256(authorized.canonical_json(payload)).hexdigest()
    manifest_contract = design["external_manifest_contract"]
    manifest = {
        "schema": manifest_contract["schema"],
        "status": manifest_contract["required_status"],
        "manifest_id": manifest_id,
        "release_path": str(release_path),
        "release_id": release_id,
        "release_payload_sha256": payload_sha,
        "authorization_design_sha256": authorized.AUTHORIZATION_DESIGN_SHA256,
        "grant_release_manifest_design_sha256": authorized.PAIR_DESIGN_SHA256,
        "implementation_result_record_sha256": authorized.IMPLEMENTATION_RECORD_SHA256,
        "canonical_paths_digest": program["grant_release_manifest_contract"]["canonical_paths_digest"],
        "one_shot": True,
    }
    assert set(manifest) == set(manifest_contract["exact_keys"])
    _write(manifest_path, manifest, 0o444)
    release_contract = design["external_release_contract"]
    release = {
        "schema": release_contract["schema"],
        "status": release_contract["required_status"],
        "release_id": release_id,
        "payload": payload,
        "payload_sha256": payload_sha,
        "manifest_path": str(manifest_path),
        "manifest_id": manifest_id,
        "manifest_sha256": authorized.sha256_file(manifest_path),
    }
    assert set(release) == set(release_contract["exact_keys"])
    _write(release_path, release, 0o444)
    grant_contract = design["local_grant_contract"]
    grant_value = {
        "schema": grant_contract["schema"],
        "status": grant_contract["required_status"],
        "grant_id": grant_id,
        "one_shot": True,
        "authorization_design_path": str(authorized.AUTHORIZATION_DESIGN.relative_to(authorized.ROOT)),
        "authorization_design_commit": "e6ba2c3482855a6c8c16aa8068df83bbfb9c62e8",
        "authorization_design_sha256": authorized.AUTHORIZATION_DESIGN_SHA256,
        "grant_release_manifest_design_path": str(authorized.PAIR_DESIGN.relative_to(authorized.ROOT)),
        "grant_release_manifest_design_commit": "3d0a2d2eed3dd9fbfc0cc88d4a705586c156021f",
        "grant_release_manifest_design_sha256": authorized.PAIR_DESIGN_SHA256,
        "implementation_commit": "7eb25554abec278a3710b99aed90e73c39f37b9b",
        "implementation_result_record_path": str(authorized.IMPLEMENTATION_RECORD.relative_to(authorized.ROOT)),
        "implementation_result_record_sha256": authorized.IMPLEMENTATION_RECORD_SHA256,
        "implementation_file_sha256_map": protected,
        "fixed_science_digest": program["grant_release_manifest_contract"]["fixed_science_digest"],
        "canonical_paths_digest": program["grant_release_manifest_contract"]["canonical_paths_digest"],
        "release_path": str(release_path), "release_id": release_id,
        "release_payload_sha256": payload_sha,
        "release_sha256": authorized.sha256_file(release_path),
        "manifest_path": str(manifest_path), "manifest_id": manifest_id,
        "manifest_sha256": authorized.sha256_file(manifest_path),
        "receipt_root": str(receipts), "cache_root": str(cache),
        "data_root": str(data), "state_root": str(state),
        "authorization": grant_contract["authorization_exact"],
    }
    assert set(grant_value) == set(grant_contract["exact_keys"])
    _write(grant, grant_value, 0o644)
    monkeypatch.setattr(authorized, "load_program", lambda: program)
    monkeypatch.setattr(authorized, "_validate_grant_git_lineage", lambda _program: "4" * 40)
    result = authorized.validate_authorization(program)
    return program, result, (grant, release_path, manifest_path, receipts, data, state, cache)


def test_program_is_closed_and_public_refuses_before_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    program = authorized.load_program()
    assert not any(program["authorization"].values())
    with pytest.raises(PermissionError):
        authorized.run_authorized_production(authorized.PROGRAM)
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(PermissionError):
        authorized.run_authorized_production(tmp_path / "wrong-program.json")


def test_pair_and_grant_are_exact_and_cross_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    program, result, paths = _fake_authorization(monkeypatch, tmp_path)
    assert result["grant"]["grant_id"] == "3" * 64
    manifest = json.loads(paths[2].read_text())
    manifest["release_id"] = "9" * 64
    paths[2].chmod(0o644)
    _write(paths[2], manifest, 0o444)
    with pytest.raises(PermissionError):
        authorized.validate_authorization(program)


def test_duplicate_json_and_git_lineage_mismatch_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    git_lineage_validator = authorized._validate_grant_git_lineage
    _, _, paths = _fake_authorization(monkeypatch, tmp_path)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"key":1,"key":2}\n')
    with pytest.raises(PermissionError, match="duplicate"):
        authorized._read_json(duplicate, "duplicate", canonical=True)

    head = "a" * 40
    grant_text = paths[0].read_text().strip()

    def fake_git(*args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return head
        if args == ("rev-parse", "@{upstream}"):
            return "b" * 40
        if args[:2] == ("ls-remote", "origin"):
            return f"{head}\trefs/heads/{authorized.BRANCH}"
        if args[0] == "diff-tree":
            return f"A\t{paths[0]}"
        if args[0] == "show":
            return grant_text
        return ""

    monkeypatch.setattr(authorized, "_git", fake_git)
    with pytest.raises(PermissionError, match="synchronized"):
        git_lineage_validator(authorized.load_program())


def test_wrong_wrapper_parent_chain_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    git_lineage_validator = authorized._validate_grant_git_lineage
    program, _, _ = _fake_authorization(monkeypatch, tmp_path)
    head = "a" * 40
    values = {
        ("rev-parse", "HEAD"): head,
        ("rev-parse", "HEAD^"): "b" * 40,
        ("rev-parse", "HEAD^^"): "c" * 40,
        ("rev-parse", "HEAD^^^"): "d" * 40,
        ("rev-parse", "@{upstream}"): head,
    }

    def fake_git(*args: str) -> str:
        if args[:2] == ("ls-remote", "origin"):
            return f"{head}\trefs/heads/{authorized.BRANCH}"
        return values.get(args, "")

    monkeypatch.setattr(authorized, "_git", fake_git)
    with pytest.raises(PermissionError, match="rooted"):
        git_lineage_validator(program)


def test_exact_wrapper_result_and_three_commit_chain_can_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    git_lineage_validator = authorized._validate_grant_git_lineage
    program, _, paths = _fake_authorization(monkeypatch, tmp_path)
    head, result_commit, implementation_commit = "a" * 40, "b" * 40, "c" * 40
    rows = []
    blob_by_path = {}
    for index, relative in enumerate(authorized.IMPLEMENTATION_FILES):
        raw = (authorized.ROOT / relative).read_bytes()
        blob = f"{index + 1:040x}"
        blob_by_path[relative] = blob
        rows.append({
            "path": relative, "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob_oid": blob,
            "mode": "100755" if relative.startswith("scripts/") else "100644",
        })
    record = {
        "schema": program["lineage"]["future_wrapper_implementation_result_record"]["schema"],
        "status": program["lineage"]["future_wrapper_implementation_result_record"]["status"],
        "implementation_commit": implementation_commit,
        "implementation_parent_commit": authorized.WRAPPER_DESIGN_COMMIT,
        "wrapper_design_commit": authorized.WRAPPER_DESIGN_COMMIT,
        "implementation_files": rows,
        "independent_audits": program["lineage"]["future_wrapper_implementation_result_record"]["independent_audits_exact"],
        "authorization": program["lineage"]["future_wrapper_implementation_result_record"]["authorization_exact"],
    }
    record_path = tmp_path / "wrapper-result.json"
    _write(record_path, record, 0o644)
    monkeypatch.setattr(authorized, "WRAPPER_RESULT_RECORD", record_path)

    def fake_git(*args: str) -> str:
        mapping = {
            ("rev-parse", "HEAD"): head,
            ("rev-parse", "HEAD^"): result_commit,
            ("rev-parse", "HEAD^^"): implementation_commit,
            ("rev-parse", "HEAD^^^"): authorized.WRAPPER_DESIGN_COMMIT,
            ("rev-parse", "@{upstream}"): head,
        }
        if args in mapping:
            return mapping[args]
        if args[:2] == ("ls-remote", "origin"):
            return f"{head}\trefs/heads/{authorized.BRANCH}"
        if args[0] == "diff-tree" and args[-1] == head:
            return f"A\t{authorized.GRANT_RELATIVE}"
        if args[0] == "diff-tree" and args[-1] == result_commit:
            return f"A\t{authorized.WRAPPER_RESULT_RECORD_RELATIVE}"
        if args[0] == "diff-tree" and args[-1] == implementation_commit:
            return "\n".join(f"A\t{path}" for path in authorized.IMPLEMENTATION_FILES)
        if args[0] == "ls-tree":
            relative = args[-1]
            if args[1] == result_commit:
                raw = record_path.read_bytes()
                blob = hashlib.sha1(
                    f"blob {len(raw)}\0".encode("ascii") + raw,
                ).hexdigest()
                return f"100644 blob {blob}\t{authorized.WRAPPER_RESULT_RECORD_RELATIVE}"
            mode = "100755" if relative.startswith("scripts/") else "100644"
            return f"{mode} blob {blob_by_path[relative]}\t{relative}"
        if args[0] == "status":
            return ""
        raise AssertionError(args)

    def fake_git_bytes(*args: str) -> bytes:
        spec = args[1]
        if spec.startswith(head + ":"):
            return paths[0].read_bytes()
        if spec.startswith(result_commit + ":"):
            return record_path.read_bytes()
        commit, relative = spec.split(":", 1)
        assert commit == implementation_commit
        return (authorized.ROOT / relative).read_bytes()

    monkeypatch.setattr(authorized, "_git", fake_git)
    monkeypatch.setattr(authorized, "_git_bytes", fake_git_bytes)
    assert git_lineage_validator(program) == head


def test_receipt_postlink_snapshot_and_inode_attack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _, auth, paths = _fake_authorization(monkeypatch, tmp_path)
    monkeypatch.setattr(authorized, "validate_authorization", lambda _program: auth)
    receipt, snapshot, snapshot_sha = authorized.create_receipt(auth)
    assert receipt == paths[3] / ("3" * 64) / "production"
    assert {item.name for item in receipt.iterdir()} == {"release.anchor", "snapshot.json", "RUNNING"}
    assert snapshot["release_nlink"] == 2
    assert oct((receipt / "snapshot.json").stat().st_mode & 0o777) == "0o444"
    authorized.revalidate_receipt(receipt, snapshot_sha)
    original = paths[1].read_bytes()
    paths[1].unlink()
    paths[1].write_bytes(original)
    paths[1].chmod(0o444)
    with pytest.raises(PermissionError):
        authorized.revalidate_receipt(receipt, snapshot_sha)


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ("after_receipt_mkdir", {"FAILED"}),
        ("after_release_anchor_link", {"FAILED", "release.anchor"}),
        ("after_snapshot_seal", {"FAILED", "release.anchor", "snapshot.json"}),
        ("after_RUNNING_seal", {"FAILED", "release.anchor", "snapshot.json"}),
    ],
)
def test_receipt_bootstrap_interruptions_are_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint: str, expected: set[str],
):
    _, auth, paths = _fake_authorization(monkeypatch, tmp_path)
    monkeypatch.setattr(authorized, "validate_authorization", lambda _program: auth)

    def stop(name: str) -> None:
        if name == checkpoint:
            raise InterruptedError(name)

    monkeypatch.setattr(authorized, "_receipt_checkpoint", stop)
    with pytest.raises(InterruptedError):
        authorized.create_receipt(auth)
    receipt = paths[3] / ("3" * 64) / "production"
    assert {item.name for item in receipt.iterdir()} == expected
    failure = json.loads((receipt / "FAILED").read_text())
    assert failure["snapshot_sha256"] is (None if checkpoint in {
        "after_receipt_mkdir", "after_release_anchor_link",
    } else failure["snapshot_sha256"])
    assert not (receipt / "RUNNING").exists()


@pytest.mark.parametrize("science_status", sorted(authorized.SCIENTIFIC_STATUSES))
def test_fake_science_complete_binds_state_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, science_status: str,
):
    _, auth, paths = _fake_authorization(monkeypatch, tmp_path)
    monkeypatch.setattr(authorized, "validate_authorization", lambda _program: auth)
    monkeypatch.setattr(authorized.os, "uname", lambda: SimpleNamespace(nodename="LagEunha.cluster"))
    monkeypatch.setattr(authorized, "_require_resources", lambda: None)
    monkeypatch.setattr(authorized, "_require_runtime_environment", lambda: None)
    monkeypatch.setattr(authorized.execution, "load_canonical_program", lambda **_kw: {})
    monkeypatch.setattr(authorized.capability, "load_frozen_contract", lambda: object())

    def core(_program, _contract, data: Path, _cache: Path):
        _write(data / "manifest.json", {"schema": "fake"}, 0o444)
        _write(data / "result.json", {"status": science_status}, 0o444)
        return {"status": science_status}

    monkeypatch.setattr(authorized.execution, "_execute_reserved_canonical_private", core)
    monkeypatch.setattr(
        authorized, "read_only_postcheck",
        lambda _data: {"status": science_status, "valid_scientific_complete": True},
    )
    returned = authorized.run_authorized_production(authorized.PROGRAM)
    receipt = paths[3] / ("3" * 64) / "production"
    assert returned["status"] == science_status
    assert (receipt / "COMPLETE").is_file() and not (receipt / "RUNNING").exists()
    assert (paths[5] / "COMPLETE").is_file() and not (paths[5] / "RUNNING").exists()
    assert json.loads((receipt / "COMPLETE").read_text())["result_manifest_sha256"] \
        == json.loads((paths[5] / "COMPLETE").read_text())["result_manifest_sha256"]
    assert authorized._read_only_complete_status()["science_status"] == science_status
    paths[6].chmod(0o700)
    with pytest.raises(PermissionError, match="terminal state type or mode"):
        authorized._read_only_complete_status()
    paths[6].chmod(0o555)


def test_fake_science_exception_routes_both_lifecycles_to_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _, auth, paths = _fake_authorization(monkeypatch, tmp_path)
    monkeypatch.setattr(authorized, "validate_authorization", lambda _program: auth)
    monkeypatch.setattr(authorized.os, "uname", lambda: SimpleNamespace(nodename="LagEunha"))
    monkeypatch.setattr(authorized, "_require_resources", lambda: None)
    monkeypatch.setattr(authorized, "_require_runtime_environment", lambda: None)
    monkeypatch.setattr(authorized.execution, "load_canonical_program", lambda **_kw: {})
    monkeypatch.setattr(authorized.capability, "load_frozen_contract", lambda: object())
    monkeypatch.setattr(
        authorized.execution, "_execute_reserved_canonical_private",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("fake invalid execution")),
    )
    with pytest.raises(RuntimeError, match="fake invalid"):
        authorized.run_authorized_production(authorized.PROGRAM)
    receipt = paths[3] / ("3" * 64) / "production"
    assert (receipt / "FAILED").is_file() and not (receipt / "RUNNING").exists()
    assert (paths[5] / "FAILED").is_file() and not (paths[5] / "RUNNING").exists()


def test_interruption_after_state_reservation_has_no_orphan_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _, auth, paths = _fake_authorization(monkeypatch, tmp_path)
    monkeypatch.setattr(authorized, "validate_authorization", lambda _program: auth)
    monkeypatch.setattr(authorized.os, "uname", lambda: SimpleNamespace(nodename="LAGEUNHA"))
    monkeypatch.setattr(authorized, "_require_resources", lambda: None)
    monkeypatch.setattr(authorized, "_require_runtime_environment", lambda: None)

    def stop(name: str) -> None:
        if name == "after_state_reservation":
            raise InterruptedError(name)

    monkeypatch.setattr(authorized, "_receipt_checkpoint", stop)
    with pytest.raises(InterruptedError):
        authorized.run_authorized_production(authorized.PROGRAM)
    receipt = paths[3] / ("3" * 64) / "production"
    assert {item.name for item in receipt.iterdir()} == {
        "FAILED", "release.anchor", "snapshot.json",
    }
    assert (paths[5] / "FAILED").is_file()


@pytest.mark.parametrize("exit_code", [124, 137])
def test_supervisor_closes_killed_running_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int,
):
    _, auth, paths = _fake_authorization(monkeypatch, tmp_path)
    monkeypatch.setattr(authorized, "validate_authorization", lambda _program: auth)
    receipt, _, _ = authorized.create_receipt(auth)
    for path in (paths[4], paths[5], paths[6]):
        path.mkdir(mode=0o700)
    authorized._exclusive_json(paths[5] / "RUNNING", {
        "schema": "ouruniv-cf4-v6-open-shared-schedule-production-state-marker-v1",
        "status": "running_authorized_shared_schedule_production",
        "grant_id": "3" * 64,
        "snapshot_sha256": authorized.sha256_file(receipt / "snapshot.json"),
    })
    authorized._supervisor_force_failed(
        "3" * 64, f"supervisor_timeout_or_child_exit_{exit_code}",
    )
    assert (receipt / "FAILED").is_file() and not (receipt / "RUNNING").exists()
    assert (paths[5] / "FAILED").is_file() and not (paths[5] / "RUNNING").exists()
    assert authorized._read_only_failed_status()["status"] == "failed"


@pytest.mark.parametrize("failure_point", ["unlink", "complete_write", "cache_chmod"])
def test_partial_complete_failure_converges_to_both_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str,
):
    _, auth, paths = _fake_authorization(monkeypatch, tmp_path)
    monkeypatch.setattr(authorized, "validate_authorization", lambda _program: auth)
    monkeypatch.setattr(authorized.os, "uname", lambda: SimpleNamespace(nodename="LagEunha"))
    monkeypatch.setattr(authorized, "_require_resources", lambda: None)
    monkeypatch.setattr(authorized, "_require_runtime_environment", lambda: None)
    monkeypatch.setattr(authorized.execution, "load_canonical_program", lambda **_kw: {})
    monkeypatch.setattr(authorized.capability, "load_frozen_contract", lambda: object())

    def core(_program, _contract, data: Path, _cache: Path):
        _write(data / "manifest.json", {"schema": "fake"}, 0o444)
        _write(data / "result.json", {"status": "complete_pass_production_smc"}, 0o444)
        return {"status": "complete_pass_production_smc"}

    monkeypatch.setattr(authorized.execution, "_execute_reserved_canonical_private", core)
    monkeypatch.setattr(
        authorized, "read_only_postcheck",
        lambda _data: {"status": "complete_pass_production_smc", "valid_scientific_complete": True},
    )
    fired = False
    if failure_point == "unlink":
        original = Path.unlink

        def fail_once(path: Path, *args, **kwargs):
            nonlocal fired
            if not fired and path.name == "RUNNING" and path.parent == paths[5]:
                fired = True
                raise OSError("injected unlink failure")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", fail_once)
    elif failure_point == "complete_write":
        original_write = authorized._exclusive_json

        def fail_write(path: Path, value):
            nonlocal fired
            if not fired and path.name == "COMPLETE":
                fired = True
                raise OSError("injected COMPLETE write failure")
            return original_write(path, value)

        monkeypatch.setattr(authorized, "_exclusive_json", fail_write)
    else:
        original_chmod = authorized.os.chmod

        def fail_chmod(path, mode, *args, **kwargs):
            nonlocal fired
            if not fired and Path(path) == paths[6] and mode == 0o555:
                fired = True
                raise OSError("injected cache chmod failure")
            return original_chmod(path, mode, *args, **kwargs)

        monkeypatch.setattr(authorized.os, "chmod", fail_chmod)
    with pytest.raises(OSError, match="injected"):
        authorized.run_authorized_production(authorized.PROGRAM)
    receipt = paths[3] / ("3" * 64) / "production"
    assert (receipt / "FAILED").is_file() and not (receipt / "COMPLETE").exists()
    assert (paths[5] / "FAILED").is_file() and not (paths[5] / "COMPLETE").exists()


def test_dangling_runtime_root_is_not_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _, auth, paths = _fake_authorization(monkeypatch, tmp_path)
    monkeypatch.setattr(authorized, "validate_authorization", lambda _program: auth)
    monkeypatch.setattr(authorized.os, "uname", lambda: SimpleNamespace(nodename="LagEunha"))
    monkeypatch.setattr(authorized, "_require_resources", lambda: None)
    monkeypatch.setattr(authorized, "_require_runtime_environment", lambda: None)
    paths[4].symlink_to(tmp_path / "missing-target")
    with pytest.raises(PermissionError, match="not absent"):
        authorized.run_authorized_production(authorized.PROGRAM)
    assert not paths[3].exists()


def test_failed_status_rejects_arbitrary_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _, auth, paths = _fake_authorization(monkeypatch, tmp_path)
    monkeypatch.setattr(authorized, "validate_authorization", lambda _program: auth)
    receipt, _, snapshot_sha = authorized.create_receipt(auth)
    for path in (paths[4], paths[5], paths[6]):
        path.mkdir(mode=0o700)
    authorized._receipt_failed(receipt, auth, "fake_failure", snapshot_sha)
    authorized._state_failed("fake_failure", auth, snapshot_sha)
    paths[5].chmod(0o700)
    marker = paths[5] / "FAILED"
    marker.chmod(0o600)
    marker.write_text('{"status":"failed"}\n')
    marker.chmod(0o444)
    paths[5].chmod(0o555)
    with pytest.raises(PermissionError):
        authorized._read_only_failed_status()


def test_failed_status_rejects_forensic_anchor_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    _, auth, paths = _fake_authorization(monkeypatch, tmp_path)
    monkeypatch.setattr(authorized, "validate_authorization", lambda _program: auth)
    receipt, _, snapshot_sha = authorized.create_receipt(auth)
    authorized._receipt_failed(receipt, auth, "fake_failure", snapshot_sha)
    receipt.chmod(0o700)
    anchor = receipt / "release.anchor"
    anchor.unlink()
    anchor.symlink_to(paths[1])
    receipt.chmod(0o555)
    with pytest.raises(PermissionError, match="anchor"):
        authorized._read_only_failed_status(allow_state_absent=True)


@pytest.mark.parametrize(
    ("failed_name", "expected_present"),
    [
        ("state", set()),
        ("cache", {"state"}),
        ("data", {"state", "cache"}),
    ],
)
def test_runtime_reservation_order_and_each_failure_is_forensic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    failed_name: str, expected_present: set[str],
):
    _, auth, paths = _fake_authorization(monkeypatch, tmp_path)
    monkeypatch.setattr(authorized, "validate_authorization", lambda _program: auth)
    monkeypatch.setattr(authorized.os, "uname", lambda: SimpleNamespace(nodename="LagEunha"))
    monkeypatch.setattr(authorized, "_require_resources", lambda: None)
    monkeypatch.setattr(authorized, "_require_runtime_environment", lambda: None)
    targets = {"data": paths[4], "state": paths[5], "cache": paths[6]}
    original_mkdir = Path.mkdir

    def fail_target(path: Path, *args, **kwargs):
        if path == targets[failed_name]:
            raise OSError(f"injected {failed_name} mkdir failure")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_target)
    with pytest.raises(PermissionError, match="reservation failed"):
        authorized.run_authorized_production(authorized.PROGRAM)
    receipt = paths[3] / ("3" * 64) / "production"
    assert (receipt / "FAILED").is_file() and not (receipt / "RUNNING").exists()
    assert {name for name, path in targets.items() if path.exists()} == expected_present
    if "state" in expected_present:
        assert (paths[5] / "FAILED").is_file()
        assert authorized._read_only_failed_status()["status"] == "failed"
    else:
        assert authorized._read_only_failed_status(allow_state_absent=True)["status"] == "failed"
