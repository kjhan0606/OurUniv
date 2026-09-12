from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from types import MappingProxyType

import numpy as np
import pytest

import cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v1 as subject


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"


def write_json(path: Path, value, mode: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))
    path.chmod(mode)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def program_value():
    paths = {
        "external_manifest": str(subject.MANIFEST), "external_release": str(subject.RELEASE),
        "local_grant": str(subject.GRANT.relative_to(subject.ROOT)),
        "local_program": str(subject.PROGRAM.relative_to(subject.ROOT)),
        "receipt_root": str(subject.RECEIPT_ROOT),
    }
    names = [
        "src/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v1.py",
        "scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v1.sbatch",
        "scripts/status_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v1.sh",
        "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v1.py",
        "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v1_runner.py",
    ]
    return {
        "schema": "ouruniv-cf4-v6-open-parent-key-overlap-canonical-read-only-analysis-authorized-program-v1",
        "status": "frozen_program_execution_false_until_sealed_pair_grant_and_audits",
        "date": "2026-08-21", "purpose": "test frozen program",
        "lineage": {
            "branch": "agent/freeze-zoom-pipeline", "authorization_design_commit": "36a872c0be856774b30d9e3c7bb7c5e11e4a11e6",
            "authorization_design_sha256": subject.AUTHORIZATION_DESIGN_SHA,
            "grant_release_manifest_design_commit": "85ebecad77794c83b54e3d7a6741e2727e6db4ee",
            "grant_release_manifest_design_sha256": subject.PAIR_DESIGN_SHA,
            "authorized_wrapper_design_commit": subject.WRAPPER_DESIGN_COMMIT,
            "authorized_wrapper_design_sha256": subject.WRAPPER_DESIGN_SHA,
            "loader_implementation_commit": "01251cef43bebae77841459bc9e86b58f1efc91b",
            "loader_implementation_result_record_commit": "411e20c88a63b70759a0ad082746c8975cfc0a4c",
            "loader_implementation_result_record_sha256": subject.LOADER_RESULT_RECORD_SHA,
        },
        "canonical_paths": paths,
        "resource_contract": {"cpus_per_task": 4, "login_host": "grammar", "node": "grammar-debug", "nodes": 1, "ntasks": 1, "partition": "debug", "pre_registered_maximum_expected_RSS_GiB": 6.5, "requested_memory_GiB": 8, "requested_memory_margin_over_expected_percent": 23.076923076923077, "requeue": False, "submission_mechanism": "Slurm_only", "time_limit": "01:00:00"},
        "implementation_files": [{"path": name, "sha256": hashlib.sha256(name.encode()).hexdigest(), "mode": "0755" if name.startswith("scripts/") else "0644"} for name in names],
        "authorization": dict(subject.FALSE_AUTHORIZATION),
        "next": {"canonical_analysis_execution_authorized": False, "canonical_artifact_read_authorized": False, "external_pair_and_committed_grant_audits_required": True, "immediate": "program_and_wrapper_implementation_precommit_audit_only", "receipt_creation_authorized": False, "slurm_submission_authorized": False},
    }


def install_fake_bundle(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    external = tmp_path / "external"
    receipts = tmp_path / "receipts"
    program = root / "config/program.json"
    grant_path = root / "config/grant.json"
    release_path = external / "release.json"
    manifest_path = external / "manifest.json"
    for name, value in (("ROOT", root), ("PROGRAM", program), ("GRANT", grant_path), ("RELEASE", release_path), ("MANIFEST", manifest_path), ("RECEIPT_ROOT", receipts)):
        monkeypatch.setattr(subject, name, value)
    commands = (
        ("/usr/bin/git", "rev-parse", "HEAD"), ("/usr/bin/git", "rev-parse", "@{upstream}"),
        ("/usr/bin/git", "status", "--porcelain=v1", "-z", "--untracked-files=all", "--", ".", ":(exclude)scripts/tripwire/**"),
        ("/usr/bin/git", "rev-parse", "HEAD^"), ("/usr/bin/git", "rev-list", "--parents", "-n", "1", "HEAD"),
        ("/usr/bin/git", "diff-tree", "--no-commit-id", "--name-status", "-r", "--no-renames", "HEAD^", "HEAD"),
        ("/usr/bin/git", "ls-tree", "HEAD", "--", "config/grant.json"),
    )
    monkeypatch.setattr(subject, "GIT_COMMANDS", commands)
    monkeypatch.setattr(subject, "_verify_fixed_local_pins", lambda: None)
    monkeypatch.setattr(subject, "_verify_program_implementation_files", lambda program: None)
    monkeypatch.setattr(subject, "_derive_runtime_contract", lambda head: {"git_subprocess_contract": {"expected_HEAD_and_tracking": head}})
    value = program_value()
    write_json(program, value, 0o644)
    program_sha = sha(program)
    release_id = hashlib.sha256(("ouruniv-parent-key-overlap-release-v1\0" + "85ebecad77794c83b54e3d7a6741e2727e6db4ee" + "\0" + subject.AUTHORIZATION_DESIGN_SHA + "\0" + program_sha).encode()).hexdigest()
    manifest_id = hashlib.sha256(("ouruniv-parent-key-overlap-manifest-v1\0" + "85ebecad77794c83b54e3d7a6741e2727e6db4ee" + "\0" + release_id).encode()).hexdigest()
    parent = "3" * 40
    digest = {"artifact_contract_digest": "1cab22081d83abc5b09c8dfbab81e37aeed789c09c0004344808c67986c95ff5", "execution_contract_digest": "2511effaa8860ea9af22aaf2084a91eb2b0a0b5028836f1b054e1d4b166cc5ea", "canonical_paths_digest": "e5acd198245a6eb7f9318367f9a8a72d4f3b193bf3380b008ec95cd7298c5bec", "resource_contract_digest": "faa8a358d8aeeb4caed9e6b19b3a979227fd8b1b1e938f6aaa099ef2ecdd9abc"}
    payload = {
        "schema": "ouruniv-cf4-v6-open-parent-key-overlap-canonical-read-only-analysis-execution-release-payload-v1",
        "status": "sealed_external_postcommit_lineage_release_payload", "release_id": release_id,
        "authorization_design_commit": "36a872c0be856774b30d9e3c7bb7c5e11e4a11e6", "authorization_design_sha256": subject.AUTHORIZATION_DESIGN_SHA,
        "grant_release_manifest_design_commit": "85ebecad77794c83b54e3d7a6741e2727e6db4ee", "grant_release_manifest_design_sha256": subject.PAIR_DESIGN_SHA,
        "implementation_commit": parent, "implementation_result_record_sha256": subject.LOADER_RESULT_RECORD_SHA,
        "program_sha256": program_sha,
        "wrapper_source_sha256": subject.LOADER_SHA,
        "wrapper_test_sha256": subject.LOADER_TEST_SHA,
        **digest, "one_shot": True, "authorization": dict(subject.FUTURE_AUTHORIZATION),
    }
    payload_sha = hashlib.sha256(subject._canonical_bytes(payload)).hexdigest()
    manifest = {
        "schema": "ouruniv-cf4-v6-open-parent-key-overlap-canonical-read-only-analysis-execution-manifest-v1",
        "status": "complete_paired_external_manifest", "manifest_id": manifest_id,
        "release_path": str(release_path), "release_id": release_id, "release_payload_sha256": payload_sha,
        "authorization_design_sha256": subject.AUTHORIZATION_DESIGN_SHA,
        "grant_release_manifest_design_sha256": subject.PAIR_DESIGN_SHA,
        "implementation_result_record_sha256": subject.LOADER_RESULT_RECORD_SHA,
        "program_sha256": program_sha, **digest, "one_shot": True,
    }
    write_json(manifest_path, manifest, 0o444)
    release = {
        "schema": "ouruniv-cf4-v6-open-parent-key-overlap-canonical-read-only-analysis-execution-release-v1",
        "status": "complete_pass_external_postcommit_lineage_audit", "release_id": release_id,
        "payload": payload, "payload_sha256": payload_sha, "manifest_path": str(manifest_path),
        "manifest_id": manifest_id, "manifest_sha256": sha(manifest_path),
    }
    write_json(release_path, release, 0o444)
    implementation = {str(subject.LOADER_PATH.relative_to(Path("/home/kjhan/BACKUP/CF4"))): subject.LOADER_SHA, str(subject.LOADER_TEST_PATH.relative_to(Path("/home/kjhan/BACKUP/CF4"))): subject.LOADER_TEST_SHA}
    grant_id = hashlib.sha256(("ouruniv-parent-key-overlap-grant-v1\0" + "85ebecad77794c83b54e3d7a6741e2727e6db4ee" + "\0" + sha(release_path) + "\0" + sha(manifest_path) + "\0" + program_sha).encode()).hexdigest()
    grant = {
        "schema": "ouruniv-cf4-v6-open-parent-key-overlap-canonical-read-only-analysis-execution-grant-v1",
        "status": "sealed_one_shot_parent_key_overlap_read_only_analysis_authorization", "grant_id": grant_id, "one_shot": True,
        "program_path": "config/program.json", "program_sha256": program_sha,
        "authorization_design_path": str(subject.AUTHORIZATION_DESIGN.relative_to(Path("/home/kjhan/BACKUP/CF4"))),
        "authorization_design_commit": "36a872c0be856774b30d9e3c7bb7c5e11e4a11e6", "authorization_design_sha256": subject.AUTHORIZATION_DESIGN_SHA,
        "grant_release_manifest_design_path": str(subject.PAIR_DESIGN.relative_to(Path("/home/kjhan/BACKUP/CF4"))),
        "grant_release_manifest_design_commit": "85ebecad77794c83b54e3d7a6741e2727e6db4ee", "grant_release_manifest_design_sha256": subject.PAIR_DESIGN_SHA,
        "implementation_commit": parent, "implementation_result_record_path": str(subject.LOADER_RESULT_RECORD.relative_to(Path("/home/kjhan/BACKUP/CF4"))),
        "implementation_result_record_sha256": subject.LOADER_RESULT_RECORD_SHA, "implementation_file_sha256_map": implementation,
        **digest, "release_path": str(release_path), "release_id": release_id, "release_payload_sha256": payload_sha,
        "release_sha256": sha(release_path), "manifest_path": str(manifest_path), "manifest_id": manifest_id,
        "manifest_sha256": sha(manifest_path), "receipt_root": str(receipts), "authorization": dict(subject.FUTURE_AUTHORIZATION),
    }
    write_json(grant_path, grant, 0o644)
    head = "7" * 40
    outputs = {
        commands[0]: f"{head}\n".encode(), commands[1]: f"{head}\n".encode(), commands[2]: b"",
        commands[3]: f"{parent}\n".encode(), commands[4]: f"{head} {parent}\n".encode(),
        commands[5]: b"A\tconfig/grant.json\n",
        commands[6]: f"100644 blob {subject._git_blob_oid(grant_path.read_bytes())}\tconfig/grant.json\n".encode(),
    }
    def runner(argv, **kwargs):
        assert kwargs["shell"] is False and kwargs["cwd"] == str(root) and kwargs["env"] == subject.GIT_ENV
        return subprocess.CompletedProcess(argv, 0, outputs[tuple(argv)], b"")
    environment = {"SLURM_JOB_NUM_NODES": "1", "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "4", "SLURM_MEM_PER_NODE": "8192", "SLURM_JOB_PARTITION": "debug", "SLURM_JOB_NAME": "cf4-parent-overlap-v1", "CUDA_VISIBLE_DEVICES": ""}
    return runner, environment, grant_path, release_path, receipts, grant_id


def immutable_result():
    array = np.asarray([1.0])
    array.setflags(write=False)
    return MappingProxyType({"value": array, "nested": (MappingProxyType({"ok": True}),)})


def test_public_refuses_before_metadata(monkeypatch):
    called = False
    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise subject.AuthorizationError("metadata gate refused")
    monkeypatch.setattr(subject, "_metadata_bundle", forbidden)
    monkeypatch.setattr(subject, "_run_authorized_for_test", forbidden)
    with pytest.raises((subject.AuthorizationError, FileNotFoundError, PermissionError)):
        subject.run_authorized_canonical_parent_key_overlap_read_only_analysis_v1()
    assert called


def test_canonical_program_is_canonical_and_pins_exact_five_files():
    file = subject._stable_read(subject.PROGRAM, mode=0o644, label="program")
    program = subject._validate_program(file)
    subject._verify_program_implementation_files(program)
    assert subject._canonical_file_bytes(program) == file.payload


def test_absent_real_grant_refuses_before_any_gpfs_read(monkeypatch):
    real = subject._stable_read
    seen = []
    def spy(path, **kwargs):
        value = Path(path)
        if str(value).startswith("/gpfs/"):
            raise AssertionError("external or science GPFS read occurred before the grant gate")
        seen.append(value)
        return real(value, **kwargs)
    monkeypatch.setattr(subject, "_stable_read", spy)
    with pytest.raises(subject.AuthorizationError):
        subject.run_authorized_canonical_parent_key_overlap_read_only_analysis_v1()
    assert subject.GRANT in seen


def test_valid_fake_flow_invokes_loader_once_and_seals_complete(tmp_path, monkeypatch):
    runner, environment, _, _, receipts, grant_id = install_fake_bundle(tmp_path, monkeypatch)
    count = 0
    def loader(contract):
        nonlocal count
        count += 1
        assert contract["git_subprocess_contract"]["expected_HEAD_and_tracking"] == "7" * 40
        return immutable_result()
    result = subject._run_authorized_for_test(git_runner=runner, hostname="Grammar-Debug.cluster", environment=environment, loader_factory=loader)
    assert count == 1 and subject._recursively_immutable(result)
    receipt = receipts / grant_id / "analysis"
    assert {p.name for p in receipt.iterdir()} == {"COMPLETE", "release.anchor", "snapshot.json"}
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o555


@pytest.mark.parametrize("mutation", ["tracking", "dirty", "parent", "diff", "tree"])
def test_git_lineage_failures_preserve_zero_write(tmp_path, monkeypatch, mutation):
    runner, environment, grant, _, receipts, _ = install_fake_bundle(tmp_path, monkeypatch)
    commands = subject.GIT_COMMANDS
    valid = {command: runner(list(command), cwd=str(subject.ROOT), env=subject.GIT_ENV, shell=False, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30).stdout for command in commands}
    if mutation == "tracking": valid[commands[1]] = ("8" * 40 + "\n").encode()
    elif mutation == "dirty": valid[commands[2]] = b"?? src/shadow.py\0"
    elif mutation == "parent": valid[commands[3]] = ("9" * 40 + "\n").encode()
    elif mutation == "diff": valid[commands[5]] = b"A\tconfig/grant.json\nA\tsrc/extra.py\n"
    else: valid[commands[6]] = b"100644 blob " + b"0" * 40 + b"\tconfig/grant.json\n"
    def bad(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, valid[tuple(argv)], b"")
    with pytest.raises(subject.AuthorizationError):
        subject._run_authorized_for_test(git_runner=bad, hostname="grammar-debug", environment=environment, loader_factory=lambda _: immutable_result())
    assert not receipts.exists()
    assert grant.exists()


def test_git_blob_uses_framed_object_not_raw_sha1():
    payload = b"grant\n"
    assert subject._git_blob_oid(payload) == hashlib.sha1(b"blob 6\0grant\n", usedforsecurity=False).hexdigest()
    assert subject._git_blob_oid(payload) != hashlib.sha1(payload, usedforsecurity=False).hexdigest()


def test_pair_or_resource_failure_before_receipt(tmp_path, monkeypatch):
    runner, environment, _, release, receipts, _ = install_fake_bundle(tmp_path, monkeypatch)
    release.chmod(0o644)
    with pytest.raises(subject.AuthorizationError):
        subject._run_authorized_for_test(git_runner=runner, hostname="grammar-debug", environment=environment, loader_factory=lambda _: immutable_result())
    assert not receipts.exists()
    release.chmod(0o444)
    environment["SLURM_CPUS_PER_TASK"] = "5"
    with pytest.raises(subject.AuthorizationError):
        subject._run_authorized_for_test(git_runner=runner, hostname="grammar-debug", environment=environment, loader_factory=lambda _: immutable_result())
    assert not receipts.exists()


def test_preexisting_receipt_root_is_preserved(tmp_path, monkeypatch):
    runner, environment, _, _, receipts, grant_id = install_fake_bundle(tmp_path, monkeypatch)
    receipts.mkdir(mode=0o700)
    subject._run_authorized_for_test(git_runner=runner, hostname="grammar-debug", environment=environment, loader_factory=lambda _: immutable_result())
    assert receipts.is_dir()
    assert (receipts / grant_id / "analysis" / "COMPLETE").is_file()


def test_same_bytes_release_inode_replacement_is_rejected(tmp_path, monkeypatch):
    runner, environment, _, release, receipts, _ = install_fake_bundle(tmp_path, monkeypatch)
    def interrupt(checkpoint):
        if checkpoint == "after_anchor":
            payload = release.read_bytes()
            release.unlink()
            release.write_bytes(payload)
            release.chmod(0o444)
    with pytest.raises(subject.AuthorizationError):
        subject._run_authorized_for_test(git_runner=runner, hostname="grammar-debug", environment=environment, loader_factory=lambda _: immutable_result(), interrupt=interrupt)
    assert not receipts.exists()


def test_release_and_anchor_joint_inode_replacement_is_rejected(tmp_path, monkeypatch):
    runner, environment, _, release, _, _ = install_fake_bundle(tmp_path, monkeypatch)
    bundle = subject._metadata_bundle(git_runner=runner, hostname="grammar-debug", environment=environment)
    receipt, snapshot_sha, live_bundle = subject._bootstrap_receipt(bundle)
    old_inode = release.stat().st_ino
    payload = release.read_bytes()
    (receipt / "release.anchor").unlink()
    release.unlink()
    release.write_bytes(payload)
    release.chmod(0o444)
    os.link(release, receipt / "release.anchor")
    assert release.stat().st_ino != old_inode
    with pytest.raises(subject.AuthorizationError):
        subject._revalidate_snapshot(live_bundle, receipt, snapshot_sha)


def test_rollback_refuses_to_delete_replaced_created_directory(tmp_path, monkeypatch):
    runner, environment, _, _, receipts, grant_id = install_fake_bundle(tmp_path, monkeypatch)
    receipt = receipts / grant_id / "analysis"
    original = receipt.with_name("analysis_original")
    def interrupt(checkpoint):
        if checkpoint == "after_mkdir":
            receipt.rename(original)
            receipt.mkdir(mode=0o700)
            raise RuntimeError("injected replacement")
    with pytest.raises(RuntimeError):
        subject._run_authorized_for_test(git_runner=runner, hostname="grammar-debug", environment=environment, loader_factory=lambda _: immutable_result(), interrupt=interrupt)
    assert original.is_dir() and receipt.is_dir()


def test_after_anchor_receipt_replacement_stays_untouched_and_empty(tmp_path, monkeypatch):
    runner, environment, _, _, receipts, grant_id = install_fake_bundle(tmp_path, monkeypatch)
    receipt = receipts / grant_id / "analysis"
    original = receipt.with_name("analysis_original")
    def interrupt(checkpoint):
        if checkpoint == "after_anchor":
            receipt.rename(original)
            receipt.mkdir(mode=0o700)
    with pytest.raises(subject.AuthorizationError):
        subject._run_authorized_for_test(git_runner=runner, hostname="grammar-debug", environment=environment, loader_factory=lambda _: immutable_result(), interrupt=interrupt)
    assert list(receipt.iterdir()) == []
    assert list(original.iterdir()) == []


@pytest.mark.parametrize("checkpoint", ["after_mkdir", "after_anchor"])
def test_pre_snapshot_interrupt_rolls_back_to_zero_write(tmp_path, monkeypatch, checkpoint):
    runner, environment, _, _, receipts, _ = install_fake_bundle(tmp_path, monkeypatch)
    def interrupt(value):
        if value == checkpoint:
            raise RuntimeError("injected")
    with pytest.raises(RuntimeError):
        subject._run_authorized_for_test(git_runner=runner, hostname="grammar-debug", environment=environment, loader_factory=lambda _: immutable_result(), interrupt=interrupt)
    assert not receipts.exists()


@pytest.mark.parametrize("checkpoint", ["after_snapshot", "after_RUNNING"])
def test_post_snapshot_interrupt_seals_failed(tmp_path, monkeypatch, checkpoint):
    runner, environment, _, _, receipts, grant_id = install_fake_bundle(tmp_path, monkeypatch)
    def interrupt(value):
        if value == checkpoint:
            raise RuntimeError("injected")
    with pytest.raises(RuntimeError):
        subject._run_authorized_for_test(git_runner=runner, hostname="grammar-debug", environment=environment, loader_factory=lambda _: immutable_result(), interrupt=interrupt)
    receipt = receipts / grant_id / "analysis"
    assert {p.name for p in receipt.iterdir()} == {"FAILED", "release.anchor", "snapshot.json"}


@pytest.mark.parametrize("checkpoint", ["after_snapshot", "after_RUNNING"])
def test_post_snapshot_receipt_replacement_seals_old_fd_only(tmp_path, monkeypatch, checkpoint):
    runner, environment, _, _, receipts, grant_id = install_fake_bundle(tmp_path, monkeypatch)
    receipt = receipts / grant_id / "analysis"
    original = receipt.with_name("analysis_original")
    def interrupt(value):
        if value == checkpoint:
            receipt.rename(original)
            receipt.mkdir(mode=0o700)
    with pytest.raises(subject.AuthorizationError):
        subject._run_authorized_for_test(git_runner=runner, hostname="grammar-debug", environment=environment, loader_factory=lambda _: immutable_result(), interrupt=interrupt)
    assert list(receipt.iterdir()) == []
    assert {path.name for path in original.iterdir()} == {"FAILED", "release.anchor", "snapshot.json"}
    assert stat.S_IMODE(original.stat().st_mode) == 0o555


def test_mutable_result_becomes_failed(tmp_path, monkeypatch):
    runner, environment, _, _, receipts, grant_id = install_fake_bundle(tmp_path, monkeypatch)
    with pytest.raises(subject.AuthorizationError):
        subject._run_authorized_for_test(git_runner=runner, hostname="grammar-debug", environment=environment, loader_factory=lambda _: {"mutable": []})
    receipt = receipts / grant_id / "analysis"
    assert {p.name for p in receipt.iterdir()} == {"FAILED", "release.anchor", "snapshot.json"}


def test_supervisor_maps_timeout_and_seals_running_receipt(tmp_path, monkeypatch):
    runner, environment, _, _, receipts, grant_id = install_fake_bundle(tmp_path, monkeypatch)
    bundle = subject._metadata_bundle(git_runner=runner, hostname="grammar-debug", environment=environment)
    receipt, _, live_bundle = subject._bootstrap_receipt(bundle)
    monkeypatch.setattr(subject, "_metadata_bundle", lambda **kwargs: live_bundle)
    assert subject._supervise_receipt("timeout_124") == 124
    assert {p.name for p in receipt.iterdir()} == {"FAILED", "release.anchor", "snapshot.json"}
    assert receipt == receipts / grant_id / "analysis"


def test_supervisor_rejects_joint_snapshot_and_complete_marker_forgery(tmp_path, monkeypatch):
    runner, environment, _, _, receipts, grant_id = install_fake_bundle(tmp_path, monkeypatch)
    captured = {}
    def loader(_):
        return immutable_result()
    subject._run_authorized_for_test(git_runner=runner, hostname="grammar-debug", environment=environment, loader_factory=loader)
    # Re-open the metadata after the hard link exists, matching the supervisor's bundle.
    bundle = subject._metadata_bundle(git_runner=runner, hostname="grammar-debug", environment=environment)
    monkeypatch.setattr(subject, "_metadata_bundle", lambda **kwargs: bundle)
    receipt = receipts / grant_id / "analysis"
    receipt.chmod(0o700)
    snapshot_path = receipt / "snapshot.json"
    complete_path = receipt / "COMPLETE"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["release_ino"] += 1
    snapshot_path.chmod(0o600)
    snapshot_path.write_bytes(canonical(snapshot))
    snapshot_path.chmod(0o444)
    complete = json.loads(complete_path.read_text())
    complete["snapshot_sha256"] = sha(snapshot_path)
    complete_path.chmod(0o600)
    complete_path.write_bytes(canonical(complete))
    complete_path.chmod(0o444)
    receipt.chmod(0o555)
    with pytest.raises(subject.AuthorizationError):
        subject._supervise_receipt("success_0")


def test_nonwhitelisted_git_is_refused():
    with pytest.raises(subject.AuthorizationError):
        subject._run_git(("/usr/bin/git", "log"), runner=subprocess.run)


def test_status_is_fail_closed_for_symlink(tmp_path, monkeypatch):
    grant = tmp_path / "grant"
    target = tmp_path / "target"
    target.write_text("{}")
    grant.symlink_to(target)
    monkeypatch.setattr(subject, "GRANT", grant)
    assert subject.receipt_status() == "invalid_fail_closed"
