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

import cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v2 as subject


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
        "src/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v2.py",
        "scripts/run_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v2.sbatch",
        "scripts/status_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v2.sh",
        "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v2.py",
        "tests/test_cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_authorized_v2_runner.py",
    ]
    return {
        "schema": "ouruniv-cf4-v6-open-parent-key-overlap-canonical-read-only-analysis-authorized-program-v2",
        "status": "frozen_v2_program_execution_false_until_sealed_v2_pair_grant_and_audits",
        "date": "2026-08-23",
        "purpose": "Freeze the exact no-argument execution-false v2 program binding for the remediated one-shot memory-only parent/key overlap analysis.",
        "lineage": {
            "branch": "agent/freeze-zoom-pipeline",
            "remediation_design_commit": subject.REMEDIATION_DESIGN_COMMIT,
            "remediation_design_sha256": subject.REMEDIATION_DESIGN_SHA,
            "grant_release_manifest_design_commit": subject.PAIR_DESIGN_COMMIT,
            "grant_release_manifest_design_sha256": subject.PAIR_DESIGN_SHA,
            "inherited_science_authorization_design_commit": "36a872c0be856774b30d9e3c7bb7c5e11e4a11e6",
            "inherited_science_authorization_design_sha256": subject.AUTHORIZATION_DESIGN_SHA,
            "v1_frozen_implementation_commit": "6240a77a4b4e8ec9cd8d9437cba80793162b485b",
            "v1_frozen_grant_commit": "1967825d8613fa031b5ff40e845bf2031425852a",
        },
        "canonical_paths": paths,
        "resource_contract": {"cpus_per_task": 4, "login_host": "grammar", "node": "grammar-debug", "nodes": 1, "ntasks": 1, "partition": "debug", "pre_registered_maximum_expected_RSS_GiB": 6.5, "requested_memory_GiB": 8, "requested_memory_margin_over_expected_percent": 23.076923076923077, "requeue": False, "submission_mechanism": "Slurm_only", "time_limit": "01:00:00"},
        "implementation_files": [{"path": name, "sha256": hashlib.sha256(name.encode()).hexdigest(), "mode": "0755" if name.startswith("scripts/") else "0644"} for name in names],
        "authorization": dict(subject.FALSE_AUTHORIZATION),
        "next": {"canonical_analysis_execution_authorized": False, "canonical_artifact_read_authorized": False, "external_pair_and_committed_grant_audits_required": True, "immediate": "exact_six_v2_implementation_precommit_audit_only", "receipt_creation_authorized": False, "slurm_submission_authorized": False},
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
    release_id = hashlib.sha256(("ouruniv-parent-key-overlap-release-v2\0" + subject.PAIR_DESIGN_COMMIT + "\0" + subject.REMEDIATION_DESIGN_SHA + "\0" + program_sha).encode()).hexdigest()
    manifest_id = hashlib.sha256(("ouruniv-parent-key-overlap-manifest-v2\0" + subject.PAIR_DESIGN_COMMIT + "\0" + release_id).encode()).hexdigest()
    parent = "3" * 40
    digests = {"artifact_contract_digest": "1cab22081d83abc5b09c8dfbab81e37aeed789c09c0004344808c67986c95ff5", "execution_contract_digest": "2511effaa8860ea9af22aaf2084a91eb2b0a0b5028836f1b054e1d4b166cc5ea", "canonical_paths_digest": "e5acd198245a6eb7f9318367f9a8a72d4f3b193bf3380b008ec95cd7298c5bec", "resource_contract_digest": "faa8a358d8aeeb4caed9e6b19b3a979227fd8b1b1e938f6aaa099ef2ecdd9abc"}
    implementation = {row["path"]: row["sha256"] for row in value["implementation_files"]}
    payload = {
        "schema": "ouruniv-cf4-v6-open-parent-key-overlap-canonical-read-only-analysis-execution-release-payload-v2",
        "status": "sealed_external_postcommit_lineage_release_payload", "release_id": release_id,
        "remediation_design_commit": subject.REMEDIATION_DESIGN_COMMIT,
        "remediation_design_sha256": subject.REMEDIATION_DESIGN_SHA,
        "grant_release_manifest_design_commit": subject.PAIR_DESIGN_COMMIT,
        "grant_release_manifest_design_sha256": subject.PAIR_DESIGN_SHA,
        "implementation_commit": parent, "program_sha256": program_sha,
        "implementation_file_sha256_map": implementation,
        **digests, "one_shot": True, "authorization": dict(subject.FUTURE_AUTHORIZATION),
    }
    payload_sha = hashlib.sha256(subject._canonical_bytes(payload)).hexdigest()
    manifest = {
        "schema": "ouruniv-cf4-v6-open-parent-key-overlap-canonical-read-only-analysis-execution-manifest-v2",
        "status": "complete_paired_external_manifest", "manifest_id": manifest_id,
        "release_path": str(release_path), "release_id": release_id, "release_payload_sha256": payload_sha,
        "remediation_design_sha256": subject.REMEDIATION_DESIGN_SHA,
        "grant_release_manifest_design_sha256": subject.PAIR_DESIGN_SHA,
        "implementation_commit": parent, "program_sha256": program_sha,
        "implementation_file_sha256_map": implementation,
        **digests, "one_shot": True,
    }
    write_json(manifest_path, manifest, 0o444)
    release = {
        "schema": "ouruniv-cf4-v6-open-parent-key-overlap-canonical-read-only-analysis-execution-release-v2",
        "status": "complete_pass_external_postcommit_lineage_audit", "release_id": release_id,
        "payload": payload, "payload_sha256": payload_sha, "manifest_path": str(manifest_path),
        "manifest_id": manifest_id, "manifest_sha256": sha(manifest_path),
    }
    write_json(release_path, release, 0o444)
    grant_id = hashlib.sha256(("ouruniv-parent-key-overlap-grant-v2\0" + subject.PAIR_DESIGN_COMMIT + "\0" + sha(release_path) + "\0" + sha(manifest_path) + "\0" + program_sha).encode()).hexdigest()
    grant = {
        "schema": "ouruniv-cf4-v6-open-parent-key-overlap-canonical-read-only-analysis-execution-grant-v2",
        "status": "sealed_one_shot_parent_key_overlap_read_only_analysis_authorization", "grant_id": grant_id, "one_shot": True,
        "program_path": "config/program.json", "program_sha256": program_sha,
        "remediation_design_path": str(subject.REMEDIATION_DESIGN.relative_to(Path("/home/kjhan/BACKUP/CF4"))),
        "remediation_design_commit": subject.REMEDIATION_DESIGN_COMMIT,
        "remediation_design_sha256": subject.REMEDIATION_DESIGN_SHA,
        "grant_release_manifest_design_path": str(subject.PAIR_DESIGN.relative_to(Path("/home/kjhan/BACKUP/CF4"))),
        "grant_release_manifest_design_commit": subject.PAIR_DESIGN_COMMIT,
        "grant_release_manifest_design_sha256": subject.PAIR_DESIGN_SHA,
        "implementation_commit": parent, "implementation_file_sha256_map": implementation,
        **digests, "release_path": str(release_path), "release_id": release_id, "release_payload_sha256": payload_sha,
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


def directory_fingerprint(path: Path):
    mode = stat.S_IMODE(path.stat().st_mode)
    if path.is_dir():
        return ("directory", mode, tuple(sorted((child.name, directory_fingerprint(child)) for child in path.iterdir())))
    return ("file", mode, path.read_bytes())


def test_public_refuses_before_metadata(monkeypatch):
    called = False
    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise subject.AuthorizationError("metadata gate refused")
    monkeypatch.setattr(subject, "_metadata_bundle", forbidden)
    monkeypatch.setattr(subject, "_run_authorized_for_test", forbidden)
    with pytest.raises((subject.AuthorizationError, FileNotFoundError, PermissionError)):
        subject.run_authorized_canonical_parent_key_overlap_read_only_analysis_v2()
    assert called


def test_canonical_program_is_canonical_and_pins_exact_five_files():
    file = subject._stable_read(subject.PROGRAM, mode=0o644, label="program")
    program = subject._validate_program(file)
    subject._verify_program_implementation_files(program)
    assert subject._canonical_file_bytes(program) == file.payload


def test_committed_erratum_is_hard_pinned_as_fixed_local_provenance():
    assert subject.PAIR_DESIGN_ERRATUM.relative_to(subject.ROOT) == Path(
        "config/cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_grant_release_manifest_design_v2_erratum_v1.json"
    )
    assert subject.PAIR_DESIGN_ERRATUM_COMMIT == "c9f462b478f0f7dc112c5a7431d6d456d3ead2d1"
    assert subject._stable_read(subject.PAIR_DESIGN_ERRATUM, mode=0o644, label="pair design erratum").sha256 == subject.PAIR_DESIGN_ERRATUM_SHA


@pytest.mark.parametrize("permutation", [(1, 0, 2, 3, 4), (4, 3, 2, 1, 0), (0, 2, 1, 3, 4)])
def test_program_rejects_every_tested_row_permutation(tmp_path, permutation):
    value = program_value()
    rows = value["implementation_files"]
    value["implementation_files"] = [rows[index] for index in permutation]
    path = tmp_path / "program.json"
    write_json(path, value, 0o644)
    with pytest.raises(subject.AuthorizationError, match="order or mode"):
        subject._validate_program(subject._stable_read(path, mode=0o644, label="permuted program"))


def test_absent_grant_fixture_refuses_before_any_gpfs_read(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    program = root / "config/program.json"
    grant = root / "config/absent-grant.json"
    receipts = tmp_path / "receipts"
    for name, value in (
        ("ROOT", root), ("PROGRAM", program), ("GRANT", grant),
        ("RELEASE", Path("/gpfs/forbidden-release-v2.json")),
        ("MANIFEST", Path("/gpfs/forbidden-manifest-v2.json")),
        ("RECEIPT_ROOT", receipts),
    ):
        monkeypatch.setattr(subject, name, value)
    monkeypatch.setattr(subject, "_verify_fixed_local_pins", lambda: None)
    monkeypatch.setattr(subject, "_verify_program_implementation_files", lambda value: None)
    write_json(program, program_value(), 0o644)
    real = subject._stable_read
    seen = []
    def spy(path, **kwargs):
        value = Path(path)
        if str(value).startswith("/gpfs/"):
            raise AssertionError("external or science GPFS read occurred before the grant gate")
        seen.append(value)
        return real(value, **kwargs)
    monkeypatch.setattr(subject, "_stable_read", spy)
    with pytest.raises((subject.AuthorizationError, FileNotFoundError)):
        subject.run_authorized_canonical_parent_key_overlap_read_only_analysis_v2()
    assert grant in seen and not receipts.exists()


def test_sealed_grant_metadata_validation_creates_no_receipt_or_science_read(tmp_path, monkeypatch):
    runner, environment, _, _, receipts, grant_id = install_fake_bundle(tmp_path, monkeypatch)
    bundle = subject._metadata_bundle(git_runner=runner, hostname="grammar-debug", environment=environment)
    assert bundle.grant["grant_id"] == grant_id
    assert not receipts.exists()


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
    lease = subject._bootstrap_receipt(bundle)
    receipt = lease.receipt
    old_inode = release.stat().st_ino
    payload = release.read_bytes()
    (receipt / "release.anchor").unlink()
    release.unlink()
    release.write_bytes(payload)
    release.chmod(0o444)
    os.link(release, receipt / "release.anchor")
    assert release.stat().st_ino != old_inode
    try:
        with pytest.raises(subject.AuthorizationError):
            subject._revalidate_snapshot(lease.bundle, lease.item, lease.snapshot_sha, lease=lease)
    finally:
        lease.close()


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


@pytest.mark.parametrize("ancestor", ["receipt_root", "grant_id", "analysis"])
@pytest.mark.parametrize("replacement_point", ["before_science", "during_loader", "after_analysis", "before_terminal"])
def test_full_held_chain_rejects_postbootstrap_replacement(tmp_path, monkeypatch, ancestor, replacement_point):
    runner, environment, _, _, receipts, grant_id = install_fake_bundle(tmp_path, monkeypatch)
    receipt = receipts / grant_id / "analysis"
    if ancestor == "receipt_root":
        target = receipts
        original = receipts.with_name(receipts.name + "_original")
        hidden_receipt = original / grant_id / "analysis"
    elif ancestor == "grant_id":
        target = receipt.parent
        original = target.with_name(target.name + "_original")
        hidden_receipt = original / "analysis"
    else:
        target = receipt
        original = receipt.with_name("analysis_original")
        hidden_receipt = original
    replacement_fingerprint = []

    def replace():
        target.rename(original)
        target.mkdir(mode=0o700)
        if ancestor == "receipt_root":
            (target / grant_id).mkdir(mode=0o700)
            (target / grant_id / "analysis").mkdir(mode=0o700)
        elif ancestor == "grant_id":
            (target / "analysis").mkdir(mode=0o700)
        replacement_fingerprint.append(directory_fingerprint(target))

    def interrupt(checkpoint):
        if checkpoint == replacement_point:
            replace()

    def loader(_):
        if replacement_point == "during_loader":
            replace()
        return immutable_result()

    with pytest.raises(subject.AuthorizationError):
        subject._run_authorized_for_test(
            git_runner=runner,
            hostname="grammar-debug",
            environment=environment,
            loader_factory=loader,
            interrupt=interrupt,
        )
    assert len(replacement_fingerprint) == 1
    assert directory_fingerprint(target) == replacement_fingerprint[0]
    assert {path.name for path in hidden_receipt.iterdir()} == {"FAILED", "release.anchor", "snapshot.json"}
    assert stat.S_IMODE(hidden_receipt.stat().st_mode) == 0o555


def test_snapshot_contains_exact_held_receipt_identity(tmp_path, monkeypatch):
    runner, environment, _, _, _, _ = install_fake_bundle(tmp_path, monkeypatch)
    bundle = subject._metadata_bundle(git_runner=runner, hostname="grammar-debug", environment=environment)
    lease = subject._bootstrap_receipt(bundle)
    try:
        snapshot_file = subject._stable_read_at(lease.receipt_fd, "snapshot.json", mode=0o444, label="snapshot")
        snapshot = subject._parse_canonical_object(snapshot_file.payload, "snapshot")
        assert len(snapshot) == 18
        assert set(snapshot) == {
            "grant_path", "grant_sha256", "program_path", "program_sha256", "release_path",
            "release_sha256", "release_dev", "release_ino", "release_size", "release_nlink",
            "manifest_path", "manifest_sha256", "remediation_design_sha256",
            "grant_release_manifest_design_sha256", "implementation_commit", "receipt_dev",
            "receipt_ino", "receipt_mode",
        }
        live = os.fstat(lease.receipt_fd)
        assert (snapshot["receipt_dev"], snapshot["receipt_ino"], snapshot["receipt_mode"]) == (live.st_dev, live.st_ino, "0700")
        subject._seal_failed_at(lease.bundle, lease.item, lease.snapshot_sha, lease=lease, checkpoint="during_analysis", require_canonical_name=True)
    finally:
        lease.close()


@pytest.mark.parametrize("outcome", ["success", "analysis_failure"])
def test_held_parent_and_receipt_fds_close_exactly_once(tmp_path, monkeypatch, outcome):
    runner, environment, _, _, _, _ = install_fake_bundle(tmp_path, monkeypatch)
    real_bootstrap = subject._bootstrap_receipt
    real_close = os.close
    held: set[int] = set()
    counts: dict[int, int] = {}

    def bootstrap(*args, **kwargs):
        lease = real_bootstrap(*args, **kwargs)
        held.update({lease.root_parent_fd, lease.root_item.fd, lease.parent_fd, lease.receipt_fd})
        return lease

    def close_spy(descriptor):
        if descriptor in held:
            counts[descriptor] = counts.get(descriptor, 0) + 1
        return real_close(descriptor)

    monkeypatch.setattr(subject, "_bootstrap_receipt", bootstrap)
    monkeypatch.setattr(subject.os, "close", close_spy)
    loader = (lambda _: immutable_result()) if outcome == "success" else (lambda _: (_ for _ in ()).throw(RuntimeError("injected")))
    if outcome == "success":
        subject._run_authorized_for_test(git_runner=runner, hostname="grammar-debug", environment=environment, loader_factory=loader)
    else:
        with pytest.raises(RuntimeError, match="injected"):
            subject._run_authorized_for_test(git_runner=runner, hostname="grammar-debug", environment=environment, loader_factory=loader)
    assert held and all(counts.get(descriptor) == 1 for descriptor in held)
    for descriptor in held:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_bootstrap_injected_failure_closes_held_fds_exactly_once(tmp_path, monkeypatch):
    runner, environment, _, _, _, grant_id = install_fake_bundle(tmp_path, monkeypatch)
    real_open = os.open
    real_close = os.close
    held: set[int] = set()
    counts: dict[int, int] = {}

    def open_spy(path, flags, *args, **kwargs):
        descriptor = real_open(path, flags, *args, **kwargs)
        wanted = {str(subject.RECEIPT_ROOT.parent), subject.RECEIPT_ROOT.name, grant_id, "analysis"}
        if str(path) in wanted and flags & os.O_DIRECTORY:
            held.add(descriptor)
        return descriptor

    def close_spy(descriptor):
        if descriptor in held:
            counts[descriptor] = counts.get(descriptor, 0) + 1
        return real_close(descriptor)

    monkeypatch.setattr(subject.os, "open", open_spy)
    monkeypatch.setattr(subject.os, "close", close_spy)
    with pytest.raises(RuntimeError, match="injected"):
        subject._run_authorized_for_test(
            git_runner=runner,
            hostname="grammar-debug",
            environment=environment,
            loader_factory=lambda _: immutable_result(),
            interrupt=lambda checkpoint: (_ for _ in ()).throw(RuntimeError("injected")) if checkpoint == "after_RUNNING" else None,
        )
    assert len(held) == 4 and all(counts.get(descriptor) == 1 for descriptor in held)
    for descriptor in held:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_mutable_result_becomes_failed(tmp_path, monkeypatch):
    runner, environment, _, _, receipts, grant_id = install_fake_bundle(tmp_path, monkeypatch)
    with pytest.raises(subject.AuthorizationError):
        subject._run_authorized_for_test(git_runner=runner, hostname="grammar-debug", environment=environment, loader_factory=lambda _: {"mutable": []})
    receipt = receipts / grant_id / "analysis"
    assert {p.name for p in receipt.iterdir()} == {"FAILED", "release.anchor", "snapshot.json"}


@pytest.mark.parametrize("child_class,expected", [("timeout_124", 124), ("killed_137", 137), ("terminated_143", 143), ("other_nonzero", 1)])
def test_supervisor_maps_nonzero_without_mutating_running_receipt(tmp_path, monkeypatch, child_class, expected):
    runner, environment, _, _, receipts, grant_id = install_fake_bundle(tmp_path, monkeypatch)
    bundle = subject._metadata_bundle(git_runner=runner, hostname="grammar-debug", environment=environment)
    lease = subject._bootstrap_receipt(bundle)
    receipt = lease.receipt
    lease.close()
    monkeypatch.setattr(subject, "_metadata_bundle", lambda **kwargs: lease.bundle)
    before = {path.name: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode)) for path in receipt.iterdir()}
    assert subject._supervise_receipt(child_class) == expected
    after = {path.name: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode)) for path in receipt.iterdir()}
    assert after == before
    assert set(after) == {"RUNNING", "release.anchor", "snapshot.json"}
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
    assert subject._supervise_receipt("success_0") == 65


def test_joint_replacement_is_only_advisory_and_never_supervisor_success(tmp_path, monkeypatch):
    runner, environment, _, _, receipts, grant_id = install_fake_bundle(tmp_path, monkeypatch)
    subject._run_authorized_for_test(
        git_runner=runner,
        hostname="grammar-debug",
        environment=environment,
        loader_factory=lambda _: immutable_result(),
    )
    receipt = receipts / grant_id / "analysis"
    original = receipt.with_name("analysis_original")
    receipt.rename(original)
    receipt.mkdir(mode=0o700)
    os.link(original / "release.anchor", receipt / "release.anchor")
    snapshot = json.loads((original / "snapshot.json").read_text())
    live = receipt.stat()
    snapshot["receipt_dev"] = live.st_dev
    snapshot["receipt_ino"] = live.st_ino
    write_json(receipt / "snapshot.json", snapshot, 0o444)
    complete = json.loads((original / "COMPLETE").read_text())
    complete["snapshot_sha256"] = sha(receipt / "snapshot.json")
    write_json(receipt / "COMPLETE", complete, 0o444)
    receipt.chmod(0o555)
    before = directory_fingerprint(receipt)
    assert subject.receipt_status() == "advisory_raw_complete_untrusted"
    assert subject._supervise_receipt("success_0") == 65
    assert directory_fingerprint(receipt) == before


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
