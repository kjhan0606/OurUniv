import inspect
import hashlib
import json
import os
from pathlib import Path
import subprocess

import cf4_aggregate_evidence_smc_execution_authorized_v2 as authorized
import pytest


def _program():
    return json.loads(authorized.CANONICAL_PROGRAM.read_text())


def _git(repo, *arguments):
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(path, text, *, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755 if executable else 0o644)


def _sealed_temp_git(
    tmp_path, *, grant_mutation=None, extra_grant_change=False,
    intermediate_auth_change=False,
):
    repo = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "agent/freeze-zoom-pipeline", str(repo)],
        check=True, capture_output=True,
    )
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Authorization Test")
    for index, relative in enumerate(authorized.AUTHORIZATION_V2_PATHS):
        _write(
            repo / relative,
            f"authorization-v2 fixture {index}\n",
            executable=relative.endswith(".sh"),
        )
    _git(repo, "add", *authorized.AUTHORIZATION_V2_PATHS)
    _git(repo, "commit", "-m", "authorization implementation")
    implementation_commit = _git(repo, "rev-parse", "HEAD")
    rows = []
    for relative in authorized.AUTHORIZATION_V2_PATHS:
        path = repo / relative
        mode, _, blob, _ = _git(repo, "ls-tree", "HEAD", "--", relative).split()
        rows.append({
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "git_blob_oid": blob,
            "git_mode": mode,
        })
    if intermediate_auth_change:
        changed = repo / authorized.AUTHORIZATION_V2_PATHS[2]
        changed.write_text(changed.read_text() + "intermediate mutation\n")
        _git(repo, "add", authorized.AUTHORIZATION_V2_PATHS[2])
        _git(repo, "commit", "-m", "forbidden intermediate authorization change")
    result_record = {
        "schema": (
            "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-"
            "implementation-result-record-v1"
        ),
        "status": "complete_pass_postcommit_authorization_v2_implementation",
        "commit_lineage": {"git_commit": implementation_commit},
        "committed_authorization_files": rows,
    }
    result_path = repo / authorized.AUTHORIZATION_RESULT_RELATIVE_PATH
    _write(result_path, json.dumps(result_record, sort_keys=True) + "\n")
    _git(repo, "add", authorized.AUTHORIZATION_RESULT_RELATIVE_PATH)
    _git(repo, "commit", "-m", "seal authorization result")
    result_commit = _git(repo, "rev-parse", "HEAD")
    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    program_sha = "a" * 64
    data = tmp_path / "never-created-data"
    state = tmp_path / "never-created-state"
    grant = {
        "schema": "ouruniv-cf4-aggregate-evidence-smc-execution-grant-v2",
        "status": "sealed_one_shot_execution_authorization",
        "one_shot": True,
        "grant_parent_commit": result_commit,
        "authorization_program_sha256": program_sha,
        "authorization_design_base_commit": authorized.AUTHORIZATION_DESIGN_BASE_COMMIT,
        "runner_implementation_commit": authorized.RUNNER_IMPLEMENTATION_COMMIT,
        "runner_result_record_sha256": authorized.RUNNER_RESULT_SHA256,
        "authorization_v2_implementation_commit": implementation_commit,
        "authorization_v2_implementation_result_record": {
            "path": authorized.AUTHORIZATION_RESULT_RELATIVE_PATH,
            "sha256": result_sha,
        },
        "authorization_v2_files": rows,
        "data_directory": str(data),
        "state_directory": str(state),
        "authorization": {
            "production_SMC_execution_authorized": True,
            "oracle_cache_population_authorized": True,
            "conditional_field_bank_authorized": False,
            "candidate_generation_authorized": False,
            "parent_or_seed_selection_authorized": False,
            "PM_authorized": False,
            "HOP_authorized": False,
            "RAMSES_authorized": False,
            "downstream_execution_authorized": False,
            "automatic_retry_authorized": False,
            "automatic_retune_authorized": False,
            "automatic_scale_up_authorized": False,
            "automatic_follow_on_authorized": False,
        },
        "precommit_audit_verdict": "EXECUTION GO",
    }
    if grant_mutation is not None:
        grant_mutation(grant)
    grant_path = repo / authorized.GRANT_RELATIVE_PATH
    _write(grant_path, json.dumps(grant, sort_keys=True) + "\n")
    _git(repo, "add", authorized.GRANT_RELATIVE_PATH)
    if extra_grant_change:
        _write(repo / "extra.txt", "forbidden extra grant change\n")
        _git(repo, "add", "extra.txt")
    _git(repo, "commit", "-m", "one-shot grant")
    grant_commit = _git(repo, "rev-parse", "HEAD")
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "agent/freeze-zoom-pipeline")
    release = tmp_path / "external-lineage-release.json"
    _write(release, json.dumps({
        "schema": (
            "ouruniv-cf4-aggregate-evidence-smc-execution-authorization-"
            "release-v2"
        ),
        "status": "complete_pass_external_postcommit_lineage_audit",
        "verdict": "LINEAGE GO",
        "grant_commit": grant_commit,
        "grant_parent_commit": result_commit,
        "grant_sha256": hashlib.sha256(grant_path.read_bytes()).hexdigest(),
        "authorization_v2_implementation_result_record_sha256": result_sha,
    }, sort_keys=True) + "\n")
    return {
        "repo": repo,
        "remote": remote,
        "grant": grant_path,
        "release": release,
        "program_sha": program_sha,
        "data": data,
        "state": state,
        "result_commit": result_commit,
    }


def _install_temp_git(monkeypatch, fixture):
    monkeypatch.setattr(authorized, "ROOT", fixture["repo"])
    monkeypatch.setattr(authorized, "CANONICAL_GRANT", fixture["grant"])
    monkeypatch.setattr(authorized, "EXTERNAL_RELEASE", fixture["release"])
    monkeypatch.setattr(authorized, "PROGRAM_SHA256", fixture["program_sha"])
    monkeypatch.setattr(authorized, "DATA_DIRECTORY", fixture["data"])
    monkeypatch.setattr(authorized, "STATE_DIRECTORY", fixture["state"])


def test_authorization_program_exactly_pins_runner_inputs_and_closed_matrix():
    program = _program()
    authorized.validate_authorization_program(
        program, verify_file_hashes=True
    )
    assert program["authorization_design_base_commit"] == (
        "d3213fa8fa2effe82dc6874911d21132dc088b4b"
    )
    assert program["runner_implementation_commit"] == (
        "375438fa6dc911059da57e46be95183ee45f1837"
    )
    assert program["runner_implementation_result_record"] == {
        "path": "config/cf4_aggregate_evidence_smc_runner_implementation_result_record.json",
        "sha256": "d96708a9f6b4998237aba4b4078918e2b483b7bb7cbf370bb1866da692d9f92a",
    }
    assert len(program["audited_runner_files"]) == 7
    assert program["audited_runner_files"] == [
        {
            "path": path,
            "sha256": digest,
            "git_blob_oid": blob,
            "git_mode": mode,
        }
        for path, digest, blob, mode in authorized.EXPECTED_RUNNER_FILES
    ]
    assert program["authorization"] == authorized._expected_authorization()
    assert program["authorization"][
        "versioned_one_shot_authorization_design_and_implementation_authorized"
    ] is True
    assert all(
        value is False
        for key, value in program["authorization"].items()
        if key != (
            "versioned_one_shot_authorization_design_and_implementation_authorized"
        )
    )
    assert program["future_grant_interface"]["current_grant_present"] is False
    assert not authorized.CANONICAL_GRANT.exists()


def test_frozen_input_lineage_is_exactly_the_base_program_lineage():
    program = _program()
    base = json.loads(authorized.BASE_PROGRAM.read_text())
    frozen = program["frozen_input_lineage"]
    assert frozen["pinned_local_files"] == base["pinned_local_files"]
    assert frozen["external_inputs"] == base["external_inputs"]
    assert frozen["parent_seed_range_inclusive"] == [3193, 3448]
    assert frozen["parent_count"] == 256


def test_public_entry_accepts_no_override_and_refuses_before_base_core(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        authorized.base_execution,
        "_execute_authorized_program",
        lambda program: calls.append(program),
    )
    assert set(inspect.signature(authorized.run_authorized_v2).parameters) == {
        "program_path"
    }
    with pytest.raises(PermissionError, match="canonical program"):
        authorized.run_authorized_v2(tmp_path / "wrong.json")
    with pytest.raises(PermissionError, match="grant is absent"):
        authorized.run_authorized_v2(authorized.CANONICAL_PROGRAM)
    assert calls == []
    assert list(tmp_path.iterdir()) == []


def test_public_entry_rejects_program_hash_before_grant_or_core(monkeypatch):
    calls = []
    monkeypatch.setattr(authorized, "PROGRAM_SHA256", "0" * 64)
    monkeypatch.setattr(
        authorized,
        "require_execution_authorization",
        lambda program: calls.append("grant"),
    )
    monkeypatch.setattr(
        authorized.base_execution,
        "_execute_authorized_program",
        lambda program: calls.append("core"),
    )
    with pytest.raises(RuntimeError, match="program hash mismatch"):
        authorized.run_authorized_v2(authorized.CANONICAL_PROGRAM)
    assert calls == []


def test_future_grant_wrong_path_absent_and_unsealed_are_fail_closed(
    tmp_path, monkeypatch
):
    program = _program()
    with pytest.raises(PermissionError, match="path is not canonical"):
        authorized.validate_future_grant(tmp_path / "wrong.json", program)

    grant = tmp_path / "grant.json"
    monkeypatch.setattr(authorized, "CANONICAL_GRANT", grant)
    with pytest.raises(PermissionError, match="grant is absent"):
        authorized.validate_future_grant(grant, program)

    grant.write_text(json.dumps({
        "schema": "ouruniv-cf4-aggregate-evidence-smc-execution-grant-v2",
        "status": "draft_unsealed",
    }))
    with pytest.raises(PermissionError, match="wrong or unsealed"):
        authorized.validate_future_grant(grant, program)


def test_program_rejects_missing_extra_or_open_authorization_keys():
    original = _program()
    for changed in (
        {
            **original,
            "authorization": {
                key: value for key, value in original["authorization"].items()
                if key != "HOP_authorized"
            },
        },
        {
            **original,
            "authorization": {
                **original["authorization"], "invented": False
            },
        },
        {
            **original,
            "authorization": {
                **original["authorization"],
                "production_SMC_execution_authorized": True,
            },
        },
    ):
        with pytest.raises(RuntimeError, match="matrix"):
            authorized.validate_authorization_program(
                changed, verify_file_hashes=False
            )


def test_full_predecessor_rehash_rejects_design_runner_record_and_runner_file(
    monkeypatch,
):
    program = _program()
    original_sha = authorized.sha256_file
    targets = (
        (authorized.CANONICAL_DESIGN, "design hash mismatch"),
        (authorized.RUNNER_RESULT_PATH, "result record hash mismatch"),
        (authorized.ROOT / authorized.EXPECTED_RUNNER_FILES[4][0], "runner file changed"),
    )
    for target, message in targets:
        def forged(path, *, _target=target):
            if Path(path).resolve() == _target.resolve():
                return "0" * 64
            return original_sha(path)

        monkeypatch.setattr(authorized, "sha256_file", forged)
        with pytest.raises(RuntimeError, match=message):
            authorized.validate_authorization_program(
                program, verify_file_hashes=True
            )
        monkeypatch.setattr(authorized, "sha256_file", original_sha)


def test_temp_git_valid_single_add_grant_invokes_fake_core_once_without_gpfs(
    tmp_path, monkeypatch
):
    fixture = _sealed_temp_git(tmp_path)
    _install_temp_git(monkeypatch, fixture)
    monkeypatch.setattr(
        authorized, "validate_authorization_program", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        authorized,
        "load_canonical_authorization_program",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        authorized.base_execution,
        "load_canonical_program",
        lambda **kwargs: {"base": "sealed"},
    )
    calls = []
    monkeypatch.setattr(
        authorized.base_execution,
        "_execute_authorized_program",
        lambda program: calls.append(program) or {"status": "fake-pass"},
    )
    monkeypatch.setattr(
        authorized,
        "CANONICAL_PROGRAM",
        fixture["repo"] / authorized.AUTHORIZATION_V2_PATHS[1],
    )
    result = authorized.run_authorized_v2(authorized.CANONICAL_PROGRAM)
    assert result == {"status": "fake-pass"}
    assert calls == [{"base": "sealed"}]
    assert not fixture["data"].exists()
    assert not fixture["state"].exists()


def test_temp_git_rejects_wrong_parent_and_exact_grant_keyset(
    tmp_path, monkeypatch
):
    cases = {
        "wrong_parent": lambda grant: grant.__setitem__(
            "grant_parent_commit", "0" * 40
        ),
        "missing_key": lambda grant: grant.pop("precommit_audit_verdict"),
        "extra_key": lambda grant: grant.__setitem__(
            "postcommit_audit_verdict", "LINEAGE GO"
        ),
    }
    for name, mutation in cases.items():
        case = tmp_path / name
        case.mkdir()
        fixture = _sealed_temp_git(case, grant_mutation=mutation)
        _install_temp_git(monkeypatch, fixture)
        monkeypatch.setattr(
            authorized,
            "validate_authorization_program",
            lambda *args, **kwargs: None,
        )
        message = "parent changed" if name == "wrong_parent" else "wrong or unsealed"
        with pytest.raises((PermissionError, RuntimeError), match=message):
            authorized.validate_future_grant(fixture["grant"], {})


def test_temp_git_rejects_extra_grant_commit_change(tmp_path, monkeypatch):
    fixture = _sealed_temp_git(tmp_path, extra_grant_change=True)
    _install_temp_git(monkeypatch, fixture)
    monkeypatch.setattr(
        authorized, "validate_authorization_program", lambda *args, **kwargs: None
    )
    with pytest.raises(RuntimeError, match="exact one-file addition"):
        authorized.validate_future_grant(fixture["grant"], {})


def test_temp_git_rejects_intermediate_authorization_file_commit(
    tmp_path, monkeypatch
):
    fixture = _sealed_temp_git(tmp_path, intermediate_auth_change=True)
    _install_temp_git(monkeypatch, fixture)
    monkeypatch.setattr(
        authorized, "validate_authorization_program", lambda *args, **kwargs: None
    )
    with pytest.raises(RuntimeError, match="not the direct child"):
        authorized.validate_future_grant(fixture["grant"], {})


def test_temp_git_rejects_head_local_tracking_or_remote_mismatch(
    tmp_path, monkeypatch
):
    for mismatch in ("head", "local", "tracking", "remote"):
        case = tmp_path / mismatch
        case.mkdir()
        fixture = _sealed_temp_git(case)
        _install_temp_git(monkeypatch, fixture)
        monkeypatch.setattr(
            authorized,
            "validate_authorization_program",
            lambda *args, **kwargs: None,
        )
        if mismatch == "head":
            grant_bytes = fixture["grant"].read_bytes()
            _git(fixture["repo"], "checkout", "--detach", fixture["result_commit"])
            fixture["grant"].parent.mkdir(parents=True, exist_ok=True)
            fixture["grant"].write_bytes(grant_bytes)
        elif mismatch == "local":
            _git(fixture["repo"], "checkout", "--detach", "HEAD")
            _git(
                fixture["repo"], "update-ref",
                "refs/heads/agent/freeze-zoom-pipeline",
                fixture["result_commit"],
            )
        elif mismatch == "tracking":
            _git(
                fixture["repo"], "update-ref",
                "refs/remotes/origin/agent/freeze-zoom-pipeline",
                fixture["result_commit"],
            )
        else:
            subprocess.run(
                [
                    "git", "--git-dir", str(fixture["remote"]), "update-ref",
                    "refs/heads/agent/freeze-zoom-pipeline",
                    fixture["result_commit"],
                ],
                check=True,
            )
        with pytest.raises(RuntimeError, match="identical refs"):
            authorized.validate_future_grant(fixture["grant"], {})


def test_temp_git_rejects_tracked_blob_worktree_mismatch(tmp_path, monkeypatch):
    fixture = _sealed_temp_git(tmp_path)
    _install_temp_git(monkeypatch, fixture)
    monkeypatch.setattr(
        authorized, "validate_authorization_program", lambda *args, **kwargs: None
    )
    fixture["grant"].write_text(fixture["grant"].read_text() + " ")
    with pytest.raises(RuntimeError, match="tracked blob or worktree"):
        authorized.validate_future_grant(fixture["grant"], {})


def test_temp_git_requires_external_non_self_committed_lineage_go(
    tmp_path, monkeypatch
):
    fixture = _sealed_temp_git(tmp_path)
    _install_temp_git(monkeypatch, fixture)
    monkeypatch.setattr(
        authorized, "validate_authorization_program", lambda *args, **kwargs: None
    )
    release_bytes = fixture["release"].read_bytes()
    fixture["release"].unlink()
    with pytest.raises(PermissionError, match="release is absent"):
        authorized.validate_future_grant(fixture["grant"], {})

    inside = fixture["repo"] / "untracked-self-release.json"
    inside.write_bytes(release_bytes)
    monkeypatch.setattr(authorized, "EXTERNAL_RELEASE", inside)
    with pytest.raises(PermissionError, match="must not be self-committed"):
        authorized.validate_future_grant(fixture["grant"], {})
