from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Iterator

import numpy as np
import pytest

import cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_parent_key_overlap_canonical_read_only_analysis_wrapper_v1 as wrapper


def _json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _analysis_arrays() -> dict[str, Any]:
    keys = np.array(
        [
            [-2, 0, 0, 1, 0, 0],
            [-1, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 0, 0, 1, 0, 0],
        ],
        dtype=np.int16,
    )
    log_z = np.zeros((4, 256), dtype=np.float64)
    for index in range(4):
        log_z[index, index] = 3.0
    row_max = np.max(log_z, axis=1, keepdims=True)
    parent_given_key = np.exp(log_z - row_max)
    parent_given_key /= np.sum(parent_given_key, axis=1, keepdims=True)
    log_z_bar = row_max[:, 0] + np.log(np.mean(np.exp(log_z - row_max), axis=1))
    replicate_keys = [
        keys[[0, 0, 1]],
        keys[[1, 2]],
        keys[[2, 3]],
        keys[[0, 3]],
    ]
    weights = [
        np.array([0.2, 0.3, 0.5], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
        np.array([0.25, 0.75], dtype=np.float64),
        np.array([0.5, 0.5], dtype=np.float64),
    ]
    key_mass = np.array(
        [
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 0.25, 0.75],
            [0.5, 0.0, 0.0, 0.5],
        ],
        dtype=np.float64,
    )
    p_rep = key_mass @ parent_given_key
    log_i = np.array([-1.0, -1.1, -0.9, -1.2], dtype=np.float64)
    pool_weights = np.exp(log_i - np.max(log_i))
    pool_weights /= np.sum(pool_weights)
    return {
        "keys": keys,
        "log_z": log_z,
        "log_z_bar": log_z_bar,
        "replicate_keys": replicate_keys,
        "weights": weights,
        "p_rep": p_rep,
        "log_i": log_i,
        "p_pool": pool_weights @ p_rep,
    }


def _fake_git(expected: str, *, head: bytes | None = None, tracking: bytes | None = None, status: bytes = b""):
    outputs = {
        wrapper.GIT_COMMANDS[0]: (head if head is not None else expected.encode()) + b"\n",
        wrapper.GIT_COMMANDS[1]: (tracking if tracking is not None else expected.encode()) + b"\n",
        wrapper.GIT_COMMANDS[2]: status,
    }

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert tuple(argv) in wrapper.GIT_COMMANDS
        assert kwargs == {
            "cwd": kwargs["cwd"],
            "env": wrapper.GIT_ENVIRONMENT,
            "shell": False,
            "check": True,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "timeout": 30,
        }
        return subprocess.CompletedProcess(argv, 0, outputs[tuple(argv)], b"")

    return run


@pytest.fixture
def sealed_fixture(tmp_path: Path) -> Iterator[dict[str, Any]]:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    state = tmp_path / "state"
    receipt = tmp_path / "receipt"
    for path in (repo, data, cache, state, receipt):
        path.mkdir()
    code_pins: dict[str, dict[str, str]] = {}
    for index, label in enumerate(("analysis_design", "implementation_result_record", "postmortem_record", "pure_analysis_source", "pure_analysis_test")):
        path = repo / f"pin_{index}.txt"
        path.write_text(f"{label}\n")
        path.chmod(0o644)
        code_pins[label] = {
            "path": path.name,
            "mode": "0644",
            "sha256": _digest(path),
        }

    arrays = _analysis_arrays()
    common = {
        "status": wrapper.EXPECTED_SCIENCE_STATUS,
        "outcome_kind": "scientific_fail",
        "failure_class": wrapper.EXPECTED_FAILURE_CLASS,
        "failed_channels": ["replicate_parent_probability_L1"],
        "pre_CF4_metrics": {"metric": 1.0},
        "CF4_metrics": {"metric": 2.0},
        "CF4_gates": {"gate": True},
    }
    _json(data / "result.json", common)
    manifest = {
        "status": "complete_immutable_production_manifest",
        "science_status": common["status"],
        "outcome_kind": common["outcome_kind"],
        "failure_class": common["failure_class"],
        "failed_channels": common["failed_channels"],
    }
    _json(data / "manifest.json", manifest)
    _json(data / "sealed_oracle_control_summary.json", {"status": "complete"})
    for index, (replicate_keys, weights) in enumerate(zip(arrays["replicate_keys"], arrays["weights"], strict=True)):
        indices = np.array(
            [np.flatnonzero(np.all(arrays["keys"] == row, axis=1))[0] for row in replicate_keys]
        )
        np.savez(
            data / f"replicate_{index}.npz",
            master_seed=np.asarray(wrapper.EXPECTED_MASTER_SEEDS[index], dtype=np.int64),
            keys=replicate_keys,
            weights=weights,
            log_Z_bar=arrays["log_z_bar"][indices],
            unused=np.asarray([index], dtype=np.int64),
        )
    np.savez(
        data / "terminal_parent_frozen.npz",
        master_seed=np.asarray(wrapper.EXPECTED_MASTER_SEEDS, dtype=np.int64),
        parent_seed=np.arange(3193, 3449, dtype=np.int32),
        log_I_bar=arrays["log_i"],
        P_rep=arrays["p_rep"],
        P_pool=arrays["p_pool"],
    )
    np.savez(data / "post_terminal_cf4_gates.npz", parent_seed=np.arange(256, dtype=np.int32))
    _json(
        data / "post_terminal_cf4_gates.json",
        {
            "status": common["status"],
            "failure_class": common["failure_class"],
            "pre_CF4_metrics": common["pre_CF4_metrics"],
            "CF4_metrics": common["CF4_metrics"],
            "gates": common["CF4_gates"],
        },
    )
    np.savez(
        cache / "shard_000000.npz",
        keys=arrays["keys"],
        log_Z=arrays["log_z"],
        log_Z_bar=arrays["log_z_bar"],
    )
    _json(cache / "manifest.json", {"status": "complete"})
    _json(state / "COMPLETE", {})
    _json(receipt / "COMPLETE", {})
    _json(receipt / "snapshot.json", {"status": "complete"})
    _json(receipt / "release.anchor", {"status": "released"})
    log = tmp_path / "slurm.log"
    log.write_bytes(b"")

    role_paths = {
        "immutable_top_manifest": data / "manifest.json",
        "scientific_result": data / "result.json",
        "sealed_oracle_control": data / "sealed_oracle_control_summary.json",
        **{f"replicate_{index}": data / f"replicate_{index}.npz" for index in range(4)},
        "terminal_parent_posterior": data / "terminal_parent_frozen.npz",
        "post_terminal_CF4_arrays": data / "post_terminal_cf4_gates.npz",
        "post_terminal_CF4_gates": data / "post_terminal_cf4_gates.json",
        "immutable_cache_manifest": cache / "manifest.json",
        "immutable_evidence_cache_shard": cache / "shard_000000.npz",
        "state_complete_marker": state / "COMPLETE",
        "receipt_complete_marker": receipt / "COMPLETE",
        "preexecution_identity_snapshot": receipt / "snapshot.json",
        "hardlink_release_anchor": receipt / "release.anchor",
        "slurm_combined_log": log,
    }
    kinds = {
        role: (
            "NPZ" if path.suffix == ".npz" else "LOG" if role == "slurm_combined_log" else "JSON"
        )
        for role, path in role_paths.items()
    }
    for role, path in role_paths.items():
        path.chmod(0o644 if role == "slurm_combined_log" else 0o444)
    for root in (data, cache, state, receipt):
        root.chmod(0o555)
    manifest_sha = _digest(data / "manifest.json")
    for marker in (state / "COMPLETE", receipt / "COMPLETE"):
        marker.chmod(0o644)
        value = {
            "status": "complete_valid_provenance_and_scientific_postcheck",
            "result_manifest_sha256_or_null": manifest_sha,
        }
        if marker.parent == state:
            value["science_status"] = wrapper.EXPECTED_SCIENCE_STATUS
        _json(marker, value)
        marker.chmod(0o444)

    rows = [
        {
            "path": str(path),
            "role": role,
            "kind": kinds[role],
            "mode": "0644" if role == "slurm_combined_log" else "0444",
            "size_bytes": path.stat().st_size,
            "sha256": _digest(path),
        }
        for role, path in role_paths.items()
    ]
    expected_commit = "a" * 40
    contract = {
        "canonical_paths": {
            "data_root": str(data),
            "cache_root": str(cache),
            "state_root": str(state),
            "receipt_root": str(receipt),
        },
        "artifact_contract": {
            "exact_17_rows": rows,
            "terminal_root_modes": {name: "0555" for name in wrapper.ROOT_NAMES},
            "exact_root_entry_sets": {
                "data": sorted(path.name for path in role_paths.values() if path.parent == data),
                "cache": sorted(path.name for path in role_paths.values() if path.parent == cache),
                "state": ["COMPLETE"],
                "receipt": ["COMPLETE", "release.anchor", "snapshot.json"],
            },
        },
        "immutable_code_contracts": code_pins,
        "git_subprocess_contract": {"expected_HEAD_and_tracking": expected_commit},
    }
    try:
        yield {
            "contract": contract,
            "repo": repo,
            "git": _fake_git(expected_commit),
            "roots": (data, cache, state, receipt),
            "paths": role_paths,
        }
    finally:
        for root in (data, cache, state, receipt):
            if root.exists() and not root.is_symlink():
                root.chmod(0o755)
                for path in root.rglob("*"):
                    if not path.is_symlink():
                        path.chmod(0o644 if path.is_file() else 0o755)


def test_public_entry_refuses_before_file_or_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "open", lambda *args, **kwargs: pytest.fail("file opened"))
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("subprocess run"))
    with pytest.raises(PermissionError, match="not authorized"):
        wrapper.run_canonical_parent_key_overlap_read_only_analysis_wrapper_v1()


def test_valid_fixture_returns_only_immutable_in_memory_result(sealed_fixture: dict[str, Any]) -> None:
    result = wrapper._run_verified_analysis_for_test(
        sealed_fixture["contract"],
        repo_root=sealed_fixture["repo"],
        git_runner=sealed_fixture["git"],
    )
    assert result["maximum_pair"] in result["pair_order"]
    assert result["factorization_max_abs_residual"] <= 1e-12
    with pytest.raises(TypeError):
        result["new"] = 1


def test_npz_snapshot_arrays_are_eager_and_archive_closes_before_final_fstat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "fake.npz"
    path.write_bytes(b"fixed archive bytes")
    path.chmod(0o444)
    arrays = {
        "keys": np.zeros((1, 6), dtype=np.int16),
        "log_Z": np.zeros((1, 256), dtype=np.float64),
        "log_Z_bar": np.zeros(1, dtype=np.float64),
    }

    class Archive:
        files = list(arrays)
        closed = False
        accessed: list[str] = []

        def __getitem__(self, name: str) -> np.ndarray:
            self.accessed.append(name)
            return arrays[name]

        def close(self) -> None:
            self.closed = True

    archive = Archive()
    monkeypatch.setattr(wrapper.np, "load", lambda stream, allow_pickle: archive)
    original_fstat = wrapper.os.fstat
    calls = 0

    def checked_fstat(fd: int):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert archive.closed
            assert archive.accessed == ["keys", "log_Z", "log_Z_bar"]
        return original_fstat(fd)

    monkeypatch.setattr(wrapper.os, "fstat", checked_fstat)
    row = {
        "path": str(path),
        "role": "immutable_evidence_cache_shard",
        "kind": "NPZ",
        "mode": "0444",
        "size_bytes": path.stat().st_size,
        "sha256": _digest(path),
    }
    loaded = wrapper._stable_artifact_read(row, materialize=True)
    assert set(loaded.value) == set(arrays)
    assert all(not value.flags.writeable for value in loaded.value.values())


def test_retained_writable_fd_npz_mutation_and_restore_cannot_change_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "artifact.npz"
    alternate = tmp_path / "alternate.npz"
    original_arrays = {
        "keys": np.zeros((1, 6), dtype=np.int16),
        "log_Z": np.zeros((1, 256), dtype=np.float64),
        "log_Z_bar": np.zeros(1, dtype=np.float64),
    }
    changed_arrays = {
        **original_arrays,
        "log_Z": np.full((1, 256), 7.0, dtype=np.float64),
    }
    np.savez(path, **original_arrays)
    np.savez(alternate, **changed_arrays)
    original_bytes = path.read_bytes()
    alternate_bytes = alternate.read_bytes()
    assert len(original_bytes) == len(alternate_bytes)
    writable = os.open(path, os.O_RDWR)
    path.chmod(0o444)
    real_load = wrapper.np.load

    class RestoringArchive:
        def __init__(self, archive: Any) -> None:
            self.archive = archive
            self.files = archive.files

        def __getitem__(self, name: str) -> np.ndarray:
            return self.archive[name]

        def close(self) -> None:
            self.archive.close()
            os.pwrite(writable, original_bytes, 0)
            os.ftruncate(writable, len(original_bytes))
            os.fsync(writable)

    def attack(snapshot: Any, *, allow_pickle: bool):
        os.pwrite(writable, alternate_bytes, 0)
        os.ftruncate(writable, len(alternate_bytes))
        os.fsync(writable)
        return RestoringArchive(real_load(snapshot, allow_pickle=allow_pickle))

    monkeypatch.setattr(wrapper.np, "load", attack)
    row = {
        "path": str(path),
        "role": "immutable_evidence_cache_shard",
        "kind": "NPZ",
        "mode": "0444",
        "size_bytes": len(original_bytes),
        "sha256": hashlib.sha256(original_bytes).hexdigest(),
    }
    try:
        loaded = wrapper._stable_artifact_read(row, materialize=True)
        assert np.array_equal(loaded.value["log_Z"], original_arrays["log_Z"])
        assert not np.array_equal(loaded.value["log_Z"], changed_arrays["log_Z"])
        assert path.read_bytes() == original_bytes
    finally:
        os.close(writable)
        path.chmod(0o644)


def test_same_bytes_inode_replacement_is_rejected_and_fixture_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "artifact.log"
    path.write_bytes(b"unchanged")
    path.chmod(0o644)
    row = {
        "path": str(path), "role": "slurm_combined_log", "kind": "LOG",
        "mode": "0644", "size_bytes": path.stat().st_size, "sha256": _digest(path),
    }
    original = wrapper._read_fd_and_hash
    backup = tmp_path / "original"

    def replace(fd: int, *, retain_bytes: bool):
        result = original(fd, retain_bytes=retain_bytes)
        path.rename(backup)
        path.write_bytes(b"unchanged")
        path.chmod(0o644)
        return result

    monkeypatch.setattr(wrapper, "_read_fd_and_hash", replace)
    try:
        with pytest.raises(wrapper.CanonicalReadContractError, match="unstable"):
            wrapper._stable_artifact_read(row, materialize=False)
    finally:
        path.unlink(missing_ok=True)
        backup.rename(path)


@pytest.mark.parametrize("intermediate", [False, True])
def test_final_and_intermediate_symlinks_are_rejected(tmp_path: Path, intermediate: bool) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    target = actual / "value.log"
    target.write_bytes(b"x")
    target.chmod(0o644)
    if intermediate:
        linked = tmp_path / "linked"
        linked.symlink_to(actual, target_is_directory=True)
        path = linked / "value.log"
    else:
        path = tmp_path / "value.log"
        path.symlink_to(target)
    row = {
        "path": str(path), "role": "slurm_combined_log", "kind": "LOG",
        "mode": "0644", "size_bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest(),
    }
    with pytest.raises(wrapper.CanonicalReadContractError, match="symlink"):
        wrapper._stable_artifact_read(row, materialize=False)


def test_fstat_instability_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "artifact.log"
    path.write_bytes(b"x")
    path.chmod(0o644)
    row = {
        "path": str(path), "role": "slurm_combined_log", "kind": "LOG",
        "mode": "0644", "size_bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest(),
    }
    original = wrapper.os.fstat
    calls = 0

    def unstable(fd: int):
        nonlocal calls
        calls += 1
        value = original(fd)
        if calls == 2:
            return type("ChangedStat", (), {
                "st_dev": value.st_dev, "st_ino": value.st_ino,
                "st_size": value.st_size + 1, "st_mode": value.st_mode,
            })()
        return value

    monkeypatch.setattr(wrapper.os, "fstat", unstable)
    with pytest.raises(wrapper.CanonicalReadContractError, match="unstable"):
        wrapper._stable_artifact_read(row, materialize=False)


@pytest.mark.parametrize("failure", ["head", "tracking", "dirty"])
def test_git_head_tracking_and_clean_scope_are_fail_closed(
    sealed_fixture: dict[str, Any], failure: str
) -> None:
    expected = sealed_fixture["contract"]["git_subprocess_contract"]["expected_HEAD_and_tracking"]
    options: dict[str, Any] = {}
    if failure == "head":
        options["head"] = b"b" * 40
    elif failure == "tracking":
        options["tracking"] = b"b" * 40
    else:
        options["status"] = b"?? src/shadow.py\0"
    with pytest.raises(wrapper.CanonicalReadContractError, match="Git"):
        wrapper._validate_git(
            sealed_fixture["contract"],
            repo_root=sealed_fixture["repo"],
            runner=_fake_git(expected, **options),
        )


def test_non_whitelisted_subprocess_is_refused_without_runner_call(tmp_path: Path) -> None:
    called = False

    def forbidden(*args: Any, **kwargs: Any):
        nonlocal called
        called = True
        raise AssertionError

    with pytest.raises(wrapper.CanonicalReadContractError, match="non-whitelisted"):
        wrapper._run_whitelisted_git(
            ("/usr/bin/git", "ls-remote", "origin"), cwd=tmp_path, runner=forbidden
        )
    assert called is False


def test_post_analysis_extra_entry_is_rejected(sealed_fixture: dict[str, Any]) -> None:
    data = sealed_fixture["roots"][0]

    def mutate(**kwargs: Any):
        data.chmod(0o755)
        (data / "unexpected").write_bytes(b"x")
        data.chmod(0o555)
        return {"untrusted": True}

    with pytest.raises(wrapper.CanonicalReadContractError, match="entry set"):
        wrapper._run_verified_analysis_for_test(
            sealed_fixture["contract"],
            repo_root=sealed_fixture["repo"],
            git_runner=sealed_fixture["git"],
            analyzer=mutate,
        )


def test_json_science_binding_mismatch_is_rejected(sealed_fixture: dict[str, Any]) -> None:
    path = sealed_fixture["paths"]["scientific_result"]
    data = sealed_fixture["roots"][0]
    data.chmod(0o755)
    path.chmod(0o644)
    value = json.loads(path.read_text())
    value["failure_class"] = "wrong"
    _json(path, value)
    path.chmod(0o444)
    data.chmod(0o555)
    row = next(
        row for row in sealed_fixture["contract"]["artifact_contract"]["exact_17_rows"]
        if row["role"] == "scientific_result"
    )
    row["size_bytes"] = path.stat().st_size
    row["sha256"] = _digest(path)
    with pytest.raises(wrapper.CanonicalReadContractError, match="semantic binding"):
        wrapper._run_verified_analysis_for_test(
            sealed_fixture["contract"],
            repo_root=sealed_fixture["repo"],
            git_runner=sealed_fixture["git"],
        )
