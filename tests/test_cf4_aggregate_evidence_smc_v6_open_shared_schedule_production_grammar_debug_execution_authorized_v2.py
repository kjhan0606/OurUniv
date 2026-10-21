from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import cf4_aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_execution_authorized_v2 as authorized


def _write(path: Path, value: dict, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(authorized.canonical_json(value) + b"\n")
    path.chmod(mode)


def _scheduler(tmp_path: Path, job_id: str = "731") -> dict:
    log = tmp_path / f"slurm-{job_id}.log"
    log.write_bytes(b"scheduler-prefix\n")
    checkpoint = authorized._scheduler_log_checkpoint(log)
    return {
        "SLURM_JOB_ID": job_id,
        "partition": "debug",
        "node": "grammar-debug",
        "nodes": 1,
        "ntasks": 1,
        "cpus_per_task": 12,
        "memory_MiB": 98304,
        "submit_time": "2026-08-21T12:00:00",
        "start_time": "2026-08-21T12:01:00",
        "command": str(authorized.RUNNER), "workdir": str(authorized.ROOT),
        "stdout_path": str(log), "stderr_path": str(log),
        "time_limit": "12:00:00", "requeue": "0",
        "minimum_memory_node": "96G",
        "req_tres_raw": "cpu=12,mem=96G,node=1,billing=12",
        "req_tres_parsed": {"cpu": "12", "mem": "96G", "node": "1", "billing": "12"},
        "alloc_tres_raw": "cpu=12,node=1,billing=12",
        "alloc_tres_parsed": {"cpu": "12", "node": "1", "billing": "12"},
        "cgroup_version": "v2", "cgroup_controller": "unified",
        "cgroup_value_path": "/sys/fs/cgroup/job/memory.max",
        "cgroup_value_raw": "103079215104",
        "combined_log": checkpoint,
    }


def _authorization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, tuple[Path, ...]]:
    program = authorized.load_program()
    grant = tmp_path / "grant.json"
    release_path = tmp_path / "release.json"
    manifest_path = tmp_path / "manifest.json"
    receipts = tmp_path / "receipts"
    cache = tmp_path / "cache"
    data = tmp_path / "data"
    state = tmp_path / "state"
    log_template = tmp_path / "slurm-%j.log"
    for name, value in (
        ("GRANT", grant), ("RELEASE", release_path),
        ("EXTERNAL_MANIFEST", manifest_path), ("RECEIPT_ROOT", receipts),
        ("CACHE_ROOT", cache), ("DATA_ROOT", data), ("STATE_ROOT", state),
        ("SLURM_LOG_TEMPLATE", log_template),
    ):
        monkeypatch.setattr(authorized, name, value)

    contract = program["pair_grant_contract"]
    release_id, manifest_id, grant_id = "1" * 64, "2" * 64, "3" * 64
    implementation_commit = "4" * 40
    result_sha = "5" * 64
    implementation_map = {path: hashlib.sha256(path.encode()).hexdigest() for path in authorized.IMPLEMENTATION_FILES}
    payload = {
        "schema": contract["payload_schema"],
        "status": contract["payload_status"],
        "release_id": release_id,
        "design_commit": authorized.WRAPPER_DESIGN_COMMIT,
        "design_sha256": authorized.WRAPPER_DESIGN_SHA256,
        "implementation_commit": implementation_commit,
        "implementation_result_record_sha256": result_sha,
        "implementation_file_sha256_map": implementation_map,
        "fixed_science_digest": contract["fixed_science_digest"],
        "canonical_paths_digest": contract["canonical_paths_digest"],
        "resource_contract_digest": contract["resource_contract_digest"],
        "one_shot": True,
        "authorization": contract["future_runtime_authorization_exact"],
    }
    payload_sha = hashlib.sha256(authorized.canonical_json(payload)).hexdigest()
    manifest = {
        "schema": contract["manifest_schema"],
        "status": contract["manifest_status"],
        "manifest_id": manifest_id,
        "release_id": release_id,
        "release_payload_sha256": payload_sha,
        "design_sha256": authorized.WRAPPER_DESIGN_SHA256,
        "implementation_result_record_sha256": result_sha,
        "canonical_paths_digest": contract["canonical_paths_digest"],
        "resource_contract_digest": contract["resource_contract_digest"],
        "release_path": str(release_path),
        "one_shot": True,
    }
    _write(manifest_path, manifest, 0o444)
    release = {
        "schema": contract["release_schema"],
        "status": contract["release_status"],
        "release_id": release_id,
        "payload": payload,
        "payload_sha256": payload_sha,
        "manifest_id": manifest_id,
        "manifest_path": str(manifest_path),
        "manifest_sha256": authorized.sha256_file(manifest_path),
    }
    _write(release_path, release, 0o444)
    grant_value = {
        "schema": contract["grant_schema"],
        "status": contract["grant_status"],
        "grant_id": grant_id,
        "one_shot": True,
        "design_commit": authorized.WRAPPER_DESIGN_COMMIT,
        "design_sha256": authorized.WRAPPER_DESIGN_SHA256,
        "implementation_commit": implementation_commit,
        "implementation_result_record_path": authorized.WRAPPER_RESULT_RECORD_RELATIVE,
        "implementation_result_record_sha256": result_sha,
        "implementation_file_sha256_map": implementation_map,
        "fixed_science_digest": contract["fixed_science_digest"],
        "canonical_paths_digest": contract["canonical_paths_digest"],
        "resource_contract_digest": contract["resource_contract_digest"],
        "release_path": str(release_path),
        "release_id": release_id,
        "release_payload_sha256": payload_sha,
        "release_sha256": authorized.sha256_file(release_path),
        "manifest_path": str(manifest_path),
        "manifest_id": manifest_id,
        "manifest_sha256": authorized.sha256_file(manifest_path),
        "receipt_root": str(receipts),
        "cache_root": str(cache),
        "data_root": str(data),
        "state_root": str(state),
        "slurm_combined_log_template": str(log_template),
        "authorization": contract["future_runtime_authorization_exact"],
    }
    _write(grant, grant_value, 0o644)
    lineage = {
        "grant_commit": "6" * 40,
        "implementation_commit": implementation_commit,
        "implementation_result_record_sha256": result_sha,
        "implementation_file_sha256_map": implementation_map,
    }
    monkeypatch.setattr(authorized, "load_program", lambda: program)
    monkeypatch.setattr(authorized, "_validate_grant_git_lineage", lambda _program: lineage)
    result = authorized.validate_authorization(program)
    return program, result, (grant, release_path, manifest_path, receipts, cache, data, state)


def _fake_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, science_status: str,
) -> tuple[dict, tuple[Path, ...], dict]:
    _program, authorization, paths = _authorization(tmp_path, monkeypatch)
    scheduler = _scheduler(tmp_path)
    monkeypatch.setattr(authorized, "validate_authorization", lambda _program: authorization)
    monkeypatch.setattr(authorized, "_read_scheduler_context", lambda: scheduler)
    monkeypatch.setattr(authorized.os, "uname", lambda: SimpleNamespace(nodename="Grammar-Debug.cluster"))
    monkeypatch.setattr(authorized, "_require_resources", lambda _scheduler: None)
    monkeypatch.setattr(authorized, "_require_runtime_environment", lambda: None)
    monkeypatch.setattr(authorized.execution, "load_canonical_program", lambda **_kwargs: {})
    monkeypatch.setattr(authorized.capability, "load_frozen_contract", lambda: object())

    def core(_program, _contract, data: Path, _cache: Path) -> dict:
        _write(data / "manifest.json", {"schema": "fake-manifest"}, 0o444)
        _write(data / "result.json", {"status": science_status}, 0o444)
        return {"status": science_status}

    monkeypatch.setattr(authorized.execution, "_execute_reserved_canonical_private", core)
    monkeypatch.setattr(
        authorized, "read_only_postcheck",
        lambda _data: {"status": science_status, "valid_scientific_complete": True},
    )
    return authorization, paths, scheduler


def test_program_is_closed_and_public_refuses_before_any_runtime_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = authorized.load_program()
    assert set(program["authorization"]) == authorized.PROGRAM_AUTHORIZATION_KEYS
    assert not any(program["authorization"].values())
    roots = tuple(tmp_path / name for name in ("receipts", "cache", "data", "state"))
    for name, value in zip(("RECEIPT_ROOT", "CACHE_ROOT", "DATA_ROOT", "STATE_ROOT"), roots):
        monkeypatch.setattr(authorized, name, value)
    monkeypatch.setattr(authorized, "GRANT", tmp_path / "absent-grant.json")
    monkeypatch.setattr(authorized, "load_program", lambda: program)
    with pytest.raises(PermissionError):
        authorized.run_authorized_production()
    assert all(not os.path.lexists(path) for path in roots)


def test_pair_grant_exact_binding_and_mutation_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    program, result, paths = _authorization(tmp_path, monkeypatch)
    assert result["grant"]["grant_id"] == "3" * 64
    manifest = json.loads(paths[2].read_text())
    manifest["release_id"] = "9" * 64
    paths[2].chmod(0o644)
    _write(paths[2], manifest, 0o444)
    with pytest.raises(PermissionError):
        authorized.validate_authorization(program)


def test_scheduler_context_is_exact_and_allocation_mutations_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = tmp_path / "slurm-%j.log"
    log = tmp_path / "slurm-731.log"
    log.write_text("start\n")
    monkeypatch.setattr(authorized, "SLURM_LOG_TEMPLATE", template)
    values = {
        "SLURM_JOB_ID": "731", "SLURM_JOB_PARTITION": "debug",
        "SLURMD_NODENAME": "grammar-debug", "SLURM_JOB_NODELIST": "grammar-debug",
        "SLURM_NNODES": "1", "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "12", "SLURM_MEM_PER_NODE": "98304",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    details = {
        "Partition": "debug", "NodeList": "grammar-debug", "BatchHost": "grammar-debug",
        "NumNodes": "1", "NumCPUs": "12", "NumTasks": "1", "CPUs/Task": "12",
        "Command": str(authorized.RUNNER), "WorkDir": str(authorized.ROOT),
        "StdOut": str(log), "StdErr": str(log), "TimeLimit": "12:00:00",
        "Requeue": "0", "BatchFlag": "1", "MinMemoryNode": "96G",
        "ReqTRES": "cpu=12,mem=96G,node=1,billing=12",
        "AllocTRES": "cpu=12,node=1,billing=12",
        "SubmitTime": "2026-08-21T12:00:00",
        "StartTime": "2026-08-21T12:01:00",
    }
    monkeypatch.setattr(authorized, "_scontrol_job", lambda _job: details)
    monkeypatch.setattr(authorized, "_resolve_memory_cgroup", lambda: {
        "cgroup_version": "v2", "cgroup_controller": "unified",
        "cgroup_value_path": "/sys/fs/cgroup/job/memory.max",
        "cgroup_value_raw": "103079215104",
    })
    observed = authorized._read_scheduler_context()
    assert observed["memory_MiB"] == 98304
    assert observed["req_tres_raw"] == "cpu=12,mem=96G,node=1,billing=12"
    assert observed["req_tres_parsed"] == {
        "cpu": "12", "mem": "96G", "node": "1", "billing": "12",
    }
    assert observed["alloc_tres_parsed"] == {
        "cpu": "12", "node": "1", "billing": "12",
    }
    for key, bad in (
        ("SLURM_JOB_PARTITION", "batch"), ("SLURMD_NODENAME", "syn101"),
        ("SLURM_JOB_NODELIST", "grammar-debug,grammar"), ("SLURM_NNODES", "2"),
        ("SLURM_NTASKS", "2"), ("SLURM_CPUS_PER_TASK", "11"),
        ("SLURM_MEM_PER_NODE", "98303"),
    ):
        monkeypatch.setenv(key, bad)
        with pytest.raises(PermissionError):
            authorized._read_scheduler_context()
        monkeypatch.setenv(key, values[key])
    details["NumCPUs"] = "11"
    with pytest.raises(PermissionError):
        authorized._read_scheduler_context()
    details["NumCPUs"] = "12"
    for key, bad in (
        ("Command", "/tmp/other.sbatch"), ("WorkDir", "/tmp"),
        ("StdOut", str(tmp_path / "other.log")), ("StdErr", str(tmp_path / "other.log")),
        ("TimeLimit", "11:59:59"), ("Requeue", "1"),
        ("MinMemoryNode", "95G"),
        ("ReqTRES", "cpu=12,mem=95G,node=1,billing=12"),
        ("AllocTRES", "cpu=11,node=1,billing=12"),
    ):
        old = details[key]
        details[key] = bad
        with pytest.raises(PermissionError):
            authorized._read_scheduler_context()
        details[key] = old


def test_tres_parser_rejects_standalone_duplicate_unknown_and_malformed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = tmp_path / "slurm-%j.log"
    log = tmp_path / "slurm-731.log"
    log.write_text("start\n")
    monkeypatch.setattr(authorized, "SLURM_LOG_TEMPLATE", template)
    for key, value in {
        "SLURM_JOB_ID": "731", "SLURM_JOB_PARTITION": "debug",
        "SLURMD_NODENAME": "grammar-debug", "SLURM_JOB_NODELIST": "grammar-debug",
        "SLURM_NNODES": "1", "SLURM_NTASKS": "1", "SLURM_CPUS_PER_TASK": "12",
        "SLURM_MEM_PER_NODE": "98304",
    }.items():
        monkeypatch.setenv(key, value)
    details = {
        "Partition": "debug", "NodeList": "grammar-debug", "BatchHost": "grammar-debug",
        "NumNodes": "1", "NumCPUs": "12", "NumTasks": "1", "CPUs/Task": "12",
        "Command": str(authorized.RUNNER), "WorkDir": str(authorized.ROOT),
        "StdOut": str(log), "StdErr": str(log), "TimeLimit": "12:00:00",
        "Requeue": "0", "BatchFlag": "1", "MinMemoryNode": "98304M",
        "ReqTRES": "cpu=12,mem=96G,node=1,billing=12",
        "AllocTRES": "cpu=12,mem=96G,node=1,billing=12",
        "SubmitTime": "2026-08-21T12:00:00", "StartTime": "2026-08-21T12:01:00",
    }
    monkeypatch.setattr(authorized, "_scontrol_job", lambda _job: details)
    monkeypatch.setattr(authorized, "_resolve_memory_cgroup", lambda: {
        "cgroup_version": "v2", "cgroup_controller": "unified",
        "cgroup_value_path": "/cg/job/memory.max", "cgroup_value_raw": "max",
    })
    assert authorized._read_scheduler_context()["alloc_tres_parsed"]["mem"] == "96G"
    for field, bad in (
        ("ReqTRES", ""),
        ("ReqTRES", "cpu=12,mem=96G,node=1,billing=12,cpu=12"),
        ("ReqTRES", "cpu=12,mem=96G,node=1,billing=12,gpu=0"),
        ("ReqTRES", "cpu=12,,mem=96G,node=1,billing=12"),
        ("AllocTRES", "cpu=12,node=1,billing=12,gres/gpu=0"),
    ):
        old = details[field]
        details[field] = bad
        with pytest.raises(PermissionError):
            authorized._read_scheduler_context()
        details[field] = old
    details.pop("ReqTRES")
    details["TRES"] = "cpu=12,mem=96G,node=1,billing=12"
    with pytest.raises(PermissionError):
        authorized._read_scheduler_context()


def test_cgroup_v2_and_v1_current_process_resolution_and_symlink_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    membership = tmp_path / "self.cgroup"
    mountinfo = tmp_path / "self.mountinfo"
    mount = tmp_path / "cgroup2"
    target = mount / "jobs" / "731" / "memory.max"
    target.parent.mkdir(parents=True)
    target.write_text("103079215104\n")
    membership.write_text("0::/jobs/731\n")
    mountinfo.write_text(f"29 23 0:26 / {mount} rw - cgroup2 cgroup rw\n")
    monkeypatch.setattr(authorized, "PROC_SELF_CGROUP", membership)
    monkeypatch.setattr(authorized, "PROC_SELF_MOUNTINFO", mountinfo)
    resolved = authorized._resolve_memory_cgroup()
    assert resolved == {
        "cgroup_version": "v2", "cgroup_controller": "unified",
        "cgroup_value_path": str(target), "cgroup_value_raw": "103079215104",
    }
    target.unlink()
    target.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(PermissionError, match="symlink"):
        authorized._resolve_memory_cgroup()

    v1_mount = tmp_path / "memory"
    v1_target = v1_mount / "slurm" / "731" / "memory.limit_in_bytes"
    v1_target.parent.mkdir(parents=True)
    v1_target.write_text(str(1 << 60) + "\n")
    membership.write_text("5:memory,cpu:/slurm/731\n")
    mountinfo.write_text(f"31 23 0:28 / {v1_mount} rw - cgroup cgroup rw,memory\n")
    assert authorized._resolve_memory_cgroup() == {
        "cgroup_version": "v1", "cgroup_controller": "memory",
        "cgroup_value_path": str(v1_target), "cgroup_value_raw": str(1 << 60),
    }


def test_unlimited_cgroup_requires_exact_scheduler_and_memavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = _scheduler(tmp_path)
    scheduler.update({
        "cgroup_version": "v2", "cgroup_controller": "unified",
        "cgroup_value_path": "/cg/job/memory.max", "cgroup_value_raw": "max",
    })
    monkeypatch.setattr(authorized, "_resolve_memory_cgroup", lambda: {
        key: scheduler[key] for key in (
            "cgroup_version", "cgroup_controller", "cgroup_value_path", "cgroup_value_raw",
        )
    })
    original_read_text = Path.read_text

    def read_text(path: Path, *args, **kwargs) -> str:
        if path == Path("/proc/meminfo"):
            return "MemAvailable: 104857600 kB\n"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(authorized.shutil, "disk_usage", lambda _path: SimpleNamespace(free=50 * 1024**3))
    monkeypatch.setattr(authorized.os, "sched_getaffinity", lambda _pid: set(range(12)))
    monkeypatch.setattr(authorized.os, "cpu_count", lambda: 12)
    authorized._require_resources(scheduler)
    scheduler["memory_MiB"] = 98303
    with pytest.raises(PermissionError, match="unlimited"):
        authorized._require_resources(scheduler)


def test_scheduler_log_append_allowed_but_mutations_are_rejected(
    tmp_path: Path,
) -> None:
    log = tmp_path / "scheduler.log"
    log.write_bytes(b"sealed-prefix")
    record = authorized._scheduler_log_checkpoint(log)
    with log.open("ab") as stream:
        stream.write(b"-append")
    authorized._revalidate_scheduler_log(record)
    with log.open("r+b") as stream:
        stream.write(b"X")
    with pytest.raises(PermissionError):
        authorized._revalidate_scheduler_log(record)
    log.unlink()
    log.write_bytes(b"sealed-prefix-append")
    with pytest.raises(PermissionError):
        authorized._revalidate_scheduler_log(record)
    link = tmp_path / "link.log"
    link.symlink_to(log)
    with pytest.raises(PermissionError):
        authorized._scheduler_log_checkpoint(link)
    short = tmp_path / "short.log"
    short.write_bytes(b"abcdef")
    short_record = authorized._scheduler_log_checkpoint(short)
    short.write_bytes(b"a")
    with pytest.raises(PermissionError):
        authorized._revalidate_scheduler_log(short_record)


def test_receipt_binds_scheduler_prefix_and_rejects_inode_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, authorization, paths = _authorization(tmp_path, monkeypatch)
    scheduler = _scheduler(tmp_path)
    monkeypatch.setattr(authorized, "_read_scheduler_context", lambda: scheduler)
    receipt, _snapshot, snapshot_sha = authorized.create_receipt(authorization, scheduler)
    authorized.revalidate_receipt(receipt, snapshot_sha, scheduler)
    log = Path(scheduler["combined_log"]["path"])
    with log.open("ab") as stream:
        stream.write(b"allowed append\n")
    authorized.revalidate_receipt(receipt, snapshot_sha, scheduler)
    scheduler["time_limit"] = "11:59:59"
    with pytest.raises(PermissionError):
        authorized.revalidate_receipt(receipt, snapshot_sha, scheduler)
    scheduler["time_limit"] = "12:00:00"
    receipt.chmod(0o755)
    with pytest.raises(PermissionError, match="mode"):
        authorized.revalidate_receipt(receipt, snapshot_sha, scheduler)
    receipt.chmod(0o700)
    original = log.read_bytes()
    log.unlink()
    log.write_bytes(original)
    with pytest.raises(PermissionError):
        authorized.revalidate_receipt(receipt, snapshot_sha, scheduler)
    authorized._receipt_failed(
        receipt, authorization, "scheduler_log_replaced", snapshot_sha, scheduler,
    )
    assert (receipt / "FAILED").is_file()
    assert not (receipt / "RUNNING").exists()
    assert paths[3].is_dir()


@pytest.mark.parametrize(
    "checkpoint",
    ["after_receipt_mkdir", "after_release_anchor_link", "after_snapshot_seal", "after_RUNNING_seal"],
)
def test_receipt_bootstrap_interruptions_are_forensically_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint: str,
) -> None:
    case = tmp_path / checkpoint
    case.mkdir()
    _, authorization, _ = _authorization(case, monkeypatch)
    scheduler = _scheduler(case)
    monkeypatch.setattr(authorized, "_read_scheduler_context", lambda: scheduler)

    def interrupt(name: str) -> None:
        if name == checkpoint:
            raise InterruptedError(name)

    monkeypatch.setattr(authorized, "_receipt_checkpoint", interrupt)
    with pytest.raises(InterruptedError):
        authorized.create_receipt(authorization, scheduler)
    receipt = authorized.canonical_receipt_path(authorization["grant"]["grant_id"])
    assert receipt.is_dir()
    assert (receipt / "FAILED").is_file()
    assert not (receipt / "RUNNING").exists()


def test_runtime_reservation_order_and_dangling_roots_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, cache, data = (tmp_path / name for name in ("state", "cache", "data"))
    monkeypatch.setattr(authorized, "STATE_ROOT", state)
    monkeypatch.setattr(authorized, "CACHE_ROOT", cache)
    monkeypatch.setattr(authorized, "DATA_ROOT", data)
    authorized._reserve_runtime()
    assert state.is_dir() and cache.is_dir() and data.is_dir()
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing")
    monkeypatch.setattr(authorized, "DATA_ROOT", dangling)
    assert os.path.lexists(authorized.DATA_ROOT)


def test_host_and_runtime_environment_are_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(authorized.os, "uname", lambda: type("U", (), {"nodename": "Grammar-Debug.cluster"})())
    assert authorized._host_short_ascii_lower() == "grammar-debug"
    wanted = {
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "MALLOC_ARENA_MAX": "2",
        "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1", "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key, value in wanted.items():
        monkeypatch.setenv(key, value)
    for key in ("BASH_ENV", "ENV", "PYTHONSTARTUP", "LD_PRELOAD"):
        monkeypatch.delenv(key, raising=False)
    authorized._require_runtime_environment()
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(PermissionError):
        authorized._require_runtime_environment()
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("BASH_ENV", "/tmp/shadow")
    with pytest.raises(PermissionError, match="inherited"):
        authorized._require_runtime_environment()


def test_untracked_shadow_and_import_origin_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(authorized, "_git", lambda *_args: "?? scripts/tripwire/owned.sh\0")
    authorized._require_science_worktree_clean()
    monkeypatch.setattr(authorized, "_git", lambda *_args: "?? src/cf4_aggregate_evidence_smc.py\0")
    with pytest.raises(PermissionError, match="shadow"):
        authorized._require_science_worktree_clean()
    program = authorized.load_program()
    shadow = tmp_path / "shadow.py"
    shadow.write_text("# shadow\n")
    monkeypatch.setattr(authorized.capability, "__file__", str(shadow))
    with pytest.raises(PermissionError, match="import origin"):
        authorized._require_import_origins(program)


@pytest.mark.parametrize("science_status", sorted(authorized.SCIENTIFIC_STATUSES))
def test_public_fake_science_pass_and_scientific_fail_complete_exact_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, science_status: str,
) -> None:
    authorization, paths, _scheduler_value = _fake_run(tmp_path, monkeypatch, science_status)
    result = authorized.run_authorized_production()
    receipt = authorized.canonical_receipt_path(authorization["grant"]["grant_id"])
    assert result["status"] == science_status
    assert {item.name for item in receipt.iterdir()} == {"release.anchor", "snapshot.json", "COMPLETE"}
    assert (paths[6] / "COMPLETE").is_file() and not (paths[6] / "RUNNING").exists()
    assert authorized._read_only_complete_status()["science_status"] == science_status


def test_public_invalid_postcheck_and_core_exception_route_both_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization, paths, _scheduler_value = _fake_run(
        tmp_path, monkeypatch, "complete_pass_production_smc",
    )
    monkeypatch.setattr(
        authorized.execution, "_execute_reserved_canonical_private",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("invalid core")),
    )
    with pytest.raises(RuntimeError, match="invalid core"):
        authorized.run_authorized_production()
    receipt = authorized.canonical_receipt_path(authorization["grant"]["grant_id"])
    assert (receipt / "FAILED").is_file() and not (receipt / "RUNNING").exists()
    assert (paths[6] / "FAILED").is_file() and not (paths[6] / "RUNNING").exists()
    assert authorized._read_only_failed_status()["status"] == "failed"


@pytest.mark.parametrize("failed_root", ["state", "cache", "data"])
def test_each_runtime_reservation_failure_has_no_orphan_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_root: str,
) -> None:
    authorization, paths, _scheduler_value = _fake_run(
        tmp_path, monkeypatch, "complete_pass_production_smc",
    )
    targets = {"state": paths[6], "cache": paths[4], "data": paths[5]}
    original = Path.mkdir

    def fail_selected(path: Path, *args, **kwargs):
        if path == targets[failed_root]:
            raise OSError(f"injected {failed_root} reservation")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_selected)
    with pytest.raises(PermissionError, match="runtime reservation failed"):
        authorized.run_authorized_production()
    receipt = authorized.canonical_receipt_path(authorization["grant"]["grant_id"])
    assert (receipt / "FAILED").is_file() and not (receipt / "RUNNING").exists()
    if failed_root != "state":
        assert (paths[6] / "FAILED").is_file()


@pytest.mark.parametrize("exit_code", [124, 137, 143])
def test_timeout_and_term_supervisor_seal_failed_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int,
) -> None:
    _program, authorization, paths = _authorization(tmp_path, monkeypatch)
    scheduler = _scheduler(tmp_path)
    monkeypatch.setattr(authorized, "validate_authorization", lambda _program: authorization)
    monkeypatch.setattr(authorized, "_read_scheduler_context", lambda: scheduler)
    receipt, snapshot, snapshot_sha = authorized.create_receipt(authorization, scheduler)
    authorized._reserve_runtime()
    authorized._exclusive_json(paths[6] / "RUNNING", {
        "schema": "ouruniv-cf4-v6-open-shared-schedule-production-grammar-debug-state-marker-v2",
        "status": "running_authorized_shared_schedule_production",
        "grant_id": authorization["grant"]["grant_id"],
        "release_id": authorization["release"]["release_id"],
        "manifest_id": authorization["manifest"]["manifest_id"],
        "snapshot_sha256": snapshot_sha,
        **authorized._scheduler_running_fields(snapshot),
    })
    assert authorized._read_only_running_status()["status"] == "running"
    authorized._supervisor_force_failed(
        authorization["grant"]["grant_id"], f"supervisor_exit_{exit_code}",
    )
    assert (receipt / "FAILED").is_file() and (paths[6] / "FAILED").is_file()
    assert authorized._read_only_failed_status()["status"] == "failed"


def test_scheduler_log_unstable_fstat_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = tmp_path / "unstable.log"
    log.write_bytes(b"stable bytes")
    original = authorized.os.fstat
    calls = 0

    def unstable(descriptor: int):
        nonlocal calls
        value = original(descriptor)
        calls += 1
        if calls == 2:
            return SimpleNamespace(st_dev=value.st_dev, st_ino=value.st_ino, st_size=value.st_size + 1)
        return value

    monkeypatch.setattr(authorized.os, "fstat", unstable)
    with pytest.raises(PermissionError, match="unstable"):
        authorized._scheduler_log_checkpoint(log)
