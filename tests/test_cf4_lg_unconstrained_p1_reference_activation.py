from __future__ import annotations

import ctypes
import importlib.util
import os
import select
import struct
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/activate_cf4_lg_unconstrained_p1_reference_v1.py"
SPEC = importlib.util.spec_from_file_location("held_ref_activation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
activation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(activation)


def _loose_ref(tmp_path: Path, commit: str = "a" * 40) -> Path:
    ref = tmp_path / "refs/remotes/origin/branch"
    ref.parent.mkdir(parents=True)
    ref.write_text(commit + "\n", encoding="ascii")
    return ref


def _watch_moved_to(directory: Path) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    assert fd >= 0
    watch = libc.inotify_add_watch(fd, os.fsencode(directory), 0x00000080)
    assert watch >= 0
    return fd


def test_atomic_identical_ref_relay_emits_local_moved_to(tmp_path: Path):
    commit = "1" * 40
    ref = _loose_ref(tmp_path, commit)
    old_inode = ref.stat().st_ino
    old_mode = ref.stat().st_mode & 0o777
    fd = _watch_moved_to(ref.parent)
    try:
        activation.relay_exact_loose_ref(ref, commit)
        assert select.select([fd], [], [], 1.0)[0] == [fd]
        data = os.read(fd, 65536)
    finally:
        os.close(fd)
    watch, mask, cookie, length = struct.unpack_from("iIII", data, 0)
    name = data[16:16 + length].split(b"\0", 1)[0]
    assert watch >= 0 and cookie > 0
    assert mask & 0x00000080
    assert name == ref.name.encode()
    assert ref.read_text(encoding="ascii") == commit + "\n"
    assert ref.stat().st_ino != old_inode
    assert ref.stat().st_mode & 0o777 == old_mode
    assert not ref.with_name(ref.name + ".lock").exists()


@pytest.mark.parametrize("commit", ["", "a" * 39, "A" * 40, "g" * 40, "a" * 41])
def test_relay_rejects_malformed_commit(tmp_path: Path, commit: str):
    ref = _loose_ref(tmp_path)
    with pytest.raises(activation.ActivationError, match="malformed expected commit"):
        activation.relay_exact_loose_ref(ref, commit)


def test_relay_rejects_ref_content_change(tmp_path: Path):
    ref = _loose_ref(tmp_path, "2" * 40)
    with pytest.raises(activation.ActivationError, match="bytes/type mismatch"):
        activation.relay_exact_loose_ref(ref, "3" * 40)
    assert ref.read_text(encoding="ascii") == "2" * 40 + "\n"


def test_relay_rejects_symlink(tmp_path: Path):
    target = _loose_ref(tmp_path, "4" * 40)
    link = tmp_path / "linked-ref"
    link.symlink_to(target)
    with pytest.raises(activation.ActivationError, match="bytes/type mismatch"):
        activation.relay_exact_loose_ref(link, "4" * 40)


def test_relay_never_removes_preexisting_git_lock(tmp_path: Path):
    ref = _loose_ref(tmp_path, "5" * 40)
    lock = ref.with_name(ref.name + ".lock")
    lock.write_text("do-not-remove\n", encoding="ascii")
    with pytest.raises(activation.ActivationError, match="lock already exists"):
        activation.relay_exact_loose_ref(ref, "5" * 40)
    assert lock.read_text(encoding="ascii") == "do-not-remove\n"
    assert ref.read_text(encoding="ascii") == "5" * 40 + "\n"


def test_scontrol_parser_is_exact_and_non_scanning():
    fields = activation._parse_scontrol_job(
        "JobId=326158 JobState=RUNNING NodeList=syn05 ReqTRES=cpu=16,mem=20G,gres/gpu=1"
    )
    assert fields == {
        "JobId": "326158", "JobState": "RUNNING", "NodeList": "syn05",
        "ReqTRES": "cpu=16,mem=20G,gres/gpu=1",
    }


def test_held_log_requires_exact_final_state(tmp_path: Path):
    log = tmp_path / "job.log"
    log.write_text("receipt\n" + activation.HELD_LINE + "\n", encoding="ascii")
    assert activation._last_nonempty_line(log) == activation.HELD_LINE
    log.write_text("receipt\n" + activation.HELD_LINE + "\nSCIENCE_GATE_OPEN\n", encoding="ascii")
    assert activation._last_nonempty_line(log) == "SCIENCE_GATE_OPEN"


def test_controller_uses_one_attached_exact_node_step(tmp_path: Path, monkeypatch, capsys):
    log = tmp_path / "logs/unconstrained_p1_reference_v1-42.log"
    log.parent.mkdir()
    log.write_text(activation.HELD_LINE + "\n", encoding="ascii")
    raw = {
        "NumNodes": "1", "NumTasks": "1", "NumCPUs": "16", "CPUs/Task": "16",
        "ReqTRES": "cpu=16,mem=20G,node=1,gres/gpu=1", "TresPerNode": "gres/gpu:1",
        "AllocTRES": "cpu=16,node=1,gres/gpu=1", "TimeLimit": "1-00:00:00",
        "Requeue": "0", "Restarts": "0", "Partition": "a40", "JobState": "RUNNING",
        "NodeList": "syn05",
    }
    context = {
        "repo": tmp_path, "job_id": "42", "node": "syn05", "head": "a" * 40,
        "grant_sha256": "b" * 64,
        "grant": {"runtime_pins": {"Slurm_raw_resource_fields": raw}},
    }
    calls = []

    def fake_run(argv, *, cwd=None):
        calls.append(tuple(argv))
        if tuple(argv) == ("hostname",):
            return "syntax"
        if tuple(argv) == ("scontrol", "show", "config"):
            return "ClusterName             = syntax"
        if argv[:4] == ("scontrol", "show", "job", "--oneliner"):
            fields = {"JobId": "42", "StdOut": str(log), **raw}
            return " ".join(f"{key}={value}" for key, value in fields.items())
        if argv[0] == "srun":
            return "SAME_ALLOCATION_EXACT_REF_EVENT_EMITTED"
        raise AssertionError(argv)

    monkeypatch.setattr(activation, "_run", fake_run)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    activation._controller(context, SCRIPT)
    assert capsys.readouterr().out.strip() == "SAME_ALLOCATION_EXACT_REF_EVENT_EMITTED"
    srun = next(call for call in calls if call[0] == "srun")
    assert srun[:8] == (
        "srun", "--jobid=42", "--overlap", "--nodes=1", "--ntasks=1",
        "--cpus-per-task=1", "--nodelist=syn05", "--kill-on-bad-exit=1",
    )
    assert "sbatch" not in srun and "scancel" not in srun


def test_static_protocol_has_no_polling_process_scan_or_manual_compute():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "srun" in source and "--jobid=" in source and "--overlap" in source
    assert "SAME_ALLOCATION_EXACT_REF_EVENT_EMITTED" in source
    assert "os.replace(lock, ref_path)" in source
    assert "while True" not in source
    assert "pgrep" not in source
    assert "ps " not in source
    assert "glob(" not in source
    assert "rglob(" not in source
    assert "scancel" not in source and "sbatch" not in source


def test_cli_requires_explicit_repo_and_job_id():
    with pytest.raises(SystemExit):
        activation._parse_args([])
    args = activation._parse_args(["--repo", "/x", "--job-id", "42"])
    assert args.repo == Path("/x") and args.job_id == "42"
    assert args.relay_on_node is False
