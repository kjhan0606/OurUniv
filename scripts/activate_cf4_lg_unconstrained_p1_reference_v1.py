#!/usr/bin/env python3
"""Relay a pushed Git ref event into one already-held Slurm allocation.

The science worker still performs the authoritative grant, lineage, runtime,
and output checks.  This controller only closes the event-delivery gap seen
when a push updates a shared filesystem ref on a different client node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


SHA1_RE = re.compile(r"[0-9a-f]{40}")
JOB_ID_RE = re.compile(r"[1-9][0-9]*")
NODE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
PROGRAM_REL = Path("config/cf4_lg_unconstrained_p1_reference_program_v1.json")
HELD_LINE = "HELD_ALLOCATION_BLOCKED_NO_SCIENCE_WAITING_EXACT_GRANT"


class ActivationError(RuntimeError):
    """Fail-closed activation error."""


def _run(argv: Sequence[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        list(argv), cwd=cwd, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ActivationError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise ActivationError(f"{label} is not canonical JSON")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_scontrol_job(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in text.split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def _absolute_git_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def _validate_repo(repo: Path) -> Path:
    repo = repo.resolve(strict=True)
    if not repo.is_dir() or _run(("git", "rev-parse", "--show-toplevel"), cwd=repo) != str(repo):
        raise ActivationError("--repo must be the exact Git worktree root")
    return repo


def _load_context(repo: Path, requested_job_id: str) -> dict[str, Any]:
    if JOB_ID_RE.fullmatch(requested_job_id) is None:
        raise ActivationError("malformed Slurm job id")
    program_path = repo / PROGRAM_REL
    program = _load_canonical(program_path, "program")
    grant_rel = Path(program["lineage"]["future_grant_path"])
    if grant_rel.is_absolute() or ".." in grant_rel.parts:
        raise ActivationError("grant path escapes the repository")
    grant_path = repo / grant_rel
    grant = _load_canonical(grant_path, "grant")
    contract = program["grant_contract"]
    if grant.get("schema") != contract.get("schema") \
            or grant.get("status") != contract.get("status"):
        raise ActivationError("grant schema/status mismatch")
    receipt = grant.get("allocation_receipt")
    pins = grant.get("runtime_pins")
    if not isinstance(receipt, Mapping) or not isinstance(pins, Mapping) \
            or pins != receipt.get("runtime_pins") \
            or receipt.get("one_live_held_allocation") is not True \
            or receipt.get("slurm_job_id") != requested_job_id:
        raise ActivationError("grant is not bound to the requested held allocation")
    node = receipt.get("slurm_nodelist")
    if not isinstance(node, str) or NODE_RE.fullmatch(node) is None \
            or pins.get("node") != node or pins.get("nodelist") != node:
        raise ActivationError("grant node identity is not one exact hostname")
    implementation = grant.get("implementation")
    if not isinstance(implementation, Mapping):
        raise ActivationError("grant implementation identity missing")
    implementation_commit = implementation.get("commit")
    if not isinstance(implementation_commit, str) or SHA1_RE.fullmatch(implementation_commit) is None:
        raise ActivationError("malformed implementation commit")
    head = _run(("git", "rev-parse", "HEAD"), cwd=repo)
    upstream_ref = _run(("git", "rev-parse", "--symbolic-full-name", "@{upstream}"), cwd=repo)
    upstream = _run(("git", "rev-parse", upstream_ref), cwd=repo)
    parents = _run(("git", "rev-list", "--parents", "-n", "1", head), cwd=repo).split()[1:]
    if SHA1_RE.fullmatch(head) is None or head != upstream or parents != [implementation_commit]:
        raise ActivationError("target worktree is not the pushed direct-child grant commit")
    if not upstream_ref.startswith("refs/remotes/"):
        raise ActivationError("upstream is not an exact remote-tracking ref")
    git_common_raw = _run(("git", "rev-parse", "--git-common-dir"), cwd=repo)
    git_common = _absolute_git_path(repo, git_common_raw).resolve(strict=True)
    ref_raw = _run(("git", "rev-parse", "--git-path", upstream_ref), cwd=repo)
    ref_path = _absolute_git_path(repo, ref_raw)
    expected_ref = git_common / upstream_ref
    if os.path.abspath(ref_path) != os.path.abspath(expected_ref):
        raise ActivationError("Git returned an unexpected remote-ref path")
    _validate_loose_ref(ref_path, head)
    grant_sha256 = _sha256(grant_path)
    return {
        "repo": repo, "program": program, "grant": grant, "grant_path": grant_path,
        "grant_sha256": grant_sha256, "job_id": requested_job_id, "node": node,
        "head": head, "upstream_ref": upstream_ref, "ref_path": ref_path,
    }


def _validate_live_job(context: Mapping[str, Any]) -> dict[str, str]:
    fields = _parse_scontrol_job(_run(("scontrol", "show", "job", "--oneliner",
                                       context["job_id"])))
    pins = context["grant"]["runtime_pins"]
    stored = pins["Slurm_raw_resource_fields"]
    exact_keys = (
        "NumNodes", "NumTasks", "NumCPUs", "CPUs/Task", "ReqTRES", "TresPerNode",
        "AllocTRES", "TimeLimit", "Requeue", "Restarts", "Partition", "JobState", "NodeList",
    )
    if fields.get("JobId") != context["job_id"] \
            or {key: fields.get(key) for key in exact_keys} != stored \
            or fields.get("JobState") != "RUNNING" \
            or fields.get("NodeList") != context["node"]:
        raise ActivationError("live Slurm allocation differs from the signed receipt")
    return fields


def _last_nonempty_line(path: Path, limit: int = 65536) -> str:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ActivationError("Slurm log is absent") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise ActivationError("Slurm log is not one regular nonempty file")
    with path.open("rb") as handle:
        handle.seek(max(0, metadata.st_size - limit))
        tail = handle.read(limit)
    lines = [line for line in tail.splitlines() if line]
    if not lines:
        raise ActivationError("Slurm log has no nonempty line")
    try:
        return lines[-1].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ActivationError("Slurm log tail is not ASCII") from exc


def _validate_held_log(context: Mapping[str, Any], fields: Mapping[str, str]) -> None:
    expected = context["repo"] / "logs" / f"unconstrained_p1_reference_v1-{context['job_id']}.log"
    stdout = fields.get("StdOut")
    if not isinstance(stdout, str) or os.path.abspath(stdout) != os.path.abspath(expected):
        raise ActivationError("Slurm stdout is not the exact frozen job log")
    if _last_nonempty_line(expected) != HELD_LINE:
        raise ActivationError("worker is not in the exact no-science held state")


def _validate_loose_ref(ref_path: Path, expected_commit: str) -> None:
    if SHA1_RE.fullmatch(expected_commit) is None:
        raise ActivationError("malformed expected commit")
    try:
        metadata = ref_path.lstat()
    except FileNotFoundError as exc:
        raise ActivationError("remote-tracking ref is not loose") from exc
    if not stat.S_ISREG(metadata.st_mode) or ref_path.is_symlink() \
            or ref_path.read_bytes() != (expected_commit + "\n").encode("ascii"):
        raise ActivationError("remote-tracking ref bytes/type mismatch")


def relay_exact_loose_ref(ref_path: Path, expected_commit: str) -> None:
    """Atomically rewrite identical loose-ref bytes, producing local IN_MOVED_TO."""
    _validate_loose_ref(ref_path, expected_commit)
    original_mode = stat.S_IMODE(ref_path.lstat().st_mode)
    parent = ref_path.parent
    lock = parent / (ref_path.name + ".lock")
    payload = (expected_commit + "\n").encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd: int | None = None
    created = False
    try:
        fd = os.open(lock, flags, 0o644)
        created = True
        os.fchmod(fd, original_mode)
        offset = 0
        while offset < len(payload):
            offset += os.write(fd, payload[offset:])
        os.fsync(fd)
        os.close(fd)
        fd = None
        _validate_loose_ref(ref_path, expected_commit)
        os.replace(lock, ref_path)
        created = False
    except FileExistsError as exc:
        raise ActivationError("Git ref lock already exists; refusing activation") from exc
    finally:
        if fd is not None:
            os.close(fd)
        if created:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass


def _relay_on_node(context: Mapping[str, Any], expected_commit: str,
                   expected_grant_sha256: str) -> None:
    if os.environ.get("SLURM_JOB_ID") != context["job_id"] \
            or os.environ.get("SLURM_STEP_ID") in (None, "batch", "extern"):
        raise ActivationError("relay must run as an attached step of the exact Slurm job")
    host = _run(("hostname",))
    if host != context["node"] or expected_commit != context["head"] \
            or expected_grant_sha256 != context["grant_sha256"]:
        raise ActivationError("relay job/node/commit/grant binding mismatch")
    fields = _validate_live_job(context)
    _validate_held_log(context, fields)
    relay_exact_loose_ref(context["ref_path"], expected_commit)
    print("SAME_ALLOCATION_EXACT_REF_EVENT_EMITTED", flush=True)


def _controller(context: Mapping[str, Any], script: Path) -> None:
    if _run(("hostname",)) != "syntax" or os.environ.get("SLURM_JOB_ID"):
        raise ActivationError("controller activation must run on syntax outside an allocation")
    cluster = _run(("scontrol", "show", "config"))
    if "ClusterName             = syntax" not in cluster:
        raise ActivationError("wrong Slurm controller")
    fields = _validate_live_job(context)
    _validate_held_log(context, fields)
    argv = (
        "srun", f"--jobid={context['job_id']}", "--overlap", "--nodes=1", "--ntasks=1",
        "--cpus-per-task=1", f"--nodelist={context['node']}", "--kill-on-bad-exit=1",
        str(Path(sys.executable).resolve(strict=True)), "-P", str(script), "--relay-on-node",
        "--repo", str(context["repo"]),
        "--job-id", context["job_id"], "--expected-commit", context["head"],
        "--expected-grant-sha256", context["grant_sha256"],
    )
    output = _run(argv)
    if output != "SAME_ALLOCATION_EXACT_REF_EVENT_EMITTED":
        raise ActivationError("attached relay did not return the exact success seal")
    print(output, flush=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--relay-on-node", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expected-commit", help=argparse.SUPPRESS)
    parser.add_argument("--expected-grant-sha256", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        repo = _validate_repo(args.repo)
        context = _load_context(repo, args.job_id)
        if args.relay_on_node:
            if args.expected_commit is None or SHA1_RE.fullmatch(args.expected_commit) is None \
                    or args.expected_grant_sha256 is None \
                    or re.fullmatch(r"[0-9a-f]{64}", args.expected_grant_sha256) is None:
                raise ActivationError("relay pins are absent or malformed")
            _relay_on_node(context, args.expected_commit, args.expected_grant_sha256)
        else:
            if args.expected_commit is not None or args.expected_grant_sha256 is not None:
                raise ActivationError("relay-only pins are forbidden in controller mode")
            _controller(context, Path(__file__).resolve())
        return 0
    except (ActivationError, KeyError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ACTIVATION_REFUSED: {exc}", file=sys.stderr, flush=True)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
