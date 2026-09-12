#!/usr/bin/env python3
"""Fail-closed byte publication of the sealed terminal aggregation result."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = ROOT / "config/cf4_lg_highk_terminal_publication_repair_program_v1.json"
EXPECTED_CONFIG_SHA = "e41616445a3e664df744f41f36c96b4356e3a0a8951966b039c81ed5efc1c5f8"
CONFIG_SCHEMA = "ouruniv-cf4-lg-highk-terminal-publication-repair-program-v1"
GRANT_SCHEMA = "ouruniv-cf4-lg-highk-terminal-publication-repair-grant-v1"
CONFIG_KEYS = {
    "schema", "status", "date", "purpose", "lineage", "source_staging",
    "source_artifacts", "canonical_target", "publication_protocol",
    "checker_contract", "resources", "execution", "authorization",
    "forbidden", "audit_sequence",
}
GRANT_KEYS = {
    "schema", "status", "date", "purpose", "lineage", "program",
    "implementation", "source_staging", "source_artifacts",
    "canonical_target", "authorization",
}
ARTIFACT_NAMES = (
    "input_manifest.json", "pair_recentered_p1.json", "terminal_result.json",
    "manifest.json", "COMPLETE",
)
EXPECTED_PARENT = "51877b7ebba8601216411ef4e3d36623016ec625"
EXPECTED_SOURCE = Path("/gpfs/kjhan/CF4/recon/linear_cr/.lg_highk_terminal_aggregation_v1.305221.a51b6c9282d34950a3ca52287c115c99.staging")
EXPECTED_TARGET = Path("/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_terminal_aggregation_v1")
EXPECTED_ARTIFACTS = (
    ("input_manifest.json", 121804, "93e2f29b5eb481b0244377baef684c5f58f9f224e25eb536aef7af02de517532"),
    ("pair_recentered_p1.json", 99966, "c9c8124d4dce20c4de4b1d76aa48834ddbd8557a05f4a0a7d9e44438cb8ec339"),
    ("terminal_result.json", 119847, "73e57de15a8beed571e3c08cc92a5f786ad79e386e6a28fdfed77b6b34cf8fdd"),
    ("manifest.json", 2258, "4417b4f85e5d2baf5336202cfcf59973e6eec8b857a7940ae6fb582c486e32b2"),
    ("COMPLETE", 172, "2fff9924225bfab0811304f9f0b0c6d418c42e4ca9a07c88e4d3d3c9285b9323"),
)
_PUBLICATION_TOKEN = object()
_FIXTURE_TOKEN = object()


def _parse_config(payload: bytes) -> dict[str, Any]:
    config = json.loads(payload)
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise RuntimeError("repair config exact top-level keyset differs")
    if config["schema"] != CONFIG_SCHEMA or config["status"] != "frozen_design_approved_implementation_only":
        raise RuntimeError("repair config schema or status differs")
    observed = tuple((row.get("name"), row.get("size_bytes"), row.get("sha256")) for row in config["source_artifacts"])
    execution, lineage = config["execution"], config["lineage"]
    if lineage.get("required_parent_commit") != EXPECTED_PARENT \
            or lineage.get("sealed_runtime_commit") != EXPECTED_PARENT \
            or Path(config["source_staging"].get("path", "")) != EXPECTED_SOURCE \
            or Path(config["canonical_target"].get("path", "")) != EXPECTED_TARGET \
            or observed != EXPECTED_ARTIFACTS \
            or execution.get("canonical_program_path") != str(CANONICAL_CONFIG.relative_to(ROOT)) \
            or execution.get("grant_path") != "config/cf4_lg_highk_terminal_publication_repair_grant_v1.json" \
            or execution.get("future_result_record_path") != "config/cf4_lg_highk_terminal_publication_repair_result_record_v1.json":
        raise RuntimeError("frozen program, lineage, source, target, or artifact pins differ")
    if config["authorization"] != {
        "implementation": True, "publication_execution": False,
        "grant_required_for_publication_execution": True,
        "grant_present_at_design_time": False, "slurm_submission": False,
        "scientific_recomputation": False, "promotion": False,
    }:
        raise RuntimeError("repair program authorization must remain implementation-only")
    return config


def load_config(path: Path) -> dict[str, Any]:
    supplied = Path(path)
    if supplied.resolve() != CANONICAL_CONFIG.resolve():
        raise RuntimeError("only the canonical publication repair program path is accepted")
    before = os.lstat(supplied)
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) \
            or stat.S_IMODE(before.st_mode) != 0o644 or before.st_nlink != 1:
        raise RuntimeError("canonical publication repair program metadata differs")
    fd = os.open(supplied, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened, payload = os.fstat(fd), _read_all(fd)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if _identity(before) != _identity(opened) or _identity(opened) != _identity(after):
        raise RuntimeError("canonical publication repair program changed during read")
    if hashlib.sha256(payload).hexdigest() != EXPECTED_CONFIG_SHA:
        raise RuntimeError("canonical publication repair program SHA-256 differs")
    return _parse_config(payload)


def _git_bytes(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def _git_text(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout.strip()


def validate_lineage_values(
    config: Mapping[str, Any], *, head: str, upstream: str,
    grant_parents: Sequence[str], implementation_parents: Sequence[str],
    implementation_rows: Sequence[tuple[str, str]], grant_rows: Sequence[tuple[str, str]],
    implementation_modes: Mapping[str, str], grant_modes: Mapping[str, str], clean: bool,
) -> str:
    lineage = config["lineage"]
    implementation = grant_parents[0] if len(grant_parents) == 1 else ""
    implementation_paths = lineage["required_exact_added_paths"]
    grant_paths = lineage["grant_commit"]["required_exact_added_paths"]
    if head != upstream or len(grant_parents) != 1 or list(implementation_parents) != [EXPECTED_PARENT]:
        raise RuntimeError("grant/implementation/upstream three-commit lineage differs")
    if sorted(implementation_rows) != sorted(("A", path) for path in implementation_paths):
        raise RuntimeError("implementation commit is not the exact six additions")
    if sorted(grant_rows) != sorted(("A", path) for path in grant_paths):
        raise RuntimeError("grant commit is not the exact one grant addition")
    if set(implementation_modes) != set(implementation_paths) \
            or any(mode != "100644" for mode in implementation_modes.values()) \
            or set(grant_modes) != set(grant_paths) \
            or any(mode != "100644" for mode in grant_modes.values()):
        raise RuntimeError("implementation or grant Git modes differ")
    if not clean:
        raise RuntimeError("worktree is not clean outside scripts/tripwire/")
    return implementation


def _name_status(older: str, newer: str) -> list[tuple[str, str]]:
    output = _git_text("diff", "--no-renames", "--name-status", older, newer, "--")
    rows = []
    for line in output.splitlines() if output else []:
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] not in {"A", "M", "D"}:
            raise RuntimeError("Git diff contains a rename, copy, or malformed row")
        rows.append((fields[0], fields[1]))
    return rows


def _tree_modes(commit: str, paths: Sequence[str]) -> dict[str, str]:
    modes = {}
    for path in paths:
        fields = _git_text("ls-tree", commit, "--", path).split()
        if len(fields) < 4 or fields[3] != path:
            raise RuntimeError(f"Git tree row missing: {path}")
        modes[path] = fields[0]
    return modes


def validate_lineage(config: Mapping[str, Any]) -> dict[str, str]:
    head, upstream = _git_text("rev-parse", "HEAD"), _git_text("rev-parse", "@{upstream}")
    grant_line = _git_text("rev-list", "--parents", "-n", "1", head).split()
    implementation = grant_line[1] if len(grant_line) == 2 else ""
    implementation_line = _git_text("rev-list", "--parents", "-n", "1", implementation).split() if implementation else []
    implementation_paths = config["lineage"]["required_exact_added_paths"]
    grant_paths = config["lineage"]["grant_commit"]["required_exact_added_paths"]
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain=v1", "-z", "--untracked-files=all",
         "--", ".", ":(exclude)scripts/tripwire/**"], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    validated = validate_lineage_values(
        config, head=head, upstream=upstream, grant_parents=grant_line[1:],
        implementation_parents=implementation_line[1:],
        implementation_rows=_name_status(EXPECTED_PARENT, implementation),
        grant_rows=_name_status(implementation, head),
        implementation_modes=_tree_modes(implementation, implementation_paths),
        grant_modes=_tree_modes(head, grant_paths), clean=not dirty)
    for reserved in (config["lineage"]["grant_commit"]["path"],
                     config["lineage"]["future_result_record_commit"]["path"]):
        if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{implementation}:{reserved}"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError(f"reserved file unexpectedly exists in implementation commit: {reserved}")
    result_path = config["lineage"]["future_result_record_commit"]["path"]
    if subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{head}:{result_path}"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError("future publication result record already exists")
    return {"head": head, "implementation": validated, "parent": EXPECTED_PARENT}


def _implementation_rows(config: Mapping[str, Any], commit: str) -> list[dict[str, Any]]:
    return [{"path": path, "mode": "100644", "sha256": hashlib.sha256(_git_bytes(commit, path)).hexdigest()}
            for path in config["lineage"]["required_exact_added_paths"]]


def validate_grant_values(config: Mapping[str, Any], lineage: Mapping[str, str],
                          grant: Mapping[str, Any], implementation_rows) -> None:
    if not isinstance(grant, dict) or set(grant) != GRANT_KEYS or grant.get("schema") != GRANT_SCHEMA \
            or grant.get("status") != "authorized_publication_only":
        raise PermissionError("publication grant exact schema/keyset/status differs")
    expected_source = [{key: row[key] for key in ("name", "size_bytes", "sha256")}
                       for row in config["source_artifacts"]]
    expected_authorization = {"publication_only": True, "scientific_recomputation": False,
                              "overwrite": False, "retry": False, "cleanup": False,
                              "promotion": False}
    if grant.get("lineage") != {"required_parent_commit": EXPECTED_PARENT} \
            or grant.get("program") != {"path": str(CANONICAL_CONFIG.relative_to(ROOT)),
                                        "sha256": EXPECTED_CONFIG_SHA} \
            or grant.get("implementation") != {"commit": lineage["implementation"],
                                                "files": implementation_rows} \
            or grant.get("source_staging") != str(EXPECTED_SOURCE) \
            or grant.get("source_artifacts") != expected_source \
            or grant.get("canonical_target") != str(EXPECTED_TARGET) \
            or grant.get("authorization") != expected_authorization:
        raise PermissionError("publication grant bindings or authorization differ")


def _load_grant(config: Mapping[str, Any], lineage: Mapping[str, str]) -> dict[str, Any]:
    path = ROOT / config["execution"]["grant_path"]
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode) \
            or stat.S_IMODE(info.st_mode) != 0o644 or info.st_nlink != 1:
        raise PermissionError("publication grant must be canonical regular mode-0644 nlink-1 JSON")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened, payload = os.fstat(fd), _read_all(fd)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if _identity(info) != _identity(opened) or _identity(opened) != _identity(after):
        raise PermissionError("publication grant changed during stable read")
    grant = json.loads(payload)
    validate_grant_values(
        config, lineage, grant, _implementation_rows(config, lineage["implementation"]),
    )
    return grant


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _read_all(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _open_source(config: Mapping[str, Any], source: Path):
    before = os.lstat(source)
    if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode) \
            or stat.S_IMODE(before.st_mode) != 0o555:
        raise RuntimeError("source staging must be a non-symlink directory mode 0555")
    directory_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    if (before.st_dev, before.st_ino) != (os.fstat(directory_fd).st_dev, os.fstat(directory_fd).st_ino):
        os.close(directory_fd)
        raise RuntimeError("source directory identity changed while opening")
    artifacts = {}
    try:
        if sorted(os.listdir(directory_fd)) != sorted(ARTIFACT_NAMES):
            raise RuntimeError("source staging exact entry set differs")
        for spec in config["source_artifacts"]:
            name = spec["name"]
            row = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(row.st_mode) or stat.S_ISLNK(row.st_mode) \
                    or stat.S_IMODE(row.st_mode) != 0o444 or row.st_nlink != 1:
                raise RuntimeError(f"source artifact type/mode/nlink differs: {name}")
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
            payload = _read_all(fd)
            if _identity(row) != _identity(os.fstat(fd)) or len(payload) != spec["size_bytes"] \
                    or hashlib.sha256(payload).hexdigest() != spec["sha256"] \
                    or json.loads(payload).get("schema") != spec["schema"]:
                os.close(fd)
                raise RuntimeError(f"source artifact stable bytes/schema differ: {name}")
            artifacts[name] = (fd, row, payload)
    except BaseException:
        for fd, _, _ in artifacts.values():
            os.close(fd)
        os.close(directory_fd)
        raise
    return directory_fd, artifacts


def _verify_source(directory_fd: int, artifacts) -> None:
    if sorted(os.listdir(directory_fd)) != sorted(ARTIFACT_NAMES):
        raise RuntimeError("source entry set changed during publication")
    for name, (fd, original, payload) in artifacts.items():
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _identity(entry) != _identity(original) or _identity(os.fstat(fd)) != _identity(original) \
                or _read_all(fd) != payload:
            raise RuntimeError(f"source artifact changed during publication: {name}")


def _assert_parent_binding(parent: Path, parent_fd: int) -> os.stat_result:
    nofollow, follow, opened = os.lstat(parent), os.stat(parent), os.fstat(parent_fd)
    if not stat.S_ISDIR(nofollow.st_mode) or stat.S_ISLNK(nofollow.st_mode) \
            or (nofollow.st_dev, nofollow.st_ino) != (follow.st_dev, follow.st_ino) \
            or (nofollow.st_dev, nofollow.st_ino) != (opened.st_dev, opened.st_ino):
        raise RuntimeError("canonical parent path no longer binds held parent fd")
    return opened


def _assert_target_binding(parent: Path, parent_fd: int, name: str,
                           target_fd: int, expected_mode: int) -> os.stat_result:
    _assert_parent_binding(parent, parent_fd)
    nofollow = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    follow = os.stat(name, dir_fd=parent_fd, follow_symlinks=True)
    opened = os.fstat(target_fd)
    identities = {(row.st_dev, row.st_ino) for row in (nofollow, follow, opened)}
    if len(identities) != 1 or not all(stat.S_ISDIR(row.st_mode) for row in (nofollow, follow, opened)) \
            or any(stat.S_IMODE(row.st_mode) != expected_mode for row in (nofollow, follow, opened)):
        raise RuntimeError("canonical target entry no longer binds held target fd")
    return opened


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        count = os.write(fd, view)
        if count <= 0:
            raise OSError(errno.EIO, "short publication write")
        view = view[count:]


def _copy_artifact(target_fd: int, name: str, payload: bytes) -> None:
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                 0o600, dir_fd=target_fd)
    try:
        _write_all(fd, payload)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        row = os.fstat(fd)
        if not stat.S_ISREG(row.st_mode) or stat.S_IMODE(row.st_mode) != 0o444 \
                or row.st_nlink != 1 or row.st_size != len(payload):
            raise RuntimeError(f"published artifact metadata differs: {name}")
    finally:
        os.close(fd)


def _run_private_checker(config_path: Path, source_fd: int, parent_fd: int,
                         target_fd: int, target_name: str) -> None:
    checker = ROOT / "scripts/check_cf4_lg_highk_terminal_publication_repair_v1.py"
    subprocess.run(
        ["/home/kjhan/miniconda3/envs/circle/bin/python3.11", "-I", "-P", str(checker),
         "--config", str(config_path), "--source-fd", str(source_fd),
         "--parent-fd", str(parent_fd), "--target-fd", str(target_fd),
         "--target-name", target_name], cwd=ROOT, check=True,
        pass_fds=(source_fd, parent_fd, target_fd))


def _publication_core(
    config: Mapping[str, Any], *, config_path: Path, source: Path, target: Path,
    authority: object, checker_hook: Callable[[Path, int, int, int, str], None],
) -> dict[str, Any]:
    if authority is _PUBLICATION_TOKEN:
        if source != EXPECTED_SOURCE or target != EXPECTED_TARGET \
                or config_path.resolve() != CANONICAL_CONFIG.resolve():
            raise RuntimeError("public publication must use exact canonical paths")
    elif authority is _FIXTURE_TOKEN:
        if source.resolve() == EXPECTED_SOURCE.resolve() or target.resolve() == EXPECTED_TARGET.resolve():
            raise RuntimeError("fixture publication must reject canonical source and target paths")
    else:
        raise PermissionError("invalid publication core authority token")
    source_fd, artifacts = _open_source(config, source)
    parent_fd = target_fd = None
    try:
        parent_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        _assert_parent_binding(target.parent, parent_fd)
        try:
            os.mkdir(target.name, 0o700, dir_fd=parent_fd)
        except FileExistsError as error:
            raise FileExistsError(f"canonical target already exists; no retry: {target}") from error
        target_fd = os.open(target.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=parent_fd)
        _assert_target_binding(target.parent, parent_fd, target.name, target_fd, 0o700)
        for name in ARTIFACT_NAMES:
            _assert_target_binding(target.parent, parent_fd, target.name, target_fd, 0o700)
            _copy_artifact(target_fd, name, artifacts[name][2])
            _assert_target_binding(target.parent, parent_fd, target.name, target_fd, 0o700)
        _verify_source(source_fd, artifacts)
        os.fsync(target_fd)
        _assert_target_binding(target.parent, parent_fd, target.name, target_fd, 0o700)
        checker_hook(config_path, source_fd, parent_fd, target_fd, target.name)
        _assert_target_binding(target.parent, parent_fd, target.name, target_fd, 0o700)
        _verify_source(source_fd, artifacts)
        os.fchmod(target_fd, 0o555)
        _assert_target_binding(target.parent, parent_fd, target.name, target_fd, 0o555)
        os.fsync(target_fd)
        _assert_target_binding(target.parent, parent_fd, target.name, target_fd, 0o555)
        os.fsync(parent_fd)
        _assert_target_binding(target.parent, parent_fd, target.name, target_fd, 0o555)
    finally:
        if target_fd is not None:
            os.close(target_fd)
        if parent_fd is not None:
            os.close(parent_fd)
        for fd, _, _ in artifacts.values():
            os.close(fd)
        os.close(source_fd)
    return {"status": "complete_publication_repair_verified",
            "scientific_status": "complete_scientific_fail_terminal_aggregation_closed",
            "artifacts": 5}


def _publish_fixture(*, authority: object, source: Path, target: Path,
                     checker_hook: Callable[[Path, int, int, int, str], None] = _run_private_checker):
    return _publication_core(load_config(CANONICAL_CONFIG), config_path=CANONICAL_CONFIG,
                             source=Path(source), target=Path(target), authority=authority,
                             checker_hook=checker_hook)


def publish(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    lineage = validate_lineage(config)
    _load_grant(config, lineage)
    return _publication_core(config, config_path=CANONICAL_CONFIG, source=EXPECTED_SOURCE,
                             target=EXPECTED_TARGET, authority=_PUBLICATION_TOKEN,
                             checker_hook=_run_private_checker)


def validate_source_only(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    source_fd, artifacts = _open_source(config, EXPECTED_SOURCE)
    try:
        _verify_source(source_fd, artifacts)
    finally:
        for fd, _, _ in artifacts.values():
            os.close(fd)
        os.close(source_fd)
    if os.path.lexists(EXPECTED_TARGET):
        raise FileExistsError(f"canonical target already exists; no retry: {EXPECTED_TARGET}")
    return {"status": "publication_source_preflight_pass", "artifacts": 5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lineage-preflight", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()
    if args.lineage_preflight and args.test_only:
        parser.error("preflight modes are mutually exclusive")
    config = load_config(args.config)
    if args.lineage_preflight:
        result = {"status": "publication_lineage_preflight_pass", **validate_lineage(config)}
    elif args.test_only:
        result = validate_source_only(args.config)
    else:
        result = publish(args.config)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
