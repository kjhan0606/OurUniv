#!/usr/bin/env python3
"""Independent inherited-FD checker for private terminal publication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CONFIG = ROOT / "config/cf4_lg_highk_terminal_publication_repair_program_v1.json"
EXPECTED_CONFIG_SHA = "e41616445a3e664df744f41f36c96b4356e3a0a8951966b039c81ed5efc1c5f8"
CONFIG_SCHEMA = "ouruniv-cf4-lg-highk-terminal-publication-repair-program-v1"
CONFIG_KEYS = {"schema", "status", "date", "purpose", "lineage", "source_staging",
               "source_artifacts", "canonical_target", "publication_protocol",
               "checker_contract", "resources", "execution", "authorization",
               "forbidden", "audit_sequence"}
ARTIFACT_NAMES = ("input_manifest.json", "pair_recentered_p1.json", "terminal_result.json",
                  "manifest.json", "COMPLETE")
PAYLOAD_NAMES = ARTIFACT_NAMES[:3]
EXPECTED_PARENT = "51877b7ebba8601216411ef4e3d36623016ec625"
EXPECTED_ARTIFACTS = (
    ("input_manifest.json", 121804, "93e2f29b5eb481b0244377baef684c5f58f9f224e25eb536aef7af02de517532"),
    ("pair_recentered_p1.json", 99966, "c9c8124d4dce20c4de4b1d76aa48834ddbd8557a05f4a0a7d9e44438cb8ec339"),
    ("terminal_result.json", 119847, "73e57de15a8beed571e3c08cc92a5f786ad79e386e6a28fdfed77b6b34cf8fdd"),
    ("manifest.json", 2258, "4417b4f85e5d2baf5336202cfcf59973e6eec8b857a7940ae6fb582c486e32b2"),
    ("COMPLETE", 172, "2fff9924225bfab0811304f9f0b0c6d418c42e4ca9a07c88e4d3d3c9285b9323"),
)
SEALED_IMPLEMENTATION_PATHS = (
    "config/cf4_lg_highk_terminal_aggregation_v1.json",
    "scripts/check_cf4_lg_highk_terminal_aggregation_v1.py",
    "scripts/run_cf4_lg_highk_terminal_aggregation_v1.sbatch",
    "src/cf4_lg_highk_terminal_aggregation.py",
    "tests/test_cf4_lg_highk_terminal_aggregation.py",
    "tests/test_cf4_lg_highk_terminal_aggregation_slurm.py",
)


def _load_config(path: Path) -> dict[str, Any]:
    if Path(path).resolve() != CANONICAL_CONFIG.resolve():
        raise RuntimeError("checker accepts only the canonical program path")
    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) \
            or stat.S_IMODE(before.st_mode) != 0o644 or before.st_nlink != 1:
        raise RuntimeError("canonical repair program metadata differs")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened, payload = os.fstat(fd), _read_all(fd)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if _identity(before) != _identity(opened) or _identity(opened) != _identity(after):
        raise RuntimeError("canonical repair program changed during read")
    if hashlib.sha256(payload).hexdigest() != EXPECTED_CONFIG_SHA:
        raise RuntimeError("canonical repair program SHA-256 differs")
    config = json.loads(payload)
    observed = tuple((row.get("name"), row.get("size_bytes"), row.get("sha256"))
                     for row in config.get("source_artifacts", []))
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS \
            or config.get("schema") != CONFIG_SCHEMA or observed != EXPECTED_ARTIFACTS \
            or config.get("lineage", {}).get("sealed_runtime_commit") != EXPECTED_PARENT:
        raise RuntimeError("checker frozen config keyset/schema/pins differ")
    return config


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


def _bind_private_target(parent_fd: int, target_fd: int, target_name: str) -> None:
    nofollow = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
    follow = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=True)
    opened = os.fstat(target_fd)
    if len({(row.st_dev, row.st_ino) for row in (nofollow, follow, opened)}) != 1 \
            or not all(stat.S_ISDIR(row.st_mode) for row in (nofollow, follow, opened)) \
            or any(stat.S_IMODE(row.st_mode) != 0o700 for row in (nofollow, follow, opened)):
        raise RuntimeError("private target parent entry does not bind inherited target fd")


def _read_artifacts(directory_fd: int, specs: Sequence[Mapping[str, Any]]):
    if sorted(os.listdir(directory_fd)) != sorted(ARTIFACT_NAMES):
        raise RuntimeError("publication exact entry set differs")
    data, stats = {}, {}
    for spec in specs:
        name = spec["name"]
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) \
                or stat.S_IMODE(before.st_mode) != 0o444 or before.st_nlink != 1:
            raise RuntimeError(f"artifact type/mode/nlink differs: {name}")
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        try:
            opened, payload = os.fstat(fd), _read_all(fd)
            after = os.fstat(fd)
        finally:
            os.close(fd)
        if _identity(before) != _identity(opened) or _identity(opened) != _identity(after) \
                or len(payload) != spec["size_bytes"] \
                or hashlib.sha256(payload).hexdigest() != spec["sha256"] \
                or json.loads(payload).get("schema") != spec["schema"]:
            raise RuntimeError(f"artifact stable bytes/schema differ: {name}")
        data[name], stats[name] = payload, after
    return data, stats


def _git_bytes(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{path}"], check=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def _git_mode(commit: str, path: str) -> str:
    fields = subprocess.run(["git", "-C", str(ROOT), "ls-tree", commit, "--", path], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True).stdout.split()
    if len(fields) < 4 or fields[3] != path:
        raise RuntimeError(f"historical Git row missing: {path}")
    return fields[0]


def _verify_historical(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    commit = config["lineage"]["sealed_runtime_commit"]
    pin = config["lineage"]["terminal_config"]
    if commit != EXPECTED_PARENT or manifest.get("runtime", {}).get("git_commit") != commit \
            or hashlib.sha256(_git_bytes(commit, pin["path"])).hexdigest() != pin["sha256"] \
            or manifest.get("config_sha256") != pin["sha256"]:
        raise RuntimeError("historical terminal config/runtime provenance differs")
    rows = manifest.get("runtime", {}).get("implementation_files")
    if not isinstance(rows, list) or [row.get("path") for row in rows] != list(SEALED_IMPLEMENTATION_PATHS):
        raise RuntimeError("sealed implementation rows differ")
    for row in rows:
        historical = _git_bytes(commit, row["path"])
        if _git_mode(commit, row["path"]) != "100644" or row.get("mode") != "0644" \
                or hashlib.sha256(historical).hexdigest() != row.get("sha256"):
            raise RuntimeError(f"historical implementation blob differs: {row['path']}")


def _verify_seal(config: Mapping[str, Any], data: Mapping[str, bytes]) -> None:
    specs = {row["name"]: row for row in config["source_artifacts"]}
    manifest, complete = json.loads(data["manifest.json"]), json.loads(data["COMPLETE"])
    rows = [{key: specs[name][key] for key in ("name", "schema", "sha256", "size_bytes")}
            for name in PAYLOAD_NAMES]
    if manifest.get("status") != "sealed" or manifest.get("files") != rows \
            or complete != {"manifest_sha256": specs["manifest.json"]["sha256"],
                            "schema": specs["COMPLETE"]["schema"], "status": "complete"}:
        raise RuntimeError("manifest/COMPLETE seal differs")
    _verify_historical(config, manifest)


def _verify_fixed_failure(config: Mapping[str, Any], data: Mapping[str, bytes]) -> None:
    result = json.loads(data["terminal_result.json"])
    fixed = config["checker_contract"]["verify_fixed_scientific_failure_metrics"]
    for field in ("status", "scientific_pass", "jointly_eligible_rows", "normalized_row_weight_ESS",
                  "maximum_single_normalized_row_weight", "positive_weight_parent_count",
                  "positive_weight_geometry_key_count", "positive_weight_fine_field_seed_count"):
        if result.get(field) != fixed[field]:
            raise RuntimeError(f"fixed terminal failure metric differs: {field}")
    if any(result.get(field) is not False for field in
           ("automatic_promotion", "RAMSES_authorized", "same_model_extension_authorized")):
        raise RuntimeError("terminal authorization flags differ")
    checks = result.get("checks", {})
    expected_checks = {
        "maximum_single_bridge_group_normalized_weight": True,
        "maximum_single_geometry_key_normalized_weight": True,
        "maximum_single_normalized_row_weight": True,
        "maximum_single_parent_normalized_weight": True,
        "minimum_bridge_group_weight_ESS": False,
        "minimum_geometry_key_weight_ESS": False,
        "minimum_jointly_eligible_rows": False,
        "minimum_normalized_row_weight_ESS": False,
        "minimum_parent_weight_ESS": False,
    }
    if checks != expected_checks:
        raise RuntimeError("terminal scientific gate statuses differ")
    if len(result.get("rows", [])) != fixed["row_count"] \
            or any(row.get("normalized_weight") != 0.0 or row.get("jointly_eligible_pair_count") != 0
                   for row in result.get("rows", [])):
        raise RuntimeError("terminal fixed row metrics differ")
    grouped = result.get("grouped_normalized_weights", {})
    if set(grouped) != set(fixed["grouped_identity_counts"]):
        raise RuntimeError("terminal grouped weights keyset differs")
    for name, count in fixed["grouped_identity_counts"].items():
        if len(grouped[name]) != count or any(row.get("normalized_weight") != 0.0 for row in grouped[name]):
            raise RuntimeError(f"terminal grouped weights differ: {name}")
    support = result.get("grouped_support", {})
    if set(support) != {"bridge", "fine_field_seed", "geometry", "parent"} \
            or any(row != {"ESS": 0.0, "maximum_weight": 0.0} for row in support.values()):
        raise RuntimeError("terminal grouped support differs")


def check_private_publication(config_path: Path, *, source_fd: int, parent_fd: int,
                              target_fd: int, target_name: str) -> dict[str, Any]:
    config = _load_config(config_path)
    _bind_private_target(parent_fd, target_fd, target_name)
    source_data, source_stats = _read_artifacts(source_fd, config["source_artifacts"])
    target_data, target_stats = _read_artifacts(target_fd, config["source_artifacts"])
    _bind_private_target(parent_fd, target_fd, target_name)
    for name in ARTIFACT_NAMES:
        if source_data[name] != target_data[name]:
            raise RuntimeError(f"target is not byte-identical: {name}")
        left, right = source_stats[name], target_stats[name]
        if (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino) \
                or left.st_nlink != 1 or right.st_nlink != 1:
            raise RuntimeError(f"source/target inode separation differs: {name}")
    _verify_seal(config, target_data)
    _verify_fixed_failure(config, target_data)
    _bind_private_target(parent_fd, target_fd, target_name)
    return {"status": "private_publication_repair_verified", "artifacts": 5}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-fd", type=int, required=True)
    parser.add_argument("--parent-fd", type=int, required=True)
    parser.add_argument("--target-fd", type=int, required=True)
    parser.add_argument("--target-name", required=True)
    args = parser.parse_args()
    result = check_private_publication(args.config, source_fd=args.source_fd,
                                       parent_fd=args.parent_fd, target_fd=args.target_fd,
                                       target_name=args.target_name)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
