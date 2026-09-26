#!/usr/bin/env python3
"""Independent checker for the sealed high-k terminal aggregation.

This module intentionally shares no aggregation or private validation helper
with the producer, so a producer-side formula error cannot self-certify.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "ouruniv-cf4-lg-highk-terminal-aggregation-program-v1"
INPUT_SCHEMA = "ouruniv-cf4-lg-highk-terminal-aggregation-input-manifest-v1"
P1_SCHEMA = "ouruniv-cf4-lg-highk-pair-recentered-p1-result-v1"
RESULT_SCHEMA = "ouruniv-cf4-lg-highk-terminal-aggregation-result-v1"
MANIFEST_SCHEMA = "ouruniv-cf4-lg-highk-terminal-aggregation-seal-manifest-v1"
COMPLETE_SCHEMA = "ouruniv-cf4-lg-highk-terminal-aggregation-complete-v1"
OUTPUT_NAMES = {
    "input_manifest.json", "pair_recentered_p1.json", "terminal_result.json",
    "manifest.json", "COMPLETE",
}
FIVE_P1_GATES = (
    "Virgo", "Coma", "LocalVoid", "BootesVoid", "ObserverEnvironment",
)
TOP_LEVEL_CONFIG_KEYS = {
    "schema", "status", "date", "purpose", "lineage", "inputs",
    "observed_integrity_anchor", "reuse_policy", "science_contract",
    "pair_contract", "aggregation_contract", "grouped_support_gates",
    "outputs", "resources", "execution", "authorization", "forbidden",
    "audit_sequence",
}
DEFAULT_CONFIG = ROOT / "config/cf4_lg_highk_terminal_aggregation_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("utf-8")


def _resolved(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or set(value) != TOP_LEVEL_CONFIG_KEYS \
            or value.get("schema") != CONFIG_SCHEMA \
            or value.get("status") != "frozen_user_authorized_terminal_aggregation":
        raise RuntimeError("terminal config identity or exact keyset changed")
    return value


def _regular_readonly(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or stat.S_IMODE(path.stat().st_mode) != 0o444:
        raise RuntimeError(f"sealed entry is not a regular 0444 file: {path}")


def _load_canonical(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"malformed sealed JSON: {path}") from error
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise RuntimeError(f"sealed JSON is not canonical compact JSON: {path}")
    return value


def _no_nonfinite(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for name, item in value.items():
            _no_nonfinite(item, f"{path}.{name}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _no_nonfinite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"sealed JSON contains a nonfinite number at {path}")


def _equal(left: Any, right: Any, path: str = "value") -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise RuntimeError(f"{path} keysets differ")
        for key in left:
            _equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise RuntimeError(f"{path} lengths differ")
        for index, (a, b) in enumerate(zip(left, right)):
            _equal(a, b, f"{path}[{index}]")
    elif left != right:
        raise RuntimeError(f"{path} values differ")


def _stable_logsumexp(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return -math.inf
    maximum = float(np.max(array))
    if not math.isfinite(maximum):
        return maximum
    return maximum + math.log(float(np.exp(array - maximum).sum()))


def _finite_record(value: float) -> dict[str, Any]:
    return {"finite": math.isfinite(value), "value": value if math.isfinite(value) else None}


def _normalize_once(log_weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(log_weights)
    weights = np.zeros(256, dtype=np.float64)
    if np.any(finite):
        weights[finite] = np.exp(
            log_weights[finite] - _stable_logsumexp(log_weights[finite])
        )
        weights /= np.sum(weights, dtype=np.float64)
    return weights


def _grouped_weights(
    rows: Sequence[Mapping[str, Any]], weights: np.ndarray, key: str,
) -> list[dict[str, Any]]:
    totals: dict[Any, float] = defaultdict(float)
    for row, weight in zip(rows, weights):
        raw = row[key]
        identity = tuple(raw) if isinstance(raw, list) else raw
        totals[identity] += float(weight)
    return [{
        "identity": list(identity) if isinstance(identity, tuple) else identity,
        "normalized_weight": totals[identity],
    } for identity in sorted(totals)]


def _ess_and_max(groups: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    positive = np.asarray([
        row["normalized_weight"] for row in groups if row["normalized_weight"] > 0.0
    ], dtype=np.float64)
    if not positive.size:
        return {"ESS": 0.0, "maximum_weight": 0.0}
    return {
        "ESS": float(1.0 / np.sum(positive**2)),
        "maximum_weight": float(np.max(positive)),
    }


def independent_aggregate(
    rows: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute every terminal statistic without importing producer code."""
    if len(rows) != 256 or [row.get("schedule_index") for row in rows] != list(range(256)):
        raise RuntimeError("independent aggregation lacks 256 ordered rows")
    eligible_pairs: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        if pair.get("p1_pass") is True:
            score = pair.get("log_likelihood")
            if type(score) not in (int, float) or not math.isfinite(float(score)):
                raise RuntimeError("P1-eligible pair likelihood is not finite")
            eligible_pairs[int(pair["schedule_index"])].append(pair)
    log_weights = np.full(256, -math.inf, dtype=np.float64)
    reports = []
    for row in rows:
        index = int(row["schedule_index"])
        denominator = len(row["loose_pair_ids"])
        eligible = eligible_pairs.get(index, [])
        log_evidence = -math.inf
        if denominator > 0 and eligible:
            log_evidence = _stable_logsumexp([
                float(pair["log_likelihood"]) for pair in eligible
            ]) - math.log(denominator)
        log_weight = -math.log(256.0) + log_evidence
        log_weights[index] = log_weight
        reports.append({
            **{name: row[name] for name in (
                "schedule_index", "parent_seed", "bridge_group", "geometry_key",
                "fine_field_seed",
            )},
            "all_loose_pair_count": denominator,
            "jointly_eligible_pair_count": len(eligible),
            "jointly_eligible_pair_ids": [
                [int(pair["halo_i"]), int(pair["halo_j"])] for pair in eligible
            ],
            "row_log_evidence": _finite_record(log_evidence),
            "unnormalized_row_log_weight": _finite_record(log_weight),
        })
    weights = _normalize_once(log_weights)
    for row, weight in zip(reports, weights):
        row["normalized_weight"] = float(weight)
    definitions = {
        "parent": "parent_seed", "geometry": "geometry_key",
        "fine_field_seed": "fine_field_seed", "bridge": "bridge_group",
    }
    grouped = {
        name: _grouped_weights(reports, weights, field)
        for name, field in definitions.items()
    }
    support = {name: _ess_and_max(group) for name, group in grouped.items()}
    row_ess = float(1.0 / np.sum(weights**2)) if np.any(weights) else 0.0
    row_max = float(np.max(weights, initial=0.0))
    eligible_rows = int(np.count_nonzero(np.isfinite(log_weights)))
    contract = config["aggregation_contract"]
    gates = config["grouped_support_gates"]
    checks = {
        "minimum_jointly_eligible_rows": eligible_rows >= int(contract["minimum_jointly_eligible_rows"]),
        "minimum_normalized_row_weight_ESS": row_ess >= float(contract["minimum_normalized_row_weight_ESS"]),
        "maximum_single_normalized_row_weight": row_max <= float(contract["maximum_single_normalized_row_weight"]),
        "minimum_parent_weight_ESS": support["parent"]["ESS"] >= float(gates["minimum_parent_weight_ESS"]),
        "maximum_single_parent_normalized_weight": support["parent"]["maximum_weight"] <= float(gates["maximum_single_parent_normalized_weight"]),
        "minimum_geometry_key_weight_ESS": support["geometry"]["ESS"] >= float(gates["minimum_geometry_key_weight_ESS"]),
        "maximum_single_geometry_key_normalized_weight": support["geometry"]["maximum_weight"] <= float(gates["maximum_single_geometry_key_normalized_weight"]),
        "minimum_bridge_group_weight_ESS": support["bridge"]["ESS"] >= float(gates["minimum_bridge_group_weight_ESS"]),
        "maximum_single_bridge_group_normalized_weight": support["bridge"]["maximum_weight"] <= float(gates["maximum_single_bridge_group_normalized_weight"]),
    }
    passed = all(checks.values())
    return {
        "schema": RESULT_SCHEMA,
        "status": (
            "complete_pass_terminal_aggregation_waiting_independent_review"
            if passed else "complete_scientific_fail_terminal_aggregation_closed"
        ),
        "scientific_pass": passed, "automatic_promotion": False,
        "RAMSES_authorized": False, "same_model_extension_authorized": False,
        "jointly_eligible_rows": eligible_rows,
        "normalized_row_weight_ESS": row_ess,
        "maximum_single_normalized_row_weight": row_max,
        "positive_weight_parent_count": sum(row["normalized_weight"] > 0.0 for row in grouped["parent"]),
        "positive_weight_geometry_key_count": sum(row["normalized_weight"] > 0.0 for row in grouped["geometry"]),
        "positive_weight_fine_field_seed_count": sum(row["normalized_weight"] > 0.0 for row in grouped["fine_field_seed"]),
        "grouped_support": support, "checks": checks,
        "grouped_normalized_weights": grouped, "rows": reports,
    }


def _verify_anchors(
    rows: Sequence[Mapping[str, Any]], intersections: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    observed = {
        "canonical_rows": len(rows),
        "rows_with_loose_pairs": sum(bool(row["loose_pair_ids"]) for row in rows),
        "loose_pairs": sum(len(row["loose_pair_ids"]) for row in rows),
        "rows_with_hard_pairs": sum(bool(row["hard_pair_ids"]) for row in rows),
        "hard_pairs": sum(len(row["hard_pair_ids"]) for row in rows),
        "hard_loose_same_identity_pairs": len(intersections),
        "hard_pairs_with_same_identity_loose_pair": sum(len(row["intersection_pair_ids"]) for row in rows),
        "recentered_p1_unique_parents": len({row["parent_seed"] for row in intersections}),
        "geometry_keys": len({tuple(row["geometry_key"]) for row in rows}),
        "bridge_group_counts": [sum(row["bridge_group"] == group for row in rows) for group in range(4)],
    }
    if observed != config["observed_integrity_anchor"] \
            or any(row["posterior_weight"] != 1.0 / 256.0 for row in rows):
        raise RuntimeError("independent observed anchors changed")


def _verify_inputs(inputs: Mapping[str, Any], production_root: Path) -> None:
    for row in inputs["rows"]:
        directory = production_root / f"row_{int(row['schedule_index']):03d}"
        if sha256_file(directory / "result.json") != row["result_sha256"] \
                or sha256_file(directory / "halos.npz") != row["halo_catalogue_sha256"]:
            raise RuntimeError("sealed production input changed")
    for parent in inputs["parent_fields"]:
        if sha256_file(Path(parent["path"])) != parent["sha256"]:
            raise RuntimeError("sealed parent input changed")


def _verify_p1_manifest_bindings(
    intersections: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]],
) -> None:
    """Independently bind every P1 row to its exact sealed input-manifest row."""
    fields = (
        "parent_seed", "bridge_group", "geometry_key", "fine_field_seed",
        "midpoint_mpc_h", "log_likelihood",
    )
    manifest_map: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    for row in intersections:
        identity = (int(row["schedule_index"]), int(row["halo_i"]), int(row["halo_j"]))
        if identity in manifest_map:
            raise RuntimeError("input manifest contains a duplicate pair identity")
        manifest_map[identity] = row
    pair_map: dict[tuple[int, int, int], Mapping[str, Any]] = {}
    for pair in pairs:
        identity = (int(pair["schedule_index"]), int(pair["halo_i"]), int(pair["halo_j"]))
        if identity in pair_map or identity not in manifest_map:
            raise RuntimeError("P1 pair identity is duplicate or absent from input manifest")
        pair_map[identity] = pair
        manifest = manifest_map[identity]
        for field in fields:
            if pair.get(field) != manifest.get(field):
                raise RuntimeError(f"P1 pair {identity} differs from input manifest field {field}")
        midpoint = np.asarray(manifest["midpoint_mpc_h"], dtype=np.float64)
        observed_offset = np.asarray(pair.get("observer_offset_mpc_h"), dtype=np.float64)
        expected_offset = ((midpoint - 192.0 + 192.0) % 384.0) - 192.0
        if midpoint.shape != (3,) or observed_offset.shape != (3,) \
                or not np.all(np.isfinite(midpoint)) or not np.all(np.isfinite(observed_offset)) \
                or not np.array_equal(observed_offset, expected_offset):
            raise RuntimeError(f"P1 pair {identity} observer offset differs from periodic midpoint")
        metrics = pair.get("p1_metrics")
        if not isinstance(metrics, dict) \
                or metrics.get("gates") != pair.get("p1_gates") \
                or metrics.get("pass") is not pair.get("p1_pass"):
            raise RuntimeError(f"P1 pair {identity} metrics gates/pass differ from top level")
    if set(pair_map) != set(manifest_map):
        raise RuntimeError("P1 and input-manifest pair identity maps differ")


def _current_implementation(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(config["lineage"]["required_commit_scope"]):
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("implementation path changed")
        rows.append({
            "path": relative, "sha256": sha256_file(path),
            "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
        })
    return rows


def _verify_runtime(runtime: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if set(runtime) != {"slurm", "python", "git_commit", "implementation_files"}:
        raise RuntimeError("runtime provenance keyset changed")
    resources = config["resources"]
    slurm = runtime["slurm"]
    expected_slurm_keys = {
        "job_id", "node", "partition", "nodes", "tasks", "cpus_per_task",
        "memory_MiB", "gpu_count", "visible_GPU",
    }
    if set(slurm) != expected_slurm_keys or not str(slurm["job_id"]).isdigit() \
            or not slurm["node"] or slurm["partition"] not in resources["partitions"].split(",") \
            or (slurm["nodes"], slurm["tasks"], slurm["cpus_per_task"], slurm["memory_MiB"], slurm["gpu_count"]) != (1, 1, 16, 20480, 1):
        raise RuntimeError("sealed Slurm allocation differs from the contract")
    gpu = slurm["visible_GPU"]
    if set(gpu) != {"uuid", "name", "memory_MiB"} or not gpu["uuid"] \
            or not gpu["name"] or int(gpu["memory_MiB"]) < 40960:
        raise RuntimeError("sealed GPU provenance differs from the contract")
    if runtime["python"] != {
        "executable": resources["python_executable"],
        "version": resources["python_version"],
        "packages": resources["python_packages"],
    }:
        raise RuntimeError("sealed Python provenance differs from the circle contract")
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()
    if runtime["git_commit"] != commit or len(commit) != 40:
        raise RuntimeError("sealed Git commit differs from current HEAD")
    if runtime["implementation_files"] != _current_implementation(config):
        raise RuntimeError("sealed exact implementation hashes or modes changed")
    if "SLURM_JOB_ID" in os.environ:
        current = {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "cpus_per_task": int(os.environ.get("SLURM_CPUS_PER_TASK", "0")),
            "memory_MiB": int(os.environ.get("SLURM_MEM_PER_NODE", "0")),
        }
        if any(slurm[name] != value for name, value in current.items()):
            raise RuntimeError("checker Slurm context differs from producer context")


def check_output(config_path: Path, output_root: Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_config(config_path)
    for name, spec in config["inputs"].items():
        if "sha256" in spec and sha256_file(_resolved(spec["path"])) != spec["sha256"]:
            raise RuntimeError(f"pinned input changed: {name}")
    output = Path(output_root) if output_root is not None else Path(config["outputs"]["canonical_root"])
    if output.is_symlink() or not output.is_dir() or stat.S_IMODE(output.stat().st_mode) != 0o555:
        raise RuntimeError("canonical output is not a real 0555 directory")
    if {path.name for path in output.iterdir()} != OUTPUT_NAMES:
        raise RuntimeError("canonical output does not have the exact sealed entry set")
    for name in OUTPUT_NAMES:
        _regular_readonly(output / name)
    values = {name: _load_canonical(output / name) for name in OUTPUT_NAMES}
    for name, value in values.items():
        _no_nonfinite(value, name)
    inputs, p1, result = (
        values["input_manifest.json"], values["pair_recentered_p1.json"],
        values["terminal_result.json"],
    )
    manifest, complete = values["manifest.json"], values["COMPLETE"]
    schemas = {
        "input_manifest.json": INPUT_SCHEMA,
        "pair_recentered_p1.json": P1_SCHEMA,
        "terminal_result.json": RESULT_SCHEMA,
    }
    if any(values[name].get("schema") != schema for name, schema in schemas.items()):
        raise RuntimeError("one or more payload schemas changed")
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "sealed":
        raise RuntimeError("manifest is not sealed")
    if complete != {"schema": COMPLETE_SCHEMA, "status": "complete", "manifest_sha256": sha256_file(output / "manifest.json")}:
        raise RuntimeError("COMPLETE does not seal this manifest")
    if manifest.get("config") != str(config_path.resolve()) or manifest.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("manifest config provenance changed")
    expected_records = [{
        "name": name, "sha256": sha256_file(output / name),
        "size_bytes": (output / name).stat().st_size, "schema": schemas[name],
    } for name in sorted(schemas)]
    if manifest.get("files") != expected_records:
        raise RuntimeError("manifest payload hashes, sizes, or schemas changed")
    _verify_runtime(manifest.get("runtime", {}), config)

    rows, pairs, intersections = (
        inputs.get("rows"), p1.get("pairs"),
        inputs.get("hard_loose_same_identity_pairs"),
    )
    if not isinstance(rows, list) or not isinstance(pairs, list) or not isinstance(intersections, list):
        raise RuntimeError("sealed input or P1 rows are malformed")
    _verify_anchors(rows, intersections, config)
    _verify_inputs(inputs, Path(config["inputs"]["production_root"]["path"]))
    if p1.get("pair_count") != 18 or len(pairs) != 18 or p1.get("unique_parent_count") != 12 \
            or p1.get("exact_gate_names") != list(FIVE_P1_GATES):
        raise RuntimeError("sealed P1 anchors changed")
    identities = []
    for pair in pairs:
        if set(pair.get("p1_gates", {})) != set(FIVE_P1_GATES) \
                or pair.get("p1_pass") is not all(pair["p1_gates"].values()):
            raise RuntimeError("P1 exact-five gate conjunction changed")
        identities.append((pair["schedule_index"], pair["halo_i"], pair["halo_j"]))
    sealed_intersections = [(row["schedule_index"], row["halo_i"], row["halo_j"]) for row in intersections]
    row_intersections = sorted(
        (row["schedule_index"], identity[0], identity[1])
        for row in rows for identity in row["intersection_pair_ids"]
    )
    if identities != sorted(identities) or len(set(identities)) != 18 \
            or sealed_intersections != sorted(sealed_intersections) \
            or identities != sealed_intersections or identities != row_intersections:
        raise RuntimeError("sealed hard/loose/P1 identities differ")
    _verify_p1_manifest_bindings(intersections, pairs)
    if result.get("config_sha256") != sha256_file(config_path) \
            or result.get("input_manifest_sha256") != sha256_file(output / "input_manifest.json") \
            or result.get("pair_recentered_p1_sha256") != sha256_file(output / "pair_recentered_p1.json"):
        raise RuntimeError("terminal result payload provenance changed")
    recomputed = independent_aggregate(rows, pairs, config)
    _equal({key: result[key] for key in recomputed}, recomputed, "terminal_result")
    if type(result.get("seconds")) not in (int, float) or result["seconds"] <= 0.0:
        raise RuntimeError("terminal runtime is invalid")
    return {
        "status": result["status"], "scientific_pass": result["scientific_pass"],
        "canonical_rows": len(rows), "pair_count": len(pairs),
        "jointly_eligible_rows": result["jointly_eligible_rows"],
        "normalized_row_weight_ESS": result["normalized_row_weight_ESS"],
        "maximum_single_normalized_row_weight": result["maximum_single_normalized_row_weight"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(check_output(args.config, args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
