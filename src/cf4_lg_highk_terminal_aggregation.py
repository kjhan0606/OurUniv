#!/usr/bin/env python3
"""Terminal pair-recentred P1 and z=0 aggregation for frozen high-k rows.

The runner is deliberately terminal: it evaluates every frozen row, writes a
sealed scientific pass or fail record, and never selects or promotes a row.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
import errno
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_cf4_lg_highk_streaming_pm_production_v1 import audit_production_root  # noqa: E402
from cf4_lg_highk_covariance_cache import sha256_file  # noqa: E402
from cf4_lg_highk_streaming_forward import _load_schedule, _parent_entries  # noqa: E402
from cf4_lg_z0_likelihood import (  # noqa: E402
    logmeanexp, logsumexp, pair_log_likelihood, score_catalog,
)
from cf4_p2_screen import find_pairs, load_config, rank_score  # noqa: E402
from cf4_parent_p1 import score_member  # noqa: E402


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
CATALOG_ABS_TOLERANCE = {
    "m1_fof_msun_h": 131072.0,
    "m2_fof_msun_h": 131072.0,
    "masses_msun_h": 262144.0,
    "mass_ratio": 9.5367431640625e-7,
    "separation_mpc_h": 3.0517578125e-5,
    "midpoint_mpc_h": 1.52587890625e-5,
    "midpoint_offset_vector_mpc_h": 1.52587890625e-5,
    "midpoint_offset_mpc_h": 3.0517578125e-5,
    "isolation_mpc_h": 3.0517578125e-5,
    "peculiar_radial_velocity_km_s": 0.0078125,
    "total_radial_velocity_km_s": 0.0078125,
    "tangential_velocity_km_s": 0.0078125,
    "ranking_score": 6.103515625e-5,
    "host_separation_mpc_h": 3.0517578125e-5,
    "mass_fof_msun_h": 131072.0,
}
LIKELIHOOD_ATOL = 1e-8
P1_FLOAT32_ULP_FIELDS = {
    "target_delta", "peak_delta", "mean_delta", "centre_delta",
    "probe_mean_delta", "excess_mass_msun_h",
}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("utf-8")


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(_canonical_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(path, 0o444)
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _pinned(spec: Mapping[str, Any], name: str) -> Path:
    try:
        path = _resolve(spec["path"])
        expected = str(spec["sha256"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"input {name!r} needs path and sha256") from error
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{name} SHA256 changed: {actual} != {expected}")
    return path


def load_terminal_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    exact = {
        "schema", "status", "date", "purpose", "lineage", "inputs",
        "observed_integrity_anchor", "reuse_policy", "science_contract",
        "pair_contract", "aggregation_contract", "grouped_support_gates",
        "outputs", "resources", "execution", "authorization", "forbidden",
        "audit_sequence",
    }
    if not isinstance(config, dict) or set(config) != exact:
        raise ValueError("terminal config does not have the exact frozen top-level keyset")
    if config["schema"] != CONFIG_SCHEMA:
        raise ValueError("unexpected terminal config schema")
    if config["status"] != "frozen_user_authorized_terminal_aggregation":
        raise PermissionError("terminal aggregation program is not frozen and authorized")
    if config["authorization"].get("terminal_aggregation_execution") is not True:
        raise PermissionError("terminal aggregation execution is not authorized")
    for name in ("automatic_promotion", "RAMSES", "same_model_extension"):
        if config["authorization"].get(name) is not False:
            raise ValueError(f"authorization.{name} must remain false")
    return config


def validate_two_commit_lineage_values(
    config: Mapping[str, Any], *, head: str, upstream: str,
    head_parents: Sequence[str], baseline_parents: Sequence[str],
    baseline_rows: Sequence[tuple[str, str]],
    correction_rows: Sequence[tuple[str, str]],
    baseline_modes: Mapping[str, str], head_modes: Mapping[str, str],
) -> None:
    """Pure validation of the exact baseline-plus-correction commit grammar."""
    contract = config["lineage"]["two_commit_execution_lineage"]
    baseline = contract["baseline_commit"]
    parent = contract["baseline_parent_commit"]
    expected_baseline = sorted(
        ("A", path) for path in contract["baseline_exact_added_paths"]
    )
    expected_correction = sorted(
        ("M", path) for path in contract["correction_exact_modified_paths"]
    )
    if head != upstream or list(head_parents) != [baseline] \
            or list(baseline_parents) != [parent]:
        raise RuntimeError("runtime HEAD/upstream or two-commit parent lineage changed")
    if sorted(baseline_rows) != expected_baseline:
        raise RuntimeError("baseline commit is not the exact original six additions")
    if sorted(correction_rows) != expected_correction:
        raise RuntimeError("correction commit is not the exact four modifications")
    required_mode = contract["required_mode"]
    if set(baseline_modes) != set(contract["baseline_exact_added_paths"]) \
            or set(head_modes) != set(contract["correction_exact_modified_paths"]) \
            or any(mode != required_mode for mode in baseline_modes.values()) \
            or any(mode != required_mode for mode in head_modes.values()):
        raise RuntimeError("baseline or correction file modes changed")


def _git_text(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *arguments], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()


def _name_status_rows(older: str, newer: str) -> list[tuple[str, str]]:
    output = _git_text(
        "diff", "--no-renames", "--name-status", older, newer, "--",
    )
    rows: list[tuple[str, str]] = []
    for line in output.splitlines() if output else []:
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] not in {"A", "M", "D"}:
            raise RuntimeError("Git diff contains a rename, copy, or malformed status row")
        rows.append((fields[0], fields[1]))
    return rows


def _tree_modes(commit: str, paths: Sequence[str]) -> dict[str, str]:
    modes: dict[str, str] = {}
    for path in paths:
        fields = _git_text("ls-tree", commit, "--", path).split()
        if len(fields) < 4 or fields[3] != path:
            raise RuntimeError(f"Git tree entry is absent or malformed: {path}")
        modes[path] = fields[0]
    return modes


def validate_two_commit_lineage(config: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only runtime validation called by the Slurm preflight."""
    contract = config["lineage"]["two_commit_execution_lineage"]
    baseline = contract["baseline_commit"]
    parent = contract["baseline_parent_commit"]
    head = _git_text("rev-parse", "HEAD")
    upstream = _git_text("rev-parse", "@{upstream}")
    head_line = _git_text("rev-list", "--parents", "-n", "1", head).split()
    baseline_line = _git_text("rev-list", "--parents", "-n", "1", baseline).split()
    baseline_paths = contract["baseline_exact_added_paths"]
    correction_paths = contract["correction_exact_modified_paths"]
    validate_two_commit_lineage_values(
        config, head=head, upstream=upstream,
        head_parents=head_line[1:], baseline_parents=baseline_line[1:],
        baseline_rows=_name_status_rows(parent, baseline),
        correction_rows=_name_status_rows(baseline, head),
        baseline_modes=_tree_modes(baseline, baseline_paths),
        head_modes=_tree_modes(head, correction_paths),
    )
    status = subprocess.run(
        [
            "git", "-C", str(ROOT), "status", "--porcelain=v1", "-z",
            "--untracked-files=all", "--", ".", ":(exclude)scripts/tripwire/**",
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout
    if status:
        raise RuntimeError("worktree is not clean outside scripts/tripwire/")
    for path, expected in contract["untouched_files"].items():
        if sha256_file(ROOT / path) != expected:
            raise RuntimeError(f"untouched lineage file hash changed: {path}")
    return {"head": head, "baseline": baseline, "baseline_parent": parent}


def _pair_id(pair: Mapping[str, Any]) -> tuple[int, int]:
    i, j = int(pair["halo_i"]), int(pair["halo_j"])
    if not 0 <= i < j:
        raise RuntimeError("pair identity is not a sorted unordered halo pair")
    return i, j


def _pair_id_json(identity: tuple[int, int]) -> list[int]:
    return [int(identity[0]), int(identity[1])]


def _serialization_close(name: str, fresh: Any, stored: Any, path: str) -> None:
    """Compare only measured float32-serialization-sensitive catalogue fields."""
    tolerance = CATALOG_ABS_TOLERANCE[name]
    left = np.asarray(fresh, dtype=np.float64)
    right = np.asarray(stored, dtype=np.float64)
    if left.shape != right.shape or not np.all(np.isfinite(left)) \
            or not np.all(np.isfinite(right)) \
            or not np.all(np.abs(left - right) <= tolerance):
        raise RuntimeError(f"{path} exceeds measured serialization ULP bound {tolerance}")


def _compare_catalogue_pair(
    fresh: Mapping[str, Any], stored: Mapping[str, Any], *, kind: str, path: str,
    likelihood: Mapping[str, Any] | None = None,
) -> None:
    if set(fresh) != set(stored):
        raise RuntimeError(f"{path} keysets differ")
    if _pair_id(fresh) != _pair_id(stored):
        raise RuntimeError(f"{path} pair identity differs")
    exact = {"halo_i", "halo_j"}
    ignored_likelihood = {"log_likelihood", "log_likelihood_components"}
    for name in fresh:
        if name in exact or name in ignored_likelihood:
            continue
        if name == "m33_candidate":
            left, right = fresh[name], stored[name]
            if (left is None) != (right is None):
                raise RuntimeError(f"{path}.m33_candidate presence differs")
            if left is not None:
                if set(left) != set(right) or left["halo_index"] != right["halo_index"] \
                        or left["host_index"] != right["host_index"]:
                    raise RuntimeError(f"{path}.m33_candidate identity differs")
                for field in ("mass_fof_msun_h", "host_separation_mpc_h"):
                    _serialization_close(field, left[field], right[field], f"{path}.m33_candidate.{field}")
            continue
        if name not in CATALOG_ABS_TOLERANCE:
            raise RuntimeError(f"{path}.{name} has no preregistered serialization bound")
        _serialization_close(name, fresh[name], stored[name], f"{path}.{name}")
    if kind == "loose":
        if likelihood is None:
            raise RuntimeError("loose catalogue comparison lacks likelihood definition")
        score, components = pair_log_likelihood(stored, likelihood["z0_likelihood"])
        if not math.isclose(score, float(stored["log_likelihood"]), rel_tol=0.0, abs_tol=LIKELIHOOD_ATOL):
            raise RuntimeError(f"{path}.log_likelihood formula differs")
        if set(components) != set(stored["log_likelihood_components"]):
            raise RuntimeError(f"{path}.log_likelihood_components keyset differs")
        for name, value in components.items():
            if not math.isclose(value, float(stored["log_likelihood_components"][name]), rel_tol=0.0, abs_tol=LIKELIHOOD_ATOL):
                raise RuntimeError(f"{path}.log_likelihood_components.{name} differs")
def _compare_catalogues(
    fresh: Sequence[Mapping[str, Any]], stored: Sequence[Mapping[str, Any]],
    *, kind: str, path: str, likelihood: Mapping[str, Any] | None = None,
) -> None:
    if len(fresh) != len(stored):
        raise RuntimeError(f"{path} pair counts differ")
    fresh_ids = [_pair_id(pair) for pair in fresh]
    stored_ids = [_pair_id(pair) for pair in stored]
    if fresh_ids != stored_ids:
        raise RuntimeError(f"{path} ordered pair identities differ")
    for index, (left, right) in enumerate(zip(fresh, stored)):
        _compare_catalogue_pair(
            left, right, kind=kind, path=f"{path}[{index}]", likelihood=likelihood,
        )


def _p1_values_close(left: Any, right: Any, *, path: str = "P1") -> None:
    """Compare legacy P1 using small field-aware float32 ULP bounds."""
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            raise RuntimeError(f"{path} keysets differ")
        for key in left:
            _p1_values_close(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise RuntimeError(f"{path} lengths differ")
        for index, (a, b) in enumerate(zip(left, right)):
            _p1_values_close(a, b, path=f"{path}[{index}]")
        return
    if type(left) is bool or type(right) is bool or left is None or right is None \
            or isinstance(left, (str, int)) or isinstance(right, (str, int)):
        if left != right:
            raise RuntimeError(f"{path} values differ")
        return
    if isinstance(left, float) and isinstance(right, float):
        leaf = path.rsplit(".", 1)[-1]
        if "percentile" in leaf:
            tolerance = 1e-10
            policy = "percentile absolute bound"
        elif leaf == "nearest_box_face_mpc_h":
            tolerance = 4.0 * max(math.ulp(float(left)), math.ulp(float(right)))
            policy = "binary64 ULP bound"
        elif leaf in P1_FLOAT32_ULP_FIELDS or ".mean_delta_profile." in path:
            scale = np.float32(max(abs(left), abs(right), 1.0))
            tolerance = float(8.0 * abs(np.spacing(scale)))
            policy = "float32 ULP bound"
        else:
            tolerance = 0.0
            policy = "exact bound"
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
            raise RuntimeError(f"{path} exceeds P1 {policy} {tolerance}")
        return
    if left != right:
        raise RuntimeError(f"{path} values differ")


def _recompute_catalogues(
    *, production_root: Path, hard_config: Mapping[str, Any],
    likelihood_program: Mapping[str, Any], box_size: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute both pair catalogues and return row inputs and intersections."""
    centre = np.full(3, box_size / 2.0, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    intersections: list[dict[str, Any]] = []
    for index in range(256):
        row_dir = production_root / f"row_{index:03d}"
        result_path, halo_path = row_dir / "result.json", row_dir / "halos.npz"
        stored = json.loads(result_path.read_text())
        with np.load(halo_path, allow_pickle=False) as item:
            halos = {
                "pos": np.asarray(item["halo_pos"]),
                "vel": np.asarray(item["halo_vel"]),
                "mass": np.asarray(item["halo_mass"]),
            }
        fresh_hard = find_pairs(
            halos, centre, hard_config["screen"], hard_config["m33_subpeak_gate"]
        )
        for pair in fresh_hard:
            pair["ranking_score"] = rank_score(pair, hard_config["ranking"])
        fresh_hard.sort(key=lambda pair: pair["ranking_score"])
        fresh_loose = score_catalog(
            halos["pos"], halos["vel"], halos["mass"], centre=centre,
            box_size=box_size, program=likelihood_program,
        )
        _compare_catalogues(
            fresh_hard, stored["hard_p2_pairs"], kind="hard",
            path=f"row_{index}.hard",
        )
        _compare_catalogues(
            fresh_loose["candidate_pairs"], stored["z0_likelihood"]["candidate_pairs"],
            kind="loose", path=f"row_{index}.loose", likelihood=likelihood_program,
        )
        stored_scores = np.asarray([
            pair["log_likelihood"] for pair in stored["z0_likelihood"]["candidate_pairs"]
        ], dtype=np.float64)
        expected_mixture = logmeanexp(stored_scores)
        if not math.isclose(
            expected_mixture, float(stored["z0_likelihood"]["log_likelihood"]),
            rel_tol=0.0, abs_tol=LIKELIHOOD_ATOL,
        ):
            raise RuntimeError(f"row_{index}.loose mixture likelihood differs")
        if stored["z0_likelihood"]["n_candidate_pairs"] != len(stored_scores) \
                or stored["z0_likelihood"]["best_pair"] != (
                    stored["z0_likelihood"]["candidate_pairs"][0]
                    if len(stored_scores) else None
                ):
            raise RuntimeError(f"row_{index}.loose count or best-pair identity differs")
        if stored.get("hard_p2_pass") is not bool(stored["hard_p2_pairs"]):
            raise RuntimeError(f"row_{index}.hard pass flag differs from exact pair count")
        for pair_number, pair in enumerate(stored["hard_p2_pairs"]):
            expected_rank = rank_score(pair, hard_config["ranking"])
            if not math.isclose(
                expected_rank, float(pair["ranking_score"]),
                rel_tol=0.0, abs_tol=LIKELIHOOD_ATOL,
            ):
                raise RuntimeError(f"row_{index}.hard[{pair_number}].ranking_score differs")

        hard_by_id = {_pair_id(pair): pair for pair in stored["hard_p2_pairs"]}
        loose_by_id = {
            _pair_id(pair): pair for pair in stored["z0_likelihood"]["candidate_pairs"]
        }
        if len(hard_by_id) != len(fresh_hard) or len(loose_by_id) != len(stored_scores):
            raise RuntimeError(f"row {index} contains duplicate pair identities")
        if not set(hard_by_id).issubset(loose_by_id):
            missing = sorted(set(hard_by_id).difference(loose_by_id))
            raise RuntimeError(f"row {index} hard-P2 pair lacks same-id loose pair: {missing}")
        common = sorted(set(hard_by_id).intersection(loose_by_id))
        row_record = {
            "schedule_index": index,
            "parent_seed": int(stored["parent_seed"]),
            "bridge_group": int(stored["group_id"]),
            "geometry_key": [int(value) for value in stored["geometry_key"]],
            "fine_field_seed": int(stored["fine_field_seed"]),
            "posterior_weight": float(stored["posterior_weight"]),
            "result_sha256": sha256_file(result_path),
            "halo_catalogue_sha256": sha256_file(halo_path),
            "hard_pair_ids": [_pair_id_json(value) for value in sorted(hard_by_id)],
            "loose_pair_ids": [_pair_id_json(value) for value in sorted(loose_by_id)],
            "intersection_pair_ids": [_pair_id_json(value) for value in common],
        }
        rows.append(row_record)
        for identity in common:
            loose = loose_by_id[identity]
            intersections.append({
                "schedule_index": index,
                "parent_seed": row_record["parent_seed"],
                "bridge_group": row_record["bridge_group"],
                "geometry_key": row_record["geometry_key"],
                "fine_field_seed": row_record["fine_field_seed"],
                "halo_i": identity[0], "halo_j": identity[1],
                "midpoint_mpc_h": [float(value) for value in loose["midpoint_mpc_h"]],
                "log_likelihood": float(loose["log_likelihood"]),
            })
    return rows, intersections


def _assert_observed_anchors(
    rows: Sequence[Mapping[str, Any]], intersections: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    anchor = config["observed_integrity_anchor"]
    observed = {
        "canonical_rows": len(rows),
        "rows_with_loose_pairs": sum(bool(row["loose_pair_ids"]) for row in rows),
        "loose_pairs": sum(len(row["loose_pair_ids"]) for row in rows),
        "rows_with_hard_pairs": sum(bool(row["hard_pair_ids"]) for row in rows),
        "hard_pairs": sum(len(row["hard_pair_ids"]) for row in rows),
        "hard_loose_same_identity_pairs": len(intersections),
        "hard_pairs_with_same_identity_loose_pair": sum(
            len(row["intersection_pair_ids"]) for row in rows
        ),
        "recentered_p1_unique_parents": len({row["parent_seed"] for row in intersections}),
        "geometry_keys": len({tuple(row["geometry_key"]) for row in rows}),
        "bridge_group_counts": [
            sum(row["bridge_group"] == group for row in rows) for group in range(4)
        ],
    }
    if observed != anchor:
        raise RuntimeError(f"observed production anchors changed: {observed} != {anchor}")
    if any(row["posterior_weight"] != 1.0 / 256.0 for row in rows):
        raise RuntimeError("production rows are not exact equal-weight 1/256 draws")


def _periodic_offset(midpoint: Sequence[float], box_size: float) -> np.ndarray:
    delta = np.asarray(midpoint, np.float64) - box_size / 2.0
    return (delta + box_size / 2.0) % box_size - box_size / 2.0


def _p1_science(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: metrics[name] for name in (
            "clusters", "secondary_cluster_anchors", "local_void", "bootes_void",
            "observer_environment", "gates", "n_gates_passed", "pass",
        )
    }


def _load_parent_field(parent: Mapping[str, Any]) -> tuple[np.ndarray, float, dict[str, float]]:
    parent_path = Path(parent["parent_field"])
    if sha256_file(parent_path) != parent["parent_field_sha256"]:
        raise RuntimeError(f"parent field {parent['seed']} SHA256 changed")
    with np.load(parent_path, allow_pickle=False) as item:
        seed = int(item["sample_seed"])
        initial = np.asarray(item["s_out"], dtype=np.float32)
        nmesh, spacing, box_size = int(item["N"]), float(item["spacing"]), float(item["L"])
        cosmology = {
            "Om": float(item["Om"]), "Ob": float(item["Ob"]),
            "h": float(item["hh"]), "A_s_1e9": float(item["A_s_1e9"]),
            "ns": float(item["ns"]),
        }
    if seed != int(parent["seed"]) or nmesh != 192 or not math.isclose(box_size, 384.0):
        raise RuntimeError("parent field identity or N192 geometry changed")
    return initial, spacing, cosmology


def _make_parent_forward(cosmology: Mapping[str, float]) -> Any:
    import jax.numpy as jnp
    from mock_pipeline import make_forward
    _, _, forward = make_forward(
        192, 2.0, jnp.float32, return_dens=True, cosmology=dict(cosmology),
    )
    return forward


def _forward_parent_density(
    parent: Mapping[str, Any], config: Mapping[str, Any], *, forward: Any,
    expected_cosmology: Mapping[str, float],
) -> tuple[np.ndarray, float, float]:
    initial, spacing, cosmology = _load_parent_field(parent)
    if cosmology != dict(expected_cosmology) or not math.isclose(spacing, 2.0):
        raise RuntimeError("N192 parent cosmology or spacing changed within aggregation")
    import jax.numpy as jnp
    density, _ = forward(jnp.asarray(initial))
    if hasattr(density, "block_until_ready"):
        density.block_until_ready()
    smoothed = gaussian_filter(
        np.asarray(density, dtype=np.float32),
        float(config["density_smoothing_mpc_h"]) / spacing,
        mode="wrap",
    )
    delta = smoothed / np.mean(smoothed, dtype=np.float64) - 1.0
    del density, smoothed, initial
    return np.asarray(delta, dtype=np.float32), spacing, float(cosmology["Om"])


def evaluate_pair_recentered_p1(
    *, intersections: Sequence[Mapping[str, Any]], parent_entries: Mapping[int, Mapping[str, Any]],
    p1_config: Mapping[str, Any], legacy_rows: Mapping[int, Mapping[str, Any]],
    box_size: float,
) -> list[dict[str, Any]]:
    """Forward one N192 parent at a time and evaluate every same-id pair."""
    by_parent: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in intersections:
        by_parent[int(pair["parent_seed"])].append(pair)
    output: list[dict[str, Any]] = []
    parent_seeds = sorted(by_parent)
    if not parent_seeds:
        return output
    first_seed = parent_seeds[0]
    if first_seed not in parent_entries:
        raise RuntimeError(f"parent {first_seed} lacks a parent-response entry")
    first_initial, first_spacing, cosmology = _load_parent_field(parent_entries[first_seed])
    del first_initial
    if not math.isclose(first_spacing, 2.0):
        raise RuntimeError("first N192 parent spacing changed")
    forward = _make_parent_forward(cosmology)
    for parent_seed in parent_seeds:
        if parent_seed not in parent_entries or parent_seed not in legacy_rows:
            raise RuntimeError(f"parent {parent_seed} lacks a sealed P1 identity")
        delta, spacing, omega_m = _forward_parent_density(
            parent_entries[parent_seed], p1_config, forward=forward,
            expected_cosmology=cosmology,
        )
        zero = score_member(delta, spacing, p1_config, omega_m=omega_m)
        _p1_values_close(
            _p1_science(zero), _p1_science(legacy_rows[parent_seed]),
            path=f"parent_{parent_seed}.zero_offset_P1",
        )
        for pair in sorted(
            by_parent[parent_seed],
            key=lambda row: (row["schedule_index"], row["halo_i"], row["halo_j"]),
        ):
            offset = _periodic_offset(pair["midpoint_mpc_h"], box_size)
            metrics = score_member(
                delta, spacing, p1_config, omega_m=omega_m, observer_offset=offset,
            )
            gates = metrics.get("gates")
            if not isinstance(gates, dict) or set(gates) != set(FIVE_P1_GATES):
                raise RuntimeError("pair-recentred P1 did not return the exact five gates")
            passed = all(gates[name] is True for name in FIVE_P1_GATES)
            if bool(metrics.get("pass")) != passed:
                raise RuntimeError("P1 pass differs from conjunction of exact five gates")
            output.append({
                **dict(pair),
                "observer_offset_mpc_h": offset.tolist(),
                "p1_gates": {name: bool(gates[name]) for name in FIVE_P1_GATES},
                "p1_pass": passed,
                "p1_metrics": _p1_science(metrics),
            })
        del delta
    output.sort(key=lambda row: (row["schedule_index"], row["halo_i"], row["halo_j"]))
    return output


def _finite_json(value: float) -> dict[str, Any]:
    return {"finite": bool(math.isfinite(value)), "value": float(value) if math.isfinite(value) else None}


def _normalized(log_values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(log_values)
    weights = np.zeros(log_values.size, dtype=np.float64)
    if np.any(finite):
        weights[finite] = np.exp(log_values[finite] - logsumexp(log_values[finite]))
        # Finish the single normalization in linear space so grouped sums and
        # preregistered equality-bound gates do not depend on last-bit drift.
        weights /= np.sum(weights, dtype=np.float64)
    return weights


def _group_weights(rows: Sequence[Mapping[str, Any]], weights: np.ndarray, key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, float] = defaultdict(float)
    for row, weight in zip(rows, weights):
        raw = row[key]
        group = tuple(raw) if isinstance(raw, list) else raw
        grouped[group] += float(weight)
    result = []
    for group in sorted(grouped):
        identity: Any = list(group) if isinstance(group, tuple) else group
        result.append({"identity": identity, "normalized_weight": grouped[group]})
    return result


def _ess_max(groups: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    positive = np.asarray(
        [row["normalized_weight"] for row in groups if row["normalized_weight"] > 0.0],
        dtype=np.float64,
    )
    if positive.size == 0:
        return 0.0, 0.0
    return float(1.0 / np.sum(positive**2)), float(np.max(positive))


def aggregate_terminal(
    rows: Sequence[Mapping[str, Any]], p1_pairs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen full-denominator mixture and grouped support gates."""
    if len(rows) != 256 or [int(row["schedule_index"]) for row in rows] != list(range(256)):
        raise ValueError("terminal aggregation requires 256 canonical ordered rows")
    eligible_by_row: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in p1_pairs:
        if pair["p1_pass"]:
            eligible_by_row[int(pair["schedule_index"])].append(pair)
    row_reports: list[dict[str, Any]] = []
    log_weights = np.full(256, -math.inf, dtype=np.float64)
    for row in rows:
        index = int(row["schedule_index"])
        denominator = len(row["loose_pair_ids"])
        eligible = eligible_by_row.get(index, [])
        log_evidence = -math.inf
        if denominator and eligible:
            log_evidence = logsumexp(np.asarray(
                [pair["log_likelihood"] for pair in eligible], dtype=np.float64
            )) - math.log(denominator)
        log_weight = -math.log(256.0) + log_evidence
        log_weights[index] = log_weight
        row_reports.append({
            **{name: row[name] for name in (
                "schedule_index", "parent_seed", "bridge_group", "geometry_key",
                "fine_field_seed",
            )},
            "all_loose_pair_count": denominator,
            "jointly_eligible_pair_count": len(eligible),
            "jointly_eligible_pair_ids": [
                [int(pair["halo_i"]), int(pair["halo_j"])] for pair in eligible
            ],
            "row_log_evidence": _finite_json(log_evidence),
            "unnormalized_row_log_weight": _finite_json(log_weight),
        })
    weights = _normalized(log_weights)
    for row, weight in zip(row_reports, weights):
        row["normalized_weight"] = float(weight)

    group_map = {
        "parent": "parent_seed", "geometry": "geometry_key",
        "fine_field_seed": "fine_field_seed", "bridge": "bridge_group",
    }
    grouped = {name: _group_weights(row_reports, weights, key) for name, key in group_map.items()}
    row_ess = float(1.0 / np.sum(weights**2)) if np.any(weights) else 0.0
    row_max = float(np.max(weights, initial=0.0))
    support = {name: dict(zip(("ESS", "maximum_weight"), _ess_max(value)))
               for name, value in grouped.items()}
    eligible_rows = int(np.count_nonzero(np.isfinite(log_weights)))
    contract = config["aggregation_contract"]
    gates_cfg = config["grouped_support_gates"]
    checks = {
        "minimum_jointly_eligible_rows": eligible_rows >= int(contract["minimum_jointly_eligible_rows"]),
        "minimum_normalized_row_weight_ESS": row_ess >= float(contract["minimum_normalized_row_weight_ESS"]),
        "maximum_single_normalized_row_weight": row_max <= float(contract["maximum_single_normalized_row_weight"]),
        "minimum_parent_weight_ESS": support["parent"]["ESS"] >= float(gates_cfg["minimum_parent_weight_ESS"]),
        "maximum_single_parent_normalized_weight": support["parent"]["maximum_weight"] <= float(gates_cfg["maximum_single_parent_normalized_weight"]),
        "minimum_geometry_key_weight_ESS": support["geometry"]["ESS"] >= float(gates_cfg["minimum_geometry_key_weight_ESS"]),
        "maximum_single_geometry_key_normalized_weight": support["geometry"]["maximum_weight"] <= float(gates_cfg["maximum_single_geometry_key_normalized_weight"]),
        "minimum_bridge_group_weight_ESS": support["bridge"]["ESS"] >= float(gates_cfg["minimum_bridge_group_weight_ESS"]),
        "maximum_single_bridge_group_normalized_weight": support["bridge"]["maximum_weight"] <= float(gates_cfg["maximum_single_bridge_group_normalized_weight"]),
    }
    passed = all(checks.values())
    return {
        "schema": RESULT_SCHEMA,
        "status": (
            "complete_pass_terminal_aggregation_waiting_independent_review"
            if passed else "complete_scientific_fail_terminal_aggregation_closed"
        ),
        "scientific_pass": passed,
        "automatic_promotion": False,
        "RAMSES_authorized": False,
        "same_model_extension_authorized": False,
        "jointly_eligible_rows": eligible_rows,
        "normalized_row_weight_ESS": row_ess,
        "maximum_single_normalized_row_weight": row_max,
        "positive_weight_parent_count": sum(row["normalized_weight"] > 0.0 for row in grouped["parent"]),
        "positive_weight_geometry_key_count": sum(row["normalized_weight"] > 0.0 for row in grouped["geometry"]),
        "positive_weight_fine_field_seed_count": sum(row["normalized_weight"] > 0.0 for row in grouped["fine_field_seed"]),
        "grouped_support": support,
        "checks": checks,
        "grouped_normalized_weights": grouped,
        "rows": row_reports,
    }


def _implementation_files(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(config["lineage"]["required_commit_scope"]):
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"implementation file is not regular: {relative}")
        rows.append({
            "path": relative, "sha256": sha256_file(path),
            "mode": f"{stat.S_IMODE(path.stat().st_mode):04o}",
        })
    return rows


def _required_slurm_integer(name: str, expected: int | None = None) -> int:
    raw = os.environ.get(name, "")
    if not raw.isdecimal() or int(raw) < 1:
        raise RuntimeError(f"{name} is not a positive Slurm integer")
    value = int(raw)
    if expected is not None and value != expected:
        raise RuntimeError(f"{name} differs from the frozen resource contract")
    return value


def runtime_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    """Capture the actual allocation and exact executable that produces the seal."""
    resources = config["resources"]
    expected_python = str(resources["python_executable"])
    if str(Path(sys.executable).resolve()) != str(Path(expected_python).resolve()) \
            or sys.version.split()[0] != resources["python_version"]:
        raise RuntimeError("Python executable or version differs from the frozen circle runtime")
    packages = {name: package_version(name) for name in ("jax", "jaxlib", "pmwd")}
    if packages != resources["python_packages"]:
        raise RuntimeError("jax/jaxlib/pmwd versions differ from the frozen circle runtime")
    partition = os.environ.get("SLURM_JOB_PARTITION", "")
    if partition not in str(resources["partitions"]).split(","):
        raise RuntimeError("Slurm partition is outside the frozen partition set")
    node = os.environ.get("SLURMD_NODENAME", "")
    if not node:
        raise RuntimeError("SLURMD_NODENAME is absent")
    memory_mib = _required_slurm_integer("SLURM_MEM_PER_NODE", 20480)
    _required_slurm_integer("SLURM_NNODES", 1)
    _required_slurm_integer("SLURM_NTASKS", 1)
    cpus = _required_slurm_integer("SLURM_CPUS_PER_TASK", 16)
    gpu_rows = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv,noheader,nounits"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.splitlines()
    if len(gpu_rows) != 1:
        raise RuntimeError("terminal allocation must expose exactly one GPU")
    parts = [part.strip() for part in gpu_rows[0].split(",")]
    if len(parts) != 3 or not parts[2].isdigit() or int(parts[2]) < 40960:
        raise RuntimeError("visible GPU identity or memory is below the frozen minimum")
    git_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ).stdout.strip()
    if len(git_commit) != 40 or any(character not in "0123456789abcdef" for character in git_commit):
        raise RuntimeError("runtime Git commit is not a full lowercase commit")
    return {
        "slurm": {
            "job_id": str(_required_slurm_integer("SLURM_JOB_ID")),
            "node": node, "partition": partition, "nodes": 1, "tasks": 1,
            "cpus_per_task": cpus, "memory_MiB": memory_mib, "gpu_count": 1,
            "visible_GPU": {"uuid": parts[0], "name": parts[1], "memory_MiB": int(parts[2])},
        },
        "python": {
            "executable": expected_python, "version": sys.version.split()[0],
            "packages": packages,
        },
        "git_commit": git_commit,
        "implementation_files": _implementation_files(config),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_replace(staging: Path, output: Path) -> None:
    """Atomically publish a directory without any overwrite race."""
    if staging.parent.resolve() != output.parent.resolve() \
            or staging.parent.stat().st_dev != output.parent.stat().st_dev:
        raise RuntimeError("staging and final output are not sibling paths on one filesystem")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace terminal output: {output}")
    if stat.S_IMODE(staging.stat().st_mode) != 0o555 \
            or {path.name for path in staging.iterdir()} != OUTPUT_NAMES:
        raise RuntimeError("refusing to publish a partial or unsealed staging directory")
    for path in staging.iterdir():
        if path.is_symlink() or not path.is_file() \
                or stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise RuntimeError("refusing to publish a staging directory with unsealed entries")
    complete = json.loads((staging / "COMPLETE").read_text())
    if complete != {
        "schema": COMPLETE_SCHEMA, "status": "complete",
        "manifest_sha256": sha256_file(staging / "manifest.json"),
    }:
        raise RuntimeError("refusing to publish staging without a valid COMPLETE seal")
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(
        -100, os.fsencode(staging), -100, os.fsencode(output), 1,
    ) != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(f"refusing to replace terminal output: {output}")
        raise OSError(code, os.strerror(code), str(output))
    _fsync_directory(output.parent)


def _seal(
    staging: Path, config_path: Path, values: Mapping[str, Mapping[str, Any]],
    runtime: Mapping[str, Any],
) -> None:
    for name, value in values.items():
        _write_exclusive(staging / name, value)
    records = []
    for name in sorted(values):
        path = staging / name
        records.append({
            "name": name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
            "schema": values[name]["schema"],
        })
    manifest = {
        "schema": MANIFEST_SCHEMA, "status": "sealed",
        "config": str(config_path.resolve()), "config_sha256": sha256_file(config_path),
        "runtime": dict(runtime),
        "files": records,
    }
    _write_exclusive(staging / "manifest.json", manifest)
    complete = {
        "schema": COMPLETE_SCHEMA, "status": "complete",
        "manifest_sha256": sha256_file(staging / "manifest.json"),
    }
    _write_exclusive(staging / "COMPLETE", complete)
    if {path.name for path in staging.iterdir()} != OUTPUT_NAMES:
        raise RuntimeError("staging output does not have the exact sealed entry set")
    os.chmod(staging, 0o555)
    _fsync_directory(staging)


def _legacy_rows(path: Path, config_sha256: str) -> dict[int, Mapping[str, Any]]:
    value = json.loads(path.read_text())
    if value.get("schema") != "cf4-p1-result-v2-observer" or value.get("status") != "complete" \
            or value.get("config_sha256") != config_sha256 or len(value.get("members", [])) != 256:
        raise RuntimeError("legacy zero-offset P1 seal changed")
    rows = {int(row["seed"]): row for row in value["members"]}
    if len(rows) != 256:
        raise RuntimeError("legacy zero-offset P1 seed identity changed")
    return rows


def verify_manifest_inputs_unchanged(
    input_manifest: Mapping[str, Any], production_root: Path,
) -> None:
    """Re-hash every consumed mutable GPFS input immediately before publish."""
    for row in input_manifest["rows"]:
        index = int(row["schedule_index"])
        row_dir = production_root / f"row_{index:03d}"
        if sha256_file(row_dir / "result.json") != row["result_sha256"] \
                or sha256_file(row_dir / "halos.npz") != row["halo_catalogue_sha256"]:
            raise RuntimeError(f"production row {index} changed during aggregation")
    for parent in input_manifest["parent_fields"]:
        if sha256_file(Path(parent["path"])) != parent["sha256"]:
            raise RuntimeError(
                f"parent field {parent['parent_seed']} changed during aggregation"
            )


def preflight(config_path: Path, *, require_output_absent: bool = True) -> dict[str, Any]:
    config = load_terminal_config(config_path)
    inputs = config["inputs"]
    paths = {name: _pinned(spec, name) for name, spec in inputs.items() if "sha256" in spec}
    production_root = _resolve(inputs["production_root"]["path"])
    if production_root != Path("/gpfs/kjhan/CF4/recon/linear_cr/lg_highk_streaming_pm_production_v1"):
        raise RuntimeError("production root changed")
    audit = audit_production_root(
        program_path=paths["production_program"],
        cache_path=_resolve(inputs["covariance_cache"]["path"]),
        output_root=production_root, require_complete=True,
    )
    if audit["complete_count"] != 256 or audit["missing_count"] != 0 or audit["staging_directories"]:
        raise RuntimeError("production root is not canonical and complete")
    output = _resolve(config["outputs"]["canonical_root"])
    staging_glob = config["outputs"]["sibling_staging_glob"]
    if require_output_absent and output.exists():
        raise FileExistsError(f"canonical terminal output already exists: {output}")
    if require_output_absent and list(output.parent.glob(staging_glob)):
        raise FileExistsError("terminal sibling staging root already exists")
    return {"config": config, "paths": paths, "production_root": production_root, "output": output}


def run(config_path: Path, *, test_only: bool = False) -> dict[str, Any]:
    state = preflight(config_path, require_output_absent=True)
    config, paths = state["config"], state["paths"]
    hard = load_config(paths["hard_p2_config"])
    likelihood = json.loads(paths["v8_likelihood_program"].read_text())
    allowed = set(config["reuse_policy"]["v8_allowed_sections"])
    if allowed != {"candidate_preselection", "z0_likelihood"}:
        raise RuntimeError("V8 reuse must be exactly candidate_preselection plus z0_likelihood")
    schedule = _load_schedule(paths["schedule"], json.loads(paths["production_program"].read_text()))
    rows, intersections = _recompute_catalogues(
        production_root=state["production_root"], hard_config=hard,
        likelihood_program={name: likelihood[name] for name in allowed},
        box_size=float(config["science_contract"]["box_size_mpc_h"]),
    )
    _assert_observed_anchors(rows, intersections, config)
    if test_only:
        return {"status": "preflight_pass", "canonical_rows": len(rows), "intersection_pairs": len(intersections)}

    runtime = runtime_provenance(config)
    parent_entries = _parent_entries(paths["parent_manifest"])
    p1_config = json.loads(paths["p1_config"].read_text())
    legacy = _legacy_rows(paths["legacy_p1_result"], config["inputs"]["p1_config"]["sha256"])
    started = time.monotonic()
    p1_pairs = evaluate_pair_recentered_p1(
        intersections=intersections, parent_entries=parent_entries,
        p1_config=p1_config, legacy_rows=legacy,
        box_size=float(config["science_contract"]["box_size_mpc_h"]),
    )
    input_manifest = {
        "schema": INPUT_SCHEMA, "status": "complete_verified",
        "config_sha256": sha256_file(config_path),
        "production_program_sha256": sha256_file(paths["production_program"]),
        "schedule_sha256": sha256_file(paths["schedule"]),
        "parent_manifest_sha256": sha256_file(paths["parent_manifest"]),
        "rows": rows,
        "hard_loose_same_identity_pairs": [
            {name: pair[name] for name in (
                "schedule_index", "parent_seed", "bridge_group", "geometry_key",
                "fine_field_seed", "halo_i", "halo_j", "midpoint_mpc_h",
                "log_likelihood",
            )} for pair in intersections
        ],
        "parent_fields": [
            {"parent_seed": seed, "path": parent_entries[seed]["parent_field"],
             "sha256": parent_entries[seed]["parent_field_sha256"]}
            for seed in sorted({int(pair["parent_seed"]) for pair in intersections})
        ],
    }
    p1_result = {
        "schema": P1_SCHEMA, "status": "complete",
        "config_sha256": sha256_file(config_path),
        "p1_config_sha256": sha256_file(paths["p1_config"]),
        "unique_parent_count": len({pair["parent_seed"] for pair in p1_pairs}),
        "pair_count": len(p1_pairs), "exact_gate_names": list(FIVE_P1_GATES),
        "parent_forward_mesh": 192, "parent_forward_cache_policy": "one_parent_in_memory_at_a_time",
        "pairs": p1_pairs,
    }
    terminal = aggregate_terminal(rows, p1_pairs, config)
    terminal.update({
        "config_sha256": sha256_file(config_path),
        "input_manifest_sha256": hashlib.sha256(_canonical_bytes(input_manifest)).hexdigest(),
        "pair_recentered_p1_sha256": hashlib.sha256(_canonical_bytes(p1_result)).hexdigest(),
        "seconds": time.monotonic() - started,
    })
    verify_manifest_inputs_unchanged(input_manifest, state["production_root"])
    for name, spec in config["inputs"].items():
        if "sha256" in spec:
            _pinned(spec, name)
    if runtime["implementation_files"] != _implementation_files(config):
        raise RuntimeError("implementation bytes or modes changed during aggregation")
    output = state["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    staging.mkdir(mode=0o700)
    _seal(staging, Path(config_path), {
        "input_manifest.json": input_manifest,
        "pair_recentered_p1.json": p1_result,
        "terminal_result.json": terminal,
    }, runtime)
    _publish_no_replace(staging, output)
    return terminal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--lineage-preflight", action="store_true")
    args = parser.parse_args()
    if args.lineage_preflight:
        if args.test_only:
            parser.error("--lineage-preflight and --test-only are mutually exclusive")
        result = {"status": "lineage_preflight_pass", **validate_two_commit_lineage(
            load_terminal_config(args.config)
        )}
    else:
        result = run(args.config, test_only=args.test_only)
    summary = {name: result[name] for name in (
        "status", "scientific_pass", "jointly_eligible_rows",
        "normalized_row_weight_ESS", "maximum_single_normalized_row_weight",
        "canonical_rows", "intersection_pairs",
    ) if name in result}
    print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
