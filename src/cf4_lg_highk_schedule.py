#!/usr/bin/env python3
"""Validate and resample the promoted CF4+LG joint posterior.

This module creates only a geometry/parent/seed schedule.  It does not generate
fields, run PM, or select a parent.  The promoted bank already includes the LG
peak evidence, so every scheduled row has equal posterior weight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from cf4_aggregate_evidence_oracle import geometry_key


EXPECTED_PARTICLES = 8192
EXPECTED_GROUPS = 4
EXPECTED_PER_GROUP = 2048
EXPECTED_PARENTS = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bank(path: Path) -> dict[str, np.ndarray]:
    required = {
        "cycle", "midpoint_mpc_h", "axis", "keys", "group_id",
        "group_particle", "weight", "parent_seed",
        "parent_conditional_probability", "P_parent",
    }
    with np.load(Path(path), allow_pickle=False) as item:
        missing = required.difference(item.files)
        if missing:
            raise ValueError(f"posterior bank is missing arrays: {sorted(missing)}")
        return {name: np.asarray(item[name]) for name in required}


def validate_bank(bank: Mapping[str, np.ndarray]) -> dict[str, Any]:
    midpoint = np.asarray(bank["midpoint_mpc_h"], dtype=np.float64)
    axis = np.asarray(bank["axis"], dtype=np.float64)
    keys = np.asarray(bank["keys"], dtype=np.int16)
    group = np.asarray(bank["group_id"], dtype=np.int64)
    within = np.asarray(bank["group_particle"], dtype=np.int64)
    weight = np.asarray(bank["weight"], dtype=np.float64)
    parent_seed = np.asarray(bank["parent_seed"], dtype=np.int64)
    conditional = np.asarray(
        bank["parent_conditional_probability"], dtype=np.float64
    )
    parent = np.asarray(bank["P_parent"], dtype=np.float64)

    if np.asarray(bank["cycle"]).shape != () or int(bank["cycle"]) != 16:
        raise ValueError("posterior bank is not the promoted cycle-16 bank")
    if midpoint.shape != (EXPECTED_PARTICLES, 3) \
            or axis.shape != (EXPECTED_PARTICLES, 3) \
            or keys.shape != (EXPECTED_PARTICLES, 6):
        raise ValueError("posterior geometry array shape changed")
    if group.shape != (EXPECTED_PARTICLES,) \
            or within.shape != (EXPECTED_PARTICLES,) \
            or weight.shape != (EXPECTED_PARTICLES,):
        raise ValueError("posterior identity array shape changed")
    if conditional.shape != (EXPECTED_PARTICLES, EXPECTED_PARENTS) \
            or parent.shape != (EXPECTED_PARENTS,):
        raise ValueError("posterior parent array shape changed")
    if not np.array_equal(parent_seed, np.arange(3193, 3449)):
        raise ValueError("parent seed lineage changed")
    if not np.all(np.isfinite(midpoint)) or not np.all(np.isfinite(axis)):
        raise ValueError("non-finite posterior geometry")
    axis_error = float(np.max(np.abs(np.linalg.norm(axis, axis=1) - 1.0)))
    if axis_error > 1.0e-12:
        raise ValueError("posterior axes are not normalized")

    group_count = np.bincount(group, minlength=EXPECTED_GROUPS)
    if not np.array_equal(group_count, np.full(EXPECTED_GROUPS, EXPECTED_PER_GROUP)):
        raise ValueError("bridge groups are not exactly balanced")
    for group_id in range(EXPECTED_GROUPS):
        observed = np.sort(within[group == group_id])
        if not np.array_equal(observed, np.arange(EXPECTED_PER_GROUP)):
            raise ValueError("within-group particle identity changed")

    if np.any(weight < 0.0) or not np.all(np.isfinite(weight)):
        raise ValueError("invalid particle weight")
    uniform = np.full(EXPECTED_PARTICLES, 1.0 / EXPECTED_PARTICLES)
    maximum_weight_difference = float(np.max(np.abs(weight - uniform)))
    if maximum_weight_difference > 1.0e-15:
        raise ValueError("promoted geometry particles are not equal weight")
    if np.any(conditional < 0.0) or not np.all(np.isfinite(conditional)):
        raise ValueError("invalid conditional parent probability")
    row_sum_error = float(np.max(np.abs(conditional.sum(axis=1) - 1.0)))
    if row_sum_error > 1.0e-12:
        raise ValueError("conditional parent rows are not normalized")
    parent_reconstruction_error = float(
        np.max(np.abs(weight @ conditional - parent))
    )
    if parent_reconstruction_error > 1.0e-12:
        raise ValueError("conditional rows do not reconstruct parent marginal")

    reconstructed = np.asarray([
        geometry_key(q, a) for q, a in zip(midpoint, axis)
    ], dtype=np.int16)
    if not np.array_equal(reconstructed, keys):
        raise ValueError("geometry keys do not reconstruct from midpoint and axis")

    return {
        "particle_count": EXPECTED_PARTICLES,
        "group_count": EXPECTED_GROUPS,
        "particles_per_group": EXPECTED_PER_GROUP,
        "parent_count": EXPECTED_PARENTS,
        "maximum_axis_norm_error": axis_error,
        "maximum_particle_weight_difference": maximum_weight_difference,
        "maximum_conditional_row_sum_error": row_sum_error,
        "maximum_parent_reconstruction_error": parent_reconstruction_error,
        "parent_ESS": float(1.0 / np.sum(parent**2)),
        "unique_geometry_key_count": int(len(np.unique(keys, axis=0))),
    }


def _systematic_indices(
    probability: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    if probability.ndim != 1 or count <= 0 \
            or np.any(probability < 0.0) or not np.all(np.isfinite(probability)):
        raise ValueError("invalid systematic-resampling input")
    total = float(probability.sum())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("systematic-resampling probability is not normalized")
    threshold = (float(rng.random()) + np.arange(count, dtype=np.float64)) / count
    cdf = np.cumsum(probability)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, threshold, side="left").astype(np.int64)


def build_joint_schedule(
    bank: Mapping[str, np.ndarray],
    *,
    count_per_group: int = 64,
    master_seed: int = 2026082501,
    fine_seed_start: int = 2026090001,
    noise_seed_start: int = 2026091001,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    validation = validate_bank(bank)
    if count_per_group <= 0:
        raise ValueError("count_per_group must be positive")
    group = np.asarray(bank["group_id"], dtype=np.int64)
    within = np.asarray(bank["group_particle"], dtype=np.int64)
    conditional = np.asarray(
        bank["parent_conditional_probability"], dtype=np.float64
    )
    parent_seed = np.asarray(bank["parent_seed"], dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64DXSM(int(master_seed)))

    selected_row, selected_parent = [], []
    for group_id in range(EXPECTED_GROUPS):
        rows = np.flatnonzero(group == group_id)
        rows = rows[np.argsort(within[rows], kind="stable")]
        joint = conditional[rows].reshape(-1) / len(rows)
        flat = _systematic_indices(joint, count_per_group, rng)
        selected_row.append(rows[flat // EXPECTED_PARENTS])
        selected_parent.append(flat % EXPECTED_PARENTS)
    row = np.concatenate(selected_row).astype(np.int32)
    parent_index = np.concatenate(selected_parent).astype(np.int16)
    count = len(row)
    schedule = {
        "schedule_index": np.arange(count, dtype=np.int32),
        "bank_row": row,
        "group_id": group[row].astype(np.int16),
        "group_particle": within[row].astype(np.int32),
        "parent_index": parent_index,
        "parent_seed": parent_seed[parent_index],
        "midpoint_mpc_h": np.asarray(bank["midpoint_mpc_h"])[row],
        "axis": np.asarray(bank["axis"])[row],
        "keys": np.asarray(bank["keys"])[row],
        "fine_field_seed": fine_seed_start + np.arange(count, dtype=np.int64),
        "likelihood_noise_seed": noise_seed_start + np.arange(count, dtype=np.int64),
        "posterior_weight": np.full(count, 1.0 / count, dtype=np.float64),
    }
    empirical = np.bincount(parent_index, minlength=EXPECTED_PARENTS) / float(count)
    target = np.asarray(bank["P_parent"], dtype=np.float64)
    metadata = {
        "validation": validation,
        "schedule_count": count,
        "count_per_group": count_per_group,
        "master_seed": int(master_seed),
        "fine_field_seed_start": int(fine_seed_start),
        "likelihood_noise_seed_start": int(noise_seed_start),
        "group_count": np.bincount(
            schedule["group_id"], minlength=EXPECTED_GROUPS
        ).tolist(),
        "unique_parent_count": int(np.count_nonzero(empirical)),
        "maximum_parent_replication": int(
            np.max(np.bincount(parent_index, minlength=EXPECTED_PARENTS))
        ),
        "empirical_parent_L1": float(np.sum(np.abs(empirical - target))),
        "equal_posterior_weight": float(1.0 / count),
        "peak_evidence_reapplied": False,
        "single_parent_selected": False,
    }
    return schedule, metadata


def parent_l1_null(
    bank: Mapping[str, np.ndarray],
    *,
    observed_l1: float,
    count_per_group: int = 64,
    draws: int = 50000,
    seed: int = 2026082502,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Calibrate schedule parent-L1 under the exact group-stratified sampler."""
    validate_bank(bank)
    if draws <= 0 or chunk_size <= 0 or not np.isfinite(observed_l1):
        raise ValueError("invalid parent-L1 null parameters")
    group = np.asarray(bank["group_id"], dtype=np.int64)
    within = np.asarray(bank["group_particle"], dtype=np.int64)
    conditional = np.asarray(
        bank["parent_conditional_probability"], dtype=np.float64
    )
    target = np.asarray(bank["P_parent"], dtype=np.float64)
    cdf = []
    for group_id in range(EXPECTED_GROUPS):
        rows = np.flatnonzero(group == group_id)
        rows = rows[np.argsort(within[rows], kind="stable")]
        probability = conditional[rows].reshape(-1) / len(rows)
        current = np.cumsum(probability)
        current[-1] = 1.0
        cdf.append(current)

    rng = np.random.Generator(np.random.PCG64DXSM(int(seed)))
    null = np.empty(draws, dtype=np.float64)
    lattice = np.arange(count_per_group, dtype=np.float64)
    total_count = count_per_group * EXPECTED_GROUPS
    for start in range(0, draws, chunk_size):
        stop = min(start + chunk_size, draws)
        size = stop - start
        offsets = rng.random((size, EXPECTED_GROUPS))
        counts = np.zeros((size, EXPECTED_PARENTS), dtype=np.int16)
        draw_index = np.repeat(np.arange(size), count_per_group)
        for group_id in range(EXPECTED_GROUPS):
            threshold = (
                offsets[:, group_id, None] + lattice[None, :]
            ) / count_per_group
            flat = np.searchsorted(
                cdf[group_id], threshold.reshape(-1), side="left"
            )
            parent_index = flat % EXPECTED_PARENTS
            np.add.at(counts, (draw_index, parent_index), 1)
        empirical = counts.astype(np.float64) / total_count
        null[start:stop] = np.sum(np.abs(empirical - target), axis=1)
    q99, q999 = np.quantile(null, [0.99, 0.999], method="higher")
    tail_probability = float(
        (1 + np.count_nonzero(null >= observed_l1)) / (draws + 1)
    )
    return {
        "draws": int(draws),
        "seed": int(seed),
        "observed_parent_L1": float(observed_l1),
        "q99": float(q99),
        "q999": float(q999),
        "tail_probability": tail_probability,
        "passes_q999": bool(observed_l1 <= q999),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--count-per-group", type=int, default=64)
    parser.add_argument("--master-seed", type=int, default=2026082501)
    parser.add_argument("--null-draws", type=int, default=50000)
    parser.add_argument("--null-seed", type=int, default=2026082502)
    args = parser.parse_args()
    if args.expected_sha256 and sha256_file(args.bank) != args.expected_sha256:
        raise RuntimeError("posterior bank SHA256 changed")
    bank = load_bank(args.bank)
    validation = validate_bank(bank)
    if args.output_root is None:
        print(json.dumps({"status": "complete_read_only_preflight", **validation}, sort_keys=True))
        return
    schedule, metadata = build_joint_schedule(
        bank, count_per_group=args.count_per_group, master_seed=args.master_seed
    )
    null = parent_l1_null(
        bank,
        observed_l1=metadata["empirical_parent_L1"],
        count_per_group=args.count_per_group,
        draws=args.null_draws,
        seed=args.null_seed,
    )
    args.output_root.mkdir(parents=False, exist_ok=False)
    if null["passes_q999"]:
        _atomic_npz(args.output_root / "schedule.npz", schedule)
    _atomic_json(args.output_root / "result.json", {
        "schema": "ouruniv-cf4-lg-highk-joint-schedule-v1",
        "status": (
            "complete_pass_schedule_only_no_field_generated"
            if null["passes_q999"]
            else "complete_fail_schedule_parent_null_no_field_generated"
        ),
        "source_bank": str(args.bank),
        "source_bank_sha256": sha256_file(args.bank),
        "parent_L1_null": null,
        **metadata,
    })
    print(json.dumps({**metadata, "parent_L1_null": null}, sort_keys=True))
    if not null["passes_q999"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
