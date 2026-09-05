#!/usr/bin/env python3
"""Particle-count-matched analysis of the cross-mode bridge pilot."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


NULL_DRAWS = 20_000
NULL_SEED = 2_026_082_401
ORIGINAL_PARTICLES_PER_GROUP = 2_048


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _maximum_parent_l1(parent_probability: np.ndarray) -> float:
    value = np.asarray(parent_probability, dtype=np.float64)
    if (
        value.shape != (4, 256)
        or not np.all(np.isfinite(value))
        or np.any(value < 0.0)
        or not np.allclose(value.sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("parent probabilities must be four normalized 256-vectors")
    return float(max(
        np.abs(value[left] - value[right]).sum()
        for left, right in itertools.combinations(range(4), 2)
    ))


def _matched_particle_null(
    parent_probability: np.ndarray,
    particle_count: int,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    observed = _maximum_parent_l1(parent_probability)
    if particle_count < 1 or draws < 1:
        raise ValueError("matched-null particle and draw counts must be positive")
    pooled = np.asarray(parent_probability, dtype=np.float64).mean(axis=0)
    rng = np.random.Generator(np.random.PCG64DXSM(int(seed)))
    counts = rng.multinomial(int(particle_count), pooled, size=(int(draws), 4))
    maxima = np.zeros(draws, dtype=np.int64)
    for left, right in itertools.combinations(range(4), 2):
        maxima = np.maximum(
            maxima, np.abs(counts[:, left] - counts[:, right]).sum(axis=1)
        )
    statistics = maxima.astype(np.float64) / particle_count
    q99, q999 = np.quantile(statistics, [0.99, 0.999], method="higher")
    return {
        "particle_count_per_group": int(particle_count),
        "draws": int(draws),
        "seed": int(seed),
        "observed_maximum_parent_L1": observed,
        "q99": float(q99),
        "q999": float(q999),
        "tail_probability": float(
            (np.count_nonzero(statistics >= observed) + 1) / (draws + 1)
        ),
        "common_parent_not_rejected_at_q999": bool(observed <= q999),
    }


def load_parent_probabilities(
    artifact_root: Path,
    bridge_cycles: list[int],
    control_sweeps: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    root = Path(artifact_root)
    bridge, control = [], []
    for cycle, sweep in zip(bridge_cycles, control_sweeps):
        bridge_rows, control_rows = [], []
        for group in range(4):
            with np.load(
                root / f"group_{group}_bridge_cycle_{cycle}.npz", allow_pickle=False
            ) as item:
                bridge_rows.append(np.asarray(item["P_parent"], dtype=np.float64))
            with np.load(
                root / f"group_{group}_control_sweep_{sweep}.npz", allow_pickle=False
            ) as item:
                control_rows.append(np.asarray(item["P_parent"], dtype=np.float64))
        bridge.append(bridge_rows)
        control.append(control_rows)
    return np.asarray(bridge), np.asarray(control)


def analyze_bridge_result(
    result: dict[str, Any],
    bridge_parent_probability: np.ndarray,
    control_parent_probability: np.ndarray,
    *,
    null_draws: int = NULL_DRAWS,
    null_seed: int = NULL_SEED,
) -> dict[str, Any]:
    if result.get("schema") != "ouruniv-cf4-lowk-cross-mode-bridge-pilot-v1" \
            or result.get("status") != "complete_diagnostic":
        raise ValueError("cross-mode bridge result is incomplete or has the wrong schema")
    checkpoints = result.get("checkpoints")
    groups = result.get("groups")
    particle_count = int(result.get("particles_per_group", 0))
    if not isinstance(checkpoints, list) or len(checkpoints) != 3:
        raise ValueError("three bridge checkpoints are required")
    if not isinstance(groups, list) or len(groups) != 4:
        raise ValueError("four start groups are required")
    bridge_parent = np.asarray(bridge_parent_probability, dtype=np.float64)
    control_parent = np.asarray(control_parent_probability, dtype=np.float64)
    if bridge_parent.shape != (3, 4, 256) or control_parent.shape != (3, 4, 256):
        raise ValueError("checkpoint parent arrays must have shape [3,4,256]")

    checkpoint_analysis = []
    previous_cycle = 0
    for index, row in enumerate(checkpoints):
        cycle = int(row["bridge_cycle"])
        sweeps = int(row["matched_mh_sweeps"])
        if cycle <= previous_cycle or sweeps != 2 * cycle:
            raise ValueError("bridge checkpoints are not ordered or sweep matched")
        previous_cycle = cycle
        bridge_l1 = _finite_float(row["bridge_maximum_parent_L1"], "bridge L1")
        control_l1 = _finite_float(row["control_maximum_parent_L1"], "control L1")
        if not np.isclose(
            bridge_l1, _maximum_parent_l1(bridge_parent[index]), rtol=0.0, atol=1e-12
        ) or not np.isclose(
            control_l1, _maximum_parent_l1(control_parent[index]), rtol=0.0, atol=1e-12
        ):
            raise ValueError("reported parent L1 does not match checkpoint arrays")
        bridge_overlap = _finite_float(
            row["bridge_minimum_exact_overlap"], "bridge overlap"
        )
        control_overlap = _finite_float(
            row["control_minimum_exact_overlap"], "control overlap"
        )
        if min(bridge_l1, control_l1, bridge_overlap, control_overlap) < 0.0 \
                or max(bridge_overlap, control_overlap) > 1.0:
            raise ValueError("bridge diagnostic lies outside its physical range")
        checkpoint_analysis.append({
            "bridge_cycle": cycle,
            "matched_mh_sweeps": sweeps,
            "bridge_maximum_parent_L1": bridge_l1,
            "control_maximum_parent_L1": control_l1,
            "parent_L1_improvement_control_minus_bridge": control_l1 - bridge_l1,
            "bridge_minimum_exact_overlap": bridge_overlap,
            "control_minimum_exact_overlap": control_overlap,
            "exact_overlap_improvement_bridge_minus_control": (
                bridge_overlap - control_overlap
            ),
            "bridge_matched_particle_null": _matched_particle_null(
                bridge_parent[index], particle_count,
                draws=null_draws, seed=null_seed + 2 * index,
            ),
            "control_matched_particle_null": _matched_particle_null(
                control_parent[index], particle_count,
                draws=null_draws, seed=null_seed + 2 * index + 1,
            ),
        })

    roundtrip, swap = [], []
    for expected_group, row in enumerate(groups):
        if int(row["group"]) != expected_group:
            raise ValueError("bridge groups are not in canonical order")
        roundtrip.append(_finite_float(row["roundtrip_fraction"], "roundtrip fraction"))
        values = np.asarray(row["swap_acceptance_fraction"], dtype=np.float64)
        if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError("swap acceptance vector is invalid")
        if np.min(values) < 0.0 or np.max(values) > 1.0:
            raise ValueError("swap acceptance lies outside [0,1]")
        swap.extend(float(value) for value in values)
    if min(roundtrip) < 0.0 or max(roundtrip) > 1.0:
        raise ValueError("roundtrip fraction lies outside [0,1]")

    cross_mode_transport_observed = bool(max(roundtrip) > 0.0)
    matched_null_pass = bool(all(
        row["bridge_matched_particle_null"]["common_parent_not_rejected_at_q999"]
        for row in checkpoint_analysis
    ))
    original_resolution_reached = particle_count >= ORIGINAL_PARTICLES_PER_GROUP
    return {
        "schema": "ouruniv-cf4-lowk-cross-mode-bridge-analysis-v2",
        "status": "complete_particle_matched_read_only_analysis",
        "supersedes": "ouruniv-cf4-lowk-cross-mode-bridge-analysis-v1",
        "checkpoints": checkpoint_analysis,
        "transport": {
            "roundtrip_fraction_by_group": roundtrip,
            "minimum_swap_acceptance_fraction": min(swap),
            "maximum_swap_acceptance_fraction": max(swap),
            "cross_mode_transport_observed": cross_mode_transport_observed,
        },
        "science_evidence": {
            "particle_count_per_group": particle_count,
            "all_bridge_checkpoints_pass_particle_matched_q999": matched_null_pass,
            "original_2048_particle_resolution_reached": original_resolution_reached,
            "cross_mode_transport_demonstrated": cross_mode_transport_observed,
            "original_parent_incoherence_resolved": bool(
                matched_null_pass and original_resolution_reached
            ),
            "exact_geometry_overlap_is_descriptive_only": True,
        },
        "decision": {
            "parent_posterior_promotion_authorized": False,
            "reason": (
                "Parent promotion requires the original 2048-particle resolution "
                "and an independent science audit."
            ),
        },
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.result.open() as stream:
        result = json.load(stream)
    cycles = [int(row["bridge_cycle"]) for row in result["checkpoints"]]
    sweeps = [int(row["matched_mh_sweeps"]) for row in result["checkpoints"]]
    bridge_parent, control_parent = load_parent_probabilities(
        args.artifact_root, cycles, sweeps
    )
    analysis = analyze_bridge_result(result, bridge_parent, control_parent)
    _atomic_json(args.output, analysis)
    print(json.dumps(analysis, sort_keys=True))


if __name__ == "__main__":
    main()
