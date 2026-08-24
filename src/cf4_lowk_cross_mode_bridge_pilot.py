#!/usr/bin/env python3
"""Execute the low-k cross-mode bridge pilot with a matched beta=1 control."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cf4_aggregate_evidence_parallel_oracle import ParallelExactAtlasEvaluator
from cf4_aggregate_evidence_smc import initialize_particles
from cf4_lowk_cross_mode_bridge import (
    PopulationState,
    run_grouped_parallel_tempering_bridge,
)
from cf4_lowk_terminal_rejuvenation_pilot import (
    ATLAS_MANIFEST,
    ATLAS_SHA256,
    CachedParentOracle,
    FILTER,
    FILTER_SHA256,
    PHYSICAL_MODEL,
    PHYSICAL_MODEL_SHA256,
)


SOURCE_ROOT = Path(
    "/gpfs/kjhan/CF4/recon/linear_cr/lowk_terminal_rejuvenation_pilot_v1"
)
BASE_CACHE = Path(
    "/gpfs/kjhan/CF4/recon/linear_cr/"
    "aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2_cache/"
    "shard_000000.npz"
)
CACHE_SHARDS = (BASE_CACHE,) + tuple(
    SOURCE_ROOT / f"new_evidence_cache_replicate_{replicate}.npz"
    for replicate in range(4)
)
BETAS = np.asarray([0.0, 0.15314, 0.318198, 0.512947, 0.736999, 1.0])
BRIDGE_CYCLES = (4, 8, 16)
CONTROL_SWEEPS = (8, 16, 32)
PARTICLES_PER_GROUP = 128
LOWER_TEMPERATURE_BURNIN_SWEEPS = 8


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, path)


def _load_cache(shards: Iterable[Path]):
    evidence: dict[tuple[int, ...], np.ndarray] = {}
    parent_count = None
    for shard in shards:
        with np.load(Path(shard), allow_pickle=False) as item:
            keys = np.asarray(item["keys"], dtype=np.int16)
            log_z = np.asarray(item["log_Z"], dtype=np.float64)
        if keys.ndim != 2 or keys.shape[1:] != (6,) or log_z.shape[0] != len(keys):
            raise ValueError(f"invalid evidence cache shard: {shard}")
        if parent_count is None:
            parent_count = log_z.shape[1]
        elif log_z.shape[1] != parent_count:
            raise ValueError("evidence cache parent width changed")
        for key, row in zip(keys, log_z):
            value = tuple(int(x) for x in key)
            if value in evidence and not np.array_equal(evidence[value], row):
                raise ValueError("evidence cache collision")
            evidence[value] = row
    if parent_count is None:
        raise ValueError("at least one evidence cache shard is required")
    return evidence, int(parent_count)


def _systematic_thin_indices(source_count: int, target_count: int, seed: int) -> np.ndarray:
    if target_count <= 0 or source_count < target_count:
        raise ValueError("invalid systematic thinning counts")
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(
        int(seed), spawn_key=(7, 0)
    )))
    offset = float(rng.random()) / target_count
    positions = offset + np.arange(target_count, dtype=np.float64) / target_count
    return np.minimum(source_count - 1, np.floor(positions * source_count).astype(np.int64))


def _key_mass(keys: np.ndarray) -> dict[tuple[int, ...], float]:
    unique, count = np.unique(keys, axis=0, return_counts=True)
    return {tuple(int(x) for x in key): float(n / len(keys)) for key, n in zip(unique, count)}


def _minimum_overlap(states: list[PopulationState]) -> float:
    masses = [_key_mass(row.keys) for row in states]
    result = []
    for left, right in itertools.combinations(range(4), 2):
        result.append(sum(
            min(masses[left].get(key, 0.0), masses[right].get(key, 0.0))
            for key in masses[left].keys() | masses[right].keys()
        ))
    return float(min(result))


def _maximum_parent_l1(parent_probability: np.ndarray) -> float:
    return float(max(
        np.abs(parent_probability[left] - parent_probability[right]).sum()
        for left, right in itertools.combinations(range(4), 2)
    ))


def run_pilot(
    *,
    source_root: Path,
    cache_shards: Iterable[Path],
    output_root: Path,
    evaluator: Any,
    particle_count: int = PARTICLES_PER_GROUP,
    betas: np.ndarray = BETAS,
    bridge_cycles=BRIDGE_CYCLES,
    control_sweeps=CONTROL_SWEEPS,
    lower_burnin_sweeps: int = LOWER_TEMPERATURE_BURNIN_SWEEPS,
) -> dict[str, Any]:
    source_root, output_root = Path(source_root), Path(output_root)
    if tuple(2 * int(cycle) for cycle in bridge_cycles) != tuple(control_sweeps):
        raise ValueError("bridge and control checkpoints must use equal MH sweep counts")
    output_root.mkdir(parents=False, exist_ok=False)
    evidence, parent_count = _load_cache(cache_shards)
    original_keys = set(evidence)
    saved_keys = set(original_keys)
    oracle = CachedParentOracle(evaluator, evidence)

    ladders = []
    bridge_seeds = []
    lower_q = []
    lower_axis = []
    for group in range(4):
        with np.load(source_root / f"replicate_{group}_sweep_32.npz", allow_pickle=False) as item:
            source_count = len(item["keys"])
            source_seed = int(item["master_seed"])
            selected = _systematic_thin_indices(source_count, particle_count, source_seed)
            top = PopulationState(
                np.asarray(item["midpoint_mpc_h"])[selected],
                np.asarray(item["axis"])[selected],
                np.asarray(item["keys"], dtype=np.int16)[selected],
                np.asarray(item["log_Z_bar"])[selected],
            )

        bridge_seed = source_seed + 10_000_000
        bridge_seeds.append(bridge_seed)
        ladder = [None] * len(betas)
        ladder[-1] = top
        for temperature in range(len(betas)):
            if temperature == len(betas) - 1:
                continue
            initial_seed = bridge_seed + 100 + temperature
            q, a = initialize_particles(initial_seed, particle_count)
            ladder[temperature] = (len(lower_q), len(lower_q) + particle_count)
            lower_q.extend(q)
            lower_axis.extend(a)
        ladders.append(ladder)

    lower_keys, lower_log_z = oracle.evaluate(
        np.asarray(lower_q, dtype=np.float64),
        np.asarray(lower_axis, dtype=np.float64),
    )
    for ladder in ladders:
        for temperature in range(len(betas) - 1):
            start, stop = ladder[temperature]
            ladder[temperature] = PopulationState(
                np.asarray(lower_q[start:stop], dtype=np.float64),
                np.asarray(lower_axis[start:stop], dtype=np.float64),
                lower_keys[start:stop],
                lower_log_z[start:stop],
            )

    bridge = run_grouped_parallel_tempering_bridge(
        ladders=ladders,
        betas=np.asarray(betas),
        oracle=oracle,
        master_seeds=bridge_seeds,
        checkpoints=bridge_cycles,
        sweeps_per_cycle=2,
        lower_burnin_sweeps=lower_burnin_sweeps,
        namespace=4_000_000,
    )

    bridge_states_by_checkpoint = [[] for _ in bridge_cycles]
    bridge_parent_by_checkpoint = [[] for _ in bridge_cycles]
    control_states_by_checkpoint = [[] for _ in control_sweeps]
    control_parent_by_checkpoint = [[] for _ in control_sweeps]
    group_summaries = []
    for group in range(4):
        for index, row in enumerate(bridge):
            bridge_state = row.bridge_top[group]
            probability = oracle.parent_probabilities(
                bridge_state.keys, np.full(particle_count, 1.0 / particle_count)
            )
            bridge_states_by_checkpoint[index].append(bridge_state)
            bridge_parent_by_checkpoint[index].append(probability)
            _atomic_npz(output_root / f"group_{group}_bridge_cycle_{row.cycle}.npz", {
                "cycle": np.asarray(row.cycle, dtype=np.int64),
                "midpoint_mpc_h": bridge_state.midpoint_mpc_h,
                "axis": bridge_state.axis,
                "keys": bridge_state.keys,
                "log_Z_bar": bridge_state.log_z_bar,
                "P_parent": probability,
                "top_origin_id": row.top_origin_id[group],
                "swap_proposal_count": row.swap_proposal_count[group],
                "swap_acceptance_count": row.swap_acceptance_count[group],
                "original_top_roundtrip_count": np.asarray(
                    row.original_top_roundtrip_count[group], dtype=np.int64
                ),
            })
            control_state = row.control[group]
            probability = oracle.parent_probabilities(
                control_state.keys, np.full(particle_count, 1.0 / particle_count)
            )
            control_states_by_checkpoint[index].append(control_state)
            control_parent_by_checkpoint[index].append(probability)
            _atomic_npz(output_root / f"group_{group}_control_sweep_{control_sweeps[index]}.npz", {
                "sweep": np.asarray(control_sweeps[index], dtype=np.int64),
                "midpoint_mpc_h": control_state.midpoint_mpc_h,
                "axis": control_state.axis,
                "keys": control_state.keys,
                "log_Z_bar": control_state.log_z_bar,
                "P_parent": probability,
            })

        final_bridge = bridge[-1]
        group_summary = {
            "group": group,
            "roundtrip_fraction": (
                final_bridge.original_top_roundtrip_count[group] / particle_count
            ),
            "swap_acceptance_fraction": (
                final_bridge.swap_acceptance_count[group]
                / final_bridge.swap_proposal_count[group]
            ).tolist(),
            "top_original_fraction": float(
                np.mean(final_bridge.top_origin_id[group] >= 0)
            ),
        }
        group_summaries.append(group_summary)
        _atomic_json(output_root / f"group_{group}_summary.json", group_summary)

    delta_keys = sorted(set(evidence).difference(saved_keys))
    _atomic_npz(output_root / "new_evidence_cache.npz", {
        "keys": np.asarray(delta_keys, dtype=np.int16).reshape(-1, 6),
        "log_Z": np.stack([evidence[key] for key in delta_keys])
        if delta_keys else np.empty((0, parent_count), dtype=np.float64),
    })

    checkpoint_results = []
    for index, (cycle, sweep) in enumerate(zip(bridge_cycles, control_sweeps)):
        bridge_parent = np.stack(bridge_parent_by_checkpoint[index])
        control_parent = np.stack(control_parent_by_checkpoint[index])
        checkpoint_results.append({
            "bridge_cycle": int(cycle),
            "matched_mh_sweeps": int(sweep),
            "bridge_maximum_parent_L1": _maximum_parent_l1(bridge_parent),
            "control_maximum_parent_L1": _maximum_parent_l1(control_parent),
            "bridge_minimum_exact_overlap": _minimum_overlap(
                bridge_states_by_checkpoint[index]
            ),
            "control_minimum_exact_overlap": _minimum_overlap(
                control_states_by_checkpoint[index]
            ),
        })

    result = {
        "schema": "ouruniv-cf4-lowk-cross-mode-bridge-pilot-v1",
        "status": "complete_diagnostic",
        "particles_per_group": int(particle_count),
        "betas": np.asarray(betas).tolist(),
        "lower_temperature_burnin_sweeps": int(lower_burnin_sweeps),
        "checkpoints": checkpoint_results,
        "groups": group_summaries,
        "base_cache_key_count": len(original_keys),
        "new_cache_key_count": len(set(evidence).difference(original_keys)),
    }
    _atomic_json(output_root / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--particle-count", type=int, default=PARTICLES_PER_GROUP)
    parser.add_argument("--extra-cache-shard", type=Path, action="append", default=[])
    args = parser.parse_args()
    evaluator = ParallelExactAtlasEvaluator(
        ATLAS_MANIFEST, ATLAS_SHA256, FILTER, FILTER_SHA256,
        PHYSICAL_MODEL, PHYSICAL_MODEL_SHA256,
    )
    try:
        result = run_pilot(
            source_root=SOURCE_ROOT,
            cache_shards=CACHE_SHARDS + tuple(args.extra_cache_shard),
            output_root=args.output_root,
            evaluator=evaluator,
            particle_count=args.particle_count,
        )
    finally:
        evaluator.close()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
