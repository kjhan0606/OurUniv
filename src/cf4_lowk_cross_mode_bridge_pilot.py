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
from cf4_aggregate_evidence_smc import initialize_particles, mh_rejuvenation_sweep
from cf4_lowk_cross_mode_bridge import (
    PopulationState,
    run_beta_one_control,
    run_parallel_tempering_bridge,
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


def _burnin_state(
    state: PopulationState,
    *,
    beta: float,
    oracle: Any,
    master_seed: int,
    sweeps: int,
    namespace: int,
) -> PopulationState:
    q, a, k, z = (
        np.asarray(state.midpoint_mpc_h).copy(),
        np.asarray(state.axis).copy(),
        np.asarray(state.keys).copy(),
        np.asarray(state.log_z_bar).copy(),
    )
    for sweep in range(sweeps):
        q, a, k, z, _ = mh_rejuvenation_sweep(
            q, a, k, z, float(beta), oracle, int(master_seed),
            int(namespace), sweep,
        )
    return PopulationState(q, a, k, z)


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

    bridge_states_by_checkpoint = [[] for _ in bridge_cycles]
    bridge_parent_by_checkpoint = [[] for _ in bridge_cycles]
    control_states_by_checkpoint = [[] for _ in control_sweeps]
    control_parent_by_checkpoint = [[] for _ in control_sweeps]
    group_summaries = []

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
        ladder = []
        for temperature, beta in enumerate(np.asarray(betas)):
            if temperature == len(betas) - 1:
                ladder.append(top)
                continue
            initial_seed = bridge_seed + 100 + temperature
            q, a = initialize_particles(initial_seed, particle_count)
            k, z = oracle.evaluate(q, a)
            ladder.append(_burnin_state(
                PopulationState(q, a, k, z),
                beta=float(beta),
                oracle=oracle,
                master_seed=bridge_seed,
                sweeps=lower_burnin_sweeps,
                namespace=4_000_000 + group * 10_000 + temperature * 100,
            ))

        bridge = run_parallel_tempering_bridge(
            states=ladder,
            betas=np.asarray(betas),
            oracle=oracle,
            master_seed=bridge_seed,
            checkpoints=bridge_cycles,
            sweeps_per_cycle=2,
            namespace=5_000_000 + group * 10_000,
        )
        control = run_beta_one_control(
            state=top,
            oracle=oracle,
            master_seed=bridge_seed,
            checkpoints=control_sweeps,
            namespace=6_000_000 + group * 10_000,
        )

        for index, row in enumerate(bridge):
            probability = oracle.parent_probabilities(
                row.top.keys, np.full(particle_count, 1.0 / particle_count)
            )
            bridge_states_by_checkpoint[index].append(row.top)
            bridge_parent_by_checkpoint[index].append(probability)
            _atomic_npz(output_root / f"group_{group}_bridge_cycle_{row.cycle}.npz", {
                "cycle": np.asarray(row.cycle, dtype=np.int64),
                "midpoint_mpc_h": row.top.midpoint_mpc_h,
                "axis": row.top.axis,
                "keys": row.top.keys,
                "log_Z_bar": row.top.log_z_bar,
                "P_parent": probability,
                "top_origin_id": row.top_origin_id,
                "swap_proposal_count": row.swap_proposal_count,
                "swap_acceptance_count": row.swap_acceptance_count,
                "original_top_roundtrip_count": np.asarray(
                    row.original_top_roundtrip_count, dtype=np.int64
                ),
            })
        for index, row in enumerate(control):
            probability = oracle.parent_probabilities(
                row.keys, np.full(particle_count, 1.0 / particle_count)
            )
            control_states_by_checkpoint[index].append(row)
            control_parent_by_checkpoint[index].append(probability)
            _atomic_npz(output_root / f"group_{group}_control_sweep_{control_sweeps[index]}.npz", {
                "sweep": np.asarray(control_sweeps[index], dtype=np.int64),
                "midpoint_mpc_h": row.midpoint_mpc_h,
                "axis": row.axis,
                "keys": row.keys,
                "log_Z_bar": row.log_z_bar,
                "P_parent": probability,
            })

        final_bridge = bridge[-1]
        group_summary = {
            "group": group,
            "roundtrip_fraction": final_bridge.original_top_roundtrip_count / particle_count,
            "swap_acceptance_fraction": (
                final_bridge.swap_acceptance_count / final_bridge.swap_proposal_count
            ).tolist(),
            "top_original_fraction": float(np.mean(final_bridge.top_origin_id >= 0)),
        }
        group_summaries.append(group_summary)
        _atomic_json(output_root / f"group_{group}_summary.json", group_summary)

        delta_keys = sorted(set(evidence).difference(saved_keys))
        _atomic_npz(output_root / f"new_evidence_cache_group_{group}.npz", {
            "keys": np.asarray(delta_keys, dtype=np.int16).reshape(-1, 6),
            "log_Z": np.stack([evidence[key] for key in delta_keys])
            if delta_keys else np.empty((0, parent_count), dtype=np.float64),
        })
        saved_keys.update(delta_keys)

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
    args = parser.parse_args()
    evaluator = ParallelExactAtlasEvaluator(
        ATLAS_MANIFEST, ATLAS_SHA256, FILTER, FILTER_SHA256,
        PHYSICAL_MODEL, PHYSICAL_MODEL_SHA256,
    )
    try:
        result = run_pilot(
            source_root=SOURCE_ROOT,
            cache_shards=CACHE_SHARDS,
            output_root=args.output_root,
            evaluator=evaluator,
        )
    finally:
        evaluator.close()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
