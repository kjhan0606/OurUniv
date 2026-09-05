#!/usr/bin/env python3
"""Driver-only statistical audit of the final 2048-particle bridge result."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from cf4_lowk_cross_mode_bridge_analysis import load_parent_probabilities


FAMILYWISE_DRAWS = 50_000
STATIONARITY_DRAWS = 20_000
AUDIT_SEED = 2_026_082_501


def _maximum_l1(values: np.ndarray) -> tuple[float, tuple[int, int]]:
    rows = [
        (float(np.abs(values[left] - values[right]).sum()), (left, right))
        for left, right in itertools.combinations(range(4), 2)
    ]
    return max(rows)


def driver_audit(
    result: dict[str, Any],
    matched_analysis: dict[str, Any],
    bridge_parent_probability: np.ndarray,
    control_parent_probability: np.ndarray,
    source_pool: np.ndarray,
    source_parent_seed: np.ndarray,
    *,
    familywise_draws: int = FAMILYWISE_DRAWS,
    stationarity_draws: int = STATIONARITY_DRAWS,
    seed: int = AUDIT_SEED,
) -> dict[str, Any]:
    if result.get("status") != "complete_diagnostic" \
            or matched_analysis.get("status") != "complete_particle_matched_read_only_analysis":
        raise ValueError("bridge result and matched analysis must both be complete")
    particle_count = int(result["particles_per_group"])
    bridge = np.asarray(bridge_parent_probability, dtype=np.float64)
    control = np.asarray(control_parent_probability, dtype=np.float64)
    source = np.asarray(source_pool, dtype=np.float64)
    parent_seed = np.asarray(source_parent_seed, dtype=np.int64)
    if bridge.shape != (3, 4, 256) or control.shape != (3, 4, 256) \
            or source.shape != (256,) or parent_seed.shape != (256,):
        raise ValueError("driver audit parent arrays have the wrong shape")
    if familywise_draws < 1 or stationarity_draws < 1:
        raise ValueError("driver audit draw counts must be positive")

    cycles = [int(row["bridge_cycle"]) for row in result["checkpoints"]]
    observed_by_cycle, worst_pair_by_cycle = [], []
    for values in bridge:
        observed, pair = _maximum_l1(values)
        observed_by_cycle.append(observed)
        worst_pair_by_cycle.append(list(pair))

    rng = np.random.Generator(np.random.PCG64DXSM(int(seed)))
    familywise = np.zeros(familywise_draws, dtype=np.float64)
    for values in bridge:
        pooled = values.mean(axis=0)
        counts = rng.multinomial(
            particle_count, pooled, size=(familywise_draws, 4)
        )
        maxima = np.zeros(familywise_draws, dtype=np.int64)
        for left, right in itertools.combinations(range(4), 2):
            maxima = np.maximum(
                maxima, np.abs(counts[:, left] - counts[:, right]).sum(axis=1)
            )
        familywise = np.maximum(familywise, maxima / particle_count)
    observed_familywise = max(observed_by_cycle)
    familywise_q99, familywise_q999 = np.quantile(
        familywise, [0.99, 0.999], method="higher"
    )
    familywise_tail = float(
        (np.count_nonzero(familywise >= observed_familywise) + 1)
        / (familywise_draws + 1)
    )

    pools = bridge.mean(axis=1)
    stationarity = []
    for pair_index, (left, right) in enumerate(itertools.combinations(range(3), 2)):
        observed = float(np.abs(pools[left] - pools[right]).sum())
        pooled = (pools[left] + pools[right]) / 2.0
        pooled /= pooled.sum()
        local_rng = np.random.Generator(np.random.PCG64DXSM(
            int(seed) + 100 + pair_index
        ))
        count = 4 * particle_count
        first = local_rng.multinomial(count, pooled, size=stationarity_draws)
        second = local_rng.multinomial(count, pooled, size=stationarity_draws)
        null = np.abs(first - second).sum(axis=1) / count
        q99, q999 = np.quantile(null, [0.99, 0.999], method="higher")
        stationarity.append({
            "cycles": [cycles[left], cycles[right]],
            "pooled_parent_L1": observed,
            "q99": float(q99),
            "q999": float(q999),
            "tail_probability": float(
                (np.count_nonzero(null >= observed) + 1) / (stationarity_draws + 1)
            ),
            "passes_q999": bool(observed <= q999),
        })

    minimum_overlap = [
        float(row["bridge_minimum_exact_overlap"]) for row in result["checkpoints"]
    ]
    bridge_better_than_control = [
        observed_by_cycle[index] < _maximum_l1(control[index])[0]
        for index in range(3)
    ]
    source_shift = [float(np.abs(pool - source).sum()) for pool in pools]
    final_pool = pools[-1]
    top = np.argsort(final_pool)[-5:][::-1]

    gates = {
        "particle_count_is_original_2048": particle_count == 2048,
        "matched_q999_passes_all_checkpoints": bool(
            matched_analysis["science_evidence"][
                "all_bridge_checkpoints_pass_particle_matched_q999"
            ]
        ),
        "familywise_q999_pass": bool(observed_familywise <= familywise_q999),
        "pooled_checkpoint_stationarity_q999_pass": all(
            row["passes_q999"] for row in stationarity
        ),
        "exact_geometry_overlap_monotonic": bool(
            np.all(np.diff(minimum_overlap) > 0.0)
        ),
        "bridge_L1_below_control_at_all_checkpoints": all(
            bridge_better_than_control
        ),
        "cross_mode_transport_demonstrated": bool(
            matched_analysis["science_evidence"][
                "cross_mode_transport_demonstrated"
            ]
        ),
    }
    driver_go = all(gates.values())
    return {
        "schema": "ouruniv-cf4-lowk-cross-mode-bridge-driver-audit-v1",
        "status": "complete_driver_only_audit",
        "auditor": "Codex driver; not independent Fable audit",
        "familywise_checkpoint_null": {
            "draws": int(familywise_draws),
            "seed": int(seed),
            "assumption": "independent checkpoints; conservative for correlated chain history",
            "observed_maximum_parent_L1": observed_familywise,
            "q99": float(familywise_q99),
            "q999": float(familywise_q999),
            "tail_probability": familywise_tail,
            "passes_q999": bool(observed_familywise <= familywise_q999),
        },
        "checkpoint_diagnostics": {
            "cycles": cycles,
            "maximum_parent_L1": observed_by_cycle,
            "worst_pair": worst_pair_by_cycle,
            "minimum_exact_geometry_overlap": minimum_overlap,
            "bridge_L1_below_control": bridge_better_than_control,
        },
        "pooled_checkpoint_stationarity": stationarity,
        "source_comparison_descriptive_only": {
            "bridge_pool_to_original_SMC_pool_L1": source_shift,
            "reason_not_gated": (
                "The original SMC populations had unequal evidence weights and low "
                "genealogical ESS; the bridge was introduced to repair that incoherence."
            ),
        },
        "final_parent_summary": {
            "ESS": float(1.0 / np.sum(final_pool**2)),
            "top_five": [
                {
                    "parent_index": int(index),
                    "parent_seed": int(parent_seed[index]),
                    "probability": float(final_pool[index]),
                }
                for index in top
            ],
        },
        "gates": gates,
        "decision": {
            "driver_science_go_for_independent_audit": driver_go,
            "parent_posterior_promotion_authorized": False,
            "seed_selection_authorized": False,
            "reason": (
                "The driver gate supports independent audit entry, but promotion "
                "requires the separately authorized independent Fable audit."
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
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (args.artifact_root / "result.json").open() as stream:
        result = json.load(stream)
    with (args.artifact_root / "analysis_v2.json").open() as stream:
        analysis = json.load(stream)
    cycles = [int(row["bridge_cycle"]) for row in result["checkpoints"]]
    sweeps = [int(row["matched_mh_sweeps"]) for row in result["checkpoints"]]
    bridge, control = load_parent_probabilities(args.artifact_root, cycles, sweeps)
    with np.load(args.source_terminal, allow_pickle=False) as item:
        source_pool = np.asarray(item["P_pool"], dtype=np.float64)
        parent_seed = np.asarray(item["parent_seed"], dtype=np.int64)
    value = driver_audit(
        result, analysis, bridge, control, source_pool, parent_seed
    )
    _atomic_json(args.output, value)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
