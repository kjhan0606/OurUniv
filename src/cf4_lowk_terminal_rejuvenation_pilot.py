#!/usr/bin/env python3
"""Run the low-k beta=1 continuation pilot against the exact CF4 oracle."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from cf4_aggregate_evidence_oracle import geometry_key, logmeanexp_parent
from cf4_aggregate_evidence_parallel_oracle import ParallelExactAtlasEvaluator
from cf4_lowk_terminal_rejuvenation import continue_terminal_population


INPUT_ROOT = Path(
    "/gpfs/kjhan/CF4/recon/linear_cr/"
    "aggregate_evidence_smc_v6_open_shared_schedule_production_grammar_debug_v2"
)
CACHE_SHARD = Path(str(INPUT_ROOT) + "_cache/shard_000000.npz")
ATLAS_MANIFEST = Path(
    "/gpfs/kjhan/CF4/recon/linear_cr/"
    "aggregate_evidence_parent_response_atlas_v1/manifest.json"
)
ATLAS_SHA256 = "47049d0047aa626912652c82ac34757f01ebe4adc0654d07f674d6b943db4211"
FILTER = Path(
    "/gpfs/kjhan/CF4/recon/linear_cr/peak_evidence_phase_control_v2/"
    "density_filter_rfft.npy"
)
FILTER_SHA256 = "1e1d2ce4b022c908b8a3a64257e82fb8621c40cf8f77b1fd206f1347cdd2f59a"
PHYSICAL_MODEL = Path("config/p2_lg_z0_forward_importance_v8.json")
PHYSICAL_MODEL_SHA256 = "6a89f5027f253282e18f21201146dde384837f0d689d725a25022def8ea7e6f2"
MOVE_NAMES = ("q_local", "axis_local", "joint_local", "prior_independence")


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


class CachedParentOracle:
    def __init__(self, evaluator: Any, evidence: dict[tuple[int, ...], np.ndarray]):
        self.evaluator = evaluator
        self.evidence = evidence

    def evaluate(self, midpoint_mpc_h: np.ndarray, axis: np.ndarray):
        midpoint = np.asarray(midpoint_mpc_h, dtype=np.float64)
        axes = np.asarray(axis, dtype=np.float64)
        keys = [geometry_key(q, a) for q, a in zip(midpoint, axes)]
        missing = sorted(set(keys).difference(self.evidence))
        if missing:
            evaluated_keys, log_z = self.evaluator(missing)
            if evaluated_keys != missing:
                raise RuntimeError("exact evaluator changed sorted key order")
            for key, row in zip(evaluated_keys, np.asarray(log_z, dtype=np.float64)):
                self.evidence[key] = np.asarray(row, dtype=np.float64).copy()
        parent_log_z = np.stack([self.evidence[key] for key in keys])
        return np.asarray(keys, dtype=np.int16), logmeanexp_parent(parent_log_z)

    def parent_probabilities(self, keys: np.ndarray, weights: np.ndarray) -> np.ndarray:
        rows = np.stack([self.evidence[tuple(int(x) for x in key)] for key in keys])
        maximum = np.max(rows, axis=1, keepdims=True)
        conditional = np.exp(rows - maximum)
        conditional /= np.sum(conditional, axis=1, keepdims=True)
        result = np.asarray(weights, dtype=np.float64) @ conditional
        return result / np.sum(result)


def _weighted_key_overlap(
    left_keys: np.ndarray,
    left_weights: np.ndarray,
    right_keys: np.ndarray,
    right_weights: np.ndarray,
) -> float:
    def mass(keys, weights):
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        values = np.bincount(inverse, weights=weights, minlength=len(unique))
        return {tuple(int(x) for x in key): float(value) for key, value in zip(unique, values)}

    left = mass(left_keys, left_weights)
    right = mass(right_keys, right_weights)
    return float(sum(min(left.get(key, 0.0), right.get(key, 0.0)) for key in left.keys() | right.keys()))


def _pair_metrics(parent_probability, checkpoints):
    rows = []
    for left, right in itertools.combinations(range(4), 2):
        rows.append({
            "pair": [left, right],
            "parent_L1": float(np.abs(parent_probability[left] - parent_probability[right]).sum()),
            "exact_geometry_weighted_overlap": _weighted_key_overlap(
                checkpoints[left].keys,
                checkpoints[left].weights,
                checkpoints[right].keys,
                checkpoints[right].weights,
            ),
        })
    return rows


def run_pilot(
    *,
    input_root: Path,
    cache_shard: Path,
    output_root: Path,
    evaluator: Any,
    checkpoints=(8, 16, 32),
) -> dict[str, Any]:
    input_root = Path(input_root)
    cache_shard = Path(cache_shard)
    output_root = Path(output_root)
    output_root.mkdir(parents=False, exist_ok=False)

    with np.load(cache_shard, allow_pickle=False) as item:
        cache_keys = np.asarray(item["keys"], dtype=np.int16)
        cache_log_z = np.asarray(item["log_Z"], dtype=np.float64)
    if cache_keys.ndim != 2 or cache_keys.shape[1:] != (6,) or cache_log_z.shape[0] != len(cache_keys):
        raise ValueError("invalid exact evidence cache")
    evidence = {
        tuple(int(x) for x in key): row
        for key, row in zip(cache_keys, cache_log_z)
    }
    original_keys = set(evidence)
    oracle = CachedParentOracle(evaluator, evidence)

    with np.load(input_root / "terminal_parent_frozen.npz", allow_pickle=False) as item:
        original_parent = np.asarray(item["P_rep"], dtype=np.float64)
        log_i_bar = np.asarray(item["log_I_bar"], dtype=np.float64)
    pooling_weight = np.exp(log_i_bar - np.max(log_i_bar))
    pooling_weight /= np.sum(pooling_weight)

    histories = []
    parent_by_checkpoint = [[] for _ in checkpoints]
    saved_keys = set(original_keys)
    original_geometry = []
    for replicate in range(4):
        with np.load(input_root / f"replicate_{replicate}.npz", allow_pickle=False) as item:
            master_seed = int(item["master_seed"])
            original_geometry.append((
                np.asarray(item["keys"], dtype=np.int16).copy(),
                np.asarray(item["weights"], dtype=np.float64).copy(),
            ))
            history = continue_terminal_population(
                master_seed=master_seed,
                midpoint_mpc_h=item["midpoint_mpc_h"],
                axis=item["axis"],
                keys=item["keys"],
                log_z_bar=item["log_Z_bar"],
                weights=item["weights"],
                ancestor_labels=item["ancestor_labels"],
                oracle=oracle,
                checkpoints=checkpoints,
                continuation_id=1,
            )
            histories.append(history)
        for checkpoint_index, (sweep, row) in enumerate(zip(checkpoints, history)):
            parent_probability = oracle.parent_probabilities(row.keys, row.weights)
            parent_by_checkpoint[checkpoint_index].append(parent_probability)
            _atomic_npz(output_root / f"replicate_{replicate}_sweep_{sweep}.npz", {
                "master_seed": np.asarray(master_seed, dtype=np.int64),
                "sweep": np.asarray(sweep, dtype=np.int64),
                "midpoint_mpc_h": row.midpoint_mpc_h,
                "axis": row.axis,
                "keys": row.keys,
                "log_Z_bar": row.log_z_bar,
                "weights": row.weights,
                "ancestor_labels": row.ancestor_labels,
                "move_proposal_count": row.move_proposal_count,
                "move_acceptance_count": row.move_acceptance_count,
                "P_rep": parent_probability,
            })
        delta_keys = sorted(set(evidence).difference(saved_keys))
        _atomic_npz(output_root / f"new_evidence_cache_replicate_{replicate}.npz", {
            "keys": np.asarray(delta_keys, dtype=np.int16).reshape(-1, 6),
            "log_Z": np.stack([evidence[key] for key in delta_keys])
            if delta_keys else np.empty((0, cache_log_z.shape[1]), dtype=np.float64),
        })
        saved_keys.update(delta_keys)

    checkpoint_results = []
    for checkpoint_index, sweep in enumerate(checkpoints):
        snapshots = [history[checkpoint_index] for history in histories]
        parent_probability = np.stack(parent_by_checkpoint[checkpoint_index])
        pooled = pooling_weight @ parent_probability
        pair_rows = _pair_metrics(parent_probability, snapshots)
        checkpoint_results.append({
            "sweep": int(sweep),
            "maximum_pair_parent_L1": max(row["parent_L1"] for row in pair_rows),
            "minimum_exact_geometry_weighted_overlap": min(
                row["exact_geometry_weighted_overlap"] for row in pair_rows
            ),
            "pool_top_parent_index": int(np.argmax(pooled)),
            "pool_top_parent_seed": int(3193 + np.argmax(pooled)),
            "pool_top_parent_probability": float(np.max(pooled)),
            "pairs": pair_rows,
        })

    baseline_pairs = [
        float(np.abs(original_parent[left] - original_parent[right]).sum())
        for left, right in itertools.combinations(range(4), 2)
    ]
    baseline_geometry_overlap = [
        _weighted_key_overlap(
            original_geometry[left][0], original_geometry[left][1],
            original_geometry[right][0], original_geometry[right][1],
        )
        for left, right in itertools.combinations(range(4), 2)
    ]
    new_keys = set(evidence).difference(original_keys)
    result = {
        "schema": "ouruniv-cf4-lowk-terminal-rejuvenation-pilot-v1",
        "status": "complete_diagnostic",
        "checkpoints": checkpoint_results,
        "baseline_maximum_pair_parent_L1": max(baseline_pairs),
        "baseline_pair_parent_L1": baseline_pairs,
        "baseline_minimum_exact_geometry_weighted_overlap": min(baseline_geometry_overlap),
        "baseline_exact_geometry_weighted_overlap": baseline_geometry_overlap,
        "base_cache_key_count": len(original_keys),
        "new_cache_key_count": len(new_keys),
        "pooling_weights": pooling_weight.tolist(),
    }
    _atomic_json(output_root / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    evaluator = ParallelExactAtlasEvaluator(
        ATLAS_MANIFEST,
        ATLAS_SHA256,
        FILTER,
        FILTER_SHA256,
        PHYSICAL_MODEL,
        PHYSICAL_MODEL_SHA256,
    )
    try:
        result = run_pilot(
            input_root=INPUT_ROOT,
            cache_shard=CACHE_SHARD,
            output_root=args.output_root,
            evaluator=evaluator,
        )
    finally:
        evaluator.close()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
