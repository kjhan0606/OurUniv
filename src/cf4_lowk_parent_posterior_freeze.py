#!/usr/bin/env python3
"""Freeze the promoted low-k bridge ensemble without selecting one parent seed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from cf4_aggregate_evidence_oracle import logmeanexp_parent


FINAL_CYCLE = 16
EXPECTED_GROUPS = 4
EXPECTED_PARTICLES_PER_GROUP = 2048
EXPECTED_PARENT_COUNT = 256


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


def _void_rows(keys: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(keys, dtype=np.int16)
    if values.ndim != 2 or values.shape[1:] != (6,):
        raise ValueError("geometry keys must have shape [n,6]")
    return values.view(np.dtype((np.void, values.dtype.itemsize * 6))).ravel()


def load_target_evidence(
    cache_shards: Iterable[Path], unique_keys: np.ndarray
) -> np.ndarray:
    target = np.asarray(unique_keys, dtype=np.int16)
    target_void = _void_rows(target)
    rows = np.empty((len(target), EXPECTED_PARENT_COUNT), dtype=np.float64)
    found = np.zeros(len(target), dtype=bool)
    for shard in cache_shards:
        with np.load(Path(shard), allow_pickle=False) as item:
            keys = np.asarray(item["keys"], dtype=np.int16)
            common, shard_index, target_index = np.intersect1d(
                _void_rows(keys), target_void, assume_unique=True, return_indices=True
            )
            if len(common) == 0:
                continue
            if np.any(found[target_index]):
                raise RuntimeError("target evidence appears in multiple cache shards")
            log_z = np.asarray(item["log_Z"], dtype=np.float64)
            if log_z.shape != (len(keys), EXPECTED_PARENT_COUNT):
                raise ValueError(f"invalid evidence cache parent matrix: {shard}")
            rows[target_index] = log_z[shard_index]
            found[target_index] = True
    if not np.all(found):
        raise RuntimeError(f"missing exact evidence for {np.count_nonzero(~found)} keys")
    if not np.all(np.isfinite(rows)):
        raise RuntimeError("non-finite exact evidence in promoted posterior")
    return rows


def freeze_parent_posterior(
    *,
    artifact_root: Path,
    cache_shards: Iterable[Path],
    source_terminal: Path,
    output_root: Path,
) -> dict[str, Any]:
    artifact_root, output_root = Path(artifact_root), Path(output_root)
    output_root.mkdir(parents=False, exist_ok=False)
    midpoint, axis, keys, log_z_bar, group_id, group_particle = [], [], [], [], [], []
    reported_parent = []
    for group in range(EXPECTED_GROUPS):
        path = artifact_root / f"group_{group}_bridge_cycle_{FINAL_CYCLE}.npz"
        with np.load(path, allow_pickle=False) as item:
            count = len(item["keys"])
            if count != EXPECTED_PARTICLES_PER_GROUP or int(item["cycle"]) != FINAL_CYCLE:
                raise ValueError("final bridge checkpoint has the wrong identity")
            midpoint.append(np.asarray(item["midpoint_mpc_h"], dtype=np.float64))
            axis.append(np.asarray(item["axis"], dtype=np.float64))
            keys.append(np.asarray(item["keys"], dtype=np.int16))
            log_z_bar.append(np.asarray(item["log_Z_bar"], dtype=np.float64))
            reported_parent.append(np.asarray(item["P_parent"], dtype=np.float64))
        group_id.append(np.full(count, group, dtype=np.int16))
        group_particle.append(np.arange(count, dtype=np.int32))

    midpoint = np.concatenate(midpoint)
    axis = np.concatenate(axis)
    keys = np.concatenate(keys)
    log_z_bar = np.concatenate(log_z_bar)
    group_id = np.concatenate(group_id)
    group_particle = np.concatenate(group_particle)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    evidence = load_target_evidence(cache_shards, unique_keys)
    reconstructed_log_z = logmeanexp_parent(evidence)[inverse]
    maximum_log_z_difference = float(np.max(np.abs(reconstructed_log_z - log_z_bar)))
    if maximum_log_z_difference > 1.0e-12:
        raise RuntimeError("cache evidence does not reproduce bridge log_Z_bar")

    maximum = np.max(evidence, axis=1, keepdims=True)
    conditional_unique = np.exp(evidence - maximum)
    conditional_unique /= conditional_unique.sum(axis=1, keepdims=True)
    conditional = conditional_unique[inverse]
    weight = np.full(len(keys), 1.0 / len(keys), dtype=np.float64)
    parent_probability = weight @ conditional
    reported = np.mean(np.stack(reported_parent), axis=0)
    maximum_parent_difference = float(np.max(np.abs(parent_probability - reported)))
    if maximum_parent_difference > 1.0e-12:
        raise RuntimeError("frozen joint bank does not reproduce reported parent marginal")

    with np.load(source_terminal, allow_pickle=False) as item:
        parent_seed = np.asarray(item["parent_seed"], dtype=np.int64)
    if parent_seed.shape != (EXPECTED_PARENT_COUNT,) \
            or not np.array_equal(parent_seed, np.arange(3193, 3449)):
        raise ValueError("parent seed order changed")

    _atomic_npz(output_root / "posterior_bank.npz", {
        "cycle": np.asarray(FINAL_CYCLE, dtype=np.int64),
        "midpoint_mpc_h": midpoint,
        "axis": axis,
        "keys": keys,
        "log_Z_bar": log_z_bar,
        "group_id": group_id,
        "group_particle": group_particle,
        "weight": weight,
        "parent_seed": parent_seed,
        "parent_conditional_probability": conditional,
        "P_parent": parent_probability,
    })
    top = np.argsort(parent_probability)[-10:][::-1]
    result = {
        "schema": "ouruniv-cf4-promoted-lowk-parent-posterior-v1",
        "status": "complete_promoted_posterior_ensemble_no_seed_selected",
        "source_cycle": FINAL_CYCLE,
        "particle_count": len(keys),
        "particles_per_group": EXPECTED_PARTICLES_PER_GROUP,
        "unique_geometry_key_count": len(unique_keys),
        "parent_count": EXPECTED_PARENT_COUNT,
        "equal_particle_weight": float(weight[0]),
        "parent_probability_sum": float(parent_probability.sum()),
        "parent_ESS": float(1.0 / np.sum(parent_probability**2)),
        "maximum_log_Z_bar_reconstruction_difference": maximum_log_z_difference,
        "maximum_parent_marginal_reconstruction_difference": maximum_parent_difference,
        "top_ten_parent": [
            {
                "parent_seed": int(parent_seed[index]),
                "probability": float(parent_probability[index]),
            }
            for index in top
        ],
        "decision": {
            "lowk_posterior_ensemble_promoted": True,
            "single_parent_seed_selected": False,
            "new_random_draw_used": False,
            "new_oracle_evaluation_used": False,
            "authorized_as_input_to_highk_conditioning_design": True,
            "highk_conditioning_execution_authorized": False,
        },
    }
    _atomic_json(output_root / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cache-shard", type=Path, action="append", required=True)
    parser.add_argument("--source-terminal", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_parent_posterior(
        artifact_root=args.artifact_root,
        cache_shards=args.cache_shard,
        source_terminal=args.source_terminal,
        output_root=args.output_root,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
