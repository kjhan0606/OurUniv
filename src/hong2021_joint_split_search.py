#!/usr/bin/env python
"""Jointly optimize spatial validity and feature balance of Hong splits.

The compact-neighborhood search supplies valid initial states but can retain
large cosmic-environment differences.  This second search swaps validation
observers while incrementally preserving at least 432 candidates whose cubes
do not overlap any validation cube.  It freezes several deterministic split
realizations so split uncertainty can be measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_balanced_split import (
    FEATURE_NAMES,
    N_TRAIN,
    N_VALIDATION,
    distribution_metrics,
    feature_matrix,
    minimum_cross_split_linf,
    optimize_training_subset,
    within_split_overlap_fraction,
)
from hong2021_prepare_tng import CUBE_MPC_H, periodic_abs_delta


def balance_objective(training: np.ndarray, validation: np.ndarray) -> float:
    pooled = np.sqrt(0.5 * (training.var(axis=0) + validation.var(axis=0)))
    smd = np.abs(
        np.divide(
            training.mean(axis=0) - validation.mean(axis=0),
            pooled,
            out=np.full(training.shape[1], np.inf),
            where=pooled > 0,
        )
    )
    return float(np.max(smd) + 0.2 * np.mean(smd))


def mutate_validation(
    initial_validation: np.ndarray,
    conflict: np.ndarray,
    features: np.ndarray,
    seed: int,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, float, int, int]:
    rng = np.random.default_rng(seed)
    validation = np.zeros(len(features), dtype=bool)
    validation[initial_validation] = True
    blocked_count = conflict[:, validation].sum(axis=1).astype(np.int16)
    training_pool = np.flatnonzero(blocked_count == 0)
    score = balance_objective(features[training_pool], features[validation])
    best = score, validation.copy(), training_pool.copy()
    valid_proposals = accepted = 0
    for iteration in range(iterations):
        outgoing = rng.choice(np.flatnonzero(validation))
        incoming = rng.choice(np.flatnonzero(~validation))
        proposed_count = (
            blocked_count - conflict[:, outgoing] + conflict[:, incoming]
        )
        proposed_pool = np.flatnonzero(proposed_count == 0)
        if len(proposed_pool) < N_TRAIN:
            continue
        valid_proposals += 1
        proposed_validation = validation.copy()
        proposed_validation[outgoing] = False
        proposed_validation[incoming] = True
        proposed_score = balance_objective(
            features[proposed_pool], features[proposed_validation]
        )
        temperature = 0.035 * (1.0 - iteration / iterations) + 0.0005
        if proposed_score < score or rng.random() < np.exp(
            (score - proposed_score) / temperature
        ):
            validation = proposed_validation
            blocked_count = proposed_count
            training_pool = proposed_pool
            score = proposed_score
            accepted += 1
            if score < best[0]:
                best = score, validation.copy(), training_pool.copy()
    return (
        np.flatnonzero(best[1]),
        best[2],
        best[0],
        valid_proposals,
        accepted,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--initial-splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=91000)
    parser.add_argument("--restarts", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=150000)
    parser.add_argument("--number", type=int, default=3)
    args = parser.parse_args()
    with h5py.File(args.metadata, "r") as handle:
        positions = handle["center_position_mpc_h"][:].astype(np.float64)
        subhalo_ids = handle["center_subhalo_id"][:].astype(np.int64)
        features = feature_matrix(handle)
    initial_report = json.loads(args.initial_splits.read_text())
    initial_validation = np.asarray(
        initial_report["splits"][0]["validation_candidate_indices"],
        dtype=np.int64,
    )
    if len(initial_validation) != N_VALIDATION:
        raise SystemExit("initial split does not contain 93 validation observers")
    delta = periodic_abs_delta(
        positions[:, None, :], positions[None, :, :]
    )
    conflict = np.all(delta < CUBE_MPC_H, axis=2)
    results: list[dict[str, Any]] = []
    for restart in range(args.restarts):
        validation, pool, mutation_score, valid, accepted = mutate_validation(
            initial_validation,
            conflict,
            features,
            args.seed + restart,
            args.iterations,
        )
        if len(pool) == N_TRAIN:
            training = np.sort(pool)
            subset_score = mutation_score
        else:
            training, subset_score = optimize_training_subset(
                pool,
                validation,
                features,
                args.seed + 10_000 + restart,
                random_trials=1000,
                swap_trials=10_000,
            )
        metrics = distribution_metrics(features[training], features[validation])
        result = {
            "restart": restart,
            "restart_seed": args.seed + restart,
            "mutation_objective": mutation_score,
            "training_subset_proxy_score": subset_score,
            "valid_mutations": valid,
            "accepted_mutations": accepted,
            "available_training_pool": int(len(pool)),
            "training_candidate_indices": training.tolist(),
            "validation_candidate_indices": np.sort(validation).tolist(),
            "training_subhalo_ids": subhalo_ids[training].tolist(),
            "validation_subhalo_ids": subhalo_ids[np.sort(validation)].tolist(),
            "balance": metrics,
            "minimum_cross_split_Linf_mpc_h": minimum_cross_split_linf(
                training, validation, positions
            ),
            "within_training_pair_overlap_fraction": within_split_overlap_fraction(
                training, positions
            ),
            "within_validation_pair_overlap_fraction": within_split_overlap_fraction(
                validation, positions
            ),
        }
        results.append(result)
        print(
            f"[joint] restart={restart + 1}/{args.restarts} "
            f"pool={len(pool)} max_smd="
            f"{metrics['max_abs_standardized_mean_difference']:.3f} "
            f"max_ks={metrics['max_ks_distance']:.3f}",
            flush=True,
        )
    results.sort(
        key=lambda item: (
            item["balance"]["max_abs_standardized_mean_difference"],
            item["balance"]["max_ks_distance"],
        )
    )
    selected: list[dict[str, Any]] = []
    for candidate in results:
        validation = set(candidate["validation_candidate_indices"])
        if any(
            len(validation & set(other["validation_candidate_indices"]))
            / len(validation | set(other["validation_candidate_indices"]))
            > 0.8
            for other in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == args.number:
            break
    if len(selected) < args.number:
        for candidate in results:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == args.number:
                break
    report = {
        "schema": "hong2021-joint-balanced-spatial-splits-v1",
        "metadata": str(args.metadata),
        "initial_splits": str(args.initial_splits),
        "feature_names": list(FEATURE_NAMES),
        "selection": {
            "n_train": N_TRAIN,
            "n_validation": N_VALIDATION,
            "seed": args.seed,
            "restarts": args.restarts,
            "iterations_per_restart": args.iterations,
            "validation_jaccard_preference_max": 0.8,
        },
        "splits": selected,
        "all_restart_summaries": [
            {
                "restart": item["restart"],
                "restart_seed": item["restart_seed"],
                "available_training_pool": item["available_training_pool"],
                "max_abs_smd": item["balance"][
                    "max_abs_standardized_mean_difference"
                ],
                "max_ks": item["balance"]["max_ks_distance"],
            }
            for item in results
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "selected": [
                    {
                        "restart": item["restart"],
                        "max_smd": item["balance"][
                            "max_abs_standardized_mean_difference"
                        ],
                        "max_ks": item["balance"]["max_ks_distance"],
                    }
                    for item in selected
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
