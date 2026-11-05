#!/usr/bin/env python
"""Select representative nonoverlapping Hong TNG100 train/validation splits.

The paper does not publish its observer IDs.  We therefore search spatially
valid 432/93 partitions and rank them using preregistered observer/input/target
features rather than choosing the validation region solely by available-pool
size.  Cross-split cube overlap is forbidden; overlap within one split is
reported but is not forbidden by the paper.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_prepare_tng import CUBE_MPC_H, periodic_abs_delta


N_TRAIN = 432
N_VALIDATION = 93
FEATURE_NAMES = (
    "log10_center_stellar_mass",
    "log1p_galaxies_l0",
    "log1p_occupied_cells_l0",
    "log1p_velocity_std",
    "target_mean",
    "target_std",
)


@dataclass
class Proposal:
    anchor: int
    validation: np.ndarray
    pool: np.ndarray
    pool_score: float


def feature_matrix(handle: h5py.File) -> np.ndarray:
    raw = np.column_stack(
        (
            np.log10(handle["center_stellar_mass_paper_msun"][:]),
            np.log1p(handle["galaxies_l0_paper"][:]),
            np.log1p(handle["occupied_cells_l0"][:]),
            np.log1p(handle["occupied_cell_mean_velocity_std_kms"][:]),
            handle["target_mean"][:],
            handle["target_std"][:],
        )
    ).astype(np.float64)
    if not np.isfinite(raw).all():
        raise RuntimeError("non-finite split feature")
    scale = raw.std(axis=0)
    if np.any(scale == 0):
        raise RuntimeError("constant split feature")
    return (raw - raw.mean(axis=0)) / scale


def proxy_score(training: np.ndarray, validation: np.ndarray) -> float:
    """Cheap balance objective on globally standardized feature arrays."""
    mean_difference = np.abs(training.mean(axis=0) - validation.mean(axis=0))
    std_difference = np.abs(training.std(axis=0) - validation.std(axis=0))
    return float(np.max(mean_difference) + 0.5 * np.max(std_difference))


def ks_distance(first: np.ndarray, second: np.ndarray) -> float:
    values = np.sort(np.concatenate((first, second)))
    first_cdf = np.searchsorted(np.sort(first), values, side="right") / len(first)
    second_cdf = np.searchsorted(np.sort(second), values, side="right") / len(second)
    return float(np.max(np.abs(first_cdf - second_cdf)))


def distribution_metrics(
    training: np.ndarray, validation: np.ndarray
) -> dict[str, Any]:
    pooled_std = np.sqrt(0.5 * (training.var(axis=0) + validation.var(axis=0)))
    smd = np.divide(
        training.mean(axis=0) - validation.mean(axis=0),
        pooled_std,
        out=np.full(training.shape[1], np.inf),
        where=pooled_std > 0,
    )
    ks = np.asarray(
        [ks_distance(training[:, i], validation[:, i]) for i in range(training.shape[1])]
    )
    return {
        "feature_names": list(FEATURE_NAMES),
        "standardized_mean_difference": smd.tolist(),
        "ks_distance": ks.tolist(),
        "max_abs_standardized_mean_difference": float(np.max(np.abs(smd))),
        "max_ks_distance": float(np.max(ks)),
        "all_abs_smd_below_0.25": bool(np.all(np.abs(smd) < 0.25)),
    }


def proposals(positions: np.ndarray, features: np.ndarray) -> list[Proposal]:
    found: dict[tuple[int, ...], Proposal] = {}
    for anchor in range(len(positions)):
        delta = periodic_abs_delta(positions, positions[anchor])
        distance2 = np.einsum("ij,ij->i", delta, delta)
        validation = np.argsort(distance2, kind="stable")[:N_VALIDATION]
        cross_delta = periodic_abs_delta(
            positions[:, None, :], positions[validation][None, :, :]
        )
        blocked = np.any(np.all(cross_delta < CUBE_MPC_H, axis=2), axis=1)
        pool = np.flatnonzero(~blocked)
        if len(pool) < N_TRAIN:
            continue
        key = tuple(sorted(validation.tolist()))
        score = proxy_score(features[pool], features[validation])
        prior = found.get(key)
        proposal = Proposal(anchor, validation, pool, score)
        if prior is None or proposal.pool_score < prior.pool_score:
            found[key] = proposal
    return sorted(found.values(), key=lambda item: item.pool_score)


def optimize_training_subset(
    pool: np.ndarray,
    validation: np.ndarray,
    features: np.ndarray,
    seed: int,
    random_trials: int,
    swap_trials: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    validation_features = features[validation]
    best = rng.choice(pool, N_TRAIN, replace=False)
    best_score = proxy_score(features[best], validation_features)
    for _ in range(random_trials):
        trial = rng.choice(pool, N_TRAIN, replace=False)
        score = proxy_score(features[trial], validation_features)
        if score < best_score:
            best, best_score = trial, score

    selected = np.zeros(len(features), dtype=bool)
    selected[best] = True
    pool_mask = np.zeros(len(features), dtype=bool)
    pool_mask[pool] = True
    if len(pool) == N_TRAIN:
        return np.sort(best), best_score
    for _ in range(swap_trials):
        inside = rng.choice(np.flatnonzero(selected))
        outside = rng.choice(np.flatnonzero(pool_mask & ~selected))
        selected[inside] = False
        selected[outside] = True
        trial = np.flatnonzero(selected)
        score = proxy_score(features[trial], validation_features)
        if score <= best_score:
            best, best_score = trial, score
        else:
            selected[outside] = False
            selected[inside] = True
    return np.sort(best), best_score


def within_split_overlap_fraction(indices: np.ndarray, positions: np.ndarray) -> float:
    delta = periodic_abs_delta(
        positions[indices][:, None, :], positions[indices][None, :, :]
    )
    overlap = np.all(delta < CUBE_MPC_H, axis=2)
    upper = overlap[np.triu_indices(len(indices), k=1)]
    return float(np.mean(upper))


def minimum_cross_split_linf(
    training: np.ndarray, validation: np.ndarray, positions: np.ndarray
) -> float:
    delta = periodic_abs_delta(
        positions[training][:, None, :], positions[validation][None, :, :]
    )
    return float(np.min(np.max(delta, axis=2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--number", type=int, default=3)
    parser.add_argument("--top-proposals", type=int, default=40)
    parser.add_argument("--random-trials", type=int, default=500)
    parser.add_argument("--swap-trials", type=int, default=5000)
    args = parser.parse_args()
    with h5py.File(args.metadata, "r") as handle:
        positions = handle["center_position_mpc_h"][:].astype(np.float64)
        subhalo_ids = handle["center_subhalo_id"][:].astype(np.int64)
        features = feature_matrix(handle)
    candidates = proposals(positions, features)
    if not candidates:
        raise SystemExit("no spatially valid split proposals")
    print(
        f"[split] unique_valid_proposals={len(candidates)} "
        f"best_pool_score={candidates[0].pool_score:.4f}",
        flush=True,
    )
    evaluated: list[dict[str, Any]] = []
    for rank, proposal in enumerate(candidates[: args.top_proposals]):
        training, optimized_score = optimize_training_subset(
            proposal.pool,
            proposal.validation,
            features,
            args.seed + rank,
            args.random_trials,
            args.swap_trials,
        )
        metrics = distribution_metrics(
            features[training], features[proposal.validation]
        )
        validation_sorted = np.sort(proposal.validation)
        evaluated.append(
            {
                "proposal_rank": rank,
                "anchor_candidate_index": proposal.anchor,
                "available_training_pool": int(len(proposal.pool)),
                "pool_proxy_score": proposal.pool_score,
                "optimized_proxy_score": optimized_score,
                "training_candidate_indices": training.tolist(),
                "validation_candidate_indices": validation_sorted.tolist(),
                "training_subhalo_ids": subhalo_ids[training].tolist(),
                "validation_subhalo_ids": subhalo_ids[validation_sorted].tolist(),
                "balance": metrics,
                "minimum_cross_split_Linf_mpc_h": minimum_cross_split_linf(
                    training, proposal.validation, positions
                ),
                "within_training_pair_overlap_fraction": within_split_overlap_fraction(
                    training, positions
                ),
                "within_validation_pair_overlap_fraction": within_split_overlap_fraction(
                    proposal.validation, positions
                ),
            }
        )
        print(
            f"[split] proposal={rank + 1}/{min(args.top_proposals, len(candidates))} "
            f"max_smd={metrics['max_abs_standardized_mean_difference']:.3f} "
            f"max_ks={metrics['max_ks_distance']:.3f}",
            flush=True,
        )
    evaluated.sort(
        key=lambda item: (
            item["balance"]["max_abs_standardized_mean_difference"],
            item["balance"]["max_ks_distance"],
        )
    )
    selected: list[dict[str, Any]] = []
    for candidate in evaluated:
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
        for candidate in evaluated:
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) == args.number:
                break
    report = {
        "schema": "hong2021-balanced-spatial-splits-v1",
        "metadata": str(args.metadata),
        "feature_names": list(FEATURE_NAMES),
        "cross_split_overlap_forbidden": True,
        "selection": {
            "n_train": N_TRAIN,
            "n_validation": N_VALIDATION,
            "seed": args.seed,
            "unique_spatial_proposals": len(candidates),
            "evaluated_top_proposals": min(args.top_proposals, len(candidates)),
            "validation_jaccard_preference_max": 0.8,
        },
        "splits": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "splits": selected}, indent=2), flush=True)


if __name__ == "__main__":
    main()
