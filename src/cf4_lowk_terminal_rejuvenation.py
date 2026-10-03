#!/usr/bin/env python3
"""Continue a completed low-k SMC population with beta=1 resample-move steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from cf4_aggregate_evidence_smc import (
    mh_rejuvenation_sweep,
    systematic_resampling,
)


@dataclass(frozen=True)
class RejuvenationCheckpoint:
    sweep: int
    midpoint_mpc_h: np.ndarray
    axis: np.ndarray
    keys: np.ndarray
    log_z_bar: np.ndarray
    weights: np.ndarray
    ancestor_labels: np.ndarray
    move_proposal_count: np.ndarray
    move_acceptance_count: np.ndarray


def _frozen_copy(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.flags.writeable = False
    return result


def _validated_checkpoints(checkpoints: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in checkpoints)
    if not result or any(value <= 0 for value in result):
        raise ValueError("checkpoints must contain positive sweep counts")
    if tuple(sorted(set(result))) != result:
        raise ValueError("checkpoints must be strictly increasing and unique")
    return result


def continue_terminal_population(
    *,
    master_seed: int,
    midpoint_mpc_h: np.ndarray,
    axis: np.ndarray,
    keys: np.ndarray,
    log_z_bar: np.ndarray,
    weights: np.ndarray,
    ancestor_labels: np.ndarray,
    oracle: Any,
    checkpoints: Iterable[int] = (8, 16, 32),
    continuation_id: int = 0,
) -> tuple[RejuvenationCheckpoint, ...]:
    """Resample once, then apply target-invariant MH sweeps at beta=1.

    The continuation uses a random-number namespace disjoint from the original
    annealed SMC stages.  Returned checkpoints are immutable snapshots.
    """

    checkpoint_values = _validated_checkpoints(checkpoints)
    master_seed = int(master_seed)
    continuation_id = int(continuation_id)
    if continuation_id < 0:
        raise ValueError("continuation_id must be non-negative")

    midpoint = np.asarray(midpoint_mpc_h, dtype=np.float64).copy()
    axes = np.asarray(axis, dtype=np.float64).copy()
    current_keys = np.asarray(keys, dtype=np.int16).copy()
    current_log_z = np.asarray(log_z_bar, dtype=np.float64).copy()
    current_weights = np.asarray(weights, dtype=np.float64).copy()
    ancestors = np.asarray(ancestor_labels, dtype=np.int64).copy()
    count = len(current_weights)
    if (
        count == 0
        or midpoint.shape != (count, 3)
        or axes.shape != (count, 3)
        or current_keys.shape != (count, 6)
        or current_log_z.shape != (count,)
        or ancestors.shape != (count,)
        or not np.all(np.isfinite(midpoint))
        or not np.all(np.isfinite(axes))
        or not np.all(np.isfinite(current_log_z))
        or not np.all(np.isfinite(current_weights))
        or np.any(current_weights < 0.0)
        or not np.isclose(np.sum(current_weights), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("invalid terminal SMC population")

    resampling_rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(
        master_seed, spawn_key=(3, continuation_id)
    )))
    selected = systematic_resampling(current_weights, resampling_rng)
    midpoint = midpoint[selected]
    axes = axes[selected]
    current_keys = current_keys[selected]
    current_log_z = current_log_z[selected]
    ancestors = ancestors[selected]
    current_weights = np.full(count, 1.0 / count, dtype=np.float64)

    proposal_total = np.zeros(4, dtype=np.int64)
    acceptance_total = np.zeros(4, dtype=np.int64)
    output = []
    checkpoint_set = set(checkpoint_values)
    stage_namespace = 1_000_000 + continuation_id
    for sweep in range(1, checkpoint_values[-1] + 1):
        midpoint, axes, current_keys, current_log_z, move = mh_rejuvenation_sweep(
            midpoint,
            axes,
            current_keys,
            current_log_z,
            1.0,
            oracle,
            master_seed,
            stage_namespace,
            sweep - 1,
        )
        proposal_total += np.asarray([
            move["proposal_count"][name]
            for name in ("q_local", "axis_local", "joint_local", "prior_independence")
        ], dtype=np.int64)
        acceptance_total += np.asarray([
            move["acceptance_count"][name]
            for name in ("q_local", "axis_local", "joint_local", "prior_independence")
        ], dtype=np.int64)
        if sweep in checkpoint_set:
            output.append(RejuvenationCheckpoint(
                sweep=sweep,
                midpoint_mpc_h=_frozen_copy(midpoint, np.float64),
                axis=_frozen_copy(axes, np.float64),
                keys=_frozen_copy(current_keys, np.int16),
                log_z_bar=_frozen_copy(current_log_z, np.float64),
                weights=_frozen_copy(current_weights, np.float64),
                ancestor_labels=_frozen_copy(ancestors, np.int64),
                move_proposal_count=_frozen_copy(proposal_total, np.int64),
                move_acceptance_count=_frozen_copy(acceptance_total, np.int64),
            ))
    return tuple(output)
