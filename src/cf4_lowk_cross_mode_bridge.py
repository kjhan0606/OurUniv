#!/usr/bin/env python3
"""Parallel-tempering bridge diagnostic for separated low-k SMC populations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from cf4_aggregate_evidence_smc import mh_rejuvenation_sweep


@dataclass(frozen=True)
class PopulationState:
    midpoint_mpc_h: np.ndarray
    axis: np.ndarray
    keys: np.ndarray
    log_z_bar: np.ndarray


@dataclass(frozen=True)
class BridgeCheckpoint:
    cycle: int
    top: PopulationState
    top_origin_id: np.ndarray
    swap_proposal_count: np.ndarray
    swap_acceptance_count: np.ndarray
    original_top_roundtrip_count: int


def _copy(value: np.ndarray, dtype: Any, *, freeze: bool = False) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    if freeze:
        result.flags.writeable = False
    return result


def _validate_checkpoints(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if not result or result != tuple(sorted(set(result))) or result[0] <= 0:
        raise ValueError("checkpoints must be strictly increasing positive integers")
    return result


def _validated_ladder(states: Iterable[PopulationState], betas: np.ndarray):
    values = tuple(states)
    beta = np.asarray(betas, dtype=np.float64)
    if (
        len(values) < 2
        or beta.shape != (len(values),)
        or not np.all(np.isfinite(beta))
        or not np.all(np.diff(beta) > 0.0)
        or beta[0] != 0.0
        or beta[-1] != 1.0
    ):
        raise ValueError("temperature ladder must increase strictly from zero to one")
    count = len(np.asarray(values[0].log_z_bar))
    if count == 0:
        raise ValueError("bridge population cannot be empty")
    midpoint, axes, keys, log_z = [], [], [], []
    for state in values:
        q = _copy(state.midpoint_mpc_h, np.float64)
        a = _copy(state.axis, np.float64)
        k = _copy(state.keys, np.int16)
        z = _copy(state.log_z_bar, np.float64)
        if (
            q.shape != (count, 3)
            or a.shape != (count, 3)
            or k.shape != (count, 6)
            or z.shape != (count,)
            or not np.all(np.isfinite(q))
            or not np.all(np.isfinite(a))
            or not np.all(np.isfinite(z))
        ):
            raise ValueError("every ladder population must have aligned finite arrays")
        midpoint.append(q)
        axes.append(a)
        keys.append(k)
        log_z.append(z)
    return beta, np.stack(midpoint), np.stack(axes), np.stack(keys), np.stack(log_z)


def _swap_rows(array: np.ndarray, left: int, right: int, accepted: np.ndarray) -> None:
    temporary = array[left, accepted].copy()
    array[left, accepted] = array[right, accepted]
    array[right, accepted] = temporary


def run_parallel_tempering_bridge(
    *,
    states: Iterable[PopulationState],
    betas: np.ndarray,
    oracle: Any,
    master_seed: int,
    checkpoints: Iterable[int] = (4, 8, 16),
    sweeps_per_cycle: int = 2,
    namespace: int = 2_000_000,
) -> tuple[BridgeCheckpoint, ...]:
    """Run fixed-ladder replica exchange and return beta=1 checkpoints."""

    checkpoint_values = _validate_checkpoints(checkpoints)
    if sweeps_per_cycle <= 0 or namespace < 0:
        raise ValueError("sweep count and RNG namespace must be positive")
    beta, midpoint, axes, keys, log_z = _validated_ladder(states, betas)
    temperatures, count = log_z.shape
    origin_id = np.full((temperatures, count), -1, dtype=np.int64)
    origin_id[-1] = np.arange(count, dtype=np.int64)
    visited_zero = np.zeros(count, dtype=bool)
    returned_to_one = np.zeros(count, dtype=bool)
    swap_proposal = np.zeros(temperatures - 1, dtype=np.int64)
    swap_acceptance = np.zeros(temperatures - 1, dtype=np.int64)
    output = []
    checkpoint_set = set(checkpoint_values)

    for cycle in range(1, checkpoint_values[-1] + 1):
        for temperature in range(temperatures):
            for sweep in range(sweeps_per_cycle):
                stage = namespace + (cycle - 1) * temperatures + temperature
                midpoint[temperature], axes[temperature], keys[temperature], log_z[temperature], _ = (
                    mh_rejuvenation_sweep(
                        midpoint[temperature],
                        axes[temperature],
                        keys[temperature],
                        log_z[temperature],
                        float(beta[temperature]),
                        oracle,
                        int(master_seed),
                        stage,
                        sweep,
                    )
                )

        for phase in (0, 1):
            for left in range(phase, temperatures - 1, 2):
                right = left + 1
                log_acceptance = (beta[left] - beta[right]) * (
                    log_z[right] - log_z[left]
                )
                rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(
                    int(master_seed),
                    spawn_key=(6, int(namespace), cycle, phase, left),
                )))
                uniform = rng.random(count)
                accepted = np.log(uniform) <= np.minimum(0.0, log_acceptance)
                swap_proposal[left] += count
                swap_acceptance[left] += int(np.sum(accepted))
                for array in (midpoint, axes, keys, log_z, origin_id):
                    _swap_rows(array, left, right, accepted)

            at_zero = origin_id[0]
            valid_zero = at_zero >= 0
            visited_zero[at_zero[valid_zero]] = True
            at_one = origin_id[-1]
            valid_one = at_one >= 0
            returned_to_one[at_one[valid_one]] |= visited_zero[at_one[valid_one]]

        if cycle in checkpoint_set:
            output.append(BridgeCheckpoint(
                cycle=cycle,
                top=PopulationState(
                    midpoint_mpc_h=_copy(midpoint[-1], np.float64, freeze=True),
                    axis=_copy(axes[-1], np.float64, freeze=True),
                    keys=_copy(keys[-1], np.int16, freeze=True),
                    log_z_bar=_copy(log_z[-1], np.float64, freeze=True),
                ),
                top_origin_id=_copy(origin_id[-1], np.int64, freeze=True),
                swap_proposal_count=_copy(swap_proposal, np.int64, freeze=True),
                swap_acceptance_count=_copy(swap_acceptance, np.int64, freeze=True),
                original_top_roundtrip_count=int(np.sum(returned_to_one)),
            ))
    return tuple(output)


def run_beta_one_control(
    *,
    state: PopulationState,
    oracle: Any,
    master_seed: int,
    checkpoints: Iterable[int] = (8, 16, 32),
    namespace: int = 3_000_000,
) -> tuple[PopulationState, ...]:
    """Matched beta=1 continuation without temperature swaps."""

    checkpoint_values = _validate_checkpoints(checkpoints)
    _, midpoint, axes, keys, log_z = _validated_ladder(
        (state, state), np.asarray([0.0, 1.0])
    )
    midpoint, axes, keys, log_z = midpoint[-1], axes[-1], keys[-1], log_z[-1]
    output = []
    for sweep in range(1, checkpoint_values[-1] + 1):
        midpoint, axes, keys, log_z, _ = mh_rejuvenation_sweep(
            midpoint, axes, keys, log_z, 1.0, oracle, int(master_seed),
            int(namespace), sweep - 1,
        )
        if sweep in set(checkpoint_values):
            output.append(PopulationState(
                midpoint_mpc_h=_copy(midpoint, np.float64, freeze=True),
                axis=_copy(axes, np.float64, freeze=True),
                keys=_copy(keys, np.int16, freeze=True),
                log_z_bar=_copy(log_z, np.float64, freeze=True),
            ))
    return tuple(output)
