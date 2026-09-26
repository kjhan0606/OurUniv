#!/usr/bin/env python3
"""Parallel-tempering bridge diagnostic for separated low-k SMC populations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from cf4_aggregate_evidence_smc import mh_rejuvenation_sweep
from cf4_aggregate_evidence_smc import mh_log_acceptance, propose_particle


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


@dataclass(frozen=True)
class GroupedBridgeCheckpoint:
    cycle: int
    bridge_top: tuple[PopulationState, ...]
    control: tuple[PopulationState, ...]
    top_origin_id: np.ndarray
    swap_proposal_count: np.ndarray
    swap_acceptance_count: np.ndarray
    original_top_roundtrip_count: np.ndarray


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


def batched_mh_rejuvenation_sweep(
    *,
    states: Iterable[PopulationState],
    betas: Iterable[float],
    oracle: Any,
    master_seeds: Iterable[int],
    stages: Iterable[int],
    sweep: int | Iterable[int],
) -> tuple[PopulationState, ...]:
    """Apply independent MH kernels while sharing one oracle evaluation call."""

    values = tuple(states)
    beta_values = tuple(float(value) for value in betas)
    seed_values = tuple(int(value) for value in master_seeds)
    stage_values = tuple(int(value) for value in stages)
    if np.isscalar(sweep):
        sweep_values = (int(sweep),) * len(values)
    else:
        sweep_values = tuple(int(value) for value in sweep)
    if (
        not values
        or not (
            len(values) == len(beta_values) == len(seed_values)
            == len(stage_values) == len(sweep_values)
        )
        or any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in beta_values)
        or any(value < 0 for value in sweep_values)
    ):
        raise ValueError("invalid batched MH controller inputs")

    current = []
    proposed_q, proposed_axis, metadata, counts = [], [], [], []
    for state, master_seed, stage, population_sweep in zip(
        values, seed_values, stage_values, sweep_values
    ):
        q = _copy(state.midpoint_mpc_h, np.float64)
        axis = _copy(state.axis, np.float64)
        keys = _copy(state.keys, np.int16)
        log_z = _copy(state.log_z_bar, np.float64)
        count = len(log_z)
        if (
            count == 0 or q.shape != (count, 3) or axis.shape != (count, 3)
            or keys.shape != (count, 6) or not np.all(np.isfinite(log_z))
        ):
            raise ValueError("invalid batched MH population")
        q_proposed = np.empty_like(q)
        axis_proposed = np.empty_like(axis)
        population_metadata = []
        for particle in range(count):
            proposal = propose_particle(
                q[particle], axis[particle], master_seed, stage,
                population_sweep, particle,
            )
            q_proposed[particle] = proposal[0]
            axis_proposed[particle] = proposal[1]
            population_metadata.append(proposal[2:])
        current.append((q, axis, keys, log_z))
        proposed_q.append(q_proposed)
        proposed_axis.append(axis_proposed)
        metadata.append(population_metadata)
        counts.append(count)

    all_keys, all_log_z = oracle.evaluate(
        np.concatenate(proposed_q, axis=0), np.concatenate(proposed_axis, axis=0)
    )
    all_keys = np.asarray(all_keys, dtype=np.int16)
    all_log_z = np.asarray(all_log_z, dtype=np.float64)
    if all_keys.shape != (sum(counts), 6) or all_log_z.shape != (sum(counts),):
        raise RuntimeError("batched oracle changed aligned proposal shape")

    output = []
    start = 0
    for index, (state, beta, population_metadata, count) in enumerate(
        zip(current, beta_values, metadata, counts)
    ):
        q, axis, keys, log_z = state
        proposed_keys = all_keys[start:start + count]
        proposed_log_z = all_log_z[start:start + count]
        accepted = np.zeros(count, dtype=bool)
        for particle, (move, _, _, rng) in enumerate(population_metadata):
            log_acceptance = mh_log_acceptance(
                move, q[particle], proposed_q[index][particle], log_z[particle],
                proposed_log_z[particle], beta,
            )
            uniform = float(rng.random())
            log_uniform = -np.inf if uniform == 0.0 else np.log(uniform)
            accepted[particle] = log_uniform <= log_acceptance
        output.append(PopulationState(
            np.where(accepted[:, None], proposed_q[index], q),
            np.where(accepted[:, None], proposed_axis[index], axis),
            np.where(accepted[:, None], proposed_keys, keys),
            np.where(accepted, proposed_log_z, log_z),
        ))
        start += count
    return tuple(output)


def _frozen_state(state: PopulationState) -> PopulationState:
    return PopulationState(
        _copy(state.midpoint_mpc_h, np.float64, freeze=True),
        _copy(state.axis, np.float64, freeze=True),
        _copy(state.keys, np.int16, freeze=True),
        _copy(state.log_z_bar, np.float64, freeze=True),
    )


def run_grouped_parallel_tempering_bridge(
    *,
    ladders: Iterable[Iterable[PopulationState]],
    betas: np.ndarray,
    oracle: Any,
    master_seeds: Iterable[int],
    checkpoints: Iterable[int] = (4, 8, 16),
    sweeps_per_cycle: int = 2,
    lower_burnin_sweeps: int = 8,
    namespace: int = 4_000_000,
) -> tuple[GroupedBridgeCheckpoint, ...]:
    """Bridge all start groups and matched controls with shared oracle batches."""

    checkpoint_values = _validate_checkpoints(checkpoints)
    groups = [list(group) for group in ladders]
    seeds = tuple(int(value) for value in master_seeds)
    if not groups or len(groups) != len(seeds) or sweeps_per_cycle <= 0 \
            or lower_burnin_sweeps < 0:
        raise ValueError("invalid grouped bridge inputs")
    beta = np.asarray(betas, dtype=np.float64)
    validated = [_validated_ladder(group, beta) for group in groups]
    temperature_count = len(beta)
    particle_count = validated[0][-1].shape[1]
    if any(value[-1].shape[1] != particle_count for value in validated):
        raise ValueError("all bridge groups must use the same particle count")
    groups = [
        [PopulationState(q[t], a[t], k[t], z[t]) for t in range(temperature_count)]
        for _, q, a, k, z in validated
    ]
    controls = [_frozen_state(group[-1]) for group in groups]

    for burnin in range(lower_burnin_sweeps):
        states, beta_rows, seed_rows, stage_rows, locations = [], [], [], [], []
        for group_index, group in enumerate(groups):
            for temperature in range(temperature_count - 1):
                states.append(group[temperature])
                beta_rows.append(beta[temperature])
                seed_rows.append(seeds[group_index])
                stage_rows.append(
                    namespace + group_index * 10_000 + temperature * 100
                )
                locations.append((group_index, temperature))
        updated = batched_mh_rejuvenation_sweep(
            states=states, betas=beta_rows, oracle=oracle,
            master_seeds=seed_rows, stages=stage_rows, sweep=burnin,
        )
        for location, state in zip(locations, updated):
            groups[location[0]][location[1]] = state

    group_count = len(groups)
    origin_id = np.full(
        (group_count, temperature_count, particle_count), -1, dtype=np.int64
    )
    origin_id[:, -1] = np.arange(particle_count, dtype=np.int64)
    visited_zero = np.zeros((group_count, particle_count), dtype=bool)
    returned_to_one = np.zeros((group_count, particle_count), dtype=bool)
    swap_proposal = np.zeros((group_count, temperature_count - 1), dtype=np.int64)
    swap_acceptance = np.zeros_like(swap_proposal)
    output = []
    checkpoint_set = set(checkpoint_values)

    for cycle in range(1, checkpoint_values[-1] + 1):
        for local_sweep in range(sweeps_per_cycle):
            states, beta_rows, seed_rows, stage_rows, sweep_rows, locations = (
                [], [], [], [], [], []
            )
            for group_index, group in enumerate(groups):
                for temperature, state in enumerate(group):
                    states.append(state)
                    beta_rows.append(beta[temperature])
                    seed_rows.append(seeds[group_index])
                    bridge_namespace = namespace + 1_000_000 + group_index * 10_000
                    stage_rows.append(
                        bridge_namespace + (cycle - 1) * temperature_count + temperature
                    )
                    sweep_rows.append(local_sweep)
                    locations.append(("bridge", group_index, temperature))
            for group_index, state in enumerate(controls):
                states.append(state)
                beta_rows.append(1.0)
                seed_rows.append(seeds[group_index])
                stage_rows.append(namespace + 2_000_000 + group_index * 10_000)
                sweep_rows.append((cycle - 1) * sweeps_per_cycle + local_sweep)
                locations.append(("control", group_index, 0))
            updated = batched_mh_rejuvenation_sweep(
                states=states, betas=beta_rows, oracle=oracle,
                master_seeds=seed_rows, stages=stage_rows, sweep=sweep_rows,
            )
            for location, state in zip(locations, updated):
                if location[0] == "bridge":
                    groups[location[1]][location[2]] = state
                else:
                    controls[location[1]] = state

        for group_index, group in enumerate(groups):
            q = np.stack([state.midpoint_mpc_h for state in group])
            a = np.stack([state.axis for state in group])
            k = np.stack([state.keys for state in group])
            z = np.stack([state.log_z_bar for state in group])
            for phase in (0, 1):
                for left in range(phase, temperature_count - 1, 2):
                    right = left + 1
                    log_acceptance = (beta[left] - beta[right]) * (z[right] - z[left])
                    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(
                        seeds[group_index],
                        spawn_key=(
                            6, int(namespace + 1_000_000 + group_index * 10_000),
                            cycle, phase, left,
                        ),
                    )))
                    accepted = np.log(rng.random(particle_count)) <= np.minimum(
                        0.0, log_acceptance
                    )
                    swap_proposal[group_index, left] += particle_count
                    swap_acceptance[group_index, left] += int(np.sum(accepted))
                    for array in (q, a, k, z, origin_id[group_index]):
                        _swap_rows(array, left, right, accepted)
                at_zero = origin_id[group_index, 0]
                valid_zero = at_zero >= 0
                visited_zero[group_index, at_zero[valid_zero]] = True
                at_one = origin_id[group_index, -1]
                valid_one = at_one >= 0
                returned_to_one[group_index, at_one[valid_one]] |= visited_zero[
                    group_index, at_one[valid_one]
                ]
            groups[group_index] = [
                PopulationState(q[t], a[t], k[t], z[t]) for t in range(temperature_count)
            ]

        if cycle in checkpoint_set:
            output.append(GroupedBridgeCheckpoint(
                cycle=cycle,
                bridge_top=tuple(_frozen_state(group[-1]) for group in groups),
                control=tuple(_frozen_state(state) for state in controls),
                top_origin_id=_copy(origin_id[:, -1], np.int64, freeze=True),
                swap_proposal_count=_copy(swap_proposal, np.int64, freeze=True),
                swap_acceptance_count=_copy(swap_acceptance, np.int64, freeze=True),
                original_top_roundtrip_count=_copy(
                    np.sum(returned_to_one, axis=1), np.int64, freeze=True
                ),
            ))
    return tuple(output)


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
