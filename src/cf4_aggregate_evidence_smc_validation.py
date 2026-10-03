#!/usr/bin/env python3
"""Prospective synthetic validation for aggregate-evidence CF4 SMC."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from cf4_aggregate_evidence_oracle import AggregateEvidenceControllerOracle
from cf4_aggregate_evidence_smc import (
    conditional_parent_probabilities,
    genealogical_ess,
    particle_ess,
    pool_parent_probabilities,
    replicate_parent_probability,
    run_smc_replicate,
    select_temperature_increment,
    systematic_resampling,
    update_weights_and_normalizer,
)


SYNTHETIC_MASTER_SEED = 2026082305
SYNTHETIC_REPLICATES = 4
SYNTHETIC_PARTICLES = 2048
DISCRETE_STATES = 64
DISCRETE_PARENTS = 8


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def null_parent_evaluator(keys):
    return keys, np.zeros((len(keys), 256), dtype=np.float64)


def run_null_likelihood_validation() -> dict[str, Any]:
    oracle = AggregateEvidenceControllerOracle(null_parent_evaluator)
    master_seeds = (2026082301, 2026082302, 2026082303, 2026082304)
    replicates = []
    for master_seed in master_seeds:
        replicate = run_smc_replicate(master_seed, oracle)
        oracle.register_terminal_history(master_seed, replicate.keys)
        replicates.append(replicate)
    oracle.seal_terminal_histories()
    parent = []
    for master_seed, replicate in zip(master_seeds, replicates):
        parent_log_z = oracle.terminal_parent_log_z(
            master_seed, replicate.keys
        )
        parent.append(replicate_parent_probability(
            replicate.weights, parent_log_z
        ))
    parent = np.stack(parent)
    uniform_error = float(np.max(np.abs(parent - 1.0 / 256.0)))
    normalizer_error = float(np.max(np.abs([
        replicate.log_normalizer for replicate in replicates
    ])))
    passed = bool(
        all(np.array_equal(
            replicate.beta_history, np.asarray([0.0, 1.0])
        ) for replicate in replicates)
        and normalizer_error <= 1e-12
        and uniform_error <= 1e-12
    )
    return {
        "pass": passed,
        "beta_history": [
            replicate.beta_history.tolist() for replicate in replicates
        ],
        "log_normalizer_absolute_error": normalizer_error,
        "uniform_parent_max_error": uniform_error,
        "genealogical_ESS": [
            replicate.genealogical_ess for replicate in replicates
        ],
    }


def dense_discrete_log_z() -> np.ndarray:
    """Freeze a reproducible eight-parent, multi-mode 64-state likelihood."""
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(
        SYNTHETIC_MASTER_SEED, spawn_key=(9,)
    )))
    state = np.arange(DISCRETE_STATES, dtype=np.float64)[:, None]
    phase = rng.uniform(0.0, 2.0 * math.pi, size=(1, DISCRETE_PARENTS))
    phase_two = rng.uniform(0.0, 2.0 * math.pi, size=(1, DISCRETE_PARENTS))
    parent_bias = rng.normal(0.0, 0.24, size=(1, DISCRETE_PARENTS))
    angle = 2.0 * math.pi * state / DISCRETE_STATES
    return (
        9.00 * np.cos(2.0 * angle - phase)
        + 5.60 * np.cos(5.0 * angle - phase_two)
        + 2.00 * np.cos(9.0 * angle + 0.5 * phase)
        + parent_bias
    )


def _discrete_log_z_bar(log_z: np.ndarray) -> np.ndarray:
    maximum = np.max(log_z, axis=1, keepdims=True)
    return (
        maximum[:, 0]
        + np.log(np.sum(np.exp(log_z - maximum), axis=1))
        - math.log(log_z.shape[1])
    )


def run_discrete_replicate(
    replicate_index: int,
    log_z: np.ndarray,
) -> dict[str, Any]:
    log_z_bar_state = _discrete_log_z_bar(log_z)
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(
        SYNTHETIC_MASTER_SEED, spawn_key=(10, int(replicate_index))
    )))
    state = rng.integers(0, DISCRETE_STATES, size=SYNTHETIC_PARTICLES)
    weights = np.full(
        SYNTHETIC_PARTICLES, 1.0 / SYNTHETIC_PARTICLES, dtype=np.float64
    )
    ancestors = np.arange(SYNTHETIC_PARTICLES, dtype=np.int64)
    beta = 0.0
    log_normalizer = 0.0
    beta_history = [beta]
    resampling_events = 0
    for stage in range(256):
        current_log_z = log_z_bar_state[state]
        delta = select_temperature_increment(beta, weights, current_log_z)
        if 1.0 - beta > 1e-10 and delta <= 1e-12:
            raise RuntimeError("dense discrete temperature schedule stagnated")
        weights, log_increment = update_weights_and_normalizer(
            weights, current_log_z, delta
        )
        beta = min(1.0, beta + delta)
        if 1.0 - beta <= np.finfo(float).eps:
            beta = 1.0
        log_normalizer += log_increment
        if particle_ess(weights) < 0.5 * SYNTHETIC_PARTICLES:
            selected = systematic_resampling(weights, rng)
            state = state[selected]
            ancestors = ancestors[selected]
            weights.fill(1.0 / SYNTHETIC_PARTICLES)
            resampling_events += 1
        for _ in range(4):
            proposal = rng.integers(
                0, DISCRETE_STATES, size=SYNTHETIC_PARTICLES
            )
            log_acceptance = np.minimum(
                0.0,
                beta * (log_z_bar_state[proposal] - log_z_bar_state[state]),
            )
            uniform = rng.random(SYNTHETIC_PARTICLES)
            accepted = np.log(uniform) <= log_acceptance
            state = np.where(accepted, proposal, state)
        beta_history.append(beta)
        if beta == 1.0:
            break
    if beta != 1.0:
        raise RuntimeError("dense discrete SMC did not reach beta one")
    conditional = conditional_parent_probabilities(log_z[state])
    parent_probability = np.sum(weights[:, None] * conditional, axis=0)
    parent_probability /= np.sum(parent_probability)
    return {
        "log_normalizer": float(log_normalizer),
        "parent_probability": parent_probability,
        "beta_history": np.asarray(beta_history),
        "genealogical_ESS": genealogical_ess(
            ancestors, SYNTHETIC_PARTICLES
        ),
        "resampling_events": resampling_events,
    }


def run_dense_discrete_validation() -> dict[str, Any]:
    log_z = dense_discrete_log_z()
    z = np.exp(log_z)
    exact_parent_integral = np.mean(z, axis=0)
    exact_parent_probability = exact_parent_integral / np.sum(
        exact_parent_integral
    )
    exact_i_bar = float(np.mean(exact_parent_integral))
    exact_log_i_bar = math.log(exact_i_bar)
    replicates = [
        run_discrete_replicate(index, log_z)
        for index in range(SYNTHETIC_REPLICATES)
    ]
    log_normalizers = np.asarray([
        row["log_normalizer"] for row in replicates
    ])
    parent = np.stack([row["parent_probability"] for row in replicates])
    pooled_parent, pooled_log_i_bar = pool_parent_probabilities(
        log_normalizers, parent
    )
    normalizer_error = abs(pooled_log_i_bar - exact_log_i_bar)
    parent_l1 = float(np.sum(np.abs(
        pooled_parent - exact_parent_probability
    )))
    multimodal = bool(
        np.count_nonzero(
            (_discrete_log_z_bar(log_z) > np.roll(_discrete_log_z_bar(log_z), 1))
            & (_discrete_log_z_bar(log_z) > np.roll(_discrete_log_z_bar(log_z), -1))
        ) >= 2
    )
    resampling_events = [row["resampling_events"] for row in replicates]
    passed = bool(
        multimodal
        and sum(resampling_events) >= 1
        and normalizer_error <= 0.05
        and parent_l1 <= 0.05
    )
    return {
        "pass": passed,
        "geometry_states": DISCRETE_STATES,
        "synthetic_parents": DISCRETE_PARENTS,
        "multimodal": multimodal,
        "exact_log_normalizer": exact_log_i_bar,
        "pooled_log_normalizer": pooled_log_i_bar,
        "pooled_log_normalizer_absolute_error": normalizer_error,
        "pooled_parent_probability_L1": parent_l1,
        "replicate_log_normalizer": log_normalizers.tolist(),
        "replicate_genealogical_ESS": [
            row["genealogical_ESS"] for row in replicates
        ],
        "replicate_resampling_events": resampling_events,
        "replicate_beta_history": [
            row["beta_history"].tolist() for row in replicates
        ],
    }


def run_validation() -> dict[str, Any]:
    null = run_null_likelihood_validation()
    dense = run_dense_discrete_validation()
    return {
        "schema": "ouruniv-cf4-aggregate-evidence-smc-synthetic-validation-v1",
        "status": "complete_pass" if null["pass"] and dense["pass"] else "complete_fail",
        "synthetic_master_seed": SYNTHETIC_MASTER_SEED,
        "null_likelihood": null,
        "dense_discrete_toy": dense,
        "all_pass": bool(null["pass"] and dense["pass"]),
        "decision": {
            "production_SMC_authorized": False,
            "conditional_field_bank_authorized": False,
            "PM_or_RAMSES_authorized": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run_validation()
    atomic_json(args.out.resolve(), result)
    print(f"[smc-validation] status={result['status']}", flush=True)


if __name__ == "__main__":
    main()
