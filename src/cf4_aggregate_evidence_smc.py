#!/usr/bin/env python3
"""Prospective aggregate-evidence annealed SMC controller for CF4."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from cf4_aggregate_evidence_oracle import canonical_axis


PARTICLE_COUNT = 2048
TARGET_CESS_FRACTION = 0.8
RESAMPLING_ESS_FRACTION = 0.5
MAXIMUM_TEMPERATURE_STAGES = 256
TEMPERATURE_INTERVAL_TOLERANCE = 1e-10
TEMPERATURE_STAGNATION_DELTA = 1e-12
TEMPERATURE_ENDPOINT_TOLERANCE = 1e-10
SWEEPS_PER_STAGE = 4
PRIOR_MEAN_MPC_H = np.asarray([0.0, -6.0, 4.0], dtype=np.float64)
PRIOR_SIGMA_MPC_H = np.asarray([3.0, 3.0, 3.0], dtype=np.float64)
MOVE_NAMES = ("q_local", "axis_local", "joint_local", "prior_independence")
MOVE_PROBABILITIES = np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
Q_SCALES = np.asarray([0.25, 0.6, 1.5], dtype=np.float64)
Q_SCALE_PROBABILITIES = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)
AXIS_KAPPA = np.asarray([100.0, 10.0, 1.0], dtype=np.float64)
AXIS_KAPPA_PROBABILITIES = np.asarray([0.5, 0.3, 0.2], dtype=np.float64)


def _validate_probability_vector(probability: np.ndarray, name: str) -> None:
    value = np.asarray(probability, dtype=np.float64)
    if (
        value.ndim != 1
        or len(value) == 0
        or np.any(value < 0.0)
        or not np.all(np.isfinite(value))
        or not np.isclose(np.sum(value), 1.0, rtol=0.0, atol=1e-15)
    ):
        raise ValueError(f"{name} must be a normalized probability vector")


for _probability, _name in (
    (MOVE_PROBABILITIES, "move probabilities"),
    (Q_SCALE_PROBABILITIES, "q-scale probabilities"),
    (AXIS_KAPPA_PROBABILITIES, "axis-scale probabilities"),
):
    _validate_probability_vector(_probability, _name)


def logsumexp(values: np.ndarray) -> float:
    value = np.asarray(values, dtype=np.float64)
    if (
        value.ndim != 1
        or len(value) == 0
        or np.any(np.isnan(value))
        or np.any(np.isposinf(value))
        or not np.any(np.isfinite(value))
    ):
        raise ValueError("logsumexp input must contain a finite log weight")
    maximum = float(np.max(value))
    return maximum + math.log(float(np.sum(np.exp(value - maximum))))


def normalized_weights_from_log(log_weights: np.ndarray) -> np.ndarray:
    value = np.asarray(log_weights, dtype=np.float64)
    normalization = logsumexp(value)
    result = np.exp(value - normalization)
    result /= np.sum(result)
    return result


def particle_ess(weights: np.ndarray) -> float:
    value = np.asarray(weights, dtype=np.float64)
    if (
        value.ndim != 1
        or len(value) == 0
        or np.any(value < 0.0)
        or not np.all(np.isfinite(value))
        or not np.isclose(np.sum(value), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("particle weights must be finite and normalized")
    return float(1.0 / np.sum(value**2))


def conditional_ess(
    weights: np.ndarray,
    log_z_bar: np.ndarray,
    delta: float,
) -> float:
    value = np.asarray(weights, dtype=np.float64)
    evidence = np.asarray(log_z_bar, dtype=np.float64)
    if value.shape != evidence.shape or value.ndim != 1:
        raise ValueError("weights and aggregate evidence must align")
    if np.any(value < 0.0) or not np.all(np.isfinite(evidence)):
        raise ValueError("CESS requires nonnegative weights and finite evidence")
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError("temperature increment must be finite and nonnegative")
    with np.errstate(divide="ignore"):
        log_weight = np.log(value)
    log_first = logsumexp(log_weight + float(delta) * evidence)
    log_second = logsumexp(log_weight + 2.0 * float(delta) * evidence)
    return float(len(value) * math.exp(2.0 * log_first - log_second))


def select_temperature_increment(
    beta: float,
    weights: np.ndarray,
    log_z_bar: np.ndarray,
    *,
    target_fraction: float = TARGET_CESS_FRACTION,
    interval_tolerance: float = TEMPERATURE_INTERVAL_TOLERANCE,
) -> float:
    if not np.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0,1]")
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target CESS fraction must lie in (0,1]")
    remaining = 1.0 - float(beta)
    if remaining == 0.0:
        return 0.0
    target = float(len(weights)) * float(target_fraction)
    if conditional_ess(weights, log_z_bar, remaining) >= target:
        return remaining
    low = 0.0
    high = remaining
    while high - low > interval_tolerance:
        midpoint = 0.5 * (low + high)
        if conditional_ess(weights, log_z_bar, midpoint) >= target:
            low = midpoint
        else:
            high = midpoint
    return low


def update_weights_and_normalizer(
    weights: np.ndarray,
    log_z_bar: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, float]:
    value = np.asarray(weights, dtype=np.float64)
    evidence = np.asarray(log_z_bar, dtype=np.float64)
    if value.shape != evidence.shape or np.any(value < 0.0):
        raise ValueError("weight update requires aligned nonnegative weights")
    with np.errstate(divide="ignore"):
        unnormalized = np.log(value) + float(delta) * evidence
    log_increment = logsumexp(unnormalized)
    updated = np.exp(unnormalized - log_increment)
    updated /= np.sum(updated)
    return updated, log_increment


def systematic_resampling(
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    value = np.asarray(weights, dtype=np.float64)
    particle_ess(value)
    count = len(value)
    initial = float(rng.random()) / count
    thresholds = initial + np.arange(count, dtype=np.float64) / count
    cdf = np.cumsum(value)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, thresholds, side="left").astype(np.int64)


def genealogical_ess(ancestor_labels: np.ndarray, particle_count: int) -> float:
    labels = np.asarray(ancestor_labels, dtype=np.int64)
    if labels.shape != (particle_count,) or np.any(labels < 0) or np.any(
        labels >= particle_count
    ):
        raise ValueError("ancestor labels are outside the initial particle range")
    fraction = np.bincount(labels, minlength=particle_count) / float(particle_count)
    return float(1.0 / np.sum(fraction**2))


def diagonal_normal_logpdf(
    midpoint_mpc_h: np.ndarray,
    mean: np.ndarray = PRIOR_MEAN_MPC_H,
    sigma: np.ndarray = PRIOR_SIGMA_MPC_H,
) -> np.ndarray:
    value = np.asarray(midpoint_mpc_h, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    if value.shape[-1] != 3 or mean.shape != (3,) or sigma.shape != (3,):
        raise ValueError("diagonal-normal geometry must end in dimension three")
    if np.any(sigma <= 0.0) or not np.all(np.isfinite(value)):
        raise ValueError("invalid diagonal-normal input")
    standardized = (value - mean) / sigma
    return -0.5 * np.sum(standardized**2, axis=-1) - np.sum(
        np.log(sigma)
    ) - 1.5 * math.log(2.0 * math.pi)


def sample_isotropic_axis(rng: np.random.Generator) -> np.ndarray:
    z = 2.0 * float(rng.random()) - 1.0
    phi = 2.0 * math.pi * float(rng.random())
    radius = math.sqrt(max(0.0, 1.0 - z * z))
    return canonical_axis(np.asarray([
        radius * math.cos(phi), radius * math.sin(phi), z
    ]))


def _orthogonal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.zeros(3, dtype=np.float64)
    reference[int(np.argmin(np.abs(direction)))] = 1.0
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    return first, second


def antipodal_vmf_logpdf(
    axis: np.ndarray,
    direction: np.ndarray,
    kappa: float,
) -> np.ndarray:
    value = np.asarray(axis, dtype=np.float64)
    if value.ndim == 1:
        value = value[None]
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError("vMF axes must have shape (n,3)")
    norms = np.linalg.norm(value, axis=1)
    if np.any(norms == 0.0) or not np.all(np.isfinite(norms)):
        raise ValueError("vMF axes must be finite and nonzero")
    value = value / norms[:, None]
    centre = canonical_axis(direction)
    kappa = float(kappa)
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("vMF kappa must be positive and finite")
    log_sinh = (
        math.log(math.sinh(kappa))
        if kappa < 1.0
        else kappa + math.log1p(-math.exp(-2.0 * kappa)) - math.log(2.0)
    )
    argument = kappa * (value @ centre)
    absolute = np.abs(argument)
    log_cosh = absolute + np.log1p(np.exp(-2.0 * absolute)) - math.log(2.0)
    return (
        math.log(kappa) - math.log(4.0 * math.pi) - log_sinh + log_cosh
    )


def sample_antipodal_vmf_axis(
    rng: np.random.Generator,
    direction: np.ndarray,
    kappa: float,
) -> np.ndarray:
    centre = canonical_axis(direction)
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("vMF kappa must be positive and finite")
    sign = -1.0 if float(rng.random()) < 0.5 else 1.0
    signed_centre = sign * centre
    uniform = float(rng.random())
    cosine = 1.0 + math.log(
        uniform + (1.0 - uniform) * math.exp(-2.0 * float(kappa))
    ) / float(kappa)
    cosine = min(1.0, max(-1.0, cosine))
    azimuth = 2.0 * math.pi * float(rng.random())
    first, second = _orthogonal_basis(centre)
    signed_first = sign * first
    sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
    value = (
        cosine * signed_centre
        + sine * (math.cos(azimuth) * signed_first + math.sin(azimuth) * second)
    )
    return canonical_axis(value)


def initialize_particles(
    master_seed: int,
    particle_count: int = PARTICLE_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    midpoint = np.empty((particle_count, 3), dtype=np.float64)
    axis = np.empty((particle_count, 3), dtype=np.float64)
    for particle in range(particle_count):
        rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(
            int(master_seed), spawn_key=(0, particle)
        )))
        midpoint[particle] = rng.normal(PRIOR_MEAN_MPC_H, PRIOR_SIGMA_MPC_H)
        axis[particle] = sample_isotropic_axis(rng)
    return midpoint, axis


def _categorical_index(
    rng: np.random.Generator,
    probability: np.ndarray,
) -> int:
    draw = float(rng.random())
    cumulative = np.cumsum(probability)
    cumulative[-1] = 1.0
    return int(np.searchsorted(cumulative, draw, side="left"))


def propose_particle(
    midpoint: np.ndarray,
    axis: np.ndarray,
    master_seed: int,
    stage: int,
    sweep: int,
    particle: int,
) -> tuple[np.ndarray, np.ndarray, str, int | None, int | None, np.random.Generator]:
    rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(
        int(master_seed), spawn_key=(2, int(stage), int(sweep), int(particle))
    )))
    move_index = _categorical_index(rng, MOVE_PROBABILITIES)
    move = MOVE_NAMES[move_index]
    proposed_q = np.asarray(midpoint, dtype=np.float64).copy()
    proposed_axis = canonical_axis(axis)
    q_scale_index = None
    axis_scale_index = None
    if move in ("q_local", "joint_local"):
        q_scale_index = _categorical_index(rng, Q_SCALE_PROBABILITIES)
        proposed_q += (
            PRIOR_SIGMA_MPC_H * Q_SCALES[q_scale_index] * rng.normal(size=3)
        )
    if move in ("axis_local", "joint_local"):
        axis_scale_index = _categorical_index(rng, AXIS_KAPPA_PROBABILITIES)
        proposed_axis = sample_antipodal_vmf_axis(
            rng, proposed_axis, AXIS_KAPPA[axis_scale_index]
        )
    if move == "prior_independence":
        proposed_q = rng.normal(PRIOR_MEAN_MPC_H, PRIOR_SIGMA_MPC_H)
        proposed_axis = sample_isotropic_axis(rng)
    return (
        proposed_q,
        proposed_axis,
        move,
        q_scale_index,
        axis_scale_index,
        rng,
    )


def mh_log_acceptance(
    move: str,
    current_midpoint: np.ndarray,
    proposed_midpoint: np.ndarray,
    current_log_z_bar: float,
    proposed_log_z_bar: float,
    beta: float,
) -> float:
    if move not in MOVE_NAMES:
        raise ValueError("unknown frozen MH move")
    if not np.all(np.isfinite([
        current_log_z_bar, proposed_log_z_bar, beta
    ])):
        raise ValueError("MH acceptance inputs must be finite")
    result = float(beta) * (float(proposed_log_z_bar) - float(current_log_z_bar))
    if move in ("q_local", "joint_local"):
        result += float(
            diagonal_normal_logpdf(np.asarray(proposed_midpoint)[None])[0]
            - diagonal_normal_logpdf(np.asarray(current_midpoint)[None])[0]
        )
    return min(0.0, result)


def mh_rejuvenation_sweep(
    midpoint: np.ndarray,
    axis: np.ndarray,
    keys: np.ndarray,
    log_z_bar: np.ndarray,
    beta: float,
    oracle,
    master_seed: int,
    stage: int,
    sweep: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    count = len(midpoint)
    proposed_q = np.empty_like(midpoint)
    proposed_axis = np.empty_like(axis)
    metadata = []
    for particle in range(count):
        proposal = propose_particle(
            midpoint[particle], axis[particle], master_seed, stage, sweep, particle
        )
        proposed_q[particle] = proposal[0]
        proposed_axis[particle] = proposal[1]
        metadata.append(proposal[2:])
    proposed_keys, proposed_log_z = oracle.evaluate(proposed_q, proposed_axis)
    accepted = np.zeros(count, dtype=bool)
    proposal_count = {name: 0 for name in MOVE_NAMES}
    acceptance_count = {name: 0 for name in MOVE_NAMES}
    q_scale_proposal = np.zeros(len(Q_SCALES), dtype=np.int64)
    q_scale_acceptance = np.zeros(len(Q_SCALES), dtype=np.int64)
    axis_scale_proposal = np.zeros(len(AXIS_KAPPA), dtype=np.int64)
    axis_scale_acceptance = np.zeros(len(AXIS_KAPPA), dtype=np.int64)
    for particle, (move, q_index, axis_index, rng) in enumerate(metadata):
        proposal_count[move] += 1
        if q_index is not None:
            q_scale_proposal[q_index] += 1
        if axis_index is not None:
            axis_scale_proposal[axis_index] += 1
        log_acceptance = mh_log_acceptance(
            move,
            midpoint[particle],
            proposed_q[particle],
            log_z_bar[particle],
            proposed_log_z[particle],
            beta,
        )
        uniform = float(rng.random())
        log_uniform = -math.inf if uniform == 0.0 else math.log(uniform)
        if log_uniform <= log_acceptance:
            accepted[particle] = True
            acceptance_count[move] += 1
            if q_index is not None:
                q_scale_acceptance[q_index] += 1
            if axis_index is not None:
                axis_scale_acceptance[axis_index] += 1
    midpoint = np.where(accepted[:, None], proposed_q, midpoint)
    axis = np.where(accepted[:, None], proposed_axis, axis)
    keys = np.where(accepted[:, None], proposed_keys, keys)
    log_z_bar = np.where(accepted, proposed_log_z, log_z_bar)
    if not np.all(np.isfinite(log_z_bar)):
        raise RuntimeError("MH oracle produced nonfinite aggregate evidence")
    return midpoint, axis, keys, log_z_bar, {
        "proposal_count": proposal_count,
        "acceptance_count": acceptance_count,
        "q_scale_proposal_count": q_scale_proposal,
        "q_scale_acceptance_count": q_scale_acceptance,
        "axis_scale_proposal_count": axis_scale_proposal,
        "axis_scale_acceptance_count": axis_scale_acceptance,
    }


@dataclass
class SMCReplicate:
    master_seed: int
    midpoint_mpc_h: np.ndarray
    axis: np.ndarray
    keys: np.ndarray
    weights: np.ndarray
    log_z_bar: np.ndarray
    ancestor_labels: np.ndarray
    beta_history: np.ndarray
    conditional_ess_history: np.ndarray
    particle_ess_history: np.ndarray
    log_normalizer_increment: np.ndarray
    resampling_ancestors: list[np.ndarray]
    move_history: list[list[dict[str, Any]]]
    log_normalizer: float

    @property
    def genealogical_ess(self) -> float:
        return genealogical_ess(self.ancestor_labels, len(self.weights))


def run_smc_replicate(
    master_seed: int,
    oracle,
    *,
    particle_count: int = PARTICLE_COUNT,
    target_cess_fraction: float = TARGET_CESS_FRACTION,
    resampling_ess_fraction: float = RESAMPLING_ESS_FRACTION,
    maximum_temperature_stages: int = MAXIMUM_TEMPERATURE_STAGES,
    sweeps_per_stage: int = SWEEPS_PER_STAGE,
) -> SMCReplicate:
    midpoint, axis = initialize_particles(master_seed, particle_count)
    keys, log_z_bar = oracle.evaluate(midpoint, axis)
    if not np.all(np.isfinite(log_z_bar)):
        raise RuntimeError("initial aggregate evidence is nonfinite")
    weights = np.full(particle_count, 1.0 / particle_count, dtype=np.float64)
    ancestors = np.arange(particle_count, dtype=np.int64)
    beta = 0.0
    log_normalizer = 0.0
    beta_history = [beta]
    cess_history = []
    ess_history = [particle_ess(weights)]
    normalizer_increment = []
    resampling_ancestors = []
    move_history = []
    resampling_event = 0
    for stage in range(maximum_temperature_stages):
        delta = select_temperature_increment(
            beta, weights, log_z_bar, target_fraction=target_cess_fraction
        )
        remaining = 1.0 - beta
        if beta < 1.0 and delta <= 0.0:
            raise RuntimeError("positive remaining temperature requires positive delta")
        if remaining > TEMPERATURE_ENDPOINT_TOLERANCE and delta <= (
            TEMPERATURE_STAGNATION_DELTA
        ):
            raise RuntimeError("temperature schedule stagnated")
        if delta == 0.0 and remaining == 0.0:
            break
        cess = conditional_ess(weights, log_z_bar, delta)
        weights, log_increment = update_weights_and_normalizer(
            weights, log_z_bar, delta
        )
        beta = min(1.0, beta + delta)
        if 1.0 - beta <= np.finfo(float).eps:
            beta = 1.0
        log_normalizer += log_increment
        pre_resampling_ess = particle_ess(weights)
        cess_history.append(cess)
        normalizer_increment.append(log_increment)
        if pre_resampling_ess < resampling_ess_fraction * particle_count:
            rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(
                int(master_seed), spawn_key=(1, resampling_event)
            )))
            selected = systematic_resampling(weights, rng)
            resampling_ancestors.append(selected)
            midpoint = midpoint[selected]
            axis = axis[selected]
            keys = keys[selected]
            log_z_bar = log_z_bar[selected]
            ancestors = ancestors[selected]
            weights = np.full(
                particle_count, 1.0 / particle_count, dtype=np.float64
            )
            resampling_event += 1
        stage_moves = []
        for sweep in range(sweeps_per_stage):
            midpoint, axis, keys, log_z_bar, move = mh_rejuvenation_sweep(
                midpoint,
                axis,
                keys,
                log_z_bar,
                beta,
                oracle,
                master_seed,
                stage,
                sweep,
            )
            stage_moves.append(move)
        move_history.append(stage_moves)
        beta_history.append(beta)
        ess_history.append(pre_resampling_ess)
        if beta == 1.0:
            break
    if beta != 1.0:
        raise RuntimeError("SMC replicate did not reach beta=1")
    return SMCReplicate(
        master_seed=int(master_seed),
        midpoint_mpc_h=midpoint,
        axis=axis,
        keys=keys,
        weights=weights,
        log_z_bar=log_z_bar,
        ancestor_labels=ancestors,
        beta_history=np.asarray(beta_history),
        conditional_ess_history=np.asarray(cess_history),
        particle_ess_history=np.asarray(ess_history),
        log_normalizer_increment=np.asarray(normalizer_increment),
        resampling_ancestors=resampling_ancestors,
        move_history=move_history,
        log_normalizer=float(log_normalizer),
    )


def conditional_parent_probabilities(parent_log_z: np.ndarray) -> np.ndarray:
    value = np.asarray(parent_log_z, dtype=np.float64)
    if value.ndim != 2 or not np.all(np.isfinite(value)):
        raise ValueError("terminal parent evidence must be a finite matrix")
    maximum = np.max(value, axis=1, keepdims=True)
    result = np.exp(value - maximum)
    result /= np.sum(result, axis=1, keepdims=True)
    return result


def replicate_parent_probability(
    weights: np.ndarray,
    parent_log_z: np.ndarray,
) -> np.ndarray:
    value = np.asarray(weights, dtype=np.float64)
    conditional = conditional_parent_probabilities(parent_log_z)
    if value.shape != (len(conditional),):
        raise ValueError("terminal weights and parent evidence do not align")
    if (
        not np.all(np.isfinite(value))
        or np.any(value < 0.0)
        or not np.isclose(np.sum(value), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("terminal weights must be finite, nonnegative, and normalized")
    result = np.sum(value[:, None] * conditional, axis=0)
    result /= np.sum(result)
    return result


def pool_parent_probabilities(
    log_normalizers: np.ndarray,
    replicate_probabilities: np.ndarray,
) -> tuple[np.ndarray, float]:
    log_i = np.asarray(log_normalizers, dtype=np.float64)
    probability = np.asarray(replicate_probabilities, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[0] != len(log_i):
        raise ValueError("replicate normalizers and parent vectors do not align")
    if not np.all(np.isfinite(log_i)) or np.any(probability < 0.0):
        raise ValueError("replicate pooling inputs must be finite probabilities")
    if not np.allclose(
        np.sum(probability, axis=1), 1.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("each replicate parent vector must be normalized")
    maximum = float(np.max(log_i))
    relative_integral = np.exp(log_i - maximum)
    parent_integral = np.mean(relative_integral[:, None] * probability, axis=0)
    pooled = parent_integral / np.sum(parent_integral)
    pooled_log_i_bar = maximum + math.log(float(np.mean(relative_integral)))
    return pooled, pooled_log_i_bar
