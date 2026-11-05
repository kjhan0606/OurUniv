"""v6-only shared-annealing contracts; no execution entry point is exposed.

The pilot is deliberately a schedule-only disposable computation.  This file
does not name a v5 namespace and cannot issue a grant, create a receipt, or
open a production evaluator.  A future separately authorized runner must
provide the two factories required by :func:`run_shared_annealing_plan`.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from typing import Callable, Protocol, Sequence

import numpy as np

from cf4_aggregate_evidence_oracle import (
    PRODUCTION_PARENT_SEEDS,
    PRODUCTION_REPLICATE_MASTER_SEEDS,
)
from cf4_aggregate_evidence_smc import (
    MAXIMUM_TEMPERATURE_STAGES,
    PARTICLE_COUNT,
    RESAMPLING_ESS_FRACTION,
    SWEEPS_PER_STAGE,
    TARGET_CESS_FRACTION,
)


V6_MASTER_SEEDS = (2026082301, 2026082302, 2026082303, 2026082304)
V6_PARTICLE_COUNT = 2048
V6_PARENT_SEEDS = tuple(range(3193, 3449))
NULL_CALIBRATION_SEED = 2026081801
NULL_CALIBRATION_DRAWS = 20_000
NULL_TAIL_PASS_MINIMUM = 0.001
LOG_I_RANGE_MAXIMUM = 0.2
LOG_I_SE_MAXIMUM = 0.1
L1_DIAGNOSTIC_THRESHOLD = 0.2


class ArchitectureFailure(RuntimeError):
    """An invalid shared-annealing architecture, never a scientific result."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def validate_frozen_v6_parameters() -> None:
    """Hard-bind v6 to the frozen v5/base scientific constants."""
    if not (
        PRODUCTION_REPLICATE_MASTER_SEEDS == V6_MASTER_SEEDS
        and PARTICLE_COUNT == V6_PARTICLE_COUNT
        and PRODUCTION_PARENT_SEEDS == V6_PARENT_SEEDS
        and TARGET_CESS_FRACTION == 0.8
        and RESAMPLING_ESS_FRACTION == 0.5
        and SWEEPS_PER_STAGE == 4
        and MAXIMUM_TEMPERATURE_STAGES == 256
    ):
        raise ArchitectureFailure("v6 frozen scientific parameters differ from base")


@dataclass(frozen=True)
class SharedBetaSchedule:
    """The sole permitted pilot-derived value passed into production."""

    beta: tuple[float, ...]
    pilot_master_seed: int
    pilot_particle_count: int
    pilot_parent_seeds: tuple[int, ...]
    schedule_sha256: str


def freeze_shared_beta_schedule(
    beta_history: Sequence[float],
    *,
    pilot_master_seed: int,
    pilot_particle_count: int,
    pilot_parent_seeds: Sequence[int],
) -> SharedBetaSchedule:
    """Validate and freeze the pilot's deterministic beta schedule only."""
    validate_frozen_v6_parameters()
    beta = tuple(float(value) for value in beta_history)
    if (
        pilot_master_seed != V6_MASTER_SEEDS[0]
        or pilot_particle_count != V6_PARTICLE_COUNT
        or tuple(pilot_parent_seeds) != V6_PARENT_SEEDS
        or len(beta) < 2
        or beta[0] != 0.0
        or beta[-1] != 1.0
        or not all(np.isfinite(value) for value in beta)
        or any(not left < right for left, right in zip(beta, beta[1:]))
        or len(beta) - 1 > MAXIMUM_TEMPERATURE_STAGES
    ):
        raise ArchitectureFailure("v6 pilot schedule contract failed")
    payload = {
        "schema": "ouruniv-cf4-shared-beta-schedule-v6",
        "beta": beta,
        "pilot_master_seed": pilot_master_seed,
        "pilot_particle_count": pilot_particle_count,
        "pilot_parent_seeds": tuple(pilot_parent_seeds),
    }
    return SharedBetaSchedule(
        beta=beta,
        pilot_master_seed=pilot_master_seed,
        pilot_particle_count=pilot_particle_count,
        pilot_parent_seeds=tuple(pilot_parent_seeds),
        schedule_sha256=_sha256(payload),
    )


class DisposablePilot(Protocol):
    beta_history: Sequence[float]
    master_seed: int
    particle_count: int
    parent_seeds: Sequence[int]
    closed: bool
    cache_disposed: bool

    def close(self) -> None: ...


@dataclass(frozen=True)
class FreshProductionReplicate:
    master_seed: int
    beta_history: tuple[float, ...]
    evaluator_namespace: str
    cache_namespace: str
    pilot_cache_reused: bool = False
    pilot_posterior_reused: bool = False
    pilot_scientific_result_reused: bool = False


def _verify_disposed_pilot(pilot: DisposablePilot) -> None:
    if not bool(pilot.closed) or not bool(pilot.cache_disposed):
        raise ArchitectureFailure("v6 pilot evaluator/cache was not closed and disposed")


def verify_stage_parity(
    schedule: SharedBetaSchedule, records: Sequence[FreshProductionReplicate]
) -> None:
    expected = schedule.beta
    if tuple(record.master_seed for record in records) != V6_MASTER_SEEDS:
        raise ArchitectureFailure("v6 production masters are not the fixed four seeds")
    for record in records:
        if (
            record.beta_history != expected
            or "v6" not in record.evaluator_namespace
            or "v6" not in record.cache_namespace
            or record.pilot_cache_reused
            or record.pilot_posterior_reused
            or record.pilot_scientific_result_reused
        ):
            raise ArchitectureFailure("shared_schedule_stage_parity_architecture_failure")


def run_shared_annealing_plan(
    pilot_factory: Callable[[], DisposablePilot],
    production_factory: Callable[[int, SharedBetaSchedule], FreshProductionReplicate],
) -> tuple[SharedBetaSchedule, tuple[FreshProductionReplicate, ...]]:
    """Orchestrate the future runner without exposing pilot objects to production.

    The returned production records are metadata-only contracts.  Factories are
    intentionally supplied by a future authorized runner; this v6 addition has
    no CLI and cannot execute science on import.
    """
    pilot = pilot_factory()
    try:
        schedule = freeze_shared_beta_schedule(
            pilot.beta_history,
            pilot_master_seed=pilot.master_seed,
            pilot_particle_count=pilot.particle_count,
            pilot_parent_seeds=pilot.parent_seeds,
        )
    finally:
        pilot.close()
    _verify_disposed_pilot(pilot)
    records = tuple(production_factory(seed, schedule) for seed in V6_MASTER_SEEDS)
    verify_stage_parity(schedule, records)
    return schedule, records


@dataclass(frozen=True)
class L1NullCalibration:
    draws: int
    seed: int
    q99: float
    q999: float
    tail_probability: float


def max_six_pairwise_l1(p_rep: np.ndarray) -> float:
    value = np.asarray(p_rep, dtype=np.float64)
    if (
        value.shape != (4, 256)
        or not np.all(np.isfinite(value))
        or np.any(value < 0.0)
        or not np.allclose(value.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("v6 replicate parent probabilities must be four normalized 256-vectors")
    return float(max(np.sum(np.abs(value[left] - value[right])) for left, right in itertools.combinations(range(4), 2)))


def calibrate_max_l1_null(
    p_pool: np.ndarray, observed_s: float, *, draws: int = NULL_CALIBRATION_DRAWS,
    seed: int = NULL_CALIBRATION_SEED,
) -> L1NullCalibration:
    """PCG64DXSM familywise calibration from four N=2048 multinomial draws."""
    p = np.asarray(p_pool, dtype=np.float64)
    if (
        p.shape != (256,) or not np.all(np.isfinite(p)) or np.any(p < 0.0)
        or not np.isclose(p.sum(), 1.0, rtol=0.0, atol=1e-12)
        or not np.isfinite(observed_s) or observed_s < 0.0 or draws < 1
    ):
        raise ValueError("v6 null calibration inputs are invalid")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    counts = rng.multinomial(V6_PARTICLE_COUNT, p, size=(draws, 4)).astype(np.int32)
    maxima = np.zeros(draws, dtype=np.int32)
    for left, right in itertools.combinations(range(4), 2):
        maxima = np.maximum(maxima, np.abs(counts[:, left] - counts[:, right]).sum(axis=1))
    statistics = maxima.astype(np.float64) / V6_PARTICLE_COUNT
    # "higher" preserves an actual discrete null statistic and is deterministic.
    q99, q999 = np.quantile(statistics, [0.99, 0.999], method="higher")
    tail = float((np.count_nonzero(statistics >= observed_s) + 1) / (draws + 1))
    return L1NullCalibration(draws=draws, seed=seed, q99=float(q99), q999=float(q999), tail_probability=tail)


def paired_incoherence_diagnostics(
    log_i_bar: np.ndarray, p_rep: np.ndarray, p_pool: np.ndarray,
    *, calibration_draws: int = NULL_CALIBRATION_DRAWS,
) -> dict[str, object]:
    """Evaluate v6 coherence while retaining log-I gates as diagnostics."""
    log_i = np.asarray(log_i_bar, dtype=np.float64)
    if log_i.shape != (4,) or not np.all(np.isfinite(log_i)):
        raise ValueError("v6 log-I diagnostics require four finite values")
    l1_statistic = max_six_pairwise_l1(p_rep)
    log_range = float(np.max(log_i) - np.min(log_i))
    log_se = float(np.std(log_i, ddof=1) / np.sqrt(4.0))
    calibration = calibrate_max_l1_null(
        p_pool, l1_statistic, draws=calibration_draws
    )
    failed_channels: list[str] = []
    if log_range > LOG_I_RANGE_MAXIMUM:
        failed_channels.append("replicate_log_I_bar_range")
    if log_se > LOG_I_SE_MAXIMUM:
        failed_channels.append("replicate_log_I_bar_sample_SE")
    if calibration.tail_probability < NULL_TAIL_PASS_MINIMUM:
        failed_channels.append("replicate_parent_probability_L1")
    if {"replicate_log_I_bar_range", "replicate_parent_probability_L1"}.issubset(failed_channels):
        primary = "paired_incoherence"
    elif failed_channels:
        primary = failed_channels[0]
    else:
        primary = None
    return {
        "failed_channels": failed_channels,
        "primary_failure": primary,
        "log_I_bar_range": log_range,
        "log_I_bar_range_pass": log_range <= LOG_I_RANGE_MAXIMUM,
        "log_I_bar_sample_SE": log_se,
        "log_I_bar_sample_SE_pass": log_se <= LOG_I_SE_MAXIMUM,
        "max_six_pairwise_parent_probability_L1": l1_statistic,
        "L1_diagnostic_threshold": L1_DIAGNOSTIC_THRESHOLD,
        "L1_diagnostic_threshold_pass": l1_statistic <= L1_DIAGNOSTIC_THRESHOLD,
        "null_calibration": {
            "bit_generator": "PCG64DXSM", "seed": calibration.seed,
            "draws": calibration.draws, "q99": calibration.q99,
            "q999": calibration.q999, "tail_probability": calibration.tail_probability,
            "coherent_pass": calibration.tail_probability >= NULL_TAIL_PASS_MINIMUM,
            "tail_probability_minimum": NULL_TAIL_PASS_MINIMUM,
        },
    }
