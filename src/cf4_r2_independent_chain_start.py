"""Independent prior starts and isolated random streams for conditional-v6 chains.

Only an explicitly identified zero count likelihood may be retried. Support
construction errors and every other nonfinite/evaluation failure abort the
chain instead of selecting around numerical problems.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np


CHAIN_COUNT = 4
STREAM_IDS = {"prior": 0, "audit": 1, "transition": 2}
MAX_PRIOR_START_ATTEMPTS = 4


class ZeroCountLikelihoodStart(Exception):
    """A start has count score -inf while its raw-mark score is finite."""

    def __init__(self, *, count_score, raw_score, reason):
        count_score = float(count_score)
        raw_score = float(raw_score)
        if not (math.isinf(count_score) and count_score < 0
                and math.isfinite(raw_score)):
            raise ValueError("only -inf count score with finite raw score is retryable")
        self.count_score = count_score
        self.raw_score = raw_score
        self.reason = str(reason)
        if not self.reason:
            raise ValueError("a zero-count retry requires a reason")
        super().__init__(self.reason)


class PriorStartSelectionError(RuntimeError):
    """Fail-closed prior-start error with JSON-safe attempt history."""

    status = "PRIOR_START_SELECTION_FAILED"

    def __init__(self, message, attempts, *, attempt_limit=MAX_PRIOR_START_ATTEMPTS):
        self.attempts = tuple(dict(row) for row in attempts)
        self.attempt_limit = int(attempt_limit)
        super().__init__(message)

    def as_report(self):
        return dict(status=self.status,
                    max_attempts=self.attempt_limit,
                    attempts=[dict(row) for row in self.attempts])


class NoFinitePriorStart(PriorStartSelectionError):
    """The allowed prior draws all had a genuine zero count likelihood."""

    status = "NO_FINITE_PRIOR_START_WITHIN_RETRY_CAP"


class PriorStartEvaluationError(PriorStartSelectionError):
    """A prior target evaluation failed numerically or unexpectedly."""

    status = "PRIOR_START_EVALUATION_FAILED"


@dataclass(frozen=True)
class PriorStartSelection:
    state: np.ndarray
    target_energy: float
    attempts: tuple[dict, ...]
    attempt_limit: int

    def as_report(self):
        return dict(status="FINITE_PRIOR_START_SELECTED",
                    max_attempts=self.attempt_limit,
                    attempts=[dict(row) for row in self.attempts],
                    selected_attempt=len(self.attempts),
                    selected_state_sha256=self.attempts[-1]["state_sha256"],
                    selected_target_energy=self.target_energy)


def is_retryable_zero_count_likelihood(total_score, score_parts, *, order,
                                       gradient, retry_enabled):
    """Apply the sole permitted prior-start retry classification."""
    if not retry_enabled or order != 2 or gradient:
        return False
    try:
        total_score = float(total_score)
        parts = np.asarray(score_parts, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return False
    return (parts.shape == (2,) and np.isneginf(total_score)
            and np.isneginf(parts[0]) and math.isfinite(float(parts[1])))


def make_independent_chain_streams(master_seed, chain_index, *, chain_count=CHAIN_COUNT):
    """Return deterministic, chain-indexed prior/audit/transition RNGs.

    The project contract is exactly four chains. Stream identities use
    SeedSequence spawn keys, so drawing from one purpose cannot advance either
    of the others.
    """
    if (not isinstance(master_seed, (int, np.integer))
            or isinstance(master_seed, (bool, np.bool_)) or int(master_seed) < 0):
        raise ValueError("master seed must be a nonnegative integer")
    if (not isinstance(chain_index, (int, np.integer))
            or isinstance(chain_index, (bool, np.bool_))
            or not 0 <= int(chain_index) < CHAIN_COUNT):
        raise ValueError("chain index must be one of 0, 1, 2, 3")
    if chain_count != CHAIN_COUNT:
        raise ValueError("the frozen conditional-v6 design requires exactly four chains")
    master_seed = int(master_seed)
    chain_index = int(chain_index)
    generators = {
        name: np.random.default_rng(np.random.SeedSequence(
            master_seed, spawn_key=(chain_index, stream_id)))
        for name, stream_id in STREAM_IDS.items()
    }
    metadata = dict(master_seed=master_seed, chain_index=chain_index,
        chain_count=CHAIN_COUNT, bit_generator="PCG64",
        streams={name: dict(stream_id=stream_id,
            spawn_key=[chain_index, stream_id])
            for name, stream_id in STREAM_IDS.items()})
    return generators, metadata


def _state_sha256(state):
    canonical = np.ascontiguousarray(np.asarray(state, dtype=np.dtype("<f8")))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _failure_reason(error):
    return f"{type(error).__name__}: {error}"


def select_prior_start(prior_rng, dimension, probe, *, max_attempts=MAX_PRIOR_START_ATTEMPTS):
    """Draw complete standard-normal states until finite support or fail.

    ``probe(q)`` must return a finite scalar target energy. It may raise
    :class:`ZeroCountLikelihoodStart` only for the narrowly retryable case.
    Any other exception or nonfinite energy aborts immediately.
    """
    if not isinstance(prior_rng, np.random.Generator):
        raise TypeError("prior_rng must be a NumPy Generator")
    if (not isinstance(dimension, (int, np.integer))
            or isinstance(dimension, (bool, np.bool_)) or int(dimension) < 1):
        raise ValueError("positive canonical-state dimension required")
    if (not isinstance(max_attempts, (int, np.integer))
            or isinstance(max_attempts, (bool, np.bool_))
            or not 1 <= int(max_attempts) <= MAX_PRIOR_START_ATTEMPTS):
        raise ValueError("prior-start attempts must be between one and the frozen cap of four")
    dimension = int(dimension)
    max_attempts = int(max_attempts)
    attempts = []

    for attempt_number in range(1, max_attempts + 1):
        state = prior_rng.standard_normal(dimension, dtype=np.float64)
        checksum = _state_sha256(state)
        try:
            raw_energy = probe(state)
            if np.ndim(raw_energy) != 0 or isinstance(raw_energy, (bool, np.bool_)):
                raise TypeError("prior target probe must return one scalar energy")
            energy = float(raw_energy)
            if not math.isfinite(energy):
                raise FloatingPointError("prior target probe returned a nonfinite energy")
        except ZeroCountLikelihoodStart as error:
            attempts.append(dict(attempt=attempt_number,
                status="ZERO_COUNT_LIKELIHOOD", state_sha256=checksum,
                target_energy=None, count_score="-inf", raw_score=error.raw_score,
                reason=error.reason))
            continue
        except Exception as error:
            attempts.append(dict(attempt=attempt_number,
                status="NUMERICAL_OR_TARGET_FAILURE", state_sha256=checksum,
                target_energy=None, reason=_failure_reason(error)))
            raise PriorStartEvaluationError(
                "prior target evaluation failed; retry is not permitted", attempts,
                attempt_limit=max_attempts) from error

        attempts.append(dict(attempt=attempt_number, status="FINITE",
            state_sha256=checksum, target_energy=energy, reason="finite full target"))
        return PriorStartSelection(state=state, target_energy=energy,
                                   attempts=tuple(attempts),attempt_limit=max_attempts)

    raise NoFinitePriorStart(
        f"no finite prior start in {max_attempts} allowed draws", attempts,
        attempt_limit=max_attempts)
