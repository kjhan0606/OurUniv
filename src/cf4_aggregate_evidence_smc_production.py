#!/usr/bin/env python3
"""Fixed four-replicate SMC driver; execution requires a future sealed program."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from cf4_aggregate_evidence_oracle import (
    PRODUCTION_PARENT_SEEDS,
    PRODUCTION_PARTICLE_COUNT,
    PRODUCTION_REPLICATE_MASTER_SEEDS,
)
from cf4_aggregate_evidence_smc import (
    MAXIMUM_TEMPERATURE_STAGES,
    PARTICLE_COUNT,
    SWEEPS_PER_STAGE,
    SMCReplicate,
    pool_parent_probabilities,
    replicate_parent_probability,
    run_smc_replicate,
)


@dataclass
class FourReplicateSMC:
    replicates: tuple[SMCReplicate, ...]
    replicate_parent_probability: np.ndarray
    pooled_parent_probability: np.ndarray
    pooled_log_i_bar: float


ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = ROOT / "config/cf4_aggregate_evidence_annealed_smc_design.json"
DESIGN_SHA256 = "47397a00b497dc09e0e517cebe61a285cebdf9aa4b0a0ef1688af588c2c58609"


def run_four_production_replicates(oracle) -> FourReplicateSMC:
    """Fail closed until a future sealed production capability is audited."""
    actual_sha = hashlib.sha256(DESIGN_PATH.read_bytes()).hexdigest()
    if actual_sha != DESIGN_SHA256:
        raise RuntimeError("committed SMC design hash mismatch")
    design = json.loads(DESIGN_PATH.read_text())
    if design["authorization"].get("production_execution_authorized") is not True:
        raise PermissionError("production SMC is not authorized by the sealed design")
    raise PermissionError(
        "a separate sealed production program and capability are still required"
    )


def _run_four_replicates_core_for_validation(oracle) -> FourReplicateSMC:
    """Private fixed core used only by tests before production authorization."""
    if (
        PARTICLE_COUNT != 2048
        or PRODUCTION_PARTICLE_COUNT != 2048
        or len(PRODUCTION_PARENT_SEEDS) != 256
        or PRODUCTION_REPLICATE_MASTER_SEEDS
        != (2026082301, 2026082302, 2026082303, 2026082304)
        or SWEEPS_PER_STAGE != 4
        or MAXIMUM_TEMPERATURE_STAGES != 256
    ):
        raise RuntimeError("compiled production constants differ from the design")
    replicates = []
    for master_seed in PRODUCTION_REPLICATE_MASTER_SEEDS:
        replicate = run_smc_replicate(master_seed, oracle)
        if (
            replicate.master_seed != master_seed
            or replicate.keys.shape != (2048, 6)
            or replicate.weights.shape != (2048,)
            or replicate.beta_history[-1] != 1.0
            or len(replicate.beta_history) - 1 > 256
            or any(len(stage) != 4 for stage in replicate.move_history)
        ):
            raise RuntimeError("SMC replicate broke the fixed production contract")
        oracle.register_terminal_history(master_seed, replicate.keys)
        replicates.append(replicate)
    oracle.seal_terminal_histories()
    parent_vectors = []
    for master_seed, replicate in zip(
        PRODUCTION_REPLICATE_MASTER_SEEDS, replicates
    ):
        parent_log_z = oracle.terminal_parent_log_z(
            master_seed, replicate.keys
        )
        if parent_log_z.shape != (2048, 256):
            raise RuntimeError("terminal parent evidence shape mismatch")
        parent_vectors.append(replicate_parent_probability(
            replicate.weights, parent_log_z
        ))
    parent_vectors = np.stack(parent_vectors)
    pooled_parent, pooled_log_i_bar = pool_parent_probabilities(
        np.asarray([replicate.log_normalizer for replicate in replicates]),
        parent_vectors,
    )
    return FourReplicateSMC(
        replicates=tuple(replicates),
        replicate_parent_probability=parent_vectors,
        pooled_parent_probability=pooled_parent,
        pooled_log_i_bar=pooled_log_i_bar,
    )
