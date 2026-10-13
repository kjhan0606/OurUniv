#!/usr/bin/env python3
"""Run the preregistered 192-mock Q3 development-only exact-grouping check."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from cf4_q1_cell_integrated_convolution import cell_integrated_tsc_deposit, q1_candidate_oracle_gate
from cf4_q3_source_compressed_operator import (
    aggregate_mass_basis,
    evaluate_grouped_q1_operator,
    exact_geometry_compress,
)


N_MOCKS = 192
SEED_NAMESPACE = "cf4-q3-development-20260904"


def mock_seed(index: int) -> int:
    digest = hashlib.sha256(f"{SEED_NAMESPACE}:{index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def run() -> dict[str, object]:
    seeds = [mock_seed(index) for index in range(N_MOCKS)]
    if len(set(seeds)) != N_MOCKS:
        raise RuntimeError("development seed collision")
    max_relative = 0.0
    max_absolute = 0.0
    ratios: list[float] = []
    basis_checks = 0
    for index, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        unique_positions = rng.uniform(0.5, 7.5, size=(2, 3))
        positions = np.vstack([unique_positions[0], unique_positions[0], unique_positions[1], unique_positions[1]])
        los = np.zeros((4, 3), dtype=np.float64)
        los[0] = los[1] = np.array([1.0, 0.0, 0.0])
        los[2] = los[3] = np.array([0.0, 1.0, 0.0])
        scale = np.asarray([0.25, 0.25, 0.75, 0.75], dtype=np.float64)
        masses = rng.uniform(0.1, 2.0, size=(6, 4))
        compressed = exact_geometry_compress(positions, los, scale)
        grouped = evaluate_grouped_q1_operator(compressed, masses, 8, 8.0)
        reference = np.stack(
            [cell_integrated_tsc_deposit(positions, masses[p], los, scale, 8, 8.0) for p in range(6)]
        )
        gate = q1_candidate_oracle_gate(grouped, reference)
        if gate["status"] != "PASS":
            raise RuntimeError(f"development gate failed at mock {index}: {gate}")
        max_relative = max(max_relative, float(gate["relative_l1"]))
        max_absolute = max(max_absolute, float(gate["absolute_l1"]))
        ratios.append(compressed.compression_ratio)
        basis = rng.normal(size=(6, 4, 5))
        grouped_basis = aggregate_mass_basis(compressed, basis)
        expected_basis = basis[:, 0, :] + basis[:, 1, :], basis[:, 2, :] + basis[:, 3, :]
        if not np.allclose(grouped_basis[:, 0, :], expected_basis[0], rtol=0.0, atol=0.0):
            raise RuntimeError(f"mass-basis preservation failed at mock {index}")
        if not np.allclose(grouped_basis[:, 1, :], expected_basis[1], rtol=0.0, atol=0.0):
            raise RuntimeError(f"mass-basis preservation failed at mock {index}")
        basis_checks += 1
    return {
        "schema": "ouruniv-cf4-q3-development-candidate-result-v1",
        "candidate": "exact_geometry_grouping",
        "development_mock_count": N_MOCKS,
        "seed_namespace": SEED_NAMESPACE,
        "seed_derivation": "SHA256(namespace+':'+index) first 8 bytes, masked to 63 bits",
        "source_count_per_mock": 4,
        "group_count_per_mock": 2,
        "compression_ratio": {"min": min(ratios), "max": max(ratios), "mean": float(np.mean(ratios))},
        "q1_oracle_gate": "PASS for all mocks and all six populations",
        "max_relative_l1": max_relative,
        "max_absolute_l1": max_absolute,
        "mass_basis_checks": basis_checks,
        "heldout_evaluation": False,
        "posterior_sampling": False,
        "science_claim_authorized": False,
        "production_execution": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
