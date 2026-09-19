#!/usr/bin/env python3
"""Audit whether the promoted high-k schedule can survive legacy P1 prefiltering."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_support(
    parent_seed: np.ndarray,
    parent_probability: np.ndarray,
    schedule_parent_seed: np.ndarray,
    schedule_weight: np.ndarray,
    p1: dict[str, Any],
) -> dict[str, Any]:
    seed = np.asarray(parent_seed, dtype=np.int64)
    probability = np.asarray(parent_probability, dtype=np.float64)
    scheduled = np.asarray(schedule_parent_seed, dtype=np.int64)
    scheduled_weight = np.asarray(schedule_weight, dtype=np.float64)
    if seed.ndim != 1 or probability.shape != seed.shape \
            or len(np.unique(seed)) != len(seed):
        raise ValueError("invalid parent marginal")
    if np.any(probability < 0.0) or not np.isclose(
        probability.sum(), 1.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("parent marginal is not normalized")
    if scheduled.ndim != 1 or scheduled_weight.shape != scheduled.shape \
            or not np.isclose(scheduled_weight.sum(), 1.0, atol=1.0e-12):
        raise ValueError("invalid scheduled posterior")
    rows = {int(row["seed"]): row for row in p1["members"]}
    if set(rows) != set(map(int, seed)):
        raise ValueError("P1 and promoted parent seed sets differ")
    if not set(map(int, scheduled)).issubset(rows):
        raise ValueError("schedule contains a parent absent from P1")
    gate_names = list(next(iter(rows.values()))["gates"])

    posterior_gate_mass = {
        name: float(sum(
            weight for value, weight in zip(seed, probability)
            if rows[int(value)]["gates"][name]
        ))
        for name in gate_names
    }
    schedule_gate_mass = {
        name: float(sum(
            weight for value, weight in zip(scheduled, scheduled_weight)
            if rows[int(value)]["gates"][name]
        ))
        for name in gate_names
    }
    full_parent = {value for value, row in rows.items() if row["pass"]}
    posterior_full_mass = float(sum(
        weight for value, weight in zip(seed, probability)
        if int(value) in full_parent
    ))
    schedule_full_mask = np.asarray([
        int(value) in full_parent for value in scheduled
    ])
    schedule_full_count = int(np.count_nonzero(schedule_full_mask))
    schedule_full_mass = float(scheduled_weight[schedule_full_mask].sum())
    independent_zero_probability = float(
        (1.0 - posterior_full_mass) ** len(scheduled)
    )
    prefilter_supports_any_state = schedule_full_count > 0
    return {
        "parent_count": len(seed),
        "schedule_count": len(scheduled),
        "legacy_full_P1_parent_seeds": sorted(full_parent),
        "posterior_gate_mass": posterior_gate_mass,
        "schedule_gate_mass": schedule_gate_mass,
        "posterior_full_P1_mass": posterior_full_mass,
        "expected_full_P1_count_under_independent_256_draws": float(
            len(scheduled) * posterior_full_mass
        ),
        "independent_probability_of_zero_full_P1_rows": independent_zero_probability,
        "schedule_full_P1_count": schedule_full_count,
        "schedule_full_P1_mass": schedule_full_mass,
        "decision": {
            "legacy_parent_centered_P1_prefilter_has_support": (
                prefilter_supports_any_state
            ),
            "legacy_parent_centered_P1_prefilter_authorized": False,
            "forward_all_scheduled_states_before_pair_recentered_P1": True,
            "pair_recentered_P1_final_environment_gate_retained": True,
            "lowk_posterior_promotion_revoked": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-sha256", required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--schedule-sha256", required=True)
    parser.add_argument("--p1-result", type=Path, required=True)
    parser.add_argument("--p1-result-sha256", required=True)
    args = parser.parse_args()
    for label, path, expected in (
        ("bank", args.bank, args.bank_sha256),
        ("schedule", args.schedule, args.schedule_sha256),
        ("P1 result", args.p1_result, args.p1_result_sha256),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"{label} SHA256 changed")
    with np.load(args.bank, allow_pickle=False) as bank:
        parent_seed = np.asarray(bank["parent_seed"])
        parent_probability = np.asarray(bank["P_parent"])
    with np.load(args.schedule, allow_pickle=False) as schedule:
        schedule_parent_seed = np.asarray(schedule["parent_seed"])
        schedule_weight = np.asarray(schedule["posterior_weight"])
    result = audit_support(
        parent_seed,
        parent_probability,
        schedule_parent_seed,
        schedule_weight,
        json.loads(args.p1_result.read_text()),
    )
    print(json.dumps({
        "status": "complete_P1_support_audit",
        **result,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
