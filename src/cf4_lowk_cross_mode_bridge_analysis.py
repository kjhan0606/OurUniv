#!/usr/bin/env python3
"""Summarize the cross-mode bridge pilot without promoting a parent seed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


FROZEN_PARENT_L1_REFERENCE_Q999 = 0.3310546875


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def analyze_bridge_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("schema") != "ouruniv-cf4-lowk-cross-mode-bridge-pilot-v1" \
            or result.get("status") != "complete_diagnostic":
        raise ValueError("cross-mode bridge result is incomplete or has the wrong schema")
    checkpoints = result.get("checkpoints")
    groups = result.get("groups")
    if not isinstance(checkpoints, list) or len(checkpoints) != 3:
        raise ValueError("three bridge checkpoints are required")
    if not isinstance(groups, list) or len(groups) != 4:
        raise ValueError("four start groups are required")

    checkpoint_analysis = []
    previous_cycle = 0
    for row in checkpoints:
        cycle = int(row["bridge_cycle"])
        sweeps = int(row["matched_mh_sweeps"])
        if cycle <= previous_cycle or sweeps != 2 * cycle:
            raise ValueError("bridge checkpoints are not ordered or sweep matched")
        previous_cycle = cycle
        bridge_l1 = _finite_float(row["bridge_maximum_parent_L1"], "bridge L1")
        control_l1 = _finite_float(row["control_maximum_parent_L1"], "control L1")
        bridge_overlap = _finite_float(
            row["bridge_minimum_exact_overlap"], "bridge overlap"
        )
        control_overlap = _finite_float(
            row["control_minimum_exact_overlap"], "control overlap"
        )
        if min(bridge_l1, control_l1, bridge_overlap, control_overlap) < 0.0 \
                or max(bridge_overlap, control_overlap) > 1.0:
            raise ValueError("bridge diagnostic lies outside its physical range")
        checkpoint_analysis.append({
            "bridge_cycle": cycle,
            "matched_mh_sweeps": sweeps,
            "bridge_maximum_parent_L1": bridge_l1,
            "control_maximum_parent_L1": control_l1,
            "parent_L1_improvement_control_minus_bridge": control_l1 - bridge_l1,
            "bridge_minimum_exact_overlap": bridge_overlap,
            "control_minimum_exact_overlap": control_overlap,
            "exact_overlap_improvement_bridge_minus_control": (
                bridge_overlap - control_overlap
            ),
        })

    roundtrip = []
    swap = []
    for expected_group, row in enumerate(groups):
        if int(row["group"]) != expected_group:
            raise ValueError("bridge groups are not in canonical order")
        roundtrip.append(_finite_float(row["roundtrip_fraction"], "roundtrip fraction"))
        values = np.asarray(row["swap_acceptance_fraction"], dtype=np.float64)
        if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
            raise ValueError("swap acceptance vector is invalid")
        if np.min(values) < 0.0 or np.max(values) > 1.0:
            raise ValueError("swap acceptance lies outside [0,1]")
        swap.extend(float(value) for value in values)
    if min(roundtrip) < 0.0 or max(roundtrip) > 1.0:
        raise ValueError("roundtrip fraction lies outside [0,1]")

    terminal = checkpoint_analysis[-1]
    final_bridge_better_than_control = bool(
        terminal["parent_L1_improvement_control_minus_bridge"] > 0.0
        and terminal["exact_overlap_improvement_bridge_minus_control"] > 0.0
    )
    cross_mode_transport_observed = bool(max(roundtrip) > 0.0)
    frozen_parent_reference_pass = bool(
        terminal["bridge_maximum_parent_L1"]
        <= FROZEN_PARENT_L1_REFERENCE_Q999
    )
    return {
        "schema": "ouruniv-cf4-lowk-cross-mode-bridge-analysis-v1",
        "status": "complete_read_only_analysis",
        "checkpoints": checkpoint_analysis,
        "transport": {
            "roundtrip_fraction_by_group": roundtrip,
            "minimum_swap_acceptance_fraction": min(swap),
            "maximum_swap_acceptance_fraction": max(swap),
            "cross_mode_transport_observed": cross_mode_transport_observed,
        },
        "science_evidence": {
            "final_bridge_better_than_matched_beta_one_control": (
                final_bridge_better_than_control
            ),
            "frozen_parent_L1_reference_q999": FROZEN_PARENT_L1_REFERENCE_Q999,
            "frozen_parent_L1_reference_pass": frozen_parent_reference_pass,
            "bridge_mechanism_supported": bool(
                final_bridge_better_than_control and cross_mode_transport_observed
            ),
        },
        "decision": {
            "parent_posterior_promotion_authorized": False,
            "reason": (
                "This pilot supplies bridge-mechanism evidence; parent promotion "
                "still requires an independent science audit."
            ),
        },
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.result.open() as stream:
        result = json.load(stream)
    analysis = analyze_bridge_result(result)
    _atomic_json(args.output, analysis)
    print(json.dumps(analysis, sort_keys=True))


if __name__ == "__main__":
    main()
