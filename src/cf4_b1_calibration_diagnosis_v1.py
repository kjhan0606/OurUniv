"""Development-only diagnosis of the B1 integrated calibration gate.

The runner replays only the sealed development members and decomposes the
pre-registered strict gate into named failure reasons.  It does not tune the
model, inspect replacement validation seeds, or produce an observational
posterior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

import cf4_b1_integrated_joint_calibration_v3 as integrated


TARGET_68 = 0.6826894921370859
TARGET_95 = 0.9544997361036416


def gate_failures(metrics: dict[str, object]) -> list[str]:
    """Return every failed component of the frozen strict member gate."""

    failures: list[str] = []
    if not 0.8 <= float(metrics["response"]) <= 1.2:
        failures.append("response")
    if float(metrics["correlation_r"]) < 0.7:
        failures.append("correlation")
    if float(metrics["residual_power_ratio"]) > 0.5:
        failures.append("residual_power")
    if abs(float(metrics["coverage68"]) - TARGET_68) > 0.05:
        failures.append("coverage68")
    if abs(float(metrics["coverage95"]) - TARGET_95) > 0.025:
        failures.append("coverage95")
    if float(metrics["heldout_log_score_improvement"]) <= 0.0:
        failures.append("heldout_gain")
    if not bool(metrics["fit_success"]):
        failures.append("optimizer_convergence")
    if float(metrics["joint_log_likelihood_abs_error"]) >= 1.0e-7:
        failures.append("joint_factor_consistency")
    return failures


def run_diagnosis() -> dict[str, object]:
    arms = ("A", "B", "C", "D")
    members = []
    for index in range(integrated.MOCK_COUNT):
        arm = arms[index // 16]
        row = integrated.run_mock(index, arm)
        failures = gate_failures(row["metrics"])
        members.append({"index": index, "arm": arm, "seed": row["seed"],
                        "metrics": row["metrics"], "gate_failures": failures,
                        "strict_low_k_gate": not failures})

    by_arm: dict[str, dict[str, object]] = {}
    numeric = ("response", "correlation_r", "residual_power_ratio", "coverage68",
               "coverage95", "heldout_log_score_improvement", "joint_log_likelihood_abs_error")
    for arm in arms:
        rows = [row for row in members if row["arm"] == arm]
        by_arm[arm] = {
            "member_count": len(rows),
            "strict_gate_pass_count": sum(bool(row["strict_low_k_gate"]) for row in rows),
            "failure_count_by_component": {
                name: sum(name in row["gate_failures"] for row in rows)
                for name in ("response", "correlation", "residual_power", "coverage68", "coverage95",
                             "heldout_gain", "optimizer_convergence", "joint_factor_consistency")
            },
            "metric_median": {name: float(np.median([row["metrics"][name] for row in rows])) for name in numeric},
            "metric_min": {name: float(np.min([row["metrics"][name] for row in rows])) for name in numeric},
            "metric_max": {name: float(np.max([row["metrics"][name] for row in rows])) for name in numeric},
        }

    failure_histogram: dict[str, int] = {}
    for row in members:
        for name in row["gate_failures"]:
            failure_histogram[name] = failure_histogram.get(name, 0) + 1
    return {
        "schema": "ouruniv-cf4-b1-calibration-diagnosis-result-v1",
        "status": "COMPLETE_DEVELOPMENT_ONLY_NO_SCIENCE_CLAIM",
        "source_result": "config/cf4_bundle_b1_integrated_joint_development_calibration_result_v3.json",
        "seed_firewall": {
            "development_start_inclusive": integrated.SEED_START,
            "development_stop_exclusive": integrated.SEED_STOP,
            "validation_opened": False,
            "contaminated_quarantine": [2026083064, 2026083128],
            "replacement_validation_sealed": [2026083320, 2026083576],
        },
        "frozen_gate": {
            "required_member_pass_count": integrated.MOCK_COUNT,
            "actual_member_pass_count": sum(bool(row["strict_low_k_gate"]) for row in members),
            "pass": all(bool(row["strict_low_k_gate"]) for row in members),
            "component_definitions": ["response", "correlation", "residual_power", "coverage68", "coverage95",
                                       "heldout_gain", "optimizer_convergence", "joint_factor_consistency"],
        },
        "failure_histogram": failure_histogram,
        "by_arm": by_arm,
        "members": members,
        "interpretation": {
            "primary_failure": "coverage calibration and member-level variance, not response/correlation or factor consistency",
            "stress_arm_note": "D is an overdispersed model-discrepancy stress arm; its coverage failure is diagnostic, not a reason to discard the arm",
            "model_change_performed": False,
            "observational_posterior": "NOT_CREATED",
            "validation_opened": False,
            "B2_IC_FORWARD": "NOT_STARTED",
        },
        "next_action": "Use development-only diagnosis to revise the declared uncertainty/stress-arm calibration, then rerun the same 64 seeds before any validation opening or promotion.",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = run_diagnosis()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pass_count": result["frozen_gate"]["actual_member_pass_count"],
                      "failure_histogram": result["failure_histogram"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
