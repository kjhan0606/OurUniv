#!/usr/bin/env python
"""Select and confirm V8 using only the two declared development domains."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from hong2021_v6_gate import HIGH_K_BANDS, field_gate


def candidate_metrics(path: Path, label: str = "edm") -> dict[str, Any]:
    report = json.loads(path.read_text())
    return report["candidates"][label]


def calibration_score(metrics: dict[str, Any]) -> float:
    power = metrics["fourier_log_density"]["generated_total_power_over_truth"]
    ratios = [power[band] for band in HIGH_K_BANDS]
    ratios.append(metrics["residual_calibration"]["generated_over_truth_rms"])
    if any(not math.isfinite(value) or value <= 0 for value in ratios):
        return float("inf")
    return max(abs(math.log(value)) for value in ratios)


def select(args: argparse.Namespace) -> None:
    candidates = []
    for step in args.steps:
        root = args.root / f"step_{step:06d}"
        tng_path = root / "tng/ensemble_evaluation/metrics.json"
        simba_path = root / "simba_dev/ensemble_evaluation/metrics.json"
        tng = candidate_metrics(tng_path)
        simba = candidate_metrics(simba_path)
        tng_score = calibration_score(tng)
        simba_score = calibration_score(simba)
        candidates.append(
            {
                "step": step,
                "checkpoint": str(
                    (args.training / "validation_checkpoints" / f"step_{step:06d}.pt").resolve()
                ),
                "tng_metrics": str(tng_path.resolve()),
                "simba_development_metrics": str(simba_path.resolve()),
                "tng_calibration_score": tng_score,
                "simba_development_calibration_score": simba_score,
                "selection_score": max(tng_score, simba_score),
                "screening_field_gate_diagnostic": {
                    "tng": field_gate(tng),
                    "simba_development": field_gate(simba),
                },
            }
        )
    chosen = min(candidates, key=lambda row: (row["selection_score"], row["step"]))
    report = {
        "schema": "hong2021-v8-development-checkpoint-selection-v1",
        "historically_inspected_simba_cv0_15_used": False,
        "eagle_used": False,
        "predeclared_steps": args.steps,
        "selection_rule": (
            "minimize the worst domain maximum absolute log deviation from "
            "one over high-k 3-6, high-k 6-10.0531, and residual RMS ratios; "
            "tie break by smaller step"
        ),
        "candidates": candidates,
        "selected_step": chosen["step"],
        "selected_checkpoint": chosen["checkpoint"],
        "selected_score": chosen["selection_score"],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def confirm(args: argparse.Namespace) -> None:
    tng = candidate_metrics(args.tng_metrics)
    simba = candidate_metrics(args.simba_metrics)
    gates = {"tng": field_gate(tng), "simba_development": field_gate(simba)}
    report = {
        "schema": "hong2021-v8-development-confirmation-v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "historically_inspected_simba_cv0_15_used": False,
        "eagle_used": False,
        "metrics": {
            "tng": str(args.tng_metrics.resolve()),
            "simba_development": str(args.simba_metrics.resolve()),
        },
        "field_gates": gates,
        "both_field_gates_pass": all(value["pass"] for value in gates.values()),
        "next": (
            "run_historical_SIMBA_stress_without_model_selection"
            if all(value["pass"] for value in gates.values())
            else "stop_without_opening_sealed_EAGLE_confirmation"
        ),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    selection = sub.add_parser("select")
    selection.add_argument("--root", type=Path, required=True)
    selection.add_argument("--training", type=Path, required=True)
    selection.add_argument(
        "--steps", type=int, nargs="+", default=[500, 2000, 5000, 10000]
    )
    selection.add_argument("--out", type=Path, required=True)
    confirmation = sub.add_parser("confirm")
    confirmation.add_argument("--checkpoint", type=Path, required=True)
    confirmation.add_argument("--tng-metrics", type=Path, required=True)
    confirmation.add_argument("--simba-metrics", type=Path, required=True)
    confirmation.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    {"select": select, "confirm": confirm}[args.mode](args)


if __name__ == "__main__":
    main()
