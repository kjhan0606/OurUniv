#!/usr/bin/env python
"""Apply the unchanged full field gate to predeclared V9 checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hong2021_v6_gate import field_gate


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text())["candidates"]["edm"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[1000, 3000, 5000])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--report-schema",
        default="hong2021-v9-full-fidelity-development-selection-v1",
    )
    args = parser.parse_args()
    candidates = []
    for step in args.steps:
        root = args.root / f"step_{step:06d}"
        tng_path = root / "tng/ensemble_evaluation/metrics.json"
        simba_path = root / "simba_dev/ensemble_evaluation/metrics.json"
        tng_gate = field_gate(load_metrics(tng_path))
        simba_gate = field_gate(load_metrics(simba_path))
        candidates.append(
            {
                "step": step,
                "checkpoint": str(
                    (
                        args.training
                        / "validation_checkpoints"
                        / f"step_{step:06d}.pt"
                    ).resolve()
                ),
                "tng_metrics": str(tng_path.resolve()),
                "simba_development_metrics": str(simba_path.resolve()),
                "tng_field_gate": tng_gate,
                "simba_development_field_gate": simba_gate,
                "both_pass": tng_gate["pass"] and simba_gate["pass"],
            }
        )
    passing = [row for row in candidates if row["both_pass"]]
    selected = min(passing, key=lambda row: row["step"]) if passing else None
    report = {
        "schema": args.report_schema,
        "historical_simba_cv0_15_used": False,
        "sealed_eagle_used": False,
        "predeclared_steps": args.steps,
        "selection_rule": (
            "smallest predeclared step passing all eight unchanged field "
            "checks in both full-fidelity development domains"
        ),
        "candidates": candidates,
        "selected_step": None if selected is None else selected["step"],
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "development_pass": selected is not None,
        "next": (
            "run_historical_SIMBA_stress_without_model_selection"
            if selected is not None
            else "stop_without_opening_sealed_EAGLE_confirmation"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
