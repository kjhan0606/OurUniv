#!/usr/bin/env python
"""Apply the unchanged eight field checks to all three V14 development codes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hong2021_v6_gate import field_gate


DOMAINS = ("tng", "simba_dev", "swift_eagle_dev")


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())["candidates"]["edm"]


def select_candidate_rows(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [row for row in rows if row["all_three_pass"]]
    return min(passing, key=lambda row: row["step"]) if passing else None


def evaluate(root: Path, training: Path, steps: list[int]) -> dict[str, Any]:
    candidates = []
    for step in steps:
        step_root = root / f"step_{step:06d}"
        domains = {}
        for domain in DOMAINS:
            metrics_path = step_root / domain / "ensemble_evaluation" / "metrics.json"
            gate = field_gate(load_metrics(metrics_path))
            domains[domain] = {
                "metrics": str(metrics_path.resolve()),
                "field_gate": gate,
            }
        candidates.append(
            {
                "step": step,
                "checkpoint": str(
                    (
                        training
                        / "validation_checkpoints"
                        / f"step_{step:06d}.pt"
                    ).resolve()
                ),
                "domains": domains,
                "all_three_pass": all(
                    value["field_gate"]["pass"] for value in domains.values()
                ),
            }
        )
    selected = select_candidate_rows(candidates)
    return {
        "schema": "hong2021-v14-three-domain-full-field-development-selection-v1",
        "EAGLE_RefL0100N1504_used": False,
        "Astrid_used": False,
        "historical_SIMBA_CV0_15_used": False,
        "predeclared_steps": steps,
        "selection_rule": (
            "smallest predeclared step passing all eight unchanged field "
            "checks independently in TNG100, SIMBA development, and "
            "Swift-EAGLE development"
        ),
        "candidates": candidates,
        "selected_step": None if selected is None else selected["step"],
        "selected_checkpoint": None if selected is None else selected["checkpoint"],
        "development_pass": selected is not None,
        "next": (
            "freeze_exact_v14_hashes_and_astrid_one_shot_command"
            if selected is not None
            else "stop_without_downloading_or_opening_astrid"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[2000, 5000, 10000])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.steps != sorted(set(args.steps)):
        raise ValueError("development candidate steps must be unique and ordered")
    report = evaluate(args.root, args.training, args.steps)
    if args.out.exists():
        raise RuntimeError(f"refusing to overwrite development decision: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
