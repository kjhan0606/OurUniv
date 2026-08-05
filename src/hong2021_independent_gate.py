#!/usr/bin/env python
"""Apply the frozen V6 gate to an independent-simulation ensemble."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hong2021_v6_gate import field_gate, paired_hop_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-metrics", type=Path, required=True)
    parser.add_argument("--hop", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--simulation", default="CAMELS-SIMBA-CV")
    parser.add_argument("--bootstrap", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()
    ensemble = json.loads(args.ensemble_metrics.read_text())
    hop: dict[str, Any] | None = None
    if args.hop is not None:
        hop = json.loads(args.hop.read_text())
    candidates: dict[str, Any] = {}
    for method, metrics in ensemble["candidates"].items():
        field = field_gate(metrics)
        hop_result = None
        if hop is not None:
            hop_result = paired_hop_bootstrap(
                hop, method, args.bootstrap, args.seed
            )
        candidates[method] = {
            "field_gate": field,
            "grid_hop_gate": hop_result,
            "overall_pass": (
                None if hop_result is None else field["pass"] and hop_result["pass"]
            ),
        }
    is_simba = args.simulation.startswith("CAMELS-SIMBA")
    report = {
        "schema": "hong2021-independent-simulation-gate-v1",
        "simulation": args.simulation,
        "provisional": is_simba,
        "ensemble_metrics": str(args.ensemble_metrics.resolve()),
        "hop_report": None if args.hop is None else str(args.hop.resolve()),
        "candidates": candidates,
        "accepted_methods": [
            method
            for method, result in candidates.items()
            if result["overall_pass"] is True
        ],
        "interpretation": (
            "SIMBA cross-code evidence does not replace the EAGLE "
            "particle-data gate or forward particle dynamics"
            if is_simba
            else "EAGLE is the paper-matched independent particle-data gate; "
            "passing still does not replace forward particle dynamics"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
