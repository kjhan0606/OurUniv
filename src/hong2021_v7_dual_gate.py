#!/usr/bin/env python
"""Require the frozen field (and optional grid-HOP) gate in both domains."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from hong2021_v6_gate import field_gate, paired_hop_bootstrap


def load_candidate(path: Path, method: str) -> dict[str, Any]:
    report = json.loads(path.read_text())
    try:
        return report["candidates"][method]
    except KeyError as error:
        raise KeyError(f"candidate {method!r} is absent from {path}") from error


def domain_result(
    metrics_path: Path,
    hop_path: Path | None,
    method: str,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    field = field_gate(load_candidate(metrics_path, method))
    hop = None
    if hop_path is not None:
        hop = paired_hop_bootstrap(
            json.loads(hop_path.read_text()), method, bootstrap, seed
        )
    return {
        "metrics": str(metrics_path.resolve()),
        "hop_report": None if hop_path is None else str(hop_path.resolve()),
        "field_gate": field,
        "grid_hop_gate": hop,
        "overall_pass": None if hop is None else field["pass"] and hop["pass"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tng-metrics", type=Path, required=True)
    parser.add_argument("--simba-metrics", type=Path, required=True)
    parser.add_argument("--tng-hop", type=Path)
    parser.add_argument("--simba-hop", type=Path)
    parser.add_argument("--method", default="edm")
    parser.add_argument("--bootstrap", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.tng_hop is None) != (args.simba_hop is None):
        raise ValueError("supply both --tng-hop and --simba-hop, or neither")

    domains = {
        "TNG100_representative_validation": domain_result(
            args.tng_metrics, args.tng_hop, args.method,
            args.bootstrap, args.seed,
        ),
        "locked_CAMELS_SIMBA_CV_0_15": domain_result(
            args.simba_metrics, args.simba_hop, args.method,
            args.bootstrap, args.seed + 1,
        ),
    }
    both_field = all(value["field_gate"]["pass"] for value in domains.values())
    hop_was_run = args.tng_hop is not None
    both_hop = (
        all(value["grid_hop_gate"]["pass"] for value in domains.values())
        if hop_was_run else None
    )
    report = {
        "schema": "hong2021-v7-tng-simba-dual-gate-v1",
        "method": args.method,
        "locked_test_policy": (
            "SIMBA CV 0-15 was not used for training, normalization, "
            "checkpoint selection, or hyperparameter selection"
        ),
        "domains": domains,
        "both_field_gates_pass": both_field,
        "grid_hop_run": hop_was_run,
        "both_grid_hop_gates_pass": both_hop,
        "advance_to_forward_dynamics": bool(both_field and both_hop),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
