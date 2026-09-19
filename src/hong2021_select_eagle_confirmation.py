#!/usr/bin/env python
"""Predeclare a spatial EAGLE confirmation subset from the sealed reserve."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hong2021_prepare_eagle import farthest_point_subset


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seal = json.loads(args.seal.read_text())
    reserve = seal["reserved_confirmation"]
    indices = np.asarray(reserve["indices"], dtype=np.int64)
    identifiers = np.asarray(reserve["galaxy_ids"], dtype=np.int64)
    positions = np.asarray(reserve["positions_mpc_h"], dtype=np.float64)
    local = farthest_point_subset(positions, identifiers, args.count)
    selected = indices[local]
    report = {
        "schema": "hong2021-eagle-confirmation-selection-v1",
        "status": "sealed_unopened",
        "selection_uses_density_truth": False,
        "source_seal": str(args.seal.resolve()),
        "source_seal_sha256": sha256(args.seal),
        "algorithm": (
            "deterministic Euclidean farthest-point selection in observer "
            "position; smallest GalaxyID anchor"
        ),
        "opening_rule": (
            "Open exactly once only after the frozen TNG development and "
            "locked SIMBA field gates both pass"
        ),
        "indices": selected.tolist(),
        "galaxy_ids": identifiers[local].tolist(),
        "positions_mpc_h": positions[local].tolist(),
        "objects": int(len(selected)),
        "ensemble_per_object": 16,
        "sampling_steps": 40,
        "seed": 3777,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "indices": selected.tolist()}, indent=2))


if __name__ == "__main__":
    main()
