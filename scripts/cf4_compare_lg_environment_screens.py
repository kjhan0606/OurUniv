#!/usr/bin/env python3
"""Attribute strict LG-search failures to pair formation or environment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-result", type=Path, required=True)
    parser.add_argument("--pair-result", type=Path, required=True)
    parser.add_argument("--pair-catalogue-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mass-threshold", type=float, default=5.0e12)
    parser.add_argument("--radius", type=float, default=8.0)
    args = parser.parse_args()

    strict = json.loads(args.strict_result.read_text())
    pair = json.loads(args.pair_result.read_text())
    strict_rows = strict.get("rows", [])
    pair_rows = pair.get("rows", [])
    if len(strict_rows) != 256 or len(pair_rows) != 256:
        raise RuntimeError("both searches must contain exactly 256 rows")

    compared = []
    for strict_row, pair_row in zip(strict_rows, pair_rows):
        identity = ("index", "field_seed", "geometry_seed", "likelihood_noise_seed", "midpoint_seed", "field_sha256")
        if any(strict_row[key] != pair_row[key] for key in identity):
            raise RuntimeError("strict and pair-only row identities differ")
        if strict_row["screen_pass"]:
            raise RuntimeError("strict search unexpectedly contains a passing row")
        if not pair_row["screen_pass"]:
            continue

        index = int(pair_row["index"])
        catalogue = args.pair_catalogue_dir / f"passing_halos_{index:03d}.npz"
        with np.load(catalogue, allow_pickle=False) as data:
            pos = np.asarray(data["halo_pos"], dtype=np.float64)
            mass = np.asarray(data["halo_mass"], dtype=np.float64)
        centre = np.full(3, float(pair["grid"]["box_mpc_h"]) / 2.0)
        best = pair_row["best_pair"]
        midpoint = np.asarray(best["midpoint_mpc_h"], dtype=np.float64)
        massive = np.flatnonzero(mass >= args.mass_threshold)
        observer_distance = np.linalg.norm(pos[massive] - centre, axis=1)
        midpoint_distance = np.linalg.norm(pos[massive] - midpoint, axis=1)
        observer_offenders = massive[observer_distance < args.radius]
        midpoint_offenders = massive[midpoint_distance < args.radius]
        compared.append({
            "index": index,
            "field_seed": int(pair_row["field_seed"]),
            "ranking_score": float(best["ranking_score"]),
            "pair": best,
            "observer_offenders": [
                {
                    "halo_index": int(i),
                    "mass_msun_h": float(mass[i]),
                    "distance_mpc_h": float(np.linalg.norm(pos[i] - centre)),
                    "position_mpc_h": pos[i].tolist(),
                }
                for i in observer_offenders
            ],
            "midpoint_offenders": [
                {
                    "halo_index": int(i),
                    "mass_msun_h": float(mass[i]),
                    "distance_mpc_h": float(np.linalg.norm(pos[i] - midpoint)),
                    "position_mpc_h": pos[i].tolist(),
                }
                for i in midpoint_offenders
                if int(i) not in (int(best["halo_i"]), int(best["halo_j"]))
            ],
        })

    n_with_environment_offender = sum(
        bool(row["observer_offenders"] or row["midpoint_offenders"])
        for row in compared
    )
    if not compared:
        decision = "PAIR_FORMATION_IS_LIMITING"
    elif n_with_environment_offender == len(compared):
        decision = "LOCAL_VOLUME_ENVIRONMENT_IS_LIMITING"
    else:
        decision = "SCREEN_DISAGREEMENT_REQUIRES_DEBUG"
    result = {
        "schema": "ouruniv-cf4-lg-environment-screen-comparison-v1",
        "strict_result": str(args.strict_result.resolve()),
        "strict_result_sha256": sha256_file(args.strict_result),
        "pair_result": str(args.pair_result.resolve()),
        "pair_result_sha256": sha256_file(args.pair_result),
        "mass_threshold_msun_h": args.mass_threshold,
        "radius_mpc_h": args.radius,
        "n_proposals": 256,
        "n_pair_screen_pass": len(compared),
        "n_pair_pass_with_environment_offender": n_with_environment_offender,
        "rows": compared,
        "decision": decision,
    }
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "n_pair_screen_pass": len(compared),
        "n_with_environment_offender": n_with_environment_offender,
        "decision": decision,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
