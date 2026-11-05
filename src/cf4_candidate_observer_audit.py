#!/usr/bin/env python3
"""Post-hoc audit of near-miss LG pairs as possible observer locations.

The preregistered P2 screen requires the pair midpoint to lie within 5 Mpc/h
of the box centre.  This diagnostic does not alter that result.  It finds
otherwise valid pairs just outside that radius, treats each pair midpoint as a
hypothetical observer, and repeats every P1 cosmography/environment gate about
that location.  A survivor can motivate a separately preregistered
candidate-centred stage; it is not retroactively counted as a P2 pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def periodic_offset(position: np.ndarray, centre: np.ndarray, box: float) -> np.ndarray:
    offset = np.asarray(position, np.float64) - np.asarray(centre, np.float64)
    return (offset + box / 2.0) % box - box / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-result", type=Path, required=True)
    parser.add_argument("--p2-config", type=Path, required=True)
    parser.add_argument("--p1-result", type=Path, required=True)
    parser.add_argument("--p1-config", type=Path, required=True)
    parser.add_argument("--audit-max-offset-mpc-h", type=float, default=6.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from scipy.ndimage import gaussian_filter
    from cf4_p2_screen import find_pairs, load_config, rank_score
    from cf4_parent_p1 import file_hash, score_member

    p2_result = json.loads(args.p2_result.read_text())
    p2_config = load_config(args.p2_config)
    p1_result = json.loads(args.p1_result.read_text())
    p1_config = json.loads(args.p1_config.read_text())
    if p2_result.get("status") != "complete":
        parser.error("P2 result must be complete")
    if p2_result.get("config_sha256") != file_hash(args.p2_config):
        parser.error("P2 result/config hash mismatch")

    parent_seeds = sorted({int(row["parent_seed"]) for row in p2_result["results"]})
    if len(parent_seeds) != 1:
        parser.error("this diagnostic currently requires exactly one P1 parent")
    parent_seed = parent_seeds[0]
    parent_rows = [
        row for row in p1_result["members"]
        if row["pass"] and int(row["seed"]) == parent_seed
    ]
    if len(parent_rows) != 1:
        parser.error("cannot resolve one passing parent field")
    parent_path = Path(parent_rows[0]["input"])
    with np.load(parent_path, allow_pickle=False) as data:
        initial = data["s_out"].astype(np.float32)
        nmesh = int(data["N"])
        spacing = float(data["spacing"])
        box = float(data["L"])
        cosmology = {
            "Om": float(data["Om"]), "Ob": float(data["Ob"]),
            "h": float(data["hh"]), "A_s_1e9": float(data["A_s_1e9"]),
            "ns": float(data["ns"]),
        }

    import jax.numpy as jnp
    from mock_pipeline import make_forward

    _, _, forward = make_forward(
        nmesh, spacing, jnp.float32, return_dens=True, cosmology=cosmology)
    density, _ = forward(jnp.asarray(initial))
    density.block_until_ready()
    smoothed = gaussian_filter(
        np.asarray(density, np.float32),
        p1_config["density_smoothing_mpc_h"] / spacing,
        mode="wrap",
    )
    delta = smoothed / np.mean(smoothed, dtype=np.float64) - 1.0

    screen = dict(p2_config["screen"])
    screen["pair_midpoint_max_offset_mpc_h"] = args.audit_max_offset_mpc_h
    centre = np.full(3, box / 2.0)
    rows = []
    halo_dir = args.p2_result.parent
    for result_row in p2_result["results"]:
        parent = int(result_row["parent_seed"])
        small = int(result_row["small_scale_seed"])
        halo_path = halo_dir / f"halos_p{parent}_s{small}.npz"
        with np.load(halo_path, allow_pickle=False) as data:
            halos = {
                "pos": data["halo_pos"], "vel": data["halo_vel"],
                "mass": data["halo_mass"],
            }
        pairs = find_pairs(
            halos, centre, screen, p2_config["m33_subpeak_gate"])
        if not pairs:
            continue
        for pair in pairs:
            pair["ranking_score"] = rank_score(pair, p2_config["ranking"])
        pair = min(pairs, key=lambda row: row["ranking_score"])
        observer_offset = periodic_offset(pair["midpoint_mpc_h"], centre, box)
        p1_metrics = score_member(
            delta, spacing, p1_config, omega_m=cosmology["Om"],
            observer_offset=observer_offset)
        halo_distance = np.linalg.norm(
            halos["pos"].astype(np.float64) - np.asarray(pair["midpoint_mpc_h"]),
            axis=1,
        )
        massive_near = (
            (halos["mass"] >= screen["isolation_mass_threshold_msun_h"])
            & (halo_distance <= 8.0)
        )
        rows.append({
            "parent_seed": parent,
            "small_scale_seed": small,
            "observer_offset_mpc_h": observer_offset.tolist(),
            "observer_offset_norm_mpc_h": float(np.linalg.norm(observer_offset)),
            "pair": pair,
            "p1_recentered": p1_metrics,
            "n_halos_ge_5e12_within_8_mpc_h": int(massive_near.sum()),
            "candidate_centred_pass": bool(
                p1_metrics["pass"] and not np.any(massive_near)),
        })
        print(
            f"[observer-audit] s{small}: r={np.linalg.norm(observer_offset):.2f} "
            f"P1={p1_metrics['n_gates_passed']}/"
            f"{len(p1_metrics['gates'])} massive8={massive_near.sum()} "
            f"=> {'SURVIVE' if rows[-1]['candidate_centred_pass'] else 'fail'}",
            flush=True,
        )

    result = {
        "schema": "ouruniv-candidate-observer-audit-v1",
        "status": "post_hoc_diagnostic",
        "interpretation": (
            "Does not change the preregistered P2 result; tests whether a new, "
            "separately frozen candidate-centred stage is scientifically warranted."
        ),
        "audit_max_offset_mpc_h": args.audit_max_offset_mpc_h,
        "p2_result": str(args.p2_result.resolve()),
        "p2_result_sha256": file_hash(args.p2_result),
        "p2_config": str(args.p2_config.resolve()),
        "p2_config_sha256": file_hash(args.p2_config),
        "p1_result": str(args.p1_result.resolve()),
        "p1_config": str(args.p1_config.resolve()),
        "parent_field": str(parent_path.resolve()),
        "parent_field_sha256": file_hash(parent_path),
        "rows": rows,
        "surviving_combinations": [
            [row["parent_seed"], row["small_scale_seed"]]
            for row in rows if row["candidate_centred_pass"]
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[observer-audit] survivors={result['surviving_combinations']}")
    print(f"[observer-audit] wrote {args.out}")


if __name__ == "__main__":
    main()
