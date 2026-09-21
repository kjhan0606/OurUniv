#!/usr/bin/env python3
"""Estimate the Lagrangian preimage of the observer-centred local volume."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mock_pipeline import make_forward  # noqa: E402


def min_image(delta: np.ndarray, box: float) -> np.ndarray:
    return (delta + box / 2.0) % box - box / 2.0


def periodic_mean(points: np.ndarray, box: float) -> np.ndarray:
    angles = 2.0 * np.pi * points / box
    mean_angle = np.arctan2(np.sin(angles).mean(0), np.cos(angles).mean(0))
    return np.mod(mean_angle, 2.0 * np.pi) * box / (2.0 * np.pi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radii", type=float, nargs="+", default=[3.0, 5.0, 8.0])
    args = parser.parse_args()

    with np.load(args.candidate, allow_pickle=False) as data:
        white = np.asarray(data["s_out"], dtype=np.float32)
        n = int(data["N"])
        spacing = float(data["spacing"])
        box = float(data["L"])
        cosmology = {
            "Om": float(data["Om"]), "Ob": float(data["Ob"]),
            "h": float(data["hh"]), "A_s_1e9": float(data["A_s_1e9"]),
            "ns": float(data["ns"]),
        }
    _, _, forward = make_forward(
        n, spacing, jnp.float32, return_dens=False, cosmology=cosmology
    )
    particles = forward(jnp.asarray(white))
    final = np.asarray(particles.pos(), dtype=np.float32)
    q = np.asarray(particles.pmid, dtype=np.float32) * spacing
    centre = np.full(3, box / 2.0)

    rows = []
    for radius in args.radii:
        select = np.linalg.norm(min_image(final - centre, box), axis=1) <= radius
        selected_q = q[select]
        mean_q = periodic_mean(selected_q, box)
        offset = min_image(mean_q - centre, box)
        spread = np.sqrt(np.mean(min_image(selected_q - mean_q, box) ** 2, axis=0))
        rows.append({
            "eulerian_radius_mpc_h": radius,
            "particle_count": int(select.sum()),
            "lagrangian_centre_mpc_h": mean_q.tolist(),
            "lagrangian_offset_from_observer_mpc_h": offset.tolist(),
            "lagrangian_axis_rms_mpc_h": spread.tolist(),
        })
        print(f"[preimage] R={radius:g} N={select.sum()} offset={offset}", flush=True)

    result = {
        "schema": "ouruniv-cf4-parent-observer-preimage-v1",
        "status": "complete_pm_proxy",
        "candidate": str(args.candidate.resolve()),
        "grid": {"N": n, "spacing_mpc_h": spacing, "box_mpc_h": box},
        "definition": "Periodic centroid of initial particle-grid coordinates whose z=0 PM positions lie inside each observer-centred sphere.",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
