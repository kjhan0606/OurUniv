#!/usr/bin/env python3
"""Estimate the Lagrangian point that flows to the z=0 box-centre observer.

The Local Group peak likelihood is imposed in initial (Lagrangian)
coordinates, whereas its required location is Eulerian at z=0.  This tool
forwards a small indexed cube of particles from a validated parent, smooths
their periodic displacement in Lagrangian coordinates, and finds the grid
point whose predicted final position is closest to the observer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def periodic_delta(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    delta = np.asarray(a) - np.asarray(b)
    return (delta + box / 2.0) % box - box / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--half-width-mpc-h", type=float, default=16.0)
    parser.add_argument("--smooth-radius-mpc-h", type=float, default=4.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.parent, allow_pickle=False) as data:
        initial = data["s_out"].astype(np.float32)
        n = int(data["N"])
        spacing = float(data["spacing"])
        box = float(data["L"])
        cosmology = {
            "Om": float(data["Om"]), "Ob": float(data["Ob"]),
            "h": float(data["hh"]), "A_s_1e9": float(data["A_s_1e9"]),
            "ns": float(data["ns"]),
        }
    half_cells = int(np.ceil(args.half_width_mpc_h / spacing))
    centre_index = n // 2
    axis = np.arange(centre_index - half_cells, centre_index + half_cells + 1)
    ii, jj, kk = np.meshgrid(axis, axis, axis, indexing="ij")
    grid_shape = ii.shape
    indices = np.ravel_multi_index(
        (ii.ravel() % n, jj.ravel() % n, kk.ravel() % n), (n, n, n))
    q = np.column_stack((ii.ravel(), jj.ravel(), kk.ravel())).astype(np.float32)
    q *= spacing

    import jax.numpy as jnp
    from scipy.ndimage import gaussian_filter
    from mock_pipeline import make_forward

    _, _, forward = make_forward(
        n, spacing, jnp.float32, return_dens=False, cosmology=cosmology,
        return_particle_arrays=True, particle_indices=jnp.asarray(indices))
    final_pos, _ = forward(jnp.asarray(initial))
    final_pos = np.asarray(final_pos, np.float32)
    displacement = periodic_delta(final_pos, q, box).reshape(grid_shape + (3,))
    sigma_cells = args.smooth_radius_mpc_h / spacing
    smooth = np.stack([
        gaussian_filter(displacement[..., component], sigma_cells, mode="nearest")
        for component in range(3)
    ], axis=-1)
    predicted = np.mod(q.reshape(grid_shape + (3,)) + smooth, box)
    observer = np.full(3, box / 2.0)
    residual = periodic_delta(predicted, observer, box)
    distance = np.linalg.norm(residual, axis=-1)
    best_local = np.unravel_index(int(np.argmin(distance)), grid_shape)
    q_best = q.reshape(grid_shape + (3,))[best_local]
    disp_best = smooth[best_local]
    predicted_best = predicted[best_local]
    centre_local = (half_cells, half_cells, half_cells)
    result = {
        "schema": "ouruniv-lagrangian-observer-origin-v1",
        "parent": str(args.parent.resolve()),
        "mesh_size": n,
        "spacing_mpc_h": spacing,
        "box_size_mpc_h": box,
        "search_half_width_mpc_h": args.half_width_mpc_h,
        "displacement_smoothing_radius_mpc_h": args.smooth_radius_mpc_h,
        "n_traced_particles": int(indices.size),
        "box_centre_lagrangian_displacement_mpc_h": smooth[centre_local].tolist(),
        "box_centre_predicted_eulerian_offset_mpc_h": periodic_delta(
            predicted[centre_local], observer, box).tolist(),
        "best_lagrangian_position_mpc_h": q_best.tolist(),
        "best_lagrangian_offset_mpc_h": periodic_delta(
            q_best, observer, box).tolist(),
        "best_smoothed_displacement_mpc_h": disp_best.tolist(),
        "best_predicted_eulerian_position_mpc_h": predicted_best.tolist(),
        "best_predicted_eulerian_offset_mpc_h": periodic_delta(
            predicted_best, observer, box).tolist(),
        "best_residual_mpc_h": float(distance[best_local]),
        "scope": "pre-forward coordinate mapping; no halo or P2 thresholds are fitted",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
