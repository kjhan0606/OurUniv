#!/usr/bin/env python3
"""Cheap preregistered P1 recheck at each promoted P2 screen midpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cf4_parent_p1 import file_hash, score_member  # noqa: E402
from cf4_zoom_z0_gate import min_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-result", type=Path, default=Path(
        "/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_lg_peak_p2_v3_inverse/"
        "p2_screen_result.json"))
    parser.add_argument("--conditioned-p1-result", type=Path, default=Path(
        "/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_lg_peak_p1_v3_inverse/"
        "p1_result.json"))
    parser.add_argument("--p1-config", type=Path,
                        default=ROOT / "config/p1_targets_v2_observer.json")
    parser.add_argument("--halo-directory", type=Path, default=Path(
        "/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_lg_peak_p2_v3_inverse"))
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=None,
        help="optional proposal seeds (default: every P2 screen survivor)")
    parser.add_argument("--out", type=Path, default=Path(
        "/gpfs/kjhan/CF4/recon/linear_cr/v3_bgc_lg_peak_p2_v3_inverse/"
        "recentered_p1_preview.json"))
    args = parser.parse_args()

    p2 = json.loads(args.p2_result.read_text())
    p1 = json.loads(args.conditioned_p1_result.read_text())
    config = json.loads(args.p1_config.read_text())
    passing_rows = [row for row in p2["results"] if row["screen_pass"]]
    seeds = (
        [int(value) for value in args.seeds]
        if args.seeds is not None
        else [int(row["small_scale_seed"]) for row in passing_rows]
    )
    candidates = {
        int(row["small_scale_seed"]): row for row in passing_rows
        if int(row["small_scale_seed"]) in seeds
    }
    p1_rows = {int(row["seed"]): row for row in p1["members"]}
    if set(candidates) != set(seeds):
        parser.error("not every requested seed is a passing P2 candidate")

    if not seeds:
        result = {
            "schema": "ouruniv-p2-recentered-p1-preview-v2",
            "status": "no_p2_screen_survivors",
            "p2_result": str(args.p2_result.resolve()),
            "p2_result_sha256": file_hash(args.p2_result),
            "conditioned_p1_result": str(args.conditioned_p1_result.resolve()),
            "conditioned_p1_result_sha256": file_hash(args.conditioned_p1_result),
            "p1_config": str(args.p1_config.resolve()),
            "p1_config_sha256": file_hash(args.p1_config),
            "rows": [],
            "passing_combinations": [],
            "passing_seeds": [],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"[done] no P2 survivors; wrote {args.out}", flush=True)
        return

    import jax.numpy as jnp
    from scipy.ndimage import gaussian_filter
    from mock_pipeline import make_forward

    first_path = Path(p1_rows[seeds[0]]["input"])
    with np.load(first_path, allow_pickle=False) as data:
        nmesh = int(data["N"])
        spacing = float(data["spacing"])
        box = float(data["L"])
        cosmology = {
            "Om": float(data["Om"]), "Ob": float(data["Ob"]),
            "h": float(data["hh"]), "A_s_1e9": float(data["A_s_1e9"]),
            "ns": float(data["ns"]),
        }
    _, _, forward = make_forward(
        nmesh, spacing, jnp.float32, return_dens=True, cosmology=cosmology)
    rows = []
    for seed in seeds:
        started = time.time()
        input_path = Path(p1_rows[seed]["input"])
        with np.load(input_path, allow_pickle=False) as data:
            initial = data["s_out"].astype(np.float32)
        density, _ = forward(jnp.asarray(initial))
        density.block_until_ready()
        smoothed = gaussian_filter(
            np.asarray(density, np.float32),
            config["density_smoothing_mpc_h"] / spacing,
            mode="wrap")
        delta = smoothed / np.mean(smoothed, dtype=np.float64) - 1.0
        parent_seed = int(candidates[seed]["parent_seed"])
        pairs = candidates[seed].get("screen_pairs")
        if pairs is None:
            pairs = [candidates[seed]["best_pair"]]
        halo_path = args.halo_directory / f"halos_p{parent_seed}_s{seed}.npz"
        with np.load(halo_path, allow_pickle=False) as data:
            halo_mass = np.asarray(data["halo_mass"], np.float64)
            halo_pos = np.asarray(data["halo_pos"], np.float64)
        pair_rows = []
        for pair_index, pair in enumerate(pairs):
            midpoint = np.asarray(pair["midpoint_mpc_h"], np.float64)
            observer_offset = min_image(midpoint - box / 2.0, box)
            metrics = score_member(
                delta, spacing, config, omega_m=cosmology["Om"],
                observer_offset=observer_offset)
            distance = np.linalg.norm(min_image(halo_pos - midpoint, box), axis=1)
            massive = np.flatnonzero((halo_mass >= 5e12) & (distance <= 8.0))
            massive_rows = [{
                "halo_index": int(k),
                "mass_fof_msun_h": float(halo_mass[k]),
                "distance_mpc_h": float(distance[k]),
            } for k in massive[np.argsort(-halo_mass[massive])]]
            pair_rows.append({
                "pair_index": pair_index,
                "screen_midpoint_mpc_h": midpoint.tolist(),
                "observer_offset_mpc_h": observer_offset.tolist(),
                "screen_pair": pair,
                "p1_recentered": metrics,
                "massive_screen_halos_within_8_mpc_h": massive_rows,
                "preview_pass": bool(metrics["pass"] and not massive_rows),
            })
        pair_rows.sort(key=lambda row: (
            not row["preview_pass"],
            -row["p1_recentered"]["n_gates_passed"],
            row["screen_pair"]["ranking_score"],
        ))
        best = pair_rows[0]
        rows.append({
            "parent_seed": parent_seed,
            "small_scale_seed": seed,
            "n_pairs_checked": len(pair_rows),
            "pair_rows": pair_rows,
            "best_recentered_pair": best,
            # Compatibility aliases for the original one-pair preview.
            **{key: best[key] for key in (
                "screen_midpoint_mpc_h", "observer_offset_mpc_h", "screen_pair",
                "p1_recentered", "massive_screen_halos_within_8_mpc_h",
                "preview_pass")},
            "seconds": time.time() - started,
        })
        print(f"[preview] s{seed}: pairs={len(pair_rows)} best P1="
              f"{best['p1_recentered']['n_gates_passed']}/"
              f"{len(best['p1_recentered']['gates'])} massive8="
              f"{len(best['massive_screen_halos_within_8_mpc_h'])} "
              f"pass={best['preview_pass']}", flush=True)
    result = {
        "schema": "ouruniv-p2-recentered-p1-preview-v2",
        "status": "screen_midpoint_preview_not_final_ramses_midpoint",
        "p2_result": str(args.p2_result.resolve()),
        "p2_result_sha256": file_hash(args.p2_result),
        "conditioned_p1_result": str(args.conditioned_p1_result.resolve()),
        "conditioned_p1_result_sha256": file_hash(args.conditioned_p1_result),
        "p1_config": str(args.p1_config.resolve()),
        "p1_config_sha256": file_hash(args.p1_config),
        "rows": rows,
        "passing_combinations": [
            [row["parent_seed"], row["small_scale_seed"]]
            for row in rows if row["preview_pass"]
        ],
        "passing_seeds": [row["small_scale_seed"] for row in rows
                          if row["preview_pass"]],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"[done] {args.out}", flush=True)


if __name__ == "__main__":
    main()
