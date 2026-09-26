#!/usr/bin/env python3
"""High-k stability screen for the recovered parent candidate.

The default reproduces the eight-seed N256 P1 plus cluster screen.  At the
L9/N512 parent resolution, ``--cluster-only`` avoids materialising a full
density mesh and exports only final particle positions for the two local FoF
queries.  No full field or particle catalogue is persisted in either mode.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cf4_make_ic import (  # noqa: E402
    embed_ic,
    embed_ic_projected,
    fourier_resample_white_field,
)
from cf4_parent_cluster_pm_fof_scan import local_fof, sg_xyz  # noqa: E402
from cf4_parent_p1 import score_member  # noqa: E402
from mock_pipeline import RHO_CRIT, make_forward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--field-key", default="s_out")
    parser.add_argument("--p1-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(2001, 2009)))
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument(
        "--canonical-N",
        type=int,
        default=None,
        help=(
            "define the random realization on this mesh and Fourier-project to N; "
            "use this to recheck the exact phase family of an L9 candidate at N256"
        ),
    )
    parser.add_argument(
        "--cluster-only",
        action="store_true",
        help="skip P1 density scoring and export only final particle positions",
    )
    args = parser.parse_args()

    config = json.loads(args.p1_config.read_text())
    with np.load(args.candidate, allow_pickle=False) as data:
        coarse = np.asarray(data[args.field_key], dtype=np.float32)
        box = float(data["L"])
        cosmology = {
            "Om": float(data["Om"]), "Ob": float(data["Ob"]),
            "h": float(data["hh"]), "A_s_1e9": float(data["A_s_1e9"]),
            "ns": float(data["ns"]),
        }
    source_n = int(coarse.shape[0])
    effective_canonical_n = (
        source_n
        if source_n >= args.N
        else args.N if args.canonical_N is None else args.canonical_N
    )
    spacing = box / args.N
    particle_mass = cosmology["Om"] * RHO_CRIT * spacing**3
    if args.cluster_only:
        _, _, forward = make_forward(
            args.N,
            spacing,
            jnp.float32,
            return_dens=False,
            cosmology=cosmology,
            return_particle_positions=True,
        )
    else:
        _, _, forward = make_forward(
            args.N, spacing, jnp.float32, return_dens=True, cosmology=cosmology
        )
    observer = np.full(3, box / 2.0)
    targets = {}
    for name in ("Virgo", "Coma"):
        spec = config["clusters"][name]
        targets[name] = {
            "position": (observer + sg_xyz(
                spec["sgl_deg"], spec["sgb_deg"],
                spec["distance_mpc"] * config["cosmology_h"],
            )) % box,
            "search_radius": float(spec["search_radius_mpc_h"]),
        }

    rows = []
    started = time.time()
    for number, seed in enumerate(args.seeds, 1):
        if coarse.shape[0] == args.N:
            white = coarse.copy()
        elif coarse.shape[0] > args.N:
            white = fourier_resample_white_field(coarse, args.N)
        elif args.canonical_N is None:
            white = embed_ic(coarse, args.N, seed)
        else:
            white = embed_ic_projected(coarse, args.canonical_N, args.N, seed)
        if args.cluster_only:
            positions = np.asarray(forward(jnp.asarray(white)), dtype=np.float32)
            p1 = None
        else:
            density, particles = forward(jnp.asarray(white))
            density.block_until_ready()
            positions = np.asarray(particles.pos(), dtype=np.float32)
            smoothed = gaussian_filter(
                np.asarray(density, dtype=np.float32),
                config["density_smoothing_mpc_h"] / spacing,
                mode="wrap",
            )
            delta = smoothed / np.mean(smoothed, dtype=np.float64) - 1.0
            p1 = score_member(delta, spacing, config, omega_m=cosmology["Om"])
        anchors = {
            name: local_fof(
                positions, spec["position"], spec["search_radius"], 10.0,
                box, spacing, particle_mass,
            )
            for name, spec in targets.items()
        }
        mass_pass = bool(
            anchors["Virgo"]["maximum_mass_msun_h"] is not None
            and anchors["Virgo"]["maximum_mass_msun_h"] >= 1.0e14
            and anchors["Coma"]["maximum_mass_msun_h"] is not None
            and anchors["Coma"]["maximum_mass_msun_h"] >= 5.0e14
        )
        rows.append({
            "small_scale_seed": seed,
            "p1": p1,
            "anchors": anchors,
            "cluster_mass_pass": mass_pass,
            "pass": bool((p1 is None or p1["pass"]) and mass_pass),
        })
        print(
            f"[n{args.N}] {number}/{len(args.seeds)} seed={seed} "
            f"p1={None if p1 is None else p1['pass']} cluster_mass={mass_pass}",
            flush=True,
        )
        del white, positions
        if not args.cluster_only:
            del density, particles, smoothed, delta
        gc.collect()

    passing = [row["small_scale_seed"] for row in rows if row["pass"]]
    result = {
        "schema": "ouruniv-cf4-parent-candidate-n256-stability-v1",
        "status": "complete_no_field_persisted",
        "candidate": str(args.candidate.resolve()),
        "p1_config": str(args.p1_config.resolve()),
        "grid": {"N": args.N, "box_mpc_h": box, "spacing_mpc_h": spacing},
        "source_N": source_n,
        "canonical_N": effective_canonical_n,
        "mode": "cluster_only" if args.cluster_only else "p1_and_clusters",
        "particle_mass_msun_h": particle_mass,
        "small_scale_seeds": args.seeds,
        "rows": rows,
        "passing_seeds": passing,
        "pass_fraction": len(passing) / len(rows),
        "seconds": time.time() - started,
        "decision": (
            f"N{args.N}_STABLE"
            if len(passing) == len(rows)
            else f"N{args.N}_NOT_UNIFORMLY_STABLE"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passing": passing, "decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
