#!/usr/bin/env python3
"""PM+FoF screen of parent fields at the canonical Virgo and Coma targets.

Only parents that passed the inexpensive linear top-hat screen are forwarded.
FoF is run on buffered local particle subsets, avoiding a full-box tree while
retaining the frozen global linking length.  This remains a PM-resolution
screen; RAMSES promotion is not authorized by this script.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fof import fof  # noqa: E402
from mock_pipeline import RHO_CRIT, make_forward  # noqa: E402


def sg_xyz(sgl_deg: float, sgb_deg: float, distance_mpc_h: float) -> np.ndarray:
    lon, lat = np.radians([sgl_deg, sgb_deg])
    return distance_mpc_h * np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def min_image(delta: np.ndarray, box: float) -> np.ndarray:
    return (delta + box / 2.0) % box - box / 2.0


def local_fof(
    positions: np.ndarray,
    target: np.ndarray,
    search_radius: float,
    buffer: float,
    box: float,
    spacing: float,
    particle_mass: float,
) -> dict:
    displacement = min_image(positions - target, box)
    select = np.all(np.abs(displacement) <= search_radius + buffer, axis=1)
    local = displacement[select]
    catalog = fof(
        local,
        L=box,
        mean_sep=spacing,
        b=0.2,
        n_min=20,
        m_particle=particle_mass,
        periodic=False,
        verbose=False,
    )
    if catalog["mass"].size:
        centres = (catalog["pos"] + target) % box
        distance = np.linalg.norm(min_image(centres - target, box), axis=1)
        inside = np.flatnonzero(distance <= search_radius)
    else:
        centres = np.empty((0, 3))
        distance = np.empty(0)
        inside = np.empty(0, dtype=np.int64)
    order = inside[np.argsort(catalog["mass"][inside])[::-1]] if inside.size else inside
    return {
        "n_local_particles": int(local.shape[0]),
        "n_fof_halos": int(catalog["mass"].size),
        "n_halos_in_search": int(inside.size),
        "maximum_mass_msun_h": float(catalog["mass"][order[0]]) if order.size else None,
        "top_halos": [
            {
                "mass_msun_h": float(catalog["mass"][i]),
                "npart": int(catalog["n"][i]),
                "position_mpc_h": centres[i].tolist(),
                "target_separation_mpc_h": float(distance[i]),
            }
            for i in order[:10]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--physical-scan", type=Path, required=True)
    parser.add_argument("--p1-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--buffer-mpc-h", type=float, default=10.0)
    parser.add_argument(
        "--all-members",
        action="store_true",
        help="scan every manifest member instead of the linear top-hat shortlist",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    physical = json.loads(args.physical_scan.read_text())
    target_config = json.loads(args.p1_config.read_text())
    seeds = (
        [int(row["seed"]) for row in physical["rows"]]
        if args.all_members
        else [int(seed) for seed in physical["both_physical_cluster_gate_seeds"]]
    )
    sources = {}
    for path_text in manifest["outputs"]:
        path = Path(path_text)
        with np.load(path, allow_pickle=False) as data:
            sources[int(data["sample_seed"])] = path
    missing = sorted(set(seeds) - set(sources))
    if missing:
        raise RuntimeError(f"candidate seeds missing from manifest: {missing}")

    model = manifest["configuration"]
    n = int(model["N"])
    box = float(model["box_size"])
    spacing = box / n
    cosmology = {
        "Om": model["Om"], "Ob": model["Ob"], "h": model["h"],
        "A_s_1e9": model["A_s_1e9"], "ns": model["ns"],
    }
    particle_mass = float(model["Om"]) * RHO_CRIT * spacing**3
    _, _, forward = make_forward(
        n,
        spacing,
        jnp.float32,
        return_dens=False,
        cosmology=cosmology,
        return_particle_positions=True,
    )
    observer = np.full(3, box / 2.0)
    targets = {}
    for name in ("Virgo", "Coma"):
        spec = target_config["clusters"][name]
        targets[name] = {
            "position": (
                observer
                + sg_xyz(
                    spec["sgl_deg"],
                    spec["sgb_deg"],
                    spec["distance_mpc"] * target_config["cosmology_h"],
                )
            ) % box,
            "search_radius": float(spec["search_radius_mpc_h"]),
        }

    rows = []
    started = time.time()
    for number, seed in enumerate(seeds, 1):
        with np.load(sources[seed], allow_pickle=False) as data:
            white = np.asarray(data["s_out"], dtype=np.float32)
        positions = np.asarray(forward(jnp.asarray(white)), dtype=np.float32)
        anchors = {
            name: local_fof(
                positions,
                spec["position"],
                spec["search_radius"],
                args.buffer_mpc_h,
                box,
                spacing,
                particle_mass,
            )
            for name, spec in targets.items()
        }
        virgo_pass = bool(
            anchors["Virgo"]["maximum_mass_msun_h"] is not None
            and anchors["Virgo"]["maximum_mass_msun_h"] >= 1.0e14
        )
        coma_pass = bool(
            anchors["Coma"]["maximum_mass_msun_h"] is not None
            and anchors["Coma"]["maximum_mass_msun_h"] >= 5.0e14
        )
        rows.append(
            {
                "seed": seed,
                "legacy_p1_pass": bool(seed in physical["legacy_p1_and_both_physical_cluster_gate_seeds"]),
                "anchors": anchors,
                "virgo_pass": virgo_pass,
                "coma_pass": coma_pass,
                "both_pass": bool(virgo_pass and coma_pass),
            }
        )
        print(
            f"[pm-fof] {number}/{len(seeds)} seed={seed} "
            f"Virgo={anchors['Virgo']['maximum_mass_msun_h']} "
            f"Coma={anchors['Coma']['maximum_mass_msun_h']}",
            flush=True,
        )
        del positions

    both = [row["seed"] for row in rows if row["both_pass"]]
    legacy_and_both = [row["seed"] for row in rows if row["legacy_p1_pass"] and row["both_pass"]]
    result = {
        "schema": "ouruniv-cf4-parent-cluster-pm-fof-scan-v1",
        "status": "complete_screen_no_promotion",
        "manifest": str(args.manifest.resolve()),
        "physical_scan": str(args.physical_scan.resolve()),
        "p1_config": str(args.p1_config.resolve()),
        "grid": {"N": n, "box_mpc_h": box, "spacing_mpc_h": spacing},
        "fof": {
            "b": 0.2,
            "n_min": 20,
            "particle_mass_msun_h": particle_mass,
            "local_buffer_mpc_h": args.buffer_mpc_h,
            "periodic_full_box_positions_local_nonperiodic_buffered_tree": True,
        },
        "thresholds": {"Virgo_msun_h": 1.0e14, "Coma_msun_h": 5.0e14},
        "candidate_seeds": seeds,
        "candidate_mode": "all_manifest_members" if args.all_members else "linear_top_hat_shortlist",
        "both_pass_seeds": both,
        "legacy_p1_and_both_pass_seeds": legacy_and_both,
        "rows": rows,
        "seconds": time.time() - started,
        "decision": "EXISTING_PARENT_AVAILABLE" if legacy_and_both else "NEW_JOINT_PARENT_CONDITIONING_REQUIRED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"both": both, "legacy_and_both": legacy_and_both, "decision": result["decision"]}, sort_keys=True))


if __name__ == "__main__":
    main()
