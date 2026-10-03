#!/usr/bin/env python3
"""Scan the frozen CF4 parent ensemble for collapse-capable Virgo/Coma peaks.

The legacy P1 cluster gates use shell percentiles.  This diagnostic adds the
missing physical question: does the linearly extrapolated density field contain
a top-hat overdensity large enough to collapse on a specified cluster mass
scale?  It does not select or promote a new parent.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from pmwd import Configuration, SimpleLCDM, boltzmann, linear_modes


RHO_CRIT = 2.775e11  # (Msun/h) / (Mpc/h)^3
DELTA_COLLAPSE = 1.686


def sg_xyz(sgl_deg: float, sgb_deg: float, distance_mpc_h: float) -> np.ndarray:
    lon, lat = np.radians([sgl_deg, sgb_deg])
    return distance_mpc_h * np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def periodic_delta(a: np.ndarray, b: np.ndarray, box: float) -> np.ndarray:
    return (a - b + box / 2.0) % box - box / 2.0


def grid_centres_near(target: np.ndarray, radius: float, n: int, spacing: float) -> np.ndarray:
    centre = np.floor(target / spacing).astype(np.int64)
    reach = int(np.ceil(radius / spacing)) + 1
    offsets = np.stack(
        np.meshgrid(
            np.arange(-reach, reach + 1),
            np.arange(-reach, reach + 1),
            np.arange(-reach, reach + 1),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    indices = (centre[None, :] + offsets) % n
    positions = (indices.astype(np.float64) + 0.5) * spacing
    keep = np.linalg.norm(periodic_delta(positions, target, n * spacing), axis=1) <= radius
    return indices[keep]


def sphere_offsets(radius: float, spacing: float) -> np.ndarray:
    reach = int(np.ceil(radius / spacing))
    offsets = np.stack(
        np.meshgrid(
            np.arange(-reach, reach + 1),
            np.arange(-reach, reach + 1),
            np.arange(-reach, reach + 1),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    distance = np.linalg.norm(offsets.astype(np.float64) * spacing, axis=1)
    return offsets[distance <= radius]


def maximum_sphere_mean(field: np.ndarray, centres: np.ndarray, offsets: np.ndarray) -> tuple[float, np.ndarray]:
    n = field.shape[0]
    total = np.zeros(centres.shape[0], dtype=np.float64)
    for offset in offsets:
        index = (centres + offset[None, :]) % n
        total += field[index[:, 0], index[:, 1], index[:, 2]]
    mean = total / offsets.shape[0]
    best = int(np.argmax(mean))
    return float(mean[best]), centres[best]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--p1-result", type=Path, required=True)
    parser.add_argument("--p1-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    p1 = json.loads(args.p1_result.read_text())
    target_config = json.loads(args.p1_config.read_text())
    model = manifest["configuration"]
    n = int(model["N"])
    box = float(model["box_size"])
    spacing = box / n
    observer = np.full(3, box / 2.0)

    conf = Configuration(
        ptcl_spacing=spacing,
        ptcl_grid_shape=(n,) * 3,
        mesh_shape=1,
        float_dtype=jnp.float32,
    )
    cosmo = boltzmann(
        SimpleLCDM(
            conf,
            Omega_m=float(model["Om"]),
            Omega_b=float(model["Ob"]),
            h=float(model["h"]),
            A_s_1e9=float(model["A_s_1e9"]),
            n_s=float(model["ns"]),
        ),
        conf,
    )
    to_linear = jax.jit(lambda s: linear_modes(s, cosmo, conf, a=1.0, real=True))

    mass_scales = [1.0e14, 3.0e14, 5.0e14]
    radius_by_mass = {
        mass: (3.0 * mass / (4.0 * np.pi * float(model["Om"]) * RHO_CRIT)) ** (1.0 / 3.0)
        for mass in mass_scales
    }
    offsets_by_mass = {
        mass: sphere_offsets(radius, spacing) for mass, radius in radius_by_mass.items()
    }
    anchor_geometry = {}
    for name in ("Virgo", "Coma"):
        spec = target_config["clusters"][name]
        target = observer + sg_xyz(
            spec["sgl_deg"],
            spec["sgb_deg"],
            spec["distance_mpc"] * target_config["cosmology_h"],
        )
        anchor_geometry[name] = {
            "target": target % box,
            "centres": grid_centres_near(target % box, spec["search_radius_mpc_h"], n, spacing),
            "search_radius_mpc_h": float(spec["search_radius_mpc_h"]),
        }

    p1_rows = {int(row["seed"]): row for row in p1["members"]}
    rows = []
    started = time.time()
    for index, source in enumerate(manifest["outputs"], 1):
        with np.load(source, allow_pickle=False) as data:
            seed = int(data["sample_seed"])
            white = np.asarray(data["s_out"], dtype=np.float32)
        linear = np.array(to_linear(jnp.asarray(white)), dtype=np.float32, copy=True)
        linear -= np.mean(linear, dtype=np.float64)
        anchors = {}
        for name, geometry in anchor_geometry.items():
            scales = {}
            for mass in mass_scales:
                maximum, centre_index = maximum_sphere_mean(
                    linear, geometry["centres"], offsets_by_mass[mass]
                )
                centre_position = (centre_index.astype(np.float64) + 0.5) * spacing
                scales[f"{mass:.1e}"] = {
                    "lagrangian_radius_mpc_h": radius_by_mass[mass],
                    "maximum_delta_linear": maximum,
                    "collapse_capable": bool(maximum >= DELTA_COLLAPSE),
                    "peak_centre_mpc_h": centre_position.tolist(),
                    "peak_target_separation_mpc_h": float(
                        np.linalg.norm(periodic_delta(centre_position, geometry["target"], box))
                    ),
                    "n_sphere_cells": int(offsets_by_mass[mass].shape[0]),
                }
            anchors[name] = scales
        rows.append(
            {
                "seed": seed,
                "legacy_p1_pass": bool(p1_rows[seed]["pass"]),
                "anchors": anchors,
                "virgo_1e14_collapse": anchors["Virgo"]["1.0e+14"]["collapse_capable"],
                "coma_5e14_collapse": anchors["Coma"]["5.0e+14"]["collapse_capable"],
            }
        )
        if index == 1 or index % 16 == 0 or index == len(manifest["outputs"]):
            print(f"[physical-gate] {index}/{len(manifest['outputs'])} seed={seed}", flush=True)

    both = [row["seed"] for row in rows if row["virgo_1e14_collapse"] and row["coma_5e14_collapse"]]
    legacy_and_both = [row["seed"] for row in rows if row["legacy_p1_pass"] and row["virgo_1e14_collapse"] and row["coma_5e14_collapse"]]
    seed3429 = next(row for row in rows if row["seed"] == 3429)
    result = {
        "schema": "ouruniv-cf4-parent-physical-cluster-gate-scan-v1",
        "status": "complete_diagnostic_no_promotion",
        "manifest": str(args.manifest.resolve()),
        "p1_result": str(args.p1_result.resolve()),
        "p1_config": str(args.p1_config.resolve()),
        "cosmology": {key: model[key] for key in ("Om", "Ob", "h", "A_s_1e9", "ns")},
        "grid": {"N": n, "box_mpc_h": box, "spacing_mpc_h": spacing},
        "criterion": {
            "delta_c": DELTA_COLLAPSE,
            "virgo_minimum_mass_msun_h": 1.0e14,
            "coma_minimum_mass_msun_h": 5.0e14,
            "meaning": "linearly extrapolated spherical top-hat collapse screen, not a halo-mass prediction",
        },
        "counts": {
            "members": len(rows),
            "virgo_1e14_collapse": sum(row["virgo_1e14_collapse"] for row in rows),
            "coma_5e14_collapse": sum(row["coma_5e14_collapse"] for row in rows),
            "both_physical_cluster_gates": len(both),
            "legacy_p1_and_both_physical_cluster_gates": len(legacy_and_both),
        },
        "both_physical_cluster_gate_seeds": both,
        "legacy_p1_and_both_physical_cluster_gate_seeds": legacy_and_both,
        "seed3429": seed3429,
        "rows": rows,
        "seconds": time.time() - started,
        "decision": "EXISTING_PARENT_AVAILABLE" if legacy_and_both else "NEW_JOINT_PARENT_CONDITIONING_REQUIRED",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("counts", "decision", "seconds")}, sort_keys=True))


if __name__ == "__main__":
    main()
