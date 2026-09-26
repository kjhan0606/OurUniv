#!/usr/bin/env python3
"""Query the existing p3429/s5108 z=0 HOP catalogue at frozen P1 anchors.

This is a read-only lineage diagnostic.  It distinguishes failure of the
CF4-constrained parent environment from failure caused by moving the observer
from the frozen box-centre coordinate to a selected halo-pair midpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def sg_xyz(sgl_deg: float, sgb_deg: float, distance_mpc_h: float) -> np.ndarray:
    lon, lat = np.radians([sgl_deg, sgb_deg])
    return distance_mpc_h * np.array(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )


def periodic_displacement(pos: np.ndarray, target: np.ndarray, box: float) -> np.ndarray:
    return (pos - target + box / 2.0) % box - box / 2.0


def query(pos: np.ndarray, mass: np.ndarray, target: np.ndarray, radius: float, box: float) -> dict:
    distance = np.linalg.norm(periodic_displacement(pos, target, box), axis=1)
    inside = np.flatnonzero(distance <= radius)
    if inside.size:
        order = inside[np.argsort(mass[inside])[::-1]]
        rows = [
            {
                "catalog_index": int(i),
                "mass_msun_h": float(mass[i]),
                "position_mpc_h": pos[i].tolist(),
                "separation_mpc_h": float(distance[i]),
            }
            for i in order[:10]
        ]
    else:
        rows = []
    nearest = int(np.argmin(distance))
    return {
        "search_radius_mpc_h": radius,
        "n_halos": int(inside.size),
        "maximum_mass_msun_h": float(np.max(mass[inside])) if inside.size else None,
        "top_halos": rows,
        "nearest_halo": {
            "catalog_index": nearest,
            "mass_msun_h": float(mass[nearest]),
            "position_mpc_h": pos[nearest].tolist(),
            "separation_mpc_h": float(distance[nearest]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--p1-config", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.p1_config.read_text())
    gate = json.loads(args.gate.read_text())
    with np.load(args.catalog, allow_pickle=False) as data:
        pos = np.asarray(data["pos"], dtype=np.float64)
        mass = np.asarray(data["mass"], dtype=np.float64)

    box = float(gate["metadata"]["box_mpc_h"])
    canonical_observer = np.full(3, box / 2.0)
    midpoint_observer = np.asarray(gate["pair"]["midpoint_mpc_h"], dtype=np.float64)
    observers = {
        "canonical_box_centre": canonical_observer,
        "selected_pair_midpoint": midpoint_observer,
    }
    anchors = {}
    for name, spec in config["clusters"].items():
        offset = sg_xyz(
            spec["sgl_deg"], spec["sgb_deg"], spec["distance_mpc"] * config["cosmology_h"]
        )
        anchors[name] = {}
        for label, observer in observers.items():
            target = (observer + offset) % box
            base_radius = float(spec["search_radius_mpc_h"])
            anchors[name][label] = {
                "observer_mpc_h": observer.tolist(),
                "target_mpc_h": target.tolist(),
                "searches": {
                    f"{factor:g}x": query(pos, mass, target, factor * base_radius, box)
                    for factor in (1.0, 2.0, 4.0)
                },
            }

    result = {
        "schema": "ouruniv-cf4-parent3429-existing-z0-environment-query-v2",
        "scope": "read-only HOP catalogue query; no new IC or forward evolution",
        "catalog": str(args.catalog.resolve()),
        "n_catalog_halos": int(mass.size),
        "box_mpc_h": box,
        "observer_shift_mpc_h": float(np.linalg.norm(midpoint_observer - canonical_observer)),
        "anchors": anchors,
        "interpretation_contract": {
            "cluster_scale_threshold_msun_h": 1.0e14,
            "canonical_virgo_present": bool(
                anchors["Virgo"]["canonical_box_centre"]["searches"]["1x"]["maximum_mass_msun_h"] is not None
                and anchors["Virgo"]["canonical_box_centre"]["searches"]["1x"]["maximum_mass_msun_h"] >= 1.0e14
            ),
            "midpoint_recentring_is_not_a_parent_failure": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
