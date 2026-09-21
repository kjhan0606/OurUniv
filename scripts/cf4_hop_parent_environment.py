#!/usr/bin/env python3
"""Evaluate the frozen Local-Volume environment gate with RAMSES HOP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

RHO_CRIT = 2.775e11


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def min_image(vector: np.ndarray, box: float) -> np.ndarray:
    return vector - box * np.rint(vector / box)


def distance(position: np.ndarray, target: np.ndarray, box: float) -> np.ndarray:
    return np.linalg.norm(min_image(position - target, box), axis=-1)


def target_position(entry: dict, observer: np.ndarray, h: float, box: float) -> np.ndarray:
    longitude = math.radians(float(entry["sgl_deg"]))
    latitude = math.radians(float(entry["sgb_deg"]))
    radius = float(entry["distance_mpc"]) * h
    offset = radius * np.asarray(
        [math.cos(latitude) * math.cos(longitude),
         math.cos(latitude) * math.sin(longitude),
         math.sin(latitude)]
    )
    return (observer + offset) % box


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hop", type=Path, required=True)
    parser.add_argument("--galaxyfinder", type=Path, required=True)
    parser.add_argument("--p1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box", type=float, default=384.0)
    parser.add_argument("--omega-m", type=float, default=0.31)
    parser.add_argument("--h", type=float, default=0.746)
    args = parser.parse_args()

    table = np.loadtxt(args.hop, comments="#", ndmin=2)
    if table.shape[1] < 7 or not np.all(np.isfinite(table)):
        raise RuntimeError("invalid HOP position catalogue")
    box = float(args.box)
    observer = np.full(3, box / 2.0)
    total_mass = float(args.omega_m) * RHO_CRIT * box**3
    position = table[:, 4:7] * box
    mass = table[:, 2] * total_mass
    npart = table[:, 1].astype(np.int64)

    galaxyfinder = json.loads(args.galaxyfinder.read_text())
    midpoint = np.asarray(galaxyfinder["lg"]["selected"]["midpoint_mpc_h"], dtype=np.float64)
    massive = mass >= 5.0e12
    observer_distance = distance(position[massive], observer, box) if np.any(massive) else np.asarray([99.0])
    midpoint_distance = distance(position[massive], midpoint, box) if np.any(massive) else np.asarray([99.0])
    environment_pass = bool(observer_distance.min() >= 8.0 and midpoint_distance.min() >= 8.0)

    nearby = []
    if np.any(massive):
        massive_index = np.flatnonzero(massive)
        order = np.argsort(observer_distance)
        for local in order:
            index = int(massive_index[local])
            if observer_distance[local] > 12.0:
                break
            nearby.append({
                "hop_row": index,
                "group_id": int(table[index, 0]),
                "npart": int(npart[index]),
                "mass_msun_h": float(mass[index]),
                "position_mpc_h": position[index].tolist(),
                "observer_distance_mpc_h": float(observer_distance[local]),
                "midpoint_distance_mpc_h": float(distance(position[index][None, :], midpoint, box)[0]),
            })

    p1 = json.loads(args.p1.read_text())
    clusters = {}
    for name, entry in p1["clusters"].items():
        target = target_position(entry, observer, float(args.h), box)
        separation = distance(position, target, box)
        inside = np.flatnonzero(separation <= float(entry["search_radius_mpc_h"]))
        if inside.size:
            index = int(inside[np.argmax(mass[inside])])
            clusters[name] = {
                "found": True,
                "group_id": int(table[index, 0]),
                "npart": int(npart[index]),
                "mass_msun_h": float(mass[index]),
                "position_mpc_h": position[index].tolist(),
                "target_separation_mpc_h": float(separation[index]),
            }
        else:
            clusters[name] = {"found": False}

    result = {
        "schema": "ouruniv-cf4-parent-ramses-hop-environment-v1",
        "hop_catalog": str(args.hop),
        "hop_catalog_sha256": sha256(args.hop),
        "box_mpc_h": box,
        "omega_m": float(args.omega_m),
        "total_box_mass_msun_h": total_mass,
        "group_count": int(table.shape[0]),
        "clusters": clusters,
        "environment": {
            "mass_threshold_msun_h": 5.0e12,
            "radius_mpc_h": 8.0,
            "nearest_to_observer_mpc_h": float(observer_distance.min()),
            "nearest_to_lg_midpoint_mpc_h": float(midpoint_distance.min()),
            "nearby_groups": nearby,
            "pass": environment_pass,
        },
        "decision": "HOP_LOCAL_ENVIRONMENT_PASS" if environment_pass else "HOP_LOCAL_ENVIRONMENT_FAIL",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "groups": int(table.shape[0]), "nearby": len(nearby)}))


if __name__ == "__main__":
    main()
