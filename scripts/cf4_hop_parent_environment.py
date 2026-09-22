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


def find_hop_lg_pairs(
    table: np.ndarray,
    position: np.ndarray,
    mass: np.ndarray,
    observer: np.ndarray,
    box: float,
    p2: dict,
) -> list[dict]:
    """Apply the frozen geometric P2 screen directly to HOP groups."""
    screen = p2["screen"]
    mass_lo, mass_hi = map(float, screen["pair_member_mass_range_msun_h"])
    midpoint_limit = float(screen["pair_midpoint_max_offset_mpc_h"])
    observer_distance = distance(position, observer, box)
    eligible = np.flatnonzero(
        (mass >= mass_lo)
        & (mass <= mass_hi)
        & (observer_distance <= midpoint_limit + 1.0)
    )
    massive = np.flatnonzero(mass >= float(screen["isolation_mass_threshold_msun_h"]))
    sep_lo, sep_hi = map(float, screen["pair_separation_range_mpc_h"])
    ratio_limit = float(screen["pair_mass_ratio_max"])
    isolation_limit = float(screen["isolation_radius_mpc_h"])
    target_mass = float(p2["ranking"]["target_member_mass_msun_h"])
    target_sep = float(p2["ranking"]["target_separation_mpc_h"])
    rows: list[dict] = []
    for local_a in range(eligible.size):
        i = int(eligible[local_a])
        for local_b in range(local_a + 1, eligible.size):
            j = int(eligible[local_b])
            vector = min_image(position[i] - position[j], box)
            separation = float(np.linalg.norm(vector))
            if separation < sep_lo or separation > sep_hi:
                continue
            ratio = float(max(mass[i], mass[j]) / min(mass[i], mass[j]))
            if ratio > ratio_limit:
                continue
            midpoint = (position[j] + 0.5 * vector) % box
            midpoint_offset = float(distance(midpoint[None, :], observer, box)[0])
            if midpoint_offset > midpoint_limit:
                continue
            external = np.asarray(
                [
                    distance(position[k][None, :], midpoint, box)[0]
                    for k in massive if int(k) not in (i, j)
                ],
                dtype=np.float64,
            )
            isolation = float(external.min()) if external.size else 99.0
            if isolation < isolation_limit:
                continue
            third_threshold = float(min(mass[i], mass[j]))
            third_candidates = np.flatnonzero(
                (mass >= third_threshold)
                & (distance(position, midpoint, box) <= 2.5)
            )
            third_candidates = np.asarray(
                [index for index in third_candidates if int(index) not in (i, j)],
                dtype=np.int64,
            )
            standard_isolation_pass = third_candidates.size == 0
            score = (
                abs(math.log(float(mass[i]) / target_mass))
                + abs(math.log(float(mass[j]) / target_mass))
                + abs(separation - target_sep) / 0.3
                + midpoint_offset / 2.0
            )
            rows.append(
                {
                    "hop_rows": [i, j],
                    "group_ids": [int(table[i, 0]), int(table[j, 0])],
                    "npart": [int(table[i, 1]), int(table[j, 1])],
                    "masses_msun_h": [float(mass[i]), float(mass[j])],
                    "positions_mpc_h": [position[i].tolist(), position[j].tolist()],
                    "mass_ratio": ratio,
                    "separation_mpc_h": separation,
                    "midpoint_mpc_h": midpoint.tolist(),
                    "midpoint_offset_mpc_h": midpoint_offset,
                    "p2_isolation_mpc_h": isolation,
                    "standard_isolation_pass": bool(standard_isolation_pass),
                    "standard_third_halo_rows": third_candidates.astype(int).tolist(),
                    "geometry_score": float(score),
                }
            )
    rows.sort(key=lambda item: (not item["standard_isolation_pass"], item["geometry_score"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hop", type=Path, required=True)
    parser.add_argument("--galaxyfinder", type=Path, required=True)
    parser.add_argument("--p1", type=Path, required=True)
    parser.add_argument("--p2", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box", type=float, default=384.0)
    parser.add_argument("--omega-m", type=float, default=0.31)
    parser.add_argument("--h", type=float, default=0.746)
    parser.add_argument(
        "--environment-policy",
        choices=("v12-blanket-veto", "standard-lg-isolation"),
        default="v12-blanket-veto",
    )
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
    selected = galaxyfinder["lg"].get("selected")
    independent_pairs: list[dict] = []
    if selected is None:
        if args.p2 is None:
            raise RuntimeError("--p2 is required when GalaxyFinder has no selected LG pair")
        p2 = json.loads(args.p2.read_text())
        independent_pairs = find_hop_lg_pairs(
            table, position, mass, observer, box, p2
        )
        hop_selected = independent_pairs[0] if independent_pairs else None
        midpoint = (
            np.asarray(hop_selected["midpoint_mpc_h"], dtype=np.float64)
            if hop_selected is not None else observer.copy()
        )
    else:
        hop_selected = None
        midpoint = np.asarray(selected["midpoint_mpc_h"], dtype=np.float64)
    massive = mass >= 5.0e12
    observer_distance = distance(position[massive], observer, box) if np.any(massive) else np.asarray([99.0])
    midpoint_distance = distance(position[massive], midpoint, box) if np.any(massive) else np.asarray([99.0])
    environment_pass = bool(observer_distance.min() >= 8.0 and midpoint_distance.min() >= 8.0)

    pair_hop_rows: list[int] = []
    pair_match_distance: list[float] = []
    if selected is not None:
        pair_positions = np.asarray(
            [selected["position_i_mpc_h"], selected["position_j_mpc_h"]],
            dtype=np.float64,
        )
        for pair_position in pair_positions:
            separation = distance(position, pair_position, box)
            index = int(np.argmin(separation))
            pair_hop_rows.append(index)
            pair_match_distance.append(float(separation[index]))
        third_threshold = min(
            float(selected["m1_fof_msun_h"]), float(selected["m2_fof_msun_h"])
        )
    elif hop_selected is not None:
        pair_hop_rows = [int(index) for index in hop_selected["hop_rows"]]
        pair_match_distance = [0.0, 0.0]
        third_threshold = min(float(value) for value in hop_selected["masses_msun_h"])
    else:
        third_threshold = math.inf
    third_distance = distance(position, midpoint, box)
    third_candidates = np.flatnonzero(
        (mass >= third_threshold) & (third_distance <= 2.5)
    )
    third_candidates = np.asarray(
        [index for index in third_candidates if int(index) not in pair_hop_rows],
        dtype=np.int64,
    )
    standard_isolation_pass = bool(
        len(set(pair_hop_rows)) == 2
        and max(pair_match_distance, default=math.inf) <= 1.0
        and third_candidates.size == 0
    )

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

    cluster_pass = bool(
        clusters["Virgo"].get("mass_msun_h", 0.0) >= 1.0e14
        and clusters["Coma"].get("mass_msun_h", 0.0) >= 5.0e14
    )
    selected_environment_pass = (
        environment_pass
        if args.environment_policy == "v12-blanket-veto"
        else standard_isolation_pass
    )
    third_halos = [
        {
            "hop_row": int(index),
            "group_id": int(table[index, 0]),
            "npart": int(npart[index]),
            "mass_msun_h": float(mass[index]),
            "position_mpc_h": position[index].tolist(),
            "midpoint_distance_mpc_h": float(third_distance[index]),
        }
        for index in third_candidates
    ]

    result = {
        "schema": "ouruniv-cf4-parent-ramses-hop-environment-v1",
        "hop_catalog": str(args.hop),
        "hop_catalog_sha256": sha256(args.hop),
        "box_mpc_h": box,
        "omega_m": float(args.omega_m),
        "total_box_mass_msun_h": total_mass,
        "group_count": int(table.shape[0]),
        "lg": {
            "source": (
                "galaxyfinder_pair_crossmatch"
                if selected is not None else "independent_hop_screen"
            ),
            "selected": selected if selected is not None else hop_selected,
            "independent_pair_count": len(independent_pairs),
            "independent_pairs": independent_pairs[:12],
        },
        "clusters": clusters,
        "environment": {
            "selection_policy": args.environment_policy,
            "mass_threshold_msun_h": 5.0e12,
            "radius_mpc_h": 8.0,
            "nearest_to_observer_mpc_h": float(observer_distance.min()),
            "nearest_to_lg_midpoint_mpc_h": float(midpoint_distance.min()),
            "nearby_groups": nearby,
            "pass": environment_pass,
            "v12_blanket_veto_is_diagnostic_only": (
                args.environment_policy == "standard-lg-isolation"
            ),
            "standard_lg_isolation": {
                "third_halo_mass_threshold_msun_h": third_threshold,
                "radius_mpc_h": 2.5,
                "pair_hop_rows": pair_hop_rows,
                "pair_match_distance_mpc_h": pair_match_distance,
                "third_halos": third_halos,
                "pass": standard_isolation_pass,
            },
        },
        "gates": {
            "virgo_coma_mass": cluster_pass,
            "selected_environment_policy": args.environment_policy,
            "selected_environment_pass": selected_environment_pass,
            "v12_blanket_veto_diagnostic": environment_pass,
            "standard_lg_isolation": standard_isolation_pass,
        },
        "decision": (
            "HOP_PARENT_STRUCTURE_PASS"
            if cluster_pass and selected_environment_pass
            else "HOP_PARENT_STRUCTURE_FAIL"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "groups": int(table.shape[0]), "nearby": len(nearby)}))


if __name__ == "__main__":
    main()
