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
MERGED_PAIR_MASS_RELATIVE_TOLERANCE = 0.02


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


def classify_selected_pair_crossmatch(
    selected: dict,
    position: np.ndarray,
    mass: np.ndarray,
    npart: np.ndarray,
    box: float,
) -> dict:
    """Classify a GalaxyFinder pair against the regrouped HOP catalogue.

    Regrouped HOP can join a close MW/M31 analogue into one host group.  That
    outcome is useful for checking the combined parent mass and environment,
    but it is not evidence that HOP independently resolved both components.
    """
    pair_positions = np.asarray(
        [selected["position_i_mpc_h"], selected["position_j_mpc_h"]],
        dtype=np.float64,
    )
    pair_hop_rows: list[int] = []
    pair_match_distance: list[float] = []
    for pair_position in pair_positions:
        separation = distance(position, pair_position, box)
        index = int(np.argmin(separation))
        pair_hop_rows.append(index)
        pair_match_distance.append(float(separation[index]))

    distinct = len(set(pair_hop_rows)) == 2
    within_match_radius = max(pair_match_distance, default=math.inf) <= 1.0
    merged_row = pair_hop_rows[0] if pair_hop_rows and not distinct else None
    selected_npart = [int(value) for value in selected.get("npart", [])]
    selected_mass = (
        float(selected["m1_fof_msun_h"]) + float(selected["m2_fof_msun_h"])
    )
    if merged_row is None:
        hop_npart = None
        hop_mass = None
        particle_count_match = False
        mass_relative_error = None
        merged_consistency_pass = False
    else:
        hop_npart = int(npart[merged_row])
        hop_mass = float(mass[merged_row])
        particle_count_match = (
            len(selected_npart) == 2 and hop_npart == sum(selected_npart)
        )
        mass_relative_error = abs(hop_mass - selected_mass) / selected_mass
        merged_consistency_pass = bool(
            within_match_radius
            and particle_count_match
            and mass_relative_error <= MERGED_PAIR_MASS_RELATIVE_TOLERANCE
        )

    pair_supported = bool(within_match_radius and distinct)
    return {
        "mode": (
            "distinct_hop_groups"
            if distinct
            else (
                "merged_hop_group_mass_consistent"
                if merged_consistency_pass
                else "merged_hop_group_unverified"
            )
        ),
        "pair_hop_rows": pair_hop_rows,
        "pair_match_distance_mpc_h": pair_match_distance,
        "distinct_hop_groups": distinct,
        "within_match_radius": within_match_radius,
        "pair_supported_for_parent_trace": pair_supported,
        "hop_independently_resolves_pair": bool(distinct and within_match_radius),
        "merged_group": {
            "hop_row": merged_row,
            "hop_npart": hop_npart,
            "selected_pair_npart": selected_npart,
            "particle_count_match": particle_count_match,
            "hop_mass_msun_h": hop_mass,
            "selected_pair_mass_msun_h": selected_mass,
            "mass_relative_error": mass_relative_error,
            "mass_relative_tolerance": MERGED_PAIR_MASS_RELATIVE_TOLERANCE,
            "consistency_pass": merged_consistency_pass,
        },
    }


def crossmatch_raw_hop_peaks(
    path: Path,
    selected: dict,
    box: float,
) -> dict:
    """Match the selected pair to pre-regroup density peaks in a gbound file."""
    pair_positions = np.asarray(
        [selected["position_i_mpc_h"], selected["position_j_mpc_h"]],
        dtype=np.float64,
    )
    nearest: list[dict | None] = [None, None]
    peak_count = 0
    with path.open() as stream:
        for line in stream:
            stripped = line.strip()
            if stripped.startswith("###"):
                break
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) == 1 and peak_count == 0:
                continue
            if len(fields) != 7:
                raise RuntimeError(f"invalid HOP gbound peak row: {stripped[:120]}")
            group_id, group_npart = int(fields[0]), int(fields[1])
            peak_position = np.asarray([float(value) for value in fields[3:6]]) * box
            peak_density = float(fields[6])
            for member, target in enumerate(pair_positions):
                separation = float(distance(peak_position[None, :], target, box)[0])
                if nearest[member] is None or separation < nearest[member]["distance_mpc_h"]:
                    nearest[member] = {
                        "raw_group_id": group_id,
                        "raw_group_npart": group_npart,
                        "peak_position_mpc_h": peak_position.tolist(),
                        "peak_density_mean_units": peak_density,
                        "distance_mpc_h": separation,
                    }
            peak_count += 1
    if peak_count == 0 or any(item is None for item in nearest):
        raise RuntimeError("HOP gbound file contains no raw peak catalogue")
    group_ids = [int(item["raw_group_id"]) for item in nearest if item is not None]
    distances = [float(item["distance_mpc_h"]) for item in nearest if item is not None]
    return {
        "catalog": str(path),
        "peak_count": peak_count,
        "nearest": nearest,
        "distinct_raw_peaks": len(set(group_ids)) == 2,
        "both_within_1_mpc_h": max(distances) <= 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hop", type=Path, required=True)
    parser.add_argument("--galaxyfinder", type=Path, required=True)
    parser.add_argument("--p1", type=Path, required=True)
    parser.add_argument("--p2", type=Path)
    parser.add_argument("--raw-gbound", type=Path)
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
    raw_peak_crossmatch = (
        crossmatch_raw_hop_peaks(args.raw_gbound, selected, box)
        if args.raw_gbound is not None and selected is not None
        else None
    )
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

    pair_crossmatch = None
    if selected is not None:
        pair_crossmatch = classify_selected_pair_crossmatch(
            selected, position, mass, npart, box
        )
        pair_hop_rows = pair_crossmatch["pair_hop_rows"]
        pair_match_distance = pair_crossmatch["pair_match_distance_mpc_h"]
        third_threshold = min(
            float(selected["m1_fof_msun_h"]), float(selected["m2_fof_msun_h"])
        )
    elif hop_selected is not None:
        pair_hop_rows = [int(index) for index in hop_selected["hop_rows"]]
        pair_match_distance = [0.0, 0.0]
        pair_crossmatch = {
            "mode": "independent_hop_pair",
            "pair_hop_rows": pair_hop_rows,
            "pair_match_distance_mpc_h": pair_match_distance,
            "distinct_hop_groups": True,
            "within_match_radius": True,
            "pair_supported_for_parent_trace": True,
            "hop_independently_resolves_pair": True,
            "merged_group": None,
        }
        third_threshold = min(float(value) for value in hop_selected["masses_msun_h"])
    else:
        pair_hop_rows = []
        pair_match_distance = []
        pair_crossmatch = {
            "mode": "no_pair",
            "pair_hop_rows": [],
            "pair_match_distance_mpc_h": [],
            "distinct_hop_groups": False,
            "within_match_radius": False,
            "pair_supported_for_parent_trace": False,
            "hop_independently_resolves_pair": False,
            "merged_group": None,
        }
        third_threshold = None
    third_distance = distance(position, midpoint, box)
    third_candidates = (
        np.flatnonzero((mass >= third_threshold) & (third_distance <= 2.5))
        if third_threshold is not None else np.empty(0, dtype=np.int64)
    )
    third_candidates = np.asarray(
        [index for index in third_candidates if int(index) not in pair_hop_rows],
        dtype=np.int64,
    )
    standard_isolation_pass = bool(
        pair_crossmatch["pair_supported_for_parent_trace"]
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
    production_trace_pass = bool(
        cluster_pass
        and selected_environment_pass
        and pair_crossmatch["hop_independently_resolves_pair"]
    )
    decision = (
        "HOP_PARENT_STRUCTURE_PASS"
        if production_trace_pass
        else "HOP_PARENT_STRUCTURE_FAIL"
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
            "raw_peak_crossmatch": raw_peak_crossmatch,
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
            "pair_crossmatch": pair_crossmatch,
        },
        "gates": {
            "virgo_coma_mass": cluster_pass,
            "selected_environment_policy": args.environment_policy,
            "selected_environment_pass": selected_environment_pass,
            "v12_blanket_veto_diagnostic": environment_pass,
            "standard_lg_isolation": standard_isolation_pass,
            "hop_independently_resolves_pair": pair_crossmatch[
                "hop_independently_resolves_pair"
            ],
            "merged_pair_consistency": bool(
                pair_crossmatch.get("merged_group")
                and pair_crossmatch["merged_group"]["consistency_pass"]
            ),
            "production_trace_pass": production_trace_pass,
        },
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "groups": int(table.shape[0]), "nearby": len(nearby)}))


if __name__ == "__main__":
    main()
