#!/usr/bin/env python3
"""Inspect the frozen seed-40349 LG cutflow and its NewGalFinder DMO members."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from cf4_newgalfinder_zoom_validation import (  # noqa: E402
    HALO_INFO,
    SUB_INFO,
    read_catalog,
)
from cf4_galaxyfinder_parent_validation import BOX, distance, min_image, sha256  # noqa: E402


DM_RECORD = np.dtype(
    {
        "names": ["x", "y", "z", "vx", "vy", "vz", "mass", "id", "levelp", "family", "tag"],
        "formats": ["<f8"] * 7 + ["<i4", "<i4", "i1", "i1"],
        "offsets": [0, 8, 16, 24, 32, 40, 48, 56, 60, 64, 65],
        "itemsize": 72,
    }
)


def particle_summary(particles: np.ndarray) -> dict:
    if not particles.size:
        return {"particle_count": 0, "mass_sum_msun_h": 0.0, "levels": []}
    mass = particles["mass"].astype(np.float64)
    levels = []
    for level in np.unique(particles["levelp"]):
        mask = particles["levelp"] == level
        levels.append(
            {
                "levelp": int(level),
                "count": int(mask.sum()),
                "mass_sum_msun_h": float(mass[mask].sum()),
                "mass_min_msun_h": float(mass[mask].min()),
                "mass_max_msun_h": float(mass[mask].max()),
            }
        )
    return {
        "particle_count": int(particles.size),
        "mass_sum_msun_h": float(mass.sum()),
        "mass_min_msun_h": float(mass.min()),
        "mass_max_msun_h": float(mass.max()),
        "levels": levels,
    }


def read_selected_members(
    data_path: Path,
    hosts: np.ndarray,
    children: np.ndarray,
    parent: np.ndarray,
    selected: set[int],
) -> dict[int, list[dict]]:
    if np.any(hosts["ngas"] != 0) or np.any(hosts["nstar"] != 0) or np.any(hosts["nsink"] != 0):
        raise RuntimeError("DMO-only member layout required")
    if np.any(children["ngas"] != 0) or np.any(children["nstar"] != 0) or np.any(children["nsink"] != 0):
        raise RuntimeError("DMO-only component layout required")
    child_counts = np.bincount(parent, weights=children["ndm"], minlength=hosts.size).astype(np.int64)
    sizes = HALO_INFO.itemsize + hosts["nsub"].astype(np.int64) * SUB_INFO.itemsize
    sizes += child_counts * DM_RECORD.itemsize
    offsets = np.empty(hosts.size, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(sizes[:-1])
    if int(sizes.sum()) != data_path.stat().st_size:
        raise RuntimeError("GALFIND.DATA size disagrees with DMO catalogue records")
    result = {}
    with data_path.open("rb") as stream:
        for host_index in sorted(selected):
            stream.seek(int(offsets[host_index]))
            header = np.fromfile(stream, dtype=HALO_INFO, count=1)
            if header.size != 1 or int(header["nsub"][0]) != int(hosts["nsub"][host_index]):
                raise RuntimeError(f"host header mismatch at {host_index}")
            rows = []
            child_indices = np.flatnonzero(parent == host_index)
            for child_index in child_indices:
                child_header = np.fromfile(stream, dtype=SUB_INFO, count=1)
                if child_header.size != 1 or int(child_header["ndm"][0]) != int(children["ndm"][child_index]):
                    raise RuntimeError(f"component header mismatch at {child_index}")
                particles = np.fromfile(stream, dtype=DM_RECORD, count=int(children["ndm"][child_index]))
                if particles.size != int(children["ndm"][child_index]):
                    raise RuntimeError(f"truncated component {child_index}")
                summary = particle_summary(particles)
                if not np.isclose(summary["mass_sum_msun_h"], float(children["mdm"][child_index]), rtol=1e-8):
                    raise RuntimeError(f"component mass mismatch at {child_index}")
                rows.append({"child_index": int(child_index), "particles": summary})
            if stream.tell() != int(offsets[host_index] + sizes[host_index]):
                raise RuntimeError(f"host data length mismatch at {host_index}")
            result[host_index] = rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")

    validation = json.loads(args.validation.read_text())
    config = json.loads(args.config.read_text())
    if validation["catalog_sha256"] != sha256(args.catalog):
        raise RuntimeError("catalogue hash differs from frozen validation")
    hosts, children, parent = read_catalog(args.catalog)
    if hosts.size != validation["host_count"] or children.size != validation["bound_component_count"]:
        raise RuntimeError("catalogue counts differ from frozen validation")
    candidate = next(
        (row for row in validation["cutflow"]["nearest_pairs"] if row["failed_cuts"] == ["isolation"]),
        None,
    )
    if candidate is None:
        raise RuntimeError("no isolation-only rejected pair in frozen cutflow")
    pair = [int(candidate["halo_i"]), int(candidate["halo_j"])]
    position = np.column_stack((hosts["x"], hosts["y"], hosts["z"]))
    midpoint = np.asarray(candidate["midpoint_mpc_h"], dtype=np.float64)
    mass = hosts["totm"].astype(np.float64)
    massive = np.flatnonzero(mass >= float(config["screen"]["isolation_mass_threshold_msun_h"]))
    massive = massive[~np.isin(massive, pair)]
    separations = distance(position[massive], midpoint)
    intruder = int(massive[np.argmin(separations)])
    if not np.isclose(float(separations.min()), candidate["isolation_mpc_h"], atol=1e-6):
        raise RuntimeError("isolation source differs from frozen cutflow")

    eligible = np.flatnonzero(
        (mass >= float(config["screen"]["pair_member_mass_range_msun_h"][0]))
        & (mass <= float(config["screen"]["pair_member_mass_range_msun_h"][1]))
        & (distance(position, np.full(3, BOX / 2))
           <= float(config["screen"]["pair_midpoint_max_offset_mpc_h"]) + 1.0)
    )
    targets = set(map(int, eligible)) | {intruder}
    members = read_selected_members(args.data, hosts, children, parent, targets)
    gate = config["m33_subpeak_gate"]
    host_rows = []
    for host_index in sorted(targets):
        child_indices = np.flatnonzero(parent == host_index)
        primary = int(child_indices[np.argmax(children["totm"][child_indices])])
        primary_pos = np.array([children[axis][primary] for axis in "xyz"], dtype=np.float64)
        primary_mass = float(children["totm"][primary])
        member_by_child = {row["child_index"]: row["particles"] for row in members[host_index]}
        component_rows = []
        for child_index in child_indices:
            child_index = int(child_index)
            child_pos = np.array([children[axis][child_index] for axis in "xyz"], dtype=np.float64)
            separation = float(np.linalg.norm(min_image(child_pos - primary_pos)))
            child_mass = float(children["totm"][child_index])
            m33_scale = (
                child_index != primary
                and gate["mass_range_msun_h"][0] <= child_mass <= gate["mass_range_msun_h"][1]
                and gate["m31_separation_range_mpc_h"][0] <= separation <= gate["m31_separation_range_mpc_h"][1]
                and child_mass <= gate["maximum_mass_fraction_of_m31"] * primary_mass
            )
            component_rows.append(
                {
                    "child_index": child_index,
                    "is_primary": child_index == primary,
                    "mass_msun_h": child_mass,
                    "position_mpc_h": child_pos.tolist(),
                    "separation_from_primary_mpc_h": separation,
                    "m33_scale_gate_pass": bool(m33_scale),
                    "particles": member_by_child[child_index],
                }
            )
        host_rows.append(
            {
                "host_index": host_index,
                "role": "isolation_intruder" if host_index == intruder else "eligible_host",
                "mass_msun_h": float(mass[host_index]),
                "npart": int(hosts["npall"][host_index]),
                "position_mpc_h": position[host_index].tolist(),
                "distance_to_pair_midpoint_mpc_h": float(distance(position[host_index], midpoint)),
                "component_count": int(hosts["nsub"][host_index]),
                "bound_component_mass_sum_msun_h": float(children["totm"][child_indices].sum()),
                "components": component_rows,
            }
        )
    result = {
        "schema": "ouruniv-cf4-s40349-newgalfinder-local-audit-v1",
        "interpretation": "Frozen-cut diagnostic; no threshold changed and no observed identity assigned.",
        "catalog_sha256": validation["catalog_sha256"],
        "data_path": str(args.data),
        "candidate_pair": candidate,
        "isolation_intruder_host_index": intruder,
        "isolation_intruder_mass_msun_h": float(mass[intruder]),
        "isolation_intruder_distance_to_midpoint_mpc_h": float(separations.min()),
        "eligible_host_indices": list(map(int, eligible)),
        "hosts": host_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pair": pair, "intruder": intruder, "local_hosts": len(host_rows)}))


if __name__ == "__main__":
    main()
