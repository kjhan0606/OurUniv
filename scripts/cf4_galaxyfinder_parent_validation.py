#!/usr/bin/env python3
"""Validate the selected CF4 parent with the RAMSES GalaxyFinder catalogue.

The catalogue is the native GalaxyFinder HaloQ stream (not the legacy GOTPM
bridge).  The selected Local-Group member IDs are retained for the following
Lagrangian-mask step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cf4_p2_screen import load_config, rank_score  # noqa: E402

BOX = 384.0
OBSERVER = np.full(3, BOX / 2.0)
HALOQ = np.dtype(
    {
        "names": [
            "np", "npstar", "npgas", "npdm", "npsink",
            "x", "y", "z", "mass", "mstar", "mgas", "mdm", "msink",
            "vx", "vy", "vz",
        ],
        "formats": [
            "<u8", "<u8", "<u8", "<u8", "<u8",
            "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8",
            "<f4", "<f4", "<f4",
        ],
        "offsets": [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 108, 112],
        "itemsize": 120,
    }
)
DM_RECORD = np.dtype(
    {
        "names": ["x", "y", "z", "vx", "vy", "vz", "mass", "id"],
        "formats": ["<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<f8", "<i4"],
        "offsets": [0, 8, 16, 24, 32, 40, 48, 56],
        "itemsize": 168,
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def min_image(vector: np.ndarray) -> np.ndarray:
    return vector - BOX * np.rint(vector / BOX)


def distance(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.norm(min_image(position - target), axis=-1)


def target_position(entry: dict) -> np.ndarray:
    longitude = math.radians(float(entry["sgl_deg"]))
    latitude = math.radians(float(entry["sgb_deg"]))
    radius = float(entry["distance_mpc"]) * 0.746
    offset = radius * np.array(
        [math.cos(latitude) * math.cos(longitude),
         math.cos(latitude) * math.sin(longitude),
         math.sin(latitude)]
    )
    return (OBSERVER + offset) % BOX


def read_catalog(path: Path) -> tuple[dict, np.ndarray]:
    size = path.stat().st_size
    if size < 28 or (size - 28) % HALOQ.itemsize:
        raise RuntimeError(f"invalid HaloQ catalogue size: {size}")
    header_values = np.fromfile(path, dtype="<f4", count=7)
    if header_values.size != 7:
        raise RuntimeError("truncated GalaxyFinder catalogue header")
    names = ("box_mpc_h", "hubble", "omega_m", "omega_b", "omega_l", "amax", "anow")
    header = {name: float(value) for name, value in zip(names, header_values)}
    catalog = np.fromfile(path, dtype=HALOQ, offset=28)
    return header, catalog


def find_cluster(catalog: np.ndarray, target: np.ndarray, radius: float) -> dict:
    position = np.column_stack((catalog["x"], catalog["y"], catalog["z"]))
    separation = distance(position, target)
    inside = np.flatnonzero(separation <= radius)
    if inside.size == 0:
        return {"found": False, "target_position_mpc_h": target.tolist(), "search_radius_mpc_h": radius}
    order = inside[np.argsort(catalog["mass"][inside])[::-1]]
    index = int(order[0])
    return {
        "found": True,
        "catalog_index": index,
        "target_position_mpc_h": target.tolist(),
        "search_radius_mpc_h": radius,
        "position_mpc_h": position[index].tolist(),
        "target_separation_mpc_h": float(separation[index]),
        "mass_msun_h": float(catalog["mass"][index]),
        "npart": int(catalog["np"][index]),
    }


def find_lg_pairs(catalog: np.ndarray, p2: dict) -> list[dict]:
    screen = p2["screen"]
    position = np.column_stack((catalog["x"], catalog["y"], catalog["z"]))
    velocity = np.column_stack((catalog["vx"], catalog["vy"], catalog["vz"])).astype(np.float64)
    mass = catalog["mass"].astype(np.float64)
    mass_lo, mass_hi = map(float, screen["pair_member_mass_range_msun_h"])
    radius = distance(position, OBSERVER)
    eligible = np.flatnonzero(
        (mass >= mass_lo) & (mass <= mass_hi)
        & (radius <= float(screen["pair_midpoint_max_offset_mpc_h"]) + 1.0)
    )
    massive = np.flatnonzero(mass >= float(screen["isolation_mass_threshold_msun_h"]))
    sep_lo, sep_hi = map(float, screen["pair_separation_range_mpc_h"])
    rows: list[dict] = []
    for local_a in range(eligible.size):
        i = int(eligible[local_a])
        for local_b in range(local_a + 1, eligible.size):
            j = int(eligible[local_b])
            vector = min_image(position[i] - position[j])
            separation = float(np.linalg.norm(vector))
            if separation < sep_lo or separation > sep_hi:
                continue
            ratio = float(max(mass[i], mass[j]) / min(mass[i], mass[j]))
            if ratio > float(screen["pair_mass_ratio_max"]):
                continue
            midpoint = (position[j] + 0.5 * vector) % BOX
            midpoint_offset = float(distance(midpoint[None, :], OBSERVER)[0])
            if midpoint_offset > float(screen["pair_midpoint_max_offset_mpc_h"]):
                continue
            external = [
                float(distance(position[k][None, :], midpoint)[0])
                for k in massive if int(k) not in (i, j)
            ]
            isolation = min(external) if external else 99.0
            if isolation < float(screen["isolation_radius_mpc_h"]):
                continue
            radial_hat = vector / separation
            relative_velocity = velocity[i] - velocity[j]
            peculiar_radial = float(np.dot(relative_velocity, radial_hat))
            tangential = float(np.linalg.norm(relative_velocity - peculiar_radial * radial_hat))
            row = {
                "halo_i": i,
                "halo_j": j,
                "m1_fof_msun_h": float(mass[i]),
                "m2_fof_msun_h": float(mass[j]),
                "npart": [int(catalog["np"][i]), int(catalog["np"][j])],
                "mass_ratio": ratio,
                "separation_mpc_h": separation,
                "midpoint_mpc_h": midpoint.tolist(),
                "midpoint_offset_mpc_h": midpoint_offset,
                "isolation_mpc_h": isolation,
                "peculiar_radial_velocity_km_s": peculiar_radial,
                "total_radial_velocity_km_s": peculiar_radial + 100.0 * separation,
                "tangential_velocity_km_s": tangential,
                "m33_candidate": None,
            }
            row["ranking_score"] = rank_score(row, p2["ranking"])
            rows.append(row)
    rows.sort(key=lambda item: item["ranking_score"])
    return rows


def member_ids(member_path: Path, catalog: np.ndarray, indices: list[int]) -> list[np.ndarray]:
    cumulative = np.concatenate(([0], np.cumsum(catalog["np"], dtype=np.uint64)))
    expected = int(cumulative[-1]) * DM_RECORD.itemsize
    if member_path.stat().st_size != expected:
        raise RuntimeError(
            f"member stream size mismatch: {member_path.stat().st_size} != {expected}"
        )
    arrays = []
    for index in indices:
        count = int(catalog["np"][index])
        offset = int(cumulative[index]) * DM_RECORD.itemsize
        record = np.fromfile(member_path, dtype=DM_RECORD, count=count, offset=offset)
        if record.size != count:
            raise RuntimeError(f"truncated member stream for halo {index}")
        arrays.append(record["id"].astype(np.int64))
    return arrays


def nearby_massive(catalog: np.ndarray, target: np.ndarray, radius: float = 12.0) -> list[dict]:
    position = np.column_stack((catalog["x"], catalog["y"], catalog["z"]))
    separation = distance(position, target)
    selected = np.flatnonzero((catalog["mass"] >= 5.0e12) & (separation <= radius))
    selected = selected[np.argsort(separation[selected])]
    return [
        {
            "catalog_index": int(index),
            "mass_msun_h": float(catalog["mass"][index]),
            "npart": int(catalog["np"][index]),
            "position_mpc_h": position[index].tolist(),
            "separation_mpc_h": float(separation[index]),
        }
        for index in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--members", type=Path, required=True)
    parser.add_argument("--p1", type=Path, required=True)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--member-output", type=Path, required=True)
    args = parser.parse_args()

    p1 = json.loads(args.p1.read_text())
    p2 = load_config(args.p2)
    header, catalog = read_catalog(args.catalog)
    if not np.isclose(header["box_mpc_h"], BOX, rtol=0, atol=1e-3):
        raise RuntimeError(f"catalogue box mismatch: {header['box_mpc_h']}")
    if catalog.size == 0 or not np.all(np.isfinite(catalog["mass"])):
        raise RuntimeError("empty or non-finite GalaxyFinder catalogue")
    if np.any(catalog["mass"] <= 0) or np.any(catalog["np"] < 20):
        raise RuntimeError("invalid halo mass/member count")
    if not np.all(catalog["np"] == catalog["npdm"]):
        raise RuntimeError("DM-only catalogue contains non-DM members")

    clusters = {
        name: find_cluster(
            catalog,
            target_position(entry),
            float(entry["search_radius_mpc_h"]),
        )
        for name, entry in p1["clusters"].items()
    }
    pairs = find_lg_pairs(catalog, p2)
    selected = pairs[0] if pairs else None
    position = np.column_stack((catalog["x"], catalog["y"], catalog["z"]))
    massive = np.flatnonzero(catalog["mass"] >= 5.0e12)
    exclusion_index = None
    if massive.size:
        exclusion_index = int(massive[np.argmin(distance(position[massive], OBSERVER))])
    if selected is not None:
        trace_indices = [selected["halo_i"], selected["halo_j"]]
        if exclusion_index is not None and exclusion_index not in trace_indices:
            trace_indices.append(exclusion_index)
        ids = member_ids(args.members, catalog, trace_indices)
        if any(values.size != np.unique(values).size for values in ids):
            raise RuntimeError("duplicate particle ID within selected halo")
        if np.intersect1d(ids[0], ids[1]).size:
            raise RuntimeError("selected pair shares particle IDs")
        args.member_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "halo_i": np.int64(selected["halo_i"]),
            "halo_j": np.int64(selected["halo_j"]),
            "member_ids_i": ids[0],
            "member_ids_j": ids[1],
        }
        if len(trace_indices) == 3:
            payload["exclusion_halo"] = np.int64(trace_indices[2])
            payload["exclusion_member_ids"] = ids[2]
        np.savez_compressed(args.member_output, **payload)

    massive_mask = catalog["mass"] >= 5.0e12
    observer_nearest = float(distance(position[massive_mask], OBSERVER).min()) if np.any(massive_mask) else 99.0
    midpoint_nearest = (
        float(distance(position[massive_mask], np.asarray(selected["midpoint_mpc_h"])).min())
        if selected is not None and np.any(massive_mask) else 99.0
    )
    cluster_pass = bool(
        clusters["Virgo"].get("mass_msun_h", 0.0) >= 1.0e14
        and clusters["Coma"].get("mass_msun_h", 0.0) >= 5.0e14
    )
    environment_fof_pass = observer_nearest >= 8.0 and midpoint_nearest >= 8.0
    parent_core_pass = bool(selected is not None and cluster_pass)
    result = {
        "schema": "ouruniv-cf4-lg-selected-parent-galaxyfinder-validation-v1",
        "stage": "7/8 parent RAMSES z=0 structure validation",
        "catalog": str(args.catalog),
        "catalog_sha256": sha256(args.catalog),
        "members": str(args.members),
        "members_sha256": sha256(args.members),
        "member_output": str(args.member_output) if selected is not None else None,
        "header": header,
        "haloq_record_bytes": HALOQ.itemsize,
        "dm_record_bytes": DM_RECORD.itemsize,
        "halo_count": int(catalog.size),
        "total_catalogued_particles": int(catalog["np"].sum(dtype=np.uint64)),
        "catalogue_mass_range_msun_h": [float(catalog["mass"].min()), float(catalog["mass"].max())],
        "lg": {
            "hard_pair_count": len(pairs),
            "selected": selected,
            "m33_status": "unresolved at L9; mandatory in the L12 zoom",
        },
        "clusters": clusters,
        "environment": {
            "mass_threshold_msun_h": 5.0e12,
            "nearest_to_observer_mpc_h": observer_nearest,
            "nearest_to_lg_midpoint_mpc_h": midpoint_nearest,
            "minimum_required_mpc_h": 8.0,
            "near_observer": nearby_massive(catalog, OBSERVER),
            "near_lg_midpoint": (
                nearby_massive(catalog, np.asarray(selected["midpoint_mpc_h"]))
                if selected is not None else []
            ),
            "fof_proxy_pass": environment_fof_pass,
            "status": (
                "PASS" if environment_fof_pass else
                "REQUIRES_HOP_OR_M200C_CHECK; the frozen gate is defined for HOP/M200c, not FoF mass"
            ),
        },
        "gates": {
            "lg_pair": selected is not None,
            "virgo_coma_mass": cluster_pass,
            "observer_environment_fof_proxy": environment_fof_pass,
            "observer_environment_definitive": None,
        },
        "decision": (
            "PARENT_RAMSES_HALO_PASS" if parent_core_pass and environment_fof_pass
            else "PARENT_RAMSES_HALO_CONDITIONAL_LOCAL_ENVIRONMENT" if parent_core_pass
            else "PARENT_RAMSES_HALO_NO_GO"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": result["decision"], "halo_count": int(catalog.size), "lg_pairs": len(pairs)}))
    if not parent_core_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
