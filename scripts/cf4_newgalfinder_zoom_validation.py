#!/usr/bin/env python3
"""Validate a DMO NewGalFinder catalogue for the frozen Local-Group gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cf4_p2_screen import load_config, rank_score  # noqa: E402
from cf4_galaxyfinder_parent_validation import (  # noqa: E402
    BOX,
    HALOQ,
    OBSERVER,
    diagnose_lg_pair_cutflow,
    distance,
    find_cluster,
    find_lg_pairs,
    min_image,
    sha256,
)

HALO_INFO = np.dtype(
    {
        "names": [
            "nsub", "ndm", "nstar", "nsink", "ngas", "npall",
            "totm", "mdm", "mgas", "msink", "mstar",
            "x", "y", "z", "vx", "vy", "vz",
        ],
        "formats": ["<i4"] * 6 + ["<f8"] * 11,
        "offsets": list(range(0, 24, 4)) + list(range(24, 112, 8)),
        "itemsize": 112,
    }
)
SUB_INFO = np.dtype(
    {
        "names": [
            "ndm", "ngas", "nsink", "nstar", "npall",
            "totm", "mdm", "mgas", "msink", "mstar",
            "x", "y", "z", "vx", "vy", "vz",
        ],
        "formats": ["<i4"] * 5 + ["<f8"] * 11,
        "offsets": [0, 4, 8, 12, 16] + list(range(24, 112, 8)),
        "itemsize": 112,
    }
)


def read_catalog(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hosts: list[np.void] = []
    children: list[np.void] = []
    parent: list[int] = []
    size = path.stat().st_size
    with path.open("rb") as stream:
        while stream.tell() < size:
            halo = np.fromfile(stream, dtype=HALO_INFO, count=1)
            if halo.size != 1:
                raise RuntimeError("truncated HaloInfo record")
            nsub = int(halo["nsub"][0])
            if nsub <= 0 or nsub > 100000:
                raise RuntimeError(f"invalid nsub={nsub} at byte {stream.tell() - 112}")
            sub = np.fromfile(stream, dtype=SUB_INFO, count=nsub)
            if sub.size != nsub:
                raise RuntimeError("truncated SubInfo records")
            host_index = len(hosts)
            hosts.append(halo[0])
            children.extend(sub)
            parent.extend([host_index] * nsub)
        if stream.tell() != size:
            raise RuntimeError("catalogue parser did not end at EOF")
    return (
        np.asarray(hosts, dtype=HALO_INFO),
        np.asarray(children, dtype=SUB_INFO),
        np.asarray(parent, dtype=np.int64),
    )


def as_haloq(hosts: np.ndarray) -> np.ndarray:
    catalog = np.zeros(hosts.size, dtype=HALOQ)
    catalog["np"] = hosts["npall"]
    catalog["npstar"] = hosts["nstar"]
    catalog["npgas"] = hosts["ngas"]
    catalog["npdm"] = hosts["ndm"]
    catalog["npsink"] = hosts["nsink"]
    for name in ("x", "y", "z", "vx", "vy", "vz"):
        catalog[name] = hosts[name]
    catalog["mass"] = hosts["totm"]
    catalog["mstar"] = hosts["mstar"]
    catalog["mgas"] = hosts["mgas"]
    catalog["mdm"] = hosts["mdm"]
    catalog["msink"] = hosts["msink"]
    return catalog


def add_m33_candidates(
    pairs: list[dict], children: np.ndarray, parent: np.ndarray, config: dict
) -> list[dict]:
    gate = config["m33_subpeak_gate"]
    mass_lo, mass_hi = map(float, gate["mass_range_msun_h"])
    sep_lo, sep_hi = map(float, gate["m31_separation_range_mpc_h"])
    max_fraction = float(gate["maximum_mass_fraction_of_m31"])
    position = np.column_stack((children["x"], children["y"], children["z"]))
    mass = children["totm"].astype(np.float64)
    enriched = []
    for raw in pairs:
        row = dict(raw)
        m31 = int(row["halo_i"] if row["m1_fof_msun_h"] >= row["m2_fof_msun_h"] else row["halo_j"])
        child_ids = np.flatnonzero(parent == m31)
        candidates = []
        if child_ids.size:
            primary = int(child_ids[np.argmax(mass[child_ids])])
            m31_position = position[primary]
            m31_mass = float(mass[primary])
            for child in child_ids:
                child = int(child)
                if child == primary:
                    continue
                separation = float(np.linalg.norm(min_image(position[child] - m31_position)))
                if not (mass_lo <= mass[child] <= mass_hi):
                    continue
                if not (sep_lo <= separation <= sep_hi):
                    continue
                if mass[child] > max_fraction * m31_mass:
                    continue
                candidates.append(
                    {
                        "child_index": child,
                        "parent_host_index": m31,
                        "mass_msun_h": float(mass[child]),
                        "npart": int(children["npall"][child]),
                        "position_mpc_h": position[child].tolist(),
                        "m31_child_index": primary,
                        "m31_bound_mass_msun_h": m31_mass,
                        "separation_mpc_h": separation,
                        "mass_fraction_of_m31": float(mass[child] / m31_mass),
                    }
                )
        candidates.sort(key=lambda item: (-item["mass_msun_h"], item["separation_mpc_h"]))
        row["m33_candidates"] = candidates
        row["m33_candidate"] = candidates[0] if candidates else None
        row["ranking_score"] = rank_score(row, config["ranking"])
        enriched.append(row)
    enriched.sort(key=lambda item: item["ranking_score"])
    return enriched


def reference_comparison(selected: Optional[dict], reference: dict) -> Optional[dict]:
    if selected is None:
        return None
    old = reference["lg"]["selected"]
    old_midpoint = np.asarray(old["midpoint_mpc_h"], dtype=np.float64)
    new_midpoint = np.asarray(selected["midpoint_mpc_h"], dtype=np.float64)
    return {
        "reference_midpoint_mpc_h": old_midpoint.tolist(),
        "new_midpoint_mpc_h": new_midpoint.tolist(),
        "periodic_midpoint_drift_mpc_h": float(np.linalg.norm(min_image(new_midpoint - old_midpoint))),
        "reference_pair_separation_mpc_h": float(old["separation_mpc_h"]),
        "new_pair_separation_mpc_h": float(selected["separation_mpc_h"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hosts, children, parent = read_catalog(args.catalog)
    if not hosts.size or not children.size:
        raise RuntimeError("empty NewGalFinder catalogue")
    for values, label in ((hosts, "hosts"), (children, "children")):
        for field in ("totm", "x", "y", "z", "vx", "vy", "vz"):
            if not np.all(np.isfinite(values[field])):
                raise RuntimeError(f"non-finite {label}.{field}")

    config = load_config(args.config)
    reference = json.loads(args.parent_reference.read_text())
    catalog = as_haloq(hosts)
    pairs = add_m33_candidates(find_lg_pairs(catalog, config), children, parent, config)
    selected = pairs[0] if pairs else None
    cluster_rows = {}
    for name, old in reference["clusters"].items():
        cluster_rows[name] = find_cluster(
            catalog,
            np.asarray(old["target_position_mpc_h"], dtype=np.float64),
            float(old["search_radius_mpc_h"]),
        )

    if selected is None:
        decision = "NEWGAL_DIAGNOSTIC_PAIR_NO_GO"
    elif selected["m33_candidate"] is None:
        decision = "NEWGAL_DIAGNOSTIC_PAIR_PASS_M33_UNRESOLVED"
    else:
        decision = "NEWGAL_DIAGNOSTIC_PAIR_AND_M33_CANDIDATE"
    result = {
        "schema": "ouruniv-cf4-s40349-newgalfinder-validation-v1",
        "decision": decision,
        "claim_boundary": (
            "Finder-level diagnostic only. The host pair uses frozen FoF gates; "
            "an M33 entry is a bound-child candidate from one random conditional "
            "high-k realization, not M33 information recovered from CF4."
        ),
        "catalog": str(args.catalog),
        "catalog_sha256": sha256(args.catalog),
        "config": str(args.config),
        "parent_reference": str(args.parent_reference),
        "host_count": int(hosts.size),
        "bound_component_count": int(children.size),
        "multi_component_host_count": int(np.count_nonzero(hosts["nsub"] > 1)),
        "maximum_components_per_host": int(hosts["nsub"].max()),
        "pair_count": len(pairs),
        "selected": selected,
        "nearest_pairs": pairs[:12],
        "cutflow": diagnose_lg_pair_cutflow(catalog, config),
        "reference_comparison": reference_comparison(selected, reference),
        "clusters": cluster_rows,
        "remaining_gates": [
            "measure low-resolution particle contamination for selected components",
            "inspect the selected pair and M33 candidate against particle-level membership",
            "do not promote the trace-only parent from this single random high-k draw",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": decision, "hosts": int(hosts.size), "components": int(children.size), "pairs": len(pairs)}))


if __name__ == "__main__":
    main()
