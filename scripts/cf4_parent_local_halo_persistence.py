#!/usr/bin/env python3
"""Test whether Local-Volume halo failures predate LG high-k conditioning."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def connected_clusters(points: np.ndarray, linking_length: float) -> list[np.ndarray]:
    """Return deterministic connected components of a fixed-radius graph."""
    if len(points) == 0:
        return []
    parent = np.arange(len(points), dtype=np.int64)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    for aa, bb in cKDTree(points).query_pairs(linking_length, output_type="ndarray"):
        ra, rb = root(int(aa)), root(int(bb))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for i in range(len(points)):
        groups.setdefault(root(i), []).append(i)
    return [np.asarray(groups[key], dtype=np.int64) for key in sorted(groups)]


def summarize_clusters(
    rows: list[dict], n_realizations: int, linking_length: float
) -> list[dict]:
    if not rows:
        return []
    points = np.asarray([row["position_mpc_h"] for row in rows], dtype=np.float64)
    clusters = []
    for indices in connected_clusters(points, linking_length):
        selected = [rows[int(i)] for i in indices]
        realization_ids = sorted({int(row["realization_id"]) for row in selected})
        masses = np.asarray([row["mass_msun_h"] for row in selected])
        centre = np.mean(points[indices], axis=0)
        clusters.append({
            "centre_mpc_h": centre.tolist(),
            "detection_count": len(selected),
            "unique_realization_count": len(realization_ids),
            "realization_fraction": len(realization_ids) / n_realizations,
            "realization_ids": realization_ids,
            "mass_range_msun_h": [float(masses.min()), float(masses.max())],
            "maximum_scatter_mpc_h": float(
                np.linalg.norm(points[indices] - centre, axis=1).max(initial=0.0)
            ),
        })
    return sorted(
        clusters,
        key=lambda row: (-row["unique_realization_count"], row["centre_mpc_h"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-catalogue-glob", required=True)
    parser.add_argument("--conditioned-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--box-size", type=float, default=384.0)
    parser.add_argument("--mass-threshold", type=float, default=5.0e12)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--cluster-linking-length", type=float, default=1.25)
    parser.add_argument("--parent-persistence-fraction", type=float, default=0.5)
    args = parser.parse_args()

    paths = [Path(value) for value in sorted(glob.glob(args.parent_catalogue_glob))]
    if len(paths) != 32:
        raise RuntimeError(f"expected 32 parent catalogues, found {len(paths)}")
    observer = np.full(3, args.box_size / 2.0)
    parent_rows = []
    parent_sources = []
    seed_pattern = re.compile(r"_s(\d+)\.npz$")
    for path in paths:
        match = seed_pattern.search(path.name)
        if match is None:
            raise RuntimeError(f"cannot parse seed from {path.name}")
        seed = int(match.group(1))
        with np.load(path, allow_pickle=False) as data:
            pos = np.asarray(data["halo_pos"], dtype=np.float64)
            mass = np.asarray(data["halo_mass"], dtype=np.float64)
        distance = np.linalg.norm(pos - observer, axis=1)
        selected = np.flatnonzero(
            (mass >= args.mass_threshold) & (distance < args.radius)
        )
        for index in selected:
            parent_rows.append({
                "realization_id": seed,
                "halo_index": int(index),
                "mass_msun_h": float(mass[index]),
                "distance_mpc_h": float(distance[index]),
                "position_mpc_h": pos[index].tolist(),
            })
        parent_sources.append({
            "seed": seed,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "offender_count": int(len(selected)),
        })

    conditioned_source = json.loads(args.conditioned_comparison.read_text())
    conditioned_rows = []
    for row in conditioned_source["rows"]:
        for offender in row["observer_offenders"]:
            conditioned_rows.append({
                "realization_id": int(row["index"]),
                "halo_index": int(offender["halo_index"]),
                "mass_msun_h": float(offender["mass_msun_h"]),
                "distance_mpc_h": float(offender["distance_mpc_h"]),
                "position_mpc_h": offender["position_mpc_h"],
            })
    parent_clusters = summarize_clusters(parent_rows, len(paths), args.cluster_linking_length)
    conditioned_clusters = summarize_clusters(
        conditioned_rows, len(conditioned_source["rows"]), args.cluster_linking_length
    )
    for cluster in conditioned_clusters:
        if parent_clusters:
            distances = np.linalg.norm(
                np.asarray([item["centre_mpc_h"] for item in parent_clusters])
                - np.asarray(cluster["centre_mpc_h"]),
                axis=1,
            )
            match_index = int(np.argmin(distances))
            cluster["nearest_parent_cluster_index"] = match_index
            cluster["nearest_parent_cluster_distance_mpc_h"] = float(distances[match_index])
            cluster["nearest_parent_cluster_realization_fraction"] = float(
                parent_clusters[match_index]["realization_fraction"]
            )

    recurrent_conditioned = [
        row for row in conditioned_clusters if row["realization_fraction"] >= 0.25
    ]
    matched_parent_persistent = [
        row for row in recurrent_conditioned
        if row.get("nearest_parent_cluster_distance_mpc_h", np.inf)
        <= args.cluster_linking_length
        and row.get("nearest_parent_cluster_realization_fraction", 0.0)
        >= args.parent_persistence_fraction
    ]
    decision = (
        "PARENT_ENVIRONMENT_IS_LIMITING"
        if matched_parent_persistent
        else "LG_CONDITIONING_OR_HIGH_K_IS_LIMITING"
    )
    result = {
        "schema": "ouruniv-cf4-parent-local-halo-persistence-v1",
        "mass_threshold_msun_h": args.mass_threshold,
        "radius_mpc_h": args.radius,
        "cluster_linking_length_mpc_h": args.cluster_linking_length,
        "parent_persistence_fraction": args.parent_persistence_fraction,
        "parent_realization_count": len(paths),
        "conditioned_pair_realization_count": len(conditioned_source["rows"]),
        "parent_sources": parent_sources,
        "conditioned_comparison": str(args.conditioned_comparison.resolve()),
        "conditioned_comparison_sha256": sha256_file(args.conditioned_comparison),
        "parent_offender_rows": parent_rows,
        "conditioned_offender_rows": conditioned_rows,
        "parent_clusters": parent_clusters,
        "conditioned_clusters": conditioned_clusters,
        "matched_parent_persistent_cluster_count": len(matched_parent_persistent),
        "decision": decision,
        "interpretation": (
            "A recurrent conditioned offender matching a halo location present in at "
            "least half of independent unconditioned high-k completions is attributed "
            "to the fixed parent environment, not to the LG peak constraints alone."
        ),
    }
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "parent_offender_realizations": sum(
            source["offender_count"] > 0 for source in parent_sources
        ),
        "parent_cluster_count": len(parent_clusters),
        "conditioned_cluster_count": len(conditioned_clusters),
        "matched_parent_persistent_cluster_count": len(matched_parent_persistent),
        "decision": decision,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
