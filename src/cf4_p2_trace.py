#!/usr/bin/env python3
"""Trace a promoted P2 PM candidate to its Lagrangian particle positions.

This is the inexpensive precursor to the RAMSES/HOP pilot.  It repeats the
exact full-box PM realization used by ``cf4_p2_screen.py``, selects a periodic
Eulerian sphere around the frozen LG midpoint, and records the corresponding
regular-grid particle coordinates.  Particle row numbers are retained as
stable PM IDs.  The resulting ``lagrangian`` array is accepted by
``cf4_lagrangian_mask.py``.

The trace is only a mask-construction aid.  It does not replace the definitive
RAMSES HOP/M200c, kinematic, M33, and contamination gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

SCHEMA = "ouruniv-pm-id-trace-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def trace_periodic_sphere(
    particles,
    center: np.ndarray,
    radius: float,
    *,
    chunk_size: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stable row IDs, Lagrangian positions, and final positions."""
    pmid = np.asarray(particles.pmid)
    disp = np.asarray(particles.disp, dtype=np.float32)
    cell_size = np.asarray(particles.conf.cell_size, dtype=np.float32)
    box_size = np.asarray(particles.conf.box_size, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    radius2 = float(radius) ** 2
    ids, lagrangian, final = [], [], []
    for start in range(0, pmid.shape[0], chunk_size):
        stop = min(start + chunk_size, pmid.shape[0])
        q = pmid[start:stop].astype(np.float32) * cell_size
        pos = np.mod(q + disp[start:stop], box_size)
        delta = pos - center
        delta -= box_size * np.rint(delta / box_size)
        keep = np.einsum("ij,ij->i", delta, delta) <= radius2
        if np.any(keep):
            local = np.flatnonzero(keep).astype(np.int64)
            ids.append(local + start + 1)
            lagrangian.append(q[keep])
            final.append(pos[keep])
    if not ids:
        raise RuntimeError("no PM particles found in the frozen target sphere")
    return np.concatenate(ids), np.concatenate(lagrangian), np.concatenate(final)


def trace_position_array(
    final_position_array,
    center: np.ndarray,
    radius: float,
    *,
    mesh_size: int,
    spacing: float,
    box_size: float,
    chunk_size: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace selected final positions using stable regular-grid row IDs."""
    pos_all = np.asarray(final_position_array, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    ids, lagrangian, final = [], [], []
    radius2 = float(radius) ** 2
    stride = mesh_size * mesh_size
    for start in range(0, pos_all.shape[0], chunk_size):
        stop = min(start + chunk_size, pos_all.shape[0])
        pos = pos_all[start:stop]
        delta = pos - center
        delta -= box_size * np.rint(delta / box_size)
        keep = np.einsum("ij,ij->i", delta, delta) <= radius2
        if not np.any(keep):
            continue
        zero_id = np.flatnonzero(keep).astype(np.int64) + start
        ii = zero_id // stride
        remainder = zero_id % stride
        jj = remainder // mesh_size
        kk = remainder % mesh_size
        ids.append(zero_id + 1)
        lagrangian.append(
            np.column_stack((ii, jj, kk)).astype(np.float32) * spacing)
        final.append(pos[keep].copy())
    if not ids:
        raise RuntimeError("no PM particles found in the frozen target sphere")
    return np.concatenate(ids), np.concatenate(lagrangian), np.concatenate(final)


def select_candidate(result: dict, parent: int | None, small: int | None) -> dict:
    passing = [row for row in result["results"] if row["screen_pass"]]
    if parent is not None or small is not None:
        if parent is None or small is None:
            raise ValueError("--parent-seed and --small-scale-seed must be used together")
        passing = [
            row for row in passing
            if row["parent_seed"] == parent and row["small_scale_seed"] == small
        ]
    if not passing:
        raise RuntimeError("the requested P2 result has no matching passing candidate")
    return min(passing, key=lambda row: row["best_pair"]["ranking_score"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2-result", type=Path, required=True)
    parser.add_argument("--p1-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-seed", type=int)
    parser.add_argument("--small-scale-seed", type=int)
    parser.add_argument("--radius-mpc-h", type=float, default=5.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = json.loads(args.p2_result.read_text())
    from cf4_p2_screen import load_config

    config = load_config(args.config)
    p1 = json.loads(args.p1_result.read_text())
    if result.get("status") != "complete":
        parser.error("--p2-result must be a complete full paired screen")
    if result.get("config_sha256") != sha256_file(args.config):
        parser.error("P2 result/config SHA-256 mismatch")
    candidate = select_candidate(
        result, args.parent_seed, args.small_scale_seed)
    parent_seed = int(candidate["parent_seed"])
    small_seed = int(candidate["small_scale_seed"])
    pair = candidate["best_pair"]
    center = np.asarray(pair["midpoint_mpc_h"], dtype=np.float64)

    parent_rows = [
        row for row in p1["members"]
        if row["pass"] and int(row["seed"]) == parent_seed
    ]
    if len(parent_rows) != 1:
        parser.error(f"cannot resolve one passing P1 field for seed {parent_seed}")
    parent_path = Path(parent_rows[0]["input"])
    with np.load(parent_path) as data:
        coarse = data["s_out"].astype(np.float64)
        box = float(data["L"])
        cosmology = {
            "Om": float(data["Om"]),
            "Ob": float(data["Ob"]),
            "h": float(data["hh"]),
            "A_s_1e9": float(data["A_s_1e9"]),
            "ns": float(data["ns"]),
        }
    n = int(config["screen"]["mesh_size"])
    spacing = box / n
    if not np.isclose(spacing, config["screen"]["particle_spacing_mpc_h"]):
        parser.error("configuration spacing does not match its mesh and box")

    import jax.numpy as jnp
    from cf4_make_ic import embed_ic_projected
    from mock_pipeline import make_forward

    print(
        f"[trace] repeat parent={parent_seed} small={small_seed}: "
        f"{coarse.shape[0]}^3 -> {n}^3, center={center.tolist()}, "
        f"R={args.radius_mpc_h} Mpc/h",
        flush=True,
    )
    canonical_n = int(config["screen"].get("canonical_mesh_size", n))
    proposal_manifest_value = result.get("conditioned_proposal_manifest")
    proposal_path = None
    proposal_hash = None
    if proposal_manifest_value is not None:
        proposal_manifest_path = Path(proposal_manifest_value)
        actual_manifest_hash = sha256_file(proposal_manifest_path)
        if actual_manifest_hash != result.get(
            "conditioned_proposal_manifest_sha256"):
            parser.error("conditioned proposal manifest/result SHA-256 mismatch")
        proposal_manifest = json.loads(proposal_manifest_path.read_text())
        matches = [
            entry for entry in proposal_manifest.get("entries", [])
            if int(entry["parent_seed"]) == parent_seed
            and int(entry["proposal_seed"]) == small_seed
        ]
        if len(matches) != 1:
            parser.error("cannot resolve one conditioned field for the P2 candidate")
        proposal_path = Path(matches[0]["field"])
        proposal_hash = sha256_file(proposal_path)
        if proposal_hash != matches[0]["field_sha256"]:
            parser.error("conditioned proposal field SHA-256 mismatch")
        with np.load(proposal_path, allow_pickle=False) as data:
            initial = data["s_conditioned"].astype(np.float32)
        if initial.shape != (n, n, n):
            parser.error("conditioned proposal mesh does not match the P2 mesh")
    else:
        initial = embed_ic_projected(coarse, canonical_n, n, small_seed)
    _, _, forward = make_forward(
        n, spacing, jnp.float32, return_dens=False, cosmology=cosmology,
        return_particle_positions=True)
    final_position_array = forward(jnp.asarray(initial))
    ids, lagrangian, final = trace_position_array(
        final_position_array, center, args.radius_mpc_h,
        mesh_size=n, spacing=spacing, box_size=box)
    metadata = {
        "schema": SCHEMA,
        "p2_result": str(args.p2_result.resolve()),
        "p2_result_sha256": sha256_file(args.p2_result),
        "p2_config": str(args.config.resolve()),
        "p2_config_sha256": sha256_file(args.config),
        "p1_result": str(args.p1_result.resolve()),
        "parent_field": str(parent_path.resolve()),
        "parent_field_sha256": sha256_file(parent_path),
        "conditioned_proposal": (
            str(proposal_path.resolve()) if proposal_path else None),
        "conditioned_proposal_sha256": proposal_hash,
        "parent_seed": parent_seed,
        "small_scale_seed": small_seed,
        "mesh_size": n,
        "particle_spacing_mpc_h": spacing,
        "box_size_mpc_h": box,
        "z0_center_mpc_h": center.tolist(),
        "z0_radius_mpc_h": args.radius_mpc_h,
        "n_particle_ids": int(len(ids)),
        "selection": "periodic PM z=0 sphere around the frozen P2 pair midpoint",
        "scope": "Lagrangian mask construction only; definitive halo gate is RAMSES/HOP",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        schema=np.array(SCHEMA),
        lagrangian=lagrangian.astype(np.float32),
        final_positions=final.astype(np.float32),
        particle_ids=ids,
        L=np.float64(box),
        z0_center_mpc_h=center,
        z0_radius_mpc_h=np.float64(args.radius_mpc_h),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
