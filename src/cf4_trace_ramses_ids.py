#!/usr/bin/env python3
"""Trace a z=0 RAMSES target sphere to an initial particle point cloud.

Both snapshots must use the custom particle layout documented in
``cf4_zoom_z0_gate.py`` and must preserve the same int64 particle IDs.  The
initial snapshot should be written immediately after IC loading
(``aexp <= 0.03`` by default).  Its particle positions are used as the
Lagrangian-mask point cloud; the downstream voxeliser adds a physical buffer
that safely covers the small start-time displacement.

The default target is an Eulerian sphere around an already selected LG
midpoint.  A 3--5 Mpc/h radius includes MW, M31, M33 and their immediate tidal
environment.  This script never assumes that the target is at the box centre.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cf4_zoom_z0_gate import (
    _record,
    _skip_header,
    min_image,
    particle_files,
    read_info,
)


SCHEMA = "ouruniv-ramses-id-trace-v1"


def _file_set_digest(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def _read_positions_and_ids(path: Path, box_mpc_h: float) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        _, _ = _skip_header(handle)
        pos = np.stack([_record(handle, "<f8") for _ in range(3)], axis=1)
        for _ in range(3):
            _record(handle)  # velocity
        _record(handle)      # mass
        ids = _record(handle, "<i8")
    return pos * box_mpc_h, ids


def select_ids(final_files: list[Path], center: np.ndarray, radius: float,
               box_mpc_h: float) -> np.ndarray:
    chunks = []
    r2 = radius * radius
    for index, path in enumerate(final_files, 1):
        pos, ids = _read_positions_and_ids(path, box_mpc_h)
        dr = min_image(pos - center, box_mpc_h)
        keep = np.einsum("ij,ij->i", dr, dr) <= r2
        if np.any(keep):
            chunks.append(ids[keep])
        if index % 8 == 0:
            print(f"[select] {index}/{len(final_files)} particle files", flush=True)
    if not chunks:
        raise RuntimeError("no z=0 particles found in the requested target sphere")
    selected = np.unique(np.concatenate(chunks))
    if np.any(selected <= 0):
        raise RuntimeError("selected particle IDs must be positive")
    return selected


def trace_ids(initial_files: list[Path], selected: np.ndarray,
              box_mpc_h: float) -> tuple[np.ndarray, np.ndarray]:
    selected = np.sort(np.asarray(selected, dtype=np.int64))
    positions = []
    found_ids = []
    for index, path in enumerate(initial_files, 1):
        pos, ids = _read_positions_and_ids(path, box_mpc_h)
        where = np.searchsorted(selected, ids)
        valid = where < len(selected)
        keep = np.zeros(len(ids), dtype=bool)
        keep[valid] = selected[where[valid]] == ids[valid]
        if np.any(keep):
            positions.append(pos[keep])
            found_ids.append(ids[keep])
        if index % 8 == 0:
            print(f"[trace] {index}/{len(initial_files)} particle files", flush=True)
    if not positions:
        raise RuntimeError("none of the selected z=0 IDs exist in the initial snapshot")
    pos = np.concatenate(positions)
    ids = np.concatenate(found_ids)
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("duplicate particle IDs found in the initial snapshot")
    missing = np.setdiff1d(selected, ids, assume_unique=True)
    if len(missing):
        preview = ",".join(map(str, missing[:8]))
        raise RuntimeError(f"{len(missing)} selected IDs are missing initially: {preview}")
    return pos, ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-output", type=Path, required=True)
    parser.add_argument("--initial-output", type=Path, required=True)
    parser.add_argument("--center-mpc-h", type=float, nargs=3, required=True)
    parser.add_argument("--radius-mpc-h", type=float, default=5.0)
    parser.add_argument("--box-mpc-h", type=float, default=384.0)
    parser.add_argument("--maximum-initial-aexp", type=float, default=0.03)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    initial_info = read_info(args.initial_output)
    final_info = read_info(args.final_output)
    if initial_info["aexp"] > args.maximum_initial_aexp:
        parser.error(
            f"initial aexp={initial_info['aexp']:.5f} exceeds "
            f"--maximum-initial-aexp={args.maximum_initial_aexp}; write an IC snapshot")
    if final_info["aexp"] < 0.95:
        parser.error(f"final snapshot aexp={final_info['aexp']:.5f} is not z=0")

    center = np.mod(np.asarray(args.center_mpc_h, dtype=np.float64), args.box_mpc_h)
    final_files = particle_files(args.final_output)
    initial_files = particle_files(args.initial_output)
    selected = select_ids(
        final_files, center, args.radius_mpc_h, args.box_mpc_h)
    print(f"[select] {len(selected):,} unique z=0 particle IDs", flush=True)
    lagrangian, found_ids = trace_ids(initial_files, selected, args.box_mpc_h)
    print(f"[trace] recovered {len(found_ids):,}/{len(selected):,} IDs", flush=True)

    order = np.argsort(found_ids)
    lagrangian = lagrangian[order]
    found_ids = found_ids[order]
    metadata = {
        "schema": SCHEMA,
        "initial_output": str(args.initial_output.resolve()),
        "final_output": str(args.final_output.resolve()),
        "initial_particle_file_set_digest": _file_set_digest(initial_files),
        "final_particle_file_set_digest": _file_set_digest(final_files),
        "initial_aexp": initial_info["aexp"],
        "final_aexp": final_info["aexp"],
        "box_size_mpc_h": args.box_mpc_h,
        "z0_center_mpc_h": center.tolist(),
        "z0_radius_mpc_h": args.radius_mpc_h,
        "n_particle_ids": int(len(found_ids)),
        "position_definition": "RAMSES positions in the initial output; buffered downstream",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        schema=np.array(SCHEMA),
        lagrangian=lagrangian.astype(np.float32),
        particle_ids=found_ids,
        L=np.float64(args.box_mpc_h),
        initial_aexp=np.float64(initial_info["aexp"]),
        final_aexp=np.float64(final_info["aexp"]),
        z0_center_mpc_h=center,
        z0_radius_mpc_h=np.float64(args.radius_mpc_h),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
