#!/usr/bin/env python3
"""Trace GalaxyFinder member IDs into a RAMSES initial snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(fh) -> tuple[int, int]:
    raw = fh.read(4)
    if len(raw) != 4:
        raise EOFError("missing Fortran record marker")
    size = struct.unpack("<i", raw)[0]
    offset = fh.tell()
    fh.seek(size, 1)
    end = struct.unpack("<i", fh.read(4))[0]
    if end != size:
        raise RuntimeError(f"Fortran record mismatch: {size} != {end}")
    return offset, size


def scalar_int(path: Path) -> int:
    with path.open("rb") as fh:
        offset, size = record(fh)
    if size != 4:
        raise RuntimeError(f"expected scalar integer record in {path}")
    return int(np.memmap(path, dtype="<i4", mode="r", offset=offset, shape=(1,))[0])


def periodic_summary(position: np.ndarray, box: float) -> dict:
    angle = 2.0 * np.pi * position / box
    centre = np.mod(np.arctan2(np.sin(angle).mean(axis=0), np.cos(angle).mean(axis=0)), 2.0 * np.pi)
    centre = centre * box / (2.0 * np.pi)
    offset = position - centre
    offset -= box * np.rint(offset / box)
    return {
        "count": int(position.shape[0]),
        "periodic_centre_mpc_h": centre.tolist(),
        "minimum_offset_mpc_h": offset.min(axis=0).tolist(),
        "maximum_offset_mpc_h": offset.max(axis=0).tolist(),
        "span_mpc_h": (offset.max(axis=0) - offset.min(axis=0)).tolist(),
        "maximum_radius_mpc_h": float(np.linalg.norm(offset, axis=1).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--members", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions-output", type=Path, required=True)
    parser.add_argument("--box", type=float, default=384.0)
    parser.add_argument(
        "--selection-status",
        default="unspecified",
        choices=("unspecified", "validated-parent", "trace-only-zoom-candidate"),
    )
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    args = parser.parse_args()

    with np.load(args.members, allow_pickle=False) as data:
        names = [name for name in ("member_ids_i", "member_ids_j", "exclusion_member_ids") if name in data]
        members = {name: np.asarray(data[name], dtype=np.int64) for name in names}
    all_ids = np.unique(np.concatenate(list(members.values())))
    owner = np.full(all_ids.size, -1, dtype=np.int8)
    for number, name in enumerate(names):
        locations = np.searchsorted(all_ids, members[name])
        owner[locations] = number
    if np.any(owner < 0):
        raise RuntimeError("failed to classify requested particle IDs")

    first = args.snapshot / f"part_{args.snapshot.name[-5:]}.out00001"
    ncpu = scalar_int(first)
    matched_ids: list[np.ndarray] = []
    matched_owner: list[np.ndarray] = []
    matched_position: list[np.ndarray] = []
    for cpu in range(1, ncpu + 1):
        path = args.snapshot / f"part_{args.snapshot.name[-5:]}.out{cpu:05d}"
        with path.open("rb") as fh:
            headers = [record(fh) for _ in range(8)]
            npart_offset, npart_bytes = headers[2]
            if npart_bytes != 4:
                raise RuntimeError(f"bad particle-count record in {path}")
            npart = int(np.memmap(path, dtype="<i4", mode="r", offset=npart_offset, shape=(1,))[0])
            vectors = [record(fh) for _ in range(7)]
            id_offset, id_bytes = record(fh)
        expected_vector_bytes = npart * 8
        if any(size != expected_vector_bytes for _, size in vectors):
            raise RuntimeError(f"unexpected vector width in {path}")
        if id_bytes == npart * 4:
            id_dtype = "<i4"
        elif id_bytes == npart * 8:
            id_dtype = "<i8"
        else:
            raise RuntimeError(f"unexpected ID width in {path}: {id_bytes}")
        ids = np.memmap(path, dtype=id_dtype, mode="r", offset=id_offset, shape=(npart,))
        selected = np.isin(ids, all_ids, assume_unique=False)
        if not np.any(selected):
            continue
        selected_ids = np.asarray(ids[selected], dtype=np.int64)
        location = np.searchsorted(all_ids, selected_ids)
        xyz = []
        for offset, _ in vectors[:3]:
            values = np.memmap(path, dtype="<f8", mode="r", offset=offset, shape=(npart,))
            xyz.append(np.asarray(values[selected], dtype=np.float64))
        matched_ids.append(selected_ids)
        matched_owner.append(owner[location])
        matched_position.append(np.column_stack(xyz) * float(args.box))

    found_ids = np.concatenate(matched_ids) if matched_ids else np.empty(0, np.int64)
    found_owner = np.concatenate(matched_owner) if matched_owner else np.empty(0, np.int8)
    found_position = np.concatenate(matched_position) if matched_position else np.empty((0, 3), np.float64)
    if found_ids.size != all_ids.size or np.unique(found_ids).size != all_ids.size:
        missing = np.setdiff1d(all_ids, found_ids)
        raise RuntimeError(f"matched {found_ids.size}/{all_ids.size} unique IDs; missing={missing[:20].tolist()}")

    order = np.argsort(found_ids)
    found_ids = found_ids[order]
    found_owner = found_owner[order]
    found_position = found_position[order]
    summaries = {
        name: periodic_summary(found_position[found_owner == number], float(args.box))
        for number, name in enumerate(names)
    }
    lg_position = found_position[found_owner <= 1]
    summaries["lg_combined"] = periodic_summary(lg_position, float(args.box))
    result = {
        "schema": "ouruniv-cf4-ramses-member-id-lagrangian-trace-v2",
        "snapshot": str(args.snapshot),
        "members": str(args.members),
        "members_sha256": sha256(args.members),
        "box_mpc_h": float(args.box),
        "selection_status": args.selection_status,
        "evidence": [
            {"path": str(path), "sha256": sha256(path)} for path in args.evidence
        ],
        "requested_unique_ids": int(all_ids.size),
        "matched_unique_ids": int(np.unique(found_ids).size),
        "summaries": summaries,
        "m33_resolved": False,
        "interpretation": (
            "Initial-snapshot positions for a trace-only zoom candidate. "
            "The buffered zoom mask must enclose the combined LG footprint and its environment; "
            "this trace does not promote the L9 parent or validate MW/M31/M33 structure."
            if args.selection_status == "trace-only-zoom-candidate"
            else "Initial-snapshot positions; a zoom mask must include padding around the combined LG footprint."
        ),
    }
    args.positions_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.positions_output,
        particle_id=found_ids,
        component=found_owner,
        position_mpc_h=found_position,
        component_names=np.asarray(names),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"matched": int(found_ids.size), "lg_span": summaries["lg_combined"]["span_mpc_h"]}))


if __name__ == "__main__":
    main()
