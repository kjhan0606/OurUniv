#!/usr/bin/env python3
"""Trace a z=0 RAMSES sphere to an initial snapshot and build a zoom mask.

This is intentionally a spatial environment selection, not a halo-member
selection.  It keeps every parent particle inside the requested periodic z=0
sphere, traces stable particle IDs to the initial snapshot, and voxelises the
initial positions into the sparse mask consumed by ``cf4_zoom_ic2.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cf4_lagrangian_mask import SCHEMA, build_mask
try:
    from scripts.cf4_trace_ramses_member_ids import periodic_summary, record, scalar_int, sha256
except ModuleNotFoundError:  # Direct execution with scripts/ on PYTHONPATH.
    from cf4_trace_ramses_member_ids import periodic_summary, record, scalar_int, sha256


def particle_layout(path: Path) -> tuple[int, list[tuple[int, int]], int, str]:
    """Return particle count, seven vector records, ID offset and ID dtype."""
    with path.open("rb") as stream:
        headers = [record(stream) for _ in range(8)]
        count_offset, count_bytes = headers[2]
        if count_bytes != 4:
            raise RuntimeError(f"bad particle-count record in {path}")
        count = int(np.memmap(
            path, dtype="<i4", mode="r", offset=count_offset, shape=(1,)
        )[0])
        vectors = [record(stream) for _ in range(7)]
        id_offset, id_bytes = record(stream)
    expected = count * 8
    if any(size != expected for _, size in vectors):
        raise RuntimeError(f"unexpected vector width in {path}")
    if id_bytes == count * 4:
        id_dtype = "<i4"
    elif id_bytes == count * 8:
        id_dtype = "<i8"
    else:
        raise RuntimeError(f"unexpected ID width in {path}: {id_bytes}")
    return count, vectors, id_offset, id_dtype


def particle_files(snapshot: Path) -> list[Path]:
    number = snapshot.name.rsplit("_", 1)[-1]
    first = snapshot / f"part_{number}.out00001"
    ncpu = scalar_int(first)
    paths = [snapshot / f"part_{number}.out{cpu:05d}" for cpu in range(1, ncpu + 1)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing particle files: {missing[:3]}")
    return paths


def periodic_radius_mask(position: np.ndarray, centre: np.ndarray,
                         box: float, radius: float) -> np.ndarray:
    delta = np.asarray(position, dtype=np.float64) - np.asarray(centre, dtype=np.float64)
    delta -= box * np.rint(delta / box)
    return np.einsum("ij,ij->i", delta, delta) <= radius * radius


def select_z0_sphere(snapshot: Path, centre: np.ndarray, box: float,
                     radius: float) -> tuple[np.ndarray, np.ndarray]:
    selected_ids: list[np.ndarray] = []
    selected_position: list[np.ndarray] = []
    for path in particle_files(snapshot):
        count, vectors, id_offset, id_dtype = particle_layout(path)
        xyz = np.column_stack([
            np.memmap(path, dtype="<f8", mode="r", offset=offset, shape=(count,))
            for offset, _ in vectors[:3]
        ]) * box
        keep = periodic_radius_mask(xyz, centre, box, radius)
        if np.any(keep):
            ids = np.memmap(path, dtype=id_dtype, mode="r", offset=id_offset, shape=(count,))
            selected_ids.append(np.asarray(ids[keep], dtype=np.int64))
            selected_position.append(np.asarray(xyz[keep], dtype=np.float64))
    if not selected_ids:
        raise RuntimeError("z=0 spatial selection is empty")
    ids = np.concatenate(selected_ids)
    position = np.concatenate(selected_position)
    if np.unique(ids).size != ids.size:
        raise RuntimeError("z=0 spatial selection contains duplicate particle IDs")
    return ids, position


def trace_ids(snapshot: Path, requested: np.ndarray, box: float) -> tuple[np.ndarray, np.ndarray]:
    requested = np.sort(np.asarray(requested, dtype=np.int64))
    matched_ids: list[np.ndarray] = []
    matched_position: list[np.ndarray] = []
    for path in particle_files(snapshot):
        count, vectors, id_offset, id_dtype = particle_layout(path)
        ids = np.memmap(path, dtype=id_dtype, mode="r", offset=id_offset, shape=(count,))
        keep = np.isin(ids, requested, assume_unique=False)
        if not np.any(keep):
            continue
        found = np.asarray(ids[keep], dtype=np.int64)
        xyz = np.column_stack([
            np.asarray(
                np.memmap(path, dtype="<f8", mode="r", offset=offset, shape=(count,))[keep],
                dtype=np.float64,
            )
            for offset, _ in vectors[:3]
        ]) * box
        matched_ids.append(found)
        matched_position.append(xyz)
    if not matched_ids:
        raise RuntimeError("no requested IDs were found in the initial snapshot")
    ids = np.concatenate(matched_ids)
    position = np.concatenate(matched_position)
    if ids.size != requested.size or np.unique(ids).size != requested.size:
        missing = np.setdiff1d(requested, ids)
        raise RuntimeError(
            f"matched {ids.size}/{requested.size} IDs; missing={missing[:20].tolist()}"
        )
    order = np.argsort(ids)
    return ids[order], position[order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z0-snapshot", type=Path, required=True)
    parser.add_argument("--initial-snapshot", type=Path, required=True)
    parser.add_argument("--selection-evidence", type=Path, required=True)
    parser.add_argument("--members", type=Path, required=True)
    parser.add_argument("--positions-output", type=Path, required=True)
    parser.add_argument("--mask-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--box", type=float, default=384.0)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--base-level", type=int, default=9)
    parser.add_argument("--buffer", type=float, default=1.5)
    parser.add_argument("--subbox-pad-base-cells", type=int, default=8)
    args = parser.parse_args()

    evidence = json.loads(args.selection_evidence.read_text())
    if evidence.get("decision") != "PARENT_RAMSES_HALO_PASS":
        raise RuntimeError("selection evidence is not a GalaxyFinder parent pass")
    selected = evidence.get("lg", {}).get("selected")
    if selected is None:
        raise RuntimeError("selection evidence has no selected LG pair")
    centre = np.asarray(selected["midpoint_mpc_h"], dtype=np.float64)
    if centre.shape != (3,) or not np.isfinite(centre).all():
        raise RuntimeError("invalid LG midpoint in selection evidence")

    z0_ids, z0_position = select_z0_sphere(
        args.z0_snapshot, centre, args.box, args.radius
    )
    with np.load(args.members, allow_pickle=False) as member_data:
        pair_ids = np.unique(np.concatenate([
            np.asarray(member_data["member_ids_i"], dtype=np.int64),
            np.asarray(member_data["member_ids_j"], dtype=np.int64),
        ]))
    missing_pair = np.setdiff1d(pair_ids, z0_ids)
    if missing_pair.size:
        raise RuntimeError(
            f"z=0 sphere excludes {missing_pair.size} selected-pair member IDs"
        )

    initial_ids, initial_position = trace_ids(args.initial_snapshot, z0_ids, args.box)
    z0_order = np.argsort(z0_ids)
    z0_ids = z0_ids[z0_order]
    z0_position = z0_position[z0_order]
    if not np.array_equal(z0_ids, initial_ids):
        raise RuntimeError("z=0 and initial sorted particle IDs differ")

    mask = build_mask(
        initial_position,
        args.box,
        base_level=args.base_level,
        buffer_mpc_h=args.buffer,
        subbox_pad_base_cells=args.subbox_pad_base_cells,
    )
    if mask["requires_shift"] or mask["lo"] is None or mask["hi"] is None:
        raise RuntimeError("Lagrangian mask crosses a periodic boundary; origin shift required")

    args.positions_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.positions_output,
        particle_id=initial_ids,
        initial_position_mpc_h=initial_position,
        z0_position_mpc_h=z0_position,
        z0_centre_mpc_h=centre,
        z0_radius_mpc_h=np.float64(args.radius),
        box_size_mpc_h=np.float64(args.box),
    )
    metadata = {
        "schema": SCHEMA,
        "selection_status": "trace-only-zoom-candidate",
        "source_positions": str(args.positions_output),
        "z0_snapshot": str(args.z0_snapshot),
        "initial_snapshot": str(args.initial_snapshot),
        "selection_evidence": str(args.selection_evidence),
        "selection_evidence_sha256": sha256(args.selection_evidence),
        "members": str(args.members),
        "members_sha256": sha256(args.members),
        "z0_radius_mpc_h": args.radius,
        "n_traced_points": int(initial_ids.size),
        "n_occupied_base_cells": int(mask["cells"].shape[0]),
        "buffer_mpc_h": args.buffer,
        "subbox_pad_base_cells": args.subbox_pad_base_cells,
    }
    np.savez_compressed(
        args.mask_output,
        schema=np.array(SCHEMA),
        box_size_mpc_h=np.float64(args.box),
        base_level=np.int32(args.base_level),
        base_cells=mask["cells"].astype(np.int16),
        periodic_center_mpc_h=mask["center"],
        subbox_lo_base=mask["lo"],
        subbox_hi_base=mask["hi"],
        buffer_mpc_h=np.float64(args.buffer),
        requires_periodic_origin_shift=np.bool_(False),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )

    side_base = int((mask["hi"] - mask["lo"])[0])
    levels = {}
    for level in range(args.base_level, 14):
        side = side_base * 2 ** (level - args.base_level)
        levels[str(level)] = {
            "side_cells": side,
            "cell_spacing_mpc_h": args.box / 2 ** level,
            "five_float32_fields_gib": 5.0 * side ** 3 * 4.0 / 2 ** 30,
        }
    report = {
        "schema": "ouruniv-cf4-spatial-lagrangian-trace-v1",
        "stage": "8/8 trace-only zoom preparation",
        "selection_status": "trace-only-zoom-candidate",
        "parent_promoted": False,
        "m33_resolved": False,
        "z0_selection": {
            "centre_mpc_h": centre.tolist(),
            "radius_mpc_h": args.radius,
            "particle_count": int(z0_ids.size),
            "summary": periodic_summary(z0_position, args.box),
            "selected_pair_member_count": int(pair_ids.size),
            "selected_pair_members_retained": int(pair_ids.size - missing_pair.size),
        },
        "initial_trace": periodic_summary(initial_position, args.box),
        "mask": {
            "path": str(args.mask_output),
            "base_level": args.base_level,
            "occupied_base_cells": int(mask["cells"].shape[0]),
            "buffer_mpc_h": args.buffer,
            "subbox_pad_base_cells": args.subbox_pad_base_cells,
            "subbox_lo_base": mask["lo"].tolist(),
            "subbox_hi_base": mask["hi"].tolist(),
            "level_geometry": levels,
        },
        "interpretation": (
            "All parent particles in a 5 cMpc/h z=0 LG-environment sphere were traced "
            "to the initial state. This is an engineering mask for a trace-only candidate, "
            "not parent validation, M33 identification, or a production zoom promotion."
        ),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "selected": int(z0_ids.size),
        "initial_span_mpc_h": report["initial_trace"]["span_mpc_h"],
        "subbox_side_base_cells": side_base,
        "mask": str(args.mask_output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
