#!/usr/bin/env python
"""Build a sparse, periodic Lagrangian refinement mask from traced particles.

The input must contain the *initial* positions, in Mpc/h, of particles selected
from the target environment at z=0.  The output stores occupied cells on the
global base grid rather than a large dense cube.  ``cf4_zoom_ic2.py`` expands
this sparse mask onto every nested IC level.

This module deliberately does not infer a Local-Group mask from a present-day
coordinate or from the box centre.  Stable particle IDs must first be traced
from the z=0 target back to the initial snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCHEMA = "ouruniv-lagrangian-mask-v1"


def sha256_file(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def periodic_center(points: np.ndarray, box_size: float) -> np.ndarray:
    """Circular mean in each periodic coordinate."""
    angle = 2.0 * np.pi * np.asarray(points, dtype=np.float64) / box_size
    mean = np.arctan2(np.sin(angle).mean(axis=0), np.cos(angle).mean(axis=0))
    return np.mod(mean, 2.0 * np.pi) * box_size / (2.0 * np.pi)


def _dilation_offsets(buffer_mpc_h: float, dx: float) -> np.ndarray:
    """Conservative cell-centre offsets for a physical buffer.

    One cell diagonal is included so that a particle near a cell face cannot
    lose the requested buffer after voxelisation.
    """
    radius = buffer_mpc_h / dx + math.sqrt(3.0)
    imax = int(math.ceil(radius))
    q = np.arange(-imax, imax + 1, dtype=np.int32)
    off = np.stack(np.meshgrid(q, q, q, indexing="ij"), axis=-1).reshape(-1, 3)
    return off[np.linalg.norm(off, axis=1) <= radius]


def voxelise(points: np.ndarray, box_size: float, base_level: int,
             buffer_mpc_h: float) -> np.ndarray:
    """Return unique periodic base-grid cells containing the buffered points."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError("lagrangian points must have shape (n>0, 3)")
    if not np.isfinite(points).all():
        raise ValueError("lagrangian points contain NaN or infinity")
    nbase = 2 ** base_level
    dx = box_size / nbase
    cells = np.floor(np.mod(points, box_size) / dx).astype(np.int32)
    cells = np.unique(cells, axis=0)
    offsets = _dilation_offsets(buffer_mpc_h, dx)

    # Process in bounded blocks: a typical LG trace has only a few thousand
    # unique L9 cells, but the Cartesian dilation can still be large.
    expanded = []
    block = max(1, 2_000_000 // max(1, len(offsets)))
    for first in range(0, len(cells), block):
        e = cells[first:first + block, None, :] + offsets[None, :, :]
        expanded.append(np.mod(e.reshape(-1, 3), nbase).astype(np.int32))
    return np.unique(np.concatenate(expanded, axis=0), axis=0)


def _minimal_periodic_interval(values: np.ndarray, size: int) -> tuple[int, int, bool]:
    """Smallest integer-cell interval; ``wraps`` means it crosses index zero."""
    values = np.unique(np.asarray(values, dtype=np.int64))
    if len(values) == size:
        return 0, size, False
    nxt = np.roll(values, -1)
    gaps = np.mod(nxt - values, size)
    cut = int(np.argmax(gaps))
    start = int(nxt[cut])
    stop_inclusive = int(values[cut])
    wraps = start > stop_inclusive
    if wraps:
        return start, stop_inclusive + 1, True
    return start, stop_inclusive + 1, False


def recommended_cube(cells: np.ndarray, nbase: int,
                     pad_cells: int) -> tuple[np.ndarray | None, np.ndarray | None, bool]:
    """Non-wrapping cubic base-grid subbox enclosing a sparse mask.

    GRAFIC subvolumes in the current generator cannot straddle a periodic box
    face.  Such a mask is valid scientifically but requires a periodic origin
    shift before IC export, which is reported explicitly.
    """
    intervals = [_minimal_periodic_interval(cells[:, axis], nbase) for axis in range(3)]
    if any(item[2] for item in intervals):
        return None, None, True
    lo0 = np.array([item[0] for item in intervals], dtype=np.int64)
    hi0 = np.array([item[1] for item in intervals], dtype=np.int64)
    side = int(np.max(hi0 - lo0) + 2 * pad_cells)
    center = 0.5 * (lo0 + hi0)
    lo = np.floor(center - 0.5 * side).astype(np.int64)
    hi = lo + side
    for axis in range(3):
        if lo[axis] < 0:
            hi[axis] -= lo[axis]
            lo[axis] = 0
        if hi[axis] > nbase:
            lo[axis] -= hi[axis] - nbase
            hi[axis] = nbase
    if np.any(lo < 0) or np.any(hi > nbase):
        return None, None, True
    return lo, hi, False


def build_mask(points: np.ndarray, box_size: float, base_level: int,
               buffer_mpc_h: float, subbox_pad_base_cells: int) -> dict:
    cells = voxelise(points, box_size, base_level, buffer_mpc_h)
    nbase = 2 ** base_level
    lo, hi, requires_shift = recommended_cube(
        cells, nbase=nbase, pad_cells=subbox_pad_base_cells)
    return {
        "points": np.asarray(points, dtype=np.float64),
        "cells": cells,
        "center": periodic_center(points, box_size),
        "lo": lo,
        "hi": hi,
        "requires_shift": requires_shift,
    }


def load_sparse_mask(path: str | Path, expected_box: float | None = None,
                     expected_base_level: int | None = None) -> dict:
    """Load and validate a mask consumed by ``cf4_zoom_ic2.py``."""
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["schema"].item())
        if schema != SCHEMA:
            raise ValueError(f"unsupported mask schema {schema!r}")
        box = float(data["box_size_mpc_h"])
        level = int(data["base_level"])
        cells = np.asarray(data["base_cells"], dtype=np.int32)
        requires_shift = bool(data["requires_periodic_origin_shift"])
        lo = np.asarray(data["subbox_lo_base"], dtype=np.int64)
        hi = np.asarray(data["subbox_hi_base"], dtype=np.int64)
    if expected_box is not None and not np.isclose(box, expected_box):
        raise ValueError(f"mask box {box} != IC box {expected_box}")
    if expected_base_level is not None and level != expected_base_level:
        raise ValueError(f"mask base level {level} != IC levelmin {expected_base_level}")
    if requires_shift or np.any(lo < 0) or np.any(hi <= lo):
        raise ValueError(
            "mask crosses a periodic face; shift the parent origin and rebuild the mask")
    nbase = 2 ** level
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("base_cells must have shape (n,3)")
    if np.any(cells < 0) or np.any(cells >= nbase):
        raise ValueError("base_cells contain out-of-range indices")
    if not np.all(hi - lo == hi[0] - lo[0]):
        raise ValueError("mask subbox must be cubic")
    grid = np.zeros((nbase, nbase, nbase), dtype=bool)
    grid[cells[:, 0], cells[:, 1], cells[:, 2]] = True
    return {
        "box_size_mpc_h": box,
        "base_level": level,
        "base_cells": cells,
        "base_grid": grid,
        "subbox_lo_base": lo,
        "subbox_hi_base": hi,
    }


def level_bounds(mask: dict, level: int) -> tuple[np.ndarray, np.ndarray]:
    scale = 2 ** (level - mask["base_level"])
    return mask["subbox_lo_base"] * scale, mask["subbox_hi_base"] * scale


def refmap_for_level(mask: dict, level: int, lo: np.ndarray,
                     hi: np.ndarray) -> np.ndarray:
    """Expand the sparse base mask to a nested IC subvolume."""
    scale = 2 ** (level - mask["base_level"])
    axes = [np.arange(lo[a], hi[a], dtype=np.int64) // scale for a in range(3)]
    return mask["base_grid"][np.ix_(*axes)].astype(np.float32)


def _self_test() -> None:
    box = 32.0
    points = np.array([[15.8, 16.1, 15.9], [16.4, 15.7, 16.2]])
    built = build_mask(points, box, base_level=4, buffer_mpc_h=1.0,
                       subbox_pad_base_cells=1)
    assert len(built["cells"]) > 2
    assert not built["requires_shift"]
    assert np.all(built["hi"] - built["lo"] == built["hi"][0] - built["lo"][0])

    edge = np.array([[31.8, 10.0, 10.0], [0.2, 10.0, 10.0]])
    wrapped = build_mask(edge, box, base_level=4, buffer_mpc_h=0.0,
                         subbox_pad_base_cells=1)
    assert wrapped["requires_shift"]
    print("cf4_lagrangian_mask self-test PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="NPZ containing traced initial positions")
    parser.add_argument("--key", default="lagrangian")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--box-mpc-h", type=float, default=384.0)
    parser.add_argument("--base-level", type=int, default=9)
    parser.add_argument("--buffer-mpc-h", type=float, default=1.5)
    parser.add_argument("--subbox-pad-base-cells", type=int, default=2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.input is None or args.out is None:
        parser.error("--input and --out are required unless --self-test is used")

    with np.load(args.input, allow_pickle=False) as source:
        if args.key not in source:
            parser.error(f"{args.input} has no {args.key!r} array")
        points = np.asarray(source[args.key], dtype=np.float64)
        if "L" in source and not np.isclose(float(source["L"]), args.box_mpc_h):
            parser.error(f"input L={float(source['L'])} differs from --box-mpc-h")

    result = build_mask(
        points, args.box_mpc_h, args.base_level, args.buffer_mpc_h,
        args.subbox_pad_base_cells)
    lo = np.full(3, -1, dtype=np.int64) if result["lo"] is None else result["lo"]
    hi = np.full(3, -1, dtype=np.int64) if result["hi"] is None else result["hi"]
    metadata = {
        "schema": SCHEMA,
        "source": str(args.input.resolve()),
        "source_sha256": sha256_file(args.input),
        "source_key": args.key,
        "n_traced_points": int(len(points)),
        "n_occupied_base_cells": int(len(result["cells"])),
        "box_size_mpc_h": args.box_mpc_h,
        "base_level": args.base_level,
        "buffer_mpc_h": args.buffer_mpc_h,
        "subbox_pad_base_cells": args.subbox_pad_base_cells,
        "requires_periodic_origin_shift": bool(result["requires_shift"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        schema=np.array(SCHEMA),
        box_size_mpc_h=np.float64(args.box_mpc_h),
        base_level=np.int32(args.base_level),
        base_cells=result["cells"].astype(np.int16),
        periodic_center_mpc_h=result["center"],
        subbox_lo_base=lo,
        subbox_hi_base=hi,
        buffer_mpc_h=np.float64(args.buffer_mpc_h),
        requires_periodic_origin_shift=np.bool_(result["requires_shift"]),
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"subbox base cells: lo={lo.tolist()} hi={hi.tolist()}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
