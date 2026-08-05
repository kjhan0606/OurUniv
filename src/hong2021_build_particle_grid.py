#!/usr/bin/env python
"""Build a validated periodic dark-matter grid from HDF5 snapshots.

This is the common V14 target operator for TNG and raw CAMELS suites.  It
streams only ``PartType1/Coordinates``, applies the tested NGP or cell-centred
CIC assignment, verifies equal dark-matter particle masses and exact particle
conservation, then atomically publishes a cache and its provenance record.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from hong2021_camels_density_audit import deposit_particle_counts


def numerical_snapshot_key(path: Path) -> tuple[str, int]:
    fields = path.name.split(".")
    try:
        index = int(fields[-2])
    except (IndexError, ValueError):
        index = 0
    return path.name, index


def expand_sources(pattern: str) -> list[Path]:
    paths = [Path(value) for value in glob.glob(pattern)]
    if not paths:
        raise FileNotFoundError(f"snapshot pattern matched no files: {pattern}")
    # TNG uses snap_099.0.hdf5 ... snap_099.447.hdf5.  Single-file CAMELS
    # snapshots have no numeric piece and remain unaffected.
    paths.sort(key=lambda path: (numerical_snapshot_key(path)[1], path.name))
    return paths


def _header_total(header: h5py.AttributeManager) -> int:
    low = np.asarray(header["NumPart_Total"], dtype=np.uint64)
    high = np.asarray(
        header.get("NumPart_Total_HighWord", np.zeros_like(low)),
        dtype=np.uint64,
    )
    return int(low[1] + (high[1] << np.uint64(32)))


def _uniform_dm_mass(handle: h5py.File) -> float:
    header_mass = float(np.asarray(handle["Header"].attrs["MassTable"])[1])
    if "PartType1/Masses" not in handle:
        if header_mass <= 0 or not np.isfinite(header_mass):
            raise ValueError("snapshot has no valid dark-matter particle mass")
        return header_mass
    masses = handle["PartType1/Masses"]
    minimum = np.inf
    maximum = -np.inf
    for begin in range(0, len(masses), 2_000_000):
        value = np.asarray(masses[begin : begin + 2_000_000], dtype=np.float64)
        if not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("dark-matter masses must be finite and positive")
        minimum = min(minimum, float(value.min()))
        maximum = max(maximum, float(value.max()))
    if minimum != maximum:
        raise ValueError(
            f"variable dark-matter masses are unsupported: {minimum} to {maximum}"
        )
    if header_mass > 0 and not np.isclose(header_mass, minimum, rtol=1e-7, atol=0):
        raise ValueError("header and particle dark-matter masses disagree")
    return minimum


def build_grid(
    sources: Sequence[Path],
    *,
    destination: Path,
    grid: int,
    box_mpc_h: float,
    coordinate_scale_to_mpc_h: float,
    assignment: str,
    block_particles: int,
) -> dict[str, Any]:
    if not sources:
        raise ValueError("at least one snapshot source is required")
    if block_particles <= 0:
        raise ValueError("block_particles must be positive")
    metadata_path = destination.with_suffix(".json")
    partial = destination.with_suffix(destination.suffix + ".partial")
    metadata_partial = metadata_path.with_suffix(".json.partial")
    if destination.exists() or metadata_path.exists() or partial.exists() or metadata_partial.exists():
        raise RuntimeError("refusing to overwrite an existing or partial grid cache")
    destination.parent.mkdir(parents=True, exist_ok=True)

    counts = np.zeros((grid,) * 3, dtype=np.float64)
    particles_read = 0
    expected_particles: int | None = None
    reference_mass: float | None = None
    source_rows: list[dict[str, Any]] = []
    started = time.time()
    for file_index, source in enumerate(sources, start=1):
        with h5py.File(source, "r") as handle:
            header = handle["Header"].attrs
            source_box = float(header["BoxSize"]) * coordinate_scale_to_mpc_h
            redshift = float(header["Redshift"])
            if not np.isclose(source_box, box_mpc_h, rtol=0.0, atol=1e-7):
                raise ValueError(f"{source}: BoxSize {source_box} != {box_mpc_h}")
            if abs(redshift) > 1e-6:
                raise ValueError(f"{source}: expected z=0, found {redshift}")
            total = _header_total(header)
            if expected_particles is None:
                expected_particles = total
            elif expected_particles != total:
                raise ValueError("snapshot pieces disagree on total DM particle count")
            particle_mass = _uniform_dm_mass(handle)
            if reference_mass is None:
                reference_mass = particle_mass
            elif not np.isclose(reference_mass, particle_mass, rtol=1e-7, atol=0):
                raise ValueError("snapshot pieces disagree on dark-matter particle mass")

            coordinates = handle["PartType1/Coordinates"]
            file_particles = 0
            for begin in range(0, len(coordinates), block_particles):
                end = min(begin + block_particles, len(coordinates))
                position = (
                    np.asarray(coordinates[begin:end], dtype=np.float64)
                    * coordinate_scale_to_mpc_h
                )
                counts += deposit_particle_counts(
                    position,
                    grid=grid,
                    box_mpc_h=box_mpc_h,
                    assignment=assignment,
                )
                file_particles += len(position)
        particles_read += file_particles
        source_rows.append(
            {
                "path": str(source.resolve()),
                "bytes": source.stat().st_size,
                "dm_particles": file_particles,
            }
        )
        print(
            f"[grid] {file_index}/{len(sources)} particles={particles_read:,} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    assert expected_particles is not None and reference_mass is not None
    deposited = float(counts.sum(dtype=np.float64))
    if particles_read != expected_particles:
        raise RuntimeError(
            f"read {particles_read:,} DM particles, expected {expected_particles:,}"
        )
    if not np.isclose(deposited, particles_read, rtol=0.0, atol=1e-5):
        raise RuntimeError(f"grid deposited {deposited}, read {particles_read}")
    payload = counts.astype(np.float32)
    float32_total = float(payload.sum(dtype=np.float64))
    relative_float32_mass_error = abs(float32_total / particles_read - 1.0)
    if relative_float32_mass_error > 2e-8:
        raise RuntimeError(
            f"float32 grid mass error {relative_float32_mass_error} is too large"
        )
    with partial.open("wb") as handle:
        np.save(handle, payload)
    metadata = {
        "schema": "hong2021-periodic-dm-particle-grid-v1",
        "complete": True,
        "assignment": assignment,
        "grid": grid,
        "box_mpc_h": box_mpc_h,
        "voxel_mpc_h": box_mpc_h / grid,
        "coordinate_scale_to_mpc_h": coordinate_scale_to_mpc_h,
        "source_files": source_rows,
        "source_file_count": len(source_rows),
        "dm_particles": particles_read,
        "dm_particle_mass_code_units": reference_mass,
        "float64_deposited_count": deposited,
        "float32_deposited_count": float32_total,
        "relative_float32_mass_error": relative_float32_mass_error,
        "minimum": float(payload.min()),
        "maximum": float(payload.max()),
        "zero_cells": int(np.count_nonzero(payload == 0)),
        "elapsed_seconds": time.time() - started,
    }
    metadata_partial.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(partial, destination)
    os.replace(metadata_partial, metadata_path)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True, help="shell-style file pattern")
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--box-mpc-h", type=float, required=True)
    parser.add_argument("--coordinate-scale-to-mpc-h", type=float, default=0.001)
    parser.add_argument("--assignment", choices=("ngp", "cic"), default="cic")
    parser.add_argument("--block-particles", type=int, default=20_000_000)
    args = parser.parse_args()
    sources = expand_sources(args.snapshots)
    build_grid(
        sources,
        destination=Path(args.out),
        grid=args.grid,
        box_mpc_h=args.box_mpc_h,
        coordinate_scale_to_mpc_h=args.coordinate_scale_to_mpc_h,
        assignment=args.assignment,
        block_particles=args.block_particles,
    )


if __name__ == "__main__":
    main()
