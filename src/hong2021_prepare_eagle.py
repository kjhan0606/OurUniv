#!/usr/bin/env python
"""Prepare a locked EAGLE RefL0100N1504 Hong-style test set.

This is an independent cross-simulation gate for a model trained only on
TNG100.  EAGLE is never used to fit model weights or normalization.

The EAGLE box (67.77 Mpc/h) is not an integer multiple of the Hong voxel
(0.3125 Mpc/h).  A naive periodic 217^3 grid would therefore change the cell
size.  This implementation bins particles on an exact 216^3 regular grid
covering [0, 67.5 Mpc/h)^3 and emits only observer cubes wholly contained in
that regular region.  This gives exact 20 Mpc/h, 64^3 cubes without resampling
or a shortened seam cell.  The coordinate-defined subset is selected before
the dark-matter truth is inspected.

The 470-GiB uncompressed tar archive is read in place.  HDF5 members are
opened through seekable tar member streams, so a second 470-GiB extraction is
not required.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tarfile
import time
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


SCHEMA = "hong2021-independent-eagle-ref100-input-v1"
DEFAULT_ROOT = Path("/gpfs/kjhan/EAGLE/RefL0100N1504")
SNAPSHOT_MEMBER_RE = re.compile(
    r"^RefL0100N1504/snapshot_028_z000p000/"
    r"snap_028_z000p000\.(\d+)\.hdf5$"
)
EXPECTED_ARCHIVE_BYTES = 504_327_096_320
EXPECTED_SNAPSHOT_FILES = 256
EXPECTED_DM_PARTICLES = 1504**3
BOX_MPC_H = 67.77
HUBBLE_PARAM = 0.6777
CELL_MPC_H = 0.3125
REGULAR_GRID = 216
REGULAR_EXTENT_MPC_H = REGULAR_GRID * CELL_MPC_H
CUBE_GRID = 64
CUBE_MPC_H = CUBE_GRID * CELL_MPC_H
DENSITY_SCALE = 4.5
MASK_LATITUDE_DEG = 10.0
CENTER_MASS_MIN = 4.0e10
CENTER_MASS_MAX = 1.0e11
TNG_TARGET_GALAXIES = 48_296
TNG_BOX_MPC_H = 75.0
TARGET_NUMBER_DENSITY = TNG_TARGET_GALAXIES / TNG_BOX_MPC_H**3
EXPECTED_TARGET_GALAXIES = round(TARGET_NUMBER_DENSITY * BOX_MPC_H**3)
PAPER_REPORTED_EAGLE_CENTERS = 478


def snapshot_members(members: Iterable[tarfile.TarInfo]) -> list[tarfile.TarInfo]:
    """Return the 256 snapshot members in numeric file order."""
    indexed: list[tuple[int, tarfile.TarInfo]] = []
    for member in members:
        match = SNAPSHOT_MEMBER_RE.fullmatch(member.name)
        if match is not None:
            if not member.isfile():
                raise RuntimeError(f"snapshot member is not a file: {member.name}")
            indexed.append((int(match.group(1)), member))
    indexed.sort(key=lambda item: item[0])
    indices = [item[0] for item in indexed]
    expected = list(range(EXPECTED_SNAPSHOT_FILES))
    if indices != expected:
        raise RuntimeError(
            f"snapshot archive indices are incomplete: found {len(indices)} files"
        )
    return [item[1] for item in indexed]


def load_catalog(path: Path) -> dict[str, np.ndarray]:
    """Load and validate the frozen number-density-matched EAGLE catalogue."""
    header_line: int | None = None
    with path.open() as stream:
        for line_number, line in enumerate(stream):
            if not line.startswith("#"):
                header_line = line_number
                break
    if header_line is None:
        raise ValueError(f"catalogue has no CSV header: {path}")
    table = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        skip_header=header_line,
        dtype=None,
        encoding="utf-8",
    )
    required = {
        "GalaxyID",
        "GroupNumber",
        "SubGroupNumber",
        "CentreOfPotential_x",
        "CentreOfPotential_y",
        "CentreOfPotential_z",
        "Velocity_x",
        "Velocity_y",
        "Velocity_z",
        "Mstar_30pkpc",
    }
    names = set(table.dtype.names or ())
    if names != required:
        raise ValueError(f"unexpected EAGLE catalogue schema: {sorted(names)}")
    if table.shape != (EXPECTED_TARGET_GALAXIES,):
        raise ValueError(
            f"expected {EXPECTED_TARGET_GALAXIES} target rows, got {table.shape}"
        )

    result = {
        "galaxy_id": np.asarray(table["GalaxyID"], dtype=np.int64),
        "group_number": np.asarray(table["GroupNumber"], dtype=np.int64),
        "subgroup_number": np.asarray(table["SubGroupNumber"], dtype=np.int64),
        # SQL positions are in cMpc.  Multiplication by h puts them in the
        # same Mpc/h coordinate convention as PartType1/Coordinates.
        "position": np.column_stack(
            [
                table["CentreOfPotential_x"],
                table["CentreOfPotential_y"],
                table["CentreOfPotential_z"],
            ]
        ).astype(np.float64)
        * HUBBLE_PARAM,
        "velocity": np.column_stack(
            [table["Velocity_x"], table["Velocity_y"], table["Velocity_z"]]
        ).astype(np.float64),
        "stellar_mass": np.asarray(table["Mstar_30pkpc"], dtype=np.float64),
    }
    if len(np.unique(result["galaxy_id"])) != EXPECTED_TARGET_GALAXIES:
        raise ValueError("duplicate GalaxyID in EAGLE catalogue")
    for name in ("position", "velocity", "stellar_mass"):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"non-finite catalogue field: {name}")
    position = result["position"]
    if np.any(position < 0.0) or np.any(position >= BOX_MPC_H):
        raise ValueError("catalogue position lies outside EAGLE periodic box")
    mass = result["stellar_mass"]
    if np.any(mass <= 0.0) or np.any(mass[:-1] < mass[1:]):
        raise ValueError("catalogue is not a positive descending mass-rank sample")
    return result


def center_mask(catalog: dict[str, np.ndarray]) -> np.ndarray:
    """Paper center criterion plus the implicit central-galaxy requirement."""
    mass = catalog["stellar_mass"]
    return (
        (catalog["subgroup_number"] == 0)
        & (mass > CENTER_MASS_MIN)
        & (mass < CENTER_MASS_MAX)
    )


def cube_origins(center_position: np.ndarray) -> np.ndarray:
    cells = np.floor(np.asarray(center_position) / CELL_MPC_H).astype(np.int64)
    return cells - CUBE_GRID // 2


def geometry_safe_mask(center_position: np.ndarray) -> np.ndarray:
    """Select exact-grid cubes without looking at the DM truth."""
    origins = cube_origins(center_position)
    return np.all(
        (origins >= 0) & (origins + CUBE_GRID <= REGULAR_GRID), axis=1
    )


def farthest_point_subset(
    positions: np.ndarray, identifiers: np.ndarray, count: int
) -> np.ndarray:
    """Deterministically spread an optional capped sample in position space."""
    positions = np.asarray(positions, dtype=np.float64)
    identifiers = np.asarray(identifiers, dtype=np.int64)
    if count <= 0 or count > len(positions):
        raise ValueError("invalid farthest-point subset size")
    first = int(np.argmin(identifiers))
    chosen = [first]
    minimum_distance2 = np.sum((positions - positions[first]) ** 2, axis=1)
    minimum_distance2[first] = -1.0
    while len(chosen) < count:
        next_index = int(np.argmax(minimum_distance2))
        chosen.append(next_index)
        distance2 = np.sum((positions - positions[next_index]) ** 2, axis=1)
        minimum_distance2 = np.minimum(minimum_distance2, distance2)
        minimum_distance2[chosen] = -1.0
    return np.asarray(chosen, dtype=np.int64)


def validate_snapshot(handle: h5py.File, source: str) -> dict[str, Any]:
    header = handle["Header"].attrs
    box = float(header["BoxSize"])
    hubble = float(header["HubbleParam"])
    redshift = float(header["Redshift"])
    files = int(header["NumFilesPerSnapshot"])
    total = int(np.asarray(header["NumPart_Total"], dtype=np.uint64)[1])
    if not np.isclose(box, BOX_MPC_H, rtol=0.0, atol=1.0e-8):
        raise RuntimeError(f"{source}: BoxSize={box}, expected {BOX_MPC_H}")
    if not np.isclose(hubble, HUBBLE_PARAM, rtol=0.0, atol=1.0e-8):
        raise RuntimeError(f"{source}: HubbleParam={hubble}")
    if abs(redshift) > 1.0e-10 or files != EXPECTED_SNAPSHOT_FILES:
        raise RuntimeError(f"{source}: unexpected z/files ({redshift}, {files})")
    if total != EXPECTED_DM_PARTICLES:
        raise RuntimeError(f"{source}: total DM particles={total:,}")
    coordinates = handle["PartType1/Coordinates"]
    attributes = coordinates.attrs
    if float(attributes["h-scale-exponent"]) != -1.0:
        raise RuntimeError(f"{source}: unexpected coordinate h exponent")
    if float(attributes["aexp-scale-exponent"]) != 1.0:
        raise RuntimeError(f"{source}: unexpected coordinate a exponent")
    return {
        "box_mpc_h": box,
        "hubble_param": hubble,
        "redshift": redshift,
        "snapshot_files": files,
        "dm_particles_total": total,
        "dm_particles_this_file": int(coordinates.shape[0]),
        "coordinate_dtype": str(coordinates.dtype),
        "coordinate_h_scale_exponent": float(
            attributes["h-scale-exponent"]
        ),
        "coordinate_aexp_scale_exponent": float(
            attributes["aexp-scale-exponent"]
        ),
    }


def accumulate_coordinates(
    counts: np.ndarray, coordinates: np.ndarray
) -> tuple[int, int]:
    """Accumulate coordinates in exact 0.3125-Mpc/h regular cells."""
    value = np.asarray(coordinates, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"expected (N,3) coordinates, got {value.shape}")
    if not np.isfinite(value).all():
        raise RuntimeError("non-finite DM coordinate")
    if np.any(value < 0.0) or np.any(value >= BOX_MPC_H):
        raise RuntimeError("DM coordinate outside EAGLE periodic box")
    cells = np.floor(value / CELL_MPC_H).astype(np.int16)
    keep = np.all(cells < REGULAR_GRID, axis=1)
    selected = cells[keep].astype(np.int64, copy=False)
    flat = (
        (selected[:, 0] * REGULAR_GRID + selected[:, 1]) * REGULAR_GRID
        + selected[:, 2]
    )
    histogram = np.bincount(flat, minlength=REGULAR_GRID**3).astype(
        np.uint64, copy=False
    )
    counts += histogram.reshape(counts.shape)
    return len(value), int(keep.sum())


def validated_density_cache(
    grid_path: Path, metadata_path: Path, archive: Path
) -> tuple[np.ndarray, dict[str, Any]] | None:
    if not grid_path.is_file() and not metadata_path.is_file():
        return None
    if not grid_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("density cache payload/metadata pair is incomplete")
    metadata = json.loads(metadata_path.read_text())
    counts = np.load(grid_path, mmap_mode="r")
    valid = (
        metadata.get("complete") is True
        and metadata.get("archive_bytes") == archive.stat().st_size
        and metadata.get("particles_read") == EXPECTED_DM_PARTICLES
        and metadata.get("particles_binned")
        == int(counts.sum(dtype=np.uint64))
        and counts.shape == (REGULAR_GRID,) * 3
        and counts.dtype == np.uint32
    )
    if not valid:
        raise RuntimeError(f"invalid existing EAGLE density cache: {grid_path}")
    return counts, metadata


def build_dark_matter_grid(
    archive: Path, derived: Path, block_particles: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stream all 1504^3 DM coordinates from the tar into an exact grid."""
    if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError(
            f"archive size {archive.stat().st_size} != {EXPECTED_ARCHIVE_BYTES}"
        )
    grid_path = derived / "dm_counts_exact_regular_216.npy"
    metadata_path = derived / "dm_counts_exact_regular_216.json"
    cached = validated_density_cache(grid_path, metadata_path, archive)
    if cached is not None:
        print(f"[density] using validated cache {grid_path}", flush=True)
        return cached

    partial_path = derived / "dm_counts_exact_regular_216.partial.npy"
    if partial_path.exists():
        raise RuntimeError(f"incomplete prior density cache exists: {partial_path}")
    counts = np.zeros((REGULAR_GRID,) * 3, dtype=np.uint64)
    particles_read = 0
    particles_binned = 0
    start_time = time.time()
    first_audit: dict[str, Any] | None = None

    with tarfile.open(archive, mode="r:") as tar_handle:
        members = snapshot_members(tar_handle.getmembers())
        for file_index, member in enumerate(members, start=1):
            extracted = tar_handle.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot open tar member {member.name}")
            with extracted, h5py.File(extracted, "r") as handle:
                audit = validate_snapshot(handle, member.name)
                if first_audit is None:
                    first_audit = audit
                coordinates = handle["PartType1/Coordinates"]
                file_read = 0
                file_binned = 0
                for begin in range(0, len(coordinates), block_particles):
                    end = min(begin + block_particles, len(coordinates))
                    read, binned = accumulate_coordinates(
                        counts, coordinates[begin:end]
                    )
                    file_read += read
                    file_binned += binned
            if file_read != audit["dm_particles_this_file"]:
                raise RuntimeError(f"short DM read from {member.name}")
            particles_read += file_read
            particles_binned += file_binned
            elapsed = time.time() - start_time
            print(
                f"[density] {file_index:03d}/{len(members)} "
                f"read={particles_read:,} binned={particles_binned:,} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    if particles_read != EXPECTED_DM_PARTICLES:
        raise RuntimeError(
            f"DM total mismatch: {particles_read:,} != {EXPECTED_DM_PARTICLES:,}"
        )
    if int(counts.sum(dtype=np.uint64)) != particles_binned:
        raise RuntimeError("density grid does not conserve selected particle count")
    if counts.max() > np.iinfo(np.uint32).max:
        raise RuntimeError("uint32 EAGLE density cache would overflow")
    final_counts = counts.astype(np.uint32)
    np.save(partial_path, final_counts)
    os.replace(partial_path, grid_path)
    metadata = {
        "complete": True,
        "schema": "eagle-ref100-exact-regular-dm-counts-v1",
        "source_archive": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size,
        "snapshot_files": EXPECTED_SNAPSHOT_FILES,
        "particles_read": particles_read,
        "particles_binned": particles_binned,
        "particles_outside_regular_extent": particles_read - particles_binned,
        "simulation_box_mpc_h": BOX_MPC_H,
        "regular_grid": REGULAR_GRID,
        "regular_extent_mpc_h": REGULAR_EXTENT_MPC_H,
        "voxel_mpc_h": CELL_MPC_H,
        "minimum_count": int(final_counts.min()),
        "maximum_count": int(final_counts.max()),
        "elapsed_seconds": time.time() - start_time,
        "first_file_unit_audit": first_audit,
        "method": (
            "all PartType1 coordinates streamed directly from seekable HDF5 "
            "members in the uncompressed tar; no full archive extraction"
        ),
    }
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary_metadata, metadata_path)
    return np.load(grid_path, mmap_mode="r"), metadata


def galaxy_input_grid(
    center_position: np.ndarray,
    center_velocity: np.ndarray,
    origin: np.ndarray,
    galaxy_position: np.ndarray,
    galaxy_velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    galaxy_cell = np.floor(galaxy_position / CELL_MPC_H).astype(np.int64)
    local_cell = galaxy_cell - origin[None, :]
    inside = np.all((local_cell >= 0) & (local_cell < CUBE_GRID), axis=1)
    displacement = galaxy_position - center_position
    radius = np.linalg.norm(displacement, axis=1)
    sin_latitude = np.divide(
        displacement[:, 2],
        radius,
        out=np.zeros_like(radius),
        where=radius > 0,
    )
    latitude = np.degrees(np.arcsin(np.clip(sin_latitude, -1.0, 1.0)))
    keep = inside & (radius > 0) & (np.abs(latitude) >= MASK_LATITUDE_DEG)
    local = local_cell[keep]
    relative_velocity = galaxy_velocity[keep] - center_velocity
    radial_velocity = np.einsum(
        "ij,ij->i", relative_velocity, displacement[keep] / radius[keep, None]
    )
    flat = (
        (local[:, 0] * CUBE_GRID + local[:, 1]) * CUBE_GRID + local[:, 2]
    )
    count = np.bincount(flat, minlength=CUBE_GRID**3).astype(np.float32)
    velocity_sum = np.bincount(
        flat, weights=radial_velocity, minlength=CUBE_GRID**3
    )
    mean_velocity = np.divide(
        velocity_sum,
        count,
        out=np.zeros_like(velocity_sum),
        where=count > 0,
    ).astype(np.float32)
    shape = (CUBE_GRID,) * 3
    return count.reshape(shape), mean_velocity.reshape(shape), int(keep.sum())


def select_centers(
    catalog: dict[str, np.ndarray], max_centers: int | None
) -> tuple[np.ndarray, dict[str, Any]]:
    candidates = np.flatnonzero(center_mask(catalog))
    safe = geometry_safe_mask(catalog["position"][candidates])
    selected = candidates[safe]
    if max_centers is not None and max_centers < len(selected):
        subset = farthest_point_subset(
            catalog["position"][selected], catalog["galaxy_id"][selected], max_centers
        )
        selected = selected[subset]
    if len(selected) == 0:
        raise RuntimeError("no geometry-safe EAGLE observer candidates")
    metadata = {
        "paper_reported_center_candidates": PAPER_REPORTED_EAGLE_CENTERS,
        "current_public_catalog_center_candidates": int(len(candidates)),
        "geometry_safe_center_candidates": int(np.count_nonzero(safe)),
        "selected_centers": int(len(selected)),
        "selection_uses_dark_matter_truth": False,
        "current_vs_paper_center_count_note": (
            "The current public 30-pkpc catalogue yields a different count "
            "from the paper; no post-hoc mass-threshold tuning was applied."
        ),
    }
    return selected, metadata


def write_test_set(
    destination: Path,
    catalog: dict[str, np.ndarray],
    selected: np.ndarray,
    selection_metadata: dict[str, Any],
    dm_counts: np.ndarray,
    density_metadata: dict[str, Any],
    source_catalog: Path,
) -> dict[str, Any]:
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists() or partial.exists():
        raise RuntimeError(f"refusing to overwrite {destination} or {partial}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    origins = cube_origins(catalog["position"][selected])
    cosmic_mean_count = EXPECTED_DM_PARTICLES * (
        CELL_MPC_H / BOX_MPC_H
    ) ** 3
    rows: list[dict[str, Any]] = []

    try:
        with h5py.File(partial, "w") as handle:
            handle.attrs.update(
                {
                    "schema": SCHEMA,
                    "complete": False,
                    "paper": "Hong et al. 2021 ApJ 913 76",
                    "simulation": "EAGLE RefL0100N1504",
                    "snapshot": 28,
                    "independent_test_only": True,
                    "tng_fitting_or_normalization_allowed": False,
                    "simulation_box_mpc_h": BOX_MPC_H,
                    "box_mpc_h": CUBE_MPC_H,
                    "grid": CUBE_GRID,
                    "voxel_mpc_h": CELL_MPC_H,
                    "density_scale": DENSITY_SCALE,
                    "channels": "Ngal,mean_radial_vpec_kms",
                    "galactic_mask_abs_b_deg": MASK_LATITUDE_DEG,
                    "target_definition": "log10(rho_dm/rho_dm_cosmic_mean)/4.5",
                    "target_number_density_h3_mpc3": TARGET_NUMBER_DENSITY,
                    "target_galaxies": EXPECTED_TARGET_GALAXIES,
                    "target_selection": (
                        "top 35632 EAGLE 30-pkpc stellar masses, matching "
                        "TNG100 M_B<-15 number density"
                    ),
                    "center_selection": (
                        "SubGroupNumber=0 and 4e10<Mstar_30pkpc/Msun<1e11"
                    ),
                    "grid_alignment": (
                        "exact 0.3125-Mpc/h cells in [0,67.5)^3; only cubes "
                        "not touching the non-integer periodic seam"
                    ),
                    "source_catalog": str(source_catalog.resolve()),
                }
            )
            n = len(selected)
            inputs = handle.create_dataset(
                "input",
                shape=(n, 2, CUBE_GRID, CUBE_GRID, CUBE_GRID),
                dtype="f4",
                chunks=(1, 2, CUBE_GRID, CUBE_GRID, CUBE_GRID),
                compression="lzf",
            )
            targets = handle.create_dataset(
                "target",
                shape=(n, 1, CUBE_GRID, CUBE_GRID, CUBE_GRID),
                dtype="f4",
                chunks=(1, 1, CUBE_GRID, CUBE_GRID, CUBE_GRID),
                compression="lzf",
            )
            handle.create_dataset("center_galaxy_id", data=catalog["galaxy_id"][selected])
            handle.create_dataset(
                "center_position_mpc_h", data=catalog["position"][selected]
            )
            handle.create_dataset(
                "center_velocity_kms", data=catalog["velocity"][selected]
            )
            handle.create_dataset(
                "center_stellar_mass_msun", data=catalog["stellar_mass"][selected]
            )
            handle.create_dataset("cube_origin_cell", data=origins.astype(np.int16))

            for output_index, catalog_index in enumerate(selected):
                origin = origins[output_index]
                center_position = catalog["position"][catalog_index]
                center_velocity = catalog["velocity"][catalog_index]
                count, velocity, kept = galaxy_input_grid(
                    center_position,
                    center_velocity,
                    origin,
                    catalog["position"],
                    catalog["velocity"],
                )
                slices = tuple(
                    slice(int(value), int(value) + CUBE_GRID) for value in origin
                )
                cube_count = np.asarray(dm_counts[slices], dtype=np.float32)
                if cube_count.shape != (CUBE_GRID,) * 3 or cube_count.min() <= 0:
                    raise RuntimeError("invalid or zero-count EAGLE DM cube")
                target = np.log10(cube_count / cosmic_mean_count) / DENSITY_SCALE
                if not np.isfinite(target).all():
                    raise RuntimeError("non-finite EAGLE target")
                inputs[output_index, 0] = count
                inputs[output_index, 1] = velocity
                targets[output_index, 0] = target
                row = {
                    "output_index": output_index,
                    "center_galaxy_id": int(catalog["galaxy_id"][catalog_index]),
                    "galaxies_after_cube_and_mask": kept,
                    "occupied_cells": int(np.count_nonzero(count)),
                    "target_min": float(target.min()),
                    "target_max": float(target.max()),
                    "target_mean": float(target.mean()),
                    "target_std": float(target.std()),
                }
                rows.append(row)
                if (output_index + 1) % 16 == 0 or output_index + 1 == n:
                    print(f"[cubes] {output_index + 1}/{n}", flush=True)
            handle.attrs["complete"] = True
        os.replace(partial, destination)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise

    report = {
        "schema": SCHEMA,
        "output": str(destination.resolve()),
        "samples": len(selected),
        "selection": selection_metadata,
        "density": density_metadata,
        "cosmic_mean_particles_per_voxel": cosmic_mean_count,
        "target_stellar_mass_threshold_msun": float(
            catalog["stellar_mass"].min()
        ),
        "rows": rows,
    }
    report_path = destination.with_suffix(".json")
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary_report, report_path)
    return report


def unit_audit(archive: Path, catalog_path: Path) -> dict[str, Any]:
    if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError("EAGLE archive size mismatch")
    catalog = load_catalog(catalog_path)
    selected, selection = select_centers(catalog, max_centers=None)
    with tarfile.open(archive, mode="r:") as tar_handle:
        members = snapshot_members(tar_handle.getmembers())
        extracted = tar_handle.extractfile(members[0])
        if extracted is None:
            raise RuntimeError("cannot open first EAGLE snapshot member")
        with extracted, h5py.File(extracted, "r") as handle:
            snapshot = validate_snapshot(handle, members[0].name)
            coordinate_sample = np.asarray(
                handle["PartType1/Coordinates"][:100_000], dtype=np.float64
            )
    return {
        "schema": "eagle-ref100-hong-unit-audit-v1",
        "archive": str(archive.resolve()),
        "archive_bytes": archive.stat().st_size,
        "snapshot": snapshot,
        "coordinate_sample_min_mpc_h": coordinate_sample.min(axis=0).tolist(),
        "coordinate_sample_max_mpc_h": coordinate_sample.max(axis=0).tolist(),
        "catalog": str(catalog_path.resolve()),
        "target_galaxies": len(catalog["galaxy_id"]),
        "target_number_density_h3_mpc3": TARGET_NUMBER_DENSITY,
        "target_stellar_mass_threshold_msun": float(catalog["stellar_mass"].min()),
        "catalog_position_min_mpc_h": catalog["position"].min(axis=0).tolist(),
        "catalog_position_max_mpc_h": catalog["position"].max(axis=0).tolist(),
        "selection": selection,
        "selected_center_ids_sha_policy": (
            "IDs are written to the prepared HDF5; selection uses only public "
            "catalogue mass, central flag, position, and fixed grid geometry"
        ),
        "selected_centers": len(selected),
        "unit_conclusion": (
            "Snapshot Coordinates are stored in Mpc/h at z=0; SQL positions "
            "are cMpc and are multiplied by h=0.6777; velocities are km/s."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--max-centers", type=int)
    parser.add_argument("--block-particles", type=int, default=4_000_000)
    args = parser.parse_args()
    archive = args.archive or args.root / "raw/RefL0100N1504_snap_028.tar"
    catalog_path = args.catalog or (
        args.root / "catalogs/RefL0100N1504_Hong_targets_snap28.csv"
    )
    derived = args.root / "derived/hong2021_v1"
    derived.mkdir(parents=True, exist_ok=True)

    audit = unit_audit(archive, catalog_path)
    audit_path = derived / "unit_audit.json"
    temporary_audit = audit_path.with_suffix(".json.tmp")
    temporary_audit.write_text(json.dumps(audit, indent=2) + "\n")
    os.replace(temporary_audit, audit_path)
    print(json.dumps(audit, indent=2), flush=True)
    if args.audit_only:
        return

    catalog = load_catalog(catalog_path)
    selected, selection = select_centers(catalog, args.max_centers)
    dm_counts, density_metadata = build_dark_matter_grid(
        archive, derived, args.block_particles
    )
    destination = args.out or derived / "eagle_ref100_z0_test.h5"
    report = write_test_set(
        destination,
        catalog,
        selected,
        selection,
        dm_counts,
        density_metadata,
        catalog_path,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
