#!/usr/bin/env python
"""Build split-selection metadata for all 988 Hong TNG100 observers.

This stage reuses the validated 240^3 dark-matter grid.  It does not reread the
6.03 billion particle coordinates.  The group catalog is read once to record
observer properties and local input/target statistics needed to construct a
representative, cross-split-nonoverlapping train/validation split.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_prepare_tng import (
    BOX_MPC_H,
    CELL_MPC_H,
    CUBE_GRID,
    DENSITY_SCALE,
    FULL_GRID,
    extract_periodic_cube,
    galaxy_input_grid,
    indexed_files,
)


def load_catalog(group_files: list[Path]) -> dict[str, np.ndarray]:
    """Load only the subhalo/group fields required for split metadata."""
    subhalo_pieces: dict[str, list[np.ndarray]] = {
        "position": [],
        "velocity": [],
        "stellar_mass_paper": [],
        "b_magnitude": [],
        "flag": [],
        "star_particles": [],
        "group_number": [],
    }
    first_subhalo: list[np.ndarray] = []
    subhalo_offset = 0
    for path in group_files:
        with h5py.File(path, "r") as handle:
            header = handle["Header"].attrs
            nsub = int(header["Nsubgroups_ThisFile"])
            ngroup = int(header["Ngroups_ThisFile"])
            if ngroup:
                first_subhalo.append(handle["Group/GroupFirstSub"][:].astype(np.int64))
            if not nsub:
                continue
            subhalo = handle["Subhalo"]
            subhalo_pieces["position"].append(
                subhalo["SubhaloPos"][:].astype(np.float32) / 1000.0
            )
            subhalo_pieces["velocity"].append(
                subhalo["SubhaloVel"][:].astype(np.float32)
            )
            subhalo_pieces["stellar_mass_paper"].append(
                subhalo["SubhaloMassType"][:, 4].astype(np.float64) * 1.0e10
            )
            subhalo_pieces["b_magnitude"].append(
                subhalo["SubhaloStellarPhotometrics"][:, 1].astype(np.float32)
            )
            subhalo_pieces["flag"].append(
                subhalo["SubhaloFlag"][:].astype(np.uint8)
            )
            subhalo_pieces["star_particles"].append(
                subhalo["SubhaloLenType"][:, 4].astype(np.int32)
            )
            subhalo_pieces["group_number"].append(
                subhalo["SubhaloGrNr"][:].astype(np.int64)
            )
            subhalo_offset += nsub
    if subhalo_offset != 4_371_211:
        raise RuntimeError(
            f"expected 4,371,211 subhalos, found {subhalo_offset:,}"
        )
    catalog = {
        name: np.concatenate(pieces) for name, pieces in subhalo_pieces.items()
    }
    catalog["id"] = np.arange(subhalo_offset, dtype=np.int64)
    catalog["group_first_subhalo"] = np.concatenate(first_subhalo)
    return catalog


def local_geometry(
    center_position: np.ndarray,
    center_velocity: np.ndarray,
    origin_cell: np.ndarray,
    galaxy_position: np.ndarray,
    galaxy_velocity: np.ndarray,
    galaxy_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the paper-mask selection and galaxy radial velocities."""
    local_cell = (galaxy_cell - origin_cell[None, :]) % FULL_GRID
    inside = np.all(local_cell < CUBE_GRID, axis=1)
    displacement = (
        (galaxy_position - center_position + BOX_MPC_H / 2.0) % BOX_MPC_H
        - BOX_MPC_H / 2.0
    )
    radius = np.linalg.norm(displacement, axis=1)
    sin_latitude = np.divide(
        displacement[:, 2],
        radius,
        out=np.zeros_like(radius),
        where=radius > 0,
    )
    latitude = np.degrees(np.arcsin(np.clip(sin_latitude, -1.0, 1.0)))
    keep = inside & (radius > 0) & (np.abs(latitude) >= 10.0)
    relative_velocity = galaxy_velocity - center_velocity
    radial_velocity = np.zeros(len(galaxy_position), dtype=np.float32)
    radial_velocity[keep] = np.einsum(
        "ij,ij->i",
        relative_velocity[keep],
        displacement[keep] / radius[keep, None],
    )
    return keep, radial_velocity


def build_metadata(
    catalog: dict[str, np.ndarray], dm_counts: np.ndarray
) -> dict[str, np.ndarray]:
    center_mask = (
        (catalog["stellar_mass_paper"] > 4.0e10)
        & (catalog["stellar_mass_paper"] < 1.0e11)
    )
    l0_mask = np.isfinite(catalog["b_magnitude"]) & (
        catalog["b_magnitude"] < -15.0
    )
    target_index = np.flatnonzero(l0_mask)
    target_position = catalog["position"][target_index]
    target_velocity = catalog["velocity"][target_index]
    target_cell = np.floor(target_position / CELL_MPC_H).astype(np.int16)
    variant_masks = {
        "l0_paper": np.ones(len(target_index), dtype=bool),
        "l1_flag": catalog["flag"][target_index] == 1,
        "l2_res100": (
            (catalog["flag"][target_index] == 1)
            & (catalog["star_particles"][target_index] >= 100)
        ),
        "l3_mstar1e8": (
            (catalog["flag"][target_index] == 1)
            & (catalog["stellar_mass_paper"][target_index] >= 1.0e8)
        ),
    }
    candidate_index = np.flatnonzero(center_mask)
    if len(candidate_index) != 988:
        raise RuntimeError(f"expected 988 center candidates, found {len(candidate_index)}")
    n = len(candidate_index)
    output: dict[str, np.ndarray] = {
        "candidate_index": np.arange(n, dtype=np.int32),
        "center_subhalo_id": catalog["id"][candidate_index],
        "center_position_mpc_h": catalog["position"][candidate_index],
        "center_velocity_kms": catalog["velocity"][candidate_index],
        "center_stellar_mass_paper_msun": catalog["stellar_mass_paper"][
            candidate_index
        ],
        "center_flag": catalog["flag"][candidate_index],
        "center_star_particles": catalog["star_particles"][candidate_index],
        "center_is_central": (
            catalog["id"][candidate_index]
            == catalog["group_first_subhalo"][
                catalog["group_number"][candidate_index]
            ]
        ).astype(np.uint8),
        "cube_origin_cell": np.empty((n, 3), dtype=np.int16),
        "occupied_cells_l0": np.empty(n, dtype=np.int32),
        "radial_velocity_galaxy_mean_kms": np.empty(n, dtype=np.float32),
        "radial_velocity_galaxy_std_kms": np.empty(n, dtype=np.float32),
        "occupied_cell_mean_velocity_std_kms": np.empty(n, dtype=np.float32),
        "target_mean": np.empty(n, dtype=np.float32),
        "target_std": np.empty(n, dtype=np.float32),
        "target_min": np.empty(n, dtype=np.float32),
        "target_max": np.empty(n, dtype=np.float32),
    }
    for variant in variant_masks:
        output[f"galaxies_{variant}"] = np.empty(n, dtype=np.int32)

    mean_dm_count = float(dm_counts.sum(dtype=np.uint64) / dm_counts.size)
    for output_index, subhalo_index in enumerate(candidate_index):
        center_position = catalog["position"][subhalo_index]
        center_velocity = catalog["velocity"][subhalo_index]
        center_cell = np.floor(center_position / CELL_MPC_H).astype(np.int64)
        origin = center_cell - CUBE_GRID // 2
        output["cube_origin_cell"][output_index] = origin
        keep, radial_velocity = local_geometry(
            center_position,
            center_velocity,
            origin,
            target_position,
            target_velocity,
            target_cell,
        )
        for variant, variant_mask in variant_masks.items():
            output[f"galaxies_{variant}"][output_index] = np.count_nonzero(
                keep & variant_mask
            )
        selected_velocity = radial_velocity[keep]
        output["radial_velocity_galaxy_mean_kms"][output_index] = np.mean(
            selected_velocity
        )
        output["radial_velocity_galaxy_std_kms"][output_index] = np.std(
            selected_velocity
        )
        count, mean_velocity, _ = galaxy_input_grid(
            center_position,
            center_velocity,
            origin,
            target_position,
            target_velocity,
            target_cell,
        )
        occupied = count > 0
        output["occupied_cells_l0"][output_index] = np.count_nonzero(occupied)
        output["occupied_cell_mean_velocity_std_kms"][output_index] = np.std(
            mean_velocity[occupied]
        )
        density_count = extract_periodic_cube(dm_counts, origin).astype(np.float32)
        target = np.log10(density_count / mean_dm_count) / DENSITY_SCALE
        output["target_mean"][output_index] = np.mean(target)
        output["target_std"][output_index] = np.std(target)
        output["target_min"][output_index] = np.min(target)
        output["target_max"][output_index] = np.max(target)
        if (output_index + 1) % 50 == 0 or output_index + 1 == n:
            print(f"[metadata] {output_index + 1}/{n}", flush=True)
    return output


def atomic_write_h5(
    destination: Path, arrays: dict[str, np.ndarray], attributes: dict[str, Any]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if destination.exists() or temporary.exists():
        raise RuntimeError(f"refusing to overwrite metadata output: {destination}")
    with h5py.File(temporary, "w") as handle:
        for key, value in attributes.items():
            handle.attrs[key] = value
        for name, values in arrays.items():
            handle.create_dataset(
                name, data=values, compression="gzip", compression_opts=4
            )
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tng-root", type=Path, required=True)
    parser.add_argument("--dm-counts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    group_files = indexed_files(
        args.tng_root / "output/groups_099", "fof_subhalo_tab_099"
    )
    if len(group_files) != 448:
        raise SystemExit(f"expected 448 group catalog chunks, found {len(group_files)}")
    dm_counts = np.load(args.dm_counts, mmap_mode="r")
    if dm_counts.shape != (FULL_GRID,) * 3 or dm_counts.dtype != np.uint32:
        raise SystemExit("invalid validated DM count grid")
    catalog = load_catalog(group_files)
    arrays = build_metadata(catalog, dm_counts)
    attributes = {
        "schema": "hong2021-center-metadata-v1",
        "simulation": "TNG100-1 snapshot 99",
        "center_selection": "4e10 < SubhaloMassType[:,4]*1e10 < 1e11",
        "l0_selection": "M_B < -15",
        "l1_selection": "L0 and SubhaloFlag == 1",
        "l2_selection": "L1 and Nstar >= 100",
        "l3_selection": "L1 and stellar_mass_paper >= 1e8",
        "dm_grid": str(args.dm_counts),
    }
    atomic_write_h5(args.out, arrays, attributes)
    report = {
        "schema": attributes["schema"],
        "path": str(args.out),
        "centers": int(len(arrays["candidate_index"])),
        "maximum_local_cube_counts": {
            key.removeprefix("galaxies_"): int(np.nanmax(values))
            for key, values in arrays.items()
            if key.startswith("galaxies_")
        },
        "feature_ranges": {
            key: [float(np.min(values)), float(np.median(values)), float(np.max(values))]
            for key, values in arrays.items()
            if values.ndim == 1 and key not in {"candidate_index", "center_subhalo_id"}
        },
    }
    report_path = args.out.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
