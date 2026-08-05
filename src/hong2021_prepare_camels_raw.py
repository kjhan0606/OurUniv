#!/usr/bin/env python
"""Prepare Hong-style CAMELS cubes from the frozen V14 raw-CIC target.

The observable construction and stellar-mass cuts are inherited from the
legacy SIMBA preparation, but the target is read only from a validated
cell-centred CIC cache produced by ``hong2021_build_particle_grid.py``.  CMD
adaptive-smoothed grids are neither accepted nor opened.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_prepare_simba import (
    BOX_MPC_H,
    CELL_MPC_H,
    CUBE_GRID,
    CUBE_MPC_H,
    DENSITY_SCALE,
    GALAXY_MASS_THRESHOLD,
    MASK_LATITUDE_DEG,
    OBSERVER_MASS,
    WORK_GRID,
    choose_observer,
    extract_periodic_cube,
    galaxy_input_grid,
    load_catalog,
    observer_candidates,
)
from hong2021_v14_target import log_cic_target


SCHEMA = "hong2021-camels-raw-cic-input-v14"
EXPECTED_DM_PARTICLES = 256**3


def validated_cic_grid(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"CIC grid/metadata pair is incomplete: {path}")
    metadata = json.loads(metadata_path.read_text())
    field = np.load(path, mmap_mode="r")
    valid = (
        metadata.get("schema") == "hong2021-periodic-dm-particle-grid-v1"
        and metadata.get("complete") is True
        and metadata.get("assignment") == "cic"
        and metadata.get("grid") == WORK_GRID
        and np.isclose(metadata.get("box_mpc_h"), BOX_MPC_H)
        and metadata.get("dm_particles") == EXPECTED_DM_PARTICLES
        and field.shape == (WORK_GRID,) * 3
        and np.issubdtype(field.dtype, np.floating)
    )
    if not valid:
        raise ValueError(f"invalid V14 CIC cache: {path}")
    if not np.isfinite(field).all() or np.any(field <= 0):
        raise ValueError(f"CIC cache is not finite and strictly positive: {path}")
    relative_mass_error = abs(
        float(field.sum(dtype=np.float64)) / EXPECTED_DM_PARTICLES - 1.0
    )
    if relative_mass_error > 1e-7:
        raise ValueError(f"CIC cache mass error {relative_mass_error}: {path}")
    return field, metadata


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    destination = Path(args.out)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    report_path = destination.with_suffix(".json")
    if any(path.exists() for path in (destination, temporary, report_path)):
        raise RuntimeError(f"refusing to overwrite output for {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    realizations = [int(value) for value in args.realizations.split(",")]
    if not realizations or len(realizations) != len(set(realizations)):
        raise ValueError("realizations must be a non-empty unique list")

    catalogs = {
        realization: load_catalog(
            root / "CV" / f"CV_{realization}" / "groups_090.hdf5"
        )
        for realization in realizations
    }
    samples: list[tuple[int, int]] = []
    for realization in realizations:
        stellar_mass = np.asarray(catalogs[realization]["stellar_mass"])
        if args.observers == "single":
            selected = np.asarray([choose_observer(stellar_mass)])
        else:
            selected = observer_candidates(stellar_mass)
            if len(selected) == 0:
                raise RuntimeError(f"no observer candidates in CV_{realization}")
        samples.extend((realization, int(observer)) for observer in selected)

    rows: list[dict[str, Any]] = []
    density_cache: dict[int, np.ndarray] = {}
    density_metadata: dict[int, dict[str, Any]] = {}
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(
                {
                    "schema": SCHEMA,
                    "simulation": f"CAMELS-{args.suite}-CV",
                    "role": args.role,
                    "box_mpc_h": CUBE_MPC_H,
                    "simulation_box_mpc_h": BOX_MPC_H,
                    "voxel_mpc_h": CELL_MPC_H,
                    "density_scale": DENSITY_SCALE,
                    "channels": "Ngal,mean_radial_vpec_kms",
                    "galactic_mask_abs_b_deg": MASK_LATITUDE_DEG,
                    "galaxy_stellar_mass_threshold": GALAXY_MASS_THRESHOLD,
                    "target_definition": "log10(raw_particle_cic_cdm/box_mean)/4.5",
                    "target_operator": "periodic cell-centred CIC on the global 80^3 grid",
                    "cmd_target_used": False,
                    "observer_selection": args.observers,
                }
            )
            shape = (len(samples), 2, CUBE_GRID, CUBE_GRID, CUBE_GRID)
            input_data = handle.create_dataset(
                "input", shape=shape, dtype="f4", chunks=(1, 2, CUBE_GRID, CUBE_GRID, CUBE_GRID), compression="lzf"
            )
            target_data = handle.create_dataset(
                "target", shape=(len(samples), 1, CUBE_GRID, CUBE_GRID, CUBE_GRID), dtype="f4", chunks=(1, 1, CUBE_GRID, CUBE_GRID, CUBE_GRID), compression="lzf"
            )
            handle.create_dataset("realization", data=np.asarray([value[0] for value in samples]))
            observer_index_data = handle.create_dataset("center_subhalo_id", shape=(len(samples),), dtype="i8")
            observer_position_data = handle.create_dataset("center_position_mpc_h", shape=(len(samples), 3), dtype="f8")
            observer_velocity_data = handle.create_dataset("center_velocity_kms", shape=(len(samples), 3), dtype="f8")
            origin_data = handle.create_dataset("cube_origin_cell", shape=(len(samples), 3), dtype="i2")

            for output_index, (realization, observer) in enumerate(samples):
                catalog = catalogs[realization]
                position = np.asarray(catalog["position"])
                velocity = np.asarray(catalog["velocity"])
                stellar_mass = np.asarray(catalog["stellar_mass"])
                galaxy = stellar_mass >= GALAXY_MASS_THRESHOLD
                galaxy_position = position[galaxy]
                galaxy_velocity = velocity[galaxy]
                galaxy_cell = np.floor(galaxy_position / CELL_MPC_H).astype(np.int64)
                center_position = position[observer]
                center_velocity = velocity[observer]
                center_cell = np.floor(center_position / CELL_MPC_H).astype(np.int64)
                origin = center_cell - CUBE_GRID // 2
                count, radial_velocity, kept = galaxy_input_grid(
                    center_position,
                    center_velocity,
                    origin,
                    galaxy_position,
                    galaxy_velocity,
                    galaxy_cell,
                )
                if realization not in density_cache:
                    path = Path(args.grid_pattern.format(realization=realization))
                    field, metadata = validated_cic_grid(path)
                    density_cache = {realization: field}
                    density_metadata[realization] = metadata
                density = density_cache[realization]
                mean_density = float(np.mean(density, dtype=np.float64))
                cube_density = extract_periodic_cube(density, origin)
                try:
                    target = log_cic_target(
                        cube_density, mean_density, DENSITY_SCALE
                    )
                except (ValueError, RuntimeError) as error:
                    raise RuntimeError(
                        f"invalid CIC target in CV_{realization}"
                    ) from error
                input_data[output_index, 0] = count
                input_data[output_index, 1] = radial_velocity
                target_data[output_index, 0] = target
                observer_index_data[output_index] = observer
                observer_position_data[output_index] = center_position
                observer_velocity_data[output_index] = center_velocity
                origin_data[output_index] = origin
                row = {
                    "realization": realization,
                    "observer_subhalo_index": observer,
                    "observer_stellar_mass": float(stellar_mass[observer]),
                    "galaxies_full_box": int(np.count_nonzero(galaxy)),
                    "galaxies_after_cube_and_mask": kept,
                    "occupied_cells": int(np.count_nonzero(count)),
                    "target_min": float(target.min()),
                    "target_max": float(target.max()),
                    "target_mean": float(target.mean()),
                    "target_std": float(target.std()),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
            handle.attrs["complete"] = True
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    report = {
        "schema": SCHEMA,
        "suite": args.suite,
        "role": args.role,
        "output": str(destination.resolve()),
        "samples": len(samples),
        "realizations": realizations,
        "density_metadata": density_metadata,
        "rows": rows,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("SIMBA", "Swift-EAGLE"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--realizations", required=True)
    parser.add_argument("--observers", choices=("single", "all"), required=True)
    parser.add_argument("--role", choices=("training", "development_validation", "historical_stress"), required=True)
    parser.add_argument("--grid-pattern", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
