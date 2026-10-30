#!/usr/bin/env python
"""Build an independent CAMELS-SIMBA Hong-style evaluation set.

EAGLE particle access is currently unavailable, so this is explicitly a
provisional cross-code gate rather than an EAGLE replacement.  All selection
and normalization choices are frozen from TNG before inspecting SIMBA dark
matter truth.  One observer is used from each independent SIMBA CV box.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA = "hong2021-independent-simba-cv-input-v1"
BOX_MPC_H = 25.0
CUBE_MPC_H = 20.0
CUBE_GRID = 64
WORK_GRID = 80
SOURCE_GRID = 256
CELL_MPC_H = CUBE_MPC_H / CUBE_GRID
DENSITY_SCALE = 4.5
MASK_LATITUDE_DEG = 10.0
OBSERVER_MASS = (4.0e10, 1.0e11)
GALAXY_MASS_THRESHOLD = 97_664_166.24188423


def overlap_matrix(source: int, destination: int) -> np.ndarray:
    """Return exact piecewise-constant cell-average weights on a unit line."""
    if source <= 0 or destination <= 0:
        raise ValueError("grid sizes must be positive")
    source_edges = np.arange(source + 1, dtype=np.float64) / source
    destination_edges = np.arange(destination + 1, dtype=np.float64) / destination
    left = np.maximum(destination_edges[:-1, None], source_edges[None, :-1])
    right = np.minimum(destination_edges[1:, None], source_edges[None, 1:])
    overlap = np.maximum(right - left, 0.0)
    weights = overlap * destination
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=2.0e-14):
        raise RuntimeError("invalid overlap weights")
    return weights


def conservative_resample_cube(
    density: np.ndarray, destination: int = WORK_GRID
) -> np.ndarray:
    """Volume-average a cubic density grid while preserving the box mean."""
    value = np.asarray(density)
    if value.ndim != 3 or len(set(value.shape)) != 1:
        raise ValueError(f"expected cubic 3-D field, got {value.shape}")
    if not np.isfinite(value).all() or np.any(value < 0):
        raise ValueError("density must be finite and non-negative")
    weights = overlap_matrix(value.shape[0], destination)
    # Each contraction acts only on one dimension.  Moving the contracted
    # axis back into place keeps the physical x,y,z ordering unchanged.
    result = np.tensordot(weights, value, axes=(1, 0))
    result = np.moveaxis(np.tensordot(weights, result, axes=(1, 1)), 0, 1)
    result = np.moveaxis(np.tensordot(weights, result, axes=(1, 2)), 0, 2)
    result = np.asarray(result, dtype=np.float32)
    source_mean = float(np.mean(value, dtype=np.float64))
    destination_mean = float(np.mean(result, dtype=np.float64))
    if not np.isclose(source_mean, destination_mean, rtol=2.0e-6, atol=0.0):
        raise RuntimeError(
            f"resampling changed box mean: {source_mean} -> {destination_mean}"
        )
    return result


def extract_periodic_cube(
    field: np.ndarray, origin_cell: np.ndarray, size: int = CUBE_GRID
) -> np.ndarray:
    indices = [
        (np.arange(size, dtype=np.int64) + int(origin_cell[axis]))
        % field.shape[axis]
        for axis in range(3)
    ]
    return np.asarray(field[np.ix_(*indices)])


def choose_observer(stellar_mass: np.ndarray) -> int:
    """Choose by mass alone, with a deterministic index tie breaker."""
    mass = np.asarray(stellar_mass, dtype=np.float64)
    candidates = np.flatnonzero(
        (mass > OBSERVER_MASS[0]) & (mass < OBSERVER_MASS[1])
    )
    if len(candidates) == 0:
        raise RuntimeError("no SIMBA observer in the frozen stellar-mass interval")
    midpoint = np.sqrt(OBSERVER_MASS[0] * OBSERVER_MASS[1])
    distance = np.abs(np.log(mass[candidates] / midpoint))
    return int(candidates[np.argmin(distance)])


def observer_candidates(stellar_mass: np.ndarray) -> np.ndarray:
    mass = np.asarray(stellar_mass, dtype=np.float64)
    return np.flatnonzero(
        (mass > OBSERVER_MASS[0]) & (mass < OBSERVER_MASS[1])
    ).astype(np.int64)


def galaxy_input_grid(
    center_position: np.ndarray,
    center_velocity: np.ndarray,
    origin_cell: np.ndarray,
    galaxy_position: np.ndarray,
    galaxy_velocity: np.ndarray,
    galaxy_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    local_cell = (galaxy_cell - origin_cell[None, :]) % WORK_GRID
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
    keep = inside & (radius > 0) & (np.abs(latitude) >= MASK_LATITUDE_DEG)
    local_cell = local_cell[keep].astype(np.int64)
    displacement = displacement[keep]
    radius = radius[keep]
    relative_velocity = galaxy_velocity[keep] - center_velocity
    radial_velocity = np.einsum(
        "ij,ij->i", relative_velocity, displacement / radius[:, None]
    )
    flat = (
        (local_cell[:, 0] * CUBE_GRID + local_cell[:, 1]) * CUBE_GRID
        + local_cell[:, 2]
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


def load_catalog(path: Path) -> dict[str, np.ndarray | float]:
    with h5py.File(path, "r") as handle:
        header = handle["Header"].attrs
        box = float(header["BoxSize"]) / 1000.0
        redshift = float(header["Redshift"])
        if not np.isclose(box, BOX_MPC_H) or abs(redshift) > 1.0e-6:
            raise ValueError(f"unexpected catalogue geometry: box={box}, z={redshift}")
        subhalo = handle["Subhalo"]
        return {
            "position": np.asarray(subhalo["SubhaloPos"], dtype=np.float64)
            / 1000.0,
            "velocity": np.asarray(subhalo["SubhaloVel"], dtype=np.float64),
            "stellar_mass": np.asarray(
                subhalo["SubhaloMassType"][:, 4], dtype=np.float64
            )
            * 1.0e10,
            "hubble": float(header["HubbleParam"]),
        }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    destination = Path(args.out)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists() or temporary.exists():
        raise RuntimeError(f"refusing to overwrite {destination} or {temporary}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = np.load(args.dm_grid, mmap_mode="r")
    expected = (27, SOURCE_GRID, SOURCE_GRID, SOURCE_GRID)
    if fields.shape != expected:
        raise ValueError(f"unexpected CMD grid shape {fields.shape}, expected {expected}")
    realizations = [int(value) for value in args.realizations.split(",")]
    if len(realizations) != len(set(realizations)) or not realizations:
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
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(
                {
                    "schema": SCHEMA,
                    "simulation": "CAMELS-SIMBA-CV",
                    "provisional_independent_gate": True,
                    "box_mpc_h": CUBE_MPC_H,
                    "simulation_box_mpc_h": BOX_MPC_H,
                    "voxel_mpc_h": CELL_MPC_H,
                    "density_scale": DENSITY_SCALE,
                    "channels": "Ngal,mean_radial_vpec_kms",
                    "galactic_mask_abs_b_deg": MASK_LATITUDE_DEG,
                    "galaxy_stellar_mass_threshold": GALAXY_MASS_THRESHOLD,
                    "galaxy_selection_note": (
                        "TNG-only Mstar rank threshold matched to the global "
                        "TNG M_B<-15 count; fixed before SIMBA density inspection"
                    ),
                    "target_definition": "log10(rho_cdm/rho_cdm_box_mean)/4.5",
                    "resampling": (
                        "exact separable volume-overlap average 256^3 to 80^3"
                    ),
                    "observer_selection": args.observers,
                }
            )
            shape = (len(samples), 2, CUBE_GRID, CUBE_GRID, CUBE_GRID)
            input_data = handle.create_dataset(
                "input",
                shape=shape,
                dtype="f4",
                chunks=(1, 2, CUBE_GRID, CUBE_GRID, CUBE_GRID),
                compression="lzf",
            )
            target_data = handle.create_dataset(
                "target",
                shape=(len(samples), 1, CUBE_GRID, CUBE_GRID, CUBE_GRID),
                dtype="f4",
                chunks=(1, 1, CUBE_GRID, CUBE_GRID, CUBE_GRID),
                compression="lzf",
            )
            handle.create_dataset(
                "realization", data=np.asarray([value[0] for value in samples])
            )
            observer_index_data = handle.create_dataset(
                "center_subhalo_id", shape=(len(samples),), dtype="i8"
            )
            observer_position_data = handle.create_dataset(
                "center_position_mpc_h", shape=(len(samples), 3), dtype="f8"
            )
            observer_velocity_data = handle.create_dataset(
                "center_velocity_kms", shape=(len(samples), 3), dtype="f8"
            )
            origin_data = handle.create_dataset(
                "cube_origin_cell", shape=(len(samples), 3), dtype="i2"
            )

            density_cache: dict[int, np.ndarray] = {}
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
                    density_cache = {
                        realization: conservative_resample_cube(
                            fields[realization], WORK_GRID
                        )
                    }
                density = density_cache[realization]
                mean_density = float(np.mean(density, dtype=np.float64))
                cube_density = extract_periodic_cube(density, origin)
                if mean_density <= 0 or np.any(cube_density <= 0):
                    raise RuntimeError(f"non-positive DM density in CV_{realization}")
                target = np.log10(cube_density / mean_density) / DENSITY_SCALE
                if not np.isfinite(target).all():
                    raise RuntimeError(f"non-finite target in CV_{realization}")
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
                    "source_density_mean": float(
                        np.mean(fields[realization], dtype=np.float64)
                    ),
                    "work_density_mean": mean_density,
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
        "output": str(destination.resolve()),
        "samples": len(samples),
        "rows": rows,
    }
    report_path = destination.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="/gpfs/kjhan/CAMELS/SIMBA/L25n256"
    )
    parser.add_argument(
        "--dm-grid",
        default=(
            "/gpfs/kjhan/CAMELS/SIMBA/L25n256/CMD/"
            "Grids_Mcdm_SIMBA_CV_256_z=0.0.npy"
        ),
    )
    parser.add_argument("--realizations", default=",".join(map(str, range(16))))
    parser.add_argument("--observers", choices=("single", "all"), default="single")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
