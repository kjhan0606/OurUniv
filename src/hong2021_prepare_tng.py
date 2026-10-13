#!/usr/bin/env python
"""Prepare the TNG100-1 training cubes used by Hong et al. (2021).

The full z=0 dark-matter particle set is streamed once into a periodic
``240^3`` count grid.  This is the unique box-wide grid with the paper's
0.3125 Mpc/h cell size.  Each 20 Mpc/h sample is then a ``64^3`` periodic
sub-cube of that grid, with galaxy count and center-relative radial peculiar
velocity gridded on exactly the same cells.

The paper states only that validation cubes do not overlap training cubes; it
does not publish the split membership or random seed.  This implementation
therefore freezes a deterministic spatial split: 93 validation observers are
the nearest periodic neighbors of the anchor that leaves the largest possible
non-overlapping training pool, and 432 training observers are selected from
that pool with seed 2021.  The exact IDs are stored in the HDF5 files.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_TNG_ROOT = Path("/scratch/kjhan/IllustrisTNG/TNG100-1")
BOX_MPC_H = 75.0
BOX_KPC_H = 75_000.0
CELL_MPC_H = 0.3125
FULL_GRID = 240
CUBE_GRID = 64
CUBE_MPC_H = 20.0
N_TRAIN = 432
N_VALIDATION = 93
DENSITY_SCALE = 4.5
PAPER_ATTRIBUTES = {
    "paper": "Hong et al. 2021 ApJ 913 76",
    "voxel_mpc_h": CELL_MPC_H,
    "channels": "Ngal,mean_radial_vpec_kms",
    "galactic_mask_abs_b_deg": 10.0,
}


def indexed_files(directory: Path, stem: str) -> list[Path]:
    return sorted(
        directory.glob(f"{stem}.*.hdf5"),
        key=lambda path: int(path.name.split(".")[-2]),
    )


def load_subhalos(group_files: list[Path]) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {
        "id": [],
        "position": [],
        "velocity": [],
        "stellar_mass_paper": [],
        "b_magnitude": [],
    }
    offset = 0
    for path in group_files:
        with h5py.File(path, "r") as handle:
            nsub = int(handle["Header"].attrs["Nsubgroups_ThisFile"])
            if not nsub:
                continue
            subhalo = handle["Subhalo"]
            pieces["id"].append(np.arange(offset, offset + nsub, dtype=np.int64))
            pieces["position"].append(
                subhalo["SubhaloPos"][:].astype(np.float32) / 1000.0
            )
            pieces["velocity"].append(
                subhalo["SubhaloVel"][:].astype(np.float32)
            )
            # This intentionally follows the numerical cut that produces the
            # 988 observers reported in the paper.  TNG documents the stored
            # mass in 1e10 Msun/h; applying an additional 1/h gives 1,552.
            pieces["stellar_mass_paper"].append(
                subhalo["SubhaloMassType"][:, 4].astype(np.float64) * 1.0e10
            )
            pieces["b_magnitude"].append(
                subhalo["SubhaloStellarPhotometrics"][:, 1].astype(np.float32)
            )
            offset += nsub
    result = {
        name: np.concatenate(values) for name, values in pieces.items()
    }
    if offset != 4_371_211:
        raise RuntimeError(f"expected 4,371,211 subhalos, found {offset:,}")
    return result


def periodic_abs_delta(
    first: np.ndarray, second: np.ndarray, box_mpc_h: float = BOX_MPC_H
) -> np.ndarray:
    delta = np.abs(first - second)
    return np.minimum(delta, box_mpc_h - delta)


def choose_spatial_split(
    positions: np.ndarray,
    n_train: int = N_TRAIN,
    n_validation: int = N_VALIDATION,
    seed: int = 2021,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Choose a deterministic train/validation split with no cube overlap."""
    positions = np.asarray(positions, dtype=np.float64)
    if len(positions) < n_train + n_validation:
        raise ValueError("not enough center candidates")

    best: tuple[int, int, np.ndarray, np.ndarray] | None = None
    for anchor in range(len(positions)):
        delta = periodic_abs_delta(positions, positions[anchor])
        distance2 = np.einsum("ij,ij->i", delta, delta)
        validation = np.argsort(distance2, kind="stable")[:n_validation]
        cross_delta = periodic_abs_delta(
            positions[:, None, :], positions[validation][None, :, :]
        )
        overlaps_validation = np.any(
            np.all(cross_delta < CUBE_MPC_H, axis=2), axis=1
        )
        available = np.flatnonzero(~overlaps_validation)
        score = len(available)
        if best is None or score > best[0]:
            best = score, anchor, validation, available

    assert best is not None
    available_count, anchor, validation, available = best
    if available_count < n_train:
        raise RuntimeError(
            f"only {available_count} non-overlapping training centers; "
            f"{n_train} required"
        )
    generator = np.random.default_rng(seed)
    training = generator.permutation(available)[:n_train]

    cross_delta = periodic_abs_delta(
        positions[training][:, None, :],
        positions[validation][None, :, :],
    )
    minimum_separation = np.min(np.max(cross_delta, axis=2))
    if np.any(np.all(cross_delta < CUBE_MPC_H, axis=2)):
        raise RuntimeError("internal error: train and validation cubes overlap")
    metadata = {
        "algorithm": (
            "nearest-93 periodic cluster around the candidate anchor maximizing "
            "the non-overlapping pool; seeded selection of 432 from that pool"
        ),
        "seed": seed,
        "anchor_candidate_index": int(anchor),
        "available_nonoverlapping_training_centers": int(available_count),
        "minimum_cross_split_Linf_separation_mpc_h": float(minimum_separation),
        "discarded_candidates": int(len(positions) - n_train - n_validation),
    }
    return training, validation, metadata


def build_dark_matter_grid(
    snapshot_files: list[Path],
    derived: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read every DM coordinate and atomically cache a 240^3 count grid."""
    final_path = derived / "dm_counts_240.npy"
    metadata_path = derived / "dm_counts_240.json"
    expected_particles = 1820**3
    if final_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        counts = np.load(final_path, mmap_mode="r")
        if (
            counts.shape == (FULL_GRID,) * 3
            and counts.dtype == np.uint32
            and int(counts.sum(dtype=np.uint64)) == expected_particles
            and metadata.get("complete") is True
        ):
            print(f"[density] using validated cache {final_path}", flush=True)
            return counts, metadata
        raise RuntimeError(f"invalid existing density cache: {final_path}")

    partial_path = derived / "dm_counts_240.partial.npy"
    if partial_path.exists():
        raise RuntimeError(
            f"incomplete prior density pass exists: {partial_path}; "
            "move it aside before restarting"
        )

    counts = np.zeros((FULL_GRID,) * 3, dtype=np.uint32)
    start = time.time()
    particles_read = 0
    for file_number, path in enumerate(snapshot_files, start=1):
        with h5py.File(path, "r") as handle:
            coordinates = handle["PartType1/Coordinates"][:]
        if not np.isfinite(coordinates).all():
            raise RuntimeError(f"non-finite DM coordinate in {path}")
        cells = np.floor(coordinates / (BOX_KPC_H / FULL_GRID)).astype(
            np.int16
        )
        if cells.min() < 0 or cells.max() >= FULL_GRID:
            raise RuntimeError(f"DM coordinate outside periodic box in {path}")
        flat = (
            (cells[:, 0].astype(np.int64) * FULL_GRID + cells[:, 1])
            * FULL_GRID
            + cells[:, 2]
        )
        histogram = np.bincount(flat, minlength=FULL_GRID**3)
        if histogram.max() > np.iinfo(np.uint32).max:
            raise RuntimeError("single-chunk cell count overflow")
        counts += histogram.reshape(counts.shape).astype(np.uint32)
        particles_read += len(coordinates)
        elapsed = time.time() - start
        print(
            f"[density] {file_number:03d}/{len(snapshot_files)} "
            f"particles={particles_read:,} elapsed={elapsed:.0f}s",
            flush=True,
        )

    total = int(counts.sum(dtype=np.uint64))
    if total != expected_particles or particles_read != expected_particles:
        raise RuntimeError(
            f"DM total mismatch: grid={total:,}, read={particles_read:,}, "
            f"expected={expected_particles:,}"
        )
    np.save(partial_path, counts)
    os.replace(partial_path, final_path)
    metadata = {
        "complete": True,
        "source_files": len(snapshot_files),
        "particles": total,
        "grid": FULL_GRID,
        "box_mpc_h": BOX_MPC_H,
        "voxel_mpc_h": CELL_MPC_H,
        "mean_particles_per_voxel": total / FULL_GRID**3,
        "minimum_particles_per_voxel": int(counts.min()),
        "maximum_particles_per_voxel": int(counts.max()),
        "elapsed_seconds": time.time() - start,
        "payload_validation": (
            "all PartType1/Coordinates values read, finite, in box, and "
            "particle-count conserving"
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return np.load(final_path, mmap_mode="r"), metadata


def extract_periodic_cube(
    field: np.ndarray, origin_cell: np.ndarray, size: int = CUBE_GRID
) -> np.ndarray:
    indices = [
        (np.arange(size, dtype=np.int64) + int(origin_cell[axis]))
        % field.shape[axis]
        for axis in range(3)
    ]
    return np.asarray(field[np.ix_(*indices)])


def galaxy_input_grid(
    center_position: np.ndarray,
    center_velocity: np.ndarray,
    origin_cell: np.ndarray,
    galaxy_position: np.ndarray,
    galaxy_velocity: np.ndarray,
    galaxy_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
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


def write_training_file(
    destination: Path,
    selected: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_position: np.ndarray,
    candidate_velocity: np.ndarray,
    galaxy_position: np.ndarray,
    galaxy_velocity: np.ndarray,
    galaxy_cell: np.ndarray,
    dm_counts: np.ndarray,
    split: str,
    split_metadata: dict[str, Any],
) -> dict[str, Any]:
    partial = destination.with_suffix(destination.suffix + ".partial")
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite existing file: {destination}")
    if partial.exists():
        raise RuntimeError(f"incomplete prior output exists: {partial}")

    mean_count = float(dm_counts.sum(dtype=np.uint64) / dm_counts.size)
    occupied_galaxies: list[int] = []
    target_min = float("inf")
    target_max = float("-inf")
    with h5py.File(partial, "w") as handle:
        for key, value in PAPER_ATTRIBUTES.items():
            handle.attrs[key] = value
        handle.attrs["box_mpc_h"] = CUBE_MPC_H
        handle.attrs["density_scale"] = DENSITY_SCALE
        handle.attrs["split"] = split
        handle.attrs["split_algorithm"] = split_metadata["algorithm"]
        handle.attrs["split_seed"] = split_metadata["seed"]
        handle.attrs["grid_alignment"] = (
            "periodic global 240^3 grid; observer lies in cell 32"
        )
        handle.attrs["stellar_mass_selection_note"] = (
            "SubhaloMassType[:,4]*1e10 without dividing by h, reproducing "
            "the paper's 988 candidates"
        )
        n = len(selected)
        x_data = handle.create_dataset(
            "input",
            shape=(n, 2, CUBE_GRID, CUBE_GRID, CUBE_GRID),
            dtype="f4",
            chunks=(1, 2, CUBE_GRID, CUBE_GRID, CUBE_GRID),
            compression="lzf",
        )
        y_data = handle.create_dataset(
            "target",
            shape=(n, 1, CUBE_GRID, CUBE_GRID, CUBE_GRID),
            dtype="f4",
            chunks=(1, 1, CUBE_GRID, CUBE_GRID, CUBE_GRID),
            compression="lzf",
        )
        handle.create_dataset("center_subhalo_id", data=candidate_ids[selected])
        handle.create_dataset(
            "center_position_mpc_h", data=candidate_position[selected]
        )
        handle.create_dataset(
            "center_velocity_kms", data=candidate_velocity[selected]
        )
        origins = handle.create_dataset(
            "cube_origin_cell", shape=(n, 3), dtype="i2"
        )

        for output_index, candidate_index in enumerate(selected):
            center_position = candidate_position[candidate_index]
            center_velocity = candidate_velocity[candidate_index]
            center_cell = np.floor(center_position / CELL_MPC_H).astype(
                np.int64
            )
            origin = center_cell - CUBE_GRID // 2
            count, velocity, ngal = galaxy_input_grid(
                center_position,
                center_velocity,
                origin,
                galaxy_position,
                galaxy_velocity,
                galaxy_cell,
            )
            density_count = extract_periodic_cube(dm_counts, origin).astype(
                np.float32
            )
            if density_count.min() <= 0:
                raise RuntimeError("zero-DM voxel cannot be log transformed")
            target = np.log10(density_count / mean_count) / DENSITY_SCALE
            if target.min() < -1.0 or target.max() > 1.0:
                raise RuntimeError("target outside published tanh range")
            x_data[output_index, 0] = count
            x_data[output_index, 1] = velocity
            y_data[output_index, 0] = target
            origins[output_index] = origin
            occupied_galaxies.append(ngal)
            target_min = min(target_min, float(target.min()))
            target_max = max(target_max, float(target.max()))
            if (output_index + 1) % 32 == 0 or output_index + 1 == n:
                print(
                    f"[cubes:{split}] {output_index + 1}/{n}",
                    flush=True,
                )

    os.replace(partial, destination)
    return {
        "path": str(destination),
        "samples": len(selected),
        "target_minmax": [target_min, target_max],
        "galaxies_after_mask_min_median_max": [
            int(np.min(occupied_galaxies)),
            float(np.median(occupied_galaxies)),
            int(np.max(occupied_galaxies)),
        ],
        "bytes": destination.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tng-root", type=Path, default=DEFAULT_TNG_ROOT)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()
    output = args.tng_root / "output"
    derived = args.tng_root / "derived/hong2021"
    derived.mkdir(parents=True, exist_ok=True)
    group_files = indexed_files(output / "groups_099", "fof_subhalo_tab_099")
    snapshot_files = indexed_files(output / "snapdir_099", "snap_099")
    if len(group_files) != 448 or len(snapshot_files) != 448:
        raise SystemExit("expected 448 groupcat and 448 snapshot chunks")

    subhalos = load_subhalos(group_files)
    center_mask = (
        (subhalos["stellar_mass_paper"] > 4.0e10)
        & (subhalos["stellar_mass_paper"] < 1.0e11)
    )
    target_mask = (
        np.isfinite(subhalos["b_magnitude"])
        & (subhalos["b_magnitude"] < -15.0)
    )
    candidate_ids = subhalos["id"][center_mask]
    candidate_position = subhalos["position"][center_mask]
    candidate_velocity = subhalos["velocity"][center_mask]
    galaxy_position = subhalos["position"][target_mask]
    galaxy_velocity = subhalos["velocity"][target_mask]
    if len(candidate_ids) != 988:
        raise RuntimeError(f"expected 988 center candidates, found {len(candidate_ids)}")
    print(
        f"[catalog] centers={len(candidate_ids)} target_galaxies="
        f"{len(galaxy_position)}",
        flush=True,
    )

    training, validation, split_metadata = choose_spatial_split(
        candidate_position, seed=args.seed
    )
    split_metadata["training_candidate_indices"] = training.tolist()
    split_metadata["validation_candidate_indices"] = validation.tolist()
    (derived / "split_v1.json").write_text(
        json.dumps(split_metadata, indent=2) + "\n"
    )
    print(f"[split] {json.dumps(split_metadata, indent=2)}", flush=True)

    dm_counts, density_metadata = build_dark_matter_grid(
        snapshot_files, derived
    )
    galaxy_cell = np.floor(galaxy_position / CELL_MPC_H).astype(np.int16)
    train_report = write_training_file(
        derived / "tng100_train.h5",
        training,
        candidate_ids,
        candidate_position,
        candidate_velocity,
        galaxy_position,
        galaxy_velocity,
        galaxy_cell,
        dm_counts,
        "train",
        split_metadata,
    )
    validation_report = write_training_file(
        derived / "tng100_validation.h5",
        validation,
        candidate_ids,
        candidate_position,
        candidate_velocity,
        galaxy_position,
        galaxy_velocity,
        galaxy_cell,
        dm_counts,
        "validation",
        split_metadata,
    )
    report = {
        "simulation": "TNG100-1",
        "snapshot": 99,
        "center_candidates": len(candidate_ids),
        "target_galaxies": len(galaxy_position),
        "split": split_metadata,
        "density": density_metadata,
        "train": train_report,
        "validation": validation_report,
    }
    (derived / "preparation_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
