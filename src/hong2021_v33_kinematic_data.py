#!/usr/bin/env python
"""Build frozen V33 three-channel kinematic development cubes.

The first two channels are reconstructed from the raw galaxy catalogue and
must match V14 element by element.  The only new value is the unbiased
within-voxel dispersion of the individual radial peculiar velocities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

from hong2021_prepare_simba import GALAXY_MASS_THRESHOLD, load_catalog
from hong2021_prepare_tng import indexed_files, load_subhalos
from hong2021_v18_init import sha256_file


PROGRAM_SCHEMA = "hong2021-v33-intrinsic-velocity-second-moment-program-v1"
PROGRAM_SHA256 = "7033b585d55bdb06b8eafd0626c8d899ce1ffde0c77e33ff55e7ddcf881c0407"
OUTPUT_SCHEMA = "hong2021-v33-intrinsic-velocity-second-moment-input-v1"
CHANNELS = (
    "Ngal,mean_radial_vpec_kms,"
    "intrinsic_within_voxel_radial_velocity_dispersion_kms"
)
CUBE_GRID = 64
CELL_MPC_H = 0.3125
MASK_ABS_B_DEG = 10.0


def load_program(path: Path, repo: Path) -> dict[str, Any]:
    if sha256_file(path.resolve()) != PROGRAM_SHA256:
        raise ValueError("V33 program hash differs")
    program = json.loads(path.read_text())
    if program.get("schema") != PROGRAM_SCHEMA:
        raise ValueError("V33 program schema differs")
    parent = program["parent_evidence"]
    record_path = (repo / parent["v32_record"]).resolve()
    if sha256_file(record_path) != parent["v32_record_sha256"]:
        raise ValueError("V33 V32 record hash differs")
    record = json.loads(record_path.read_text())
    audit = record.get("audit", {})
    if (
        audit.get("classification") != parent["required_classification"]
        or audit.get("next") != parent["required_next"]
        or record.get("firewall", {}).get("Astrid_accessed") is not False
        or record.get("firewall", {}).get("historical_EAGLE_accessed") is not False
    ):
        raise ValueError("V33 V32 parent conclusion or firewall differs")
    return program


def _update_logical_digest(
    digest: "hashlib._Hash", label: str, value: np.ndarray
) -> None:
    array = np.ascontiguousarray(value)
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape).encode("ascii") + b"\0")
    digest.update(memoryview(array).cast("B"))


def dataset_payload_sha256(dataset: h5py.Dataset) -> str:
    """Hash logical array values in leading-axis order, independent of HDF5."""
    digest = hashlib.sha256()
    digest.update(dataset.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(dataset.shape).encode("ascii") + b"\0")
    if dataset.ndim == 0:
        digest.update(np.asarray(dataset[()]).tobytes(order="C"))
    else:
        for index in range(dataset.shape[0]):
            digest.update(np.ascontiguousarray(dataset[index]).tobytes(order="C"))
    return digest.hexdigest()


def galaxy_input_grid_with_dispersion(
    center_position: np.ndarray,
    center_velocity: np.ndarray,
    origin_cell: np.ndarray,
    galaxy_position: np.ndarray,
    galaxy_velocity: np.ndarray,
    galaxy_cell: np.ndarray,
    *,
    simulation_box_mpc_h: float,
    full_grid: int,
    cube_grid: int = CUBE_GRID,
    mask_abs_b_deg: float = MASK_ABS_B_DEG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Grid count, mean radial velocity, and unbiased within-cell dispersion."""
    center_position = np.asarray(center_position)
    center_velocity = np.asarray(center_velocity)
    galaxy_position = np.asarray(galaxy_position)
    galaxy_velocity = np.asarray(galaxy_velocity)
    galaxy_cell = np.asarray(galaxy_cell)
    if (
        center_position.shape != (3,)
        or center_velocity.shape != (3,)
        or galaxy_position.ndim != 2
        or galaxy_position.shape[1] != 3
        or galaxy_velocity.shape != galaxy_position.shape
        or galaxy_cell.shape != galaxy_position.shape
    ):
        raise ValueError("invalid V33 catalogue array shape")

    local_cell = (galaxy_cell - np.asarray(origin_cell)[None, :]) % full_grid
    inside = np.all(local_cell < cube_grid, axis=1)
    displacement = (
        (galaxy_position - center_position + simulation_box_mpc_h / 2.0)
        % simulation_box_mpc_h
        - simulation_box_mpc_h / 2.0
    )
    radius = np.linalg.norm(displacement, axis=1)
    sin_latitude = np.divide(
        displacement[:, 2],
        radius,
        out=np.zeros_like(radius),
        where=radius > 0,
    )
    latitude = np.degrees(np.arcsin(np.clip(sin_latitude, -1.0, 1.0)))
    keep = inside & (radius > 0) & (np.abs(latitude) >= mask_abs_b_deg)
    local_cell = local_cell[keep].astype(np.int64)
    displacement = displacement[keep]
    radius = radius[keep]
    relative_velocity = galaxy_velocity[keep] - center_velocity
    radial_velocity = np.einsum(
        "ij,ij->i", relative_velocity, displacement / radius[:, None]
    )
    if not np.isfinite(radial_velocity).all():
        raise ValueError("non-finite V33 radial velocity")
    flat = (
        (local_cell[:, 0] * cube_grid + local_cell[:, 1]) * cube_grid
        + local_cell[:, 2]
    )
    size = cube_grid**3
    count = np.bincount(flat, minlength=size).astype(np.float32)
    velocity_sum = np.bincount(flat, weights=radial_velocity, minlength=size)
    velocity_square_sum = np.bincount(
        flat, weights=np.square(radial_velocity), minlength=size
    )
    mean_velocity = np.divide(
        velocity_sum,
        count,
        out=np.zeros_like(velocity_sum),
        where=count > 0,
    )
    centered_square_sum = velocity_square_sum - np.divide(
        np.square(velocity_sum),
        count,
        out=np.zeros_like(velocity_sum),
        where=count > 0,
    )
    sample_variance = np.divide(
        np.maximum(centered_square_sum, 0.0),
        count - 1.0,
        out=np.zeros_like(centered_square_sum),
        where=count >= 2,
    )
    shape = (cube_grid,) * 3
    return (
        count.reshape(shape),
        mean_velocity.astype(np.float32).reshape(shape),
        np.sqrt(sample_variance).astype(np.float32).reshape(shape),
        int(keep.sum()),
    )


def _copy_source_except_input(source: h5py.File, destination: h5py.File) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value
    for name in source:
        if name != "input":
            source.copy(name, destination, name=name)


def _verify_centers(
    source: h5py.File,
    row_indices: np.ndarray,
    catalogue_position: np.ndarray,
    catalogue_velocity: np.ndarray,
    catalogue_ids: np.ndarray,
) -> None:
    raw_position = catalogue_position[catalogue_ids]
    raw_velocity = catalogue_velocity[catalogue_ids]
    source_position = np.asarray(source["center_position_mpc_h"][row_indices])
    source_velocity = np.asarray(source["center_velocity_kms"][row_indices])
    if not np.array_equal(raw_position, source_position):
        raise ValueError("raw observer positions differ from V14")
    if not np.array_equal(raw_velocity, source_velocity):
        raise ValueError("raw observer velocities differ from V14")


def _tng_catalogue_rows(
    specification: dict[str, Any], source: h5py.File
) -> tuple[
    Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    "hashlib._Hash",
]:
    root = Path(specification["raw_catalogue_root"])
    files = indexed_files(root, "fof_subhalo_tab_099")
    if len(files) != specification["raw_catalogue_files_expected"]:
        raise ValueError("V33 TNG raw catalogue file count differs")
    subhalos = load_subhalos(files)
    if len(subhalos["id"]) != specification["raw_subhalos_expected"]:
        raise ValueError("V33 TNG raw subhalo count differs")
    selected = np.isfinite(subhalos["b_magnitude"]) & (subhalos["b_magnitude"] < -15.0)
    galaxy_position = np.asarray(subhalos["position"])[selected]
    galaxy_velocity = np.asarray(subhalos["velocity"])[selected]
    galaxy_cell = np.floor(galaxy_position / CELL_MPC_H).astype(np.int16)
    center_ids = np.asarray(source["center_subhalo_id"], dtype=np.int64)
    rows = np.arange(len(center_ids), dtype=np.int64)
    _verify_centers(
        source,
        rows,
        np.asarray(subhalos["position"]),
        np.asarray(subhalos["velocity"]),
        center_ids,
    )
    digest = hashlib.sha256()
    _update_logical_digest(digest, "TNG100:selected_position", galaxy_position)
    _update_logical_digest(digest, "TNG100:selected_velocity", galaxy_velocity)
    _update_logical_digest(digest, "TNG100:observer_ids", center_ids)

    def iterator() -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        for row, center_id in enumerate(center_ids):
            yield (
                row,
                np.asarray(subhalos["position"])[center_id],
                np.asarray(subhalos["velocity"])[center_id],
                galaxy_position,
                galaxy_velocity,
                galaxy_cell,
            )

    return iterator(), digest


def _camels_catalogue_rows(
    specification: dict[str, Any], source: h5py.File
) -> tuple[
    Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    "hashlib._Hash",
]:
    realizations = np.asarray(source["realization"], dtype=np.int64)
    center_ids = np.asarray(source["center_subhalo_id"], dtype=np.int64)
    expected_realizations = np.asarray(specification["realizations"], dtype=np.int64)
    if not np.array_equal(np.unique(realizations), expected_realizations):
        raise ValueError("V33 CAMELS realization membership differs")
    pattern = specification["raw_catalogue_pattern"]
    digest = hashlib.sha256()
    def iterator() -> Iterator[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        for realization in expected_realizations:
            rows = np.flatnonzero(realizations == realization)
            catalogue = load_catalog(
                Path(pattern.format(realization=int(realization)))
            )
            position = np.asarray(catalogue["position"])
            velocity = np.asarray(catalogue["velocity"])
            stellar_mass = np.asarray(catalogue["stellar_mass"])
            galaxy = stellar_mass >= GALAXY_MASS_THRESHOLD
            galaxy_position = position[galaxy]
            galaxy_velocity = velocity[galaxy]
            galaxy_cell = np.floor(galaxy_position / CELL_MPC_H).astype(np.int64)
            _verify_centers(source, rows, position, velocity, center_ids[rows])
            label = f"CV_{int(realization)}"
            _update_logical_digest(
                digest, f"{label}:selected_position", galaxy_position
            )
            _update_logical_digest(
                digest, f"{label}:selected_velocity", galaxy_velocity
            )
            _update_logical_digest(
                digest, f"{label}:observer_ids", center_ids[rows]
            )
            for row in rows:
                center_id = center_ids[row]
                yield (
                    int(row),
                    position[center_id],
                    velocity[center_id],
                    galaxy_position,
                    galaxy_velocity,
                    galaxy_cell,
                )

    return iterator(), digest


def build_dataset(
    program: dict[str, Any], domain: str, split: str
) -> dict[str, Any]:
    specification = program["datasets"][domain]
    artifact = specification[split]
    source_path = Path(artifact["source"])
    output_path = Path(artifact["output"])
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    report_path = output_path.with_suffix(".json")
    if sha256_file(source_path) != artifact["source_sha256"]:
        raise ValueError("V33 frozen source hash differs")
    if any(path.exists() for path in (output_path, partial_path, report_path)):
        raise FileExistsError(f"refusing to overwrite V33 output for {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count_max_abs_difference = 0.0
    velocity_max_abs_difference = 0.0
    total_galaxies = 0
    galaxies_in_multi_cells = 0
    nonzero_dispersion_cells = 0
    multi_cells = 0
    dispersion_sum_multi = 0.0
    maximum_dispersion = 0.0
    target_source_sha256 = ""
    catalogue_sha256 = ""
    try:
        with h5py.File(source_path, "r") as source, h5py.File(partial_path, "w") as destination:
            if source["input"].shape != (artifact["objects"], 2, CUBE_GRID, CUBE_GRID, CUBE_GRID):
                raise ValueError("V33 source input shape differs")
            _copy_source_except_input(source, destination)
            destination.attrs["schema"] = OUTPUT_SCHEMA
            destination.attrs["channels"] = CHANNELS
            destination.attrs["source_v14_data"] = str(source_path.resolve())
            destination.attrs["source_v14_sha256"] = artifact["source_sha256"]
            destination.attrs["intrinsic_velocity_dispersion_definition"] = (
                "equal-weight unbiased sample standard deviation of individual "
                "selected-galaxy radial peculiar velocities; zero for N<2"
            )
            destination.attrs["observational_sigma_mean_included"] = False
            destination.attrs["complete"] = False
            output = destination.create_dataset(
                "input",
                shape=(artifact["objects"], 3, CUBE_GRID, CUBE_GRID, CUBE_GRID),
                dtype="f4",
                chunks=(1, 3, CUBE_GRID, CUBE_GRID, CUBE_GRID),
                compression="lzf",
            )
            if domain == "TNG100":
                rows, catalogue_digest = _tng_catalogue_rows(specification, source)
                simulation_box_mpc_h, full_grid = 75.0, 240
            else:
                rows, catalogue_digest = _camels_catalogue_rows(
                    {**specification, **artifact}, source
                )
                simulation_box_mpc_h, full_grid = 25.0, 80
            seen = np.zeros(artifact["objects"], dtype=bool)
            for completed, row_data in enumerate(rows, start=1):
                row, center_position, center_velocity, galaxy_position, galaxy_velocity, galaxy_cell = row_data
                if seen[row]:
                    raise RuntimeError("V33 duplicate output row")
                seen[row] = True
                count, mean_velocity, dispersion, kept = galaxy_input_grid_with_dispersion(
                    center_position,
                    center_velocity,
                    np.asarray(source["cube_origin_cell"][row]),
                    galaxy_position,
                    galaxy_velocity,
                    galaxy_cell,
                    simulation_box_mpc_h=simulation_box_mpc_h,
                    full_grid=full_grid,
                )
                old_count = np.asarray(source["input"][row, 0])
                old_velocity = np.asarray(source["input"][row, 1])
                count_difference = float(np.max(np.abs(count - old_count)))
                velocity_difference = float(np.max(np.abs(mean_velocity - old_velocity)))
                count_max_abs_difference = max(count_max_abs_difference, count_difference)
                velocity_max_abs_difference = max(velocity_max_abs_difference, velocity_difference)
                if not np.array_equal(count, old_count):
                    raise ValueError(f"V33 {domain} row {row} count differs from V14")
                if not np.array_equal(mean_velocity, old_velocity):
                    raise ValueError(f"V33 {domain} row {row} mean velocity differs from V14")
                if (
                    not np.isfinite(dispersion).all()
                    or np.any(dispersion < 0)
                    or np.any(dispersion[count < 2] != 0)
                ):
                    raise ValueError("invalid V33 intrinsic velocity dispersion")
                output[row, 0] = count
                output[row, 1] = mean_velocity
                output[row, 2] = dispersion
                multiple = count >= 2
                total_galaxies += kept
                galaxies_in_multi_cells += int(np.rint(count[multiple].sum()))
                multi_cells += int(multiple.sum())
                nonzero_dispersion_cells += int((dispersion > 0).sum())
                dispersion_sum_multi += float(dispersion[multiple].sum(dtype=np.float64))
                maximum_dispersion = max(maximum_dispersion, float(dispersion.max()))
                if completed % 32 == 0 or completed == artifact["objects"]:
                    print(f"[v33:{domain}:{split}] {completed}/{artifact['objects']}", flush=True)
            if not seen.all():
                raise RuntimeError("V33 output row coverage incomplete")
            catalogue_sha256 = catalogue_digest.hexdigest()
            target_source_sha256 = dataset_payload_sha256(source["target"])
            target_output_sha256 = dataset_payload_sha256(destination["target"])
            if target_source_sha256 != target_output_sha256:
                raise ValueError("V33 target payload changed")
            destination.attrs["target_payload_sha256"] = target_source_sha256
            destination.attrs["raw_selected_catalogue_logical_sha256"] = catalogue_sha256
            destination.attrs["complete"] = True
        os.replace(partial_path, output_path)
    except BaseException:
        if partial_path.exists():
            partial_path.unlink()
        raise

    report = {
        "schema": OUTPUT_SCHEMA,
        "status": "complete",
        "program_sha256": PROGRAM_SHA256,
        "domain": domain,
        "split": split,
        "host": socket.gethostname(),
        "source": str(source_path.resolve()),
        "source_sha256": artifact["source_sha256"],
        "output": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
        "objects": artifact["objects"],
        "target_payload_sha256": target_source_sha256,
        "raw_selected_catalogue_logical_sha256": catalogue_sha256,
        "count_max_abs_difference_from_v14": count_max_abs_difference,
        "mean_velocity_max_abs_difference_from_v14_kms": velocity_max_abs_difference,
        "galaxies_after_cube_and_mask": total_galaxies,
        "galaxies_in_multi_galaxy_cells": galaxies_in_multi_cells,
        "galaxy_fraction_in_multi_galaxy_cells": galaxies_in_multi_cells / total_galaxies,
        "multi_galaxy_cells": multi_cells,
        "nonzero_dispersion_cells": nonzero_dispersion_cells,
        "mean_dispersion_in_multi_galaxy_cells_kms": dispersion_sum_multi / multi_cells,
        "maximum_dispersion_kms": maximum_dispersion,
        "observational_sigma_mean_included": False,
        "posthoc_Ak_used": False,
        "Astrid_accessed": False,
        "historical_EAGLE_accessed": False,
    }
    temporary_report = report_path.with_suffix(report_path.suffix + ".partial")
    temporary_report.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary_report, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--program",
        type=Path,
        default=Path("config/hong2021_v33_intrinsic_velocity_moment_program.json"),
    )
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--domain", choices=("TNG100", "SIMBA", "Swift"), required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    args = parser.parse_args()
    program = load_program(args.program, args.repo.resolve())
    expected_host = program["datasets"][args.domain]["execution_host"]
    actual_host = socket.gethostname().split(".")[0]
    if actual_host.lower() != expected_host.lower():
        raise SystemExit(
            f"V33 {args.domain} is frozen for {expected_host}, not {actual_host}"
        )
    print(json.dumps(build_dataset(program, args.domain, args.split), indent=2), flush=True)


if __name__ == "__main__":
    main()
