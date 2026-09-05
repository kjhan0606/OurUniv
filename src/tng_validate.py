#!/usr/bin/env python
"""Validate the local TNG100-1 snapshot-99 download used by Hong et al.

The check is deliberately restricted to data needed by this project.  It
verifies the downloader manifests, file numbering and byte sizes, opens every
HDF5 chunk, checks header consistency, and checks the shapes of the dark-matter
coordinates and required subhalo fields.  Reading every dark-matter coordinate
is deferred to ``hong2021_prepare_tng.py`` because that pass also constructs the
density grid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_OUTPUT = Path("/scratch/kjhan/IllustrisTNG/TNG100-1/output")
SNAPSHOT_FIELDS = ("PartType1/Coordinates",)
SUBHALO_FIELDS = (
    "SubhaloPos",
    "SubhaloVel",
    "SubhaloMassType",
    "SubhaloStellarPhotometrics",
)


def indexed_files(directory: Path, stem: str) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in directory.glob(f"{stem}.*.hdf5"):
        found.append((int(path.name.split(".")[-2]), path))
    return sorted(found)


def manifest_sizes(path: Path) -> tuple[dict[int, int], dict[str, Any]]:
    manifest = json.loads(path.read_text())
    sizes = {
        int(item["index"]): int(item["size_bytes"])
        for item in manifest["results"]
    }
    return sizes, manifest


def validate_snapshot(output: Path) -> dict[str, Any]:
    directory = output / "snapdir_099"
    files = indexed_files(directory, "snap_099")
    expected_sizes, manifest = manifest_sizes(
        directory / "download_snapshot_manifest.json"
    )
    failures: list[str] = []
    summed = np.zeros(6, dtype=np.uint64)
    header_total: np.ndarray | None = None
    redshifts: list[float] = []
    numfiles: set[int] = set()

    indices = [index for index, _ in files]
    if indices != list(range(448)):
        failures.append("snapshot_file_indices_not_0_through_447")
    if manifest.get("files_complete") != 448:
        failures.append("snapshot_manifest_not_complete")

    for index, path in files:
        if path.stat().st_size != expected_sizes.get(index):
            failures.append(f"snapshot_size_mismatch:{index}")
        try:
            with h5py.File(path, "r") as handle:
                attrs = handle["Header"].attrs
                this_file = np.asarray(
                    attrs["NumPart_ThisFile"], dtype=np.uint64
                )
                low = np.asarray(attrs["NumPart_Total"], dtype=np.uint64)
                high = np.asarray(
                    attrs["NumPart_Total_HighWord"], dtype=np.uint64
                )
                current_total = low + (high << np.uint64(32))
                if header_total is None:
                    header_total = current_total
                elif not np.array_equal(current_total, header_total):
                    failures.append(f"snapshot_total_header_mismatch:{index}")
                summed += this_file
                numfiles.add(int(attrs["NumFilesPerSnapshot"]))
                redshifts.append(float(attrs["Redshift"]))
                if this_file[1]:
                    dataset = handle.get(SNAPSHOT_FIELDS[0])
                    if dataset is None:
                        failures.append(f"missing_dm_coordinates:{index}")
                    elif dataset.shape != (int(this_file[1]), 3):
                        failures.append(f"dm_coordinate_shape:{index}")
        except Exception as error:  # a corrupt HDF5 chunk must not stop the scan
            failures.append(f"snapshot_open:{index}:{error!r}")

    if header_total is None or not np.array_equal(summed, header_total):
        failures.append("snapshot_particle_sum_mismatch")
    if numfiles != {448}:
        failures.append("snapshot_NumFilesPerSnapshot_not_448")
    if not redshifts or not np.allclose(redshifts, 0.0, atol=1.0e-12):
        failures.append("snapshot_not_z0")

    return {
        "pass": not failures,
        "directory": str(directory),
        "files": len(files),
        "manifest_bytes": int(manifest.get("bytes_complete", -1)),
        "summed_NumPart_ThisFile": summed.tolist(),
        "header_NumPart_Total": (
            header_total.tolist() if header_total is not None else None
        ),
        "dark_matter_particles": int(summed[1]),
        "dark_matter_expected_1820_cubed": 1820**3,
        "failures": failures,
    }


def validate_groupcat(output: Path) -> dict[str, Any]:
    directory = output / "groups_099"
    files = indexed_files(directory, "fof_subhalo_tab_099")
    expected_sizes, manifest = manifest_sizes(
        directory / "download_groupcat_manifest.json"
    )
    failures: list[str] = []
    summed = np.zeros(3, dtype=np.uint64)
    header_total: tuple[int, int, int] | None = None
    redshifts: list[float] = []
    numfiles: set[int] = set()
    center_candidates = 0
    target_galaxies = 0

    indices = [index for index, _ in files]
    if indices != list(range(448)):
        failures.append("groupcat_file_indices_not_0_through_447")
    if manifest.get("files_complete") != 448:
        failures.append("groupcat_manifest_not_complete")

    for index, path in files:
        if path.stat().st_size != expected_sizes.get(index):
            failures.append(f"groupcat_size_mismatch:{index}")
        try:
            with h5py.File(path, "r") as handle:
                attrs = handle["Header"].attrs
                this_file = np.array(
                    [
                        attrs["Ngroups_ThisFile"],
                        attrs["Nsubgroups_ThisFile"],
                        attrs["Nids_ThisFile"],
                    ],
                    dtype=np.uint64,
                )
                current_total = (
                    int(attrs["Ngroups_Total"]),
                    int(attrs["Nsubgroups_Total"]),
                    int(attrs["Nids_Total"]),
                )
                if header_total is None:
                    header_total = current_total
                elif current_total != header_total:
                    failures.append(f"groupcat_total_header_mismatch:{index}")
                summed += this_file
                numfiles.add(int(attrs["NumFiles"]))
                redshifts.append(float(attrs["Redshift"]))

                nsub = int(this_file[1])
                if nsub:
                    subhalo = handle["Subhalo"]
                    for field in SUBHALO_FIELDS:
                        if field not in subhalo:
                            failures.append(f"missing_{field}:{index}")
                    if all(field in subhalo for field in SUBHALO_FIELDS):
                        for field in ("SubhaloPos", "SubhaloVel"):
                            if subhalo[field].shape != (nsub, 3):
                                failures.append(f"{field}_shape:{index}")
                        if subhalo["SubhaloMassType"].shape != (nsub, 6):
                            failures.append(f"SubhaloMassType_shape:{index}")
                        if subhalo["SubhaloStellarPhotometrics"].shape != (
                            nsub,
                            8,
                        ):
                            failures.append(
                                f"SubhaloStellarPhotometrics_shape:{index}"
                            )

                        # The factor below exactly reproduces the paper's 988
                        # reported observer galaxies.  Dividing by h would give
                        # 1,552 and therefore does not reproduce their cut.
                        stellar_mass = (
                            subhalo["SubhaloMassType"][:, 4].astype(np.float64)
                            * 1.0e10
                        )
                        center_candidates += int(
                            (
                                (stellar_mass > 4.0e10)
                                & (stellar_mass < 1.0e11)
                            ).sum()
                        )
                        b_magnitude = subhalo[
                            "SubhaloStellarPhotometrics"
                        ][:, 1]
                        target_galaxies += int(
                            (np.isfinite(b_magnitude) & (b_magnitude < -15.0)).sum()
                        )
        except Exception as error:
            failures.append(f"groupcat_open:{index}:{error!r}")

    if header_total is None or tuple(int(v) for v in summed) != header_total:
        failures.append("groupcat_object_sum_mismatch")
    if numfiles != {448}:
        failures.append("groupcat_NumFiles_not_448")
    if not redshifts or not np.allclose(redshifts, 0.0, atol=1.0e-12):
        failures.append("groupcat_not_z0")
    if center_candidates != 988:
        failures.append("center_candidate_count_not_paper_988")

    return {
        "pass": not failures,
        "directory": str(directory),
        "files": len(files),
        "manifest_bytes": int(manifest.get("bytes_complete", -1)),
        "summed_groups_subhalos_ids": summed.tolist(),
        "header_groups_subhalos_ids": header_total,
        "paper_center_candidates": center_candidates,
        "target_galaxies_MB_lt_minus15": target_galaxies,
        "failures": failures,
    }


def build_report(output: Path) -> dict[str, Any]:
    snapshot = validate_snapshot(output)
    groupcat = validate_groupcat(output)
    return {
        "simulation": "TNG100-1",
        "snapshot": 99,
        "validation_scope": (
            "manifests, byte sizes, all HDF5 headers, and required dataset "
            "shapes; DM coordinate payload is read in full during preparation"
        ),
        "pass": snapshot["pass"] and groupcat["pass"],
        "snapshot_validation": snapshot,
        "groupcat_validation": groupcat,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("recon/hong2021/tng100_download_validation.json"),
    )
    args = parser.parse_args()
    report = build_report(args.output_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
