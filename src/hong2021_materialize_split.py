#!/usr/bin/env python
"""Materialize one frozen L0-paper split from the validated TNG grids."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hong2021_prepare_tng import (
    CELL_MPC_H,
    FULL_GRID,
    indexed_files,
    load_subhalos,
    periodic_abs_delta,
    write_training_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tng-root", type=Path, required=True)
    parser.add_argument("--dm-counts", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split-index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.split_manifest.read_text())
    try:
        split = manifest["splits"][args.split_index]
    except (KeyError, IndexError) as error:
        raise SystemExit("split index is absent from manifest") from error
    training = np.asarray(split["training_candidate_indices"], dtype=np.int64)
    validation = np.asarray(split["validation_candidate_indices"], dtype=np.int64)
    if len(training) != 432 or len(validation) != 93:
        raise SystemExit("frozen split must contain 432 train and 93 validation cubes")

    group_files = indexed_files(
        args.tng_root / "output/groups_099", "fof_subhalo_tab_099"
    )
    if len(group_files) != 448:
        raise SystemExit("expected 448 TNG100 group catalog chunks")
    subhalos = load_subhalos(group_files)
    center_mask = (
        (subhalos["stellar_mass_paper"] > 4.0e10)
        & (subhalos["stellar_mass_paper"] < 1.0e11)
    )
    target_mask = np.isfinite(subhalos["b_magnitude"]) & (
        subhalos["b_magnitude"] < -15.0
    )
    candidate_ids = subhalos["id"][center_mask]
    candidate_position = subhalos["position"][center_mask]
    candidate_velocity = subhalos["velocity"][center_mask]
    galaxy_position = subhalos["position"][target_mask]
    galaxy_velocity = subhalos["velocity"][target_mask]
    galaxy_cell = np.floor(galaxy_position / CELL_MPC_H).astype(np.int16)
    if len(candidate_ids) != 988 or len(galaxy_position) != 48_296:
        raise RuntimeError("literal Hong L0 catalog counts changed")
    np.testing.assert_array_equal(
        candidate_ids[training], np.asarray(split["training_subhalo_ids"])
    )
    np.testing.assert_array_equal(
        candidate_ids[validation], np.asarray(split["validation_subhalo_ids"])
    )
    cross_delta = periodic_abs_delta(
        candidate_position[training][:, None, :],
        candidate_position[validation][None, :, :],
    )
    if np.any(np.all(cross_delta < 20.0, axis=2)):
        raise RuntimeError("manifest contains cross-split cube overlap")

    dm_counts = np.load(args.dm_counts, mmap_mode="r")
    if dm_counts.shape != (FULL_GRID,) * 3 or dm_counts.dtype != np.uint32:
        raise RuntimeError("invalid validated DM count grid")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_metadata = {
        "algorithm": manifest["schema"],
        "seed": manifest["selection"]["seed"],
        "manifest": str(args.split_manifest),
        "split_index": args.split_index,
    }
    train_report = write_training_file(
        args.out_dir / "tng100_train.h5",
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
        args.out_dir / "tng100_validation.h5",
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
        "schema": "hong2021-materialized-split-v1",
        "selection": "L0-paper: M_B < -15",
        "split_manifest": str(args.split_manifest),
        "split_index": args.split_index,
        "balance": split["balance"],
        "minimum_cross_split_Linf_mpc_h": split[
            "minimum_cross_split_Linf_mpc_h"
        ],
        "train": train_report,
        "validation": validation_report,
    }
    (args.out_dir / "materialization_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
