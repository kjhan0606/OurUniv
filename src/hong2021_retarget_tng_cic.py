#!/usr/bin/env python
"""Replace only a TNG Hong target with the frozen V14 CIC target."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from hong2021_prepare_tng import DENSITY_SCALE, extract_periodic_cube
from hong2021_v14_target import log_cic_target


def retarget(source: Path, grid_path: Path, destination: Path) -> dict[str, Any]:
    metadata_path = grid_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    grid = np.load(grid_path, mmap_mode="r")
    if (
        metadata.get("schema") != "hong2021-periodic-dm-particle-grid-v1"
        or metadata.get("assignment") != "cic"
        or metadata.get("complete") is not True
        or tuple(grid.shape) != (240, 240, 240)
        or not np.isfinite(grid).all()
        or np.any(grid <= 0)
    ):
        raise ValueError("invalid TNG V14 CIC grid cache")
    partial = destination.with_suffix(destination.suffix + ".partial")
    report_path = destination.with_suffix(".json")
    if any(path.exists() for path in (destination, partial, report_path)):
        raise RuntimeError(f"refusing to overwrite output for {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    mean_density = float(grid.mean(dtype=np.float64))
    rows = []
    try:
        with h5py.File(source, "r") as old, h5py.File(partial, "w") as new:
            for name, value in old.attrs.items():
                new.attrs[name] = value
            new.attrs.update(
                {
                    "schema": "hong2021-tng100-raw-cic-input-v14",
                    "target_definition": "log10(raw_particle_cic_cdm/box_mean)/4.5",
                    "target_operator": "periodic cell-centred CIC on the global 240^3 grid",
                    "retargeted_from": str(source.resolve()),
                    "input_and_split_changed": False,
                }
            )
            for name in old:
                if name != "target":
                    old.copy(name, new)
            old_target = old["target"]
            target = new.create_dataset(
                "target",
                shape=old_target.shape,
                dtype="f4",
                chunks=old_target.chunks,
                compression="lzf",
            )
            cube_grid = old_target.shape[-1]
            if old_target.shape[-3:] != (cube_grid,) * 3:
                raise ValueError("source target is not cubic")
            origins = np.asarray(old["cube_origin_cell"], dtype=np.int64)
            for index, origin in enumerate(origins):
                density = extract_periodic_cube(grid, origin, size=cube_grid)
                try:
                    value = log_cic_target(density, mean_density, DENSITY_SCALE)
                except (ValueError, RuntimeError) as error:
                    raise RuntimeError(
                        f"invalid CIC target in TNG cube {index}"
                    ) from error
                target[index, 0] = value
                rows.append(
                    {
                        "index": index,
                        "minimum": float(value.min()),
                        "maximum": float(value.max()),
                        "mean": float(value.mean()),
                        "std": float(value.std()),
                    }
                )
        os.replace(partial, destination)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    report = {
        "schema": "hong2021-tng100-cic-retarget-report-v1",
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "grid": str(grid_path.resolve()),
        "samples": len(rows),
        "non_target_datasets_copied_by_hdf5": True,
        "rows": rows,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    retarget(Path(args.source), Path(args.grid), Path(args.out))


if __name__ == "__main__":
    main()
