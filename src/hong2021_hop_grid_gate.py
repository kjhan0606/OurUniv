#!/usr/bin/env python
"""Run the user's HOP build on truth and generated Eulerian density grids.

This is a field-level HOP gate, not an N-body halo catalog.  Every 64^3 voxel
centre is represented by one weighted pseudo-particle and the grid density is
passed to HOP through its documented external-density file.  No Poisson
particles, sub-voxel structure, velocities, or additional random information
are introduced.  Truth, deterministic mean, EDM, and flow use identical
positions and HOP thresholds.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA = "hong2021-grid-hop-gate-v1"
DENSITY_LOG_SCALE = 4.5
RHO_CRIT_M_SUN_H_PER_MPC_H3 = 2.77536627e11


def write_record(handle: Any, value: np.ndarray) -> None:
    data = np.ascontiguousarray(value).tobytes()
    handle.write(struct.pack("=i", len(data)))
    handle.write(data)
    handle.write(struct.pack("=i", len(data)))


def write_simple_lattice(path: Path, grid: int) -> None:
    """Write HOP's documented simple Fortran-unformatted particle input."""
    coordinate = (np.arange(grid, dtype=np.float32) + 0.5) / grid
    x, y, z = np.meshgrid(coordinate, coordinate, coordinate, indexing="ij")
    size = grid**3
    with path.open("wb") as handle:
        write_record(handle, np.asarray([size], dtype=np.int32))
        for value in (x.ravel(), y.ravel(), z.ravel()):
            write_record(handle, value)


def write_density(path: Path, density: np.ndarray) -> None:
    value = np.asarray(density, dtype=np.float32).ravel()
    with path.open("wb") as handle:
        handle.write(np.asarray([len(value)], dtype=np.int32).tobytes())
        handle.write(value.tobytes())


def read_tags(path: Path, expected: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if len(raw) != expected + 2 or int(raw[0]) != expected:
        raise RuntimeError(f"invalid HOP tag file {path}: {raw[:2]} length={len(raw)}")
    return raw[2:]


def load_field(task: dict[str, Any]) -> np.ndarray:
    with h5py.File(task["path"], "r") as handle:
        object_index = int(task["object_index"])
        if task["kind"] == "truth":
            field = handle["truth"][object_index, 0]
        elif task["kind"] == "deterministic":
            field = handle["conditional_mean"][object_index, 0]
        else:
            field = handle["sample"][
                object_index, int(task["member_index"]), 0
            ]
        return np.asarray(field, dtype=np.float32)


def hop_one(task: dict[str, Any]) -> dict[str, Any]:
    field = load_field(task)
    density = np.power(10.0, DENSITY_LOG_SCALE * field, dtype=np.float64)
    if not np.isfinite(density).all():
        raise RuntimeError(f"non-finite density for {task}")
    grid = field.shape[-1]
    work_root = Path(task["work_root"])
    with tempfile.TemporaryDirectory(dir=work_root) as temporary_name:
        temporary = Path(temporary_name)
        density_path = temporary / "field.den"
        write_density(density_path, density)
        root = Path("hop")
        hop_command = [
            task["hop_bin"],
            "-in",
            task["particle_prefix"],
            "-den",
            density_path.name,
            "-nh",
            str(task["n_hop"]),
            "-nm",
            str(task["n_merge"]),
            "-o",
            str(root),
        ]
        hop = subprocess.run(
            hop_command, cwd=temporary, capture_output=True, text=True,
            timeout=task["timeout"]
        )
        if hop.returncode:
            raise RuntimeError(f"HOP failed for {task}:\n{hop.stdout}\n{hop.stderr}")
        group_root = Path("groups")
        regroup_command = [
            task["regroup_bin"],
            "-root",
            str(root),
            "-den",
            density_path.name,
            "-douter",
            str(task["douter"]),
            "-dsaddle",
            str(task["dsaddle"]),
            "-dpeak",
            str(task["dpeak"]),
            "-mingroup",
            str(task["min_group_voxels"]),
            "-o",
            str(group_root),
        ]
        regroup = subprocess.run(
            regroup_command, cwd=temporary, capture_output=True, text=True,
            timeout=task["timeout"]
        )
        if regroup.returncode:
            raise RuntimeError(
                f"regroup failed for {task}:\n{regroup.stdout}\n{regroup.stderr}"
            )
        tags = read_tags(temporary / group_root.with_suffix(".tag"), grid**3)

    valid = tags >= 0
    number = int(tags[valid].max() + 1) if np.any(valid) else 0
    cell_mass = (
        task["omega_m"]
        * RHO_CRIT_M_SUN_H_PER_MPC_H3
        * task["voxel_mpc_h"] ** 3
    )
    masses = np.bincount(
        tags[valid], weights=density.ravel()[valid], minlength=number
    ) * cell_mass
    voxels = np.bincount(tags[valid], minlength=number)
    peaks = np.zeros(number, dtype=np.float64)
    np.maximum.at(peaks, tags[valid], density.ravel()[valid])
    return {
        "kind": task["kind"],
        "method": task.get("method"),
        "object_index": int(task["object_index"]),
        "member_index": task.get("member_index"),
        "n_groups": number,
        "group_mass_m_sun_h": masses.tolist(),
        "group_voxels": voxels.tolist(),
        "group_peak_density": peaks.tolist(),
        "density_mean": float(density.mean()),
        "density_max": float(density.max()),
    }


def summarize(rows: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    categories: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = row["kind"] if row["kind"] != "generated" else row["method"]
        categories.setdefault(label, []).append(row)
    result: dict[str, Any] = {}
    for label, selected in categories.items():
        per_cube = {
            f"ge_{threshold:.0e}": np.asarray(
                [np.count_nonzero(np.asarray(row["group_mass_m_sun_h"]) >= threshold)
                 for row in selected],
                dtype=np.float64,
            )
            for threshold in thresholds
        }
        maxima = np.asarray(
            [max(row["group_mass_m_sun_h"], default=0.0) for row in selected]
        )
        result[label] = {
            "cubes": len(selected),
            "group_count_mean": float(np.mean([row["n_groups"] for row in selected])),
            "group_count_std": float(np.std([row["n_groups"] for row in selected])),
            "count_above_mass": {
                key: {"mean": float(value.mean()), "std": float(value.std())}
                for key, value in per_cube.items()
            },
            "maximum_group_mass_m_sun_h": {
                "median": float(np.median(maxima)),
                "mean": float(np.mean(maxima)),
            },
        }
    truth = result["truth"]
    for method in ("edm", "flow"):
        candidate = result.get(method)
        if candidate is None:
            continue
        candidate["ratios_to_truth"] = {
            "group_count_mean": candidate["group_count_mean"]
            / truth["group_count_mean"],
            "count_above_mass": {
                key: candidate["count_above_mass"][key]["mean"] / value["mean"]
                if value["mean"] > 0 else None
                for key, value in truth["count_above_mass"].items()
            },
            "maximum_mass_median": candidate["maximum_group_mass_m_sun_h"]["median"]
            / truth["maximum_group_mass_m_sun_h"]["median"],
        }
    return result


def build_tasks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    base = {
        "hop_bin": str(args.hop_dir / "hop"),
        "regroup_bin": str(args.hop_dir / "regroup"),
        "particle_prefix": str(args.work / "lattice_particles.bin"),
        "work_root": str(args.work),
        "n_hop": args.n_hop,
        "n_merge": args.n_merge,
        "douter": args.douter,
        "dsaddle": args.dsaddle,
        "dpeak": args.dpeak,
        "min_group_voxels": args.min_group_voxels,
        "omega_m": args.omega_m,
        "voxel_mpc_h": args.voxel_mpc_h,
        "timeout": args.timeout,
    }
    with h5py.File(args.edm, "r") as handle:
        objects, ensemble = handle["sample"].shape[:2]
        grid = handle["sample"].shape[-1]
    members = list(range(ensemble))
    if args.members is not None:
        members = members[: args.members]
    tasks: list[dict[str, Any]] = []
    if args.objects is not None:
        objects = min(objects, args.objects)
    for object_index in range(objects):
        tasks.append({**base, "path": str(args.edm), "kind": "truth",
                      "object_index": object_index})
        tasks.append({**base, "path": str(args.edm), "kind": "deterministic",
                      "object_index": object_index})
        candidates = [("edm", args.edm)]
        if args.flow is not None:
            candidates.append(("flow", args.flow))
        for method, path in candidates:
            for member in members:
                tasks.append({**base, "path": str(path), "kind": "generated",
                              "method": method, "object_index": object_index,
                              "member_index": member})
    return tasks, grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edm", type=Path, required=True)
    parser.add_argument(
        "--flow", type=Path, default=None,
        help="optional flow candidate; omit for a single-EDM downstream gate",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--hop-dir", type=Path,
        default=Path("/home/kjhan/BACKUP/lagRamses-de-nonstd/utils/f90/hop_ramses"),
    )
    parser.add_argument("--members", type=int, default=4)
    parser.add_argument("--objects", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--voxel-mpc-h", type=float, default=0.3125)
    parser.add_argument("--omega-m", type=float, default=0.3089)
    parser.add_argument("--n-hop", type=int, default=16)
    parser.add_argument("--n-merge", type=int, default=4)
    parser.add_argument("--douter", type=float, default=80.0)
    parser.add_argument("--dsaddle", type=float, default=200.0)
    parser.add_argument("--dpeak", type=float, default=240.0)
    parser.add_argument("--min-group-voxels", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--mass-thresholds", default="1e12,3e12,1e13,3e13"
    )
    args = parser.parse_args()
    for executable in (args.hop_dir / "hop", args.hop_dir / "regroup"):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise FileNotFoundError(executable)
    args.work.mkdir(parents=True, exist_ok=True)
    tasks, grid = build_tasks(args)
    particle_path = args.work / "lattice_particles.bin"
    if not particle_path.exists():
        write_simple_lattice(particle_path, grid)
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(hop_one, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(
                f"[hop] {index}/{len(tasks)} {row['kind']} "
                f"{row.get('method') or ''} object={row['object_index']} "
                f"member={row.get('member_index')} groups={row['n_groups']}",
                flush=True,
            )
    rows.sort(
        key=lambda row: (
            row["kind"], row.get("method") or "", row["object_index"],
            -1 if row.get("member_index") is None else row["member_index"],
        )
    )
    thresholds = [float(value) for value in args.mass_thresholds.split(",")]
    report = {
        "schema": SCHEMA,
        "interpretation": (
            "Eulerian grid-HOP field gate; not an N-body bound-halo catalog and "
            "not a substitute for HOP on a forward simulation"
        ),
        "density_mapping": "rho/rho_mean = 10**(4.5 * normalized_log_density)",
        "pseudo_particles": (
            "one weighted particle at every voxel centre; external HOP density; "
            "no Poisson or sub-voxel information"
        ),
        "grid": grid,
        "voxel_mpc_h": args.voxel_mpc_h,
        "hop": {
            "binary": str((args.hop_dir / "hop").resolve()),
            "regroup": str((args.hop_dir / "regroup").resolve()),
            "n_hop": args.n_hop,
            "n_merge": args.n_merge,
            "douter": args.douter,
            "dsaddle": args.dsaddle,
            "dpeak": args.dpeak,
            "min_group_voxels": args.min_group_voxels,
            "periodic": False,
        },
        "members_per_environment": args.members,
        "mass_thresholds_m_sun_h": thresholds,
        "summary": summarize(rows, thresholds),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
