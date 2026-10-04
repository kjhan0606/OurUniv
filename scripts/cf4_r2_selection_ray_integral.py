"""Map-aware, cell-integrated 2M++ exposure for the N128 R2 count grid.

This is a numerical observation-operator artifact only. It reads pinned
selection maps and a previous exposure file for comparison, but no catalogue,
count/key, train/holdout, field, or native-truth arrays.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from itertools import product

import healpy as hp
import h5py
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from cf4_actual_selection import base  # noqa: E402
from cf4_r2_ray_cost_profile import (  # noqa: E402
    REFERENCE_KEYS,
    candidate_cap,
    ray_box_intervals,
    ray_geometry,
)
from cf4_twompp_joint_information_budget_pilot_v1 import (  # noqa: E402
    _cosmology_distance_table,
    schechter_fraction,
)

N = 128
BOX = 384.0
DX = BOX / N
HALF = BOX / 2.0
R_INNER = 5.0
R_OUTER = 180.0
RADIAL_EDGES = np.array([5., 30., 60., 90., 120., 150., 180.], dtype=np.float64)
NSIDES = (1024, 2048)
CHUNK_SIZES = (65_536, 250_000)
OLD_CANDIDATE_LIMIT = 1_500_000
RADIUS_SUMMARY_EDGES = np.array([0., 15., 30., 60., 90., 120., 150., 180., 186.])
DATA = Path("/gpfs/kjhan/CF4/z0_density/r2_common_catalogue_128_v1")
OLD_SELECTION = Path("/gpfs/kjhan/CF4/z0_density/r2_common_selection_128_v1/selection_3.h5")
TECHNICAL_INPUT = Path(
    "/gpfs/kjhan/CF4/kf_design/twompp_disjoint_tracer_v4/pilot/result.json")
PROGRAM = ROOT / "config/cf4_twompp_disjoint_tracer_pilot_program_v1.json"
COSMOLOGY = ROOT / "config/cf4_r2_common_cosmology_v1.json"
OUT_DEFAULT = Path(
    "/gpfs/kjhan/CF4/z0_density/r2_ray_selection_n128_nside1024_2048_20261004_v1")

_WORKER_INPUTS: dict | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def observer_rotation() -> np.ndarray:
    return SkyCoord(CartesianRepresentation(np.eye(3)),
                    frame="supergalactic").icrs.cartesian.xyz.value


def load_physics_inputs():
    tracer = json.loads(PROGRAM.read_text())
    cosmology_contract = json.loads(COSMOLOGY.read_text())
    common = cosmology_contract["common_cosmology"]
    cosmology = dict(
        h=common["h"], H0_km_s_Mpc=common["H0_km_s_Mpc"],
        Omega_m=common["Omega_m"], Omega_b=common["Omega_b"],
        Tcmb_K=common["Tcmb_K"])
    if N != cosmology_contract["N"] or BOX != cosmology_contract["box_cMpc_h"]:
        raise ValueError("frozen N128/384 exposure grid differs from cosmology contract")

    map_paths = [Path(tracer["inputs"][name]["path"]) for name in
                 ("completeness_11_5", "completeness_12_5")]
    technical = json.loads(TECHNICAL_INPUT.read_text())
    if technical.get("status") != "PASS_TECHNICAL_INPUT_GATE_NO_FIELD_INFERENCE":
        raise ValueError("pinned map technical-input gate is not passing")
    expected_hashes = [technical["bound_inputs"][name]["sha256"] for name in
                       ("completeness_11_5", "completeness_12_5")]
    actual_hashes = [sha256_file(path) for path in map_paths]
    if actual_hashes != expected_hashes:
        raise ValueError("angular map digest differs from pinned technical-input result")
    maps = [np.asarray(base.load_completeness_map(path, 512), dtype=np.float64)
            for path in map_paths]
    if any(value.shape != (hp.nside2npix(512),) for value in maps):
        raise ValueError("pinned completeness map is not NSIDE=512 RING")
    if any(not np.isfinite(value).all() or np.any((value < 0.) | (value > 1.))
           for value in maps):
        raise ValueError("completeness map contains non-finite or out-of-range values")

    edges = np.asarray(tracer["tracer_design"]["absolute_K_edges"], dtype=np.float64)
    if len(edges) != 4:
        raise ValueError("expected exactly three absolute-magnitude bins")
    rtab = np.linspace(R_INNER, R_OUTER, 200_001, dtype=np.float64)
    luminosity_distance = _cosmology_distance_table(rtab, cosmology) * cosmology["h"]
    radial_fraction = np.asarray([
        schechter_fraction(
            luminosity_distance,
            None if p // 3 == 0 else 11.5,
            11.5 if p // 3 == 0 else 12.5,
            edges[p % 3], edges[p % 3 + 1], -23.28, -0.94)
        for p in range(6)
    ], dtype=np.float64)
    cumulative = np.concatenate((
        np.zeros((6, 1), dtype=np.float64),
        np.cumsum(
            .5 * (rtab[1:] ** 2 * radial_fraction[:, 1:]
                  + rtab[:-1] ** 2 * radial_fraction[:, :-1])
            * np.diff(rtab)[None, :], axis=1)), axis=1)
    cumulative_edges = np.asarray([
        np.interp(RADIAL_EDGES, rtab, cumulative[p]) for p in range(6)
    ], dtype=np.float64)
    rotation = observer_rotation()
    for array in [*maps, cumulative, cumulative_edges, rtab, rotation]:
        array.setflags(write=False)
    return {
        "maps": maps,
        "map_hashes": actual_hashes,
        "cosmology": cosmology,
        "radial_magnitude_edges": edges.tolist(),
        "rtab": rtab,
        "cumulative": cumulative,
        "cumulative_edges": cumulative_edges,
        "rotation": rotation,
    }


def exact_geometry_census(nside_for_cap: int = 2048):
    """Enumerate active support and save the cap/boundary cross-tab exactly."""
    index = np.arange(N, dtype=np.float64)
    lows = index * DX - HALF
    highs = lows + DX
    centers_1d = .5 * (lows + highs)
    near_1d = np.where((lows <= 0.) & (highs >= 0.), 0.,
                       np.minimum(np.abs(lows), np.abs(highs)))
    far_1d = np.maximum(np.abs(lows), np.abs(highs))
    cy, cz = np.meshgrid(centers_1d, centers_1d, indexing="ij")
    ny, nz = np.meshgrid(near_1d, near_1d, indexing="ij")
    fy, fz = np.meshgrid(far_1d, far_1d, indexing="ij")
    cy, cz, ny, nz, fy, fz = [a.ravel() for a in (cy, cz, ny, nz, fy, fz)]
    corner_offsets = np.asarray(list(product((-DX / 2., DX / 2.), repeat=3)))
    npix = hp.nside2npix(nside_for_cap)

    active_by_x = []
    flat_by_x = []
    radii_by_x = []
    counts = {
        "active_shell_intersection": 0,
        "fully_inside_shell": 0,
        "inner_boundary_partial": 0,
        "outer_boundary_partial": 0,
        "both_boundaries_partial": 0,
        "active_cells_over_cap_estimate_limit": 0,
        "inner_partial_and_cap_over_limit": 0,
        "inner_partial_and_cap_not_over_limit": 0,
        "not_inner_partial_and_cap_over_limit": 0,
        "not_inner_partial_and_cap_not_over_limit": 0,
        "cap_estimate_pixel_sum": 0.0,
    }
    joint_status_crosstab = [
        {"inner_boundary_partial": inner,
         "outer_boundary_partial": outer,
         "cap_estimate_over_limit": over_cap,
         "active_cells": 0}
        for inner, outer, over_cap in product((False, True), repeat=3)
    ]
    max_cap = -1.0
    max_cells = []
    for ix in range(N):
        xlo, xhi = lows[ix], highs[ix]
        xc = centers_1d[ix]
        nx, fx = near_1d[ix], far_1d[ix]
        rmin = np.sqrt(nx * nx + ny * ny + nz * nz)
        rmax = np.sqrt(fx * fx + fy * fy + fz * fz)
        active = (rmax > R_INNER) & (rmin < R_OUTER)
        active_indices = np.flatnonzero(active)
        if not len(active_indices):
            active_by_x.append(np.empty((0, 2), dtype=np.int16))
            flat_by_x.append(np.empty(0, dtype=np.int64))
            radii_by_x.append(np.empty(0, dtype=np.float64))
            continue

        y = cy[active]
        z = cz[active]
        centers = np.column_stack((np.full(len(y), xc), y, z))
        radii = np.linalg.norm(centers, axis=1)
        min_cos = np.full(len(y), np.inf)
        for offset in corner_offsets:
            vertex = centers + offset[None, :]
            norms = np.linalg.norm(vertex, axis=1)
            nonzero = norms > 0.
            cosine = np.full(len(y), np.inf)
            cosine[nonzero] = (np.sum(centers[nonzero] * vertex[nonzero], axis=1)
                               / (radii[nonzero] * norms[nonzero]))
            min_cos = np.minimum(min_cos, cosine)
        caps = np.minimum(np.pi, np.arccos(np.clip(min_cos, -1., 1.)) + 1e-6)
        candidate_estimate = npix * (1. - np.cos(caps)) / 2.
        inner = (rmin[active] < R_INNER) & (rmax[active] > R_INNER)
        outer = (rmin[active] < R_OUTER) & (rmax[active] > R_OUTER)
        over_cap = candidate_estimate > OLD_CANDIDATE_LIMIT

        counts["active_shell_intersection"] += int(len(y))
        counts["fully_inside_shell"] += int(np.count_nonzero(
            (rmin[active] >= R_INNER) & (rmax[active] <= R_OUTER)))
        counts["inner_boundary_partial"] += int(np.count_nonzero(inner))
        counts["outer_boundary_partial"] += int(np.count_nonzero(outer))
        counts["both_boundaries_partial"] += int(np.count_nonzero(inner & outer))
        counts["active_cells_over_cap_estimate_limit"] += int(np.count_nonzero(over_cap))
        counts["inner_partial_and_cap_over_limit"] += int(np.count_nonzero(inner & over_cap))
        counts["inner_partial_and_cap_not_over_limit"] += int(np.count_nonzero(inner & ~over_cap))
        counts["not_inner_partial_and_cap_over_limit"] += int(np.count_nonzero(~inner & over_cap))
        counts["not_inner_partial_and_cap_not_over_limit"] += int(np.count_nonzero(~inner & ~over_cap))
        counts["cap_estimate_pixel_sum"] += float(np.sum(candidate_estimate, dtype=np.float64))
        for status in joint_status_crosstab:
            selected = ((inner == status["inner_boundary_partial"])
                        & (outer == status["outer_boundary_partial"])
                        & (over_cap == status["cap_estimate_over_limit"]))
            status["active_cells"] += int(np.count_nonzero(selected))

        iy = active_indices // N
        iz = active_indices % N
        active_by_x.append(np.column_stack((iy, iz)).astype(np.int16))
        flat_by_x.append(((ix * N + iy) * N + iz).astype(np.int64))
        radii_by_x.append(radii.astype(np.float64, copy=True))

        local_max = float(np.max(caps))
        if local_max > max_cap + 1e-12:
            max_cap = local_max
            max_cells = []
        if abs(local_max - max_cap) <= 1e-12:
            for j in np.flatnonzero(np.abs(caps - max_cap) <= 1e-12):
                k = int(active_indices[j])
                iy0, iz0 = divmod(k, N)
                max_cells.append({
                    "ijk": [int(ix), int(iy0), int(iz0)],
                    "flat_index": int((ix * N + iy0) * N + iz0),
                    "low": [float(xlo), float(lows[iy0]), float(lows[iz0])],
                    "high": [float(xhi), float(highs[iy0]), float(highs[iz0])],
                    "center": centers[j].tolist(),
                    "cap_rad": float(caps[j]),
                    "cap_estimate_pixels": float(candidate_estimate[j]),
                })

    if counts["active_shell_intersection"] != (
            counts["fully_inside_shell"] + counts["inner_boundary_partial"]
            + counts["outer_boundary_partial"] - counts["both_boundaries_partial"]):
        raise AssertionError("shell-cell partition does not close")
    if sum(row["active_cells"] for row in joint_status_crosstab) != counts[
            "active_shell_intersection"]:
        raise AssertionError("inner/outer/cap status cross-tab does not close")
    max_cells.sort(key=lambda row: row["ijk"])
    flat_indices = np.concatenate(flat_by_x)
    if not np.all(flat_indices[1:] > flat_indices[:-1]):
        raise AssertionError("active flat-cell indices must be strictly increasing")
    return {
        "counts": counts,
        "joint_status_crosstab": joint_status_crosstab,
        "active_by_x": active_by_x,
        "flat_indices": flat_indices,
        "active_radii": np.concatenate(radii_by_x),
        "max_cap_rad": max_cap,
        "maximum_cap_cells": max_cells,
    }


def _shell_indices(rlo: np.ndarray, rhi: np.ndarray):
    lo_shell = np.searchsorted(RADIAL_EDGES, rlo, side="right") - 1
    hi_shell = np.searchsorted(RADIAL_EDGES, rhi, side="left") - 1
    lo_shell = np.clip(lo_shell, 0, len(RADIAL_EDGES) - 2).astype(np.int16)
    hi_shell = np.clip(hi_shell, 0, len(RADIAL_EDGES) - 2).astype(np.int16)
    if np.any(hi_shell < lo_shell) or np.any(hi_shell - lo_shell > 1):
        raise ValueError("one voxel ray interval crosses multiple radial-shell edges")
    return lo_shell, hi_shell


def integrate_candidate_chunks(candidates: np.ndarray, nside: int,
                               low: np.ndarray, high: np.ndarray,
                               physics: dict, chunk_size: int):
    """Accumulate six populations by six radial shells without dropping rays."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    exposure_mass = np.zeros((6, 6), dtype=np.float64)
    geometry_mass = np.zeros(6, dtype=np.float64)
    box_hits = 0
    shell_hits = 0
    rotation = physics["rotation"]
    maps = physics["maps"]
    cumulative = physics["cumulative"]
    cumulative_edges = physics["cumulative_edges"]
    rtab = physics["rtab"]
    for start in range(0, len(candidates), chunk_size):
        pixel_ids = candidates[start:start + chunk_size]
        sky_dirs = np.asarray(hp.pix2vec(nside, pixel_ids, nest=False), dtype=np.float64)
        grid_dirs = rotation.T @ sky_dirs
        enter, leave, intersects = ray_box_intervals(grid_dirs, low, high)
        box_hits += int(np.count_nonzero(intersects))
        rlo = np.maximum(enter, R_INNER)
        rhi = np.minimum(leave, R_OUTER)
        valid = intersects & (rhi > rlo)
        if not np.any(valid):
            continue
        shell_hits += int(np.count_nonzero(valid))
        lo = rlo[valid]
        hi = rhi[valid]
        lo_shell, hi_shell = _shell_indices(lo, hi)
        same_shell = lo_shell == hi_shell
        split_shell = ~same_shell
        native_pixels = hp.vec2pix(512, *sky_dirs[:, valid], nest=False)

        for p in range(6):
            cumulative_lo = np.interp(lo, rtab, cumulative[p])
            cumulative_hi = np.interp(hi, rtab, cumulative[p])
            mass = cumulative_hi - cumulative_lo
            completeness = maps[p // 3][native_pixels]
            weights = completeness * mass
            exposure_mass[p] += np.bincount(
                lo_shell[same_shell], weights=weights[same_shell], minlength=6)
            if np.any(split_shell):
                left_shell = lo_shell[split_shell]
                right_shell = hi_shell[split_shell]
                edge_value = cumulative_edges[p, left_shell + 1]
                left_mass = edge_value - cumulative_lo[split_shell]
                right_mass = cumulative_hi[split_shell] - edge_value
                exposure_mass[p] += np.bincount(
                    left_shell, weights=completeness[split_shell] * left_mass,
                    minlength=6)
                exposure_mass[p] += np.bincount(
                    right_shell, weights=completeness[split_shell] * right_mass,
                    minlength=6)
        geometry_total = (hi ** 3 - lo ** 3) / 3.0
        geometry_mass += np.bincount(
            lo_shell[same_shell], weights=geometry_total[same_shell], minlength=6)
        if np.any(split_shell):
            left_shell = lo_shell[split_shell]
            right_shell = hi_shell[split_shell]
            edge = RADIAL_EDGES[left_shell + 1]
            left_geometry = (edge ** 3 - lo[split_shell] ** 3) / 3.0
            right_geometry = (hi[split_shell] ** 3 - edge ** 3) / 3.0
            geometry_mass += np.bincount(
                left_shell, weights=left_geometry, minlength=6)
            geometry_mass += np.bincount(
                right_shell, weights=right_geometry, minlength=6)

    factor = hp.nside2pixarea(nside) / (DX ** 3)
    return {
        "exposure": exposure_mass * factor,
        "geometry": geometry_mass * factor,
        "box_hit_rays": box_hits,
        "shell_hit_rays": shell_hits,
    }


def integrate_cell(ijk, nside: int, physics: dict, chunk_size: int):
    ijk = np.asarray(ijk, dtype=np.int64)
    low = ijk.astype(np.float64) * DX - HALF
    high = low + DX
    center_sky = physics["rotation"] @ (low + high) / 2.0
    axis, cap = candidate_cap(center_sky, low, high, physics["rotation"])
    candidates = hp.query_disc(nside, axis, cap, inclusive=True, nest=False)
    result = integrate_candidate_chunks(
        candidates, nside, low, high, physics, chunk_size)
    result.update({
        "candidate_pixels": int(len(candidates)),
        "cap_rad": float(cap),
        "estimated_cap_pixels": float(
            hp.nside2npix(nside) * (1. - np.cos(cap)) / 2.),
    })
    return result


def legacy_exposure_from_ray_geometry(geometry: dict, nside: int, physics: dict):
    """Reconstruct the old unsplit six-channel ray integral from saved rays."""
    if geometry.get("skipped"):
        raise ValueError("cannot reconstruct exposure from a skipped geometry")
    native = np.asarray(geometry["native_pixels"], dtype=np.int64)
    rlo = np.asarray(geometry["rlo"], dtype=np.float64)
    rhi = np.asarray(geometry["rhi"], dtype=np.float64)
    if native.shape != rlo.shape or native.shape != rhi.shape:
        raise ValueError("legacy ray geometry arrays have inconsistent lengths")
    result = np.zeros(6, dtype=np.float64)
    for population in range(6):
        radial_mass = (np.interp(rhi, physics["rtab"], physics["cumulative"][population])
                       - np.interp(rlo, physics["rtab"], physics["cumulative"][population]))
        completeness = physics["maps"][population // 3][native]
        result[population] = (hp.nside2pixarea(nside)
                              * np.dot(completeness, radial_mass) / DX ** 3)
    return result


def _relative_difference(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = np.maximum(np.abs(a), np.abs(b))
    result = np.full(np.broadcast_shapes(a.shape, b.shape), np.nan, dtype=np.float64)
    np.divide(np.abs(a - b), denominator, out=result, where=denominator > 0.)
    return result


def _max_relative_difference(a: np.ndarray, b: np.ndarray) -> float:
    delta = _relative_difference(a, b)
    finite = delta[np.isfinite(delta)]
    return float(np.max(finite)) if finite.size else 0.0


def _assert_same(a, b, label: str, rtol: float = 1e-12, atol: float = 1e-14):
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{label} differs: max_relative={_max_relative_difference(a, b):.6g}")


def run_preflight(geometry: dict, physics: dict):
    checks = {"focused_slurm_pytest": "passed before script start",
              "max_cap_chunk_comparisons": [],
              "fixed_reference_comparisons": []}
    if len(geometry["maximum_cap_cells"]) != 8:
        raise AssertionError("expected eight cells attaining the measured maximum cap")

    # Stress both chunk sizes against one-piece accumulation on the hardest
    # NSIDE=2048 cells. One-piece references are sequential, never concurrent.
    for cell in geometry["maximum_cap_cells"]:
        low = np.asarray(cell["low"], dtype=np.float64)
        high = np.asarray(cell["high"], dtype=np.float64)
        center_sky = physics["rotation"] @ np.asarray(cell["center"], dtype=np.float64)
        axis, cap = candidate_cap(center_sky, low, high, physics["rotation"])
        candidates = hp.query_disc(2048, axis, cap, inclusive=True, nest=False)
        chunk_sizes = (*CHUNK_SIZES, max(1, len(candidates)))
        values = [integrate_candidate_chunks(
            candidates, 2048, low, high, physics, chunk_size)
            for chunk_size in chunk_sizes]
        for chunk_size, value in zip(chunk_sizes, values):
            _assert_same(values[0]["exposure"], value["exposure"],
                         f"max-cap {cell['ijk']} chunk={chunk_size}")
            _assert_same(values[0]["geometry"], value["geometry"],
                         f"max-cap geometry {cell['ijk']} chunk={chunk_size}")
        checks["max_cap_chunk_comparisons"].append({
            "ijk": cell["ijk"], "candidate_pixels": int(len(candidates)),
            "chunk_sizes": list(chunk_sizes),
            "maximum_relative_difference": max(
                _max_relative_difference(values[0]["exposure"], value["exposure"])
                for value in values[1:]),
        })

    # Keep the two established references as an independent unchunked control.
    for key, expected_by_nside in REFERENCE_KEYS.items():
        flat = int(key % N ** 3)
        population = int(key // N ** 3)
        ijk = np.unravel_index(flat, (N,) * 3)
        for nside in (512, 1024, 2048):
            old = ray_geometry(flat, nside, physics["rotation"], OLD_CANDIDATE_LIMIT)
            if old.get("skipped"):
                raise AssertionError(
                    f"fixed reference key={key} exceeds old unchunked control cap at {nside}")
            candidate_count = int(old["candidate_pixels"])
            old_exposure = legacy_exposure_from_ray_geometry(old, nside, physics)
            current_values = [integrate_cell(ijk, nside, physics, chunk_size)
                              for chunk_size in CHUNK_SIZES]
            for chunk_size, current in zip(CHUNK_SIZES, current_values):
                _assert_same(current["exposure"].sum(axis=1),
                             old_exposure,
                             f"fixed reference key={key} nside={nside} chunk={chunk_size}")
            for chunk_size, current in zip(CHUNK_SIZES, current_values):
                if abs(current["exposure"][population].sum()
                       - expected_by_nside[nside]) > 1e-7:
                    raise AssertionError(
                        f"fixed NSIDE={nside} reference changed for key={key}")
            checks["fixed_reference_comparisons"].append({
                "key": int(key), "flat_index": flat, "population": population,
                "nside": int(nside), "candidate_pixels": candidate_count,
                "reference_exposure": float(expected_by_nside[nside]),
            })
    return checks


def _worker_init(physics: dict, active_by_x: list[np.ndarray]):
    global _WORKER_INPUTS
    _WORKER_INPUTS = {"physics": physics, "active_by_x": active_by_x}


def _integrate_x_slab(ix: int):
    if _WORKER_INPUTS is None:
        raise RuntimeError("worker input was not initialized before spawn")
    physics = _WORKER_INPUTS["physics"]
    active_ij = _WORKER_INPUTS["active_by_x"][ix]
    e1024 = np.zeros((6, 6, N, N), dtype=np.float64)
    e2048 = np.zeros((6, 6, N, N), dtype=np.float64)
    g1024 = np.zeros((6, N, N), dtype=np.float64)
    g2048 = np.zeros((6, N, N), dtype=np.float64)
    hits2048 = np.zeros((N, N), dtype=np.uint32)
    stats = {str(nside): {"candidate_pixels": 0, "box_hit_rays": 0,
                          "shell_hit_rays": 0, "zero_hit_cells": []}
             for nside in NSIDES}
    for iy0, iz0 in active_ij:
        ijk = (ix, int(iy0), int(iz0))
        for nside, exposure_slab in ((1024, e1024), (2048, e2048)):
            value = integrate_cell(ijk, nside, physics, CHUNK_SIZES[-1])
            exposure_slab[:, :, iy0, iz0] = value["exposure"]
            if nside == 1024:
                g1024[:, iy0, iz0] = value["geometry"]
            stat = stats[str(nside)]
            stat["candidate_pixels"] += value["candidate_pixels"]
            stat["box_hit_rays"] += value["box_hit_rays"]
            stat["shell_hit_rays"] += value["shell_hit_rays"]
            if value["shell_hit_rays"] == 0:
                stat["zero_hit_cells"].append(
                    int((ix * N + int(iy0)) * N + int(iz0)))
            if nside == 2048:
                g2048[:, iy0, iz0] = value["geometry"]
                hits2048[iy0, iz0] = value["shell_hit_rays"]
    return ix, e1024, e2048, g1024, g2048, hits2048, stats


def fullsky_map_sums(nside: int, maps: list[np.ndarray], chunk_size: int = 250_000):
    npix = hp.nside2npix(nside)
    sums = np.zeros(6, dtype=np.float64)
    for start in range(0, npix, chunk_size):
        ids = np.arange(start, min(start + chunk_size, npix), dtype=np.int64)
        sky_dirs = np.asarray(hp.pix2vec(nside, ids, nest=False), dtype=np.float64)
        parent = hp.vec2pix(512, *sky_dirs, nest=False)
        for p in range(6):
            sums[p] += np.sum(maps[p // 3][parent], dtype=np.float64)
    return sums


def _channel_radius_quantiles(values: np.ndarray, active_radii: np.ndarray):
    rows = []
    radius_bin = np.clip(
        np.searchsorted(RADIUS_SUMMARY_EDGES, active_radii, side="right") - 1,
        0, len(RADIUS_SUMMARY_EDGES) - 2)
    for p in range(values.shape[0]):
        for shell in range(values.shape[1]):
            for radial_bin in range(len(RADIUS_SUMMARY_EDGES) - 1):
                selected = radius_bin == radial_bin
                current = np.asarray(values[p, shell, selected], dtype=np.float64)
                current = current[np.isfinite(current)]
                quantiles = (np.quantile(current, [0.5, 0.95, 0.99, 1.0]).tolist()
                             if current.size else [None] * 4)
                rows.append({"population": p, "radial_shell": shell,
                             "center_radius_bin_cMpc_h": [
                                 float(RADIUS_SUMMARY_EDGES[radial_bin]),
                                 float(RADIUS_SUMMARY_EDGES[radial_bin + 1])],
                             "finite_cells": int(current.size),
                             "p50_p95_p99_max": quantiles})
    return rows


def old_new_summary(relative: np.ndarray, active_radii: np.ndarray,
                    support_counts: dict):
    result = _channel_radius_quantiles(relative, active_radii)
    for row in result:
        p, shell = row["population"], row["radial_shell"]
        bounds = row["center_radius_bin_cMpc_h"]
        b = int(np.searchsorted(RADIUS_SUMMARY_EDGES, bounds[0], side="right") - 1)
        row.update({name: int(value[p, shell, b]) for name, value in support_counts.items()})
    return result


def _validate_previous_exposure():
    with h5py.File(OLD_SELECTION, "r") as handle:
        status = handle.attrs.get("status", "")
        if isinstance(status, bytes):
            status = status.decode("utf-8")
        if status != "INTEGRATION_COMPLETE_NOT_CALIBRATED":
            raise ValueError(f"old exposure status is not an approved comparison: {status}")
        dataset = handle.get("selection_shells")
        if dataset is None or dataset.shape != (6, 6, N, N, N):
            raise ValueError("old selection exposure shape differs from N128 contract")
        return {"status": status, "shape": list(dataset.shape),
                "dtype": str(dataset.dtype)}


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("full map-aware exposure integration must run through Slurm")
    expected_commit = os.environ.get("CF4_EXPECTED_COMMIT")
    out_value = os.environ.get("CF4_R2_OUT_DIR")
    if not expected_commit or not out_value:
        raise RuntimeError("CF4_EXPECTED_COMMIT and CF4_R2_OUT_DIR are required")
    actual_commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"source commit mismatch: expected {expected_commit}, found {actual_commit}")
    out = Path(out_value)
    if out.exists():
        raise FileExistsError(out)

    started = time.monotonic()
    physics = load_physics_inputs()
    previous_contract = _validate_previous_exposure()
    geometry = exact_geometry_census(2048)
    if geometry["counts"]["active_shell_intersection"] != 938_128:
        raise AssertionError("exact active shell-cell count changed from frozen census")
    preflight = run_preflight(geometry, physics)

    worker_count = int(os.environ.get("CF4_R2_WORKERS", "15"))
    if worker_count < 1 or worker_count > int(os.environ.get("SLURM_CPUS_PER_TASK", "1")) - 1:
        raise ValueError("worker count must leave one allocated CPU for the parent")
    active_flat = geometry["flat_indices"]
    row_offsets = np.zeros(N + 1, dtype=np.int64)
    for ix in range(N):
        row_offsets[ix + 1] = row_offsets[ix] + len(geometry["active_by_x"][ix])
    if row_offsets[-1] != len(active_flat):
        raise AssertionError("per-slab active cell order differs from flat-cell index")

    out.mkdir(parents=True)
    relative_resolution = np.full((6, 6, len(active_flat)), np.nan, dtype=np.float32)
    relative_old_new = np.full((6, 6, len(active_flat)), np.nan, dtype=np.float32)
    support_counts = {
        "old_positive_cells": np.zeros((6, 6, len(RADIUS_SUMMARY_EDGES) - 1), dtype=np.int64),
        "new_positive_cells": np.zeros((6, 6, len(RADIUS_SUMMARY_EDGES) - 1), dtype=np.int64),
        "old_positive_new_zero_cells": np.zeros((6, 6, len(RADIUS_SUMMARY_EDGES) - 1), dtype=np.int64),
        "old_zero_new_positive_cells": np.zeros((6, 6, len(RADIUS_SUMMARY_EDGES) - 1), dtype=np.int64),
    }
    totals = {str(nside): {"candidate_pixels": 0, "box_hit_rays": 0,
                            "shell_hit_rays": 0, "zero_hit_cells": []}
              for nside in NSIDES}
    observed = {str(nside): np.zeros((6, 6), dtype=np.float64) for nside in NSIDES}
    geometry_observed = {str(nside): np.zeros(6, dtype=np.float64) for nside in NSIDES}

    h5_path = out / "selection_ray.h5"
    result_path = out / "result.json"
    with h5py.File(OLD_SELECTION, "r") as old_handle, h5py.File(h5_path, "x") as output:
        old_dataset = old_handle["selection_shells"]
        exposure_dataset = output.create_dataset(
            "selection_shells", shape=(6, 6, N, N, N), dtype="f8",
            chunks=(1, 1, 1, N, N), compression="gzip", compression_opts=1)
        geometry_dataset = output.create_dataset(
            "geometry_shells", shape=(6, N, N, N), dtype="f8",
            chunks=(1, 1, N, N), compression="gzip", compression_opts=1)
        hit_dataset = output.create_dataset(
            "ray_hit_count_nside2048", shape=(N, N, N), dtype="u4",
            chunks=(1, N, N), compression="gzip", compression_opts=1)
        output.create_dataset("active_flat_cell_index", data=active_flat, dtype="i8")
        output.attrs.update(
            status="INCOMPLETE", source_commit=expected_commit,
            grid_N=N, box_cMpc_h=BOX, dx_cMpc_h=DX,
            map_nside=512, map_ordering="RING",
            output_nside=2048, diagnostic_nside=1024,
            radial_edges_cMpc_h=RADIAL_EDGES,
            cosmology_json=json.dumps(physics["cosmology"], sort_keys=True),
            map_hashes_json=json.dumps(physics["map_hashes"]),
            selection_status="NUMERICAL_SELECTION_ARTIFACT_NOT_LIKELIHOOD_READY")

        ctx = mp.get_context("spawn")
        completed_slabs = 0
        with ctx.Pool(processes=worker_count, initializer=_worker_init,
                      initargs=(physics, geometry["active_by_x"])) as pool:
            for ix, e1024, e2048, g1024, g2048, hit2048, stats in pool.imap_unordered(
                    _integrate_x_slab, range(N), chunksize=1):
                if (not np.isfinite(e1024).all() or not np.isfinite(e2048).all()
                        or np.any(e1024 < 0.) or np.any(e2048 < 0.)
                        or not np.isfinite(g1024).all() or np.any(g1024 < 0.)
                        or not np.isfinite(g2048).all() or np.any(g2048 < 0.)):
                    raise ValueError(f"non-finite or negative exposure in x slab {ix}")
                exposure_dataset[:, :, ix, :, :] = e2048
                geometry_dataset[:, ix, :, :] = g2048
                hit_dataset[ix, :, :] = hit2048
                observed["1024"] += e1024.sum(axis=(2, 3), dtype=np.float64)
                observed["2048"] += e2048.sum(axis=(2, 3), dtype=np.float64)
                geometry_observed["1024"] += g1024.sum(axis=(1, 2), dtype=np.float64)
                geometry_observed["2048"] += g2048.sum(axis=(1, 2), dtype=np.float64)

                active_ij = geometry["active_by_x"][ix]
                row_slice = slice(int(row_offsets[ix]), int(row_offsets[ix + 1]))
                if len(active_ij):
                    ys, zs = active_ij[:, 0], active_ij[:, 1]
                    active_radii = geometry["active_radii"][row_slice]
                    radius_bins = np.clip(
                        np.searchsorted(RADIUS_SUMMARY_EDGES, active_radii, side="right") - 1,
                        0, len(RADIUS_SUMMARY_EDGES) - 2)
                    new_low = e1024[:, :, ys, zs]
                    new_high = e2048[:, :, ys, zs]
                    old_slab = np.asarray(old_dataset[:, :, ix, :, :], dtype=np.float64)
                    old_active = old_slab[:, :, ys, zs]
                    relative_resolution[:, :, row_slice] = _relative_difference(
                        new_low, new_high).astype(np.float32)
                    relative_old_new[:, :, row_slice] = _relative_difference(
                        old_active, new_high).astype(np.float32)
                    old_positive = old_active > 0.
                    new_positive = new_high > 0.
                    for radial_bin in range(len(RADIUS_SUMMARY_EDGES) - 1):
                        selected_radius = radius_bins == radial_bin
                        support_counts["old_positive_cells"][:, :, radial_bin] += np.count_nonzero(
                            old_positive[:, :, selected_radius], axis=-1)
                        support_counts["new_positive_cells"][:, :, radial_bin] += np.count_nonzero(
                            new_positive[:, :, selected_radius], axis=-1)
                        support_counts["old_positive_new_zero_cells"][:, :, radial_bin] += np.count_nonzero(
                            old_positive[:, :, selected_radius]
                            & ~new_positive[:, :, selected_radius], axis=-1)
                        support_counts["old_zero_new_positive_cells"][:, :, radial_bin] += np.count_nonzero(
                            ~old_positive[:, :, selected_radius]
                            & new_positive[:, :, selected_radius], axis=-1)

                for nside in NSIDES:
                    target = totals[str(nside)]
                    current = stats[str(nside)]
                    for name in ("candidate_pixels", "box_hit_rays", "shell_hit_rays"):
                        target[name] += int(current[name])
                    target["zero_hit_cells"].extend(current["zero_hit_cells"])
                completed_slabs += 1
                if completed_slabs % 8 == 0:
                    output.flush()
                    print(json.dumps({
                        "progress_slabs_completed": completed_slabs,
                        "slabs_total": N,
                        "elapsed_seconds": time.monotonic() - started,
                        "nside2048_candidate_pixels_completed":
                            totals["2048"]["candidate_pixels"],
                    }), flush=True)

        output.create_dataset(
            "zero_ray_flat_index_nside2048",
            data=np.asarray(totals["2048"]["zero_hit_cells"], dtype=np.int64),
            dtype="i8")
        output.flush()

        closure = {}
        for nside in NSIDES:
            map_sums = fullsky_map_sums(nside, physics["maps"])
            expected = np.asarray([
                hp.nside2pixarea(nside) * map_sums[p]
                * np.diff(physics["cumulative_edges"][p]) / DX ** 3
                for p in range(6)
            ], dtype=np.float64)
            expected_geometry = (4. * np.pi / 3. *
                                 (RADIAL_EDGES[1:] ** 3 - RADIAL_EDGES[:-1] ** 3)
                                 / DX ** 3)
            got = observed[str(nside)]
            got_geometry = geometry_observed[str(nside)]
            rel = _relative_difference(got, expected)
            geom_rel = _relative_difference(got_geometry, expected_geometry)
            max_rel = float(np.nanmax(rel))
            max_geom_rel = float(np.nanmax(geom_rel))
            closure[str(nside)] = {
                "population_shell_max_relative_error": max_rel,
                "pure_geometry_shell_max_relative_error": max_geom_rel,
                "expected_population_shell": expected.tolist(),
                "observed_population_shell": got.tolist(),
                "expected_pure_geometry_shell": expected_geometry.tolist(),
                "observed_pure_geometry_shell": got_geometry.tolist(),
            }
            if max_rel > 1e-10 or max_geom_rel > 1e-10:
                output.attrs["status"] = "FAILED_CLOSURE"
                output.flush()
                failure = {
                    "status": "FAILED_CLOSURE",
                    "slurm_job_id": os.environ["SLURM_JOB_ID"],
                    "source_commit": expected_commit,
                    "closure": closure,
                }
                result_path.write_text(json.dumps(failure, indent=2, allow_nan=False) + "\n")
                raise RuntimeError(f"global selection/geometry closure failed at NSIDE={nside}")

        resolution_summary = _channel_radius_quantiles(
            relative_resolution, geometry["active_radii"])
        old_new_comparison = old_new_summary(
            relative_old_new, geometry["active_radii"], support_counts)
        elapsed = time.monotonic() - started
        parent_peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
        child_peak_gib = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024 ** 2
        report = {
            "classification": "NUMERICAL_SELECTION_ARTIFACT_NOT_LIKELIHOOD_READY",
            "status": "COMPLETE_NUMERICAL_SELECTION_ARTIFACT_NOT_LIKELIHOOD_READY",
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "source_commit": expected_commit,
            "grid": {"N": N, "box_cMpc_h": BOX, "dx_cMpc_h": DX,
                     "active_shell_cells": int(len(active_flat))},
            "radial_edges_cMpc_h": RADIAL_EDGES.tolist(),
            "angular_maps": {"native_nside": 512, "ordering": "RING",
                             "sha256": physics["map_hashes"]},
            "angular_quadrature_nsides": list(NSIDES),
            "population_definition": "two apparent-K limits times three fixed absolute-K bins",
            "shell_semantics": "ray intervals clipped at each radial edge; count likelihood aggregates shells by summing axis 1",
            "no_skip": True,
            "data_access": {"catalogue": False, "counts_or_keys": False,
                            "train_or_holdout": False, "field_states": False,
                            "native_truth_identities": False},
            "previous_exposure_comparison_input": {
                "path": str(OLD_SELECTION), **previous_contract,
                "sha256": sha256_file(OLD_SELECTION),
                "use": "post-integration cellwise diagnostic only; not an input to new exposure values",
            },
            "geometry_census": {
                "counts": geometry["counts"],
                "joint_inner_outer_cap_status": geometry["joint_status_crosstab"],
                "maximum_cap_rad": geometry["max_cap_rad"],
                "maximum_cap_cells": geometry["maximum_cap_cells"],
            },
            "ray_work": totals,
            "preflight": preflight,
            "closure": closure,
            "cellwise_nside1024_to_2048_relative_change": resolution_summary,
            "center_radius_summary_edges_cMpc_h": RADIUS_SUMMARY_EDGES.tolist(),
            "cellwise_order6_old_to_nside2048_relative_change": old_new_comparison,
            "closure_tolerance_relative": 1e-10,
            "resolution_relative_metric": "abs(a-b)/max(abs(a),abs(b)); both-zero cells excluded",
            "zero_ray_active_cell_count_nside1024": len(totals["1024"]["zero_hit_cells"]),
            "zero_ray_active_cell_count_nside2048": len(totals["2048"]["zero_hit_cells"]),
            "zero_ray_cells_are_quadrature_diagnostics_not_native_truth_labels": True,
            "selection_bias_calibrated": False,
            "shared_multimember_covariance_calibrated": False,
            "shared_latent_count_mark_law_complete": False,
            "posterior_or_density_field_created": False,
            "R2_complete": False,
            "MW_M31_M33": "MW/M31 roles remain ambiguous and M33 unresolved; their observables must later constrain those same roles in the NEW evolved field at LG <=0.3 cMpc/h; native truth IDs remain calibration/evaluation-only.",
            "elapsed_seconds": elapsed,
            "parent_peak_GiB": parent_peak_gib,
            "max_child_process_peak_GiB": child_peak_gib,
            "conservative_parallel_peak_upper_bound_GiB":
                parent_peak_gib + worker_count * child_peak_gib,
            "workers": worker_count,
        }
        output.attrs["status"] = report["status"]
        output.attrs["closure_json"] = json.dumps(closure, allow_nan=False)
        output.flush()
        result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": "COMPLETE_NUMERICAL_SELECTION_ARTIFACT_NOT_LIKELIHOOD_READY",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "active_cells": int(len(active_flat)),
        "elapsed_seconds": time.monotonic() - started,
        "closure_2048": closure["2048"]["population_shell_max_relative_error"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
