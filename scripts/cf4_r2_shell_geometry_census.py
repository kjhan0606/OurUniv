"""Exact grid-shell census and bounded inner-cell ray census for R2.

Geometry only: this job reads no catalogue, field, train/holdout, or selection
map data. It counts every N=128 voxel intersecting 5 < r < 180 cMpc/h, sums
solid-angle candidate estimates over that exact support, and directly counts
HEALPix candidate/intersecting pixel-center rays for every maximum-cap voxel.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from itertools import product

import healpy as hp
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from cf4_r2_ray_cost_profile import ray_box_intervals  # noqa: E402

N = 128
DX = 3.0
BOX = 384.0
HALF = BOX / 2.0
R_INNER = 5.0
R_OUTER = 180.0
NSIDE = 2048
RADIAL_BINS = np.array([0., 5., 15., 30., 60., 90., 120., 150., 180., 186.])
CHUNK_PIXELS = 250_000


def interval_distance_bounds(low: np.ndarray, high: np.ndarray):
    """Exact min/max distance from origin to an axis-aligned 3-D box."""
    nearest = np.where((low <= 0.) & (high >= 0.), 0.,
                       np.minimum(np.abs(low), np.abs(high)))
    farthest = np.maximum(np.abs(low), np.abs(high))
    return float(np.linalg.norm(nearest)), float(np.linalg.norm(farthest))


def cell_cap(center: np.ndarray, half_width: float = DX / 2.0) -> float:
    """Conservative angular cap about the center ray, padded for roundoff."""
    radius = float(np.linalg.norm(center))
    if radius == 0.:
        return float(np.pi)
    corners = center[None, :] + np.asarray(list(product((-half_width, half_width), repeat=3)))
    norms = np.linalg.norm(corners, axis=1)
    valid = norms > 0.
    cosines = corners[valid] @ (center / radius) / norms[valid]
    return float(min(np.pi, np.arccos(np.clip(np.min(cosines), -1., 1.)) + 1e-6))


def self_test():
    low = np.array([0., 0., 0.])
    high = np.array([3., 3., 3.])
    rmin, rmax = interval_distance_bounds(low, high)
    if not np.isclose(rmin, 0.) or not np.isclose(rmax, 3. * np.sqrt(3.)):
        raise AssertionError("origin-corner voxel distance bounds are wrong")
    cap = cell_cap((low + high) / 2.)
    if not np.isclose(cap, np.arccos(1. / np.sqrt(3.)) + 1e-6):
        raise AssertionError("origin-corner voxel angular cap is wrong")
    if not (rmin < R_INNER < rmax):
        raise AssertionError("near-observer control must partially intersect inner shell")
    low_outer = np.array([180., 0., 0.])
    high_outer = low_outer + DX
    tangent_min, tangent_max = interval_distance_bounds(low_outer, high_outer)
    if tangent_min != R_OUTER or tangent_max <= R_INNER:
        raise AssertionError("outer-shell tangency control changed")
    if (tangent_max > R_INNER) and (tangent_min < R_OUTER):
        raise AssertionError("outer-shell strict-intersection convention changed")
    # The cap contains every nonzero box-corner direction; convexity then bounds
    # every ray through the box. This fixture checks all eight corner directions.
    center = (low + high) / 2.
    for corner in corners_of_box(low, high):
        if np.linalg.norm(corner) == 0.:
            continue
        angle = np.arccos(np.clip(np.dot(corner, center) /
                                  (np.linalg.norm(corner) * np.linalg.norm(center)), -1., 1.))
        if angle > cap:
            raise AssertionError("angular cap excludes a box-corner direction")
    return 4


def corners_of_box(low: np.ndarray, high: np.ndarray):
    return np.asarray(list(product(*zip(low, high))), dtype=np.float64)


def observer_rotation():
    return SkyCoord(CartesianRepresentation(np.eye(3)), frame="supergalactic").icrs.cartesian.xyz.value


def census_all_cells():
    """Stream 128 x 128 x 128 cells by x slab; return exact support counts."""
    indices = np.arange(N, dtype=np.float64)
    lows = indices * DX - HALF
    highs = lows + DX
    centers_1d = (lows + highs) / 2.
    near_1d = np.where((lows <= 0.) & (highs >= 0.), 0.,
                       np.minimum(np.abs(lows), np.abs(highs)))
    far_1d = np.maximum(np.abs(lows), np.abs(highs))
    cy, cz = np.meshgrid(centers_1d, centers_1d, indexing="ij")
    ny, nz = np.meshgrid(near_1d, near_1d, indexing="ij")
    fy, fz = np.meshgrid(far_1d, far_1d, indexing="ij")
    cy, cz, ny, nz, fy, fz = [a.ravel() for a in (cy, cz, ny, nz, fy, fz)]

    counts = {
        "active_shell_intersection": 0,
        "fully_inside_shell": 0,
        "inner_boundary_partial": 0,
        "outer_boundary_partial": 0,
        "both_boundaries_partial": 0,
        "center_radius_bins": [0] * (len(RADIAL_BINS) - 1),
        "center_radius_bins_cap_estimate_pixels": [0.0] * (len(RADIAL_BINS) - 1),
        "active_cells_over_1p5m_cap_estimate": 0,
        "cap_estimate_pixel_sum": 0.0,
    }
    max_cap = -1.0
    max_cells = []
    npix = int(hp.nside2npix(NSIDE))
    corner_offsets = np.asarray(list(product((-DX / 2., DX / 2.), repeat=3)))

    for ix in range(N):
        xlo, xhi = lows[ix], highs[ix]
        xc = centers_1d[ix]
        nx, fx = near_1d[ix], far_1d[ix]
        rmin = np.sqrt(nx * nx + ny * ny + nz * nz)
        rmax = np.sqrt(fx * fx + fy * fy + fz * fz)
        active = (rmax > R_INNER) & (rmin < R_OUTER)
        if not np.any(active):
            continue

        active_indices = np.flatnonzero(active)
        y = cy[active]
        z = cz[active]
        centers = np.column_stack((np.full(len(y), xc), y, z))
        radii = np.linalg.norm(centers, axis=1)
        min_cos = np.full(len(y), np.inf)
        for offset in corner_offsets:
            vx, vy, vz = centers[:, 0] + offset[0], centers[:, 1] + offset[1], centers[:, 2] + offset[2]
            vnorm = np.sqrt(vx * vx + vy * vy + vz * vz)
            nonzero = vnorm > 0.
            cosine = np.full(len(y), np.inf)
            cosine[nonzero] = (centers[nonzero, 0] * vx[nonzero]
                               + centers[nonzero, 1] * vy[nonzero]
                               + centers[nonzero, 2] * vz[nonzero]) / (radii[nonzero] * vnorm[nonzero])
            min_cos = np.minimum(min_cos, cosine)
        caps = np.arccos(np.clip(min_cos, -1., 1.)) + 1e-6
        candidates = npix * (1. - np.cos(caps)) / 2.
        bins = np.clip(np.searchsorted(RADIAL_BINS, radii, side="right") - 1,
                       0, len(RADIAL_BINS) - 2)

        counts["active_shell_intersection"] += int(len(y))
        inner = (rmin[active] < R_INNER) & (rmax[active] > R_INNER)
        outer = (rmin[active] < R_OUTER) & (rmax[active] > R_OUTER)
        counts["inner_boundary_partial"] += int(np.count_nonzero(inner))
        counts["outer_boundary_partial"] += int(np.count_nonzero(outer))
        counts["both_boundaries_partial"] += int(np.count_nonzero(inner & outer))
        counts["fully_inside_shell"] += int(np.count_nonzero(
            (rmin[active] >= R_INNER) & (rmax[active] <= R_OUTER)))
        counts["active_cells_over_1p5m_cap_estimate"] += int(np.count_nonzero(candidates > 1_500_000.))
        counts["cap_estimate_pixel_sum"] += float(np.sum(candidates, dtype=np.float64))
        for b in range(len(RADIAL_BINS) - 1):
            selected = bins == b
            counts["center_radius_bins"][b] += int(np.count_nonzero(selected))
            counts["center_radius_bins_cap_estimate_pixels"][b] += float(
                np.sum(candidates[selected], dtype=np.float64))

        local_max = float(np.max(caps))
        if local_max > max_cap + 1e-12:
            max_cap = local_max
            max_cells = []
        if abs(local_max - max_cap) <= 1e-12:
            loc = np.flatnonzero(np.abs(caps - max_cap) <= 1e-12)
            for j in loc:
                kyz = int(active_indices[j])
                iy, iz = divmod(kyz, N)
                max_cells.append({
                    "ijk": [int(ix), int(iy), int(iz)],
                    "low_cMpc_h": [float(xlo), float(lows[iy]), float(lows[iz])],
                    "high_cMpc_h": [float(xhi), float(highs[iy]), float(highs[iz])],
                    "center_cMpc_h": centers[j].tolist(),
                    "center_radius_cMpc_h": float(radii[j]),
                    "rmin_cMpc_h": float(rmin[kyz]),
                    "rmax_cMpc_h": float(rmax[kyz]),
                    "cap_rad": float(caps[j]),
                    "cap_area_candidate_estimate": float(candidates[j]),
                })

    if counts["active_shell_intersection"] != (
            counts["fully_inside_shell"] + counts["inner_boundary_partial"]
            + counts["outer_boundary_partial"] - counts["both_boundaries_partial"]):
        raise AssertionError("shell-cell partition does not close")
    max_cells.sort(key=lambda row: row["ijk"])
    return counts, max_cap, max_cells


def exact_ray_counts(cell: dict, rotation: np.ndarray):
    low = np.asarray(cell["low_cMpc_h"], dtype=np.float64)
    high = np.asarray(cell["high_cMpc_h"], dtype=np.float64)
    center = np.asarray(cell["center_cMpc_h"], dtype=np.float64)
    center_sky = rotation @ center
    axis = center_sky / np.linalg.norm(center_sky)
    corners = (rotation @ corners_of_box(low, high).T).T
    norms = np.linalg.norm(corners, axis=1)
    cosines = (corners[norms > 0.] @ axis) / norms[norms > 0.]
    cap = float(min(np.pi, np.arccos(np.clip(np.min(cosines), -1., 1.)) + 1e-6))
    pixels = hp.query_disc(NSIDE, axis, cap, inclusive=True, nest=False)
    box_hits = 0
    shell_hits = 0
    for start in range(0, len(pixels), CHUNK_PIXELS):
        part = pixels[start:start + CHUNK_PIXELS]
        sky_dirs = np.asarray(hp.pix2vec(NSIDE, part, nest=False))
        directions = rotation.T @ sky_dirs
        enter, leave, intersects = ray_box_intervals(directions, low, high)
        box_hits += int(np.count_nonzero(intersects))
        shell_hits += int(np.count_nonzero(
            intersects & (np.minimum(leave, R_OUTER) > np.maximum(enter, R_INNER))))
    return {
        "ijk": cell["ijk"],
        "cap_rad": cap,
        "candidate_pixels_exact_query_disc_count": int(len(pixels)),
        "candidate_center_rays_intersecting_box": box_hits,
        "candidate_center_rays_intersecting_5_to_180_shell": shell_hits,
        "candidate_chunk_pixels": CHUNK_PIXELS,
    }


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("shell geometry census must run through Slurm")
    expected = os.environ.get("CF4_EXPECTED_COMMIT")
    out_value = os.environ.get("CF4_R2_OUT_DIR")
    if not expected or not out_value:
        raise RuntimeError("CF4_EXPECTED_COMMIT and CF4_R2_OUT_DIR are required")
    out = Path(out_value)
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    tests = self_test()
    counts, max_cap, max_cells = census_all_cells()
    rotation = observer_rotation()
    max_cell_rays = [exact_ray_counts(cell, rotation) for cell in max_cells]
    result = {
        "classification": "GEOMETRY_ONLY_EXACT_SHELL_CELL_AND_MAX_CAP_RAY_CENSUS",
        "status": "GEOMETRY_CENSUS_COMPLETE_NO_SELECTION_INTEGRATION",
        "source_commit": expected,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "grid": {"N": N, "dx_cMpc_h": DX, "box_cMpc_h": BOX},
        "radial_shell_cMpc_h": [R_INNER, R_OUTER],
        "angular_quadrature": {"nside": NSIDE, "ordering": "RING", "candidate_inclusive": True,
                               "chunk_pixels": CHUNK_PIXELS},
        "data_access": {"catalogues": False, "selection_maps": False, "field": False,
                        "train_or_holdout_arrays": False, "likelihood_or_counts": False},
        "geometry_self_tests_passed": tests,
        "all_grid_cell_census": counts,
        "maximum_cap_rad": max_cap,
        "maximum_cap_cells": max_cells,
        "exact_healpix_rays_for_maximum_cap_cells": max_cell_rays,
        "cap_area_pixel_sum_is_estimate_not_exact_query_disc_sum": True,
        "angular_accuracy_certified": False,
        "selection_calibration_complete": False,
        "quantitative_joint_likelihood_ready": False,
        "R2_complete": False,
        "Q_GOAL": "Resolve omitted observer-near/partial-shell geometry for the same CF4-conditioned z=0 field; this census infers no field.",
        "Q_LEAN": "One geometry-only grid census and exact rays for the maximum-cap cells; no catalogue, field, heldout, fit, PM evolution, or simulation.",
        "MW_M31_M33": "MW/M31 remain role-ambiguous and M33 unresolved; their observables must later constrain those same roles in the NEW evolved field at LG <=0.3 cMpc/h; native truth IDs remain calibration/evaluation-only.",
        "open_R2_limitations": ["cellwise angular accuracy", "selection/bias calibration", "multi-member covariance", "shared-latent count/mark conditional"],
        "elapsed_seconds": time.monotonic() - started,
        "host_peak_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
    }
    out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "classification": result["classification"],
        "all_grid_cell_census": counts,
        "maximum_cap_rad": max_cap,
        "maximum_cap_cell_count": len(max_cells),
        "exact_healpix_rays_for_maximum_cap_cells": max_cell_rays,
        "elapsed_seconds": result["elapsed_seconds"],
        "host_peak_GiB": result["host_peak_GiB"],
    }, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
