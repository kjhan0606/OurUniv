"""Repair missed selection support, using survey geometry, never galaxy occupancy.

Preserve v1. This is a bounded quadrature repair, not selection calibration:
nonzero v1 exposures retain their original quadrature error.
"""
import json
from pathlib import Path
import time

import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import CartesianRepresentation, SkyCoord
from scipy.spatial import cKDTree
from scipy.stats import qmc

from cf4_actual_selection import base
from cf4_twompp_joint_information_budget_pilot_v1 import (
    _cosmology_distance_table, schechter_fraction,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 2026090801
NSIDE = 512
SHELLS = np.array([5., 30., 60., 90., 120., 150., 180.])


def boundary_tree(mask):
    """Both sides of every positive/zero HEALPix footprint edge."""
    positive = mask > 0
    boundary = np.zeros(len(mask), dtype=bool)
    for low in range(0, len(mask), 65536):
        ids = np.arange(low, min(low + 65536, len(mask)))
        neighbours = hp.get_all_neighbours(NSIDE, ids)
        boundary[ids] = np.any(
            (neighbours >= 0)
            & (positive[np.maximum(neighbours, 0)] != positive[ids][None, :]), axis=0)
    ids = np.flatnonzero(boundary)
    return cKDTree(np.array(hp.pix2vec(NSIDE, ids)).T) if len(ids) else None


def footprint_possible(center, dx, rotation, mask, tree):
    """Conservative angular cap around a cube, expanded by HEALPix pixel radius."""
    radius = np.linalg.norm(center, axis=1)
    half_diagonal = np.sqrt(3) * dx / 2
    unit = (rotation @ center.T / np.maximum(radius, 1e-30)).T
    pixels = hp.vec2pix(NSIDE, *unit.T, nest=False)
    possible = mask[pixels] > 0
    if tree is not None:
        cap = np.arcsin(np.minimum(1, half_diagonal / np.maximum(radius, 1e-30)))
        cap += hp.max_pixrad(NSIDE)
        possible |= tree.query(unit, workers=2)[0] <= 2 * np.sin(cap / 2)
    return possible | (radius <= half_diagonal)


def integrate_cells(center, dx, offsets, rotation, maps, rtab, radial):
    """Equal-volume scrambled Sobol cubature; returns population/shell/cell means."""
    result = np.zeros((6, 6, len(center)))
    for low in range(0, len(center), 64):
        cc = center[low:low + 64]
        pos = cc[:, None, :] + dx * offsets[None, :, :]
        rr = np.linalg.norm(pos, axis=2)
        active = (rr >= SHELLS[0]) & (rr <= SHELLS[-1])
        ci, qi = np.nonzero(active)
        if not len(ci):
            continue
        radius = rr[ci, qi]
        pixels = hp.vec2pix(NSIDE, *(rotation @ pos[ci, qi].T), nest=False)
        shell = np.clip(np.searchsorted(SHELLS, radius, side="right") - 1, 0, 5)
        key = shell * len(cc) + ci
        for p in range(6):
            value = maps[p // 3][pixels] * np.interp(radius, rtab, radial[p])
            result[p, :, low:low + len(cc)] = np.bincount(
                key, weights=value, minlength=6 * len(cc)).reshape(6, len(cc)) / len(offsets)
    return result


def main():
    start = time.monotonic()
    cfg = json.loads((ROOT / "config/cf4_bundle_c_v1.json").read_text())
    tr = json.loads((ROOT / cfg["tracer_program"]).read_text())
    root = Path(cfg["output_root"])
    previous = root / "selection_1p5_v1"
    out = root / "selection_1p5_v2"
    out.mkdir(exist_ok=False)
    old_report = json.loads((previous / "result.json").read_text())
    maps = [base.load_completeness_map(tr["inputs"][name]["path"], NSIDE)
            for name in ("completeness_11_5", "completeness_12_5")]
    trees = [boundary_tree(mask) for mask in maps]
    rotation = SkyCoord(CartesianRepresentation(np.eye(3)), frame="supergalactic").icrs.cartesian.xyz.value
    rtab = np.linspace(5., 180., 200001)
    dl = _cosmology_distance_table(rtab, tr["cosmology"]) * tr["cosmology"]["h"]
    absolute = tr["tracer_design"]["absolute_K_edges"]
    radial = np.array([schechter_fraction(
        dl, None if p // 3 == 0 else 11.5, 11.5 if p // 3 == 0 else 12.5,
        absolute[p % 3], absolute[p % 3 + 1], -23.28, -.94) for p in range(6)])
    support = []
    for table in radial:
        indices = np.flatnonzero(table > 0)
        support.append((rtab[max(0, indices[0] - 1)], rtab[min(len(rtab) - 1, indices[-1] + 1)]))
    # Fixed before opening any galaxy/count file; one identical rule for empty cells.
    offsets = qmc.Sobol(3, scramble=True, seed=SEED).random_base2(11) - .5
    n, dx = cfg["background_N"], cfg["background_dx_cMpc_h"]
    axis = (np.arange(n) + .5) * dx - cfg["box_cMpc_h"] / 2
    totals = np.zeros((6, 6))
    candidate_keys, added_keys, added_values = [], [], []
    controls = []
    with h5py.File(previous / "selection.h5", "r") as src, h5py.File(out / "selection.h5", "x") as dst:
        if src.attrs["status"] != "INTEGRATION_COMPLETE_NOT_SELECTION_CALIBRATION":
            raise ValueError("v1 integration must be complete")
        ds = dst.create_dataset("selection_shells", shape=src["selection_shells"].shape,
            dtype="f4", chunks=src["selection_shells"].chunks, compression="gzip", compression_opts=1)
        dst.attrs.update(dict(src.attrs))
        dst.attrs.update(status="INCOMPLETE", support_method="geometry_selected_Sobol_2048", support_seed=SEED)
        for low in range(0, n, 8):
            center = np.array(np.meshgrid(axis[low:low + 8], axis, axis, indexing="ij")).reshape(3, -1).T
            block = src["selection_shells"][:, :, low:low + 8].reshape(6, 6, -1)
            exposure = block.sum(axis=1)
            # Exact radial min/max of an axis-aligned cube, not just its centre.
            rmin = np.linalg.norm(np.maximum(abs(center) - dx / 2, 0), axis=1)
            rmax = np.linalg.norm(abs(center) + dx / 2, axis=1)
            possible = np.array([footprint_possible(center, dx, rotation, mask, tree)
                                 for mask, tree in zip(maps, trees)])
            candidate = np.array([(exposure[p] == 0) & possible[p // 3]
                & (rmax > support[p][0]) & (rmin < support[p][1]) for p in range(6)])
            cells = np.flatnonzero(candidate.any(axis=0))
            integral = integrate_cells(center[cells], dx, offsets, rotation, maps, rtab, radial)
            for p in range(6):
                chosen = candidate[p, cells]
                jj = cells[chosen]
                values = integral[p][:, chosen]
                keys = p * n**3 + low * n**2 + jj
                candidate_keys.extend(keys.tolist())
                positive = values.sum(axis=0) > 0
                added_keys.extend(keys[positive].tolist())
                added_values.extend(values[:, positive].sum(axis=0).tolist())
                block[p][:, jj] = values.astype(np.float32)
            # Fixed spatially distributed, occupancy-blind convergence controls.
            if len(cells):
                sample = np.unique(np.linspace(0, len(cells) - 1, min(4, len(cells)), dtype=int))
                for j in sample:
                    controls.append((center[cells[j]], candidate[:, cells[j]], integral[:, :, j]))
            if not np.all(np.isfinite(block)) or np.any(block < 0) or np.any(block.sum(axis=1) > 1 + 1e-6):
                raise ValueError("invalid cell selection probabilities")
            totals += block.astype(np.float64).sum(axis=2) * dx**3
            ds[:, :, low:low + 8] = block.reshape(6, 6, 8, n, n)
            dst.flush()
            print(f"support x={low + 8}/{n}, candidates={len(candidate_keys)}, added={len(added_keys)}, seconds={time.monotonic()-start:.1f}", flush=True)
        dst.attrs["status"] = "INTEGRATION_COMPLETE_NOT_SELECTION_CALIBRATION"
    # Counts are opened only AFTER the output integrals and control set are fixed.
    with np.load(root / cfg["native_data_subdir"] / "counts_1p5_sparse.npz") as data:
        occupied = data["all_keys"]
    keys = np.array(added_keys, dtype=np.int64)
    np.savez_compressed(out / "support_changes.npz", candidate_keys=np.array(candidate_keys, dtype=np.int64),
                        added_keys=keys, added_exposure=np.array(added_values))
    zero = []
    with h5py.File(out / "selection.h5", "r") as h:
        pop = occupied // n**3
        cell = np.array(np.unravel_index(occupied % n**3, (n,) * 3)).T
        for low in range(0, n, 8):
            take = np.flatnonzero((cell[:, 0] >= low) & (cell[:, 0] < low + 8))
            block = h["selection_shells"][:, :, low:low + 8].sum(axis=1)
            values = block[pop[take], cell[take, 0] - low, cell[take, 1], cell[take, 2]]
            zero.extend(occupied[take[values <= 0]].tolist())
    control_report = {}
    if controls:
        cc, mask, coarse = map(np.array, zip(*controls))
        reference_offsets = qmc.Sobol(3, scramble=True, seed=SEED + 1).random_base2(15) - .5
        reference = integrate_cells(cc, dx, reference_offsets, rotation, maps, rtab, radial).transpose(2, 0, 1)
        error = np.abs(coarse - reference).sum(axis=2)[mask]
        control_report = dict(cells=len(cc), independent_reference_points=32768,
            max_shell_L1_absolute_error=float(error.max()), mean_shell_L1_absolute_error=float(error.mean()),
            reference_positive_missed_by_2048=int(np.count_nonzero(mask & (reference.sum(axis=2) > 0) & (coarse.sum(axis=2) == 0))))
    recovered_occupied = np.intersect1d(keys, occupied)
    report = dict(status="SUPPORT_REPAIRED_NOT_SELECTION_CALIBRATION" if not zero else "UNRESOLVED_POSITIVE_COUNT_SUPPORT",
        source=str(previous), N=n, dx_cMpc_h=dx, method="2048-point fixed scrambled Sobol for geometry-eligible zero-exposure population/cells",
        seed=SEED, candidates=len(candidate_keys), added_population_cells=len(keys),
        added_occupied_keys=recovered_occupied.tolist(), added_unoccupied_population_cells=int(len(keys)-len(recovered_occupied)),
        zero_exposure_positive_count_keys=zero, raw_effective_volumes_cMpc_h3=totals.tolist(),
        effective_volume_change_cMpc_h3=(totals - np.array(old_report["raw_effective_volumes_cMpc_h3"])).tolist(),
        controls=control_report, elapsed_seconds=time.monotonic()-start,
        diagnosis="v1 order4 misses a narrow positive footprint in cell[107,129,128], population2. Source rows6921/6935 have mask0.51851851 and LF1; no photometry/distance mismatch.",
        limits="Finite cubature can still miss very thin support. Nonzero v1 entries are unchanged, not precision-certified. This repairs numerical zero support, not angular-map/survival/bias calibration, a density posterior, or a resolved LG operator.")
    (out / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)
    if zero:
        raise RuntimeError("occupied support remains unresolved; do not run a count likelihood")


if __name__ == "__main__":
    main()
