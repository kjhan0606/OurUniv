"""Recompute full N128 NSIDE1024 exposures, then audit training-only impact.

The data-free operator is written, closed, and SHA256-bound before this script
opens the sparse training keys/counts. It never reads all-key or heldout arrays.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import cf4_r2_selection_ray_integral as ray  # noqa: E402

N = ray.N
DX = ray.DX
OUT_DEFAULT = Path(
    "/gpfs/kjhan/CF4/z0_density/"
    "r2_ray_selection_n128_nside1024_trainingimpact_20261004_v1")
RAY2048_DIR = Path(
    "/gpfs/kjhan/CF4/z0_density/"
    "r2_ray_selection_n128_nside1024_2048_20261004_v1")
RAY2048_H5 = RAY2048_DIR / "selection_ray.h5"
RAY2048_RESULT = RAY2048_DIR / "result.json"
DATA = Path("/gpfs/kjhan/CF4/z0_density/r2_common_catalogue_128_v1")
OLD_SELECTION = ray.OLD_SELECTION
CHUNK_SIZE = 250_000

# Freeze the decision rule in source before training counts are opened.
DELTA_LOG_LIKELIHOOD_TOL_NATS = 1.0
MAX_CELL_SHOT_NOISE_Z = 0.5
FRACTION_COUNTS_ABOVE_Z01_TOL = 0.01

_WORKER_STATE: dict | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _worker_init(physics: dict, active_by_x: list[np.ndarray]):
    global _WORKER_STATE
    _WORKER_STATE = {"physics": physics, "active_by_x": active_by_x}


def _integrate_nside1024_x_slab(ix: int):
    if _WORKER_STATE is None:
        raise RuntimeError("NSIDE1024 worker was not initialized")
    physics = _WORKER_STATE["physics"]
    active_ij = _WORKER_STATE["active_by_x"][ix]
    exposure = np.zeros((6, 6, N, N), dtype=np.float64)
    geometry = np.zeros((6, N, N), dtype=np.float64)
    hit_count = np.zeros((N, N), dtype=np.uint32)
    stats = {"candidate_pixels": 0, "box_hit_rays": 0,
             "shell_hit_rays": 0, "zero_hit_cells": []}
    for iy0, iz0 in active_ij:
        iy, iz = int(iy0), int(iz0)
        value = ray.integrate_cell((ix, iy, iz), 1024, physics, CHUNK_SIZE)
        if (not np.isfinite(value["exposure"]).all()
                or np.any(value["exposure"] < 0.)
                or not np.isfinite(value["geometry"]).all()
                or np.any(value["geometry"] < 0.)):
            raise ValueError(f"invalid NSIDE1024 value in cell {(ix, iy, iz)}")
        exposure[:, :, iy, iz] = value["exposure"]
        geometry[:, iy, iz] = value["geometry"]
        hit_count[iy, iz] = value["shell_hit_rays"]
        for name in ("candidate_pixels", "box_hit_rays", "shell_hit_rays"):
            stats[name] += int(value[name])
        if value["shell_hit_rays"] == 0:
            stats["zero_hit_cells"].append(int((ix * N + iy) * N + iz))
    return ix, exposure, geometry, hit_count, stats


def _interior_cell_mask(flat_indices: np.ndarray):
    ijk = np.asarray(np.unravel_index(flat_indices, (N, N, N))).T
    low = ijk.astype(np.float64) * DX - ray.HALF
    high = low + DX
    nearest = np.where((low <= 0.) & (high >= 0.), 0.,
                       np.minimum(np.abs(low), np.abs(high)))
    farthest = np.maximum(np.abs(low), np.abs(high))
    rmin = np.linalg.norm(nearest, axis=1)
    rmax = np.linalg.norm(farthest, axis=1)
    return (rmin >= ray.R_INNER) & (rmax <= ray.R_OUTER), ijk


def _expected_closure(nside: int, physics: dict):
    map_sums = ray.fullsky_map_sums(nside, physics["maps"])
    expected_population = np.asarray([
        ray.hp.nside2pixarea(nside) * map_sums[p]
        * np.diff(physics["cumulative_edges"][p]) / DX ** 3
        for p in range(6)], dtype=np.float64)
    expected_geometry = (4. * np.pi / 3. *
                         (ray.RADIAL_EDGES[1:] ** 3 - ray.RADIAL_EDGES[:-1] ** 3)
                         / DX ** 3)
    return expected_population, expected_geometry


def _interior_geometry_audit(h5_path: Path, geometry: dict):
    flat = geometry["flat_indices"]
    interior_mask, ijk = _interior_cell_mask(flat)
    deviations = []
    with h5py.File(h5_path, "r") as handle:
        dataset = handle["geometry_shells"]
        for ix in range(N):
            rows = np.flatnonzero((ijk[:, 0] == ix) & interior_mask)
            if not len(rows):
                continue
            slab = np.asarray(dataset[:, ix, :, :], dtype=np.float64).sum(axis=0)
            deviations.extend(np.abs(slab[ijk[rows, 1], ijk[rows, 2]] - 1.).tolist())
    values = np.asarray(deviations, dtype=np.float64)
    if len(values) != 870_528:
        raise AssertionError(f"interior geometry audit saw {len(values)} cells")
    return {
        "fully_interior_cells": int(len(values)),
        "target_geometry_exposure": 1.0,
        "absolute_cellwise_deviation_p50_p95_p99_max":
            np.quantile(values, [0.5, 0.95, 0.99, 1.0]).tolist(),
        "cells_abs_deviation_gt_1e-3": int(np.count_nonzero(values > 1e-3)),
        "cells_abs_deviation_gt_1e-2": int(np.count_nonzero(values > 1e-2)),
        "data_free": True,
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("weighted quantile inputs must be aligned vectors")
    if not len(values) or np.any(weights <= 0.) or not np.isfinite(values).all():
        raise ValueError("weighted quantile requires finite values and positive weights")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    index = int(np.searchsorted(cumulative, q * cumulative[-1], side="left"))
    return float(values[order[min(index, len(order) - 1)]])


def _classify_zero_exposure(rmin: float, rmax: float, ray_hits: int):
    """Separate domain exclusion from a finite-ray miss in active geometry."""
    if rmax <= ray.R_INNER or rmin >= ray.R_OUTER:
        return "outside_selection_domain"
    if ray_hits == 0:
        return "positive_volume_domain_intersection_missed_by_finite_rays"
    return "rays_intersect_domain_but_population_exposure_is_zero"


def _training_cell_geometry(keys: np.ndarray):
    flat = np.asarray(keys, dtype=np.int64) % N ** 3
    ijk = np.asarray(np.unravel_index(flat, (N, N, N))).T
    low = ijk.astype(np.float64) * DX - ray.HALF
    high = low + DX
    nearest = np.where((low <= 0.) & (high >= 0.), 0.,
                       np.minimum(np.abs(low), np.abs(high)))
    farthest = np.maximum(np.abs(low), np.abs(high))
    rmin = np.linalg.norm(nearest, axis=1)
    rmax = np.linalg.norm(farthest, axis=1)
    crossed = [
        [float(edge) for edge in ray.RADIAL_EDGES
         if rmin[row] < edge < rmax[row]]
        for row in range(len(flat))
    ]
    active = (rmax > ray.R_INNER) & (rmin < ray.R_OUTER)
    return flat, ijk, rmin, rmax, crossed, active


def summarize_exposure_triplet(keys, counts, old_exposure,
                               exposure1024, exposure2048):
    """Summarize support and predeclared weighted resolution diagnostics."""
    keys = np.asarray(keys, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    old_exposure = np.asarray(old_exposure, dtype=np.float64)
    exposure1024 = np.asarray(exposure1024, dtype=np.float64)
    exposure2048 = np.asarray(exposure2048, dtype=np.float64)
    shape = keys.shape
    if keys.ndim != 1 or any(value.shape != shape for value in (
            counts, old_exposure, exposure1024, exposure2048)):
        raise ValueError("training key/count/exposure vectors are misaligned")
    if np.any(counts <= 0):
        raise ValueError("training counts must be positive")
    for name, values in (("old", old_exposure), ("NSIDE1024", exposure1024),
                         ("NSIDE2048", exposure2048)):
        if not np.isfinite(values).all() or np.any(values < 0.):
            raise ValueError(f"{name} exposure is non-finite or negative")
    population = keys // N ** 3
    if np.any((population < 0) | (population >= 6)):
        raise ValueError("training key is outside the six-population namespace")

    rows = []
    for p in range(6):
        selected = population == p
        n = counts[selected]
        old = old_exposure[selected]
        e1024 = exposure1024[selected]
        e2048 = exposure2048[selected]
        old_positive = old > 0.
        low_positive = e1024 > 0.
        high_positive = e2048 > 0.
        common = low_positive & high_positive
        log_ratio = np.log(e2048[common] / e1024[common])
        common_counts = n[common]
        z = np.abs(log_ratio) * np.sqrt(common_counts.astype(np.float64))
        weighted_delta = (float(np.dot(common_counts, log_ratio))
                          if len(log_ratio) else 0.0)
        common_support_for_all = bool(np.all(common))
        signed_delta = weighted_delta if common_support_for_all else None
        frac_gt_01 = (float(n[common][z > 0.1].sum() / n.sum())
                      if n.sum() > 0 else 0.0)
        any_z_gt_05 = bool(np.any(z > 0.5))
        p50_p95_max = ([_weighted_quantile(np.abs(log_ratio), common_counts, q)
                        for q in (0.5, 0.95, 1.0)] if len(log_ratio) else [None] * 3)
        support = {
            "training_keys": int(np.count_nonzero(selected)),
            "training_galaxies": int(n.sum()),
            "old_order6_zero_keys": int(np.count_nonzero(~old_positive)),
            "old_order6_zero_galaxies": int(n[~old_positive].sum()),
            "old_zero_nside2048_positive_keys": int(np.count_nonzero(~old_positive & high_positive)),
            "old_zero_nside2048_positive_galaxies": int(n[~old_positive & high_positive].sum()),
            "nside1024_zero_keys": int(np.count_nonzero(~low_positive)),
            "nside1024_zero_galaxies": int(n[~low_positive].sum()),
            "nside2048_zero_keys": int(np.count_nonzero(~high_positive)),
            "nside2048_zero_galaxies": int(n[~high_positive].sum()),
            "nside1024_zero_nside2048_positive_keys":
                int(np.count_nonzero(~low_positive & high_positive)),
            "nside1024_zero_nside2048_positive_galaxies":
                int(n[~low_positive & high_positive].sum()),
            "nside1024_positive_nside2048_zero_keys":
                int(np.count_nonzero(low_positive & ~high_positive)),
            "nside1024_positive_nside2048_zero_galaxies":
                int(n[low_positive & ~high_positive].sum()),
        }
        thresholds_pass = (
            common_support_for_all
            and abs(signed_delta) < DELTA_LOG_LIKELIHOOD_TOL_NATS
            and not any_z_gt_05
            and frac_gt_01 < FRACTION_COUNTS_ABOVE_Z01_TOL)
        rows.append({
            "population": p,
            "support": support,
            "nside2048_vs_nside1024_common_support": {
                "signed_count_weighted_delta_log_exposure_nats": weighted_delta,
                "full_signed_delta_finite": common_support_for_all,
                "full_signed_delta_log_likelihood_nats": signed_delta,
                "count_weighted_abs_log_ratio_p50_p95_max": p50_p95_max,
                "training_count_fraction_in_cells_z_gt_0_1": frac_gt_01,
                "cells_with_z_gt_0_5": int(np.count_nonzero(z > 0.5)),
                "galaxies_in_cells_z_gt_0_1": int(n[common][z > 0.1].sum()),
            },
            "predeclared_negligible_rule_pass": bool(thresholds_pass),
            "resolution_gate_pass": bool(thresholds_pass and support["nside2048_zero_keys"] == 0),
        })
    return rows


def _read_training_exposure_by_key(path: Path, keys: np.ndarray, dataset_name: str):
    flat = keys % N ** 3
    population = keys // N ** 3
    ijk = np.asarray(np.unravel_index(flat, (N, N, N))).T
    result = np.zeros(len(keys), dtype=np.float64)
    with h5py.File(path, "r") as handle:
        dataset = handle[dataset_name]
        if dataset.shape != (6, 6, N, N, N):
            raise ValueError(f"unexpected exposure dataset shape in {path}")
        for ix in np.unique(ijk[:, 0]):
            rows = np.flatnonzero(ijk[:, 0] == ix)
            slab = np.asarray(dataset[:, :, int(ix), :, :], dtype=np.float64)
            collapsed = slab.sum(axis=1, dtype=np.float64)
            result[rows] = collapsed[population[rows], ijk[rows, 1], ijk[rows, 2]]
    return result


def _read_training_scalar_by_flat_key(path: Path, keys: np.ndarray,
                                      dataset_name: str):
    flat = np.asarray(keys, dtype=np.int64) % N ** 3
    ijk = np.asarray(np.unravel_index(flat, (N, N, N))).T
    result = np.zeros(len(keys), dtype=np.uint32)
    with h5py.File(path, "r") as handle:
        dataset = handle[dataset_name]
        if dataset.shape != (N, N, N):
            raise ValueError(f"unexpected scalar dataset shape in {path}")
        for ix in np.unique(ijk[:, 0]):
            rows = np.flatnonzero(ijk[:, 0] == ix)
            slab = np.asarray(dataset[int(ix), :, :], dtype=np.uint32)
            result[rows] = slab[ijk[rows, 1], ijk[rows, 2]]
    return result


def _training_impact(data_free: dict, output_dir: Path, expected_commit: str):
    count_path = DATA / "counts_3_sparse.npz"
    # Deliberately access exactly these two arrays; do not touch all_keys or any
    # holdout key/count arrays from the archive.
    with np.load(count_path, allow_pickle=False) as archive:
        keys = np.asarray(archive["train_keys"], dtype=np.int64).copy()
        counts = np.asarray(archive["train_counts"], dtype=np.int64).copy()
    if (keys.ndim != 1 or counts.shape != keys.shape or len(np.unique(keys)) != len(keys)
            or np.any(keys < 0) or np.any(keys >= 6 * N ** 3)):
        raise ValueError("frozen training key/count arrays are malformed")

    exposure1024 = _read_training_exposure_by_key(
        output_dir / "selection_nside1024.h5", keys, "selection_shells")
    exposure2048 = _read_training_exposure_by_key(
        RAY2048_H5, keys, "selection_shells")
    old_exposure = _read_training_exposure_by_key(
        OLD_SELECTION, keys, "selection_shells")
    _, ijk, rmin, rmax, crossed_edges, active = _training_cell_geometry(keys)
    hits1024 = _read_training_scalar_by_flat_key(
        output_dir / "selection_nside1024.h5", keys, "ray_hit_count_nside1024")
    hits2048 = _read_training_scalar_by_flat_key(
        RAY2048_H5, keys, "ray_hit_count_nside2048")
    zero_ray_rows = hits2048 == 0
    zero_rows = np.flatnonzero((exposure1024 == 0.) | (exposure2048 == 0.))
    zero_exposure_training_rows = []
    for row in zero_rows:
        p = int(keys[row] // N ** 3)
        zero_exposure_training_rows.append({
            "training_key": int(keys[row]),
            "population": p,
            "count": int(counts[row]),
            "ijk": ijk[row].astype(int).tolist(),
            "rmin_cMpc_h": float(rmin[row]),
            "rmax_cMpc_h": float(rmax[row]),
            "active_positive_volume_domain_intersection": bool(active[row]),
            "radial_edges_crossed_cMpc_h": crossed_edges[row],
            "exposure_order6": float(old_exposure[row]),
            "exposure_nside1024": float(exposure1024[row]),
            "exposure_nside2048": float(exposure2048[row]),
            "ray_hit_count_nside1024": int(hits1024[row]),
            "ray_hit_count_nside2048": int(hits2048[row]),
            "nside1024_zero_category": (
                _classify_zero_exposure(rmin[row], rmax[row], int(hits1024[row]))
                if exposure1024[row] == 0. else None),
            "nside2048_zero_category": (
                _classify_zero_exposure(rmin[row], rmax[row], int(hits2048[row]))
                if exposure2048[row] == 0. else None),
        })

    per_population = summarize_exposure_triplet(
        keys, counts, old_exposure, exposure1024, exposure2048)
    gate_pass = all(row["resolution_gate_pass"] for row in per_population)
    result = {
        "classification": "TRAINING_ONLY_SELECTION_SUPPORT_AND_RESOLUTION_IMPACT",
        "status": "TRAINING_DIAGNOSTIC_ONLY_NOT_CALIBRATED_NOT_FIELD_INFERENCE",
        "source_commit": expected_commit,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "training_keys_only": True,
        "heldout_or_all_key_arrays_read": False,
        "train_keys_sha256": hashlib.sha256(keys.tobytes()).hexdigest(),
        "train_counts_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
        "train_key_count": int(len(keys)),
        "train_galaxy_count": int(counts.sum()),
        "training_key_count_archive": str(count_path),
        "data_free_nside1024_artifact": data_free,
        "old_selection_artifact": {
            "path": str(OLD_SELECTION),
            "sha256": sha256_file(OLD_SELECTION),
            "classification": "order-six numerical exposure, not calibrated",
        },
        "nside2048_artifact": {
            "path": str(RAY2048_H5),
            "sha256": data_free["nside2048_input_sha256"],
        },
        "nside2048_zero_ray_training_keys": int(np.count_nonzero(zero_ray_rows)),
        "nside2048_zero_ray_training_galaxies": int(counts[zero_ray_rows].sum()),
        "zero_exposure_training_rows": zero_exposure_training_rows,
        "per_population": per_population,
        "all_population_resolution_gate_pass": bool(gate_pass),
        "decision_rule": {
            "count_weighted_abs_delta_log_likelihood_nats_lt": DELTA_LOG_LIKELIHOOD_TOL_NATS,
            "max_cell_abs_log_ratio_times_sqrt_count_le": MAX_CELL_SHOT_NOISE_Z,
            "fraction_of_training_counts_with_z_gt_0_1_lt": FRACTION_COUNTS_ABOVE_Z01_TOL,
            "support_mismatch_is_non_negligible": True,
            "zero_exposure_floor_or_smoothing": False,
            "only_common_positive_support_has_a_finite_log_ratio": True,
        },
        "exposure_likelihood_term_only": True,
        "delta_log_likelihood_interpretation":
            "At a spatially uniform delta=0 field, the equal global exposure closure cancels the expected-rate term; this is not a fitted field likelihood and does not assess exposure-density interactions. Passing NSIDE1024-to-2048 is a pairwise proxy, not a direct error bound on NSIDE2048.",
        "nside1024_nside2048_pairwise_proxy_gate_pass": bool(gate_pass),
        "nside2048_absolute_convergence_certified": False,
        "selection_bias_or_survival_calibrated": False,
        "shared_group_covariance_or_count_mark_law_calibrated": False,
        "posterior_or_density_field_created": False,
        "R2_complete": False,
        "Q_GOAL": "Training-only support/resolution impact of the selection denominator for the same CF4-conditioned z=0 R2 field; no density field is inferred.",
        "Q_LEAN": "One full data-free NSIDE1024 comparison followed by one sparse training-only count-weighted check; NSIDE4096 only if a predeclared support/resolution condition fails.",
        "MW_M31_M33": "MW/M31 roles remain ambiguous and M33 unresolved. Their observables must later constrain those same roles on the same NEW evolved LG field at <=0.3 cMpc/h; native truth IDs remain calibration/evaluation-only.",
        "heldout_remains_untouched_by_this_run": True,
    }
    (output_dir / "training_impact.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "status": result["status"],
        "train_key_count": result["train_key_count"],
        "train_galaxy_count": result["train_galaxy_count"],
        "nside2048_zero_ray_training_keys": result["nside2048_zero_ray_training_keys"],
        "all_population_resolution_gate_pass": result["all_population_resolution_gate_pass"],
        "result": str(output_dir / "training_impact.json"),
    }, indent=2), flush=True)
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("full NSIDE1024 operator must run under Slurm")
    expected_commit = os.environ.get("CF4_EXPECTED_COMMIT")
    out_value = os.environ.get("CF4_R2_OUT_DIR")
    if not expected_commit or not out_value:
        raise RuntimeError("CF4_EXPECTED_COMMIT and CF4_R2_OUT_DIR are required")
    actual_commit = __import__("subprocess").run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    if actual_commit != expected_commit:
        raise RuntimeError(f"source commit mismatch: {expected_commit} != {actual_commit}")
    output_dir = Path(out_value)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    started = time.monotonic()
    physics = ray.load_physics_inputs()
    previous_contract = ray._validate_previous_exposure()
    with h5py.File(RAY2048_H5, "r") as handle:
        status = handle.attrs.get("status", "")
        if isinstance(status, bytes):
            status = status.decode("utf-8")
        if status != "COMPLETE_NUMERICAL_SELECTION_ARTIFACT_NOT_LIKELIHOOD_READY":
            raise ValueError(f"NSIDE2048 source artifact is not complete: {status}")
        if handle["selection_shells"].shape != (6, 6, N, N, N):
            raise ValueError("NSIDE2048 source artifact shape differs")
    nside2048_hash = sha256_file(RAY2048_H5)
    nside2048_result = json.loads(RAY2048_RESULT.read_text())
    if nside2048_result.get("status") != "COMPLETE_NUMERICAL_SELECTION_ARTIFACT_NOT_LIKELIHOOD_READY":
        raise ValueError("NSIDE2048 result JSON is not complete")
    geometry = ray.exact_geometry_census(2048)
    if geometry["counts"]["active_shell_intersection"] != 938_128:
        raise AssertionError("active-cell census changed")
    interior_geometry = _interior_geometry_audit(RAY2048_H5, geometry)

    worker_count = int(os.environ.get("CF4_R2_WORKERS", "15"))
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    if worker_count < 1 or worker_count > cpus - 1:
        raise ValueError("worker count must leave one allocated CPU for the parent")
    output_dir.mkdir(parents=True)
    h5_path = output_dir / "selection_nside1024.h5"
    observed = np.zeros((6, 6), dtype=np.float64)
    geometry_observed = np.zeros(6, dtype=np.float64)
    totals = {"candidate_pixels": 0, "box_hit_rays": 0,
              "shell_hit_rays": 0, "zero_hit_cells": []}
    row_offsets = np.zeros(N + 1, dtype=np.int64)
    for ix in range(N):
        row_offsets[ix + 1] = row_offsets[ix] + len(geometry["active_by_x"][ix])
    active_flat = geometry["flat_indices"]

    with h5py.File(h5_path, "x") as output:
        selection = output.create_dataset(
            "selection_shells", shape=(6, 6, N, N, N), dtype="f8",
            chunks=(1, 1, 1, N, N), compression="gzip", compression_opts=1)
        geom = output.create_dataset(
            "geometry_shells", shape=(6, N, N, N), dtype="f8",
            chunks=(1, 1, N, N), compression="gzip", compression_opts=1)
        hits = output.create_dataset(
            "ray_hit_count_nside1024", shape=(N, N, N), dtype="u4",
            chunks=(1, N, N), compression="gzip", compression_opts=1)
        output.create_dataset("active_flat_cell_index", data=active_flat, dtype="i8")
        output.attrs.update(
            status="INCOMPLETE", source_commit=expected_commit,
            grid_N=N, box_cMpc_h=ray.BOX, dx_cMpc_h=DX,
            angular_quadrature_nside=1024,
            map_nside=512, map_ordering="RING",
            radial_edges_cMpc_h=ray.RADIAL_EDGES,
            map_hashes_json=json.dumps(physics["map_hashes"]),
            selection_status="NUMERICAL_SELECTION_ARTIFACT_NOT_LIKELIHOOD_READY")

        context = mp.get_context("spawn")
        completed = 0
        with context.Pool(processes=worker_count, initializer=_worker_init,
                          initargs=(physics, geometry["active_by_x"])) as pool:
            for ix, e, g, hit, stats in pool.imap_unordered(
                    _integrate_nside1024_x_slab, range(N), chunksize=1):
                selection[:, :, ix, :, :] = e
                geom[:, ix, :, :] = g
                hits[ix, :, :] = hit
                observed += e.sum(axis=(2, 3), dtype=np.float64)
                geometry_observed += g.sum(axis=(1, 2), dtype=np.float64)
                for name in ("candidate_pixels", "box_hit_rays", "shell_hit_rays"):
                    totals[name] += int(stats[name])
                totals["zero_hit_cells"].extend(stats["zero_hit_cells"])
                completed += 1
                if completed % 8 == 0:
                    output.flush()
                    print(json.dumps({
                        "nside1024_slabs_completed": completed,
                        "slabs_total": N,
                        "elapsed_seconds": time.monotonic() - started,
                        "candidate_pixels_completed": totals["candidate_pixels"],
                    }), flush=True)

        output.create_dataset("zero_ray_flat_index_nside1024",
                              data=np.asarray(totals["zero_hit_cells"], dtype=np.int64),
                              dtype="i8")
        output.flush()
        expected_population, expected_geometry = _expected_closure(1024, physics)
        pop_rel = ray._relative_difference(observed, expected_population)
        geom_rel = ray._relative_difference(geometry_observed, expected_geometry)
        closure = {
            "population_shell_max_relative_error": float(np.nanmax(pop_rel)),
            "pure_geometry_shell_max_relative_error": float(np.nanmax(geom_rel)),
            "expected_population_shell": expected_population.tolist(),
            "observed_population_shell": observed.tolist(),
            "expected_pure_geometry_shell": expected_geometry.tolist(),
            "observed_pure_geometry_shell": geometry_observed.tolist(),
        }
        if (closure["population_shell_max_relative_error"] > 1e-10
                or closure["pure_geometry_shell_max_relative_error"] > 1e-10):
            output.attrs["status"] = "FAILED_CLOSURE"
            output.flush()
            raise RuntimeError("data-free NSIDE1024 closure failed")
        output.attrs["closure_json"] = json.dumps(closure, allow_nan=False)
        output.attrs["status"] = "COMPLETE_NUMERICAL_SELECTION_ARTIFACT_NOT_LIKELIHOOD_READY"
        output.flush()

    # Freeze and hash the complete data-free operator before opening counts.
    data_free = {
        "classification": "DATA_FREE_FULL_GRID_NSIDE1024_SELECTION_OPERATOR",
        "status": "COMPLETE_NUMERICAL_SELECTION_ARTIFACT_NOT_LIKELIHOOD_READY",
        "source_commit": expected_commit,
        "parent_job_id": os.environ["SLURM_JOB_ID"],
        "path": str(h5_path),
        "sha256": sha256_file(h5_path),
        "source_nside2048_path": str(RAY2048_H5),
        "source_nside2048_sha256": nside2048_hash,
        "source_nside2048_commit": nside2048_result["source_commit"],
        "previous_order6_contract": previous_contract,
        "grid": {"N": N, "box_cMpc_h": ray.BOX, "dx_cMpc_h": DX,
                 "active_shell_cells": int(len(active_flat))},
        "geometry_census_counts": geometry["counts"],
        "interior_geometry_nside2048_audit": interior_geometry,
        "ray_work_nside1024": totals,
        "closure_nside1024": closure,
        "closure_tolerance_relative": 1e-10,
        "selection_bias_calibrated": False,
        "shared_group_covariance_calibrated": False,
        "posterior_or_density_field_created": False,
        "training_counts_opened": False,
        "heldout_or_all_key_arrays_opened": False,
        "elapsed_data_free_seconds": time.monotonic() - started,
    }
    frozen_path = output_dir / "data_free_result.json"
    frozen_path.write_text(json.dumps(data_free, indent=2, allow_nan=False) + "\n")
    if sha256_file(h5_path) != data_free["sha256"]:
        raise RuntimeError("data-free NSIDE1024 artifact hash changed before training read")
    with h5py.File(h5_path, "r") as handle:
        status = handle.attrs.get("status", "")
        if isinstance(status, bytes):
            status = status.decode("utf-8")
        if status != data_free["status"]:
            raise RuntimeError("data-free artifact is not frozen complete before training read")

    result = _training_impact(data_free, output_dir, expected_commit)
    print(json.dumps({
        "status": "TRAINING_IMPACT_WRITTEN",
        "job_id": os.environ["SLURM_JOB_ID"],
        "nside1024_candidates": totals["candidate_pixels"],
        "nside1024_closure": closure["population_shell_max_relative_error"],
        "training_galaxies": result["train_galaxy_count"],
        "resolution_gate_pass": result["all_population_resolution_gate_pass"],
        "elapsed_seconds": time.monotonic() - started,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
