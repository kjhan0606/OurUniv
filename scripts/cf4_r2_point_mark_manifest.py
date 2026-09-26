"""Bind every eligible 2M++ point to its survey marks and CF4 group edges.

Source audit and lossless observation representation only. No likelihood is
evaluated or catalogue changed. Run on a compute node through Slurm.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import healpy as hp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cf4_twompp_disjoint_tracer_pilot_v3 import load_catalog, base
from cf4_twompp_joint_information_budget_pilot_v1 import (
    _cosmology_distance_table, schechter_fraction,
)

OUT = Path("/gpfs/kjhan/CF4/z0_density/r2_point_mark_manifest_v1")
PARENT = Path("/gpfs/kjhan/CF4/z0_density/r2_inclusive_count_diagnostic_v1/row_diagnostics.npz")
NATIVE = Path("/gpfs/kjhan/CF4/z0_density/bundle_c_v1/native_data_v2/CF4_native.npz")
GROUP_HASH = "bfdc0cfc0f172b48468e3a8fd05e87978c1ec68c341fb2d929fc1200f0123334"
MATCH_CLASS = {"unmatched": 0, "secure_joint_mark": 1,
               "coordinate_redshift_conflict": 2,
               "nonreciprocal_collision": 3,
               "extended_review_candidate": 4}


def bound(path, record):
    raw = path.read_bytes()
    if len(raw) != record["bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"source binding changed: {path}")
    return path


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def point_selection(radius, population, map_value, cosmology, absolute_edges):
    dl_h = _cosmology_distance_table(radius, cosmology) * cosmology["h"]
    result = np.empty(len(radius), dtype=np.float64)
    for p in range(6):
        chosen = population == p
        apparent, absolute = divmod(p, 3)
        fraction = schechter_fraction(
            dl_h[chosen], None if apparent == 0 else 11.5,
            11.5 if apparent == 0 else 12.5,
            absolute_edges[absolute], absolute_edges[absolute + 1], -23.28, -.94)
        result[chosen] = map_value[chosen] * fraction
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    if OUT.exists():
        raise FileExistsError(OUT)
    tracer = json.loads((ROOT / "config/cf4_twompp_disjoint_tracer_pilot_program_v1.json").read_text())
    inputs = tracer["inputs"]
    catalogue_path = bound(ROOT / inputs["twompp_catalog"]["path"], inputs["twompp_catalog"])
    crossmatch_path = bound(ROOT / inputs["cf4_twompp_crossmatch"]["path"],
                            inputs["cf4_twompp_crossmatch"])
    bright_path = bound(ROOT / inputs["completeness_11_5"]["path"], inputs["completeness_11_5"])
    faint_path = bound(ROOT / inputs["completeness_12_5"]["path"], inputs["completeness_12_5"])
    group_path = ROOT / "data/cf4_groups.csv"
    if hashlib.sha256(group_path.read_bytes()).hexdigest() != GROUP_HASH:
        raise ValueError("CF4 group source changed")
    old = json.loads((ROOT / "config/cf4_datum_bearing_z0_twompp_datum_builder_program_v2.json").read_text())
    prior_path = bound(Path(old["bindings"]["excluded_recnos"]["path"]),
                       old["bindings"]["excluded_recnos"])
    prior = {int(row["recno"]) for row in csv_rows(prior_path)}
    if len(prior) != 319:
        raise ValueError("prior exception manifest changed")

    source = load_catalog(catalogue_path)
    with np.load(PARENT, allow_pickle=False) as saved:
        rows = {key: saved[key] for key in saved.files}
    recno = rows["recno"]
    sorted_source = np.argsort(source["recno"])
    source_idx = sorted_source[np.searchsorted(source["recno"][sorted_source], recno)]
    np.testing.assert_array_equal(source["recno"][source_idx], recno)
    population = rows["population"].astype(np.int8)
    radius = np.linalg.norm(rows["position_cMpc_h"].astype(np.float64) - 192., axis=1)
    cell = np.floor(rows["position_cMpc_h"] / 3.).astype(np.int32)
    flat_cell = np.ravel_multi_index(cell.T, (128,) * 3).astype(np.int32)
    pixel = hp.ang2pix(512, np.deg2rad(90. - source["DEC"][source_idx]),
                       np.deg2rad(source["RA"][source_idx]), nest=False)
    bright = base.load_completeness_map(bright_path, 512)
    faint = base.load_completeness_map(faint_path, 512)
    map_value = np.where(population >= 3, faint[pixel], bright[pixel])
    catalogue_mark = np.where(population >= 3, source["c12_5"][source_idx],
                              source["c11_5"][source_idx])
    mark_mismatch = ~np.isfinite(catalogue_mark) | (np.abs(catalogue_mark - map_value) > .05)
    point_zero = map_value <= 0
    prior_exception = np.isin(recno, np.fromiter(prior, dtype=np.int64))
    quality = ~(point_zero | mark_mismatch | prior_exception)
    np.testing.assert_array_equal(rows["source_survivor"], ~rows["match"] & quality)
    common = json.loads((ROOT / "config/cf4_r2_common_cosmology_v1.json").read_text())[
        "common_cosmology"]
    cosmology = dict(h=common["h"], H0_km_s_Mpc=common["H0_km_s_Mpc"],
                     Omega_m=common["Omega_m"], Omega_b=common["Omega_b"],
                     Tcmb_K=common["Tcmb_K"])
    selection = point_selection(radius, population, map_value, cosmology,
                                tracer["tracer_design"]["absolute_K_edges"])
    if np.any(~np.isfinite(selection)) or np.any(selection < 0):
        raise ValueError("invalid pointwise selection")
    groups = {int(row["1PGC"]): row for row in csv_rows(group_path)}
    with np.load(NATIVE, allow_pickle=False) as saved:
        native_pgc = saved["CF4_pgc"].astype(np.int64)
        native_holdout = saved["CF4_holdout"].astype(bool)
    if len(native_pgc) != 19313 or len(np.unique(native_pgc)) != len(native_pgc):
        raise ValueError("CF4 native split changed")
    native = {int(g): bool(h) for g, h in zip(native_pgc, native_holdout)}
    row_index = {int(r): i for i, r in enumerate(recno)}
    if len(row_index) != len(recno):
        raise ValueError("eligible recno duplicated")
    edges = []
    all_matched = set()
    for edge in csv_rows(crossmatch_path):
        kind = edge["match_class"]
        if kind == "unmatched":
            continue
        if kind not in MATCH_CLASS:
            raise ValueError(f"unknown match class: {kind}")
        target = int(edge["twompp_recno"])
        all_matched.add(target)
        index = row_index.get(target)
        if index is None:
            continue
        group_id = int(edge["1PGC"])
        edges.append((index, int(edge["cf4_recno"]), group_id, MATCH_CLASS[kind]))
    if len(all_matched) != 17007:
        raise ValueError("frozen crossmatch cardinality changed")
    edge = np.asarray(edges, dtype=np.int64)
    represented = np.unique(edge[:, 0])
    np.testing.assert_array_equal(np.flatnonzero(rows["match"]), represented)
    edge_groups = edge[:, 2]
    native_status = np.array([g in native for g in edge_groups], dtype=bool)
    native_test = np.array([native.get(int(g), False) for g in edge_groups], dtype=bool)
    canonical_status = np.array([int(g) in groups for g in edge_groups], dtype=bool)
    def group_number(group_id, column):
        row = groups.get(int(group_id))
        return float(row[column]) if row is not None and row[column] else np.nan
    group_v = np.array([group_number(g, "Vcmb") for g in edge_groups])
    group_dm = np.array([group_number(g, "DMzp") for g in edge_groups])
    group_dm_error = np.array([group_number(g, "e_DMzp") for g in edge_groups])
    group_ngal = np.array([int(groups[int(g)]["Ngal"]) if int(g) in groups else -1
                           for g in edge_groups])
    key = population.astype(np.int64) * 128**3 + flat_cell
    all_key, all_count = np.unique(key, return_counts=True)
    previous = Path("/gpfs/kjhan/CF4/z0_density/r2_inclusive_count_diagnostic_v1/")
    with np.load(previous / "inclusive_counts_3_sparse.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(all_key, saved["parent_keys"])
        np.testing.assert_array_equal(all_count, saved["parent_counts"])

    report = dict(classification="R2_SOURCE_BOUND_POINT_MARK_MANIFEST_NO_LIKELIHOOD",
                  eligible_points=int(len(recno)), crossmatched_points=int(len(represented)),
                  crossmatch_edges=int(len(edge)), groups_in_edges=int(len(np.unique(edge_groups))),
                  edges_without_canonical_group=int(np.count_nonzero(~canonical_status)),
                  native_cf4_edges=int(native_status.sum()),
                  secure_edges=int(np.count_nonzero(edge[:, 3] == MATCH_CLASS["secure_joint_mark"])),
                  map_zero_points=int(point_zero.sum()),
                  mark_mismatch_points=int(mark_mismatch.sum()),
                  prior_exception_points=int(prior_exception.sum()),
                  quality_failure_points=int((~quality).sum()),
                  positive_voxel_exposure_but_zero_point_selection=int(
                      np.count_nonzero((rows["exposure"] > 0) & (selection == 0))),
                  edge_group_missing_distance=int(np.count_nonzero(~np.isfinite(group_dm))),
                  edge_group_missing_velocity=int(np.count_nonzero(~np.isfinite(group_v))),
                  exact_inclusive_count_projection=True,
                  source_hashes={"2mpp": inputs["twompp_catalog"]["sha256"],
                                 "crossmatch": inputs["cf4_twompp_crossmatch"]["sha256"],
                                 "cf4_groups": GROUP_HASH},
                  scientific_status="Not a likelihood. Full-map point process has zero support for at least one observed point; quality thinning, CF4 group selection, and conditional group marks remain uncalibrated.")
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT / "points.npz", recno=recno, population=population,
                        flat_cell=flat_cell, ra_deg=source["RA"][source_idx],
                        dec_deg=source["DEC"][source_idx],
                        vcmb_km_s=source["Vcmb"][source_idx],
                        ksmag=source["Ksmag"][source_idx],
                        radius_cMpc_h=radius, point_selection=selection,
                        map_completeness=map_value, catalogue_completeness=catalogue_mark,
                        integrated_cell_exposure=rows["exposure"],
                        mark_mismatch=mark_mismatch, map_zero=point_zero,
                        prior_exception=prior_exception, quality=quality,
                        train=rows["train"], holdout=rows["holdout"],
                        calibration=rows["calibration"])
    np.savez_compressed(OUT / "edges.npz", point_index=edge[:, 0].astype(np.int32),
                        cf4_recno=edge[:, 1].astype(np.int32),
                        group_1pgc=edge_groups.astype(np.int64),
                        match_class_code=edge[:, 3].astype(np.int8),
                        canonical_group=canonical_status,
                        native_group=native_status, native_holdout=native_test,
                        group_vcmb_km_s=group_v, group_dmzp=group_dm,
                        group_dmzp_error=group_dm_error, group_ngal=group_ngal)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
