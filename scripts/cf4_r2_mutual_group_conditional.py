"""Source-bound mutual-group velocity and selection-denominator check.

The 2M++ Lavaux--Hudson GID is not a CF4/Tully group identity. A mutual
one-to-one observed bridge is a restricted diagnostic subset, not a parent
population for CF4 distance-indicator inclusion. No field likelihood is fit.
"""

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/gpfs/kjhan/CF4/z0_density/r2_mutual_group_conditional_v2")
POINTS = Path("/gpfs/kjhan/CF4/z0_density/r2_point_mark_manifest_v1/points.npz")
NATIVE = Path("/gpfs/kjhan/CF4/z0_density/bundle_c_v1/native_data_v2/CF4_native.npz")
HASHES = {
    "cf4_groups.csv": "bfdc0cfc0f172b48468e3a8fd05e87978c1ec68c341fb2d929fc1200f0123334",
    "2mpp_catalog.csv": "05d39f49af58caa7aa199420cc7354b3aa9fe3dbacf8d6c33222479c288fb23d",
    "2mpp_groups.csv": "e83bcad0ebe97f6048f36b3235f66babaadf995fc2257302313160f35980057a",
    "cf4_2mpp_crossmatch_v1.csv": "64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf",
}
DF = 4.0
METHODS = ("o_DMcal", "o_DMsnIa", "o_DMfp", "o_DMtf", "o_DMsbfo", "o_DMsbfi")


def read_rows(name):
    path = ROOT / "data" / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
        raise ValueError(f"source hash changed: {name}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def method_pattern(row):
    parts = [name[4:] for name in METHODS if row[name] and int(row[name]) > 0]
    if row["DMsnII"]:
        parts.append("snII")
    return "+".join(parts) if parts else "unlabelled"


def summaries(values):
    arr = np.asarray(values, dtype=np.float64)
    if not arr.size:
        return None
    return dict(n=int(arr.size), median=float(np.median(arr)),
                p90=float(np.percentile(arr, 90)),
                p99=float(np.percentile(arr, 99)))


def fit_group_velocity(residual, sigma, rich):
    """One prespecified t4 conditional candidate, train only.

    The group velocity dispersion / sqrt(richness) is a *trial* scale proxy,
    not a measured CF4--2M++ cross-catalogue error. The fitted floor absorbs
    membership/definition mismatch; it is not source calibration.
    """
    residual = np.asarray(residual, dtype=np.float64)
    known = np.maximum(np.asarray(sigma, dtype=np.float64), 0.0) / np.sqrt(rich)

    def objective(p):
        floor = np.exp(p[1])
        scale = np.sqrt(floor**2 + known**2)
        return -float(np.sum(t.logpdf(residual, DF, loc=p[0], scale=scale)))

    fitted = minimize(objective, [float(np.median(residual)), np.log(80.0)],
                      method="L-BFGS-B", bounds=[(None, None), (np.log(1.0), np.log(1e4))])
    if not fitted.success or not np.isfinite(fitted.fun):
        raise RuntimeError(f"conditional fit failed: {fitted.message}")
    return float(fitted.x[0]), float(np.exp(fitted.x[1]))


def evaluate(residual, sigma, rich, loc, floor):
    residual = np.asarray(residual, dtype=np.float64)
    scale = np.sqrt(floor**2 + (np.maximum(sigma, 0.0) / np.sqrt(rich))**2)
    result = dict(n=int(residual.size),
                  mean_log_score=float(np.mean(t.logpdf(residual, DF, loc=loc, scale=scale))),
                  absolute_residual_km_s=summaries(np.abs(residual)))
    for level in (0.68, 0.90, 0.95):
        width = t.ppf((1.0 + level) / 2.0, DF) * scale
        result[f"central_{int(level*100)}_coverage"] = float(np.mean(np.abs(residual-loc) <= width))
    return result


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    if OUT.exists():
        raise FileExistsError(OUT)
    data = {name: read_rows(name) for name in HASHES}
    cf4 = {int(r["1PGC"]): r for r in data["cf4_groups.csv"]}
    galaxy = {int(r["recno"]): r for r in data["2mpp_catalog.csv"]}
    groups = {int(r["GID"]): r for r in data["2mpp_groups.csv"]}
    with np.load(POINTS, allow_pickle=False) as saved:
        eligible = set(map(int, saved["recno"]))
    with np.load(NATIVE, allow_pickle=False) as saved:
        split = {int(g): bool(h) for g, h in zip(saved["CF4_pgc"], saved["CF4_holdout"])}
    if len(cf4) != 38053 or len(galaxy) != 72973 or len(groups) != 4002 or len(eligible) != 57238:
        raise ValueError("source cardinality changed")

    eligible_gid = Counter()
    for recno in eligible:
        gid = galaxy[recno]["GID"]
        if gid:
            eligible_gid[int(gid)] += 1
    if not set(eligible_gid).issubset(groups):
        raise ValueError("eligible galaxy has unknown 2M++ GID")

    all_pgc_gids = defaultdict(set)
    secure_edges = 0
    secure_without_gid = 0
    for edge in data["cf4_2mpp_crossmatch_v1.csv"]:
        if edge["match_class"] != "secure_joint_mark":
            continue
        recno = int(edge["twompp_recno"])
        if recno not in eligible:
            continue
        secure_edges += 1
        gid = galaxy[recno]["GID"]
        if not gid:
            secure_without_gid += 1
            continue
        pgc = int(edge["1PGC"])
        if pgc in cf4:
            all_pgc_gids[pgc].add(int(gid))
    pgc_gids = {pgc: gids for pgc, gids in all_pgc_gids.items() if pgc in split}
    gid_pgcs = defaultdict(set)
    for pgc, gids in all_pgc_gids.items():
        for gid in gids:
            gid_pgcs[gid].add(pgc)
    mutual = [(pgc, next(iter(gids))) for pgc, gids in pgc_gids.items()
              if len(gids) == 1 and len(gid_pgcs[next(iter(gids))]) == 1]
    mutual.sort()

    records = []
    for pgc, gid in mutual:
        c, g = cf4[pgc], groups[gid]
        if not c["Vcmb"] or not g["Vcmb"] or not g["Rich"] or not g["sigma"]:
            continue
        rich = int(g["Rich"])
        sigma = float(g["sigma"])
        if rich <= 0 or sigma < 0:
            continue
        records.append((pgc, gid, float(c["Vcmb"])-float(g["Vcmb"]), sigma,
                        rich, split[pgc], method_pattern(c),
                        float(c["e_DMzp"]) if c["e_DMzp"] else np.nan))
    if len(records) < 100:
        raise ValueError("too few mutual source pairs")
    pgc, gid, residual, sigma, rich, holdout, method, dm_error = map(np.asarray, zip(*records))
    residual = residual.astype(np.float64)
    sigma = sigma.astype(np.float64)
    rich = rich.astype(np.float64)
    holdout = holdout.astype(bool)
    dm_error = dm_error.astype(np.float64)
    if min(np.count_nonzero(~holdout), np.count_nonzero(holdout)) < 100:
        raise ValueError("frozen train/holdout has too few mutual groups")
    loc, floor = fit_group_velocity(residual[~holdout], sigma[~holdout], rich[~holdout])
    tested = evaluate(residual[holdout], sigma[holdout], rich[holdout], loc, floor)
    # Predeclared adequacy screen, not an empirical promotion of the complete
    # joint law. Stratified sky/method checks are deferred if overall fails.
    tolerance90 = 2.0 * np.sqrt(.9*.1 / tested["n"])
    tolerance95 = 2.0 * np.sqrt(.95*.05 / tested["n"])
    velocity_coverage_screen = (abs(tested["central_90_coverage"]-.9) <= tolerance90
                                and abs(tested["central_95_coverage"]-.95) <= tolerance95)
    all_native_methods = Counter(method_pattern(cf4[p]) for p in split if p in cf4)
    mutual_methods = Counter(method.tolist())
    report = dict(
        classification="R2_MUTUAL_GROUP_SOURCE_CONDITIONAL_DIAGNOSTIC_NOT_JOINT_LIKELIHOOD",
        source_sha256=HASHES,
        eligible_2mpp_points=len(eligible), eligible_2mpp_group_ids=len(eligible_gid),
        secure_eligible_edges=secure_edges, secure_eligible_edges_without_2mpp_gid=secure_without_gid,
        native_cf4_groups=len(split), native_cf4_groups_with_mapped_gid=len(pgc_gids),
        native_cf4_groups_with_multiple_gids=sum(len(gids)>1 for gids in pgc_gids.values()),
        mapped_2mpp_gids_linked_to_multiple_any_cf4_groups=sum(len(pgcs)>1 for pgcs in gid_pgcs.values()),
        native_cf4_graph_is_subgraph_of_all_cf4=True,
        mutual_one_to_one_pairs=len(mutual), usable_mutual_pairs=len(records),
        unlinked_eligible_2mpp_group_ids=len(set(eligible_gid)-set(gid_pgcs)),
        native_cf4_method_patterns=dict(all_native_methods.most_common(12)),
        mutual_method_patterns=dict(mutual_methods.most_common(12)),
        mutual_distance_modulus_error_mag=summaries(dm_error[np.isfinite(dm_error)]),
        velocity_candidate=dict(df=DF, location_km_s=loc, fitted_floor_km_s=floor,
            trial_scale="sqrt(floor^2 + (2M++ group sigma/sqrt(Rich))^2)",
            train=evaluate(residual[~holdout], sigma[~holdout], rich[~holdout], loc, floor),
            holdout=tested, heldout_90_tolerance_2se=tolerance90,
            heldout_95_tolerance_2se=tolerance95,
            coverage_screen_pass=bool(velocity_coverage_screen)),
        selection_identified=False,
        selection_reason=("Unlinked GIDs are not known CF4 distance-nonobservations: "
            "the group definitions differ, the crossmatch is an observed edge graph, "
            "and CF4 methods have different selection. No unmeasured CF4 parent denominator exists."),
        limitation=("Even a passing restricted redshift conditional would not calibrate "
            "CF4 distance-indicator inclusion, distance-modulus error/selection, "
            "ambiguous links, or field dependence. The native holdout is not "
            "an independent sky volume. No posterior or IC follows."))
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT / "mutual_pairs.npz", cf4_1pgc=pgc.astype(np.int64),
                        twompp_gid=gid.astype(np.int64), delta_vcmb_km_s=residual,
                        twompp_group_sigma_km_s=sigma, twompp_group_rich=rich,
                        cf4_holdout=holdout, cf4_method_pattern=method,
                        cf4_e_dmzp_mag=dm_error)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
