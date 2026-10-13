"""Bind secure singleton TF marks to the counted 2M++ observed point."""

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path("/gpfs/kjhan/CF4/z0_density")
TF = BASE / "r2_tf_source_link_v1/tf_only_groups.npz"
EDGES = BASE / "r2_point_mark_manifest_v1/edges.npz"
POINTS = BASE / "r2_point_mark_manifest_v1/points.npz"
GROUPS = ROOT / "data/cf4_groups.csv"
GROUP_SHA = "bfdc0cfc0f172b48468e3a8fd05e87978c1ec68c341fb2d929fc1200f0123334"
OUT = BASE / "r2_tf_matched_point_bridge_v1"


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    if OUT.exists():
        raise FileExistsError(OUT)
    if hashlib.sha256(GROUPS.read_bytes()).hexdigest() != GROUP_SHA:
        raise ValueError("CF4 source groups changed")
    with np.load(TF, allow_pickle=False) as saved:
        tf = {key: saved[key].copy() for key in saved.files}
    if len(tf["group_pgc"]) != 8502:
        raise ValueError("TF-only catalogue changed")
    ngal = {}
    with GROUPS.open(newline="") as stream:
        for row in csv.DictReader(stream):
            ngal[int(row["1PGC"])] = int(row["Ngal"])
    with np.load(EDGES, allow_pickle=False) as saved:
        edge_group = saved["group_1pgc"].astype(np.int64)
        edge_point = saved["point_index"].astype(np.int64)
        edge_class = saved["match_class_code"].astype(np.int8)
    with np.load(POINTS, allow_pickle=False) as saved:
        point_cz = saved["vcmb_km_s"].astype(float)
        point_pop = saved["population"].astype(np.int8)
    groups = tf["group_pgc"].astype(np.int64)
    index = {int(group): i for i, group in enumerate(groups)}
    if len(index) != len(groups):
        raise ValueError("duplicate TF-only group identity")
    source_ngal = np.array([ngal[int(group)] for group in groups], dtype=np.int32)
    all_edges = [[] for _ in groups]
    for group, point, kind in zip(edge_group, edge_point, edge_class, strict=True):
        i = index.get(int(group))
        if i is not None:
            all_edges[i].append((int(point), int(kind)))
    population = np.full(len(groups), -1, dtype=np.int8)
    point_index = np.full(len(groups), -1, dtype=np.int32)
    original_cz = tf["observed_cz"].copy()
    matched = 0
    for i, rows in enumerate(all_edges):
        if source_ngal[i] != 1 or len(rows) != 1 or rows[0][1] != 1:
            continue
        point = rows[0][0]
        if not 0 <= int(point_pop[point]) < 6 or not np.isfinite(point_cz[point]):
            raise ValueError("invalid secure point population/redshift")
        population[i] = point_pop[point]
        point_index[i] = point
        tf["observed_cz"][i] = point_cz[point]
        matched += 1
    chosen = population >= 0
    if matched < 1000 or np.any(point_index[~chosen] != -1):
        raise ValueError("unexpected secure singleton bridge cardinality")
    if len(np.unique(point_index[chosen])) != matched:
        raise ValueError("same counted point assigned to more than one TF-only group")
    offset = np.abs(original_cz[chosen]-tf["observed_cz"][chosen])
    tf.update(point_population=population, secure_point_index=point_index,
              original_group_cz=original_cz, source_Ngal=source_ngal)
    report = dict(classification="R2_TF_SINGLETON_POINT_CONDITIONAL_SOURCE_BRIDGE",
        job_id=os.environ["SLURM_JOB_ID"], TF_only_groups=len(groups),
        TF_matched_secure_singletons=matched,
        matched_training_groups=int(np.count_nonzero(chosen & ~tf["holdout"])),
        matched_heldout_groups=int(np.count_nonzero(chosen & tf["holdout"])),
        unmatched_or_ambiguous_groups=int((~chosen).sum()),
        matched_count_population=np.bincount(population[chosen], minlength=6).tolist(),
        matched_abs_group_minus_point_cz_km_s_p50_p90_p99=[float(x) for x in np.percentile(offset, [50, 90, 99])],
        identity_checks=dict(one_secure_edge_and_Ngal1_only=True, unique_count_point_per_TF_group=True,
                             original_TF_holdout_preserved=True),
        limitations=["Matched groups may share count information; this bridge changes the conditioned cz covariate, not the TF source distance summary.",
                     "Matching does not calibrate TF selection among 2M++ galaxies or the count point-process model.",
                     "Unmatched/ambiguous groups keep their original group cz; no physical association is invented."],
        R2_posterior=False,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (TF, EDGES, POINTS, GROUPS)})
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT / "tf_groups_linked.npz", **tf)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
