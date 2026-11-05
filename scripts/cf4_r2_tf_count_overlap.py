"""Exact TF-only CF4-group versus eligible 2M++ point-edge ownership."""

import json
import os
from pathlib import Path

import numpy as np

BASE = Path("/gpfs/kjhan/CF4/z0_density")
TF = BASE / "r2_tf_source_link_v1/tf_only_groups.npz"
EDGES = BASE / "r2_point_mark_manifest_v1/edges.npz"
POINTS = BASE / "r2_point_mark_manifest_v1/points.npz"
OUT = BASE / "r2_tf_count_overlap_v2"


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    if OUT.exists():
        raise FileExistsError(OUT)
    with np.load(TF, allow_pickle=False) as saved:
        ids = saved["group_pgc"].astype(np.int64)
        hold = saved["holdout"].astype(bool)
        cz = saved["observed_cz"].astype(float)
    if len(ids) != 8502 or len(np.unique(ids)) != len(ids):
        raise ValueError("TF-only source groups changed")
    with np.load(EDGES, allow_pickle=False) as saved:
        edge_group = saved["group_1pgc"].astype(np.int64)
        edge_point = saved["point_index"].astype(np.int64)
        secure = saved["match_class_code"] == 1
        native = saved["native_group"].astype(bool)
    with np.load(POINTS, allow_pickle=False) as saved:
        point_cz = saved["vcmb_km_s"].astype(float)
    # Only an observed secure crossmatch is an association. Do not infer a
    # missing edge to be a CF4 non-detection or a parent selection denominator.
    in_tf = np.isin(edge_group, ids)
    linked = np.unique(edge_group[in_tf])
    secure_groups = np.unique(edge_group[in_tf & secure])
    unique_points = np.unique(edge_point[in_tf & secure])
    tf_index = {int(group): i for i, group in enumerate(ids)}
    linked_train = sum(not hold[tf_index[int(g)]] for g in linked)
    linked_holdout = len(linked)-linked_train
    secure_train = sum(not hold[tf_index[int(g)]] for g in secure_groups)
    secure_holdout = len(secure_groups)-secure_train
    secure_tf = in_tf & secure
    cz_offset = np.abs(np.asarray([cz[tf_index[int(g)]] for g in edge_group[secure_tf]])
                       - point_cz[edge_point[secure_tf]])
    if not np.isfinite(cz_offset).all():
        raise ValueError("secure matched redshift offset nonfinite")
    report = dict(classification="R2_TF_COUNT_SOURCE_OVERLAP_NOT_JOINT_LIKELIHOOD",
        job_id=os.environ["SLURM_JOB_ID"], TF_only_groups=len(ids),
        training_groups=int((~hold).sum()), heldout_groups=int(hold.sum()),
        point_edges_total=len(edge_group), TF_only_edges=int(in_tf.sum()),
        TF_only_secure_edges=int((in_tf & secure).sum()),
        TF_only_linked_groups=len(linked),
        TF_only_linked_training_groups=linked_train,
        TF_only_linked_heldout_groups=linked_holdout,
        TF_only_secure_linked_groups=len(secure_groups),
        TF_only_secure_training_groups=secure_train,
        TF_only_secure_heldout_groups=secure_holdout,
        TF_only_unique_secure_2Mpp_points=len(unique_points),
        secure_edge_abs_CF4_group_minus_2Mpp_point_cz_km_s={
            "median_p90_p99": [float(x) for x in np.percentile(cz_offset, [50, 90, 99])],
            "within_20_km_s_fraction": float(np.mean(cz_offset <= 20)),
            "within_150_km_s_fraction": float(np.mean(cz_offset <= 150))},
        TF_only_native_edge_fraction=float(np.mean(native[in_tf])) if in_tf.any() else None,
        limitations=["An eligible 2M++ point is counted in the voxel factor; a secure CF4 edge exposes observation dependence.",
                     "Unlinked groups are not proven independent of 2M++ and do not define a selection parent.",
                     "A secure overlap alone does not prove double counting: a normalized conditional mark given the observed point can factor from a voxel count under explicit assumptions.",
                     "This graph cannot determine group inclusion, conditional group-cz law, or velocity covariance."],
        R2_posterior=False)
    OUT.mkdir(parents=True)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
