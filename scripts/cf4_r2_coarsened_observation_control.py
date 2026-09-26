"""Actual-source N128 coarsened-count support and ownership control.

Uses all eligible 2M++ cells, including the ARES-map-zero galaxy's cell.
One homogeneous published-rate baseline is evaluated only to demonstrate a
finite normalized count factor. It is not a field fit or joint CF4 posterior.
"""

import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cf4_r2_coarsened_observation import sparse_poisson_count_logpmf

MANIFEST = Path("/gpfs/kjhan/CF4/z0_density/r2_point_mark_manifest_v1")
ARES_BRIDGE = Path("/gpfs/kjhan/CF4/z0_density/r2_ares_catalogue_bridge_v1")
COUNTS = Path("/gpfs/kjhan/CF4/z0_density/r2_inclusive_count_diagnostic_v1/inclusive_counts_3_sparse.npz")
SELECTION = Path("/gpfs/kjhan/CF4/z0_density/r2_common_selection_128_v1/selection_3.h5")
OUT = Path("/gpfs/kjhan/CF4/z0_density/r2_coarsened_observation_v2")
N = 128


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Slurm allocation required")
    if OUT.exists():
        raise FileExistsError(OUT)
    with np.load(MANIFEST / "points.npz", allow_pickle=False) as saved:
        recno = saved["recno"]
        pop = saved["population"].astype(np.int64)
        flat = saved["flat_cell"].astype(np.int64)
        point_zero = saved["map_zero"]
        point_selection = saved["point_selection"]
    with np.load(COUNTS, allow_pickle=False) as saved:
        keys = saved["parent_keys"]
        counts = saved["parent_counts"]
    with np.load(ARES_BRIDGE / "bridge.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(recno, saved["eligible_recno"])
        in_ares_example = saved["eligible_in_example"]
    with np.load(MANIFEST / "edges.npz", allow_pickle=False) as saved:
        edge_point = saved["point_index"]
        native_group = saved["native_group"]
    projected_key, projected_count = np.unique(pop * N**3 + flat, return_counts=True)
    np.testing.assert_array_equal(projected_key, keys)
    np.testing.assert_array_equal(projected_count, counts)
    if int(point_zero.sum()) != 1 or int((point_selection == 0).sum()) != 1:
        raise ValueError("known pointwise support diagnosis changed")
    cfg = json.loads((ROOT / "config/cf4_r2_common_cosmology_v1.json").read_text())
    prior = cfg["published_prior"]
    nbar = np.asarray(prior["original_mean_count_per_cell_bright_first"], dtype=np.float64) * (
        3.0 / prior["original_cell_cMpc_h"]) ** 3
    if nbar.shape != (6,) or np.any(nbar <= 0):
        raise ValueError("invalid published homogeneous count-rate baseline")
    occupied_pop = keys // N**3
    occupied_flat = keys % N**3
    x, y, z = np.unravel_index(occupied_flat, (N,) * 3)
    expected_at_keys = np.full(len(keys), np.nan, dtype=np.float64)
    expected_total = 0.0
    with h5py.File(SELECTION, "r") as handle:
        if handle.attrs["status"] != "INTEGRATION_COMPLETE_NOT_CALIBRATED":
            raise ValueError("selection integration status changed")
        source = handle["selection_shells"]
        if source.shape != (6, 6, N, N, N):
            raise ValueError("selection geometry changed")
        for lower in range(0, N, 4):
            exposure = source[:, :, lower:lower + 4].sum(axis=1, dtype=np.float64)
            expected_total += float(np.sum(nbar[:, None, None, None] * exposure))
            take = np.flatnonzero((x >= lower) & (x < lower + 4))
            expected_at_keys[take] = (nbar[occupied_pop[take]] *
                exposure[occupied_pop[take], x[take] - lower, y[take], z[take]])
    if not np.all(np.isfinite(expected_at_keys)):
        raise ValueError("occupied keys not covered by source selection")
    score = sparse_poisson_count_logpmf(keys, counts, expected_at_keys, expected_total)
    zero_index = int(np.flatnonzero(point_zero)[0])
    zero_key = int(pop[zero_index] * N**3 + flat[zero_index])
    zero_key_index = int(np.searchsorted(keys, zero_key))
    if zero_key_index >= len(keys) or keys[zero_key_index] != zero_key:
        raise ValueError("zero-map point absent from count target")
    report = dict(classification="R2_COARSENED_N128_OBSERVATION_CONTROL_NOT_JOINT_POSTERIOR",
                  observed_points=int(len(recno)), occupied_population_cells=int(len(keys)),
                  pointwise_map_zero=int(point_zero.sum()),
                  zero_point_recno=int(recno[zero_index]),
                  zero_point_in_ares_example=bool(in_ares_example[zero_index]),
                  zero_point_cell_count=int(counts[zero_key_index]),
                  zero_point_cell_expected_homogeneous=float(expected_at_keys[zero_key_index]),
                  occupied_zero_expected_cells=int(np.count_nonzero(expected_at_keys == 0)),
                  all_points_counted_exactly=True,
                  published_homogeneous_nbar_per_cell=nbar.tolist(),
                  published_homogeneous_expected_total=expected_total,
                  published_homogeneous_log_count_pmf=score,
                  association_edges=int(len(edge_point)),
                  native_cf4_association_edges=int(native_group.sum()),
                  required_full_joint_factorization=(
                      "p(C,U,A,G|F,S)=p(C|F,S) p(U|C,F,S) "
                      "p(A|C,U,F,S) p(G|A,C,U,F,S)"),
                  evaluated_factor="p(C|F,S) only, at one homogeneous published-rate baseline",
                  unresolved_factors=["within-cell point-location law",
                      "observed CF4/2M++ association and group-selection law",
                      "conditional CF4 group distance/velocity mark law"],
                  limitation="An F-independent positive within-cell location law is only a proposed coarsening approximation, not source-calibrated. No p(A|C,U,F,S) or p(G|A,C,U,F,S) is implemented. The finite homogeneous score is not a joint likelihood, field fit, posterior, or evidence for 0.3-cMpc/h information.")
    OUT.mkdir(parents=True)
    (OUT / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
