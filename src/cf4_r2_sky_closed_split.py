"""Sky-volume count split with closure of shared CF4/FP group marks.

The point split is geometric and never changed by catalogue memberships.
Groups touching a held-out point cannot enter the training mark likelihood.
"""

import numpy as np


def octant_from_xyz(relative):
    xyz = np.asarray(relative)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("expected (n,3) relative coordinates")
    return ((xyz[:, 0] >= 0).astype(np.int8) * 4
            + (xyz[:, 1] >= 0).astype(np.int8) * 2
            + (xyz[:, 2] >= 0).astype(np.int8))


def close_group_marks(point_heldout, edge_point, edge_cf4,
                      seed_cf4, fp_cf4, fp_source, seed_fp):
    """Return held CF4 IDs and FP labels by bipartite graph closure.

    A CF4 group attached to *any* held-out 2M++ point is withheld. FP
    source groups and CF4 groups then propagate withholding across their
    shared source rows. Training points remain geometrically defined so the
    count intensity uses an exact sky-cell observation window.
    """
    point_heldout = np.asarray(point_heldout, dtype=bool)
    edge_point = np.asarray(edge_point, dtype=np.int64)
    edge_cf4 = np.asarray(edge_cf4, dtype=np.int64)
    fp_cf4 = np.asarray(fp_cf4, dtype=np.int64)
    fp_source = np.asarray(fp_source)
    seed_fp = np.asarray(seed_fp, dtype=bool)
    if (edge_point.shape != edge_cf4.shape or fp_cf4.shape != fp_source.shape
            or fp_cf4.shape != seed_fp.shape or np.any(edge_point < 0)
            or np.any(edge_point >= point_heldout.size)):
        raise ValueError("inconsistent group graph")
    held_cf4 = set(map(int, seed_cf4))
    held_cf4.update(map(int, edge_cf4[point_heldout[edge_point]]))
    held_fp = set(fp_source[seed_fp].tolist())
    while True:
        count = len(held_cf4) + len(held_fp)
        held_fp.update(fp_source[np.isin(fp_cf4, list(held_cf4))].tolist())
        held_cf4.update(map(int, fp_cf4[np.isin(fp_source, list(held_fp))]))
        if len(held_cf4) + len(held_fp) == count:
            return held_cf4, held_fp
