"""Small source-identity and geometry helpers for the R2 Tempel parent check."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    return np.column_stack((np.cos(dec) * np.cos(ra),
                            np.cos(dec) * np.sin(ra), np.sin(dec)))


def nearest_random_arcsec(tree: cKDTree, xyz: np.ndarray, workers: int = 1) -> np.ndarray:
    chord, _ = tree.query(np.asarray(xyz, dtype=np.float64), k=1, workers=workers)
    return np.rad2deg(2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0))) * 3600.0


def unique_velocity_matches(
    tempel_xyz: np.ndarray,
    tempel_cz_km_s: np.ndarray,
    source_xyz: np.ndarray,
    source_v_km_s: np.ndarray,
    max_sep_arcsec: float = 10.0,
    max_dv_km_s: float = 300.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Return reciprocal one-to-one sky/velocity links; preserve ambiguity counts."""
    tree = cKDTree(np.asarray(tempel_xyz, dtype=np.float64))
    chord = 2.0 * np.sin(np.deg2rad(max_sep_arcsec / 3600.0) / 2.0)
    candidate_pairs: list[tuple[int, int]] = []
    ambiguous_source = 0
    no_candidate = 0
    for source_i, candidates in enumerate(tree.query_ball_point(source_xyz, r=chord)):
        if not candidates:
            no_candidate += 1
            continue
        idx = np.asarray(candidates, dtype=np.int64)
        good = idx[np.abs(tempel_cz_km_s[idx] - source_v_km_s[source_i]) <= max_dv_km_s]
        if len(good) == 1:
            candidate_pairs.append((source_i, int(good[0])))
        elif len(good) > 1:
            ambiguous_source += 1
    if candidate_pairs:
        pairs = np.asarray(candidate_pairs, dtype=np.int64)
        unique_tempel, multiplicity = np.unique(pairs[:, 1], return_counts=True)
        reused = set(map(int, unique_tempel[multiplicity > 1]))
        reciprocal = np.array([int(t) not in reused for t in pairs[:, 1]], dtype=bool)
        kept = pairs[reciprocal]
        ambiguous_tempel = len(reused)
        reused_source = int((~reciprocal).sum())
    else:
        kept = np.empty((0, 2), dtype=np.int64)
        ambiguous_tempel = reused_source = 0
    return kept[:, 0], kept[:, 1], dict(
        candidate_source_rows=len(candidate_pairs), unique_pairs=len(kept),
        no_sky_candidate=no_candidate, ambiguous_source_rows=ambiguous_source,
        ambiguous_tempel_rows=ambiguous_tempel, source_rows_rejected_by_reuse=reused_source,
        max_sep_arcsec=float(max_sep_arcsec), max_dv_km_s=float(max_dv_km_s))


def parent_indices(group_id: np.ndarray, gal_id: np.ndarray, group_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GroupID=0 rows become distinct singleton parents, never one shared group."""
    group_id = np.asarray(group_id, dtype=np.int64)
    gal_id = np.asarray(gal_id, dtype=np.int64)
    singles = np.flatnonzero(group_id == 0)
    parent = np.empty(len(group_id), dtype=np.int64)
    members = group_id > 0
    if np.any(group_id[members] > group_count):
        raise ValueError("Tempel GroupID exceeds declared group table")
    parent[members] = group_id[members] - 1
    parent[singles] = group_count + np.arange(len(singles), dtype=np.int64)
    parent_kind = np.concatenate((np.zeros(group_count, dtype=np.int8),
                                  np.ones(len(singles), dtype=np.int8)))
    parent_id = np.concatenate((np.arange(1, group_count + 1, dtype=np.int64), gal_id[singles]))
    return parent, parent_kind, parent_id


def parent_counts(parent: np.ndarray, n_parent: int, selected: np.ndarray) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.int64)
    selected = np.asarray(selected, dtype=bool)
    if parent.shape != selected.shape or np.any((parent < 0) | (parent >= n_parent)):
        raise ValueError("parent/count input shape or index is invalid")
    return np.bincount(parent[selected], minlength=n_parent).astype(np.int32)
