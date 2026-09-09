"""Field-only density peaks and aperture proxies; NOT a bound-halo finder."""
from itertools import product

import numpy as np
from scipy.ndimage import maximum_filter, label
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


def aperture_offsets(radius_cells):
    offsets = np.array(list(product(range(-radius_cells, radius_cells+1), repeat=3)))
    return offsets[(offsets*offsets).sum(1) <= radius_cells**2]


def identify(field, dx, observer_cell, *, h):
    """No native identity, position, membership, or observation target input.

    Velocities remain BOX-frame peculiar km/s. Distances here are cMpc/h at z=0.
    Observer cell is a supplied spatial condition, not an inferred MW identity.
    """
    n = field.shape[-1]
    if field.shape != (7, n, n, n) or dx <= 0 or h <= 0 or not np.isfinite(field).all():
        raise ValueError('finite cubic seven-moment field and positive dx/h required')
    if np.any(field[0] < 0):
        raise ValueError('negative mass')
    lo, hi = np.asarray(observer_cell, float)
    if lo.shape != (3,) or hi.shape != (3,) or np.any(lo >= hi):
        raise ValueError('explicit observer-cell lower/upper three-vectors required')
    mass = field[0]
    maxima = (mass > 0) & (mass == maximum_filter(mass, size=3, mode='constant', cval=-np.inf))
    groups, count = label(maxima, structure=np.ones((3, 3, 3), bool))
    flat = np.flatnonzero(maxima)
    ids = groups.ravel()[flat]
    first = np.full(count+1, n**3, dtype=np.int64)
    np.minimum.at(first, ids, flat)
    sizes = np.bincount(ids, minlength=count+1)
    cells = np.array(np.unravel_index(first[1:], (n,)*3)).T
    interior = np.all((cells >= 2) & (cells < n-2), axis=1)
    cells = cells[interior]
    positions = (cells+.5)*dx
    tree = cKDTree(positions)
    mw = np.flatnonzero(np.all((positions >= lo) & (positions < hi), axis=1))
    margin = np.sqrt(3)*dx
    pair = [(int(i), int(j)) for i in mw for j in tree.query_ball_point(positions[i], 3*h+margin) if j != i]
    m31 = sorted({j for _, j in pair})
    satellite = [(int(j), int(k)) for j in m31
                 for k in tree.query_ball_point(positions[j], 1.5*h+margin) if k != j]
    selected = sorted(set(mw.tolist()) | {j for _, j in pair} | {k for _, k in satellite})
    selected = np.asarray(selected, dtype=int)
    remap = {int(old): new for new, old in enumerate(selected)}
    edges = np.array([(remap[i], remap[j]) for i, j in pair], dtype=int).reshape(-1, 2)
    thirds = np.array([(remap[j], remap[k]) for j, k in satellite], dtype=int).reshape(-1, 2)
    moments = []
    for radius in (1, 2):
        total = np.zeros((7, len(selected)))
        for offset in aperture_offsets(radius):
            xyz = cells[selected]+offset
            total += field[:, xyz[:, 0], xyz[:, 1], xyz[:, 2]]
        moments.append(total)
    moments = np.asarray(moments)
    velocity = moments[:, 1:4]/moments[:, :1]
    variance = moments[:, 4:7]/moments[:, :1]-velocity**2
    scale = np.maximum(moments[:, 4:7]/moments[:, :1], 1)
    if np.any(variance < -1e-7*scale):
        raise ValueError('nonrealizable aperture moments')
    # Third adjacency is conditional on M31; exclude the selected MW when
    # counting each triplet. No huge materialized triplet collection.
    neighbors = {j: set(thirds[thirds[:, 0] == j, 1].tolist()) for j in np.unique(thirds[:, 0])}
    triples = sum(len(neighbors.get(j, set())-{i}) for i, j in edges)
    return dict(cells=cells[selected], positions=positions[selected], moments=moments,
        mean_velocity=velocity, physical_sigma=np.sqrt(np.maximum(variance, 0)),
        mw_candidates=np.array([remap[int(i)] for i in mw], dtype=int),
        mw_m31_edges=edges, m31_m33_edges=thirds, triplet_count=int(triples),
        all_interior_peak_positions=positions, raw_peak_count=count,
        boundary_excluded_peak_count=int(count-interior.sum()),
        tied_peak_group_count=int(np.sum(sizes[1:] > 1)),
        label='FIELD_ONLY_PEAK_APERTURE_PROXIES_NOT_RESOLVED_HALOS')


def match_for_evaluation(candidate_positions, truth_positions, radius):
    """Truth enters ONLY here, after identify returns a frozen catalogue."""
    truth = np.asarray(truth_positions)
    candidates = np.asarray(candidate_positions).reshape(-1, 3)
    distances = np.linalg.norm(truth[:, None]-candidates[None], axis=-1)
    cost = np.where(distances <= radius, distances, 1e20)
    cost = np.c_[cost, np.full((len(truth), len(truth)), radius*1.001)]
    rows, cols = linear_sum_assignment(cost)
    matched = np.full(len(truth), -1, dtype=int)
    matched[rows[cols < len(candidates)]] = cols[cols < len(candidates)]
    nearest = np.argmin(distances, axis=1) if len(candidates) else np.full(len(truth), -1, dtype=int)
    nearest_distance = distances.min(axis=1).tolist() if len(candidates) else [None]*len(truth)
    return dict(matched=matched.tolist(), nearest=nearest.tolist(), nearest_distance=nearest_distance,
                shared_nearest_pairs=[[i, j] for i in range(len(truth)) for j in range(i+1, len(truth))
                                      if nearest[i] >= 0 and nearest[i] == nearest[j]])
