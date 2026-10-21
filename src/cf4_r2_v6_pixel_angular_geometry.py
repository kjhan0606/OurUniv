"""Score-blind geometry controls for the active v6 angular-map check.

This module uses the HEALPix-enabled Python environment and has no JAX
dependency. Its compact output is consumed by the JAX/GPU diagnostic.
"""
import healpy as hp
import numpy as np

N = 256
BOX = 384.0
DX = BOX / N
OBSERVER = np.full(3, BOX / 2.0)


def source_volume_rule(order=2):
    """Active GL2 source-volume nodes and normalized weights."""
    if order != 2:
        raise ValueError("geometry controls are frozen to GL2")
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(order)
    ijk = np.asarray([(i, j, k) for i in range(order)
                      for j in range(order) for k in range(order)])
    offsets = nodes_1d[ijk] * DX / 2.0
    weights = np.prod(weights_1d[ijk] / 2.0, axis=1)
    return offsets, weights


def flatten_cell_indices(ijk, n=N):
    """Return C-order flattened grid-cell indices for integer ``(i,j,k)``."""
    ijk = np.asarray(ijk)
    if ijk.ndim != 2 or ijk.shape[1] != 3 or not np.issubdtype(ijk.dtype, np.integer):
        raise ValueError("integer cell indices with shape (n,3) required")
    if np.any((ijk < 0) | (ijk >= n)):
        raise ValueError("cell index outside the grid")
    return (ijk[:, 0].astype(np.int64) * n + ijk[:, 1]) * n + ijk[:, 2]


def pixel_completeness(points, maps, rotation, *, observer=OBSERVER, nside=512):
    """Evaluate the pinned RING maps at fixed physical source sightlines."""
    points = np.asarray(points, dtype=np.float64)
    maps = [np.asarray(m, dtype=np.float64) for m in maps]
    rotation = np.asarray(rotation, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("finite physical positions with shape (n,3) required")
    if len(maps) != 2 or any(m.shape != (hp.nside2npix(nside),) for m in maps):
        raise ValueError("two aligned HEALPix completeness maps required")
    if not np.isfinite(rotation).all() or rotation.shape != (3, 3):
        raise ValueError("finite 3x3 sky rotation required")
    relative = points - np.asarray(observer, dtype=np.float64)
    radii = np.linalg.norm(relative, axis=1)
    if np.any(radii == 0.0):
        raise ValueError("observer-coincident source has no angular pixel")
    sky = rotation @ relative.T
    pixels = hp.vec2pix(nside, sky[0], sky[1], sky[2], nest=False)
    return np.stack([m[pixels] for m in maps], axis=0)


def _fibonacci_directions(count=1536):
    i = np.arange(count, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / count
    phi = i * (np.pi * (3.0 - np.sqrt(5.0)))
    r = np.sqrt(1.0 - z * z)
    return np.column_stack((r * np.cos(phi), r * np.sin(phi), z))


def choose_controls(maps, rotation, coarse_angular):
    """Freeze 12 score-blind, radially stratified cells."""
    offsets, _ = source_volume_rule()
    radii = np.asarray([6., 18., 30., 45., 60., 90., 120., 150., 168., 178.])
    points = (OBSERVER[None, None, :] + radii[:, None, None]
              * _fibonacci_directions()[None, :, :]).reshape(-1, 3)
    ijk = np.floor(points / DX).astype(np.int32)
    if np.any((ijk < 0) | (ijk >= N)):
        raise ValueError("geometry controls escaped the periodic box")
    flat = (ijk[:, 0] * N + ijk[:, 1]) * N + ijk[:, 2]
    _, unique = np.unique(flat, return_index=True)
    ijk = ijk[np.sort(unique)]
    centres = (ijk.astype(np.float64) + 0.5) * DX
    node_points = centres[:, None, :] + offsets[None, :, :]
    direct = pixel_completeness(node_points.reshape(-1, 3), maps, rotation)
    direct = direct.reshape(2, len(ijk), len(offsets)).transpose(1, 0, 2)
    contrast = np.max(np.ptp(direct, axis=2), axis=1)
    mean_direct = np.mean(direct, axis=2)
    centre_radius = np.linalg.norm(centres - OBSERVER[None, :], axis=1)
    parent = (ijk // 2).astype(np.int32)
    parent_flat = (parent[:, 0] * 128 + parent[:, 1]) * 128 + parent[:, 2]
    coarse_angular = np.asarray(coarse_angular)
    if coarse_angular.shape != (2, 128**3):
        raise ValueError("frozen N128 two-map angular source geometry required")
    base_angular = coarse_angular[:, parent_flat].T

    chosen = []
    used = set()

    def append_best(order, category, amount, predicate=None):
        for idx in order:
            idx = int(idx)
            if idx in used or (predicate is not None and not predicate(idx)):
                continue
            chosen.append((idx, category))
            used.add(idx)
            if sum(label == category for _, label in chosen) == amount:
                return
        raise RuntimeError(f"not enough distinct geometry controls for {category}")

    strata = ((18., 55.), (55., 95.), (95., 135.), (135., 168.))
    boundary_order = np.argsort(-contrast, kind="stable")
    for low, high in strata:
        append_best(
            boundary_order, f"map_boundary_{low:g}_{high:g}", 1,
            lambda i, lo=low, hi=high: lo <= centre_radius[i] < hi
            and float(np.max(mean_direct[i])) > 0.)
    for target in (6., 30., 90., 178.):
        radial_order = np.argsort(np.abs(centre_radius - target), kind="stable")
        append_best(radial_order, f"radial_{target:g}", 1)
    smooth_order = np.argsort(contrast, kind="stable")
    for low, high in strata:
        append_best(
            smooth_order, f"smooth_interior_{low:g}_{high:g}", 1,
            lambda i, lo=low, hi=high: lo <= centre_radius[i] < hi
            and float(np.min(mean_direct[i])) > 0.)

    candidate_indices = np.asarray([idx for idx, _ in chosen], dtype=np.int64)
    selected_ijk = ijk[candidate_indices]
    return dict(
        candidate_indices=candidate_indices,
        flat_ids=flatten_cell_indices(selected_ijk),
        ijk=selected_ijk,
        positions=centres[candidate_indices],
        radius=centre_radius[candidate_indices],
        labels=[label for _, label in chosen],
        parent_angular=base_angular[candidate_indices],
        direct_angular=direct[candidate_indices],
        direct_contrast=contrast[candidate_indices],
        quadrature_offsets=offsets,
    )
