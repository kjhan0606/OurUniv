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
    """Freeze 12 score-blind cells: 4 map boundaries, 4 radial, 4 smooth."""
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

    boundary_order = np.argsort(-contrast, kind="stable")
    append_best(boundary_order, "map_boundary", 4,
                lambda i: 18. <= centre_radius[i] <= 168.
                and float(np.max(mean_direct[i])) > 0.)
    for target in (6., 30., 90., 178.):
        radial_order = np.argsort(np.abs(centre_radius - target), kind="stable")
        append_best(radial_order, f"radial_{target:g}", 1)
    smooth_order = np.argsort(contrast, kind="stable")
    append_best(smooth_order, "smooth_interior", 4,
                lambda i: 18. <= centre_radius[i] <= 168.
                and float(np.min(mean_direct[i])) > 0.)

    ids = np.asarray([idx for idx, _ in chosen], dtype=np.int64)
    return dict(
        ids=ids,
        ijk=ijk[ids],
        positions=centres[ids],
        radius=centre_radius[ids],
        labels=[label for _, label in chosen],
        parent_angular=base_angular[ids],
        direct_angular=direct[ids],
        direct_contrast=contrast[ids],
        quadrature_offsets=offsets,
    )
