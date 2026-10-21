"""Exact native-patch translations/rotations; no new independent universes."""
from itertools import permutations, product
import h5py
import numpy as np
from cf4_resolved_moments import derived


def cube_rotations():
    rotations = []
    for perm in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            matrix = np.eye(3)[list(perm)] * np.array(signs)[:, None]
            if np.linalg.det(matrix) > .5:
                rotations.append((perm, signs))
    return rotations


def rotate_moments(value, perm, signs):
    """Rotate scalar mass, polar momentum and diagonal second moments together."""
    if value.ndim != 4 or value.shape[0] != 7 or len(set(value.shape[1:])) != 1:
        raise ValueError('seven moments on a cubic grid required')
    if (tuple(perm), tuple(signs)) not in cube_rotations():
        raise ValueError('proper cube rotation required')
    def scalar(field):
        field = field.transpose(perm)
        return np.flip(field, axis=tuple(i for i, s in enumerate(signs) if s < 0))
    return np.stack([scalar(value[0])]
        + [signs[i] * scalar(value[1 + perm[i]]) for i in range(3)]
        + [scalar(value[4 + perm[i]]) for i in range(3)])


def rotate_features(features, perm, signs):
    """Same transformation for 8 log masses +24 bulk-velocity components."""
    n = len(features)
    density = features[:, :8].reshape(n, 2, 2, 2)
    velocity = features[:, 8:].reshape(n, 3, 2, 2, 2)
    def scalar(field):
        field = field.transpose((0,) + tuple(p + 1 for p in perm))
        return np.flip(field, axis=tuple(i + 1 for i, s in enumerate(signs) if s < 0))
    return np.concatenate([scalar(density).reshape(n, 8),
        np.stack([signs[i] * scalar(velocity[:, perm[i]]) for i in range(3)], axis=1).reshape(n, 24)], axis=1)


def coarse_features(coarse, origins):
    """Exact summed-area 12-Mpc/h parent integrals in a native24-Mpc/h patch."""
    origins = np.asarray(origins, dtype=np.int64)
    if coarse.shape != (7, 50, 50, 50) or origins.ndim != 2 or origins.shape[1] != 3 or np.any(origins < 0) or np.any(origins + 16 > 50):
        raise ValueError('origins must contain full nonwrapping native patches')
    prefix = np.pad(coarse, ((0, 0), (1, 0), (1, 0), (1, 0)))
    for axis in (1, 2, 3):
        prefix = prefix.cumsum(axis=axis)
    parents = []
    for parent in product((0, 1), repeat=3):
        lo = origins + 8 * np.array(parent)
        integral = np.zeros((len(origins), 7))
        for corner in product((0, 1), repeat=3):
            loc = lo + 8 * np.array(corner)
            integral += (-1)**(3 - sum(corner)) * prefix[:, loc[:, 0], loc[:, 1], loc[:, 2]].T
        parents.append(integral)
    values = np.stack(parents, axis=2)
    mass = values[:, 0]
    if np.any(mass <= 0):
        raise ValueError('nonpositive coarse parent mass')
    return np.concatenate([np.log(mass), (values[:, 1:4] / mass[:, None]).reshape(len(origins), 24)], axis=1)


def source_groups(origins):
    """Coarse spatial grouping; NOT an estimator of independent universes.

    Group by patch centre in 24-Mpc/h native tiles. Adjacent groups can still
    share particles through overlapping patches and long modes.
    """
    tile = (np.asarray(origins) + 8) // 16
    return np.ravel_multi_index(tile.T, (4, 4, 4))


def grouped_support(weights, groups):
    totals = np.bincount(groups, weights=weights)
    return dict(ess=float(1 / np.dot(totals, totals)), max_weight=float(totals.max()))


def read_rotated_patch(path, origin, rotation_index):
    origin = np.asarray(origin, dtype=np.int64)
    if origin.shape != (3,) or np.any(origin < 0) or np.any(origin + 16 > 50):
        raise ValueError('native nonwrapping origin required')
    perm, signs = cube_rotations()[rotation_index]
    with h5py.File(path, 'r') as f:
        if f.attrs['status'] != 'NATIVE_TOTAL_MATTER_NOT_OBSERVED_LOCAL_UNIVERSE':
            raise ValueError('complete native source required')
        x, y, z = origin * 8
        native = f['fine'][:, x:x + 128, y:y + 128, z:z + 128]
        metadata = dict(source_catalogue=f.attrs['source_catalogue'], h=float(f.attrs['h']))
    value = rotate_moments(native, perm, signs)
    rotation = np.eye(3)[list(perm)] * np.array(signs)[:, None]
    metadata.update(source_center_cMpc_h=(1.5 * origin + 12).tolist(),
        source_to_patch_rotation=rotation.tolist(), patch_center_cMpc_h=[12.] * 3,
        position_rule='x_patch=R@(x_native-source_center)+[12,12,12]; v_patch=R@v_native',
        dx_cMpc_h=.1875, amplitude_or_phase_modification=False,
        limits='Native galaxy/halo coordinates MUST use this same transform. No actual MW/M31/M33 assignment or observation conditioning.')
    moments = dict(mass=value[0], momentum=value[1:4], second_moment=value[4:7])
    return dict(metadata=metadata, integrals=moments, fields=derived(moments, .1875))
