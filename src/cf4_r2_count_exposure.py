"""Population-specific Poisson exposure masks for the graph-closed R2 split."""

from __future__ import annotations

import numpy as np


def build_population_exposure_masks(
    grid_size,
    heldout_flat_voxels,
    train_window_excluded_keys=(),
    heldout_window_excluded_keys=(),
    *,
    population_count=6,
):
    """Build train/heldout exposure masks in population-major key order.

    The base split is spatial (heldout octant versus training volume), while
    graph closure removes additional observed keys from the exposure assigned
    to their role. The return arrays flatten ``(population, voxel)`` in the
    same order as keys ``population * grid_size**3 + flat_voxel``.

    Buffered keys are never exposed. Training and heldout masks are separate;
    neither is inferred as the complement of the other because the buffer is
    intentionally exposed to neither likelihood.
    """
    n = int(grid_size)
    populations = int(population_count)
    if n < 1 or populations < 1:
        raise ValueError('grid_size and population_count must be positive')
    nvoxel = n**3
    heldout_indices = np.asarray(heldout_flat_voxels)
    if (heldout_indices.ndim != 1
            or (heldout_indices.size
                and not np.issubdtype(heldout_indices.dtype, np.integer))):
        raise ValueError('heldout_flat_voxels must be a 1-D integer index array')
    if heldout_indices.size and (np.any(heldout_indices < 0)
                                 or np.any(heldout_indices >= nvoxel)):
        raise ValueError('heldout voxel index is outside the grid')
    if np.unique(heldout_indices).size != heldout_indices.size:
        raise ValueError('heldout voxel indices contain duplicates')
    heldout = np.zeros(nvoxel, dtype=bool)
    heldout[heldout_indices.astype(np.int64)] = True

    train_mask = np.broadcast_to(~heldout, (populations, nvoxel)).copy()
    heldout_mask = np.broadcast_to(heldout, (populations, nvoxel)).copy()

    def exclude_keys(mask, keys, *, expected_role):
        key_array = np.asarray(keys)
        if (key_array.ndim != 1
                or (key_array.size
                    and not np.issubdtype(key_array.dtype, np.integer))):
            raise ValueError(f'{expected_role} excluded keys must be a 1-D integer array')
        if key_array.size and (np.any(key_array < 0)
                               or np.any(key_array >= populations*nvoxel)):
            raise ValueError(f'{expected_role} excluded key is outside the key geometry')
        if np.unique(key_array).size != key_array.size:
            raise ValueError(f'{expected_role} excluded keys contain duplicates')
        if key_array.size:
            voxel = key_array % nvoxel
            if expected_role == 'train' and np.any(heldout[voxel]):
                raise ValueError('training exclusion lies in the heldout voxel octant')
            if expected_role == 'heldout' and np.any(~heldout[voxel]):
                raise ValueError('heldout exclusion lies outside the heldout voxel octant')
            mask.reshape(-1)[key_array] = False

    exclude_keys(train_mask, train_window_excluded_keys, expected_role='train')
    exclude_keys(heldout_mask, heldout_window_excluded_keys,
                 expected_role='heldout')
    if np.any(train_mask & heldout_mask):
        raise AssertionError('train and heldout exposure masks overlap')
    return train_mask.reshape(-1), heldout_mask.reshape(-1)
