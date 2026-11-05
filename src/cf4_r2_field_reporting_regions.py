"""Training-only reporting masks for R2 native-node field maps."""
from __future__ import annotations

import numpy as np


def training_count_cell_region(train_keys, *, count_grid_size,
                               field_grid_size, population_count=6):
    """Lift observed training count voxels onto an aligned native-node map.

    ``train_keys`` use population-major indexing into the frozen observed
    count grid. The returned mask is the union over populations, mapped by
    coarse-cell interval rather than interpolation: with an aligned 128 to
    256 grid, count cell ``j`` owns native PM node indices ``2*j`` and
    ``2*j+1`` on each axis. This region marks data-bearing coarse cells; it
    does not claim to contain every source cell that can affect the likelihood
    through radial displacement or the volume kernel. Held-out/buffer cells
    are absent when the caller supplies only the frozen training keys.
    """
    def dimension(name, value):
        if (not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))):
            raise ValueError(f"{name} must be an integer")
        return int(value)

    n_count = dimension("count_grid_size", count_grid_size)
    n_field = dimension("field_grid_size", field_grid_size)
    n_population = dimension("population_count", population_count)
    if min(n_count, n_field, n_population) < 1:
        raise ValueError("positive count, field, and population dimensions required")
    if n_field % n_count:
        raise ValueError("field grid must be an integer refinement of the count grid")
    factor = n_field // n_count
    keys = np.asarray(train_keys)
    nvoxel = n_count ** 3
    if (keys.ndim != 1 or (keys.size and not np.issubdtype(keys.dtype, np.integer))
            or np.any(keys < 0) or np.any(keys >= n_population * nvoxel)):
        raise ValueError("training keys must be valid population-major count-grid indices")
    if np.unique(keys).size != keys.size:
        raise ValueError("training count keys must be unique")

    coarse = np.zeros(nvoxel, dtype=bool)
    if keys.size:
        coarse[keys % nvoxel] = True
    coarse = coarse.reshape((n_count,) * 3)
    lifted = np.repeat(coarse, factor, axis=0)
    lifted = np.repeat(lifted, factor, axis=1)
    lifted = np.repeat(lifted, factor, axis=2)
    if lifted.shape != (n_field,) * 3:
        raise AssertionError("reporting-region lift changed field geometry")
    return lifted
