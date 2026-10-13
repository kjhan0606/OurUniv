"""Population-labelled, cell-modulated point-process observation kernel.

This is a numerical building block, not an accepted 2M++ likelihood.  In
particular the actual catalogue contains a point outside the published map
support, and no calibrated quality/selection-discrepancy branch exists yet.
"""

from __future__ import annotations

import numpy as np


class PointProcessSupportError(ValueError):
    """An observed point has zero intensity under the supplied survey model."""


def log_point_process(unselected_per_cell, exposure_cell, population,
                      flat_cell, point_selection, cell_volume):
    """Log density of labelled points in cMpc/h coordinates.

    ``unselected_per_cell[p,c]`` is the expected *unselected* number in a
    cell. ``exposure_cell[p,c]`` is the cell-average selection fraction and
    ``point_selection[i]`` its value at the observed redshift-space point.
    The intensity per unit volume is u[p,c] s_p(x) / cell_volume; its
    integral is sum(u * exposure).  No observed-count normalization, hidden
    floor, redshift independence, or quality thinning is inserted here.
    """

    u = np.asarray(unselected_per_cell, dtype=np.float64)
    exposure = np.asarray(exposure_cell, dtype=np.float64)
    pop = np.asarray(population)
    cell = np.asarray(flat_cell)
    selection = np.asarray(point_selection, dtype=np.float64)
    if u.ndim != 2 or u.shape != exposure.shape or u.shape[0] != 6:
        raise ValueError("expected matching six-population (P,C) arrays")
    if any(x.ndim != 1 or len(x) != len(pop) for x in (cell, selection)):
        raise ValueError("point arrays have incompatible lengths")
    if not np.issubdtype(pop.dtype, np.integer) or not np.issubdtype(cell.dtype, np.integer):
        raise ValueError("population and cell indices must be integer")
    if np.any(pop < 0) or np.any(pop >= 6) or np.any(cell < 0) or np.any(cell >= u.shape[1]):
        raise ValueError("point index outside the supplied population/cell grid")
    if not np.isfinite(cell_volume) or cell_volume <= 0:
        raise ValueError("cell_volume must be positive and finite")
    if not np.all(np.isfinite(u)) or np.any(u < 0):
        raise ValueError("unselected intensity must be finite and nonnegative")
    if not np.all(np.isfinite(exposure)) or np.any(exposure < 0):
        raise ValueError("exposure must be finite and nonnegative")
    if not np.all(np.isfinite(selection)) or np.any(selection < 0):
        raise ValueError("point selection must be finite and nonnegative")
    if np.any((selection == 0) | (exposure[pop, cell] == 0) | (u[pop, cell] == 0)):
        raise PointProcessSupportError("observed point outside positive intensity support")
    return float(-np.sum(u * exposure, dtype=np.float64)
                 + np.sum(np.log(u[pop, cell]) + np.log(selection)
                          - np.log(cell_volume), dtype=np.float64))
