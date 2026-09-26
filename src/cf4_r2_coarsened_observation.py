"""Coarsened 2M++ count factor for an overlap-aware CF4 joint model.

The data scored here are six-population N128 *cell counts*, not individual
locations. Individual redshifts remain observed covariates for the distinct
conditional CF4 group-mark factor. Any within-cell position law is therefore
left outside this count score; it must be field-independent or explicitly
modelled before calling the complete point/mark process exact.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln


def within_cell_logdensity(point_keys, intensity_at_points, integral_at_points,
                           *, unordered=True):
    """Conditional positions given cell counts, for a supplied intensity law.

    q(x|cell,F)=lambda(x,F)/Lambda_cell(F). The caller must compute Lambda
    by integrating the SAME lambda over the cell. For unordered point sets,
    include n_cell! to cancel the factorial in the count PMF. No support
    floor or field-independence assumption is made here. The formula does
    not calibrate the supplied intensity/selection law.
    """
    keys = np.asarray(point_keys)
    intensity, integral = np.asarray(intensity_at_points), np.asarray(integral_at_points)
    if keys.ndim != 1 or intensity.shape != keys.shape or integral.shape != keys.shape:
        raise ValueError('point arrays must be matching vectors')
    if not np.issubdtype(keys.dtype, np.integer) or np.any(keys < 0):
        raise ValueError('invalid population-cell keys')
    if not np.isfinite(intensity).all() or not np.isfinite(integral).all():
        raise ValueError('nonfinite point intensity or integral')
    if np.any(intensity <= 0) or np.any(integral <= 0):
        raise CountSupportError('conditional position outside positive support')
    _, counts = np.unique(keys,return_counts=True)
    order = np.argsort(keys)
    repeated = keys[order][1:] == keys[order][:-1]
    if np.any(repeated & (integral[order][1:] != integral[order][:-1])):
        raise ValueError('inconsistent cell integral for repeated key')
    value = np.sum(np.log(intensity)-np.log(integral))
    return float(value + (np.sum(gammaln(counts+1)) if unordered else 0.))


class CountSupportError(ValueError):
    """An occupied cell has zero expected count."""


def sparse_poisson_count_logpmf(keys, counts, expected_at_keys, total_expected):
    """Exact binned Poisson log PMF from occupied keys and the full integral.

    ``total_expected`` must include every cell, occupied or not. The caller
    supplies it from a field response times the same integrated selection
    used at occupied cells. It may not be estimated from observed counts.
    """
    key = np.asarray(keys)
    n = np.asarray(counts)
    mu = np.asarray(expected_at_keys, dtype=np.float64)
    if key.ndim != 1 or n.shape != key.shape or mu.shape != key.shape:
        raise ValueError("occupied-key arrays must have equal one-dimensional shape")
    if not np.issubdtype(key.dtype, np.integer) or np.any(key < 0) or \
            len(np.unique(key)) != len(key):
        raise ValueError("keys must be unique nonnegative integers")
    if not np.issubdtype(n.dtype, np.integer) or np.any(n <= 0):
        raise ValueError("occupied counts must be positive integers")
    if not np.all(np.isfinite(mu)) or np.any(mu < 0):
        raise ValueError("expected counts must be finite and nonnegative")
    if np.any(mu == 0):
        raise CountSupportError("positive observed count at zero expected count")
    if not np.isfinite(total_expected) or total_expected < np.sum(mu):
        raise ValueError("full expected-count integral is invalid")
    return float(-total_expected + np.sum(n * np.log(mu) - gammaln(n + 1)))
