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
