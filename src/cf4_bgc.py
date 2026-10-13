#!/usr/bin/env python
"""Bias Gaussianization correction (BGc) for grouped Cosmicflows data.

This module implements equations (23) and (36) of Hoffman et al. (2021),
MNRAS 505, 3380.  At fixed redshift the observed distance distribution is
lognormal.  BGc estimates its median from a fixed-count redshift neighbourhood
and maps the logarithmic distance residual to a Gaussian peculiar velocity:

    v_BGc = median(V | z)
            - sigma_V / sigma_lnD * ln[D / median(D | z)].

With the first-order paper prescription ``sigma_V = cz * sigma_lnD`` the
Gaussianization term is simply ``-cz * ln(D / D_med)``.  The paper sets the
extra Gaussian distance scatter sigma_d to zero, so ``d_BGc = D_med``.

The correction is deliberately not extrapolated into the nearby regime where
peculiar velocities compete with the Hubble velocity.  Hoffman et al. retain
the observed distances and velocities there; this module marks those rows so a
caller cannot mistake them for BGc-Gaussianized constraints.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BGcResult:
    distance: np.ndarray
    velocity: np.ndarray
    sigma_velocity: np.ndarray
    distance_median: np.ndarray
    velocity_median: np.ndarray
    corrected: np.ndarray
    neighbour_count: np.ndarray


def _validate_window(window: int, n: int) -> int:
    if window < 3:
        raise ValueError("BGc redshift window must contain at least 3 rows")
    if window % 2 == 0:
        raise ValueError("BGc redshift window must be odd")
    if n < 3:
        raise ValueError("at least 3 correctable rows are required")
    return min(window, n if n % 2 else n - 1)


def fixed_count_running_median(
    redshift_velocity: np.ndarray,
    values: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-count, redshift-ranked medians and neighbour counts.

    Interior rows receive a centred window.  At either edge the same window is
    shifted rather than shrunk, avoiding a radial change in median noise.
    Ties are resolved stably by input order.
    """
    cz = np.asarray(redshift_velocity, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    if cz.ndim != 1 or x.shape != cz.shape:
        raise ValueError("redshift_velocity and values must be matching 1D arrays")
    if not np.all(np.isfinite(cz)) or not np.all(np.isfinite(x)):
        raise ValueError("running-median inputs must be finite")

    n = cz.size
    use_window = _validate_window(int(window), n)
    order = np.argsort(cz, kind="stable")
    xs = x[order]
    half = use_window // 2
    med_sorted = np.empty(n, dtype=np.float64)
    for i in range(n):
        start = min(max(i - half, 0), n - use_window)
        med_sorted[i] = np.median(xs[start : start + use_window])
    med = np.empty_like(med_sorted)
    med[order] = med_sorted
    return med, np.full(n, use_window, dtype=np.int32)


def fixed_count_reference_median(
    reference_redshift_velocity: np.ndarray,
    reference_values: np.ndarray,
    target_redshift_velocity: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate target medians from a separate, fixed-count reference pool.

    Separating targets from references is required for honest held-out tests:
    no held-out distance or velocity may help define its own BGc correction.
    The window is shifted, rather than shortened, at either reference-pool
    boundary, matching :func:`fixed_count_running_median`.
    """
    ref_cz = np.asarray(reference_redshift_velocity, dtype=np.float64)
    ref_x = np.asarray(reference_values, dtype=np.float64)
    target_cz = np.asarray(target_redshift_velocity, dtype=np.float64)
    if ref_cz.ndim != 1 or ref_x.shape != ref_cz.shape:
        raise ValueError("reference redshifts and values must be matching 1D arrays")
    if target_cz.ndim != 1:
        raise ValueError("target redshifts must be a 1D array")
    if not np.all(np.isfinite(ref_cz)) or not np.all(np.isfinite(ref_x)):
        raise ValueError("reference-median inputs must be finite")
    if not np.all(np.isfinite(target_cz)):
        raise ValueError("target redshifts must be finite")

    use_window = _validate_window(int(window), ref_cz.size)
    order = np.argsort(ref_cz, kind="stable")
    sorted_cz = ref_cz[order]
    sorted_x = ref_x[order]
    half = use_window // 2
    insertion = np.searchsorted(sorted_cz, target_cz, side="left")
    med = np.empty(target_cz.size, dtype=np.float64)
    for i, rank in enumerate(insertion):
        start = min(max(int(rank) - half, 0), ref_cz.size - use_window)
        med[i] = np.median(sorted_x[start : start + use_window])
    return med, np.full(target_cz.size, use_window, dtype=np.int32)


def bgc_transform_from_reference(
    target_redshift_velocity: np.ndarray,
    target_observed_distance: np.ndarray,
    target_sigma_ln_distance: np.ndarray,
    reference_redshift_velocity: np.ndarray,
    reference_observed_distance: np.ndarray,
    *,
    h0: float,
    window: int,
    cz_min: float = 1500.0,
    cz_max: float = 18000.0,
) -> BGcResult:
    """Apply BGc to targets using an independent reference catalog.

    All reference rows must already have passed the caller's pool cuts.  This
    function intentionally never inserts target observations into that pool.
    """
    cz = np.asarray(target_redshift_velocity, dtype=np.float64)
    dist = np.asarray(target_observed_distance, dtype=np.float64)
    sigln = np.asarray(target_sigma_ln_distance, dtype=np.float64)
    ref_cz = np.asarray(reference_redshift_velocity, dtype=np.float64)
    ref_dist = np.asarray(reference_observed_distance, dtype=np.float64)
    if cz.ndim != 1 or dist.shape != cz.shape or sigln.shape != cz.shape:
        raise ValueError("BGc target inputs must be matching 1D arrays")
    if ref_cz.ndim != 1 or ref_dist.shape != ref_cz.shape:
        raise ValueError("BGc reference inputs must be matching 1D arrays")
    if h0 <= 0:
        raise ValueError("h0 must be positive")
    if not np.all(np.isfinite(ref_cz)) or not np.all(np.isfinite(ref_dist)):
        raise ValueError("BGc reference inputs must be finite")
    if np.any(ref_dist <= 0):
        raise ValueError("BGc reference distances must be positive")

    finite = np.isfinite(cz) & np.isfinite(dist) & np.isfinite(sigln)
    corrected = finite & (dist > 0) & (sigln > 0) & (cz >= cz_min) & (cz <= cz_max)
    if corrected.sum() < 1:
        raise ValueError("no target rows lie in the BGc correction range")

    raw_velocity = cz - h0 * dist
    ref_velocity = ref_cz - h0 * ref_dist
    dmed = dist.copy()
    vmed = raw_velocity.copy()
    neighbours = np.zeros(cz.size, dtype=np.int32)
    dmed_c, counts = fixed_count_reference_median(
        ref_cz, ref_dist, cz[corrected], window
    )
    vmed_c, _ = fixed_count_reference_median(
        ref_cz, ref_velocity, cz[corrected], window
    )
    dmed[corrected] = dmed_c
    vmed[corrected] = vmed_c
    neighbours[corrected] = counts

    velocity = raw_velocity.copy()
    velocity[corrected] = vmed_c - np.abs(cz[corrected]) * np.log(
        dist[corrected] / dmed_c
    )
    distance = dist.copy()
    distance[corrected] = dmed_c
    sigma_velocity = np.abs(cz) * sigln
    return BGcResult(
        distance=distance,
        velocity=velocity,
        sigma_velocity=sigma_velocity,
        distance_median=dmed,
        velocity_median=vmed,
        corrected=corrected,
        neighbour_count=neighbours,
    )


def bgc_transform(
    redshift_velocity: np.ndarray,
    observed_distance: np.ndarray,
    sigma_ln_distance: np.ndarray,
    *,
    h0: float,
    window: int,
    cz_min: float = 1500.0,
    cz_max: float = 30000.0,
) -> BGcResult:
    """Apply the published first-order BGc transform.

    Parameters use km/s and Mpc.  Rows outside ``[cz_min, cz_max]`` are returned
    with their direct observed distance and velocity, but ``corrected=False``.
    They must not be fed to a Gaussian BGc likelihood without a separate model.
    """
    cz = np.asarray(redshift_velocity, dtype=np.float64)
    dist = np.asarray(observed_distance, dtype=np.float64)
    sigln = np.asarray(sigma_ln_distance, dtype=np.float64)
    if cz.ndim != 1 or dist.shape != cz.shape or sigln.shape != cz.shape:
        raise ValueError("BGc inputs must be matching 1D arrays")
    if h0 <= 0:
        raise ValueError("h0 must be positive")

    finite = np.isfinite(cz) & np.isfinite(dist) & np.isfinite(sigln)
    positive = (dist > 0) & (sigln > 0)
    pool = finite & positive
    corrected = pool & (cz >= cz_min) & (cz <= cz_max)
    if corrected.sum() < 3:
        raise ValueError("fewer than 3 rows lie in the BGc correction range")

    raw_velocity = cz - h0 * dist
    dmed = dist.copy()
    vmed = raw_velocity.copy()
    neighbours = np.zeros(cz.size, dtype=np.int32)
    # The median pool extends beyond the correction interval.  Otherwise the
    # first and last target bins acquire one-sided windows and a false radial
    # monopole.  Targets remain a subset of this pool.
    pmask = np.flatnonzero(pool)
    cmask = np.flatnonzero(corrected)
    dmed_pool, count_pool = fixed_count_running_median(
        cz[pmask], dist[pmask], window
    )
    vmed_pool, _ = fixed_count_running_median(
        cz[pmask], raw_velocity[pmask], window
    )
    pool_position = np.full(cz.size, -1, dtype=np.int64)
    pool_position[pmask] = np.arange(pmask.size)
    cpos = pool_position[cmask]
    dmed_c = dmed_pool[cpos]
    vmed_c = vmed_pool[cpos]
    dmed[cmask] = dmed_c
    vmed[cmask] = vmed_c
    neighbours[cmask] = count_pool[cpos]

    velocity = raw_velocity.copy()
    # Equation (36), with sigma_V / sigma_lnD = |cz| at first order.
    velocity[cmask] = vmed_c - np.abs(cz[cmask]) * np.log(
        dist[cmask] / dmed_c
    )
    distance = dist.copy()
    distance[cmask] = dmed_c  # equation (23) with sigma_d = 0
    sigma_velocity = np.abs(cz) * sigln
    return BGcResult(
        distance=distance,
        velocity=velocity,
        sigma_velocity=sigma_velocity,
        distance_median=dmed,
        velocity_median=vmed,
        corrected=corrected,
        neighbour_count=neighbours,
    )
