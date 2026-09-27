"""Exact LF selection transfer for the observed 2M++ K-bin convention.

This is a source-model component, not a calibrated galaxy likelihood.  It
keeps the apparent magnitude at true distance and computes the catalogue's
absolute magnitude from observed-redshift distance after RSD.  In particular,
it does not reuse an observed-space voxel exposure as a source weight.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammainc


def _schechter_interval_probability(lower, upper, *, mstar, alpha):
    """Integral of a normalized Schechter LF over lower<M_h<upper."""
    shape = alpha + 1.0
    if shape <= 0:
        raise ValueError("Schechter alpha must exceed -1 for an unbounded LF")
    lower, upper = np.broadcast_arrays(np.asarray(lower, dtype=float),
                                        np.asarray(upper, dtype=float))
    x_lower = np.power(10.0, 0.4 * (mstar - lower))
    x_upper = np.power(10.0, 0.4 * (mstar - upper))
    return np.where(upper > lower,
                    np.maximum(0.0, gammainc(shape, x_lower)
                               - gammainc(shape, x_upper)), 0.0)


def observed_magnitude_transfer(
    true_modulus_h, observed_modulus_h, *,
    true_edges=(-np.inf, -25.0, -23.6666666666667,
                -22.3333333333333, -21.0, np.inf),
    observed_edges=(-25.0, -23.6666666666667,
                    -22.3333333333333, -21.0),
    mstar=-23.28, alpha=-0.94,
):
    """Return ``P(observed K population | intrinsic K bin, r, s)``.

    Output shape is ``(6, n_true_bins, *broadcast_modulus_shape)``.  Apparent
    populations are K<=11.5 and 11.5<K<=12.5; each has three observed
    absolute-K bins.  Intrinsic edges include bright/faint tails by default,
    because objects outside -25<M_h<-21 can migrate into that observed range.
    The survey angular mask, redshift-space radial cut, row survival and
    population-dependent tracer bias are deliberately NOT part of this
    conditional transfer.  A calibrated intrinsic LF/rate and bias law are
    required before this may replace the existing development operator.
    """
    mu_r, mu_s = np.broadcast_arrays(np.asarray(true_modulus_h, dtype=float),
                                     np.asarray(observed_modulus_h, dtype=float))
    if not np.all(np.isfinite(mu_r)) or not np.all(np.isfinite(mu_s)):
        raise ValueError("finite true and observed distance moduli required")
    true_edges = np.asarray(true_edges, dtype=float)
    observed_edges = np.asarray(observed_edges, dtype=float)
    if (true_edges.ndim != 1 or observed_edges.shape != (4,)
            or not np.all(np.diff(true_edges) > 0)
            or not np.all(np.diff(observed_edges) > 0)):
        raise ValueError("strictly increasing intrinsic/observed edges required")
    result = np.empty((6, len(true_edges)-1) + mu_r.shape, dtype=float)
    shift = mu_s - mu_r
    for j in range(len(true_edges)-1):
        lo_true, hi_true = true_edges[j:j+2]
        normalizer = _schechter_interval_probability(
            lo_true, hi_true, mstar=mstar, alpha=alpha)
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("intrinsic luminosity bin has zero LF measure")
        for a, (m_lo, m_hi) in enumerate(((-np.inf, 11.5), (11.5, 12.5))):
            apparent_lo = m_lo - mu_r
            apparent_hi = m_hi - mu_r
            for b in range(3):
                lo = np.maximum.reduce(np.broadcast_arrays(
                    lo_true, apparent_lo, observed_edges[b] + shift))
                hi = np.minimum.reduce(np.broadcast_arrays(
                    hi_true, apparent_hi, observed_edges[b+1] + shift))
                result[3*a+b, j] = _schechter_interval_probability(
                    lo, hi, mstar=mstar, alpha=alpha) / normalizer
    return result
