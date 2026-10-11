"""Conditional observed-K likelihood for a flux-limited 2M++ subsample.

This is a diagnostic of an assumed luminosity-function shape, not a galaxy
count likelihood or an independent normalization/bias calibration.  At fixed
observed redshift and apparent-magnitude class, the angular completeness in
the published two-map model cancels from the normalized magnitude density.
"""

import numpy as np
from scipy.special import logsumexp


_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(24)
_LOG_WEIGHTS = np.log(_WEIGHTS)
_LOG10 = np.log(10.)


def conditional_logpdf(magnitude, lower, upper, alpha, mstar):
    """Normalized Schechter log density on each row's finite K interval."""
    magnitude, lower, upper = np.broadcast_arrays(magnitude, lower, upper)
    if (not np.isfinite(magnitude).all() or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(upper <= lower)
            or np.any(magnitude < lower-1e-9)
            or np.any(magnitude > upper+1e-9)):
        raise ValueError('invalid conditional-K row support')
    centre = (lower+upper)/2.
    half = (upper-lower)/2.
    sample = centre[..., None]+half[..., None]*_NODES
    def log_shape(value):
        exponent = .4*_LOG10*(mstar-value)
        return (alpha+1.)*exponent-np.exp(exponent)
    log_normalizer = (np.log(half)
                      +logsumexp(log_shape(sample)+_LOG_WEIGHTS, axis=-1))
    return log_shape(magnitude)-log_normalizer


def conditional_pit(magnitude, lower, upper, alpha, mstar):
    """Conditional CDF at observed K; quadrature is independent of bin width."""
    magnitude, lower, upper = np.broadcast_arrays(magnitude, lower, upper)
    def log_integral(lo, hi):
        half = (hi-lo)/2.
        centre = (lo+hi)/2.
        value = centre[..., None]+half[..., None]*_NODES
        exponent = .4*_LOG10*(mstar-value)
        return np.log(half)+logsumexp(
            (alpha+1.)*exponent-np.exp(exponent)+_LOG_WEIGHTS, axis=-1)
    if np.any(upper <= lower) or np.any(magnitude < lower) or np.any(magnitude > upper):
        raise ValueError('invalid conditional-K CDF support')
    result = np.zeros_like(magnitude, dtype=float)
    interior = magnitude > lower
    result[interior] = np.exp(log_integral(lower[interior], magnitude[interior])
                              -log_integral(lower[interior], upper[interior]))
    return np.clip(result, 0., 1.)


def observed_k_intervals(modulus_h, apparent_class):
    """Observed -25<M_h<-21 intersected with the two apparent-K classes."""
    modulus_h, apparent_class = np.broadcast_arrays(modulus_h, apparent_class)
    if np.any((apparent_class != 0) & (apparent_class != 1)):
        raise ValueError('apparent-K class must be bright=0 or faint=1')
    lower = np.maximum(-25., np.where(apparent_class == 0,
                                      -np.inf, 11.5-modulus_h))
    upper = np.minimum(-21., np.where(apparent_class == 0,
                                      11.5-modulus_h, 12.5-modulus_h))
    return lower, upper
