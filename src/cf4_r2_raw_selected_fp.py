"""Selected raw optical/K mark component, NOT a calibrated field likelihood.

CPU reference implementation. The caller owns the count-conditioned source
weights, common K proxy convention, shared population parameters and incidence.
No published eta PDF, source Sn/fn, independent fitted prior, or extra count.
"""
from functools import lru_cache

import numpy as np
from scipy.special import log_ndtr, logsumexp, ndtri_exp


@lru_cache(maxsize=8)
def rule(order):
    if order < 8:
        raise ValueError('quadrature order must be >=8')
    x, w = np.polynomial.legendre.leggauss(order)
    return (x+1)/2, w/2


def log_cdf_interval(lower, upper):
    """Stable log[Phi(upper)-Phi(lower)], including the positive tail."""
    lo, hi = np.broadcast_arrays(lower, upper)
    reflect = lo > 0
    a, b = np.where(reflect, -hi, lo), np.where(reflect, -lo, hi)
    lb, la = log_ndtr(b), log_ndtr(a)
    with np.errstate(divide='ignore', invalid='ignore'):
        value = lb + np.log(-np.expm1(la-lb))
    return np.where(hi > lo, value, -np.inf)


def log_box_probability(mean, covariance, lower, upper, *, order=96):
    """Two correlated Gaussian cuts via deterministic conditional quadrature.

    Integrate in the first marginal's CDF coordinate, not independent Phi
    products. The caller must check quadrature accuracy in its parameter range.
    Leading dimensions broadcast; final dimensions are2 and(2,2).
    """
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if mean.shape[-1:] != (2,) or covariance.shape[-2:] != (2, 2):
        raise ValueError('two-dimensional Gaussian geometry required')
    sd = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
    rho = covariance[..., 0, 1]/(sd[..., 0]*sd[..., 1])
    if (not np.isfinite(sd).all() or np.any(sd <= 0)
            or not np.isfinite(rho).all() or np.any(np.abs(rho) >= 1)):
        raise ValueError('positive-definite cut covariance required')
    lo, hi = (lower-mean)/sd, (upper-mean)/sd
    if np.any(hi <= lo):
        raise ValueError('nonempty finite cut intervals required')
    flip = lo[..., 0] > 0
    a = np.where(flip, -hi[..., 0], lo[..., 0])
    b = np.where(flip, -lo[..., 0], hi[..., 0])
    correlation = np.where(flip, -rho, rho)
    mass = log_cdf_interval(a, b)
    t, w = rule(order)
    log_u = np.logaddexp(log_ndtr(a)[..., None], mass[..., None]+np.log(t))
    z = ndtri_exp(log_u)
    conditional_sd = np.sqrt(1-correlation**2)[..., None]
    shift = correlation[..., None]*z
    log_p = log_cdf_interval((lo[..., 1, None]-shift)/conditional_sd,
                             (hi[..., 1, None]-shift)/conditional_sd)
    return mass+logsumexp(np.log(w)+log_p, axis=-1)


def optical_error_covariance(er, es, ei):
    """Howlett eq17 optical prescription, without its redshift-fit FoG term.

    Perfect r/i anticorrelation is a source modelling assumption, not a
    measurement of all true error correlations. Intrinsic Sigma makes the
    total covariance positive definite. K errors are NOT declared zero here.
    """
    er, es, ei = np.broadcast_arrays(er, es, ei)
    if not np.isfinite([er, es, ei]).all() or np.any(np.array([er, es, ei]) <= 0):
        raise ValueError('positive finite reported errors required')
    cov = np.zeros(er.shape+(3, 3))
    cov[..., 0, 0], cov[..., 1, 1], cov[..., 2, 2] = er**2, es**2, ei**2
    cov[..., 0, 2] = cov[..., 2, 0] = -er*ei
    return cov


def optical_cut_geometry(x, magnitude_r, measured_sigma):
    """Exact affine r-band/raw-sigma cuts at fixed redshift/aperture metadata.

    x=(r_z,s,i). Source s is aperture-corrected; measured_sigma is not.
    Other morphology/quality/group selection is NOT represented by these cuts.
    """
    x = np.asarray(x)
    matrix = np.array([[2., 0., 1.], [.04, 1., 0.]])
    magnitude_constant = np.asarray(magnitude_r)+2.5*(2*x[..., 0]+x[..., 2])
    aperture_constant = np.log10(measured_sigma)-x[..., 1]-.04*x[..., 0]
    lower = np.stack(((magnitude_constant-17.)/2.5,
                       np.log10(70.)-aperture_constant), axis=-1)
    upper = np.stack(((magnitude_constant-10.)/2.5,
                       np.log10(420.)-aperture_constant), axis=-1)
    return matrix, lower, upper


def log_schechter(magnitude, *, mstar=-23.28, alpha=-.94):
    """Unnormalised proxy-magnitude density per magnitude, same count LF."""
    t = .4*np.log(10)*(mstar-np.asarray(magnitude))
    return (alpha+1)*t-np.exp(t)


def selected_mark_logpdf(x, observed_magnitude_by_candidate, log_weight, eta,
                         magnitude_lower, magnitude_upper, mu, beta, covariance,
                         cut_matrix, cut_lower, cut_upper, *, m0=-23.,
                         mstar=-23.28, alpha=-.94, magnitude_order=24, cut_order=96):
    """Joint optical/within-bin-K density conditional on both selections.

    Candidate weights ALREADY include count K-bin/flux selection. Each finite
    magnitude interval is the SAME intersection used by that count transfer.
    M_observed,j = observed_K - candidate_shift_j (unit Jacobian). No extra
    count factor. Optical covariance is intrinsic plus one row's error matrix.
    x can be (...,3), and M_observed (...,candidates) for integration checks.
    Scalar population parameters are NOT learned or calibrated by this function.
    """
    x, observed_magnitude_by_candidate = map(np.asarray, (x, observed_magnitude_by_candidate))
    log_weight, eta, lo, hi = map(np.asarray, (log_weight, eta, magnitude_lower, magnitude_upper))
    mu, beta, cov = map(np.asarray, (mu, beta, covariance))
    matrix, lower, upper = map(np.asarray, (cut_matrix, cut_lower, cut_upper))
    if (log_weight.ndim != 1 or any(a.shape != log_weight.shape for a in (eta, lo, hi))
            or observed_magnitude_by_candidate.shape[-1] != len(eta)
            or mu.shape != (3,) or beta.shape != (3,) or cov.shape != (3, 3)
            or matrix.shape != (2, 3) or lower.shape != (2,) or upper.shape != (2,)):
        raise ValueError('one-row, aligned finite candidate geometry required')
    if (not np.isfinite(np.r_[lo, hi, eta]).all() or np.any(hi <= lo)
            or np.isnan(log_weight).any() or np.isposinf(log_weight).any()
            or not np.isfinite(logsumexp(log_weight))):
        raise ValueError('invalid source weights or finite magnitude intersections')
    chol = np.linalg.cholesky(cov)
    inv = np.linalg.solve(chol.T, np.linalg.solve(chol, np.eye(3)))
    normalizer = 3*np.log(2*np.pi)+2*np.log(np.diag(chol)).sum()
    t, w = rule(magnitude_order)
    mag = lo[:, None]+(hi-lo)[:, None]*t
    log_lf_q = log_schechter(mag, mstar=mstar, alpha=alpha)+np.log(w)
    log_integral_lf = np.log(hi-lo)+logsumexp(log_lf_q, axis=-1)
    log_mag_probability = log_lf_q-logsumexp(log_lf_q, axis=-1)[:, None]
    mean = mu+beta*(mag[..., None]-m0)
    mean[..., 0] += eta[:, None]
    cut_cov = matrix@cov@matrix.T
    log_z = log_box_probability(mean@matrix.T, cut_cov, lower, upper, order=cut_order)
    log_selected = logsumexp(log_mag_probability+log_z, axis=-1)
    denominator = logsumexp(log_weight+log_selected)
    means = mu+beta*(observed_magnitude_by_candidate[..., None]-m0)
    means[..., 0] += eta
    delta = x[..., None, :]-means
    log_g = -.5*(np.einsum('...i,ij,...j->...', delta, inv, delta)+normalizer)
    support = ((observed_magnitude_by_candidate >= lo)
               & (observed_magnitude_by_candidate <= hi))
    log_f = (log_schechter(observed_magnitude_by_candidate, mstar=mstar, alpha=alpha)
             -log_integral_lf+log_g)
    value = logsumexp(log_weight+np.where(support, log_f, -np.inf), axis=-1)-denominator
    projected = x@matrix.T
    inside = ((projected >= lower) & (projected <= upper)).all(axis=-1)
    return np.where(inside, value, -np.inf)
