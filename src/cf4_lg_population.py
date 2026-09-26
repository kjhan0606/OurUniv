"""Continuous TNG-calibrated LG-mark distribution, NOT a fine matter-field prior.

Coordinates are invariant under a common proper rotation. Probability density
in these coordinates is NOT density in Cartesian positions/velocities: the
Jacobian below is mandatory when conditioning on observed sky directions.
"""
import numpy as np
from scipy.special import logsumexp

FEATURES = ['ln_M200_MW_Msun', 'ln_M200_M31_Msun', 'ln_bound_M33_Msun',
            'ln_r31_kpc', 'ln_r32_kpc', 'atanh_cos_r31_r32'] + [
            f'asinh_v{pair}_{axis}_over100' for pair in ('31', '32') for axis in ('r', 't', 'n')] + [
            'ln_rho_2_4_over_mean', 'ln_rho_4_8_over_mean',
            'flow_2_4_over100', 'flow_4_8_over100', 'ln_sigma_2_4_over100', 'ln_sigma_4_8_over100']


def invariant_kinematics(r31, r32, v31, v32):
    r31, r32, v31, v32 = np.broadcast_arrays(r31, r32, v31, v32)
    a, b = np.linalg.norm(r31, axis=-1), np.linalg.norm(r32, axis=-1)
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError('nonzero separations required')
    e1 = r31 / a[..., None]
    cosine = np.sum(e1 * r32, axis=-1) / b
    if np.any(abs(cosine) >= 1-1e-12):
        raise ValueError('degenerate triangle is outside this coordinate chart')
    e2 = (r32/b[..., None] - cosine[..., None]*e1)/np.sqrt(1-cosine**2)[..., None]
    rotation = np.stack([e1, e2, np.cross(e1, e2)], axis=-2)
    va = np.einsum('...ij,...j->...i', rotation, v31)
    vb = np.einsum('...ij,...j->...i', rotation, v32)
    return np.concatenate([np.log(a)[..., None], np.log(b)[..., None],
        np.arctanh(cosine)[..., None], np.arcsinh(va/100), np.arcsinh(vb/100)], axis=-1)


def sky_observable_log_jacobian(kin, distance_kpc):
    """Jacobian from distance moduli/LOS/PM at fixed sky to invariant density.

    Cartesian volume: r31^3*r32^3*(1-cos^2) dlogr31 dlogr32 datanhcos dSO(3).
    Observed DM/LOS/PM volume at fixed sky: product D^5, up to constants.
    asinh(v/100) contributes product 1/(100*cosh(u)). Fixed h/units contribute
    constants only. Includes the *conditional sky* density, not angular errors.
    """
    kin = np.asarray(kin)
    log_sech2 = 2 * (np.log(2.) - np.logaddexp(kin[..., 2], -kin[..., 2]))
    log_cosh_v = np.logaddexp(kin[..., 3:], -kin[..., 3:]) - np.log(2.)
    return (5*np.log(distance_kpc).sum(axis=-1) - 3*kin[..., :2].sum(axis=-1)
            - log_sech2 - log_cosh_v.sum(axis=-1))


def fit_gaussian(features, weights, shrink=.05):
    """One declared regularized Gaussian in mark coordinates, not Gaussian rho.

    Weighted rows target uniform retained MW observer, then companion choices.
    Shrinkage is fixed before heldout results; no kernel-width support repair.
    """
    features = np.asarray(features, float)
    weights = np.asarray(weights, float)
    if len(features) < 2*features.shape[1] or np.any(weights <= 0) or not np.isfinite(features).all():
        raise ValueError('insufficient finite positive-weight population')
    weights = weights / weights.sum()
    mean = weights @ features
    residual = features - mean
    covariance = (residual*weights[:, None]).T @ residual
    covariance = (1-shrink)*covariance + shrink*np.diag(np.diag(covariance))
    np.linalg.cholesky(covariance)
    return mean, covariance


def gaussian_logpdf(value, mean, covariance):
    L = np.linalg.cholesky(covariance)
    residual = np.linalg.solve(L, (np.asarray(value)-mean).T).T
    return -.5*np.sum(residual**2, axis=-1) - np.log(np.diag(L)).sum() - len(mean)*np.log(2*np.pi)/2


def conditional_gaussian(mean, covariance, known, value):
    known = np.asarray(known)
    other = np.setdiff1d(np.arange(len(mean)), known)
    gain = np.linalg.solve(covariance[np.ix_(known, known)], covariance[np.ix_(known, other)]).T
    mu = mean[other] + (np.asarray(value)-mean[known]) @ gain.T
    cov = covariance[np.ix_(other, other)] - gain @ covariance[np.ix_(known, other)]
    np.linalg.cholesky(cov)
    return other, mu, cov


def summarize_weights(log_weight):
    if not np.isfinite(log_weight).any():
        raise ValueError('zero posterior proposal support')
    weights = np.exp(log_weight-logsumexp(log_weight))
    return weights, float(1/np.sum(weights**2))
