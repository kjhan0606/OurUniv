"""Empirical continuous remainder-field conditional, not a certified LCDM prior.

Five jointly correlated Gaussian-copula channels represent child mass ratios,
three normalized mean-velocity residuals and scalar physical dispersion. A
conservative map conditions on all seven supplied coarse moments. The coarse
field probability law and changing halo profiles are separate requirements.
"""
import numpy as np
from scipy.special import ndtr, ndtri
from cf4_continuous_matter import restrict, check_realizable


def expand(x, ratio=8):
    for axis in (-3, -2, -1):
        x = np.repeat(x, ratio, axis=axis)
    return x


def block_sum(x, ratio=8):
    n = x.shape[-1]
    if n % ratio:
        raise ValueError('integer child/parent ratio required')
    c = n//ratio
    return x.reshape(x.shape[:-3]+(c, ratio, c, ratio, c, ratio)).sum(axis=(-5, -3, -1))


def moments_state(x):
    valid = x[0] > 0
    mean = np.divide(x[1:4], x[0], out=np.zeros_like(x[1:4]), where=valid)
    second = np.divide(x[4:], x[0], out=np.zeros_like(x[4:]), where=valid)
    variance = np.maximum(second-mean**2, 0)
    return mean, variance


def detail_channels(fine, ratio=8):
    check_realizable(fine)
    coarse = restrict(fine, ratio)
    mean, variance = moments_state(fine)
    parent_mean, parent_var = moments_state(coarse)
    parent_sigma = expand(np.sqrt(parent_var.mean(axis=0)), ratio)
    child_mean_mass = expand(coarse[0], ratio)/ratio**3
    result = np.zeros((5,)+fine.shape[1:])
    np.divide(fine[0], child_mean_mass, out=result[0], where=child_mean_mass > 0)
    np.divide(mean-expand(parent_mean, ratio), parent_sigma, out=result[1:4], where=parent_sigma > 0)
    np.divide(np.sqrt(variance.mean(axis=0)), parent_sigma, out=result[4], where=parent_sigma > 0)
    # Empty-cell kinematics are undefined, not measured zero. Their chosen
    # latent placeholder is ignored when generated child mass is zero.
    result[1:, fine[0] == 0] = 0
    return result, coarse


class Marginals:
    def __init__(self, quantiles):
        self.quantiles = np.asarray(quantiles)
        self.p = np.linspace(0, 1, self.quantiles.shape[1])

    def gaussianize(self, channels, rng):
        out = np.empty_like(channels)
        for k, knots in enumerate(self.quantiles):
            values, first, count = np.unique(knots, return_index=True, return_counts=True)
            mid = (first+(count-1)/2)/(len(knots)-1)
            probability = np.interp(channels[k], values, mid)
            # Randomized probability integral transform within resolved atoms
            # (including zero mass/dispersion); no pseudo-mass density floor.
            for j in np.flatnonzero(count > 1):
                mask = channels[k] == values[j]
                probability[mask] = rng.uniform(self.p[first[j]], self.p[first[j]+count[j]-1], size=mask.sum())
            out[k] = ndtri(np.clip(probability, 1e-6, 1-1e-6))
        return out

    def invert(self, latent):
        return np.array([np.interp(ndtr(z), self.p, knots) for z, knots in zip(latent, self.quantiles)])


def conditional_moments(channels, coarse, ratio=8):
    """Exactly preserve parent M,P,Qdiag using nonnegative child mixtures.

    This conditional energy allocation is part of the model, not a posterior
    amplitude correction. Zero mass has zero extensive moments. A supplied
    coarse state must be realizable; all-zero positive-mass parents fail.
    """
    check_realizable(coarse)
    weight = channels[0]
    if np.any(weight < 0) or not np.isfinite(channels).all():
        raise ValueError('finite channels and nonnegative mass ratios required')
    normalization = block_sum(weight, ratio)
    if np.any((normalization <= 0) & (coarse[0] > 0)):
        raise ValueError('zero conditional support in a positive-mass parent')
    mass = np.divide(weight*expand(coarse[0], ratio), expand(normalization, ratio),
        out=np.zeros_like(weight), where=expand(normalization, ratio) > 0)
    mean, variance = moments_state(coarse)
    out = np.empty((7,)+weight.shape)
    out[0] = mass
    sigma_squared = channels[4]**2
    for i in range(3):
        child_mean = np.divide(block_sum(mass*channels[1+i], ratio), coarse[0],
            out=np.zeros_like(coarse[0]), where=coarse[0] > 0)
        residual = channels[1+i]-expand(child_mean, ratio)
        base_var = np.divide(block_sum(mass*(residual**2+sigma_squared), ratio), coarse[0],
            out=np.zeros_like(coarse[0]), where=coarse[0] > 0)
        if np.any((base_var <= 0) & (variance[i] > 1e-8)):
            raise ValueError('zero conditional variance support')
        scale = np.sqrt(np.divide(variance[i], base_var, out=np.zeros_like(base_var), where=base_var > 0))
        scale = expand(scale, ratio)
        velocity = expand(mean[i], ratio)+scale*residual
        out[1+i] = mass*velocity
        out[4+i] = mass*(velocity**2+scale**2*sigma_squared)
    return out


def spectral_draw(factors, diagonal_power, rng, shape, *, shrink=.05):
    """Full cross-channel C(k) without a huge covariance matrix/eigendecompose.

    Z(k)=sqrt((1-lambda)/T) sum F_t(k)*eta_t(k)
         +sqrt(lambda*diag C(k))*xi(k), with independent real-white-noise FFTs.
    Shared eta within each five-vector retains complex cross spectra. Every
    Fourier mode has independent coefficients, NOT a whole-template mixture.
    Hermitian consistency follows from FFTs of real fields. Patch stationarity
    and Gaussian copula discard higher-order phase coupling: test, don't assume.
    """
    out = np.zeros_like(diagonal_power, dtype=np.complex128)
    for factor in factors:
        noise = np.fft.rfftn(rng.normal(size=shape), norm='ortho')
        out += factor*noise*np.sqrt((1-shrink)/len(factors))
    for k in range(5):
        noise = np.fft.rfftn(rng.normal(size=shape), norm='ortho')
        out[k] += np.sqrt(shrink*diagonal_power[k])*noise
    out[:, 0, 0, 0] = 0
    return np.fft.irfftn(out, s=shape, axes=(-3, -2, -1), norm='ortho').real
