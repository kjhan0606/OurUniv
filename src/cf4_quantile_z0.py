"""Training-only Gaussian-copula z=0 prior; neither ICs nor exact dynamics.

The latent Gaussian field has a covariance fitted *after* Gaussianizing the
training log densities. A monotone C1 map restores their one-point shape.
Sampling remains in white coordinates; this forward generative transform
does not require a change-of-variables Jacobian in the white-coordinate prior.
"""
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicHermiteSpline, PchipInterpolator
from scipy.special import ndtr

from cf4_pm_calibrated_z0 import PMCalibratedFieldModel
from cf4_z0_physical_field import unit_mean_density


def fit_quantile_map(densities, knots=65, extent=3.5):
    logs = []
    for rho in densities:
        if not np.isfinite(rho).all() or np.any(rho <= 0):
            raise ValueError("invalid training density")
        g = np.log(rho)
        logs.append((g - g.mean()).ravel())
    x = np.linspace(-extent, extent, knots)
    y = np.quantile(np.concatenate(logs), ndtr(x))
    if np.any(np.diff(y) <= 0):
        raise ValueError("training quantiles must be strictly monotone")
    slopes = PchipInterpolator(x, y).derivative()(x)
    # Positive endpoint secants make the linear tails C1 without saturation.
    slopes[0], slopes[-1] = (y[1] - y[0]) / (x[1] - x[0]), (y[-1] - y[-2]) / (x[-1] - x[-2])
    if np.any(slopes <= 0):
        raise ValueError("nonpositive quantile slope")
    return dict(x=x, y=y, slopes=slopes)


def quantile_numpy(z, mapping):
    x, y, slopes = (mapping[k] for k in ("x", "y", "slopes"))
    interior = CubicHermiteSpline(x, y, slopes)(np.clip(z, x[0], x[-1]))
    return np.where(z < x[0], y[0] + slopes[0] * (z - x[0]),
                    np.where(z > x[-1], y[-1] + slopes[-1] * (z - x[-1]), interior))


def gaussianize(log_density, mapping):
    """Vectorized inverse of the same monotone map, including linear tails."""
    g = np.asarray(log_density)
    x, y, slopes = (mapping[k] for k in ("x", "y", "slopes"))
    lo, hi = np.full_like(g, x[0]), np.full_like(g, x[-1])
    for _ in range(45):
        mid = (lo + hi) / 2
        below = quantile_numpy(mid, mapping) < g
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    result = (lo + hi) / 2
    return np.where(g < y[0], x[0] + (g - y[0]) / slopes[0],
                    np.where(g > y[-1], x[-1] + (g - y[-1]) / slopes[-1], result))


def quantile_jax(z, mapping):
    x, y, m = (jnp.asarray(mapping[k]) for k in ("x", "y", "slopes"))
    bounded = jnp.clip(z, x[0], x[-1])
    i = jnp.clip(jnp.searchsorted(x, bounded, side="right") - 1, 0, len(x) - 2)
    h = x[i + 1] - x[i]
    t = (bounded - x[i]) / h
    value = (2*t**3 - 3*t**2 + 1)*y[i] + (t**3 - 2*t**2 + t)*h*m[i]
    value += (-2*t**3 + 3*t**2)*y[i+1] + (t**3 - t**2)*h*m[i+1]
    # <=/>= also choose the intended endpoint derivative, avoiding clip's
    # half-derivative convention exactly at the two outer knots.
    return jnp.where(z <= x[0], y[0] + m[0]*(z-x[0]),
                     jnp.where(z >= x[-1], y[-1] + m[-1]*(z-x[-1]), value))


class QuantileFieldModel(PMCalibratedFieldModel):
    def __init__(self, source, covariance, mapping):
        self.__dict__.update(vars(source))
        self.covariance = {k: jnp.asarray(v) for k, v in covariance.items()}
        self.mapping = {k: jnp.asarray(v) for k, v in mapping.items()}
        if any(a.shape != (self.n,)*3 or not np.isfinite(a).all() for a in covariance.values()):
            raise ValueError("invalid Gaussianized-coordinate covariance")

    def fields(self, vector):
        z, _, velocity = super().fields(vector)
        log_density = quantile_jax(z, self.mapping)
        return log_density, unit_mean_density(log_density), velocity
