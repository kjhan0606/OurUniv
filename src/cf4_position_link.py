"""Position measurement density conditional on archive eligibility, NOT p(F|data)."""
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
import torch
import torch.nn.functional as functional

from cf4_population_locations import DX, CORE, HALO, selected

KAPPA = math.log(10)/5


def root_nonnegative(value):
    positive = value > 0
    return torch.where(positive, torch.sqrt(torch.where(positive, value, torch.ones_like(value))), 0.)


def features_torch(moments):
    """Exact four-cell halo support; double extensive moments and safe cold gradients."""
    if moments.dtype != torch.float64 or moments.shape[0] != 7:
        raise ValueError('seven FP64 native extensive moments required')
    mass = moments[0]
    if not bool(torch.isfinite(moments).all()) or bool((mass < 0).any()):
        raise ValueError('nonfinite/negative native state')
    positive = mass > 0
    safe_mass = torch.where(positive, mass, 1.)
    mean = lambda x, size: functional.avg_pool3d(x[None, None], size, stride=1)[0, 0]
    cut = (slice(4, -4),)*3
    mean3 = mean(mass, 3)[3:-3, 3:-3, 3:-3]
    mean9 = mean(mass, 9)
    velocity = torch.where(positive[None], moments[1:4]/safe_mass, 0.)
    second = torch.where(positive[None], moments[4:]/safe_mass, 0.)
    v2 = velocity.square()
    tolerance = 64*torch.finfo(moments.dtype).eps*torch.maximum(torch.maximum(second.abs(), v2), torch.ones_like(v2))
    variance = torch.where(second-v2 > tolerance, second-v2, 0.)
    rough = torch.zeros_like(mean3)
    for axis in range(3):
        local = mean(moments[1+axis], 3)[3:-3, 3:-3, 3:-3]
        local = torch.where(mean3 > 0, local/torch.where(mean3 > 0, mean3, 1.), 0.)
        rough = rough+torch.where(positive[cut], (local-velocity[axis][cut]).square(), 0.)
    return torch.stack([mass[cut], mean3, mean9,
        root_nonnegative(variance.mean(0)[cut]), root_nonnegative(rough)])


def normal_interval(lo, hi):
    """Stable interval probability, including same-sign positive tails."""
    if lo >= hi:
        return 0.
    return float(ndtr(-lo)-ndtr(-hi) if lo > 0 else ndtr(hi)-ndtr(lo))


def ray_segments(observer, direction, mean, sigma, h, n=CORE, clip=12.):
    observer, direction = np.asarray(observer, float), np.asarray(direction, float)
    if observer.shape != (3,) or direction.shape != (3,) or not np.isfinite([observer, direction]).all():
        raise ValueError('finite observer/direction required')
    if not np.isclose(np.linalg.norm(direction), 1, atol=1e-10) or np.any(observer <= 0) or np.any(observer >= n*DX):
        raise ValueError('unit ray from an interior observer required')
    lo, hi = np.exp(KAPPA*(np.array([mean-clip*sigma, mean+clip*sigma])-10))*h/1000
    crossing = []
    exits = []
    for axis in range(3):
        if direction[axis] != 0:
            distances = (np.arange(n+1)*DX-observer[axis])/direction[axis]
            crossing.extend(distances[(distances > lo) & (distances < hi)])
            exits.append(((n*DX if direction[axis] > 0 else 0)-observer[axis])/direction[axis])
    hi = min(hi, min(exits))
    if hi <= lo:
        return []
    edges = np.unique([lo, *[d for d in crossing if d < hi], hi])
    segments = []
    for low, high in zip(edges[:-1], edges[1:]):
        point = observer+(low+high)/2*direction
        cell = tuple(map(int, np.floor(point/DX)))
        limits = np.log(np.array([low, high])*1000/h)/KAPPA+10
        segments.append(dict(cell=cell, limits=limits))
    return segments


def rectangle_probability(first, second, mean, cov, tolerance):
    sigma = np.sqrt(np.diag(cov))
    rho = cov[0, 1]/np.prod(sigma)
    residual = math.sqrt(1-rho*rho)
    a, b = (np.asarray(first)-mean[0])/sigma[0]
    c, d = (np.asarray(second)-mean[1])/sigma[1]
    def integrand(x):
        return math.exp(-x*x/2)/math.sqrt(2*math.pi)*normal_interval((c-rho*x)/residual, (d-rho*x)/residual)
    points = [x for x in (0., c/rho if rho else 0., d/rho if rho else 0.) if a < x < b]
    value, error = quad(integrand, a, b, epsabs=tolerance, epsrel=tolerance, points=points, limit=150)
    return max(value, 0.), error


def observation_kernel(mw, observer, directions, modulus, covariance, h=.6774, n=CORE, tolerance=1e-11):
    """Cell-face-split Gaussian integral after exact D^3 exponential tilting.

    Outputs weights independent of F. Directions are OBSERVATIONS, not inferred
    catalogue parents. Uniform-in-cell q defines a continuous positional law.
    """
    mw, modulus, covariance = np.asarray(mw, float), np.asarray(modulus, float), np.asarray(covariance, float)
    if mw.shape != (3,) or modulus.shape != (2,) or covariance.shape != (2, 2) or not np.isfinite([modulus]).all():
        raise ValueError('invalid point/modulus/covariance shape')
    if not np.isfinite(mw).all() or not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T) or h <= 0:
        raise ValueError('invalid observations')
    np.linalg.cholesky(covariance)
    cell_w = tuple(map(int, np.floor(mw/DX)))
    if min(cell_w) < 0 or max(cell_w) >= n:
        raise ValueError('MW point outside represented field: zero support')
    tilt = np.full(2, 3*KAPPA)
    mean = modulus+covariance@tilt
    sigma = np.sqrt(np.diag(covariance))
    parts = [ray_segments(observer, directions[i], mean[i], sigma[i], h, n) for i in range(2)]
    pairs, total_error = [], 0.
    for first in parts[0]:
        for second in parts[1]:
            probability, error = rectangle_probability(first['limits'], second['limits'], mean, covariance, tolerance)
            total_error += error
            if probability > 0:
                pairs.append((first['cell'], second['cell'], probability))
    # Host marginal uses ONLY its own Jacobian tilt, not the two-companion tilt.
    host_mean = modulus[0]+covariance[0, 0]*3*KAPPA
    host_parts = ray_segments(observer, directions[0], host_mean, sigma[0], h, n)
    hosts = [(part['cell'], normal_interval(*(part['limits']-host_mean)/sigma[0])) for part in host_parts]
    b = math.log(h/1000)-10*KAPPA
    return dict(mw=cell_w, observer=np.asarray(observer)/DX, pairs=pairs, hosts=hosts,
        log_jacobian=2*math.log(KAPPA)+6*b+float(tilt@modulus+.5*tilt@covariance@tilt),
        log_host_jacobian=math.log(KAPPA)+3*b+3*KAPPA*modulus[0]+.5*(3*KAPPA)**2*covariance[0, 0],
        numerical_probability_error=total_error, clipped_tail_probability_bound=float(4*ndtr(-12)),
        tilted_probability_in_field=sum(p[2] for p in pairs), tolerance=tolerance,
        units='(cMpc/h)^-3 mag^-2 sr^-2; joint sky density, not sky-conditioned normalization')


def log_likelihood(model, features, kernel):
    """Sum observation-induced cells; no oracle native identity/candidate input."""
    if not kernel['pairs']:
        raise ValueError('zero quadrature/field support, not a finite likelihood')
    w, observer = kernel['mw'], kernel['observer']
    logw = selected(model.log_prob(features, observer, [], 0), [w])[0]
    loga = model.log_prob(features, observer, [w], 1)
    grouped = {}
    for a, t, probability in kernel['pairs']:
        grouped.setdefault(a, []).append((t, probability))
    terms = []
    for a, entries in grouped.items():
        logt = model.log_prob(features, observer, [w, a], 2)
        weights = features.new_tensor([math.log(p) for _, p in entries])
        terms.append(loga[a]+torch.logsumexp(selected(logt, [t for t, _ in entries])+weights, 0))
    log_integral = torch.logsumexp(torch.stack(terms), 0)
    joint = logw+log_integral+kernel['log_jacobian']-9*math.log(DX)
    host = [(cell, p) for cell, p in kernel['hosts'] if p > 0]
    host_log = torch.logsumexp(selected(loga, [c for c, _ in host])+features.new_tensor([math.log(p) for _, p in host]), 0)
    mw_density = logw-3*math.log(DX)
    a_density = host_log+kernel['log_host_jacobian']-3*math.log(DX)
    return joint, dict(MW=mw_density, M31_given_MW=a_density,
        M33_given_MW_M31_data=joint-mw_density-a_density,
        log_weighted_cell_probability=log_integral,
        log_jacobian=features.new_tensor(kernel['log_jacobian']))


def conservative_permutation(moments, axis):
    """Permute whole extensive seven-vectors inside aligned8^3 core parents."""
    if axis not in range(3):
        raise ValueError('axis0..2 required')
    n = moments.shape[-1]-2*HALO
    if n % 8 or moments.shape != (7, n+8, n+8, n+8):
        raise ValueError('aligned core and four-cell halo required')
    out = moments.clone()
    core = moments[:, 4:-4, 4:-4, 4:-4]
    grouped = core.reshape(7, n//8, 8, n//8, 8, n//8, 8)
    out[:, 4:-4, 4:-4, 4:-4] = torch.roll(grouped, 4, 2+2*axis).reshape_as(core)
    return out
