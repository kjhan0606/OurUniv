"""Selected-population kinematic proxy, NOT resolved members or a field prior."""
import numpy as np
from scipy.special import logsumexp
from cf4_lg_observation_contract import basis


def pair_context(positions, mean_velocity, physical_sigma, pairs, *, dx, h):
    """Field-only interface; positions cMpc/h, velocity/physical sigma km/s."""
    x, v, s = [np.asarray(a, float) for a in (positions, mean_velocity, physical_sigma)]
    pairs = np.asarray(pairs, int).reshape(-1, 2)
    if (x.ndim != 2 or x.shape[1] != 3 or v.shape != x.shape or s.shape != x.shape
            or not all(np.isfinite(a).all() for a in (x, v, s)) or np.any(s < 0)
            or not np.isfinite([dx, h]).all() or min(dx, h) <= 0):
        raise ValueError('invalid field context; not negative scientific evidence')
    if not len(pairs):
        return dict(status='UNRESOLVED_SUPPORT', pairs=pairs)
    if (pairs.min() < 0 or pairs.max() >= len(x) or np.any(pairs[:, 0] == pairs[:, 1])
            or len(np.unique(pairs, axis=0)) != len(pairs)):
        raise ValueError('distinct valid candidate pairs required')
    i, j = pairs.T
    separation = (x[j]-x[i])*1000/h
    length = np.linalg.norm(separation, axis=1)
    if np.any(length <= 0):
        raise ValueError('zero separation has no carrier direction')
    base = np.zeros((len(pairs), 4, 3))
    base[:, 0], base[:, 2] = separation, v[j]-v[i]
    scales = np.empty((len(pairs), 4))
    scales[:, :2] = 1000*dx/h
    scales[:, 2] = np.sqrt(100**2+(s[i]**2).mean(1)+(s[j]**2).mean(1))
    scales[:, 3] = np.sqrt(100**2+(s[j]**2).mean(1))
    return dict(status='SUPPORTED_DIAGNOSTIC_ONLY', pairs=pairs, base=base,
                scales=scales, direction=separation/length[:, None])


def fit(normalized_residuals, directions):
    """One uniform-unique-pair fit: four radial means, ten covariance entries."""
    y, e = np.asarray(normalized_residuals), np.asarray(directions)
    if y.ndim != 3 or y.shape[1:] != (4, 3) or len(y) < 8 or e.shape != (len(y), 3):
        raise ValueError('at least eight unique eligible training pairs required')
    if not np.isfinite(y).all() or not np.isfinite(e).all():
        raise ValueError('nonfinite training data')
    np.testing.assert_allclose(np.linalg.norm(e, axis=1), 1, atol=1e-12)
    beta = np.einsum('nga,na->ng', y, e).mean(0)
    residual = y-beta[None, :, None]*e[:, None]
    raw = np.einsum('nga,nha->gh', residual, residual)/(3*len(y))
    covariance = .9*raw+.1*np.diag(np.diag(raw))
    condition_number = float(np.linalg.cond(covariance))
    diagnostics = dict(training_pairs=len(y), fitted_coefficients=14,
        covariance_estimator='pooled axis ML denominator3N, fixed10% diagonal shrinkage',
        eigenvalues_before=np.linalg.eigvalsh(raw).tolist(),
        eigenvalues_after=np.linalg.eigvalsh(covariance).tolist(),
        condition_number=condition_number if np.isfinite(condition_number) else None)
    try:
        np.linalg.cholesky(covariance)  # No covariance tuning or floor on failure.
    except np.linalg.LinAlgError as error:
        raise ValueError(f'nonpositive covariance; fit diagnostics: {diagnostics}') from error
    return dict(beta=beta, K=covariance, diagnostics=diagnostics)


def pair_means(model, context):
    return context['base']+context['scales'][:, :, None]*model['beta'][None, :, None]*context['direction'][:, None]


def pair_logpdf(target, model, context, rows=(0, 1, 2, 3)):
    rows = np.asarray(rows, int)
    target = np.asarray(target, float)
    if target.shape != (4, 3) or not np.isfinite(target).all():
        raise ValueError('finite four relative three-vectors required')
    covariance = model['K'][np.ix_(rows, rows)]
    factor = np.linalg.cholesky(covariance)
    scale = context['scales'][:, rows]
    residual = (target[rows]-pair_means(model, context)[:, rows])/scale[:, :, None]
    white = np.einsum('gh,nha->nga', np.linalg.solve(factor, np.eye(len(rows))), residual)
    return (-.5*(white**2).sum((1, 2))-3*np.log(np.diag(factor)).sum()
            -1.5*len(rows)*np.log(2*np.pi)-3*np.log(scale).sum(1))


def score(target, model, context):
    if context['status'] != 'SUPPORTED_DIAGNOSTIC_ONLY':
        return dict(status=context['status'], log_joint=None, log_host=None, log_M33_given_host=None)
    joint = pair_logpdf(target, model, context)
    host = pair_logpdf(target, model, context, (0, 2))
    log_joint, log_host = [float(logsumexp(p)-np.log(len(p))) for p in (joint, host)]
    result = dict(status='SUPPORTED_DIAGNOSTIC_ONLY', log_joint=log_joint, log_host=log_host,
                  log_M33_given_host=log_joint-log_host, candidate_pairs=len(joint))
    for label, p in (('all_data', joint), ('host_only', host)):
        weights = np.exp(p-logsumexp(p))
        result[label+'_pair_ESS'] = float(1/(weights@weights))
        result[label+'_max_pair_weight'] = float(weights.max())
    return result


def predictive_moments(model, context):
    """Unconditioned equal-pair proxy uncertainty; NOT field physical sigma_v."""
    means = pair_means(model, context)
    variance = context['scales']**2*np.diag(model['K'])
    center = means.mean(0)
    total = (variance[:, :, None]+(means-center)**2).mean(0)
    return center, np.sqrt(total)


def latent_observables(relative, *, h, solar_position_kpc, solar_velocity_km_s):
    """Mock ICRS: [RA,Dec,DM,vlos,muRA*,muDec] for M31/M33. NOT halo certification."""
    r31, r32, v31, v32 = np.asarray(relative, float)
    rows = []
    for dr, dv in ((r31, v31), (r31+r32, v31+v32)):
        position = dr-np.asarray(solar_position_kpc)
        distance = np.linalg.norm(position)
        if distance <= 0:
            raise ValueError('positive heliocentric distance required')
        ra = np.rad2deg(np.arctan2(position[1], position[0])) % 360
        dec = np.rad2deg(np.arcsin(position[2]/distance))
        velocity = dv+100*h*dr/1000-np.asarray(solar_velocity_km_s)
        los, east, north = basis(ra, dec)@velocity
        rows.append([ra, dec, 5*np.log10(distance)+10, los,
                     east*1000/(4.74047*distance), north*1000/(4.74047*distance)])
    return np.array(rows)


def latent_from_observables(observables, *, h, solar_position_kpc, solar_velocity_km_s):
    positions, velocities = [], []
    for ra, dec, dm, los, east, north in np.asarray(observables):
        distance = 10**((dm-10)/5)
        axes = basis(ra, dec)
        dr = distance*axes[0]+np.asarray(solar_position_kpc)
        heliovel = axes.T@np.array([los, east*4.74047*distance/1000, north*4.74047*distance/1000])
        dv = heliovel+np.asarray(solar_velocity_km_s)-100*h*dr/1000
        positions.append(dr); velocities.append(dv)
    return np.array([positions[0], positions[1]-positions[0], velocities[0], velocities[1]-velocities[0]])
