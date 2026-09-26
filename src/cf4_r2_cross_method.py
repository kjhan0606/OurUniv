"""Source observation links and a CONDITIONAL relative-distance diagnostic.

Not an independent calibration prior or a complete selected-group likelihood.
Common source fitting, selection, depth and association errors remain outside
the Gaussian moment diagnostic. Keep original measurements for joint inference.
"""
import numpy as np

METHODS = ('snIa', 'tf', 'sbf', 'snII', 'trgb', 'ceph', 'mas')


def link_nonfp_rows(rows, pgc_to_source, source_labels):
    """Exact PGC/T17 association; NEVER use combined DM, DMfp or inferred1PGC."""
    labels = set(source_labels)
    linked, issues, seen = [], [], set()
    for row in rows:
        pgc, recno = int(row['PGC']), int(row['recno'])
        same = pgc_to_source.get(pgc)
        t = int(row['T17'] or 0)
        tempel = f'T{t}' if t > 0 else None
        if same is not None and tempel is not None and same != tempel:
            issues.append(dict(recno=recno, PGC=pgc, reason='conflicting_PGC_T17',
                               source_group=same, catalogue_group=tempel))
            continue
        group = same if same is not None else tempel
        if group not in labels:
            continue
        for method in METHODS:
            value, error = row.get('DM'+method, ''), row.get('e_DM'+method, '')
            if not value and not error:
                continue
            if not value or not error:
                issues.append(dict(recno=recno, method=method, reason='incomplete_measurement'))
                continue
            mu, sigma = float(value), float(error)
            if not np.isfinite([mu, sigma]).all() or sigma <= 0:
                issues.append(dict(recno=recno, method=method, reason='invalid_measurement'))
                continue
            key = (pgc, method)
            if key in seen:
                raise ValueError(f'duplicate PGC/method observation: {key}')
            seen.add(key)
            linked.append(dict(recno=recno, PGC=pgc, method=method, modulus=mu,
                error=sigma, source_group=group, CF4_group=int(row['1PGC']),
                association='same_PGC' if same is not None else 'explicit_T17',
                CF3=int(row['CF3'] or 0)))
    return linked, issues


def closed_group_holdout(fp_groups, fp_cf4, fp_holdout, anchors):
    """Close inherited holdout over source/CF4 links, including new anchors."""
    parent = {}
    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def join(s, c):
        a, b = root('source:'+str(s)), root('cf4:'+str(c))
        parent[a] = b
    for s, c in zip(fp_groups, fp_cf4, strict=True):
        join(s, c)
    for a in anchors:
        join(a['source_group'], a['CF4_group'])
    held = {root('source:'+str(s)) for s, h in zip(fp_groups, fp_holdout, strict=True) if h}
    return {str(s): root('source:'+str(s)) in held for s in set(fp_groups)}


def relative_system(y, sigma, group, fp_variance, method, nmethod, excess_variance):
    """C_g=diag(sigma^2)+(FPmean variance+excess variance)11^T.

    Uses rank-one block inverses; no dense N_observation covariance allocation.
    Method offsets are shared globally. fp_variance has one entry per group;
    method/group indices may be unsorted. Shared covariance is counted ONCE.
    """
    y, sigma = np.asarray(y, float), np.asarray(sigma, float)
    group, method = np.asarray(group, int), np.asarray(method, int)
    fp_variance = np.asarray(fp_variance, float)
    if (y.ndim != 1 or any(v.shape != y.shape for v in (sigma, group, method))
            or not np.isfinite(y).all() or not np.isfinite(sigma).all()
            or np.any(sigma <= 0) or np.any(group < 0) or np.any(group >= len(fp_variance))
            or np.any(method < 0) or np.any(method >= nmethod)
            or not np.isfinite(fp_variance).all() or np.any(fp_variance < 0)
            or not np.isfinite(excess_variance) or excess_variance < 0):
        raise ValueError('invalid relative-distance covariance inputs')
    x = np.eye(nmethod)[method]
    w = 1/sigma**2
    summed = np.bincount(group, weights=w, minlength=len(fp_variance))
    common = fp_variance+excess_variance
    shrink = common/(1+common*summed)
    def inverse(values):
        weighted = w[:, None]*values
        totals = np.stack([np.bincount(group, weights=weighted[:, k],
            minlength=len(fp_variance)) for k in range(values.shape[1])], axis=1)
        return weighted-w[:, None]*shrink[group, None]*totals[group]
    inverse_x = inverse(x)
    information = x.T@inverse_x
    rhs = x.T@inverse(y[:, None])[:, 0]
    logdet = float(np.log(sigma**2).sum()+np.log1p(common*summed).sum())
    return x, inverse, information, rhs, logdet


def fit_relative_at_variance(y, sigma, group, fp_variance, method, nmethod, variance):
    x, inverse, information, rhs, logdet = relative_system(
        y, sigma, group, fp_variance, method, nmethod, variance)
    sign, logdet_info = np.linalg.slogdet(information)
    if sign <= 0 or len(y) <= nmethod:
        raise ValueError('unidentified relative method offsets')
    offset = np.linalg.solve(information, rhs)
    residual = y-x@offset
    quadratic = float(residual@inverse(residual[:, None])[:, 0])
    reml = .5*(quadratic+logdet+logdet_info+(len(y)-nmethod)*np.log(2*np.pi))
    return dict(offset=offset, information=information, quadratic=quadratic,
                logdet=logdet, reml_nll=float(reml))


def heldout_relative_score(y, sigma, group, fp_variance, method, fit, variance):
    """Integrate shared method-offset uncertainty from TRAIN only, variance fixed.

    Assumes no group shared with training (caller owns split closure).
    Does not refit offsets/variance on heldout. Not independent validation of
    the source catalogue, whose FP/calibration fitting used all source rows.
    """
    x, inverse, info, _, logdet = relative_system(y, sigma, group, fp_variance,
        method, len(fit['offset']), variance)
    residual = y-x@fit['offset']
    ci_residual = inverse(residual[:, None])[:, 0]
    rhs = x.T@ci_residual
    joint_info = fit['information']+info
    quadratic = float(residual@ci_residual-rhs@np.linalg.solve(joint_info, rhs))
    logdet += np.linalg.slogdet(joint_info)[1]-np.linalg.slogdet(fit['information'])[1]
    return dict(rows=len(y), quadratic=quadratic,
                logpdf=float(-.5*(quadratic+logdet+len(y)*np.log(2*np.pi))))
