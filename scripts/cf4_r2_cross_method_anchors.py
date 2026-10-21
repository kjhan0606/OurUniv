"""Real non-FP observation bridge and one conditional relative calibration fit."""
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_r2_cross_method import (METHODS, link_nonfp_rows, closed_group_holdout,
                                fit_relative_at_variance, heldout_relative_score)

BASE = Path('/gpfs/kjhan/CF4/z0_density')
INPUT = BASE/'r2_source_observation_assembly_v1/observations.npz'
CATALOGUE = ROOT/'data/cf4_galaxies.csv'
OUT = BASE/'r2_cross_method_anchors_v1'


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.monotonic()
    if hashlib.sha256(CATALOGUE.read_bytes()).hexdigest() != '28e7b8bd386f53716ed84cddd67a6f7602f98bc1394923a312f906555da7f709':
        raise ValueError('CF4 source changed')
    with np.load(INPUT) as f:
        data = {k:f[k].copy() for k in f.files}
    with CATALOGUE.open() as f:
        rows = list(csv.DictReader(f))
    if len({r['recno'] for r in rows}) != len(rows):
        raise ValueError('duplicate CF4 source row')
    pgc_source = dict(zip(map(int, data['PGC']), data['source_group'], strict=True))
    if len(pgc_source) != len(data['PGC']):
        raise ValueError('nonunique FP PGC')
    labels, gid = np.unique(data['source_group'], return_inverse=True)
    lookup = dict(zip(labels, range(len(labels))))
    anchors, issues = link_nonfp_rows(rows, pgc_source, labels)
    closed = closed_group_holdout(data['source_group'], data['CF4_group'], data['holdout'], anchors)
    group_hold = np.array([closed[s] for s in labels])
    anchor_fields = ('recno', 'PGC', 'method', 'modulus', 'error', 'source_group',
                     'CF4_group', 'association', 'CF3')
    packed = {k:np.array([a[k] for a in anchors]) for k in anchor_fields}
    ag = np.array([lookup[a['source_group']] for a in anchors], dtype=int)
    ahold = group_hold[ag]
    packed.update(group_index=ag, holdout=ahold)
    variance = (5*data['eta_std'])**2
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError('invalid source FP variance')
    inverse = 1/variance
    weight = np.bincount(gid, weights=inverse)
    eta_mean = np.bincount(gid, weights=inverse*data['eta_mean'])/weight
    fp_variance = 1/weight
    nfp = np.bincount(gid)
    zg = np.bincount(gid, weights=data['zgroup'])/nfp
    if np.max(np.abs(zg[gid]-data['zgroup'])) > 1e-10:
        raise ValueError('inconsistent source group redshift')
    zgrid = np.linspace(0., .2, 20001)
    dgrid = 2997.92458*cumulative_trapezoid(1/np.sqrt(.31*(1+zgrid)**3+.69), zgrid, initial=0.)
    dl_reference = (1+zg)*np.interp(zg, zgrid, dgrid)/.746
    fp_mu = 5*np.log10(dl_reference)+25-5*eta_mean
    strata = []
    for m in METHODS:
        choose = packed['method'] == m
        strata.append(dict(method=m, rows=int(choose.sum()),
            groups=int(len(set(ag[choose]))), train_rows=int(np.sum(choose & ~ahold)),
            holdout_rows=int(np.sum(choose & ahold)),
            same_PGC_rows=int(np.sum(choose & (packed['association'] == 'same_PGC')))))
    result = dict(classification='OBSERVED_NONFP_BRIDGE_AND_CONDITIONAL_RELATIVE_CALIBRATION',
        job_id=os.environ['SLURM_JOB_ID'], FP_rows=len(gid), FP_groups=len(labels),
        FP_groups_with_multiple_FP=int(np.sum(nfp > 1)), independent_FP_contrasts=int(np.sum(nfp-1)),
        FP_contrasts_identify_common_group_shift=False, FP_contrasts_identify_common_variance=False,
        anchors=len(anchors), anchor_groups=len(set(ag)),
        new_holdout_FP_rows=int(np.sum(group_hold[gid] & ~data['holdout'])),
        train_FP_groups=int(np.sum(~group_hold)), holdout_FP_groups=int(group_hold.sum()),
        CF3_flags=dict(Counter(str(a['CF3']) for a in anchors)),
        association_counts=dict(Counter(a['association'] for a in anchors)), strata=strata,
        association_or_measurement_issues=issues, raw_method_moduli_preserved=True,
        combined_DM_and_DMfp_used_as_anchors=False, method_zero_points_free=True,
        source_calibration_shared=True, independent_holdout_validation=False,
        group_covariance_calibrated=False, central_satellite_law_calibrated=False,
        group_inclusion_calibrated=False, role_dependent_FoG_calibrated=False,
        R2_posterior=False, new_gravity_runs=0,
        source_papers=['https://arxiv.org/html/2209.11238','https://arxiv.org/html/2201.03112'],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (INPUT, CATALOGUE)},
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (Path(__file__),ROOT/'src/cf4_r2_cross_method.py')})
    train_methods = sorted(set(packed['method'][~ahold]))
    method_id = {m:i for i, m in enumerate(train_methods)}
    mid = np.array([method_id.get(m, -1) for m in packed['method']], dtype=int)
    train = ~ahold
    hold = ahold & (mid >= 0)
    if np.intersect1d(ag[train], ag[hold]).size:
        raise ValueError('group leaks train/holdout')
    if len(train_methods) and train.sum() > len(train_methods):
        residual = packed['modulus']-fp_mu[ag]
        args = (residual[train], packed['error'][train], ag[train], fp_variance,
                mid[train], len(train_methods))
        objective = lambda v: fit_relative_at_variance(*args, v)['reml_nll']
        opt = minimize_scalar(objective, bounds=(0., 1.), method='bounded',
                              options={'xatol':1e-7, 'maxiter':80})
        if not opt.success:
            raise RuntimeError('bounded relative-variance fit did not converge')
        candidates = [(0., objective(0.)), (float(opt.x), float(opt.fun)), (1., objective(1.))]
        best_v = min(candidates, key=lambda pair:pair[1])[0]
        fit = fit_relative_at_variance(*args, best_v)
        zero_fit = fit_relative_at_variance(*args, 0.)
        beta_sd = np.sqrt(np.diag(np.linalg.inv(fit['information'])))
        fitted = dict(methods=train_methods, offset_mag=fit['offset'].tolist(),
            offset_conditional_SD_mag=beta_sd.tolist(), relative_group_excess_SD_mag=float(np.sqrt(best_v)),
            variance_search_boundary=('zero' if best_v == 0 else 'upper' if best_v == 1 else 'interior'),
            reml_nll=fit['reml_nll'], no_excess_reml_nll=zero_fit['reml_nll'],
            train_quadratic=fit['quadratic'], train_rows=int(train.sum()),
            train_groups=len(set(ag[train])), optimizer_evaluations=int(opt.nfev),
            heldout_rows_with_untrained_method=int(np.sum(ahold & (mid < 0))),
            interpretation='TOTAL RELATIVE discrepancy, not pure FP shared scatter or independent calibration prior')
        if np.any(hold):
            hargs = (residual[hold], packed['error'][hold], ag[hold], fp_variance, mid[hold])
            fitted['heldout_fixed_fit'] = heldout_relative_score(*hargs, fit, best_v)
            fitted['heldout_no_excess_fixed_fit'] = heldout_relative_score(*hargs, zero_fit, 0.)
        result['conditional_diagnostic_fit'] = fitted
    else:
        result['conditional_diagnostic_fit'] = dict(status='INSUFFICIENT_TRAINING_OVERLAP')
    result['runtime_seconds'] = time.monotonic()-start
    OUT.mkdir(parents=True)
    np.savez_compressed(OUT/'anchors.npz', **packed)
    np.savez_compressed(OUT/'fp_group_moments.npz', group_labels=labels, holdout=group_hold,
        group_n_FP=nfp, group_z=zg, eta_mean=eta_mean, mu_FP_proxy=fp_mu,
        mu_FP_independent_variance=fp_variance)
    (OUT/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
