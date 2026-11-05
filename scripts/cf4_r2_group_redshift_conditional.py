"""Empirical CF4-group redshift conditional on secure 2M++ member redshifts.

This calibrates only p(V_CF4_group | matched individual V_2M++, secure-match
selection). It neither reconstructs the Tully velocity-membership process nor
supplies p(CF4 distance mark | field), selection, or a joint R2 likelihood.
"""

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, t


ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/gpfs/kjhan/CF4/z0_density/r2_group_redshift_conditional_v1')
NATIVE = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1/native_data_v2/CF4_native.npz')
ELIGIBLE = Path('/gpfs/kjhan/CF4/z0_density/r2_inclusive_count_diagnostic_v1/row_diagnostics.npz')
HASHES = {
    'cf4_groups.csv': 'bfdc0cfc0f172b48468e3a8fd05e87978c1ec68c341fb2d929fc1200f0123334',
    '2mpp_catalog.csv': '05d39f49af58caa7aa199420cc7354b3aa9fe3dbacf8d6c33222479c288fb23d',
    'cf4_2mpp_crossmatch_v1.csv': '64e4f8a1a8a612a19788ac759062930991a8ffe52bfa203635845fa1ad7a83bf',
}
DF = 4.0


def rows(name):
    path = ROOT / 'data' / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != HASHES[name]:
        raise ValueError(f'frozen input changed: {name}')
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def fit_student(residual):
    x = np.asarray(residual, dtype=np.float64)
    centre = float(np.median(x))
    scale = max(float(np.median(np.abs(x-centre))), 1.0)

    def objective(params):
        loc, log_scale = params
        return -float(np.sum(t.logpdf(x, DF, loc=loc, scale=np.exp(log_scale))))

    fitted = minimize(objective, [centre, np.log(scale)], method='L-BFGS-B',
                      bounds=[(None, None), (np.log(1e-3), np.log(1e5))])
    if not fitted.success or not np.isfinite(fitted.fun):
        raise ValueError(f'Student-t fit did not converge: {fitted.message}')
    return float(fitted.x[0]), float(np.exp(fitted.x[1]))


def evaluate(residual, loc, scale, gaussian_loc, gaussian_scale):
    x = np.asarray(residual, dtype=np.float64)
    result = {
        'n': int(x.size),
        'mean_log_score_student_t4': float(np.mean(t.logpdf(x, DF, loc=loc, scale=scale))),
        'mean_log_score_gaussian': float(np.mean(norm.logpdf(x, loc=gaussian_loc,
                                                            scale=gaussian_scale))),
        'signed_residual_median_km_s': float(np.median(x)),
        'absolute_residual_p90_km_s': float(np.percentile(np.abs(x), 90)),
    }
    for mass in (0.68, 0.90, 0.95):
        half_width = float(t.ppf((1+mass)/2, DF)*scale)
        result[f'student_t4_central_{int(mass*100)}_coverage'] = float(
            np.mean(np.abs(x-loc) <= half_width))
    return result


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm allocation required')
    if OUT.exists():
        raise FileExistsError(OUT)
    with np.load(NATIVE, allow_pickle=False) as saved:
        pgc = saved['CF4_pgc'].astype(np.int64)
        holdout = saved['CF4_holdout'].astype(bool)
    if pgc.size != 19313 or len(np.unique(pgc)) != pgc.size:
        raise ValueError('native CF4 group IDs changed or are duplicated')
    split = {int(g): bool(h) for g, h in zip(pgc, holdout)}
    with np.load(ELIGIBLE, allow_pickle=False) as saved:
        eligible = set(map(int, saved['recno']))
    cf4 = {int(row['1PGC']): row for row in rows('cf4_groups.csv')}
    mpp = {int(row['recno']): row for row in rows('2mpp_catalog.csv')}
    by_group = defaultdict(dict)
    for row in rows('cf4_2mpp_crossmatch_v1.csv'):
        if row['match_class'] != 'secure_joint_mark':
            continue
        recno = int(row['twompp_recno'])
        group = int(row['1PGC'])
        if recno not in eligible or group not in split or group not in cf4:
            continue
        galaxy = mpp.get(recno)
        if galaxy is None or not galaxy['Vcmb']:
            continue
        previous = by_group[group].get(recno)
        if previous is not None and previous != float(galaxy['Vcmb']):
            raise ValueError('conflicting secure redshift for one member')
        by_group[group][recno] = float(galaxy['Vcmb'])

    values = defaultdict(lambda: {'train': [], 'holdout': []})
    member_count = defaultdict(lambda: {'train': 0, 'holdout': 0})
    skipped_missing_group_velocity = 0
    for group, members in by_group.items():
        source = cf4[group]
        if not source['Vcmb']:
            skipped_missing_group_velocity += 1
            continue
        label = 'holdout' if split[group] else 'train'
        stratum = 'one_secure_member' if len(members) == 1 else 'multiple_secure_members'
        residual = float(source['Vcmb'])-float(np.mean(list(members.values())))
        values[stratum][label].append(residual)
        member_count[stratum][label] += len(members)

    results = {}
    for stratum, parts in values.items():
        train = np.asarray(parts['train'], dtype=np.float64)
        test = np.asarray(parts['holdout'], dtype=np.float64)
        if train.size < 100 or test.size < 30:
            raise ValueError(f'insufficient frozen split in {stratum}')
        loc, scale = fit_student(train)
        gaussian_loc, gaussian_scale = float(np.mean(train)), float(np.std(train, ddof=1))
        if gaussian_scale <= 0:
            raise ValueError('degenerate Gaussian comparison')
        results[stratum] = {
            'training_groups': int(train.size), 'heldout_groups': int(test.size),
            'training_secure_member_rows': member_count[stratum]['train'],
            'heldout_secure_member_rows': member_count[stratum]['holdout'],
            'fitted_student_t4_loc_km_s': loc,
            'fitted_student_t4_scale_km_s': scale,
            'fitted_gaussian_loc_km_s': gaussian_loc,
            'fitted_gaussian_scale_km_s': gaussian_scale,
            'training': evaluate(train, loc, scale, gaussian_loc, gaussian_scale),
            'heldout': evaluate(test, loc, scale, gaussian_loc, gaussian_scale),
        }

    report = {
        'classification': 'R2_EMPIRICAL_GROUP_REDSHIFT_CONDITIONAL_NOT_JOINT_LIKELIHOOD',
        'source_sha256': HASHES,
        'sample_definition': 'native-CF4 groups with >=1 N128-eligible secure 2M++ match',
        'conditioned_on': 'mean observed individual 2M++ Vcmb in the secure matched subset',
        'target': 'published CF4 group Vcmb',
        'split': 'preserved native CF4 group train/holdout; no refit on holdout',
        'student_degrees_of_freedom_fixed': DF,
        'groups_with_secure_eligible_members': len(by_group),
        'skipped_missing_group_velocity': skipped_missing_group_velocity,
        'strata': results,
        'limitation': ('This empirical conditional uses a secure matched subset and holds the '
                       'group-selection mechanism fixed. It has no field dependence, distance '
                       'mark law, CF4 selection model, or account of the 442 point-selection '
                       'anomalies. It cannot itself constrain an IC or be multiplied into the '
                       'old binned-count/BGc product as a production likelihood. The CF4 '
                       'holdout is not an independent sky volume.'),
    }
    OUT.mkdir(parents=True)
    (OUT/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
