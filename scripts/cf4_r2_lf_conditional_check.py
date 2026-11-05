"""One sky-heldout observed-K LF check; not an R2 posterior calibration."""

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kstest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_actual_selection import base, magnitude_h
from cf4_r2_lf_conditional import (conditional_logpdf, conditional_pit,
                                   observed_k_intervals)

BASE = Path('/gpfs/kjhan/CF4/z0_density')
OUT = BASE/'r2_lf_conditional_skyholdout_v1'
PARENT = BASE/'r2_common_catalogue_128_v1/rows.npz'
HOLDOUT_OCTANT = 3


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm allocation required')
    if OUT.exists():
        raise FileExistsError(OUT)
    tracer = json.loads((ROOT/'config/cf4_twompp_disjoint_tracer_pilot_program_v1.json').read_text())
    source = tracer['inputs']['twompp_catalog']
    path = ROOT/source['path']
    if (path.stat().st_size != source['bytes']
            or hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']):
        raise ValueError('2M++ source catalogue changed')
    with np.load(PARENT, allow_pickle=False) as saved:
        rows = {key: saved[key] for key in ('recno', 'population', 'position_cMpc_h')}
    galaxy = base.load_catalog(path)
    order = np.argsort(galaxy['recno'])
    index = order[np.searchsorted(galaxy['recno'][order], rows['recno'])]
    np.testing.assert_array_equal(galaxy['recno'][index], rows['recno'])
    common = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())['common_cosmology']
    _, magnitude_phys = base.distance_and_absolute_magnitude(
        galaxy['Vcmb'][index], galaxy['Ksmag'][index], common)
    magnitude = magnitude_h(magnitude_phys, common['h'])
    apparent = rows['population']//3
    lower, upper = observed_k_intervals(galaxy['Ksmag'][index]-magnitude, apparent)
    valid = ((galaxy['Vcmb'][index] >= 5000.)
             & (galaxy['Vcmb'][index] <= 20000.)
             & (upper > lower+1e-8)
             & (magnitude >= lower)
             & (magnitude <= upper))
    # An entire predeclared sky octant is unseen in the two-parameter fit.
    relative = rows['position_cMpc_h']-192.
    octant = ((relative[:, 0] >= 0).astype(int)*4
              +(relative[:, 1] >= 0).astype(int)*2
              +(relative[:, 2] >= 0).astype(int))
    train = valid & (octant != HOLDOUT_OCTANT)
    held = valid & (octant == HOLDOUT_OCTANT)
    if train.sum() < 1000 or held.sum() < 100:
        raise ValueError('sky split lacks supported observed-K rows')
    def nll(parameter):
        alpha, mstar = parameter
        return -float(np.sum(conditional_logpdf(magnitude[train],lower[train],
                                                  upper[train],alpha,mstar)))
    fit = minimize(nll, (-.8,-23.2), bounds=((-1.3,-.5),(-23.8,-22.7)),
                   method='L-BFGS-B', options={'maxiter':100})
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f'conditional LF fit failed: {fit.message}')
    models = {'imported_low_z_bright': (-.94,-23.28),
              'source_default': (-.73,-23.17),
              'sky_training_fit': tuple(map(float,fit.x))}
    report = dict(classification='R2_LF_CONDITIONAL_SKY_HOLDOUT_ONLY',
        job_id=os.environ['SLURM_JOB_ID'], status='COMPLETE',
        source_catalogue=str(path), parent_rows=len(rows['recno']),
        LF_window='5000<=Vcmb<=20000 km/s, -25<M_h<-21',
        supported_window_rows=int(valid.sum()),
        holdout_octant=HOLDOUT_OCTANT, train_rows=int(train.sum()),
        heldout_rows=int(held.sum()), heldout_apparent_counts=np.bincount(
            apparent[held], minlength=2).tolist(),
        model_assumption='conditional magnitude at fixed observed redshift/class; '
                         'angular map cancels only within the assumed two-class completeness law',
        models={}, independent_validation=False, rate_or_bias_calibrated=False,
        source_RSD_or_CF4_overlap_calibrated=False, R2_posterior=False)
    for label, (alpha, mstar) in models.items():
        logp = conditional_logpdf(magnitude[held],lower[held],upper[held],alpha,mstar)
        pit = conditional_pit(magnitude[held],lower[held],upper[held],alpha,mstar)
        report['models'][label] = dict(alpha=alpha,mstar=mstar,
            heldout_mean_log_density=float(logp.mean()),
            heldout_total_log_density=float(logp.sum()),
            heldout_pit_KS_D=float(kstest(pit,'uniform').statistic),
            heldout_pit_mean=float(pit.mean()))
    report['training_fit'] = dict(alpha=float(fit.x[0]),mstar=float(fit.x[1]),
                                  train_total_log_density=float(-fit.fun))
    OUT.mkdir(parents=True)
    (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
