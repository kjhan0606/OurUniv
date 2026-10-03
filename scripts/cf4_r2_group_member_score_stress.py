"""One-state grouped-member score stress for the inclusive partial likelihood.

This is NOT a survey mock or a calibrated CF4/2M++ likelihood. It tests the
mathematical sensitivity of a binned-count + one-group-mark product to shared
member velocities, distance-dependent mark selection, and duplicate marks.
Source group multiplicities and a saved PM density provide fixed geometry;
the synthetic rate, dispersion, and correlation coefficients are explicit
stress choices and must not be used as astrophysical priors.
"""

import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OVERLAP = Path('/gpfs/kjhan/CF4/z0_density/r2_inclusive_count_diagnostic_v1/row_diagnostics.npz')
CF4 = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1/native_data_v2/CF4_native.npz')
STATE = Path('/gpfs/kjhan/CF4/z0_density/r2_pm128_unconditional_v1/state.npz')
OUT = Path('/gpfs/kjhan/CF4/z0_density/r2_group_member_score_stress_v1')
GROUPS = 512
REPLICATES = 4096
SEED = 2026092603


def source_groups():
    tracer = json.loads((ROOT/'config/cf4_twompp_disjoint_tracer_pilot_program_v1.json').read_text())
    binding = tracer['inputs']['cf4_twompp_crossmatch']
    path = ROOT/binding['path']
    raw = path.read_bytes()
    if len(raw) != binding['bytes'] or hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError('crossmatch source binding changed')
    with np.load(OVERLAP,allow_pickle=False) as saved:
        eligible = set(map(int,saved['recno']))
    members = Counter()
    with path.open(newline='',encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row['match_class']=='secure_joint_mark' and int(row['twompp_recno']) in eligible:
                members[int(row['1PGC'])] += 1
    if sum(members.values()) != 14878:
        raise ValueError('eligible secure-match count changed')
    return members


def fixed_direction(members):
    with np.load(CF4,allow_pickle=False) as saved:
        pgc = saved['CF4_pgc'].astype(np.int64)
        pos = saved['CF4_pos'].astype(np.float64)
        variance = saved['CF4_variance'].astype(np.float64)
    if len(set(map(int,pgc))) != len(pgc):
        raise ValueError('native group identifiers not unique')
    available = np.flatnonzero(np.isin(pgc,list(members)))
    if len(available)<GROUPS:
        raise ValueError('insufficient secure native groups')
    rng = np.random.default_rng(SEED)
    # Fixed random *geometry sample*, not a ranked field/IC candidate.
    ids = rng.choice(available,GROUPS,replace=False)
    multiplicity = np.array([members[int(pgc[i])] for i in ids],dtype=np.int32)
    with np.load(STATE,allow_pickle=False) as saved:
        rho = saved['rho']
        if rho.shape!=(128,128,128) or np.any(rho<0) or not np.isfinite(rho).all():
            raise ValueError('saved N128 matter field invalid')
        cell = np.floor(pos[ids]/3.).astype(np.int32)%128
        log_density = np.log1p(rho[cell[:,0],cell[:,1],cell[:,2]])
    direction = log_density-log_density.mean()
    direction /= np.std(direction)
    direction = np.clip(direction,-2.5,2.5)
    direction -= direction.mean()
    direction /= np.sqrt(np.mean(direction**2))
    if np.any(variance[ids]<=0):
        raise ValueError('nonpositive native CF4 variance')
    return direction,multiplicity,np.sqrt(variance[ids])


def score_arm(rng, direction, multiplicity, *, rate_coupling, mark_coupling,
              select_on_mark=False, duplicate_mark=False):
    shape = (REPLICATES,GROUPS)
    u = rng.normal(size=shape)
    # Mean of independent member FoG residuals: variance 1/n. Common u
    # makes both individual redshifts and the group aggregate dependent.
    member_delta_km_s = 300*(u+rng.normal(size=shape)/np.sqrt(multiplicity))
    mean_member = member_delta_km_s/(300*np.sqrt(1+1/multiplicity))
    mark = mark_coupling*mean_member + np.sqrt(1-mark_coupling**2)*rng.normal(size=shape)
    lam = .8*np.exp(.2*direction)
    count = rng.poisson(lam*np.exp(rate_coupling*u-.5*rate_coupling**2))
    selected = (mark+.5*direction>-.5) if select_on_mark else np.ones(shape,dtype=bool)
    weight = multiplicity if duplicate_mark else np.ones(GROUPS,dtype=np.int32)
    scount = np.sum((count-lam)*direction,axis=1)
    smark = np.sum(selected*weight*mark*direction,axis=1)
    score = scount+smark
    hcount = float(np.sum(lam*direction**2))
    hmark = np.sum(selected*weight*direction**2,axis=1)
    hmean = float(np.mean(hcount+hmark))
    return dict(mean_score=float(score.mean()),mean_score_over_sqrt_H=float(score.mean()/np.sqrt(hmean)),
        variance_score=float(score.var(ddof=1)),mean_candidate_H=hmean,
        score_variance_to_H=float(score.var(ddof=1)/hmean),
        count_score_variance_to_H=float(scount.var(ddof=1)/hcount),
        count_mark_score_covariance=float(np.cov(scount,smark,ddof=1)[0,1]),
        mark_selection_fraction=float(selected.mean()),
        duplicate_group_marks=bool(duplicate_mark))


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm job required')
    if OUT.exists():
        raise FileExistsError(OUT)
    direction,multiplicity,sigma = fixed_direction(source_groups())
    arms = {
        'independent_reference': dict(rate_coupling=0.,mark_coupling=0.),
        'shared_member_velocity': dict(rate_coupling=.5,mark_coupling=.6),
        'distance_dependent_group_selection': dict(rate_coupling=.5,mark_coupling=.6,select_on_mark=True),
        'duplicate_group_mark_negative_control': dict(rate_coupling=0.,mark_coupling=0.,duplicate_mark=True),
    }
    result = {name:score_arm(np.random.default_rng(SEED+index+1),direction,multiplicity,**cfg)
              for index,(name,cfg) in enumerate(arms.items())}
    baseline = result['independent_reference']
    if abs(baseline['score_variance_to_H']-1)>0.12:
        raise AssertionError('independent reference does not reproduce information identity')
    if result['duplicate_group_mark_negative_control']['score_variance_to_H']<=1.05:
        raise AssertionError('duplicate group-mark negative control did not respond')
    report = dict(classification='ONE_STATE_GROUP_MEMBER_PARTIAL_LIKELIHOOD_STRESS_NOT_CALIBRATION',
        saved_field=str(STATE), secure_native_groups_sampled=GROUPS,
        source_secure_member_count_distribution={str(n):int(np.count_nonzero(multiplicity==n))
                                                  for n in np.unique(multiplicity)},
        cf4_sigma_velocity_km_s=dict(median=float(np.median(sigma)),p90=float(np.percentile(sigma,90))),
        mock_replicates_per_arm=REPLICATES,seed=SEED,
        synthetic_count_rate_per_group='.8 exp(.2 standardized log1p rho); not actual 2M++ intensity',
        synthetic_member_delta_km_s='300*(shared Gaussian + mean independent member Gaussian); standardized before mark coupling; stress only',
        arms=result,
        interpretation='Score variance/H>1 indicates candidate curvature underestimates variance for this synthetic dependence. Selection-score shift or duplicate marks are negative controls, not measured CF4 biases.',
        limitations='One unconditional N128 PM field, synthetic scalar direction/count proxy and stress coefficients. Not a full galaxy survey mock, not an actual CF4/2M++ likelihood calibration, no LG identification or posterior.')
    OUT.mkdir(parents=True)
    (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__=='__main__':
    main()
