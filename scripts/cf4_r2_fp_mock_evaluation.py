"""One fixed selected-mock FP uncertainty evaluation; no calibration fit."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import skewnorm

BASE = Path('/gpfs/kjhan/CF4/z0_density')
OUT = BASE/'r2_fp_mock_evaluation_v1'


def cdf_at_truth(truth, mean, std, alpha):
    """Convert published moments to skew-normal loc/scale, not vice versa."""
    delta = alpha/np.sqrt(1+alpha**2)
    scale = std/np.sqrt(1-2*delta**2/np.pi)
    loc = mean-scale*delta*np.sqrt(2/np.pi)
    return skewnorm.cdf(truth,alpha,loc=loc,scale=scale)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    if OUT.exists():
        raise FileExistsError(OUT)
    binding = json.loads((BASE/'r2_sdss_mock_bridge_v1/source.json').read_text())
    raw = BASE/'r2_sdss_mock_bridge_v1/first_mock.dat'
    if hashlib.sha256(raw.read_bytes()).hexdigest() != binding['member_sha256']:
        raise ValueError('mock source changed')
    original = np.genfromtxt(raw,names=True,delimiter=',')
    with np.load(BASE/'r2_sdss_mock_bridge_v3/host_member_inputs.npz') as f:
        data = {k:f[k] for k in f.files}
    mean, truth, std, alpha = [data[k] for k in
        ('eta_group_measured','eta_group_truth','eta_std','eta_alpha')]
    if any(np.any(~np.isfinite(x)) for x in (mean,truth,std,alpha)) or np.any(std<=0):
        raise ValueError('invalid FP parameters')
    if len(mean) != len(original):
        raise ValueError('source row mismatch')
    np.testing.assert_array_equal(data['z_true'],original['z_true'])
    probability = cdf_at_truth(truth,mean,std,alpha)
    original_probability = cdf_at_truth(original['logdist_true'],original['logdist'],std,alpha)
    np.testing.assert_allclose(probability,original_probability,atol=1e-12,rtol=0)
    # A Gaussian special case checks moment conversion/CDF independently.
    np.testing.assert_allclose(cdf_at_truth(np.array([0.]),np.array([0.]),np.array([1.]),np.array([0.])),.5)
    residual = (mean-truth)/std
    strata = []
    for pop,mask in (('all',np.ones(len(mean),bool)),('central',data['central']),('satellite',~data['central'])):
        for name,low,high in (('all',0.,.2),('z_lt_0p03',0.,.03),('z_0p03_0p06',.03,.06),('z_ge_0p06',.06,.2)):
            choose = mask&(data['z_true']>=low)&(data['z_true']<high)
            n = int(choose.sum())
            if n == 0:
                strata.append(dict(population=pop,redshift_bin=name,n=0))
                continue
            p = probability[choose]
            r = residual[choose]
            strata.append(dict(population=pop,redshift_bin=name,n=n,
                mean_eta_bias_dex=float(np.mean((mean-truth)[choose])),
                standardized_residual_mean=float(r.mean()),standardized_residual_sd=float(r.std()),
                source_pdf_central68_truth_fraction=float(np.mean((p>=.16)&(p<=.84))),
                source_pdf_central90_truth_fraction=float(np.mean((p>=.05)&(p<=.95))),
                source_pdf_central95_truth_fraction=float(np.mean((p>=.025)&(p<=.975))),
                source_cdf_at_truth_mean=float(p.mean()),
                eta_residual_quantiles_dex=np.quantile((mean-truth)[choose],[.05,.5,.95]).tolist()))
    result = dict(classification='ONE_SELECTED_MOCK_FP_MEASUREMENT_CHECK_NOT_GROUP_CALIBRATION',
        job_id=os.environ['SLURM_JOB_ID'],source=binding,strata=strata,
        reference_conversion_cdf_max_abs=float(np.max(np.abs(probability-original_probability))),
        independent_mock_boxes=1,source_FP_fit_shared_with_evaluation_rows=True,
        group_richness_correction_reproduced=False,Tempel_group_recovery_tested=False,
        group_covariance_calibrated=False,group_inclusion_calibrated=False,
        source_pdf_intervals_are_not_full_distance_posterior=True,R2_posterior=False,
        fitted_parameters=0,new_gravity_runs=0)
    OUT.mkdir(parents=True)
    (OUT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
