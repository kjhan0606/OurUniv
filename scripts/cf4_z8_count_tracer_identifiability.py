import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def corr(a,b):
    a=np.asarray(a); b=np.asarray(b)
    return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else 0.0

def main():
    plan=json.loads((ROOT/'config/cf4_z8_count_tracer_identifiability_plan_v1.json').read_text())
    root=Path(plan['output_root']); root.mkdir(parents=True,exist_ok=False)
    rows=[]
    for case in plan['inputs']['cases']:
        d=Path(plan['inputs']['Z7_root'])/f"task_{case['task']}"
        result=json.loads((d/'result.json').read_text())
        with np.load(d/'posterior_fields.npz',allow_pickle=False) as f:
            truth=f['truth_density']; mean=f['density_mean']; sd=f['density_SD']
        chains=[]
        for c in range(4):
            with np.load(d/f'chain_{c}.npz',allow_pickle=False) as f:
                p=f['projections']
            chains.append(p)
        proj=np.concatenate(chains)
        bias=proj[:,6:12]; white_rms=proj[:,-2]; log_posterior=proj[:,-1]
        row={'task':case['task'],'truth_index':case['truth_index'],'channel':case['channel'],
             'status':result['status'],'density':result['density'],'heldout':result['heldout_pointwise_log_predictive_gain'],
             'bias_mean':bias.mean(axis=0).tolist(),'bias_sd':bias.std(axis=0).tolist(),
             'bias_white_rms_corr':[corr(bias[:,i],white_rms) for i in range(6)],
             'bias_log_posterior_corr':[corr(bias[:,i],log_posterior) for i in range(6)],
             'mean_pointwise_posterior_uncertainty_SD':float(sd.mean()),
             'truth_spatial_density_SD':float(truth.std()),
             'posterior_mean_spatial_density_SD':float(mean.std()),
             'posterior_mean_to_truth_spatial_SD_ratio':float(mean.std()/truth.std()),
             'uncertainty_to_truth_spatial_SD_ratio':float(sd.mean()/truth.std()),
             'posterior_draw_spatial_RMS_quadratic_mean':float(np.sqrt(np.mean(mean**2+sd**2))),
             'draw_RMS_identity': 'sqrt(mean_x(mean_posterior(delta)^2 + Var_posterior(delta))); each physical draw has zero spatial mean. Not SD of the posterior-mean map.'}
        rows.append(row)
    report={'bundle':plan['bundle'],'status':'CORRECTED_STATISTICS_COMPUTED_REQUIRES_REVIEW','cases':rows,
            'conclusion':{'count_channel':'Counts-only has higher whole-field density correlation in Z7. These white-RMS and scalar log-posterior probes do not test physical bias-density identifiability.',
                          'correction': 'white_RMS combines four standardized latent blocks; logdensity in old projections is scalar log posterior. Mean pointwise uncertainty divided by truth spatial SD is not recovered amplitude. No missing nonlinear tracer physics conclusion follows.',
                          'promotion':'No actual CF4 promotion or resolution decision is made.',
                          'next':'Use audit to specify count/tracer model requirements and heldout population tests.'},
            'next_bundle_started':False}
    (root/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'bundle':plan['bundle'],'status':report['status'],'cases':len(rows)}),flush=True)

if __name__=='__main__': main()
