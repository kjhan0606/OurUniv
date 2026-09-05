"""Stored-field definition sensitivity, never a posterior repair or refit."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cf4_bgc_fixed_design_smoke as fixed
import cf4_datum_bearing_z0_phasec_pilot as phasec


def corr(x,y):
    return float(np.corrcoef(x.ravel(),y.ravel())[0,1])


def apply_filter(field,window):
    return np.fft.ifftn(np.fft.fftn(field,norm='ortho')*window,norm='ortho').real


def diagnose(truth, samples):
    mean = samples.mean(axis=0,dtype=np.float64)
    sd = samples.std(axis=0,dtype=np.float64)
    if not np.isfinite(samples).all() or np.any(sd <= 0):
        raise ValueError('invalid transformed draws')
    quantiles = np.quantile(samples,[.025,.16,.84,.975],axis=0)
    z = (truth-mean)/sd
    return {'coverage68':float(np.mean((truth>=quantiles[1]) & (truth<=quantiles[2]))),
            'coverage95':float(np.mean((truth>=quantiles[0]) & (truth<=quantiles[3]))),
            'standardized_residual_std':float(z.std()),'standardized_residual_mean':float(z.mean()),
            'mean_posterior_cell_sd':float(sd.mean()),'truth_mean_correlation':corr(truth,mean),
            'fraction_draw_cells_delta_below_minus_one':float(np.mean(samples < -1))}


def main():
    start = time.perf_counter()
    plan_path = ROOT/'config/cf4_z3_field_definition_plan_v1.json'
    plan = json.loads(plan_path.read_text())
    posterior_program = json.loads((ROOT/plan['inputs']['posterior_program']).read_text())
    base = json.loads((ROOT/plan['inputs']['base_program']).read_text())
    args = fixed.frozen_args(base['input_bindings']['CF4_catalog']['path'])
    # frozen_args only constructs a namespace; no catalog is opened.
    transfer,growth = fixed.build_density_transfer(args)
    n,box = args.N,args.box_size
    assert n==32 and box==384 and base['grid']['truth_N']==64
    import jax.numpy as jnp
    from pmwd import Configuration,SimpleLCDM,boltzmann,linear_modes
    conf = Configuration(ptcl_spacing=box/n,ptcl_grid_shape=(n,)*3,mesh_shape=1,cosmo_dtype=jnp.float64,float_dtype=jnp.float64)
    cosmo = boltzmann(SimpleLCDM(conf,Omega_m=args.Om,Omega_b=args.Ob,h=args.h,A_s_1e9=args.A_s_1e9,n_s=args.ns),conf)
    white = np.random.default_rng(2026083000).standard_normal((n,)*3)
    direct = np.asarray(linear_modes(jnp.asarray(white),cosmo,conf,a=1.,real=True))
    inferred = apply_filter(white,transfer)
    normalization_error = float(np.linalg.norm(direct-inferred)/np.linalg.norm(direct))
    if normalization_error > 1e-10:
        raise ValueError(f'normalization mismatch {normalization_error}')
    freq = 2*np.pi*np.fft.fftfreq(n,d=box/n)
    kvec = np.meshgrid(freq,freq,freq,indexing='ij')
    kmag = np.sqrt(sum(k*k for k in kvec))
    fine_dx = box/64
    mask = np.ones((n,)*3)
    for axis in range(3):
        cut=[slice(None)]*3; cut[axis]=n//2; mask[tuple(cut)]=0
    cic = np.prod([np.sinc(k*fine_dx/(2*np.pi))**2 for k in kvec],axis=0)
    block = np.prod([np.cos(k*fine_dx/2) for k in kvec],axis=0)
    phase = np.exp(.5j*fine_dx*sum(kvec))
    windows={'raw':np.ones_like(mask),'non_Nyquist_only':mask,
             'CIC6_plus_block12_centered':mask*cic*block,
             'CIC6_plus_block12_native_phase':mask*cic*block*phase}
    # Exact band-limited interpolation and block average verify normalization/phase.
    modes=np.rint(np.fft.fftfreq(n)*n).astype(int)
    coarse_idx=np.flatnonzero(np.abs(modes)<n//2)
    fine_idx=modes[coarse_idx]%64
    coarse_fft=np.fft.fftn(inferred,norm='ortho')*cic
    fine_fft=np.zeros((64,)*3,dtype=complex)
    fine_fft[np.ix_(fine_idx,fine_idx,fine_idx)]=coarse_fft[np.ix_(coarse_idx,coarse_idx,coarse_idx)]*np.sqrt(8)
    explicit=phasec.block_sum(np.fft.ifftn(fine_fft,norm='ortho').real,n)/8
    analytic=apply_filter(inferred,windows['CIC6_plus_block12_native_phase'])
    block_error=float(np.max(np.abs(explicit-analytic)))
    if block_error > 1e-11:
        raise ValueError(f'block phase check failed {block_error}')
    edges=np.linspace(0,kmag.max()*(1+1e-12),10)
    rows=[]
    for task in posterior_program['assignments']:
        i,seed,arm=task['task_index'],task['seed'],task['arm']
        name=f'posterior_v1_{i:02d}_mock_{i:02d}_seed_{seed}_arm_{arm}'
        source=Path(plan['inputs']['posterior_root'])/name/'posterior_summary.npz'
        with np.load(source,allow_pickle=False) as data:
            truth=data['truth_coarse_density'].astype(float)
            truth_v=data['truth_coarse_velocity'].astype(float)
            mean_v=data['posterior_velocity_mean'].astype(float)
            q=data['posterior_density_quantiles']
            # Copy fixed subsample then release the larger compressed member.
            white_draws=data['posterior_white_thinned'][:,::4,:].copy().reshape(-1,n,n,n)
        full_coverage=float(np.mean((truth>=q[1]) & (truth<=q[2])))
        template=apply_filter(np.random.default_rng(seed).standard_normal((n,)*3),transfer)
        truth_fft=np.fft.fftn(truth,norm='ortho')
        variants={}
        for label,window in windows.items():
            samples=np.empty_like(white_draws,dtype=np.float32)
            for begin in range(0,len(samples),16):
                stop=begin+16
                modes=np.fft.fftn(white_draws[begin:stop].astype(float),axes=(1,2,3),norm='ortho')
                samples[begin:stop]=np.fft.ifftn(modes*(transfer*window)[None],axes=(1,2,3),norm='ortho').real
            detail=diagnose(truth,samples)
            filtered_template=apply_filter(template,window)
            detail['prior_expected_cell_sd']=float(np.sqrt(np.mean(np.abs(transfer*window)**2)))
            detail['same_seed_linear_truth_correlation']=corr(truth,filtered_template)
            detail['same_seed_linear_truth_residual_RMS']=float(np.sqrt(np.mean((truth-filtered_template)**2)))
            shells=[]
            for lo,hi in zip(edges[:-1],edges[1:]):
                use=(kmag>=lo)&(kmag<hi)&(kmag>0)
                if not use.any(): continue
                prior=float(np.mean(np.abs((transfer*window)[use])**2))
                power=float(np.mean(np.abs(truth_fft[use])**2))
                shells.append({'k_h_Mpc':float(kmag[use].mean()),'truth_to_prior_power':power/prior if prior>0 else None})
            detail['shells']=shells
            variants[label]=detail
            del samples
        row={'task':name,'source':str(source),'retained_diagnostic_draws':len(white_draws),
             'truth_cell_sd':float(truth.std()),'full_draw_recorded_coverage68':full_coverage,
             'truth_velocity_sd_km_s':[float(v.std()) for v in truth_v],
             'velocity_truth_posterior_mean_correlation':[corr(truth_v[c],mean_v[c]) for c in range(3)],
             'variants':variants}
        rows.append(row)
        print(json.dumps({'task':i,'truth_sd':row['truth_cell_sd'],'variants':{key:{k:v[k] for k in ('coverage68','standardized_residual_std','same_seed_linear_truth_correlation')} for key,v in variants.items()}}),flush=True)
        del white_draws
    result={'bundle':plan['bundle'],'implementation_commit':os.environ['EXPECTED_COMMIT'],
            'job_id':os.environ['SLURM_JOB_ID'],'normalization_relative_RMS':normalization_error,
            'explicit_block_vs_analytic_max_error':block_error,'growth_rate':growth,
            'rows':rows,'elapsed_seconds':time.perf_counter()-start,
            'peak_host_MiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            'limitations':['Not refitted posterior','CIC window neglects aliases/nonlinearity','No filter selected by coverage','Fixed subsample intervals have Monte Carlo error','No observed datum or new PM evolution']}
    output=Path(plan['output_root'])/f'result_{os.environ["SLURM_JOB_ID"]}.json'
    with output.open('x') as handle: json.dump(result,handle,indent=2,allow_nan=False)
    print('RESULT '+str(output),flush=True)


if __name__=='__main__': main()
