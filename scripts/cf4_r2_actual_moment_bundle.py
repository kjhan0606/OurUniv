"""Saved actual-data development state, shared moments and closure sensitivity.

No PM evolution, optimizer, chain, heldout score or calibration-prior claim.
All1414 training raw FP links and47121 training counts, sourceGL1 only.
"""
import json
import os
from pathlib import Path
import resource
import time
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_moment_target import MomentObservationTarget

BASE=Path('/gpfs/kjhan/CF4/z0_density')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':
        raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False)
    started=time.monotonic()
    report=dict(status='STARTED',job_id=os.environ['SLURM_JOB_ID'],
        source_commit=os.environ['CF4_EXPECTED_COMMIT'],R2_complete=False,
        PM_evolutions=0,optimizer_steps=0,heldout_scored=False,volume_order=1,
        closure_prior_used=False,results=[],
        limitations='Saved nonstationary N128/3 state, coarse GL1; FP subset only, not full CF4. Diagonal covariance, population/Ks/PM discrepancy uncalibrated. MW/M31 ambiguous,M33 unresolved.')
    def save():
        report.update(seconds=time.monotonic()-started,
            host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        _,_,_,_,source,mix,o,g=load_inputs()
        statepath=BASE/'r2_raw_joint_pilot_v1/accepted_present_state.npz'
        parent=json.loads((statepath.parent/'result.json').read_text())
        if parent['status']!='RAW_JOINT_TRANSITION_PILOT_NOT_POSTERIOR':
            raise ValueError('completed preserved same-state moment product required')
        with np.load(statepath,allow_pickle=False) as f:
            r,v,var=map(jnp.asarray,(f['rho'],f['mean_velocity_km_s'],f['physical_velocity_variance_km2_s2']))
            old=np.asarray(f['tracer']);p=jnp.asarray(f['population_white'])
        # This predecessor used alpha=-1+.06*exp(.5*u7). Preserve physical
        # alpha in the new broad linear coordinate; DO NOT reuse u7 unchanged.
        t=jnp.asarray(np.r_[old[:6],.12*np.exp(.5*old[7]),old[8]])
        if r.shape!=(128,)*3 or v.shape!=(3,128,128,128) or var.shape!=v.shape:
            raise ValueError('matching native N128 moment state required')
        if not all(bool(jnp.isfinite(a).all()) for a in (r,v,var,t,p)) or bool(jnp.any(var<0)):
            raise ValueError('finite nonnegative native variances required')
        source={k:jnp.asarray(source[k]) for k in ('positions','angular')}
        with np.load(BASE/'r2_sky_closed_split_v6/split.npz',allow_pickle=False) as f:
            keys,counts=map(jnp.asarray,(f['train_keys'],f['train_counts']))
            exposure,_=build_population_exposure_masks(128,f['heldout_flat_voxels'],
                f['train_window_excluded_keys'],f['heldout_window_excluded_keys'])
        target=MomentObservationTarget(128,source,mix['population'],o,g,keys,counts,jnp.asarray(exposure),
            volume_orders=(1,),source_chunk=8192,raw_block=256,cut_order=64)
        report.update(state=str(statepath),training_count=int(counts.sum()),raw_training_rows=len(o['x']),
            physical_alpha=float(-1+.5*t[6]),source_chunk=8192,
            sensitivity_design='Fixed core30km/s,fraction.5; broad scales .5,.25,1. Not a fit or independent calibration.',
            future_regularization='Possible proper Gaussian coordinates logcore/logscale/logitfraction. No width/centre certified; native calibration discrepancy must not be fixed to zero.')
        save()
        for scale in (.5,.25,1.):
            if time.monotonic()-started>1500:raise TimeoutError('25min application budget; no automatic extension')
            c=jnp.array([jnp.log(30.),jnp.log(scale),0.]);tic=time.monotonic()
            packs,info=target.support(r,v,var,t,c,1)
            report['phase']=f'JOINT_SCALE_{scale}';report['current_support']=info;save()
            compiled=target.value.lower(r,v,var,t,p,c,packs,source,o,1).compile()
            mem=compiled.memory_analysis();stats=jax.devices()[0].memory_stats() or {}
            estimate=stats.get('bytes_in_use',0)+mem.temp_size_in_bytes+mem.output_size_in_bytes
            if stats.get('bytes_limit') and 1.2*estimate>stats['bytes_limit']:
                raise MemoryError('joint value lacks20percent device margin')
            value,parts=compiled(r,v,var,t,p,c,packs,source,o)
            value=float(value);parts=np.asarray(parts)
            row=dict(dispersion_scale=scale,value=value if np.isfinite(value) else None,
                count_logscore=float(parts[0]) if np.isfinite(parts[0]) else None,
                raw_FP_logscore=float(parts[1]) if np.isfinite(parts[1]) else None,
                finite_joint=bool(np.isfinite(parts).all()),seconds=time.monotonic()-tic,
                device_estimated_peak_GiB=estimate/1024**3,support=info)
            report['results'].append(row);save();print(json.dumps(row),flush=True)
            del compiled
            if not row['finite_joint']:
                report['status']='ACTUAL_MOMENT_JOINT_SUPPORT_FAILURE_NOT_POSTERIOR';save();return
        report['status']='ACTUAL_MOMENT_JOINT_SENSITIVITY_NOT_POSTERIOR';save()
    except Exception as error:
        report.update(status='FAILED_ACTUAL_MOMENT_BUNDLE',error=repr(error));save();raise


if __name__=='__main__':main()
