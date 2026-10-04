"""Bounded untempered raw-population/LCDM joint transitions, not posterior UQ."""
import json
import os
from pathlib import Path
import resource
import time
import jax
import jax.numpy as jnp
import numpy as np
from cf4_r1_particle_forward import make_dynamics,particle_grid
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_raw_volume_target import FreshRawSupport,raw_field_logpdf,count_field_loglike
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_prior_split_hmc import FixedSplitMetric,inverse_laplacian_metric_symbol,PilotBudgetStop
from cf4_r2_corrected_split_hmc import corrected_split_step

BASE=Path('/gpfs/kjhan/CF4/z0_density');ROOT=Path(__file__).resolve().parents[1];N=128;BOX=384.


def padded_capacity(packs,quantum):
    """State-local membership with bucketed shapes, never frozen membership."""
    result=[]
    for pack in packs:
        size=len(pack['ids']);padding=(-size)%quantum
        result.append({k:jnp.pad(v,(0,padding),constant_values=False if k=='mask' else 0)
                       for k,v in pack.items()})
    return tuple(result)


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False);started=time.monotonic();cap=6600
    report=dict(status='STARTED',job_id=os.environ['SLURM_JOB_ID'],source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        R2_complete=False,heldout_scored=False,N=N,box_cMpc_h=BOX,dx_cMpc_h=3.,trace=[],evaluations=[],
        force_volume_order=2,fine_volume_order=4,cut_rule='axis1 order256, certified marginal tolerance1e-12',
        target='same-field counts and association-clean conditional linked-point rawFP/K marks; LCDM and24 proper nuisance priors once',
        limitations='development transitions only; no stationarity/UQ; conditional graph/type selection; MW/M31 ambiguous,M33 unresolved')
    def save():
        report.update(seconds=time.monotonic()-started,host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        required={'r2_raw_field_profile_v2_linked_point_radius':'NATIVE_RAW_TARGET_CHECKED_NOT_POSTERIOR',
                  'r2_volume_count_profile_v1':'FULL_COUNT_VOLUME_PROFILE_NOT_POSTERIOR'}
        for name,status in required.items():
            if json.loads((BASE/name/'result.json').read_text())['status']!=status:raise ValueError('native observation checks required')
        oldrho,oldvel,tracer,pop,source,mix,o,g=load_inputs()
        source_jax={k:jnp.asarray(v) for k,v in source.items()}
        with np.load(BASE/'r2_sky_closed_split_v6/split.npz',allow_pickle=False) as f:
            keys,counts=map(jnp.asarray,(f['train_keys'],f['train_counts']))
            exposure,_=build_population_exposure_masks(N,f['heldout_flat_voxels'],
                f['train_window_excluded_keys'],f['heldout_window_excluded_keys'])
        exposure=jnp.asarray(exposure)
        builders={order:FreshRawSupport(source['positions'],source['angular'],mix['population'],o,g,
            source_spacing=3.,volume_order=order) for order in (2,4)}
        centred=jax.jit(native_mass_momentum_to_count_cells,static_argnums=2)
        def support(r,v,t,order):
            _,cv=centred(r,v,BOX)
            packs,info=builders[order].build(jnp.moveaxis(cv,0,-1).reshape(-1,3),t)
            return padded_capacity(packs,65536 if order==2 else 262144),info
        def data(r,v,t,p,packs,source,o,order):
            density,cv=native_mass_momentum_to_count_cells(r,v,BOX)
            cv=jnp.moveaxis(cv,0,-1).reshape(-1,3)
            count=count_field_loglike(density,cv,t,source,g,keys,counts,exposure,
                source_spacing=3.,volume_order=order)
            raw=raw_field_logpdf(density,cv,t,p,packs,source,o,g,source_spacing=3.,volume_order=order,
                cut_order=256,cut_integration_axis=1,cut_marginal_tolerance=1e-12,
                source_conditioning_radius_cMpc_h=o['source_conditioning_radius_cMpc_h']).sum()
            return count+raw,jnp.array([count,raw])
        value=jax.jit(data,static_argnums=7)
        derivative=jax.jit(jax.value_and_grad(data,argnums=(0,1,2,3),has_aux=True),static_argnums=7)
        settings=json.loads((ROOT/'config/cf4_r1_particle_entry_v1.json').read_text())
        settings={k:settings[k] for k in ('cosmology','a_start','a_stop','a_nbody_maxstep')}
        settings.update(n=N,box_cMpc_h=BOX)
        common=json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())['common_cosmology']
        if settings['cosmology']['h']!=.746 or settings['cosmology']['Om']!=common['Omega_m']:
            raise ValueError('PM/observation cosmology mismatch')
        evolve,_,conf,_,pmass=make_dynamics(settings);mass=jnp.full(N**3,pmass)
        @jax.jit
        def field(white):
            pos,vel=evolve(white);state=particle_grid(pos,vel,mass,conf)
            return state['rho'],jnp.moveaxis(state['mean_velocity_km_s'],-1,0)
        @jax.jit
        def field_with_dispersion(white):
            # Read moments from the SAME forward needed for a fine energy;
            # do not evolve an accepted endpoint a second time for its plot.
            pos,vel=evolve(white);state=particle_grid(pos,vel,mass,conf)
            return (state['rho'],jnp.moveaxis(state['mean_velocity_km_s'],-1,0),
                jnp.moveaxis(state['variance_km2_s2'],-1,0),state['valid'])
        with np.load(BASE/'r2_prior_split_long_v1/final_state.npz',allow_pickle=False) as f:
            q=np.r_[f['white_ic'].ravel(),np.asarray(tracer),np.asarray(pop)]
        n=N**3
        def split(q):return tuple(map(jnp.asarray,(q[:n],q[n:n+9],q[n+9:])))
        def deadline():
            if time.monotonic()-started>cap-240:raise PilotBudgetStop()
        latest_fine={}
        def oracle(q,order,gradient):
            nonlocal latest_fine
            deadline();tic=time.monotonic();white,t,p=split(q)
            if gradient:(r,v),pullback=jax.vjp(field,white)
            elif order==4:r,v,variance,valid=field_with_dispersion(white)
            else:r,v=field(white)
            packs,info=support(r,v,t,order)
            if gradient:
                if 'joint_device_memory' not in report:
                    compiled=derivative.lower(r,v,t,p,packs,source_jax,o,order).compile()
                    memory=compiled.memory_analysis();stats=jax.devices()[0].memory_stats() or {}
                    peak=stats.get('bytes_in_use',0)+memory.temp_size_in_bytes+memory.output_size_in_bytes
                    report['joint_device_memory']=dict(estimated_peak_GiB=peak/1024**3,
                        limit_GiB=stats.get('bytes_limit',0)/1024**3);save()
                    if stats.get('bytes_limit') and 1.2*peak>stats['bytes_limit']:
                        raise MemoryError('PM/observation adjoint lacks20percent device margin')
                    del compiled
                (score,parts),grads=derivative(r,v,t,p,packs,source_jax,o,order)
                ic,=pullback((grads[0],grads[1]))
                grad=q-np.r_[np.asarray(ic),np.asarray(grads[2]),np.asarray(grads[3])]
                if not np.isfinite(grad).all():raise FloatingPointError('nonfinite full PM raw/count derivative')
            else:(score,parts),grad=value(r,v,t,p,packs,source_jax,o,order),None
            energy=.5*float(q@q)-float(score)
            if not np.isfinite(energy):raise FloatingPointError('nonfinite full target, no floor')
            if order==4 and not gradient:
                latest_fine=dict(rho=np.asarray(r),mean_velocity_km_s=np.asarray(v),
                    physical_velocity_variance_km2_s2=np.asarray(variance),velocity_valid=np.asarray(valid))
                if not all(np.isfinite(x).all() for x in latest_fine.values()):
                    raise FloatingPointError('nonfinite accepted-state moment candidate')
            entry=dict(order=order,gradient=gradient,energy=energy,parts=np.asarray(parts).tolist(),
                seconds=time.monotonic()-tic,**info)
            report['evaluations'].append(entry);save();print(json.dumps(entry),flush=True)
            return energy,grad
        report['phase']='SAME_IC_FIELD_RECONSTRUCTION';save()
        r,v=field(jnp.asarray(q[:n]))
        report['restart_field_max_abs']=dict(rho=float(jnp.max(jnp.abs(r-oldrho))),velocity=float(jnp.max(jnp.abs(v-oldvel))))
        save()
        if max(report['restart_field_max_abs'].values())>1e-7:raise AssertionError('same saved IC/PM field mismatch')
        coarse,gradient=oracle(q,2,True)
        # Value-only vs differentiated primal at EXACTLY the same state/rule.
        direct,_=oracle(q,2,False)
        if abs(coarse-direct)>1e-7:raise AssertionError('native PM target primal changed under differentiation')
        # One joint IC/population/tracer directional control, not prior-only.
        probe_rng=np.random.default_rng(2026092914)
        wave=np.broadcast_to(np.cos(2*np.pi*np.arange(N)/N)[:,None,None],(N,)*3)
        direction=np.r_[.03*wave.ravel(),.01*probe_rng.normal(size=24)/np.sqrt(24)]
        epsilon=2e-5
        plus=oracle(q+epsilon*direction,2,False)[0]
        minus=oracle(q-epsilon*direction,2,False)[0]
        prior=float(q@direction);ad=float(gradient@direction)-prior
        fd=(plus-minus)/(2*epsilon)-prior
        error=abs(ad-fd)/max(1.,abs(ad),abs(fd))
        report['joint_PM_direction']=dict(likelihood_AD=ad,likelihood_FD=fd,relative_error=error);save()
        if error>2e-3:raise AssertionError('joint PM/raw/count derivative mismatch')
        energy,_=oracle(q,4,False)
        accepted_field=latest_fine
        metric=FixedSplitMetric(inverse_laplacian_metric_symbol(N),np.eye(24)*1e-5)
        report['metric']='fixed inverse mass: IC inverse-laplacian6000; nuisance1e-5; proposal guess, not covariance'
        rng=np.random.default_rng(2026092913);step=.1
        np.savez(out/'accepted_checkpoint.npz',canonical=q,fine_energy=energy,force_energy=coarse,force_gradient=gradient)
        report.update(phase='BOUNDED_CORRECTED_TRANSITIONS',initial_fine_energy=energy,initial_force_energy=coarse,
            initial_white_mean_square=float(np.mean(q[:n]**2)));save()
        for i in range(8):
            before=q.copy()
            try:
                q,energy,coarse,gradient,info=corrected_split_step(lambda x:oracle(x,2,True),
                    lambda x:oracle(x,4,False)[0],metric,q,energy,coarse,gradient,rng,step=step,steps=2)
            except PilotBudgetStop:break
            if info['accepted']:accepted_field=latest_fine
            row=dict(iteration=i+1,warmup=i<4,step_size=step,fine_energy=energy,force_energy=coarse,
                white_mean_square=float(np.mean(q[:n]**2)),canonical_jump_rms=float(np.sqrt(np.mean((q-before)**2))),
                **{k:(str(v) if isinstance(v,float) and not np.isfinite(v) else v) for k,v in info.items()})
            report['trace'].append(row);report['rng_state']=rng.bit_generator.state
            np.savez(out/'accepted_checkpoint.npz',canonical=q,fine_energy=energy,force_energy=coarse,force_gradient=gradient)
            save();print(json.dumps(row),flush=True)
            if i<4:step=float(np.clip(step*np.exp((np.exp(info['log_acceptance'])-.65)/np.sqrt(i+1)),1e-6,.3))
        report.update(status='RAW_JOINT_TRANSITION_PILOT_NOT_POSTERIOR',final_fine_energy=energy,
            completed_proposals=len(report['trace']),final_white_mean_square=float(np.mean(q[:n]**2)))
        np.savez(out/'accepted_present_state.npz',**accepted_field,white_ic=q[:n],
            tracer=q[n:n+9],population_white=q[n+9:],fine_energy=energy,box_cMpc_h=BOX,
            native_mesh_origin_fraction=0.,R2_complete=False)
        report['present_state_readout']='same accepted fine-energy forward; physical dispersion is NOT posterior uncertainty or tracer LOS nuisance'
        save()
    except Exception as error:
        report.update(status='FAILED_RAW_JOINT_PILOT',error=repr(error));save();raise


if __name__=='__main__':main()
