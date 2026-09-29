"""Bounded N256 transitions or two fixed-anchor force trials, not posterior UQ."""
import json
import os
from pathlib import Path
import resource
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy import fft
from cf4_r1_particle_forward import make_dynamics,particle_grid
from cf4_lg_highk_conditional_field import restrict_spectrum_preserve_dtype
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_resolution_target import source_geometry_at_resolution,ResolutionObservationTarget
from cf4_r2_prior_split_hmc import FixedSplitMetric,inverse_laplacian_metric_symbol,PilotBudgetStop
from cf4_r2_corrected_split_hmc import corrected_split_step
from cf4_r2_affine_force import AffineCorrectedForce
from cf4_r2_chain_schedule import ChainSchedule
from cf4_r2_posterior_moments import PresentMomentAccumulator

BASE=Path('/gpfs/kjhan/CF4/z0_density');ROOT=Path(__file__).resolve().parents[1]
N=256;BOX=384.;NIC=N**3


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend()!='gpu':raise RuntimeError('Slurm GPU required')
    out=Path(os.environ['CF4_R2_OUT_DIR']);out.mkdir(exist_ok=False);started=time.monotonic()
    affine=os.environ.get('CF4_R2_AFFINE_FORCE')=='1'
    long_chain=os.environ.get('CF4_R2_LONG_CHAIN')=='1'
    if long_chain and not affine:raise ValueError('long-chain mode requires checked affine force')
    schedule=ChainSchedule() if long_chain else None
    maximum=schedule.proposals if long_chain else (2 if affine else 4)
    warmup=schedule.warmup if long_chain else (0 if affine else 2)
    budget=23*3600 if long_chain else 6300
    report=dict(status='STARTED',job_id=os.environ['SLURM_JOB_ID'],source_commit=os.environ['CF4_EXPECTED_COMMIT'],
        N=N,box_cMpc_h=BOX,dx_cMpc_h=1.5,observed_count_grid=128,R2_complete=False,heldout_scored=False,
        force_volume_order=1,fine_volume_order=2,source_cell_rate_factor=.125,trace=[],evaluations=[],
        target='conditional1414 rawFP/K marks plus47121 training counts, LCDM and24 proper nuisance priors once',
        max_proposals=maximum,warmup_proposals=warmup,frozen_affine_force=affine,
        long_chain=long_chain,application_budget_seconds=budget,
        sampling_status='NOT_ASSESSED_FOR_STATIONARITY',
        limitations='bounded development transitions, no stationarity/UQ; fine2 needs fine4 sensitivity; conditional selection; MW/M31 ambiguous,M33 unresolved')
    def save():
        report.update(seconds=time.monotonic()-started,host_peak_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        (out/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        parent=Path(os.environ.get('CF4_R2_INITIAL_DIR',str(BASE/'r2_n256_dynamics_profile_v1')))
        if json.loads((parent/'result.json').read_text())['status']!='N256_DYNAMICS_INITIALIZER_NOT_POSTERIOR':
            raise ValueError('completed actual N256 initializer required')
        if json.loads((BASE/'r2_n256_source_profile_v3/result.json').read_text())['status']!='N256_SOURCE_WORKSPACE_PROFILE_NOT_N256_FIELD':
            raise ValueError('completed large-source count resource profile required')
        _,_,_,_,source,mix,o,g=load_inputs()
        source=source_geometry_at_resolution(source,N)
        with np.load(BASE/'r2_sky_closed_split_v6/split.npz',allow_pickle=False) as f:
            keys,counts=map(jnp.asarray,(f['train_keys'],f['train_counts']))
            exposure,_=build_population_exposure_masks(128,f['heldout_flat_voxels'],
                f['train_window_excluded_keys'],f['heldout_window_excluded_keys'])
        obs=ResolutionObservationTarget(N,source,mix,o,g,keys,counts,jnp.asarray(exposure),force_order=1,fine_order=2)
        restart=Path(os.environ['CF4_R2_RESTART_DIR']) if os.environ.get('CF4_R2_RESTART_DIR') else None
        startfile=parent/'initial_present_state.npz'
        if restart is not None:
            previous=json.loads((restart/'result.json').read_text())
            if previous['status']!='N256_RAW_JOINT_TRANSITION_PILOT_NOT_POSTERIOR' or previous['fine_volume_order']!=2 or previous['N']!=N:
                raise ValueError('completed same-target N256 restart required')
            startfile=restart/'accepted_present_state.npz'
        with np.load(startfile,allow_pickle=False) as f:
            q=np.r_[f['white_ic'],f['tracer'],f['population_white']]
            oldrho=f['rho'];oldvel=f['mean_velocity_km_s']
            restart_fine=float(f['fine_energy']) if 'fine_energy' in f else None
        if q.shape!=(NIC+24,):raise ValueError('matching canonical IC and24 nuisances required')
        report['initializer']=str(startfile);save()
        settings=json.loads((parent/'result.json').read_text())['settings']
        evolve,_,conf,_,pmass=make_dynamics(settings);mass=jnp.full(NIC,pmass)
        @jax.jit
        def field(white):
            pos,vel=evolve(white);state=particle_grid(pos,vel,mass,conf)
            return state['rho'],jnp.moveaxis(state['mean_velocity_km_s'],-1,0)
        @jax.jit
        def moments(white):
            pos,vel=evolve(white);state=particle_grid(pos,vel,mass,conf)
            return state['rho'],jnp.moveaxis(state['mean_velocity_km_s'],-1,0),jnp.moveaxis(state['variance_km2_s2'],-1,0),state['valid']
        latest_fine={}
        def oracle(q,order,gradient):
            nonlocal latest_fine
            if time.monotonic()-started>budget:raise PilotBudgetStop()
            tic=time.monotonic();white,t,p=map(jnp.asarray,(q[:NIC],q[NIC:NIC+9],q[NIC+9:]))
            if gradient:(r,v),pullback=jax.vjp(field,white)
            elif order==2:r,v,variance,valid=moments(white)
            else:r,v=field(white)
            packs,info=obs.support(r,v,t,order)
            if gradient:
                memory_key='joint_device_memory' if order==1 else 'fine_gradient_device_memory'
                if memory_key not in report:
                    compiled=obs.derivative.lower(r,v,t,p,packs,source,o,order).compile()
                    memory=compiled.memory_analysis();stats=jax.devices()[0].memory_stats() or {}
                    peak=stats.get('bytes_in_use',0)+memory.temp_size_in_bytes+memory.output_size_in_bytes
                    report[memory_key]=dict(estimated_peak_GiB=peak/1024**3,
                        limit_GiB=stats.get('bytes_limit',0)/1024**3);save()
                    if stats.get('bytes_limit') and 1.2*peak>stats['bytes_limit']:
                        raise MemoryError('N256 joint adjoint lacks20percent device margin')
                    del compiled
                (score,parts),grads=obs.derivative(r,v,t,p,packs,source,o,order)
                ic,=pullback((grads[0],grads[1]))
                grad=q-np.r_[np.asarray(ic),np.asarray(grads[2]),np.asarray(grads[3])]
                if not np.isfinite(grad).all():raise FloatingPointError('nonfinite joint derivative')
            else:
                if order==2 and 'fine_device_memory' not in report:
                    compiled=obs.value.lower(r,v,t,p,packs,source,o,order).compile()
                    memory=compiled.memory_analysis();stats=jax.devices()[0].memory_stats() or {}
                    peak=stats.get('bytes_in_use',0)+memory.temp_size_in_bytes+memory.output_size_in_bytes
                    report['fine_device_memory']=dict(estimated_peak_GiB=peak/1024**3,
                        limit_GiB=stats.get('bytes_limit',0)/1024**3);save()
                    if stats.get('bytes_limit') and 1.2*peak>stats['bytes_limit']:
                        raise MemoryError('N256 fine target lacks20percent device margin')
                    del compiled
                (score,parts),grad=obs.value(r,v,t,p,packs,source,o,order),None
            energy=.5*float(q@q)-float(score)
            if not np.isfinite(energy):raise FloatingPointError('nonfinite target, no floor')
            if order==2 and not gradient:
                latest_fine=dict(rho=np.asarray(r),mean_velocity_km_s=np.asarray(v),
                    physical_velocity_variance_km2_s2=np.asarray(variance),velocity_valid=np.asarray(valid))
                if not all(np.isfinite(x).all() for x in latest_fine.values()):raise FloatingPointError('nonfinite moments')
                latest_fine['count_raw_scores']=np.asarray(parts)
            row=dict(order=order,gradient=gradient,energy=energy,parts=np.asarray(parts).tolist(),seconds=time.monotonic()-tic,**info)
            report['evaluations'].append(row);save();print(json.dumps(row),flush=True)
            return energy,grad
        report['phase']='N256_SAME_IC_RECONSTRUCTION';save()
        r,v=field(jnp.asarray(q[:NIC]))
        errors=dict(rho=float(jnp.max(jnp.abs(r-oldrho))),velocity=float(jnp.max(jnp.abs(v-oldvel))))
        report['restart_field_max_abs']=errors;save()
        if max(errors.values())>1e-7:raise AssertionError('N256 initializer reconstruction mismatch')
        del oldrho,oldvel,r,v
        coarse,gradient=oracle(q,1,True)
        direct,_=oracle(q,1,False)
        if abs(coarse-direct)>1e-7:raise AssertionError('N256 AD/value primal mismatch')
        seed=int(os.environ.get('CF4_R2_CHAIN_SEED',str(2026092917 if affine else 2026092916)))
        report['seed']=seed;rng=np.random.default_rng(seed)
        wave=np.broadcast_to(np.cos(2*np.pi*np.arange(N)/N)[:,None,None],(N,)*3)
        direction=np.r_[.03*wave.ravel(),.01*rng.normal(size=24)/np.sqrt(24)]
        epsilon=2e-5;prior=float(q@direction)
        plus=oracle(q+epsilon*direction,1,False)[0];minus=oracle(q-epsilon*direction,1,False)[0]
        ad=float(gradient@direction)-prior;fd=(plus-minus)/(2*epsilon)-prior
        error=abs(ad-fd)/max(1.,abs(ad),abs(fd))
        report['joint_PM_direction']=dict(likelihood_AD=ad,likelihood_FD=fd,relative_error=error);save()
        if error>2e-3:raise AssertionError('N256 PM/raw/count directional mismatch')
        del direction,wave
        energy,_=oracle(q,2,False);accepted_field=latest_fine
        if restart_fine is not None and abs(energy-restart_fine)>1e-7:
            raise AssertionError('same-target restart fine energy changed')
        force=lambda x:oracle(x,1,True)
        if affine:
            report['phase']='FROZEN_AFFINE_REFERENCE_GRADIENT';save()
            reference_energy,reference_gradient=oracle(q,2,True)
            difference=abs(reference_energy-energy)
            if difference>1e-7:raise AssertionError('fine gradient/value primal mismatch')
            force=AffineCorrectedForce(force,q,reference_gradient-gradient)
            gradient=reference_gradient
            report.update(fine_gradient_value_abs_error=difference,
                affine_correction_norm=float(np.linalg.norm(force.correction)),
                force_policy='anchor and correction fixed for entire run; fine Metropolis target unchanged')
            np.savez(out/'fixed_force_anchor.npz',anchor=force.anchor,gradient_correction=force.correction)
            save()
        metric=FixedSplitMetric(inverse_laplacian_metric_symbol(N),np.eye(24)*1e-5)
        step=schedule.initial_step if long_chain else (.05220457782250625 if affine else .1)
        accumulator=PresentMomentAccumulator((N,)*3) if long_chain else None
        report['schedule']=schedule.__dict__ if schedule else None
        def white_summary(q):
            spectrum=fft.fftn(q[:NIC].reshape((N,)*3),norm='ortho',workers=2)
            coarse_spectrum=restrict_spectrum_preserve_dtype(spectrum,128)
            return dict(white_mean_square=float(np.mean(q[:NIC]**2)),
                inherited_low_white_mean_square=float(np.mean(np.abs(coarse_spectrum)**2)),
                fundamental_real=[float(spectrum[index].real) for index in ((1,0,0),(0,1,0),(0,0,1))],
                fundamental_imag=[float(spectrum[index].imag) for index in ((1,0,0),(0,1,0),(0,0,1))])
        report.update(phase='BOUNDED_TRANSITIONS',initial_fine_energy=energy,initial_force_energy=coarse,
            initial_white=white_summary(q))
        def checkpoint():
            np.savez(out/'accepted_checkpoint.npz',canonical=q,fine_energy=energy,force_energy=coarse,force_gradient=gradient)
        checkpoint();save()
        for i in range(maximum):
            before=q.copy()
            integrations=schedule.steps(i) if long_chain else 2
            try:
                q,energy,coarse,gradient,info=corrected_split_step(force,
                    lambda x:oracle(x,2,False)[0],metric,q,energy,coarse,gradient,rng,step=step,steps=integrations)
            except PilotBudgetStop:
                report['stopped_at_application_budget']=True
                break
            if info['accepted']:accepted_field=latest_fine
            row=dict(iteration=i+1,warmup=i<warmup,step_size=step,integration_steps=integrations,
                fine_energy=energy,force_energy=coarse,
                nuisance_white=q[NIC:].tolist(),force_linear_offset=force.offset(q) if affine else 0.,**white_summary(q),
                count_raw_scores=accepted_field['count_raw_scores'].tolist(),
                density_octant_means=[float(accepted_field['rho'][a:a+N//2,b:b+N//2,c:c+N//2].mean())
                    for a in (0,N//2) for b in (0,N//2) for c in (0,N//2)],
                canonical_jump_rms=float(np.sqrt(np.mean((q-before)**2))),
                **{k:(str(v) if isinstance(v,float) and not np.isfinite(v) else v) for k,v in info.items()})
            if long_chain and i>=warmup:
                # Every retained chain state counts, including repeated rejections.
                accumulator.update(accepted_field)
                report['retained_states']=accumulator.n
                if accumulator.n%8==0:
                    np.savez(out/f'draw_{accumulator.n:04d}.npz',canonical=q,
                        rho=accepted_field['rho'].astype(np.float32),
                        mean_velocity_km_s=accepted_field['mean_velocity_km_s'].astype(np.float32),
                        velocity_valid=accepted_field['velocity_valid'],fine_energy=energy,
                        iteration=i+1,native_mesh_origin_fraction=0.,R2_complete=False)
                    np.savez(out/'moment_checkpoint.npz',**accumulator.checkpoint_arrays())
            report['trace'].append(row);report['rng_state']=rng.bit_generator.state
            checkpoint();save();print(json.dumps(row),flush=True)
            if long_chain:step=schedule.next_step(step,i,info['log_acceptance'])
            elif i<warmup:step=float(np.clip(step*np.exp((np.exp(info['log_acceptance'])-.65)/np.sqrt(i+1)),1e-6,.3))
        if long_chain and accumulator.n:
            np.savez(out/'moment_checkpoint.npz',**accumulator.checkpoint_arrays())
            np.savez(out/'present_moments_unassessed.npz',**accumulator.arrays())
        np.savez(out/'accepted_present_state.npz',**accepted_field,white_ic=q[:NIC],tracer=q[NIC:NIC+9],
            population_white=q[NIC+9:],fine_energy=energy,box_cMpc_h=BOX,native_mesh_origin_fraction=0.,R2_complete=False)
        report.update(status=('N256_CHAIN_FINISHED_REQUIRES_DIAGNOSTICS' if long_chain else
            'N256_RAW_JOINT_TRANSITION_PILOT_NOT_POSTERIOR'),completed_proposals=len(report['trace']),
            final_white=white_summary(q),final_fine_energy=energy);save()
    except Exception as error:
        report.update(status='FAILED_N256_RAW_JOINT_PILOT',error=repr(error));save();raise


if __name__=='__main__':main()
