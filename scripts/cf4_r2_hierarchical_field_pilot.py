"""Bounded live-field hierarchical partial model; not a calibrated R2 map."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_hierarchical_marks import (decode_hyper, hierarchical_group_scores,
    joint_logdensity, marginal_group_scores)
from cf4_chunked_hmc import make_chunks, checked_record

BASE = Path('/gpfs/kjhan/CF4/z0_density')
GEOMETRY = BASE/'r2_hierarchical_field_geometry_v1'
OUT = BASE/'r2_hierarchical_field_pilot_v1'
N,BOX = 128,384.


def load_geometry(nodes):
    with np.load(GEOMETRY/f'geometry_q{nodes}.npz') as f:
        data = {k:jnp.asarray(f[k]) for k in f.files if k not in ('method_names','group_labels')}
        names = list(map(str,f['method_names']))
    if 'log_distance_weight' in data or 'redshift_logkernel' in data:
        raise ValueError('live geometry must not cache state-dependent values')
    np.testing.assert_array_equal(data['train_group_index'],np.flatnonzero(~np.asarray(data['group_holdout'])))
    return data,names


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    started = time.monotonic()
    g,names = load_geometry(257)
    size = N**3
    sd = jnp.asarray([.004]+[1.]*len(names)+[2.])
    nh = len(sd)+2
    ng = len(g['train_group_index'])
    dim = size+nh+ng
    hold = np.asarray(g['group_holdout'])
    base = json.loads((ROOT/'config/cf4_r1_particle_entry_v1.json').read_text())
    settings = {k:base[k] for k in ('cosmology','a_start','a_stop','a_nbody_maxstep')}
    settings.update(n=N,box_cMpc_h=BOX)
    common = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())['common_cosmology']
    if settings['cosmology']['h'] != common['h'] or settings['cosmology']['Om'] != common['Omega_m']:
        raise ValueError('PM/observation cosmology mismatch')
    hmc = dict(initial_step_size=.01,maximum_step_size=.05,target_acceptance=.8,
        divergence_threshold=1000.,integration_steps=6,integration_steps_range=[4,8])
    seeds = [2026092501,2026092702]
    report = dict(classification='LIVE_FIELD_HIERARCHICAL_CONDITIONAL_MARK_PILOT',
        status='STARTED',job_id=os.environ['SLURM_JOB_ID'],grid=N,box_cMpc_h=BOX,
        dx_cMpc_h=BOX/N,PM_mesh_origin_fraction=0.,settings=settings,sampler=hmc,
        prior=dict(calibration_sd=np.asarray(sd).tolist(),selected_bias_log_sd=.5,
                   excess_group_tau_median_dex=.02,excess_group_tau_log_sd=.7,
                   externally_calibrated=False,previous_same_data_fit_used=False),
        train_groups=ng,heldout_groups=int(hold.sum()),hyper_parameters=nh,
        source_FP_rows=len(g['row_group']),nonFP_rows=len(g['anchor_group']),method_names=names,
        conditioned_unique_2mpp_members=int(np.asarray(g['twompp_member_count']).sum()),
        groups_with_2mpp_members=int(np.count_nonzero(np.asarray(g['twompp_member_count']))),
        count_likelihood_included=False,BGc_included=False,original_marks_used_once=True,
        selected_group_inclusion_identified=False,source_fit_covariance_resolved=False,
        independent_holdout_validation=False,converged_posterior=False,R2_delivery=False,
        MW_M31_M33_identified=False,initial_IC_seeds=seeds,chains=[],saved_states=[],
        summary_columns=['log_target','white_IC_mean_square','white0','white1','white2','white3']
            +['FP_zero_white']+[f'{m}_offset_white' for m in names]+['kappa_white',
              'log_selected_bias_white','log_tau_white','group_white_mean','group_white_mean_square'],
        geometry_manifest_sha256=hashlib.sha256((GEOMETRY/'manifest.json').read_bytes()).hexdigest(),
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),ROOT/'src/cf4_r2_hierarchical_marks.py',
                ROOT/'src/cf4_r2_live_field_marks.py',ROOT/'src/cf4_r1_particle_forward.py',
                ROOT/'src/cf4_r2_fp_group_marginal.py',ROOT/'src/cf4_r2_fp_distance.py',
                ROOT/'src/cf4_z0_physical_field.py',ROOT/'src/cf4_chunked_hmc.py')})
    traces = {}
    def write():
        report['runtime_seconds'] = time.monotonic()-started
        report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if traces:
            np.savez_compressed(OUT/'transition_summaries.npz',**traces)
    def deadline():
        if time.monotonic()-started > 1200:
            raise TimeoutError('bounded application budget; no automatic extension')
    try:
        evolve,_initial,conf,_cosmo,particle_mass = make_dynamics(settings)
        mass = jnp.full((size,),particle_mass)
        split = lambda x: (x[:size],x[size:size+nh],x[size+nh:])
        def fields(vector):
            pos,vel = evolve(vector[:size])
            return particle_grid(pos,vel,mass,conf)
        def target_with_kernel(vector,sufficient):
            field = fields(vector)
            return joint_logdensity(*split(vector),field['rho'],
                jnp.moveaxis(field['mean_velocity_km_s'],-1,0),
                dict(g,redshift_sufficient=sufficient),sd)
        target = lambda vector: target_with_kernel(vector,g['redshift_sufficient'])
        value_grad = jax.jit(jax.value_and_grad(target_with_kernel))
        score = jax.jit(target)
        field_fn = jax.jit(fields)
        score_field = jax.jit(lambda r,v,h,u:hierarchical_group_scores(r,v,h,u,g,sd))
        def starting_vector(seed,chain):
            ic = np.random.default_rng(seed).standard_normal(size)
            u = np.random.default_rng(2026092720+chain).standard_normal(ng)
            return jnp.asarray(np.r_[ic,np.zeros(nh),u])
        def save_state(vector,label):
            state = field_fn(vector)
            ic,h,u = split(vector)
            rho = np.asarray(state['rho'])
            vel = np.asarray(jnp.moveaxis(state['mean_velocity_km_s'],-1,0))
            variance = np.asarray(jnp.moveaxis(state['variance_km2_s2'],-1,0))
            if not all(np.isfinite(x).all() for x in (rho,vel,variance)) or abs(rho.mean()-1)>1e-8:
                raise FloatingPointError('invalid live physical field')
            p,b,tau = decode_hyper(h,sd)
            scores = np.asarray(score_field(jnp.asarray(rho),jnp.asarray(vel),h,u))
            np.savez_compressed(OUT/f'{label}.npz',white_ic=np.asarray(ic),white_hyper=np.asarray(h),
                white_training_group=np.asarray(u),train_group_index=np.asarray(g['train_group_index']),
                calibration_parameters=np.asarray(p),selected_bias=float(b),tau_dex=float(tau),
                rho=rho.astype(np.float32),mean_velocity_km_s=vel.astype(np.float32),
                physical_velocity_variance_km2_s2=variance.astype(np.float32),
                valid=np.asarray(state['valid']),PM_mesh_origin_fraction=0.,box_cMpc_h=BOX)
            report['saved_states'].append(dict(label=label,training_mark_logfactor=float(scores[~hold].sum()),
                calibration_parameters=np.asarray(p).tolist(),selected_bias=float(b),tau_dex=float(tau),
                group_offset_RMS_dex=float(tau*jnp.sqrt(jnp.mean(u*u))),
                white_IC_RMS=float(jnp.sqrt(jnp.mean(ic*ic))),rho_mean=float(rho.mean()),
                rho_rms=float(rho.std()),velocity_rms_km_s=float(np.sqrt(np.mean(vel*vel)))))
            write()
        first = starting_vector(seeds[0],0)
        began = time.monotonic()
        value,grad = value_grad(first,g['redshift_sufficient'])
        grad.block_until_ready()
        report['first_joint_gradient_seconds'] = time.monotonic()-began
        if not np.isfinite(float(value)) or not np.isfinite(np.asarray(grad)).all():
            raise FloatingPointError('invalid initial target/gradient')
        direction = np.random.default_rng(2026092723).standard_normal(dim)
        direction[:size] /= np.sqrt(np.mean(direction[:size]**2))
        direction = jnp.asarray(direction)
        tangent = float(jnp.vdot(grad,direction))
        prior_tangent = -float(jnp.vdot(first,direction))
        mark_tangent = tangent-prior_tangent
        fd = []
        for eps in (2e-5,1e-5):
            finite = float((score(first+eps*direction)-score(first-eps*direction))/(2*eps))-prior_tangent
            fd.append(dict(epsilon=eps,mark_finite_difference=finite,
                mark_absolute_error=abs(finite-mark_tangent),
                mark_scaled_error=abs(finite-mark_tangent)/max(1.,abs(finite),abs(mark_tangent))))
        report['initial_joint_derivative'] = dict(total_autodiff=tangent,analytic_prior=prior_tangent,
                                                  mark_autodiff=mark_tangent,finite_differences=fd)
        value_off,grad_off = value_grad(first,g['group_only_redshift_sufficient'])
        delta = grad[:size]-grad_off[:size]
        mark_grad = grad[:size]+first[:size]
        report['conditional_2mpp_influence'] = dict(full_minus_group_only_logfactor=float(value-value_off),
            IC_gradient_difference_L2=float(jnp.linalg.norm(delta)),
            full_IC_mark_gradient_L2=float(jnp.linalg.norm(mark_grad)),
            fraction_of_full_IC_mark_gradient=float(jnp.linalg.norm(delta)/jnp.linalg.norm(mark_grad)),
            independent_information_gain=False,comparison_includes_no_count_factor=True)
        del grad_off,delta
        write()
        print(json.dumps(report['initial_joint_derivative']),flush=True)
        print(json.dumps(report['conditional_2mpp_influence']),flush=True)
        if fd[-1]['mark_scaled_error'] > .02:
            report['status'] = 'STOP_JOINT_ADJOINT_CHECK'
            return
        save_state(first,'initial_chain0')
        initialize,warm,sample,final = make_chunks(target,dim,hmc,record_steps=True)
        for chain,seed in enumerate(seeds):
            deadline()
            vector = first if chain == 0 else starting_vector(seed,chain)
            initial_ic = np.asarray(vector[:size])
            state,adaptation = initialize(vector)
            key = jax.random.fold_in(jax.random.PRNGKey(2026092725),chain)
            kw,ks = jax.random.split(key)
            keysets = [jax.random.split(kw,32),jax.random.split(ks,32)]
            buffers = {k:[] for k in ('summary','acceptance','divergent','step','leapfrog_steps','phase')}
            row = dict(chain=chain,initial_seed=seed,warmup_completed=0,retained_completed=0)
            report['chains'].append(row)
            for phase in range(2):
                if phase:
                    step = final(adaptation)
                    row['sampling_step'] = float(step)
                for offset in range(0,32,4):
                    deadline()
                    keys = keysets[phase][offset:offset+4]
                    if phase:
                        state,record = sample(state,step,keys)
                    else:
                        (state,adaptation),record = warm(state,adaptation,keys)
                    a = checked_record(record,state)
                    x,u = a[0],a[0][:,size+nh:]
                    summary = np.column_stack((a[1],np.mean(x[:,:size]**2,axis=1),x[:,:4],
                        x[:,size:size+nh],u.mean(axis=1),np.mean(u*u,axis=1)))
                    for name,val in zip(buffers,(summary,a[2],a[3],a[6],a[7],np.full(4,phase,dtype=np.int8))):
                        buffers[name].append(val)
                    traces.update({f'chain{chain}_{name}':np.concatenate(v) for name,v in buffers.items()})
                    row['retained_completed' if phase else 'warmup_completed'] = offset+4
                    write()
                    print(f'chain{chain} phase{phase} {offset+4}/32; seconds={time.monotonic()-started:.1f}',flush=True)
            retained = traces[f'chain{chain}_phase'] == 1
            row.update(mean_retained_acceptance=float(traces[f'chain{chain}_acceptance'][retained].mean()),
                retained_divergences=int(traces[f'chain{chain}_divergent'][retained].sum()),
                IC_RMS_change_from_start=float(np.sqrt(np.mean((np.asarray(state.position[:size])-initial_ic)**2))))
            save_state(state.position,f'chain{chain}_retained32')
        deadline()
        fine,fine_names = load_geometry(513)
        if names != fine_names:
            raise ValueError('method order mismatch')
        end = field_fn(state.position)
        _,h,u = split(state.position)
        r,v = end['rho'],jnp.moveaxis(end['mean_velocity_km_s'],-1,0)
        coarse_scores = np.asarray(score_field(r,v,h,u))
        fine_scores = np.asarray(jax.jit(lambda r,v,h,u:hierarchical_group_scores(r,v,h,u,fine,sd))(r,v,h,u))
        report['fixed_endpoint_Q513_minus_Q257_training_logfactor'] = float((fine_scores-coarse_scores)[~hold].sum())
        held = []
        for order in (9,17):
            nodes,weights = np.polynomial.hermite.hermgauss(order)
            pred = np.asarray(jax.jit(lambda r,v,h:marginal_group_scores(r,v,h,fine,sd,
                jnp.asarray(np.sqrt(2)*nodes),jnp.log(jnp.asarray(weights))))(r,v,h))
            held.append(pred[hold])
        correction = float((held[1]-held[0]).sum())
        report['heldout_new_group_offsets'] = dict(Q513_GH9_logfactor=float(held[0].sum()),
            Q513_GH17_logfactor=float(held[1].sum()),GH17_minus_GH9=correction,
            max_group_absolute_difference=float(np.max(abs(held[1]-held[0]))),
            integration_stable_at_checked_orders=bool(abs(correction)<.05 and np.max(abs(held[1]-held[0]))<.05),
            posterior_predictive=False,independent_validation=False)
        np.savez_compressed(OUT/'endpoint_factor_checks.npz',Q513_minus_Q257=fine_scores-coarse_scores,
                            heldout_GH9=held[0],heldout_GH17=held[1],group_holdout=hold)
        report['status'] = 'COMPLETE_HIERARCHICAL_TRANSITION_PILOT_NOT_EQUILIBRATED'
    except TimeoutError as exc:
        report.update(status='INCOMPLETE_BUDGET',error=str(exc))
    except Exception as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write()
        print(json.dumps(report,indent=2,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
