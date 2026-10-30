"""Bounded live IC + source-mark nuisance transitions; not a converged R2 map."""
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
from cf4_r2_joint_calibration import group_scores
from cf4_r2_live_field_marks import bind_state_geometry, live_group_scores, joint_white_logdensity
from cf4_chunked_hmc import make_chunks, checked_record

BASE = Path('/gpfs/kjhan/CF4/z0_density')
GEOMETRY = BASE/'r2_live_field_geometry_v1'
OUT = BASE/'r2_live_field_pilot_v1'
N,BOX = 128,384.


def geometry(nodes):
    with np.load(GEOMETRY/f'geometry_q{nodes}.npz') as f:
        data = {k:jnp.asarray(f[k]) for k in f.files if k not in ('method_names','group_labels')}
        names = list(map(str,f['method_names']))
    if 'log_distance_weight' in data or 'redshift_logkernel' in data:
        raise ValueError('live geometry must not contain frozen state-dependent arrays')
    return data,names


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    start = time.monotonic()
    g,names = geometry(257)
    size = N**3
    sd = jnp.asarray([.004]+[1.]*len(names)+[2.])
    dim = size+len(sd)
    mask = np.asarray(g['group_holdout'])
    base = json.loads((ROOT/'config/cf4_r1_particle_entry_v1.json').read_text())
    settings = {k:base[k] for k in ('cosmology','a_start','a_stop','a_nbody_maxstep')}
    settings.update(n=N,box_cMpc_h=BOX)
    common = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())['common_cosmology']
    if settings['cosmology']['h'] != common['h'] or settings['cosmology']['Om'] != common['Omega_m']:
        raise ValueError('PM/observation cosmology mismatch')
    settings_hmc = dict(initial_step_size=.01,maximum_step_size=.05,target_acceptance=.8,
        divergence_threshold=1000.,integration_steps=6,integration_steps_range=[4,8])
    initial_seeds = [2026092501,2026092702]
    report = dict(classification='LIVE_FIELD_SOURCE_MARK_TRANSITION_PILOT',
        job_id=os.environ['SLURM_JOB_ID'],status='STARTED',grid=N,box_cMpc_h=BOX,
        dx_cMpc_h=BOX/N,PM_mesh_origin_fraction=0.,settings=settings,
        source_FP_rows=len(g['row_group']),nonFP_rows=len(g['anchor_group']),
        train_groups=int((~mask).sum()),heldout_groups=int(mask.sum()),
        method_names=names,nuisance_prior_sd=np.asarray(sd).tolist(),
        original_data_used_once=True,previous_fitted_offsets_or_covariance_used=False,
        includes_inclusive_counts=False,includes_BGc=False,group_selection_calibrated=False,
        source_fit_covariance_resolved=False,converged_posterior=False,R2_delivery=False,
        LG_identified=False,initial_seeds=initial_seeds,sampler=settings_hmc,
        chains=[],saved_states=[],full_chain_positions_archived=False,
        summary_columns=['log_target','white_IC_mean_square','white0','white1','white2','white3']
            +['FP_zero_white']+[f'{m}_offset_white' for m in names]+['kappa_white'],
        geometry_manifest_sha256=hashlib.sha256((GEOMETRY/'manifest.json').read_bytes()).hexdigest(),
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),ROOT/'src/cf4_r2_live_field_marks.py',
                      ROOT/'src/cf4_r2_joint_calibration.py',ROOT/'src/cf4_r1_particle_forward.py',
                      ROOT/'src/cf4_r2_fp_group_marginal.py',ROOT/'src/cf4_r2_fp_distance.py',
                      ROOT/'src/cf4_z0_physical_field.py',
                      ROOT/'src/cf4_chunked_hmc.py')})
    traces = {}
    def write():
        report['runtime_seconds'] = time.monotonic()-start
        report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (OUT/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        if traces:
            np.savez_compressed(OUT/'transition_summaries.npz',**traces)
    def deadline():
        if time.monotonic()-start > 900:
            raise TimeoutError('bounded application budget; no automatic extension')
    try:
        with np.load(BASE/'r2_pm128_unconditional_v1/state.npz') as f:
            rho,v = jnp.asarray(f['rho'],dtype=jnp.float64),jnp.asarray(f['velocity_km_s'],dtype=jnp.float64)
        theta0 = jnp.zeros(len(sd))
        compare = jax.jit(lambda r,v,o: group_scores(theta0,
            bind_state_geometry(r,v,g,origin_fraction=o)))
        legacy = np.asarray(compare(rho,v,.5))
        corrected = np.asarray(compare(rho,v,0.))
        with np.load(BASE/'r2_joint_calibration_cache_v1/geometry_q257.npz') as f:
            old = {k:jnp.asarray(f[k]) for k in f.files if k not in ('method_names','group_labels')}
        np.testing.assert_array_equal(old['group_holdout'],g['group_holdout'])
        old_scores = np.asarray(jax.jit(group_scores)(theta0,old))
        np.testing.assert_allclose(legacy,old_scores,rtol=1e-9,atol=1e-7)
        report['coordinate_correction'] = dict(legacy_cache_max_group_difference=float(np.max(abs(legacy-old_scores))),
            corrected_minus_legacy_train_logfactor=float((corrected-legacy)[~mask].sum()),
            corrected_minus_legacy_heldout_logfactor=float((corrected-legacy)[mask].sum()),
            absolute_group_logfactor_change_p50_p90_p99=np.quantile(abs(corrected-legacy),[.5,.9,.99]).tolist(),
            sample_shift_per_axis_cMpc_h=BOX/N/2,gravity_kernel_changed=False)
        print(json.dumps(report['coordinate_correction']),flush=True)
        del old,old_scores,rho,v
        evolve,_initial,conf,_cosmo,particle_mass = make_dynamics(settings)
        mass = jnp.full((size,),particle_mass)
        def fields(vector):
            pos,vel = evolve(vector[:size])
            return particle_grid(pos,vel,mass,conf)
        def target(vector):
            field = fields(vector)
            return joint_white_logdensity(vector[:size],vector[size:],field['rho'],
                jnp.moveaxis(field['mean_velocity_km_s'],-1,0),g,sd)
        score = jax.jit(target)
        value_grad = jax.jit(jax.value_and_grad(target))
        field_fn = jax.jit(fields)
        score_field = jax.jit(lambda r,v,p: live_group_scores(r,v,p,g))
        def save_state(vector,label):
            state = field_fn(vector)
            rho = np.asarray(state['rho'])
            velocity = np.asarray(jnp.moveaxis(state['mean_velocity_km_s'],-1,0))
            variance = np.asarray(jnp.moveaxis(state['variance_km2_s2'],-1,0))
            if not all(np.isfinite(x).all() for x in (rho,velocity,variance)) or abs(rho.mean()-1) > 1e-8:
                raise FloatingPointError('invalid physical field readout')
            parameters = np.asarray(vector[size:]*sd)
            scores = np.asarray(score_field(jnp.asarray(rho),jnp.asarray(velocity),jnp.asarray(parameters)))
            np.savez_compressed(OUT/f'{label}.npz',white_ic=np.asarray(vector[:size]),
                white_nuisance=np.asarray(vector[size:]),parameters=parameters,
                rho=rho.astype(np.float32),mean_velocity_km_s=velocity.astype(np.float32),
                physical_velocity_variance_km2_s2=variance.astype(np.float32),
                valid=np.asarray(state['valid']),PM_mesh_origin_fraction=0.,box_cMpc_h=BOX)
            row = dict(label=label,train_conditional_logfactor=float(scores[~mask].sum()),
                heldout_conditional_logfactor=float(scores[mask].sum()),
                white_IC_RMS=float(np.sqrt(np.mean(np.asarray(vector[:size])**2))),
                rho_mean=float(rho.mean()),rho_rms=float(np.std(rho)),
                velocity_rms_km_s=float(np.sqrt(np.mean(velocity**2))),
                parameters=parameters.tolist())
            report['saved_states'].append(row)
            write()
            return row
        def starting_vector(seed):
            return jnp.concatenate((jnp.asarray(np.random.default_rng(seed).standard_normal(size)),jnp.zeros(len(sd))))
        first = starting_vector(initial_seeds[0])
        print('Compiling live N128 source-mark IC/nuisance adjoint',flush=True)
        began = time.monotonic()
        val,grad = value_grad(first)
        grad.block_until_ready()
        report['first_joint_gradient_seconds'] = time.monotonic()-began
        if not np.isfinite(float(val)) or not np.isfinite(np.asarray(grad)).all():
            raise FloatingPointError('nonfinite initial joint target/gradient')
        direction = np.random.default_rng(2026092703).standard_normal(dim)
        direction[:size] /= np.sqrt(np.mean(direction[:size]**2))
        direction = jnp.asarray(direction)
        tangent = float(jnp.vdot(grad,direction))
        prior_tangent = -float(jnp.vdot(first,direction))
        mark_tangent = tangent-prior_tangent
        fd = []
        for eps in (2e-5,1e-5):
            finite = float((score(first+eps*direction)-score(first-eps*direction))/(2*eps))
            mark_finite = finite-prior_tangent  # quadratic prior's central difference is exact
            fd.append(dict(epsilon=eps,total_finite_difference=finite,mark_finite_difference=mark_finite,
                mark_absolute_error=abs(mark_finite-mark_tangent),
                mark_scaled_error=abs(mark_finite-mark_tangent)/max(1.,abs(mark_finite),abs(mark_tangent))))
        report['initial_joint_derivative'] = dict(total_autodiff=tangent,analytic_prior=prior_tangent,
            mark_autodiff=mark_tangent,finite_differences=fd)
        write()
        if fd[-1]['mark_scaled_error'] > .02:
            report['status'] = 'STOP_JOINT_ADJOINT_CHECK'
            return
        save_state(first,'initial_chain0')
        initialize,warm,sample,final = make_chunks(target,dim,settings_hmc,record_steps=True)
        for chain,seed in enumerate(initial_seeds):
            deadline()
            vector = first if chain == 0 else starting_vector(seed)
            initial_vector = np.asarray(vector)
            state,adaptation = initialize(vector)
            key = jax.random.fold_in(jax.random.PRNGKey(2026092710),chain)
            warm_key,sample_key = jax.random.split(key)
            keysets = [jax.random.split(warm_key,32),jax.random.split(sample_key,32)]
            rows,accept,diverge,steps,counts,phases = [],[],[],[],[],[]
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
                    x = a[0]
                    # Scalar summaries only; avoid retaining a2GiB full chain.
                    summary = np.column_stack((a[1],np.mean(x[:,:size]**2,axis=1),x[:,:4],x[:,size:]))
                    rows.append(summary); accept.append(a[2]); diverge.append(a[3]); steps.append(a[6]); counts.append(a[7])
                    phases.append(np.full(4,phase,dtype=np.int8))
                    row['retained_completed' if phase else 'warmup_completed'] = offset+4
                    traces.update({f'chain{chain}_{name}':np.concatenate(values)
                        for name,values in (('summary',rows),('acceptance',accept),('divergent',diverge),
                                            ('step',steps),('leapfrog_steps',counts),('phase',phases))})
                    write()
                    print(f'chain{chain} phase{phase} {offset+4}/32; seconds={time.monotonic()-start:.1f}',flush=True)
                    if phase and offset+4 in (16,32):
                        save_state(state.position,f'chain{chain}_retained{offset+4}')
            retained = traces[f'chain{chain}_phase'] == 1
            row.update(mean_retained_acceptance=float(traces[f'chain{chain}_acceptance'][retained].mean()),
                retained_divergences=int(traces[f'chain{chain}_divergent'][retained].sum()),
                IC_RMS_change_from_start=float(np.sqrt(np.mean((np.asarray(state.position[:size])-initial_vector[:size])**2))))
            write()
        report['status'] = 'COMPLETE_SHORT_TRANSITION_PILOT_NOT_EQUILIBRATED'
        # One fixed endpoint check, no score-selected state or quadrature sweep.
        fine,fine_names = geometry(513)
        if fine_names != names:
            raise ValueError('fine method order mismatch')
        end = field_fn(state.position)
        p = state.position[size:]*sd
        vv = jnp.moveaxis(end['mean_velocity_km_s'],-1,0)
        coarse_score = np.asarray(score_field(end['rho'],vv,p))
        fine_score = np.asarray(jax.jit(lambda r,v,p:live_group_scores(r,v,p,fine))(end['rho'],vv,p))
        report['fixed_endpoint_Q513_minus_Q257'] = dict(train=float((fine_score-coarse_score)[~mask].sum()),
            heldout=float((fine_score-coarse_score)[mask].sum()),whole_chain_bound=False)
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
