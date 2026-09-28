"""Bounded training-only N128 MAP attempt; NOT a calibrated R2 posterior.

The host refreshes shifted-source support for every objective evaluation,
including line-search trials. PM and observation VJPs are composed explicitly
so the discrete approximate support never masquerades as a differentiable map.
No heldout counts or eta values are loaded/scored by this executable.
"""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_linked_singleton_target import partial_v6_count_singleton_parts
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_2mpp_joint_likelihood_jax import observer_centred_spherical_rsd_jax
from cf4_2mpp_joint_likelihood_jax import _gaussian_hermite_rule
from cf4_r2_linked_fp_sparse_train import load_train_singletons, FP, SOURCE

BASE = Path('/gpfs/kjhan/CF4/z0_density')
SPLIT = BASE/'r2_sky_closed_split_v6/split.npz'
N, BOX, WIDTH = 128, 384., 8192
OUT = Path(os.environ.get('CF4_R2_OUT_DIR', str(BASE/'r2_v6_partial_map_v2')))


def conditional_rate_mode(expected_count,observed_count,white_rate):
    """One exact scalar MAP update; no marginalization or prior change.

    The rate multiplies every count intensity by exp(2*t); it cancels from
    the conditional FP factor. The Gaussian prior is -.5*t**2.
    """
    if not (np.isfinite(expected_count) and expected_count>0
            and np.isfinite(observed_count) and observed_count>0
            and np.isfinite(white_rate)):
        raise ValueError('finite positive counts required for conditional rate mode')
    poisson_mode=white_rate+.5*(np.log(observed_count)-np.log(expected_count))
    if poisson_mode==0.:
        return 0.
    derivative=lambda t:2*observed_count*np.expm1(2*(t-poisson_mode))+t
    return brentq(derivative,min(0.,poisson_mode),max(0.,poisson_mode),xtol=1e-13)


def finite_gradient_curvature(fun,point,directions,*,epsilons,seconds_left,record):
    """Finite differences of full gradients, NOT exact/autodiff Hessians.

    Positive curvatures in these few directions do not prove a positive
    definite Hessian or a valid Laplace posterior. Preserve scale dependence.
    """
    for name,direction in directions.items():
        previous=None
        for epsilon in epsilons:
            if epsilon<=0 or not np.isfinite(epsilon):
                raise ValueError('positive finite curvature displacement required')
            gradients=[]
            start=time.monotonic()
            for sign in (1.,-1.):
                if seconds_left()<=0:
                    raise TimeoutError('curvature feasibility time budget')
                value,gradient=fun(point+sign*epsilon*direction)
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('curvature trial outside differentiable support')
                gradients.append(gradient)
            hv=(gradients[0]-gradients[1])/(2*epsilon)
            if not np.isfinite(hv).all():
                raise FloatingPointError('nonfinite finite-difference Hessian action')
            row=dict(direction=name,epsilon=float(epsilon),
                directional_curvature=float(direction@hv),HVP_norm=float(np.linalg.norm(hv)),
                seconds=time.monotonic()-start)
            if previous is not None:
                row['relative_HVP_change_from_larger_step']=float(np.linalg.norm(hv-previous)
                    /max(1.,np.linalg.norm(hv),np.linalg.norm(previous)))
            previous=hv
            record(row)


def bounded_lbfgs(fun, initial, callback, *, n_ic, seconds_left, maxiter=128,
                  value_only=None, initial_norm_cap=None,initial_evaluation=None):
    """Descent-only L-BFGS with bounded trial steps, not parameter bounds.

    An infinite objective means a genuine zero-probability trial and is
    rejected, never replaced by a likelihood floor. A finite objective with
    nonfinite derivative is an implementation/nondifferentiability failure;
    the objective callable must stop rather than mask that case.
    """
    x=initial.copy()
    if n_ic < 0 or x.shape != (n_ic+10,):
        raise ValueError('expected IC block plus nine tracer and one zero coordinates')
    value,grad=fun(x) if initial_evaluation is None else initial_evaluation
    if not np.isfinite(value) or not np.isfinite(grad).all():
        raise FloatingPointError('restart is not a finite differentiable state')
    history=[]
    norm_cap=initial_norm_cap
    message='iteration limit'
    for iteration in range(maxiter):
        if np.max(np.abs(grad))<1e-4:
            message='gradient tolerance'; break
        if seconds_left()<=0:
            message='application time budget'; break
        q=grad.copy()
        alphas=[]
        for s,y,inverse in reversed(history):
            a=inverse*np.dot(s,q); alphas.append(a); q-=a*y
        scale=(np.dot(history[-1][0],history[-1][1])/np.dot(history[-1][1],history[-1][1])
               if history else 1.)
        direction=scale*q
        for (s,y,inverse),a in zip(history,reversed(alphas)):
            direction+=s*(a-inverse*np.dot(y,direction))
        direction=-direction
        if np.dot(direction,grad)>=0:
            history=[]; direction=-grad
        # Coordinate limits affect trial size ONLY; the target/prior unchanged.
        ic_step=(np.linalg.norm(direction[:n_ic])/np.sqrt(n_ic)/.1 if n_ic else 0.)
        divisor=max(1.,ic_step,
                    np.max(np.abs(direction[n_ic:n_ic+9]/100.))/.1,
                    abs(direction[-1])/.5)
        direction/=divisor
        if norm_cap is not None:
            direction/=max(1.,np.linalg.norm(direction)/norm_cap)
        slope=np.dot(grad,direction)
        accepted=False
        step=1.
        for trial in range(40):
            if seconds_left()<=0:
                message='application time budget'; break
            candidate=x+step*direction
            if value_only is None:
                fv,fg=fun(candidate)
            else:
                fv=value_only(candidate)
            if np.isfinite(fv) and fv<=value+1e-4*step*slope:
                if value_only is not None:
                    checked,fg=fun(candidate)
                    if not np.isclose(checked,fv,rtol=1e-12,atol=1e-7):
                        raise FloatingPointError('accepted score-only/full objective mismatch')
                    fv=checked
                    if fv>value+1e-4*step*slope:
                        step*=.5
                        continue
                accepted=True; break
            # Safeguarded quadratic interpolation, never accept predicted gain.
            curvature=fv-value-step*slope
            proposal=(-slope*step*step/(2*curvature)
                      if np.isfinite(curvature) and curvature>0 else .5*step)
            step=float(np.clip(proposal,.1*step,.5*step))
        if not accepted:
            if seconds_left()>0:
                message='no finite sufficient-decrease trial'
            break
        s,y=candidate-x,fg-grad
        if norm_cap is not None:
            norm_cap=max(np.finfo(float).eps, np.linalg.norm(s)*(4. if trial==0 else 2.))
        sy=np.dot(s,y)
        if sy>1e-10*np.linalg.norm(s)*np.linalg.norm(y):
            history.append((s.copy(),y.copy(),1./sy)); history=history[-8:]
        x,value,grad=candidate,fv,fg
        callback(x)
    return x,value,grad,message


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    expected = os.environ['CF4_EXPECTED_COMMIT']
    subprocess.run(['git', 'diff', '--exit-code', expected, '--', 'src',
                    'scripts/cf4_r2_v6_partial_map.py',
                    'scripts/cf4_r2_linked_fp_sparse_train.py'], cwd=ROOT, check=True)
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    count_integration=os.environ.get('CF4_R2_COUNT_INTEGRATION','gh')
    cdf_order=int(os.environ.get('CF4_R2_CDF_ORDER','4'))
    cdf_segments=int(os.environ.get('CF4_R2_CDF_SEGMENTS','32'))
    if count_integration not in ('gh','shell_cdf') or min(cdf_order,cdf_segments)<1:
        raise ValueError('invalid count integration choice')
    maxiter=int(os.environ.get('CF4_R2_MAXITER','128'))
    cap=int(os.environ.get('CF4_R2_SECONDS_CAP','2700'))
    norm_cap=os.environ.get('CF4_R2_INITIAL_NORM_CAP')
    norm_cap=None if norm_cap is None else float(norm_cap)
    nuisance_only=os.environ.get('CF4_R2_NUISANCE_ONLY')=='1'
    if nuisance_only and (not os.environ.get('CF4_R2_RESTART') or any(
            os.environ.get(k)=='1' for k in ('CF4_R2_DIRECTION_DIAG',
            'CF4_R2_CURVATURE_DIAG','CF4_R2_RATE_WARM_START'))):
        raise ValueError('conditional block needs a restart and no other diagnostic/warm-start mode')
    if maxiter<1 or cap<1 or (norm_cap is not None and (not np.isfinite(norm_cap) or norm_cap<=0)):
        raise ValueError('invalid bounded fit budget/initial step norm')
    report = dict(status='STARTED', source_commit=expected,
        job_id=os.environ['SLURM_JOB_ID'], N=N, box_cMpc_h=BOX,
        classification='TRAINING_ONLY_PARTIAL_MAP_ATTEMPT_NOT_POSTERIOR',
        heldout_counts_scored=0, heldout_FP_marks_scored=0, R2_complete=False,
        N256=False, source_membership='strict ungrouped only; grouped FP excluded',
        association='constant given observed point; uncalibrated',
        covariance='one shared FP zero, prior SD .004 dex; other source-fit covariance missing',
        support='periodic shifted-position KD tree rebuilt every evaluation; 8-sigma truncation',
        quadrature_order=15, radial_window_cMpc_h=[5.,180.],
        limitations=['MAP is not posterior uncertainty; no calibrated selection/bias/FoG',
                     'continuous mark and GH15 count approximation differ near selection edges',
                     'MW/M31 roles ambiguous and M33 unresolved on this NEW coarse state; '
                     'LG observables must constrain the same field in R3; no truth IDs'],
        maxiter=maxiter,application_seconds_cap=cap,
        initial_step_norm_cap=norm_cap,score_only_line_search=True,trace=[])
    report['count_integration']=dict(family=count_integration,gh_order=15,
                                    cdf_order=cdf_order,cdf_segments=cdf_segments)
    if nuisance_only:
        report.update(classification='CONDITIONAL_NUISANCE_OPTIMIZATION_FIXED_FIELD_NOT_POSTERIOR',
            optimization_block='ten nuisances at fixed IC/field',field_changed=False,
            nuisance_trace=[],mandatory_full_gradient_after_block=True)
    if count_integration=='shell_cdf':
        report['quadrature_order']=None
        report['limitations'][1]='finite count integration and continuous mark differ; periodic-mark aliases remain uncalibrated'

    def save_report():
        report['elapsed_seconds'] = time.monotonic()-started
        report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (OUT/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')

    save_report()
    try:
        # Select only training data. The geometry identifies the heldout region,
        # but not its measured counts or distance marks.
        with np.load(SPLIT, allow_pickle=False) as f:
            split = {k:f[k].copy() for k in ('train_keys','train_counts',
                'heldout_flat_voxels','train_window_excluded_keys','heldout_window_excluded_keys')}
        exposure, _ = build_population_exposure_masks(N, split['heldout_flat_voxels'],
            split['train_window_excluded_keys'], split['heldout_window_excluded_keys'])
        if not exposure[split['train_keys']].all() or split['train_counts'].sum() != 47121:
            raise ValueError('frozen v6 training graph mismatch')
        options, point, fp = load_train_singletons(SPLIT)
        with np.load(FP, allow_pickle=False) as f:
            membership = f['membership_state'].astype(str)
            source_labels = f['source_group'].astype(str)
        if any(source_labels[o[3]] != o[0] for o in options):
            raise ValueError('source membership and mark row alignment changed')
        allowed = {'source_ungrouped_catalogue_present', 'source_ungrouped_catalogue_absent'}
        options = [o for o in options if membership[o[3]] in allowed]
        if len(options) != 429:
            raise ValueError(f'expected 429 strict singleton links, got {len(options)}')
        with np.load(SOURCE, allow_pickle=False) as f:
            source = {k:jnp.asarray(f[k]) for k in f.files}
        keys, counts, exposure = map(jnp.asarray,
                                    (split['train_keys'],split['train_counts'],exposure))
        common = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())['common_cosmology']
        base = json.loads((ROOT/'config/cf4_r1_particle_entry_v1.json').read_text())
        settings = {k:base[k] for k in ('cosmology','a_start','a_stop','a_nbody_maxstep')}
        settings.update(n=N, box_cMpc_h=BOX)
        if settings['cosmology']['h'] != common['h'] or settings['cosmology']['Om'] != common['Omega_m']:
            raise ValueError('cosmology mismatch')
        evolve, _, conf, _, particle_mass = make_dynamics(settings)
        mass = jnp.full(N**3, particle_mass)

        @jax.jit
        def field(white):
            pos, vel = evolve(white)
            result = particle_grid(pos, vel, mass, conf)
            return result['rho'], jnp.moveaxis(result['mean_velocity_km_s'], -1, 0)

        @jax.jit
        def terminal_field(white):
            # Replace the existing final forward/readout, not an extra simulation.
            # Physical particle dispersion is NOT posterior uncertainty or the
            # phenomenological tracer sigma_los nuisance in the likelihood.
            pos,vel=evolve(white)
            result=particle_grid(pos,vel,mass,conf)
            return (result['rho'],jnp.moveaxis(result['mean_velocity_km_s'],-1,0),
                    jnp.moveaxis(result['variance_km2_s2'],-1,0),result['valid'])

        @jax.jit
        def shifted_positions(rho, vel):
            _, velocity = native_mass_momentum_to_count_cells(rho, vel, BOX)
            velocity = jnp.moveaxis(velocity, 0, -1).reshape(-1,3)
            return observer_centred_spherical_rsd_jax(source['positions'],velocity,
                jnp.full(3,BOX/2),BOX,common['H0_km_s_Mpc'],little_h=common['h'],
                scale_factor=1.)[0]

        selected = {p:[o for o in options if point['population'][o[2]] == p] for p in range(6)}
        centers = {}
        metadata = {}
        for p, batch in selected.items():
            voxels = np.array([np.unravel_index(int(point['flat_cell'][o[2]]),(N,)*3)
                               for o in batch], dtype=np.int32).reshape(-1,3)
            centers[p] = (voxels+.5)*(BOX/N)
            fi = np.array([o[4] for o in batch],dtype=np.int32)
            metadata[p] = dict(voxel_ijk=jnp.asarray(voxels),
                observed_radius_cMpc_h=jnp.array([point['radius_cMpc_h'][o[2]] for o in batch]),
                **{k:jnp.asarray(fp[k][fi]) for k in ('dz_row','eta_mean','eta_std','eta_alpha')})

        def build_support(rho, vel, tracer):
            sigma = 100*np.exp(.5*tracer[6])
            radius = 8*common['h']*sigma/common['H0_km_s_Mpc'] + np.sqrt(3)*1.5*BOX/N
            if not np.isfinite(radius) or radius >= BOX/2:
                raise ValueError('LOS support exceeds bounded implementation; do not clip nuisance')
            shifted = np.asarray(shifted_positions(rho,vel)) % BOX
            tree = cKDTree(shifted, boxsize=BOX)
            links, max_neighbors = {}, 0
            for p, batch in selected.items():
                neighborhoods = tree.query_ball_point(centers[p],radius,workers=1)
                ids = np.zeros((len(batch),WIDTH),dtype=np.int32)
                active = np.zeros_like(ids,dtype=bool)
                for i, neighbors in enumerate(neighborhoods):
                    size = len(neighbors)
                    if not 0 < size <= WIDTH:
                        raise ValueError(f'support capacity {size} outside [1,{WIDTH}]; no truncation')
                    ids[i,:] = neighbors[0]
                    ids[i,:size] = neighbors
                    active[i,:size] = True
                    max_neighbors = max(max_neighbors,size)
                links[p] = dict(metadata[p],candidate_source_ids=jnp.asarray(ids),
                    candidate_mask=jnp.asarray(active),
                    association_logprob=jnp.zeros((len(batch),5,WIDTH)))
            return links,max_neighbors

        def target_parts(rho,vel,tracer,zero,links):
            return partial_v6_count_singleton_parts(rho,vel,jnp.zeros(0),tracer,
                source,links,keys,counts,exposure,box=BOX,
                hubble=common['H0_km_s_Mpc'],h=common['h'],white_fp_zero=zero,
                count_integration=count_integration,count_cdf_order=cdf_order,
                count_cdf_segments=cdf_segments)

        @jax.jit
        def data_target(rho,vel,tracer,zero,links):
            parts,_ = target_parts(rho,vel,tracer,zero,links)
            return jnp.sum(parts[:3]),parts[:3]
        data_vg = jax.jit(jax.value_and_grad(data_target,argnums=(0,1,2,3),has_aux=True))
        report.update(training_count_keys=int(keys.size),training_counts=int(counts.sum()),
                      training_singletons=len(options),IC_seed=2026092702,
                      nuisance_precondition_scale=100.)
        start_scale=float(os.environ.get('CF4_R2_IC_START_SCALE','1.'))
        if not np.isfinite(start_scale) or start_scale<=0:
            raise ValueError('positive finite optimizer starting scale required')
        initial = np.r_[start_scale*np.random.default_rng(2026092702).standard_normal(N**3),np.zeros(10)]
        report.update(IC_start_scale=start_scale,initialization=(
            'prior draw' if start_scale==1 else 'small-perturbation optimizer start, NOT a prior draw; prior unchanged'))
        restart=os.environ.get('CF4_R2_RESTART')
        if restart:
            with np.load(restart,allow_pickle=False) as f:
                initial=f['parameters'].copy()
            if initial.shape!=(N**3+10,) or not np.isfinite(initial).all():
                raise ValueError('invalid accepted restart coordinates')
            report['restart']=restart
            report['initialization']='accepted checkpoint'
            report['restart_policy']='same target/accepted state; fresh L-BFGS history, not exact optimizer continuation'
        latest = {}
        diagnostic_baseline = {}
        data_executable=None
        def score_only(x):
            tracer=jnp.asarray(x[N**3:N**3+9]/100.)
            rho,vel=field(jnp.asarray(x[:N**3]))
            links,width=build_support(rho,vel,np.asarray(tracer))
            data,parts=data_target(rho,vel,tracer,jnp.asarray(x[-1]),links)
            value=-float(data)+.5*np.dot(x[:N**3],x[:N**3])
            if np.isnan(value) or value==-np.inf:
                raise FloatingPointError('undefined diagnostic/trial score; no floor')
            return value,np.asarray(parts),(rho,vel),width

        def trial_value(x):
            value,parts,_,width=score_only(x)
            report['score_only_trials']=report.get('score_only_trials',0)+1
            print(json.dumps(dict(score_trial=report['score_only_trials'],
                objective=float(value) if np.isfinite(value) else str(value),
                max_neighbors=width)),flush=True)
            return value

        def objective(x):
            nonlocal data_executable
            tic = time.monotonic()
            white = jnp.asarray(x[:N**3])
            tracer = jnp.asarray(x[N**3:N**3+9]/100.)
            zero = jnp.asarray(x[-1])
            (rho,vel), pullback = jax.vjp(field,white)
            links,width = build_support(rho,vel,np.asarray(tracer))
            if data_executable is None:
                data_executable=data_vg.lower(rho,vel,tracer,zero,links).compile()
                analysis=data_executable.memory_analysis()
                stats=jax.devices()[0].memory_stats() or {}
                if analysis is not None:
                    report['data_derivative_memory']=dict(
                        temporary_GiB=analysis.temp_size_in_bytes/1024**3,
                        current_device_GiB=stats.get('bytes_in_use',0)/1024**3,
                        device_limit_GiB=stats.get('bytes_limit',0)/1024**3)
                    save_report()
                    peak=stats.get('bytes_in_use',0)+analysis.temp_size_in_bytes+analysis.output_size_in_bytes
                    if stats.get('bytes_limit') and 1.2*peak>stats['bytes_limit']:
                        raise MemoryError('combined derivative lacks 20 percent device-memory margin')
            (value,parts), grads = data_executable(rho,vel,tracer,zero,links)
            icgrad, = pullback((grads[0],grads[1]))
            value = float(value-.5*jnp.vdot(white,white))
            gradient = np.r_[np.asarray(icgrad-white),np.asarray(grads[2])/100.,float(grads[3])]
            if not np.isfinite(value) or not np.isfinite(gradient).all():
                bad=dict(parts=[float(v) if np.isfinite(v) else str(v) for v in np.asarray(parts)],
                    objective_finite=bool(np.isfinite(value)),
                    nonfinite_derivative_counts=[int(np.count_nonzero(~np.isfinite(np.asarray(g)))) for g in grads],
                    nonfinite_IC_derivatives=int(np.count_nonzero(~np.isfinite(gradient[:N**3]))),
                    max_neighbors=width)
                if not report.get('nonfinite_trials'):
                    np.savez(OUT/'first_nonfinite_trial.npz',parameters=x,
                             rho=np.asarray(rho),velocity_km_s=np.asarray(vel))
                report.setdefault('nonfinite_trials',[]).append(bad)
                save_report()
                if np.isfinite(value):
                    raise FloatingPointError('finite target but nonfinite derivative; preserved exact trial')
                if np.isnan(value) or value==np.inf:
                    raise FloatingPointError('undefined or positive-infinite log target; preserved exact trial')
                # Exact zero support: let line search shrink, no model floor.
                return np.inf,np.zeros_like(x)
            latest.update(parts=list(map(float,np.asarray(parts))),objective=-value,
                          gradient_inf=float(np.max(np.abs(gradient))),
                          gradient_inf_IC=float(np.max(np.abs(gradient[:N**3]))),
                          gradient_l2_IC=float(np.linalg.norm(gradient[:N**3])),
                          gradient_l2_nuisance_optimizer=float(np.linalg.norm(gradient[N**3:])),
                          gradient_max_coordinate=int(np.argmax(np.abs(gradient))),
                          objective_nuisance_gradient_optimizer_coordinates=(-gradient[N**3:]).tolist(),
                          max_neighbors=width,sigma_los_km_s=float(100*np.exp(.5*float(tracer[6]))),
                          seconds=time.monotonic()-tic)
            report['evaluations']=report.get('evaluations',0)+1
            print(json.dumps(dict(evaluation=report['evaluations'],**latest)),flush=True)
            if report['evaluations']==1:
                if nuisance_only:
                    diagnostic_baseline['fixed_field']=(rho,vel)
                    report['full_gradient_before_block']=dict(latest)
                if os.environ.get('CF4_R2_DIRECTION_DIAG') == '1':
                    diagnostic_baseline['data_gradients'] = grads
                np.savez(OUT/'initial_state.npz',white_ic=np.asarray(white),
                         rho=np.asarray(rho),velocity_km_s=np.asarray(vel),tracer=np.asarray(tracer),
                         white_fp_zero=np.asarray(zero))
                report['initial_objective']=-value
                save_report()
            return -value,-gradient

        def callback(x):
            # The step controller invokes this only after accepting an iterate.
            report['trace'].append(dict(iteration=len(report['trace'])+1,**latest))
            np.savez(OUT/'accepted_checkpoint.npz',parameters=x)
            save_report()

        # One necessary check of the newly composed PM + refreshed-support
        # adjoint, not a separate validation ladder. No heldout score involved.
        value0,gradient0=objective(initial)
        if os.environ.get('CF4_R2_RATE_WARM_START')=='1':
            if (os.environ.get('CF4_R2_DIRECTION_DIAG')=='1'
                    or os.environ.get('CF4_R2_CURVATURE_DIAG')=='1'):
                raise ValueError('do not alter the state of a fixed-state diagnostic')
            old_rate=initial[N**3]/100.
            expected_count=47121.+.5*(100.*gradient0[N**3]-old_rate)
            new_rate=conditional_rate_mode(expected_count,47121.,old_rate)
            delta=new_rate-old_rate
            predicted_change=(-2*47121.*delta+expected_count*np.expm1(2*delta)
                              +.5*(new_rate**2-old_rate**2))
            candidate=initial.copy(); candidate[N**3]=100.*new_rate
            checked,checked_gradient=objective(candidate)
            report['conditional_rate_warm_start']=dict(expected_count_before=float(expected_count),
                white_rate_before=float(old_rate),white_rate_after=float(new_rate),
                predicted_objective_change=float(predicted_change),
                actual_objective_change=float(checked-value0),
                final_scaled_rate_gradient=float(checked_gradient[N**3]),
                policy='one scalar conditional MAP update; same joint target and Gaussian prior')
            save_report()
            if (not np.isclose(checked-value0,predicted_change,rtol=0.,atol=1e-6)
                    or checked>value0+1e-8 or abs(checked_gradient[N**3])>1e-7):
                raise AssertionError('conditional rate update disagrees with the full joint target')
            initial,value0,gradient0=candidate,checked,checked_gradient
            np.savez(OUT/'accepted_checkpoint.npz',parameters=initial)
        if os.environ.get('CF4_R2_DIRECTION_DIAG') == '1':
            # Diagnose the ACTUAL fitted state rather than another prior draw.
            # Three step sizes along descent separate the observation map from
            # the PM VJP using a finite-difference field tangent. No fit/holdout.
            direction=-gradient0/np.linalg.norm(gradient0)
            reverse=float(gradient0@direction)
            grads=diagnostic_baseline['data_gradients']
            epsilons=tuple(float(v) for v in os.environ.get(
                'CF4_R2_DIAG_EPSILONS','0.01,0.0001,0.000001').split(','))
            if not epsilons or not all(np.isfinite(e) and e>0 for e in epsilons):
                raise ValueError('diagnostic epsilons must be finite and positive')
            baseline,_,base_field,_=score_only(initial)
            report['forward_repeat_objective_delta']=baseline-value0
            @jax.jit
            def cut_flags(rho,vel,tracer):
                _,v=native_mass_momentum_to_count_cells(rho,vel,BOX)
                shifted,_,rhat=observer_centred_spherical_rsd_jax(source['positions'],
                    jnp.moveaxis(v,0,-1).reshape(-1,3),jnp.full(3,BOX/2),BOX,
                    common['H0_km_s_Mpc'],little_h=common['h'],scale_factor=1.)
                nodes,_=_gaussian_hermite_rule(15)
                sigma=common['h']*100*jnp.exp(.5*tracer[6])/common['H0_km_s_Mpc']
                def one(node):
                    pos=(shifted+node*sigma*rhat)%BOX
                    radius=jnp.linalg.norm((pos-BOX/2+BOX/2)%BOX-BOX/2,axis=1)
                    return (radius>=5.)&(radius<=180.)
                return jax.lax.map(one,jnp.asarray(nodes))
            check_cuts=os.environ.get('CF4_R2_DIAG_CUTS')=='1'
            if check_cuts and count_integration!='gh':
                raise ValueError('GH atom-crossing diagnostic applies only to GH target')
            if check_cuts:
                baseline_cuts=cut_flags(*base_field,jnp.asarray(initial[N**3:N**3+9]/100.))
            base_rho=np.asarray(base_field[0])
            occupied=base_rho>0
            report['native_density_diagnostic']=dict(empty_cells=int((~occupied).sum()),
                positive_min=float(base_rho[occupied].min()),
                positive_quantiles=np.quantile(base_rho[occupied],[.001,.01,.5]).tolist())
            rows=[]
            for epsilon in epsilons:
                plus,pp,fp,wp=score_only(initial+epsilon*direction)
                minus,pm,fm,wm=score_only(initial-epsilon*direction)
                fd=(plus-minus)/(2*epsilon)
                field_tangent=tuple((a-b)/(2*epsilon) for a,b in zip(fp,fm))
                via_field_fd=-float(jnp.vdot(grads[0],field_tangent[0])
                    +jnp.vdot(grads[1],field_tangent[1])
                    +jnp.vdot(grads[2],jnp.asarray(direction[N**3:N**3+9]/100.))
                    +grads[3]*direction[-1])+float(np.dot(initial[:N**3],direction[:N**3]))
                row=dict(epsilon=epsilon,reverse=reverse,finite_difference=fd,
                    observation_vjp_with_field_fd=via_field_fd,
                    relative_discrepancy=abs(reverse-fd)/max(1.,abs(reverse),abs(fd)),
                    plus_objective_delta=plus-value0,minus_objective_delta=minus-value0,
                    log_factor_finite_differences=((pp-pm)/(2*epsilon)).tolist(),
                    max_neighbors_plus=wp,max_neighbors_minus=wm)
                row['field_tangent_rms']=[float(jnp.sqrt(jnp.mean(t*t))) for t in field_tangent]
                row['native_occupancy_changes']=int(jnp.count_nonzero((fp[0]>0)!=(fm[0]>0)))
                products=(grads[0]*field_tangent[0],grads[1]*field_tangent[1])
                row['observation_field_tangent_contributions']=[-float(jnp.sum(t)) for t in products]
                # Evidence only: these bins never alter fields, likelihoods or gradients.
                row['velocity_contribution_by_native_density']=[
                    -float(jnp.sum(jnp.where(jnp.asarray(mask)[None],products[1],0.)))
                    for mask in (base_rho==0,(base_rho>0)&(base_rho<1e-6),
                                 (base_rho>=1e-6)&(base_rho<1e-3),base_rho>=1e-3)]
                if check_cuts:
                    for name,fields,params in (('plus',fp,initial+epsilon*direction),
                                               ('minus',fm,initial-epsilon*direction)):
                        flags=cut_flags(*fields,jnp.asarray(params[N**3:N**3+9]/100.))
                        row[f'GH_radial_cut_changes_{name}']=np.asarray(
                            jnp.sum(flags!=baseline_cuts,axis=1)).tolist()
                        row[f'GH_radial_cut_sky_active_changes_{name}']=np.asarray(
                            jnp.sum((flags!=baseline_cuts)&
                                jnp.any(source['angular']>0,axis=0)[None,:],axis=1)).tolist()
                rows.append(row)
                report['descent_direction_check']=rows
                save_report()
                print(json.dumps(row,allow_nan=False),flush=True)
            report.update(status='FITTED_STATE_DIRECTION_DIAGNOSTIC_COMPLETE_NOT_POSTERIOR',
                          diagnostic_only=True,optimizer_steps=0,
                          gradient_max_coordinate=int(np.argmax(np.abs(gradient0))))
            save_report()
            return
        direction=np.random.default_rng(2026092801).standard_normal(initial.size)
        direction/=np.linalg.norm(direction)
        epsilon=float(os.environ.get('CF4_R2_ADJOINT_EPS','2e-5'))
        if not np.isfinite(epsilon) or epsilon<=0:
            raise ValueError('invalid initial adjoint epsilon')
        plus,_,_,_=score_only(initial+epsilon*direction)
        minus,_,_,_=score_only(initial-epsilon*direction)
        if not np.isfinite(plus) or not np.isfinite(minus):
            raise FloatingPointError('initial finite-difference trial outside support; no floor')
        reverse=float(gradient0@direction)
        finite=(plus-minus)/(2*epsilon)
        analytic_prior=np.dot(np.r_[initial[:N**3],initial[N**3:N**3+9]/100.**2,
                                      initial[-1]],direction)
        data_reverse,data_finite=reverse-analytic_prior,finite-analytic_prior
        error=abs(data_reverse-data_finite)/max(1.,abs(data_reverse),abs(data_finite))
        report['initial_adjoint']=dict(epsilon=epsilon,reverse=reverse,finite_difference=finite,
                                      analytic_prior_direction=analytic_prior,
                                      observation_reverse=data_reverse,
                                      observation_finite_difference=data_finite,
                                      relative_discrepancy=error)
        save_report()
        if error>=.02:
            raise AssertionError('initial refreshed-support IC adjoint mismatch')
        if os.environ.get('CF4_R2_CURVATURE_DIAG')=='1':
            # Few directions only, no posterior draw and no optimization.
            random=np.r_[np.random.default_rng(2026092809).standard_normal(N**3),np.zeros(10)]
            random/=np.linalg.norm(random)
            wave=np.broadcast_to(np.cos(2*np.pi*np.arange(N)/N)[:,None,None],(N,N,N)).reshape(-1)
            low_k=np.r_[wave/np.linalg.norm(wave),np.zeros(10)]
            rate=np.zeros_like(initial); rate[N**3]=1.
            rows=[]
            def record_curvature(row):
                rows.append(row)
                report['finite_difference_curvature']=rows
                save_report()
                print(json.dumps(row,allow_nan=False),flush=True)
            # The conditional FP factor cancels the common source amplitude.
            # Poisson expected count and rate-axis curvature follow exactly
            # from its gradient and the declared Gaussian rate prior.
            expected_count=47121.+.5*(100.*gradient0[N**3]-initial[N**3]/100.)
            report.update(diagnostic_only=True,optimizer_steps=0,
                classification='FINITE_GRADIENT_CURVATURE_COST_NOT_UNCERTAINTY',
                analytic_rate_axis_curvature=(4.*expected_count+1.)/10000.,
                count_expectation_from_rate_gradient=float(expected_count),
                curvature_limitations='finite differences, only three directions; no positive-definiteness or Laplace certification')
            finite_gradient_curvature(objective,initial,
                dict(rate_coordinate=rate,IC_fundamental_x=low_k,IC_random=random),
                epsilons=(1e-3,3e-4),seconds_left=lambda:cap-(time.monotonic()-started),
                record=record_curvature)
            report.update(status='CURVATURE_FEASIBILITY_COMPLETE_NOT_POSTERIOR')
            save_report()
            return
        if nuisance_only:
            # This is a conditional OPTIMIZER block, not a reduced posterior.
            # All ten coordinates remain free in any later joint inference.
            fixed_rho,fixed_vel=diagnostic_baseline['fixed_field']
            ic_prior=.5*np.dot(initial[:N**3],initial[:N**3])
            coordinate=(np.arange(N)+.5)*(BOX/N)-BOX/2.
            radius=np.sqrt(coordinate[:,None,None]**2+coordinate[None,:,None]**2
                           +coordinate[None,None,:]**2).reshape(-1)
            bins=jnp.asarray(np.minimum(16,(radius/12.).astype(np.int32)))
            edges=np.arange(0.,193.,12.)
            observed,_=np.histogram(radius[np.asarray(keys)%N**3],bins=edges,
                                    weights=np.asarray(counts))

            @jax.jit
            def conditional_target(rho,vel,tracer,zero,links):
                parts,intensity=target_parts(rho,vel,tracer,zero,links)
                prediction=(intensity.reshape(6,-1)*exposure.reshape(6,-1)).sum(axis=0)
                radial=jnp.bincount(bins,weights=prediction,length=17)[:16]
                return jnp.sum(parts[:3]),(parts[:3],radial)

            conditional_vg=jax.jit(jax.value_and_grad(
                conditional_target,argnums=(2,3),has_aux=True))
            conditional_compiled=None
            conditional_latest={}

            def conditional_evaluate(z,*,derivative):
                nonlocal conditional_compiled
                tic=time.monotonic()
                tracer,zero=jnp.asarray(z[:9]/100.),jnp.asarray(z[-1])
                links,width=build_support(fixed_rho,fixed_vel,np.asarray(tracer))
                args=(fixed_rho,fixed_vel,tracer,zero,links)
                if derivative:
                    if conditional_compiled is None:
                        conditional_compiled=conditional_vg.lower(*args).compile()
                        analysis=conditional_compiled.memory_analysis()
                        stats=jax.devices()[0].memory_stats() or {}
                        if analysis is not None:
                            report['nuisance_derivative_memory']=dict(
                                temporary_GiB=analysis.temp_size_in_bytes/1024**3,
                                current_device_GiB=stats.get('bytes_in_use',0)/1024**3,
                                device_limit_GiB=stats.get('bytes_limit',0)/1024**3)
                            save_report()
                            peak=(stats.get('bytes_in_use',0)+analysis.temp_size_in_bytes
                                  +analysis.output_size_in_bytes)
                            if stats.get('bytes_limit') and 1.2*peak>stats['bytes_limit']:
                                raise MemoryError('nuisance derivative lacks 20 percent device margin')
                    (log_value,(parts,radial)),grads=conditional_compiled(*args)
                    grad=-np.r_[np.asarray(grads[0])/100.,float(grads[1])]
                else:
                    log_value,(parts,radial)=conditional_target(*args)
                val=-float(log_value)+ic_prior
                if not np.isfinite(val) or (derivative and not np.isfinite(grad).all()):
                    if derivative or val!=np.inf:
                        full=initial.copy(); full[N**3:]=z
                        np.savez(OUT/'nonfinite_nuisance_trial.npz',parameters=full)
                    if val==np.inf:
                        return (val,np.zeros(10)) if derivative else val
                    raise FloatingPointError('undefined conditional target/derivative; trial preserved')
                if not derivative:
                    report['nuisance_score_only_trials']=report.get('nuisance_score_only_trials',0)+1
                    return val
                report['nuisance_evaluations']=report.get('nuisance_evaluations',0)+1
                conditional_latest.update(objective=val,parts=np.asarray(parts).tolist(),
                    conditional_gradient_inf=float(np.max(np.abs(grad))),
                    conditional_gradient_optimizer_coordinates=grad.tolist(),
                    white_tracer=(z[:9]/100.).tolist(),white_fp_zero=float(z[-1]),
                    predicted_radial_counts=np.asarray(radial).tolist(),
                    max_neighbors=width,seconds=time.monotonic()-tic)
                print(json.dumps(dict(nuisance_evaluation=report['nuisance_evaluations'],
                                      **conditional_latest)),flush=True)
                return val,grad

            z0=initial[N**3:].copy()
            cv,cg=conditional_evaluate(z0,derivative=True)
            if not np.isfinite(cv):
                raise AssertionError('conditional baseline nonfinite although the full target is finite')
            report['conditional_baseline']=dict(conditional_latest)
            report['conditional_full_baseline_agreement']=dict(
                objective_difference=float(cv-value0),
                gradient_max_difference=float(np.max(np.abs(cg-gradient0[N**3:]))))
            save_report()
            if (not np.isclose(cv,value0,rtol=0.,atol=1e-7)
                    or not np.allclose(cg,gradient0[N**3:],rtol=1e-10,atol=1e-8)):
                raise AssertionError('conditional/full baseline target or derivative mismatch')

            @jax.jit
            def fp_only(rho,vel,tracer,zero,links):
                # Only the FP output is used; count integration is dead code.
                return data_target(rho,vel,tracer,zero,links)[1][1]

            def zero_sensitivity(z,fitted_fp):
                tracer=jnp.asarray(z[:9]/100.)
                links,_=build_support(fixed_rho,fixed_vel,np.asarray(tracer))
                prior_mean=float(fp_only(fixed_rho,fixed_vel,tracer,jnp.asarray(0.),links))
                if np.isnan(prior_mean) or prior_mean==np.inf:
                    raise FloatingPointError('undefined fixed-field FP-zero sensitivity')
                return dict(fitted_white_zero=float(z[-1]),fitted_FP_score=float(fitted_fp),
                    prior_mean_zero_FP_score=prior_mean if np.isfinite(prior_mean) else '-inf',
                    fitted_minus_prior_mean_FP=float(fitted_fp-prior_mean) if np.isfinite(prior_mean) else None,
                    interpretation='fixed NEW field and other nuisances; NOT decomposition of earlier field-fit gains')

            report['FP_zero_sensitivity_before_block']=zero_sensitivity(z0,conditional_latest['parts'][1])
            save_report()
            conditional_accepted=dict(conditional_latest)

            def nuisance_callback(z):
                conditional_accepted.clear(); conditional_accepted.update(conditional_latest)
                report['nuisance_trace'].append(dict(iteration=len(report['nuisance_trace'])+1,
                                                    **conditional_latest))
                full=initial.copy(); full[N**3:]=z
                np.savez(OUT/'accepted_checkpoint.npz',parameters=full)
                save_report()

            z,conditional_value,conditional_gradient,block_message=bounded_lbfgs(
                lambda z:conditional_evaluate(z,derivative=True),z0,nuisance_callback,n_ic=0,
                seconds_left=lambda:cap-(time.monotonic()-started)-180.,maxiter=maxiter,
                value_only=lambda z:conditional_evaluate(z,derivative=False),
                initial_norm_cap=norm_cap,initial_evaluation=(cv,cg))
            solution=initial.copy(); solution[N**3:]=z
            report['conditional_endpoint']=dict(conditional_accepted)
            report['FP_zero_sensitivity_after_block']=zero_sensitivity(z,conditional_accepted['parts'][1])
            report.update(nuisance_iterations=len(report['nuisance_trace']),
                nuisance_optimizer_message=block_message,
                nuisance_optimizer_success=block_message=='gradient tolerance',
                final_conditional_gradient_inf=float(np.max(np.abs(conditional_gradient))),
                conditional_objective_gain=float(cv-conditional_value),
                training_radial_profile=dict(radius_edges_cMpc_h=edges.tolist(),
                    observed=observed.tolist(),
                    before=report['conditional_baseline']['predicted_radial_counts'],
                    after=conditional_accepted['predicted_radial_counts'],
                    interpretation='training prediction at fixed field; not heldout or posterior uncertainty'))
            save_report()
            # Required: conditional stationarity is not joint stationarity.
            value,gradient=objective(solution)
            if (not np.isclose(value,conditional_value,rtol=0.,atol=1e-7)
                    or not np.allclose(gradient[N**3:],conditional_gradient,rtol=1e-10,atol=1e-8)):
                raise AssertionError('conditional/full endpoint target or derivative mismatch')
            report['full_gradient_after_block']=dict(latest)
            message='conditional block: '+block_message
        else:
            solution,value,gradient,message=bounded_lbfgs(objective,initial,callback,n_ic=N**3,
                seconds_left=lambda:cap-(time.monotonic()-started),maxiter=maxiter,
                value_only=trial_value,initial_norm_cap=norm_cap,
                initial_evaluation=(value0,gradient0))
        rho,vel,variance,valid=terminal_field(jnp.asarray(solution[:N**3]))
        np.savez(OUT/'final_state.npz',white_ic=solution[:N**3],
                 tracer=solution[N**3:N**3+9]/100.,white_fp_zero=solution[-1],
                 rho=np.asarray(rho),velocity_km_s=np.asarray(vel),
                 physical_velocity_variance_km2_s2=np.asarray(variance),
                 velocity_valid=np.asarray(valid))
        np.savez(OUT/'accepted_checkpoint.npz',parameters=solution)
        # Keep the endpoint derivative for later decisions without another PM VJP.
        np.save(OUT/'final_gradient.npy',gradient)
        report.update(status=('CONDITIONAL_NUISANCE_OPTIMIZER_STOP_NOT_POSTERIOR' if nuisance_only
                              else 'PARTIAL_MAP_OPTIMIZER_STOP_NOT_POSTERIOR'),
            optimizer_success=not nuisance_only and message=='gradient tolerance',optimizer_message=message,
            iterations=len(report['trace']),final_objective=float(value),
            final_gradient_inf=float(np.max(np.abs(gradient))),
            physical_dispersion_saved=True,posterior_velocity_uncertainty=False)
        save_report()
        if nuisance_only:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            profile=report['training_radial_profile']
            fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
            for name,color in (('observed','black'),('before','tab:orange'),('after','tab:blue')):
                ax.stairs(profile[name],profile['radius_edges_cMpc_h'],label=name,color=color)
            ax.set(xlabel='Observed-key cell radius (cMpc/h)',ylabel='Training galaxy count',
                   title='Same FIXED field, nuisance parameters only\nNOT calibration, heldout validation or posterior uncertainty')
            ax.legend()
            fig.savefig(OUT/'conditional_radial_counts.png',dpi=150)
            plt.close(fig)
    except Exception as exc:
        report.update(status='STOPPED_NOT_POSTERIOR',error=f'{type(exc).__name__}: {exc}')
        save_report()
        raise


if __name__=='__main__':
    main()
