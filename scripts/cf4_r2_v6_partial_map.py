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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_linked_singleton_target import partial_v6_count_singleton_parts
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_2mpp_joint_likelihood_jax import observer_centred_spherical_rsd_jax
from cf4_r2_linked_fp_sparse_train import load_train_singletons, FP, SOURCE

BASE = Path('/gpfs/kjhan/CF4/z0_density')
SPLIT = BASE/'r2_sky_closed_split_v6/split.npz'
N, BOX, WIDTH = 128, 384., 8192
OUT = Path(os.environ.get('CF4_R2_OUT_DIR', str(BASE/'r2_v6_partial_map_v2')))


def bounded_lbfgs(fun, initial, callback, *, n_ic, seconds_left, maxiter=128,
                  value_only=None, initial_norm_cap=None):
    """Descent-only L-BFGS with bounded trial steps, not parameter bounds.

    An infinite objective means a genuine zero-probability trial and is
    rejected, never replaced by a likelihood floor. A finite objective with
    nonfinite derivative is an implementation/nondifferentiability failure;
    the objective callable must stop rather than mask that case.
    """
    x=initial.copy()
    value,grad=fun(x)
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
        divisor=max(1.,np.linalg.norm(direction[:n_ic])/np.sqrt(n_ic)/.1,
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
        trace=[])

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

        @jax.jit
        def data_target(rho,vel,tracer,zero,links):
            parts,_ = partial_v6_count_singleton_parts(rho,vel,jnp.zeros(0),tracer,
                source,links,keys,counts,exposure,box=BOX,
                hubble=common['H0_km_s_Mpc'],h=common['h'],white_fp_zero=zero)
            return jnp.sum(parts[:3]),parts[:3]
        data_vg = jax.jit(jax.value_and_grad(data_target,argnums=(0,1,2,3),has_aux=True))
        report.update(training_count_keys=int(keys.size),training_counts=int(counts.sum()),
                      training_singletons=len(options),IC_seed=2026092702,
                      nuisance_precondition_scale=100.,maxiter=128,
                      application_seconds_cap=2700)
        initial = np.r_[np.random.default_rng(2026092702).standard_normal(N**3),np.zeros(10)]
        restart=os.environ.get('CF4_R2_RESTART')
        if restart:
            with np.load(restart,allow_pickle=False) as f:
                initial=f['parameters'].copy()
            if initial.shape!=(N**3+10,) or not np.isfinite(initial).all():
                raise ValueError('invalid accepted restart coordinates')
            report['restart']=restart
            report['restart_policy']='same target/accepted state; fresh L-BFGS history, not exact optimizer continuation'
        latest = {}
        diagnostic_baseline = {}
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
            tic = time.monotonic()
            white = jnp.asarray(x[:N**3])
            tracer = jnp.asarray(x[N**3:N**3+9]/100.)
            zero = jnp.asarray(x[-1])
            (rho,vel), pullback = jax.vjp(field,white)
            links,width = build_support(rho,vel,np.asarray(tracer))
            (value,parts), grads = data_vg(rho,vel,tracer,zero,links)
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
                          max_neighbors=width,sigma_los_km_s=float(100*np.exp(.5*float(tracer[6]))),
                          seconds=time.monotonic()-tic)
            report['evaluations']=report.get('evaluations',0)+1
            print(json.dumps(dict(evaluation=report['evaluations'],**latest)),flush=True)
            if report['evaluations']==1:
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
        plus,_=objective(initial+epsilon*direction)
        minus,_=objective(initial-epsilon*direction)
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
        maxiter=int(os.environ.get('CF4_R2_MAXITER','128'))
        cap=int(os.environ.get('CF4_R2_SECONDS_CAP','2700'))
        norm_cap=os.environ.get('CF4_R2_INITIAL_NORM_CAP')
        norm_cap=None if norm_cap is None else float(norm_cap)
        if maxiter<1 or cap<1 or (norm_cap is not None and (not np.isfinite(norm_cap) or norm_cap<=0)):
            raise ValueError('invalid bounded fit budget/initial step norm')
        report.update(maxiter=maxiter,application_seconds_cap=cap,
                      initial_step_norm_cap=norm_cap,score_only_line_search=True)
        solution,value,gradient,message=bounded_lbfgs(objective,initial,callback,n_ic=N**3,
            seconds_left=lambda:cap-(time.monotonic()-started),maxiter=maxiter,
            value_only=trial_value,initial_norm_cap=norm_cap)
        rho,vel=field(jnp.asarray(solution[:N**3]))
        np.savez(OUT/'final_state.npz',white_ic=solution[:N**3],
                 tracer=solution[N**3:N**3+9]/100.,white_fp_zero=solution[-1],
                 rho=np.asarray(rho),velocity_km_s=np.asarray(vel))
        np.savez(OUT/'accepted_checkpoint.npz',parameters=solution)
        report.update(status='PARTIAL_MAP_OPTIMIZER_STOP_NOT_POSTERIOR',
            optimizer_success=message=='gradient tolerance',optimizer_message=message,
            iterations=len(report['trace']),final_objective=float(value),
            final_gradient_inf=float(np.max(np.abs(gradient))))
        save_report()
    except Exception as exc:
        report.update(status='STOPPED_NOT_POSTERIOR',error=f'{type(exc).__name__}: {exc}')
        save_report()
        raise


if __name__=='__main__':
    main()
