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
from scipy.optimize import minimize
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
OUT = BASE/'r2_v6_partial_map_v1'


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
                      nuisance_precondition_scale=100.,maxiter=128,maxfun=192,
                      application_seconds_cap=2700)
        initial = np.r_[np.random.default_rng(2026092702).standard_normal(N**3),np.zeros(10)]
        latest = {}

        def objective(x):
            if time.monotonic()-started > 2700:
                raise TimeoutError('bounded MAP budget reached; preserve accepted checkpoint')
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
                raise FloatingPointError('nonfinite objective/gradient; no surrogate penalty')
            latest.update(parts=list(map(float,np.asarray(parts))),objective=-value,
                          gradient_inf=float(np.max(np.abs(gradient))),
                          max_neighbors=width,sigma_los_km_s=float(100*np.exp(.5*float(tracer[6]))),
                          seconds=time.monotonic()-tic)
            report['evaluations']=report.get('evaluations',0)+1
            print(json.dumps(dict(evaluation=report['evaluations'],**latest)),flush=True)
            if report['evaluations']==1:
                np.savez(OUT/'initial_state.npz',white_ic=np.asarray(white),
                         rho=np.asarray(rho),velocity_km_s=np.asarray(vel),tracer=np.asarray(tracer))
                report['initial_objective']=-value
                save_report()
            return -value,-gradient

        def callback(x):
            # SciPy invokes this only after accepting an iterate.
            report['trace'].append(dict(iteration=len(report['trace'])+1,**latest))
            np.savez(OUT/'accepted_checkpoint.npz',parameters=x)
            save_report()

        # One necessary check of the newly composed PM + refreshed-support
        # adjoint, not a separate validation ladder. No heldout score involved.
        value0,gradient0=objective(initial)
        direction=np.random.default_rng(2026092801).standard_normal(initial.size)
        direction/=np.linalg.norm(direction)
        epsilon=2e-5
        plus,_=objective(initial+epsilon*direction)
        minus,_=objective(initial-epsilon*direction)
        reverse=float(gradient0@direction)
        finite=(plus-minus)/(2*epsilon)
        error=abs(reverse-finite)/max(1.,abs(reverse),abs(finite))
        report['initial_adjoint']=dict(reverse=reverse,finite_difference=finite,
                                      relative_discrepancy=error)
        save_report()
        if error>=.02:
            raise AssertionError('initial refreshed-support IC adjoint mismatch')
        result=minimize(objective,initial,jac=True,method='L-BFGS-B',callback=callback,
                        options=dict(maxiter=128,maxfun=192,maxcor=8,maxls=12,gtol=1e-4,ftol=1e-10))
        rho,vel=field(jnp.asarray(result.x[:N**3]))
        np.savez(OUT/'final_state.npz',white_ic=result.x[:N**3],
                 tracer=result.x[N**3:N**3+9]/100.,white_fp_zero=result.x[-1],
                 rho=np.asarray(rho),velocity_km_s=np.asarray(vel))
        report.update(status='PARTIAL_MAP_OPTIMIZER_STOP_NOT_POSTERIOR',
            optimizer_success=bool(result.success),optimizer_message=str(result.message),
            iterations=int(result.nit),final_objective=float(result.fun),
            final_gradient_inf=float(np.max(np.abs(result.jac))))
        save_report()
    except Exception as exc:
        report.update(status='STOPPED_NOT_POSTERIOR',error=f'{type(exc).__name__}: {exc}')
        save_report()
        raise


if __name__=='__main__':
    main()
