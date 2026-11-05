"""One bounded nuisance-only posterior on the preserved unconditional PM state."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
# This executable shares its basename with the src module. Prefer src rather
# than Python's initial scripts/ entry, as the other source drivers do.
sys.path.insert(0,str(ROOT/'src'))
from cf4_chunked_hmc import make_chunks, checked_record
from cf4_r2_joint_calibration import group_scores, white_logdensity

BASE = Path('/gpfs/kjhan/CF4/z0_density')
CACHE = BASE/'r2_joint_calibration_cache_v1'
OUT = BASE/'r2_joint_calibration_v1'
DATA_KEYS = ('distance','log_distance_weight','redshift_logkernel','row_group',
    'dz_row','eta_mean','eta_std','eta_alpha','predicted_modulus','anchor_group',
    'anchor_modulus','anchor_error','anchor_method','group_holdout')


def read_cache(nodes):
    with np.load(CACHE/f'geometry_q{nodes}.npz') as f:
        data = {k:jnp.asarray(f[k]) for k in DATA_KEYS}
        names = list(map(str,f['method_names']))
    if any(not np.isfinite(np.asarray(v)).all() for k,v in data.items()
           if k != 'log_distance_weight'):
        raise ValueError('nonfinite geometry or observations')
    w = np.asarray(data['log_distance_weight'])
    if np.isnan(w).any() or np.isposinf(w).any() or not np.isfinite(w).any(axis=1).all():
        raise ValueError('invalid radial support')
    if np.any(np.asarray(data['anchor_error']) <= 0):
        raise ValueError('nonpositive anchor error')
    if data['anchor_method'].size and int(data['anchor_method'].max()) >= len(names):
        raise ValueError('unknown method')
    return data,names


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.monotonic()
    OUT.mkdir(parents=True)
    data,names = read_cache(257)
    dimension = len(names)+2
    mean = jnp.zeros(dimension)
    prior_sd = np.array([.004]+[1.]*len(names)+[2.])
    prior_L = jnp.diag(jnp.asarray(prior_sd))
    parameter_names = ['FP_zero_dex']+[f'{m}_offset_mag' for m in names]+['selected_radial_kappa']
    target = jax.jit(lambda w: white_logdensity(w,data,mean,prior_L))
    value_grad = jax.jit(jax.value_and_grad(lambda w: -target(w)))
    def objective(w):
        value,grad = value_grad(jnp.asarray(w))
        value,grad = float(value),np.asarray(grad)
        if not np.isfinite(value) or not np.isfinite(grad).all():
            raise FloatingPointError('nonfinite MAP objective/gradient')
        return value,grad
    print('MAP: zero initialization; original observations once; development prior only',flush=True)
    fit = minimize(objective,np.zeros(dimension),jac=True,method='BFGS',
                   options=dict(maxiter=100,gtol=1e-4))
    map_white = jnp.asarray(fit.x)
    precision = np.asarray(jax.jit(jax.hessian(lambda w: -target(w)))(map_white))
    if not np.isfinite(precision).all() or not np.allclose(precision,precision.T,atol=1e-7):
        raise ValueError('invalid observed MAP Hessian')
    np.linalg.cholesky(precision)  # no eigenvalue clipping or silent model repair
    transform = jnp.asarray(np.linalg.cholesky(np.linalg.inv(precision)))
    map_logdensity = target(map_white)
    # A fixed linear change of coordinates preserves the exact target. The
    # Gaussian curvature is used only as a mass/coordinate preconditioner.
    transformed_target = lambda q: target(map_white+transform@q)-map_logdensity
    map_info = dict(success=bool(fit.success),message=str(fit.message),
        evaluations=int(fit.nfev),iterations=int(fit.nit),
        max_abs_white_gradient=float(np.abs(fit.jac).max()),
        parameters=np.asarray(prior_L@map_white).tolist(),
        precision_eigenvalues=np.linalg.eigvalsh(precision).tolist())
    print(json.dumps(dict(MAP=map_info,seconds=time.monotonic()-start)),flush=True)
    settings = dict(initial_step_size=.25,maximum_step_size=1.,
        target_acceptance=.85,divergence_threshold=1000.,integration_steps=5,
        integration_steps_range=[3,7])
    initialize,warm,sample,final = make_chunks(transformed_target,dimension,settings,record_steps=True)
    seed,nchain,nwarm,nkeep,chunk = 20260927,4,128,256,32
    traces,steps = [],[]
    bounded_stop = False
    def checkpoint():
        payload = dict(parameter_names=np.asarray(parameter_names),
            prior_mean=np.asarray(mean),prior_cholesky=np.asarray(prior_L),
            map_white=np.asarray(map_white),coordinate_cholesky=np.asarray(transform))
        for c,trace in enumerate(traces):
            if not trace:
                continue
            arrays = [np.concatenate([r[k] for r in trace]) for k in range(8)]
            white = np.asarray(map_white)+arrays[0]@np.asarray(transform).T
            payload[f'parameters_chain{c}'] = white@np.asarray(prior_L).T
            for k,label in enumerate(('q','logdensity','acceptance','divergent','energy',
                                       'raw_step','used_step','integration_steps')):
                payload[f'{label}_chain{c}'] = arrays[k]
        np.savez_compressed(OUT/'chains.npz',**payload)
    for chain in range(nchain):
        chain_key = jax.random.fold_in(jax.random.PRNGKey(seed),chain)
        start_key,warm_key,sample_key = jax.random.split(chain_key,3)
        state,adaptation = initialize(1.5*jax.random.normal(start_key,(dimension,),dtype=jnp.float64))
        warm_keys = jax.random.split(warm_key,nwarm)
        for offset in range(0,nwarm,chunk):
            (state,adaptation),record = warm(state,adaptation,warm_keys[offset:offset+chunk])
            checked_record(record,state)
            if time.monotonic()-start > 720:
                bounded_stop = True
                break
        if bounded_stop:
            break
        step = final(adaptation)
        steps.append(float(step))
        traces.append([])
        sample_keys = jax.random.split(sample_key,nkeep)
        for offset in range(0,nkeep,chunk):
            state,record = sample(state,step,sample_keys[offset:offset+chunk])
            traces[-1].append(checked_record(record,state))
            print(f'chain {chain+1}/{nchain}: retained {offset+chunk}/{nkeep}; seconds {time.monotonic()-start:.1f}',flush=True)
            if time.monotonic()-start > 720:
                bounded_stop = True
                break
        checkpoint()
        if bounded_stop:
            break
    checkpoint()
    complete = len(traces) == nchain and all(sum(len(r[0]) for r in t) == nkeep for t in traces)
    result = dict(classification='FIXED_FIELD_JOINT_NUISANCE_DEVELOPMENT_POSTERIOR',
        job_id=os.environ['SLURM_JOB_ID'],parameter_names=parameter_names,
        prior_mean=np.asarray(mean).tolist(),prior_sd=prior_sd.tolist(),
        prior_is_independent_external_calibration=False,prior_is_development_regularization=True,
        fitted_previous_offsets_used=False,raw_observations_reused_in_prior=False,
        kappa_is_inclusion_probability=False,group_selection_separately_identified=False,
        shared_source_shape_covariance_resolved=False,full_absolute_calibration=False,
        group_FoG_covariance_calibrated=False,independent_holdout_validation=False,
        field_inferred=False,R2_posterior=False,new_gravity_runs=0,
        train_groups=int((~np.asarray(data['group_holdout'])).sum()),
        holdout_groups=int(np.asarray(data['group_holdout']).sum()),
        MAP=map_info,sampler_settings=settings,seed=seed,
        requested_chains=nchain,warmup_per_chain=nwarm,requested_retained_per_chain=nkeep,
        actual_retained_per_chain=[sum(len(r[0]) for r in t) for t in traces],
        sampling_complete=complete,bounded_stop=bounded_stop,final_steps=steps,
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),ROOT/'src/cf4_r2_joint_calibration.py',
                      ROOT/'src/cf4_r2_fp_group_marginal.py',ROOT/'src/cf4_chunked_hmc.py')},
        cache_manifest_sha256=hashlib.sha256((CACHE/'manifest.json').read_bytes()).hexdigest())
    if complete:
        with np.load(OUT/'chains.npz') as f:
            parameters = np.stack([f[f'parameters_chain{c}'] for c in range(nchain)])
            acceptance = np.stack([f[f'acceptance_chain{c}'] for c in range(nchain)])
            divergences = np.stack([f[f'divergent_chain{c}'] for c in range(nchain)])
        split = np.concatenate((parameters[:,:nkeep//2],parameters[:,nkeep//2:]),axis=0)
        rhat = np.asarray(blackjax.diagnostics.potential_scale_reduction(jnp.asarray(split)))
        ess = np.asarray(blackjax.diagnostics.effective_sample_size(jnp.asarray(split)))
        flat = parameters.reshape(-1,dimension)
        result.update(posterior_mean=flat.mean(axis=0).tolist(),posterior_sd=flat.std(axis=0,ddof=1).tolist(),
            posterior_quantile_025_50_975=np.quantile(flat,[.025,.5,.975],axis=0).tolist(),
            split_Rhat=rhat.tolist(),split_ESS=ess.tolist(),
            Rhat_is_rank_normalized=False,mean_acceptance=float(acceptance.mean()),
            retained_divergences=int(divergences.sum()))
        selected = jnp.asarray(flat[::16])  # fixed 64 draws, not score-selected
        evaluate = jax.jit(lambda p: jax.lax.map(lambda x: group_scores(x,data),p))
        scores = np.asarray(evaluate(selected))
        hold = np.asarray(data['group_holdout'])
        hold_logfactor = scores[:,hold].sum(axis=1)
        logweights = hold_logfactor-logsumexp(hold_logfactor)
        importance_ess = float(np.exp(-logsumexp(2*logweights)))
        result['heldout_conditional_mark_factor'] = dict(draws=len(selected),
            logmean_factor_relative_to_fixed_data_reference=float(logsumexp(hold_logfactor)-np.log(len(selected))),
            importance_ESS=importance_ess,maximum_weight=float(np.exp(logweights).max()),
            absolute_predictive_logdensity=False,independent_validation=False,
            reliable_estimate_established=False)
        fine,fine_names = read_cache(513)
        if fine_names != names:
            raise ValueError('cache method order changed')
        np.testing.assert_array_equal(fine['group_holdout'],data['group_holdout'])
        fine_eval = jax.jit(lambda p: jax.lax.map(lambda x: group_scores(x,fine),p))
        fine_scores = np.asarray(fine_eval(selected[::2]))  # fixed 32 of the above draws
        correction = (fine_scores[:,~hold]-scores[::2,~hold]).sum(axis=1)
        hold_correction = (fine_scores[:,hold]-scores[::2,hold]).sum(axis=1)
        result['distance_quadrature'] = dict(coarse=257,fine=513,draws=32,
            train_logtarget_correction_min=float(correction.min()),
            train_logtarget_correction_max=float(correction.max()),
            train_logtarget_correction_range=float(np.ptp(correction)),
            heldout_logfactor_correction_range=float(np.ptp(hold_correction)),
            whole_posterior_error_bound=False)
        np.savez_compressed(OUT/'retained_checks.npz',parameters=np.asarray(selected),
            heldout_logfactor=hold_logfactor,training_quadrature_correction=correction,
            heldout_quadrature_correction=hold_correction)
        result['bounded_numerical_control_pass'] = bool(np.isfinite(rhat).all() and np.isfinite(ess).all()
            and (rhat < 1.05).all() and (ess > 100).all() and not divergences.any()
            and np.ptp(correction) < .05)
    else:
        result['bounded_numerical_control_pass'] = False
    result['runtime_seconds'] = time.monotonic()-start
    (OUT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
