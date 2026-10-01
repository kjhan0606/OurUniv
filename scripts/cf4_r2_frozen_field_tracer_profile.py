#!/usr/bin/env python3
"""Profile only the nine count-tracer nuisances at saved R2 endpoint A."""
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_chunked_volume_count import predict_chunked_volume_intensity
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_linked_fp_sparse_train import SOURCE
from cf4_r2_marked_tracer_jax import sparse_marked_poisson_log_likelihood
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_raw_volume_target import tracer_geometry, tracer_masses
from cf4_r2_resolution_target import source_geometry_at_resolution


BASE = Path('/gpfs/kjhan/CF4/z0_density')
N, NC, BOX = 256, 128, 384.0
NIC, NNUISANCE = N**3, 24
TRACER_DIM = 9
VOLUME_RATE = (BOX/N/3.0)**3
SOURCE_CHUNK = 2_097_152
RADIAL_EDGES = np.arange(0.0, 192.0 + 12.0, 12.0)
CHECKPOINT = BASE/'r2_n256_gl2_metric6000_control_a_v1/chain_a_accepted_checkpoint.npz'
COMPONENT_REPORT = BASE/'r2_n256_lowk_component_attribution_20261002_v2/result.json'
COUNT_REPORT = BASE/'r2_n256_training_count_residual_20261002_v1/result.json'
SPLIT = BASE/'r2_sky_closed_split_v6/split.npz'
OUT = Path(os.environ.get('CF4_R2_OUT_DIR',
    str(BASE/'r2_n256_frozen_field_tracer_profile_20261002_v2')))
MAX_PROFILE_EVALUATIONS = 8
APPLICATION_BUDGET_SECONDS = 2*3600 + 15*60
FINALIZATION_RESERVE_SECONDS = 10*60


class ProfileBudgetStop(RuntimeError):
    pass


def radial_population_bins(grid_size, box_size, radial_edges):
    n = int(grid_size)
    flat = np.arange(n**3, dtype=np.int64)
    i = flat//(n*n)
    j = (flat//n) % n
    k = flat % n
    xyz = (np.column_stack((i, j, k)).astype(np.float64)+.5)*(box_size/n)-box_size/2
    radius = np.linalg.norm(xyz, axis=1)
    return np.clip(np.searchsorted(radial_edges, radius, side='right')-1,
                   0, len(radial_edges)-2).astype(np.int32)


def observed_radial_population(keys, counts, grid_size, radial_bin):
    nvoxel = grid_size**3
    key = np.asarray(keys, dtype=np.int64)
    count = np.asarray(counts, dtype=np.int64)
    if key.ndim != 1 or key.shape != count.shape or np.any(key < 0) or np.any(key >= 6*nvoxel):
        raise ValueError('sparse training count keys/counts are invalid')
    pop, voxel = key//nvoxel, key % nvoxel
    nradial = int(np.max(radial_bin))+1
    return np.stack([np.bincount(radial_bin[voxel[pop == p]],
        weights=count[pop == p], minlength=nradial)[:nradial]
        for p in range(6)]).astype(np.int64)


def radial_population_table(observed, expected, edges=RADIAL_EDGES):
    obs = np.asarray(observed, dtype=np.int64)
    exp = np.asarray(expected, dtype=np.float64)
    if obs.shape != exp.shape or obs.shape != (6, len(edges)-1):
        raise ValueError('radial population summaries have incompatible geometry')
    rows = []
    for pop in range(6):
        for radial in range(len(edges)-1):
            o, e = int(obs[pop, radial]), float(exp[pop, radial])
            rows.append(dict(population=pop,
                radius_lower_cMpc_h=float(edges[radial]),
                radius_upper_cMpc_h=float(edges[radial+1]),
                observed_count=o, expected_training_count=e,
                observed_to_expected=float(o/e) if e > 0.0 else None))
    return rows


def radial_population_l1(rows):
    """Descriptive in-sample binned L1 discrepancy; never a test statistic."""
    observed = sum(row['observed_count'] for row in rows)
    if observed == 0:
        raise ValueError('cannot normalize an empty radial training table')
    return float(sum(abs(row['observed_count']-row['expected_training_count'])
                     for row in rows)/observed)


def unpack_profile_result(result):
    """Unpack JAX value_and_grad(has_aux=True): ((value, aux), gradient)."""
    (objective, (score, means)), gradient = result
    return objective, score, means, gradient


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    OUT.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='STARTED', job_id=os.environ['SLURM_JOB_ID'],
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        classification='ONE_FROZEN_FIELD_NINE_TRACER_PROFILE_NOT_POSTERIOR',
        field_checkpoint=str(CHECKPOINT), field_fixed=True,
        optimized_coordinates='the nine standard-normal count tracer nuisances only',
        Gaussian_tracer_prior_included_once=True, PMWD_forward_replays=0,
        PMWD_adjoints=0, chain_transitions=0, heldout_outcome_values_read=False,
        heldout_scored=False, active_count_law_changed=False,
        Q_GOAL='distinguish nuisance non-equilibration from count-law stress upstream of the actual R2 z=0 field; this does not identify the LG',
        Q_LEAN='one saved field endpoint, one deterministic PMWD state replay because rho/velocity were not retained, up to eight 9D count-only value/gradient evaluations; no PMWD adjoint, new chain, or heldout score',
        MW_M31='remain role-ambiguous; no native truth identity seeds or selects a generated field',
        M33='unresolved; later observables must constrain all roles on the same NEW field at <=0.3 cMpc/h',
        trace=[])

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        temp = OUT/'result.json.tmp'
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(OUT/'result.json')

    save()
    try:
        parent = json.loads((BASE/'r2_n256_dynamics_profile_v1/result.json').read_text())
        components = json.loads(COMPONENT_REPORT.read_text())
        previous = json.loads(COUNT_REPORT.read_text())
        if parent.get('status') != 'N256_DYNAMICS_INITIALIZER_NOT_POSTERIOR':
            raise ValueError('frozen N256 PMWD settings unavailable')
        if components.get('status') != 'COMPONENT_ATTRIBUTION_COMPLETE_NOT_POSTERIOR':
            raise ValueError('saved count-gradient reference unavailable')
        if previous.get('status') != 'SAME_STATE_TRAINING_COUNT_RESIDUAL_COMPLETE_NOT_POSTERIOR':
            raise ValueError('same-state radial-count reference unavailable')

        with np.load(CHECKPOINT, allow_pickle=False) as f:
            q = np.asarray(f['canonical'], dtype=np.float64)
            completed = int(f['completed_transitions'])
        if q.shape != (NIC+NNUISANCE,) or completed != 1 or not np.isfinite(q).all():
            raise ValueError('saved A endpoint layout/status mismatch')
        tracer0 = q[NIC:NIC+TRACER_DIM]
        a_component = components['checkpoints']['a']
        if Path(a_component['checkpoint']) != CHECKPOINT:
            raise ValueError('component gradient and fixed field do not share checkpoint A')

        with np.load(SOURCE, allow_pickle=False) as f:
            source_data = {name: f[name].copy() for name in
                ('positions', 'angular', 'radial_table', 'modulus_table', 'redshift_table')}
        source_xy = source_geometry_at_resolution(source_data, N)
        positions = jnp.asarray(source_xy['positions'])
        angular = jnp.asarray(source_xy['angular'])
        radial_map = jnp.asarray(radial_population_bins(NC, BOX, RADIAL_EDGES))
        growth = dict(observer=jnp.full(3, BOX/2), box_size_cMpc_h=BOX,
            hubble_km_s_Mpc=74.6, little_h=.746,
            radius_table_cMpc_h=jnp.asarray(source_data['radial_table']),
            modulus_table_h=jnp.asarray(source_data['modulus_table']),
            redshift_table=jnp.asarray(source_data['redshift_table']), grid_size=NC,
            radial_min_cMpc_h=5., radial_max_cMpc_h=180.)

        with np.load(SPLIT, allow_pickle=False) as f:
            keys = np.asarray(f['train_keys'], dtype=np.int64)
            counts = np.asarray(f['train_counts'], dtype=np.int64)
            # Only the heldout-octant mask and training graph buffer are needed
            # to construct training exposure. No heldout outcome/key arrays.
            training_exposure, _ = build_population_exposure_masks(
                NC, f['heldout_flat_voxels'], f['train_window_excluded_keys'], ())
        observed = observed_radial_population(keys, counts, NC,
            np.asarray(radial_map))
        key_j, count_j = jnp.asarray(keys), jnp.asarray(counts)
        exposure_j = jnp.asarray(training_exposure)

        evolve, _, config, _, particle_mass = make_dynamics(parent['settings'])
        particle_mass_array = jnp.full(NIC, particle_mass)

        @jax.jit
        def recover_field(white):
            pos, vel = evolve(white)
            state = particle_grid(pos, vel, particle_mass_array, config)
            return state['rho'], jnp.moveaxis(state['mean_velocity_km_s'], -1, 0)

        tic = time.monotonic()
        rho_native, velocity_native = recover_field(jnp.asarray(q[:NIC]))
        jax.block_until_ready((rho_native, velocity_native))
        density, velocity = native_mass_momentum_to_count_cells(
            rho_native, velocity_native, BOX)
        velocity = jnp.moveaxis(velocity, 0, -1).reshape(-1, 3)
        report.update(PMWD_forward_replays=1,
            PMWD_forward_seconds=time.monotonic()-tic,
            interpretation='same deterministic saved-state PMWD forward; no independent IC or gravity realization')
        save()

        radial_count = len(RADIAL_EDGES)-1

        def profile_target(tracer, density, velocity, source_pos, source_sky,
                          keys, counts, exposure):
            masses = tracer_masses(density, tracer)*VOLUME_RATE
            geometry = tracer_geometry(tracer, growth)
            intensity = predict_chunked_volume_intensity(source_pos, velocity,
                masses, source_sky, source_chunk_size=SOURCE_CHUNK,
                source_spacing=BOX/N, volume_order=2, order=4, segments=8,
                **geometry)
            score = sparse_marked_poisson_log_likelihood(
                intensity, keys, counts, selected_voxel_mask=exposure)
            means = jnp.stack([jnp.bincount(radial_map,
                weights=intensity[p].reshape(-1)*exposure.reshape(6, NC**3)[p],
                length=radial_count) for p in range(6)])
            objective = .5*jnp.vdot(tracer, tracer)-score
            return objective, (score, means)

        derivative = jax.jit(jax.value_and_grad(profile_target, argnums=0, has_aux=True))
        args = (jnp.asarray(tracer0), density, velocity, positions, angular,
                key_j, count_j, exposure_j)
        compiled = derivative.lower(*args).compile()
        memory = compiled.memory_analysis()
        stats = jax.devices()[0].memory_stats() or {}
        peak = (stats.get('bytes_in_use', 0)+memory.temp_size_in_bytes+
                memory.output_size_in_bytes)
        report['compiled_device_memory'] = dict(
            estimated_peak_GiB=peak/1024**3,
            device_limit_GiB=stats.get('bytes_limit', 0)/1024**3 if stats.get('bytes_limit') else None)
        if stats.get('bytes_limit') and 1.2*peak > stats['bytes_limit']:
            raise MemoryError('count nuisance profile misses the required 20 percent GPU memory margin')

        objective0, score0, expected0, gradient0 = unpack_profile_result(compiled(*args))
        objective0, score0 = float(objective0), float(score0)
        expected0 = np.asarray(expected0, dtype=np.float64)
        gradient0 = np.asarray(gradient0, dtype=np.float64)
        reference_score = float(a_component['component_scores']['count_2mpp'])
        score_error = abs(score0-reference_score)
        # Reuse the saved A radial table without reopening heldout observations.
        old_rows = previous['states']['a']['radial_by_population']
        expected_reference = np.array([[r['expected_training_count'] for r in old_rows
            if r['population'] == p] for p in range(6)], dtype=np.float64)
        saved_nll_gradient = np.asarray(
            a_component['component_gradients']['count_2mpp']['negative_log_gradient_tracer9'],
            dtype=np.float64)
        gradient_error = float(np.linalg.norm(gradient0-(tracer0+saved_nll_gradient)) /
                               max(1., np.linalg.norm(gradient0)))
        if score_error > 1e-7 or gradient_error > 1e-6:
            raise AssertionError('fixed-field count score/gradient does not reproduce saved endpoint')
        if expected_reference.shape != expected0.shape or np.max(np.abs(expected_reference-expected0)) > 1e-6:
            raise AssertionError('fixed-field radial expected counts do not reproduce saved A readout')

        initial_rows = radial_population_table(observed, expected0, RADIAL_EDGES)
        prior0 = .5*float(tracer0@tracer0)
        report['initial'] = dict(count_log_score=score0,
            count_score_abs_error_to_saved=score_error,
            gaussian_tracer_prior_nll=prior0, conditional_objective=objective0,
            tracer_coordinates=tracer0.tolist(),
            objective_gradient_norm=float(np.linalg.norm(gradient0)),
            score_gradient_relative_error_to_saved=gradient_error,
            radial_population_l1_fraction=radial_population_l1(initial_rows),
            radial_by_population=initial_rows)
        report['evaluations'] = 1
        report['trace'].append(dict(evaluation=1, objective=objective0,
            count_log_score=score0, gradient_norm=float(np.linalg.norm(gradient0)),
            radial_population_l1_fraction=radial_population_l1(initial_rows)))
        save()

        cache = dict(theta=tracer0.copy(), objective=objective0, score=score0,
                     gradient=gradient0.copy(), expected=expected0.copy())
        best = {key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in cache.items()}

        def evaluate(theta):
            theta = np.asarray(theta, dtype=np.float64)
            if np.array_equal(theta, cache['theta']):
                return cache['objective'], cache['gradient']
            if report['evaluations'] >= MAX_PROFILE_EVALUATIONS:
                raise ProfileBudgetStop('predeclared eight profile evaluations reached')
            if time.monotonic()-started >= APPLICATION_BUDGET_SECONDS-FINALIZATION_RESERVE_SECONDS:
                raise ProfileBudgetStop('profile time budget reserve reached')
            value, score, expected, gradient = unpack_profile_result(compiled(
                jnp.asarray(theta), density, velocity, positions, angular,
                key_j, count_j, exposure_j))
            value, score = float(value), float(score)
            gradient = np.asarray(gradient, dtype=np.float64)
            expected = np.asarray(expected, dtype=np.float64)
            if not np.isfinite(np.r_[value, score, gradient, expected]).all():
                raise FloatingPointError('nonfinite conditional profile evaluation')
            cache.update(theta=theta.copy(), objective=value, score=score,
                         gradient=gradient.copy(), expected=expected.copy())
            if value < best['objective']:
                best.update(theta=theta.copy(), objective=value, score=score,
                            gradient=gradient.copy(), expected=expected.copy())
            report['evaluations'] += 1
            rows = radial_population_table(observed, expected, RADIAL_EDGES)
            report['trace'].append(dict(evaluation=report['evaluations'],
                objective=value, count_log_score=score,
                tracer_prior_nll=.5*float(theta@theta),
                gradient_norm=float(np.linalg.norm(gradient)),
                radial_population_l1_fraction=radial_population_l1(rows)))
            save()
            print(json.dumps(report['trace'][-1]), flush=True)
            return value, gradient

        try:
            fit = minimize(evaluate, tracer0, jac=True, method='L-BFGS-B',
                options=dict(maxiter=12, maxfun=MAX_PROFILE_EVALUATIONS,
                             maxls=4, ftol=1e-9, gtol=1e-4))
            if not np.array_equal(np.asarray(fit.x), cache['theta']):
                evaluate(fit.x)
            theta_final = cache['theta'].copy()
            stop_reason = str(fit.message)
            optimizer_success = bool(fit.success)
        except ProfileBudgetStop as stop:
            fit = None
            stop_reason = str(stop)
            optimizer_success = False

        theta_final = best['theta'].copy()
        final_rows = radial_population_table(observed, best['expected'], RADIAL_EDGES)
        final_prior = .5*float(theta_final@theta_final)
        report.update(status=('FROZEN_FIELD_TRACER_PROFILE_CONVERGED_NOT_CALIBRATION'
                              if optimizer_success else
                              'FROZEN_FIELD_TRACER_PROFILE_BOUNDED_NOT_CONVERGED'),
            optimizer_success=optimizer_success, optimizer_message=stop_reason,
            iterations=int(fit.nit) if fit is not None else None,
            count_log_score=float(best['score']),
            count_score_change=float(best['score']-score0),
            gaussian_tracer_prior_nll=final_prior,
            conditional_objective=float(best['objective']),
            conditional_objective_change=float(best['objective']-objective0),
            tracer_coordinates=theta_final.tolist(),
            tracer_names=['log_rate_white']+[f'bias_white_{i}' for i in range(5)]+
                ['sigma_los_white','alpha_white','mstar_white'],
            objective_gradient_norm=float(np.linalg.norm(best['gradient'])),
            radial_population_l1_fraction=radial_population_l1(final_rows),
            radial_by_population=final_rows,
            limits='one fixed, nonstationary in-sample field; any nuisance fit or residual closure is conditional mechanics only, not calibration, validation, or a posterior')
        save()
        print(json.dumps(report, allow_nan=False), flush=True)
    except Exception as error:
        report.update(status='FROZEN_FIELD_TRACER_PROFILE_FAILED_OR_INCOMPLETE_NOT_CALIBRATION',
            error=f'{type(error).__name__}: {error}')
        save()
        raise


if __name__ == '__main__':
    main()
