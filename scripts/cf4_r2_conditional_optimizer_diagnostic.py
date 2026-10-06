"""Replay the conditional target and attribute its gradient. Not an optimization."""
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_conditional_map import (
    BOX, KNOWN_DEVICE_PEAK_GIB, MAX_SUPPORT_CELLS, N, N_IC, TERM_KEYS,
    assemble_conditional_objective, conditional_target_terms,
)
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_resolution_target import (
    ResolutionObservationTarget, linked_point_conditioning_radius, source_geometry_at_resolution,
)

BASE = Path('/gpfs/kjhan/CF4/z0_density')
ROOT = Path(__file__).resolve().parents[1]
SAVED = BASE / 'r2_conditional_map_20261005_v3'
ABSOLUTE_TERM_LIMIT = 1e-3
RELATIVE_OBJECTIVE_LIMIT = 1e-8
FD_EPS = 1e-4
FD_RELATIVE_LIMIT = 1e-2
COMPONENT_RELATIVE_L2_LIMIT = 1e-4
COMPONENT_MAX_ABS_FACTOR = 1e-3
APP_SECONDS = 70 * 60
ADJOINT_STAGE_SECONDS = 18 * 60
VALUE_STAGE_SECONDS = 12 * 60
COMPONENTS = ((0, 'count_2mpp'), (1, 'conditional_FP'))


def reproduction_failures(saved, replayed, *, absolute_term_limit=ABSOLUTE_TERM_LIMIT,
                          relative_objective_limit=RELATIVE_OBJECTIVE_LIMIT):
    """Empty means the saved evaluation and the replay agree at the declared gate."""
    failures = []
    for key in TERM_KEYS:
        error = abs(float(replayed[key]) - float(saved[key]))
        if not np.isfinite(error) or error > absolute_term_limit:
            failures.append(dict(kind='term', term=key, absolute_error=error,
                                 limit=absolute_term_limit))
    saved_objective = float(saved['objective'])
    replay_objective = float(replayed['total_negative_log_target'])
    relative = abs(replay_objective - saved_objective) / max(abs(saved_objective), 1.0)
    if not np.isfinite(relative) or relative > relative_objective_limit:
        failures.append(dict(
            kind='objective', absolute_error=abs(replay_objective - saved_objective),
            relative_error=relative, limit=relative_objective_limit))
    return failures


def block_gradient_summary(gradient, n_ic=N_IC):
    gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
    if gradient.shape != (n_ic + 24,):
        raise ValueError('gradient is not the canonical IC plus 24 nuisance coordinates')
    spans = (('IC', 0, n_ic), ('tracer', n_ic, n_ic + 9), ('population', n_ic + 9, n_ic + 24))
    flat = int(np.argmax(np.abs(gradient)))
    summary = {}
    for name, start, stop in spans:
        block = gradient[start:stop]
        summary[name] = dict(
            size=int(block.size),
            rms=float(np.sqrt(np.mean(block * block))),
            inf=float(np.max(np.abs(block))),
        )
    if flat < n_ic:
        owner, local = 'IC', flat
    elif flat < n_ic + 9:
        owner, local = 'tracer', flat - n_ic
    else:
        owner, local = 'population', flat - (n_ic + 9)
    summary['infinity_norm_location'] = dict(
        flat_index=flat, block=owner, index_in_block=int(local), value=float(gradient[flat]))
    return summary


def finite_difference_agreement(value_at_x, value_at_step, analytic_directional_derivative,
                                eps, *, relative_limit=FD_RELATIVE_LIMIT):
    if not np.isfinite(eps) or eps == 0.0:
        raise ValueError('finite-difference step must be a nonzero finite number')
    observed = (float(value_at_step) - float(value_at_x)) / float(eps)
    analytic = float(analytic_directional_derivative)
    relative = abs(observed - analytic) / max(abs(analytic), 1.0)
    return dict(observed_directional_derivative=observed, analytic_directional_derivative=analytic,
                absolute_error=abs(observed - analytic), relative_error=relative,
                limit=relative_limit, passed=bool(np.isfinite(relative) and relative <= relative_limit))


def gradient_agreement(reconstructed, reference, *, relative_l2_limit=COMPONENT_RELATIVE_L2_LIMIT,
                       max_abs_factor=COMPONENT_MAX_ABS_FACTOR):
    difference = np.asarray(reconstructed, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    relative_l2 = float(np.linalg.norm(difference) / max(float(np.linalg.norm(reference)), 1.0))
    max_abs = float(np.max(np.abs(difference)))
    max_abs_limit = max_abs_factor * max(float(np.max(np.abs(reference))), 1.0)
    passed = bool(np.isfinite(relative_l2) and np.isfinite(max_abs)
                  and relative_l2 <= relative_l2_limit and max_abs <= max_abs_limit)
    return dict(relative_l2=relative_l2, relative_l2_limit=relative_l2_limit,
                max_abs=max_abs, max_abs_limit=max_abs_limit, passed=passed)


def diagnostic_status(*, reproduction_passed, component_split, finite_difference,
                      budget_stopped_early):
    if not reproduction_passed:
        return 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
    if component_split == 'mismatch':
        return 'CONDITIONAL_OPTIMIZER_COMPONENT_SPLIT_MISMATCH'
    if finite_difference == 'failed':
        return 'CONDITIONAL_OPTIMIZER_FD_FAILED'
    if finite_difference != 'passed':
        return 'CONDITIONAL_OPTIMIZER_DIAGNOSTIC_INCOMPLETE_BUDGET'
    if budget_stopped_early:
        return 'CONDITIONAL_OPTIMIZER_DIAGNOSTIC_INCOMPLETE_BUDGET'
    if component_split == 'skipped':
        return 'CONDITIONAL_OPTIMIZER_DIAGNOSTIC_COMPLETE_BLOCK_ONLY'
    if component_split == 'passed':
        return 'CONDITIONAL_OPTIMIZER_DIAGNOSTIC_COMPLETE'
    raise ValueError('unknown component-split state')


def merge_component_score_gradient(field_pullback, gradients):
    if len(gradients) != 4:
        raise ValueError('component target must return four input-gradient blocks')
    rho_gradient, velocity_gradient, tracer_gradient, population_gradient = gradients
    ic_gradient, = field_pullback((rho_gradient, velocity_gradient))
    return np.concatenate([
        np.asarray(ic_gradient, dtype=np.float64).reshape(-1),
        np.asarray(tracer_gradient, dtype=np.float64).reshape(-1),
        np.asarray(population_gradient, dtype=np.float64).reshape(-1),
    ])


def _save(path, report, started):
    report['elapsed_seconds'] = time.monotonic() - started
    report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _snapshot_count(name):
    value = os.environ.get(name)
    return None if value is None else int(value)


def _stage_allowed(started, need_seconds, budget_seconds=APP_SECONDS):
    return budget_seconds - (time.monotonic() - started) >= need_seconds


def finite_difference_status(*, reproduction_passed, finite_difference):
    """Status for the best-state finite-difference completion. It does not authorize a fit."""
    if not reproduction_passed:
        return 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
    if finite_difference == 'passed':
        return 'CONDITIONAL_OPTIMIZER_FD_PASSED'
    if finite_difference == 'failed':
        return 'CONDITIONAL_OPTIMIZER_FD_FAILED'
    return 'CONDITIONAL_OPTIMIZER_DIAGNOSTIC_INCOMPLETE_BUDGET'


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU allocation required')
    mode = os.environ['CF4_GPU_MODE']
    expected_kind = {'h200': 'H200', 'h100': 'H100', 'a100': 'A100'}[mode]
    if expected_kind not in str(jax.devices()[0].device_kind).upper():
        raise RuntimeError('allocated device does not match the selected GPU mode')
    expected = os.environ['CF4_EXPECTED_COMMIT']
    actual = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if actual != expected:
        raise RuntimeError('source commit does not match submitted allocation')
    out = Path(os.environ['CF4_R2_OUT_DIR'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'OWNER.txt').write_text(
        'project: CF4\nrepo: /home/kjhan/BACKUP/CF4\n'
        'purpose: conditional-target reproduction and gradient attribution\n'
        'DO_NOT_CANCEL\n')
    fd_only = os.environ.get('CF4_R2_FD_ONLY') == '1'
    budget_seconds = 40 * 60 if fd_only else APP_SECONDS
    started = time.monotonic()
    report_path = out / 'result.json'
    report = dict(
        status='STARTED', job_id=os.environ['SLURM_JOB_ID'], source_commit=expected,
        N=N, box_cMpc_h=BOX, optimizer_called=False, longer_warm_start_authorized=False,
        posterior_sample=False, heldout_scored=False, heldout_count_values_loaded=0,
        heldout_FP_marks_loaded=0, likelihood_changed=False,
        estimand=('same conditional v6 training target as job 413612; no new prior or likelihood term'),
        gates=dict(absolute_term_limit=ABSOLUTE_TERM_LIMIT,
                   relative_objective_limit=RELATIVE_OBJECTIVE_LIMIT,
                   finite_difference_eps=FD_EPS, finite_difference_relative_limit=FD_RELATIVE_LIMIT,
                   component_relative_l2_limit=COMPONENT_RELATIVE_L2_LIMIT,
                   component_max_abs_factor=COMPONENT_MAX_ABS_FACTOR),
        application_seconds=budget_seconds,
        finite_difference_only=fd_only,
        LG_roles=dict(MW='ambiguous', M31='ambiguous', M33='unresolved'),
        Q_GOAL='reproduce and attribute the existing conditional target before any longer fit',
        Q_LEAN='two saved states, best-state component split, one directional finite difference; no sampler or heldout',
    )
    _save(report_path, report, started)
    try:
        stats = jax.devices()[0].memory_stats() or {}
        device_limit = int(stats.get('bytes_limit', 0) or 0)
        report['resource_evidence'] = dict(
            selected_mode=f'{mode} / gpu:{expected_kind}:1',
            h200_free_typed_gpus_at_submit=_snapshot_count('CF4_H200_FREE_TYPED_GPUS'),
            h100_free_typed_gpus_at_submit=_snapshot_count('CF4_H100_FREE_TYPED_GPUS'),
            a100_free_typed_gpus_at_submit=_snapshot_count('CF4_A100_FREE_TYPED_GPUS'),
            measured_prior_exact_GL2_peak_GiB=KNOWN_DEVICE_PEAK_GIB,
            current_device_limit_GiB=float(device_limit / 1024 ** 3) if device_limit else None,
            host_request_GiB=48,
        )
        if device_limit and 1.2 * KNOWN_DEVICE_PEAK_GIB * 1024 ** 3 > device_limit:
            raise MemoryError('measured exact-GL2 device peak lacks 20 percent headroom here')

        saved = json.loads((SAVED / 'result.json').read_text())
        if saved.get('status') != 'CONDITIONAL_MAP_ATTEMPT_INCOMPLETE':
            raise ValueError('frozen conditional MAP artifact is not the incomplete 413612 attempt')
        if len(saved.get('evaluations', ())) != 6:
            raise ValueError('frozen conditional MAP artifact does not have six evaluations')
        saved_initial = saved['evaluations'][0]
        saved_best = saved['evaluations'][5]
        with np.load(SAVED / 'best_parameters.npz', allow_pickle=False) as archive:
            q_best = np.concatenate([
                np.asarray(archive['white_ic'], dtype=np.float64),
                np.asarray(archive['tracer_white'], dtype=np.float64),
                np.asarray(archive['population_white'], dtype=np.float64),
            ])
            stored_objective = float(archive['objective'])
        if q_best.shape != (N_IC + 24,) or not np.isfinite(q_best).all():
            raise ValueError('saved best parameters are not a finite canonical state')
        if abs(stored_objective - float(saved_best['objective'])) > 1e-6:
            raise ValueError('best parameter objective does not match evaluation 6')

        parent = BASE / 'r2_n256_dynamics_profile_v1'
        parent_result = json.loads((parent / 'result.json').read_text())
        if parent_result.get('status') != 'N256_DYNAMICS_INITIALIZER_NOT_POSTERIOR':
            raise ValueError('frozen N256 initializer is absent or has an unexpected status')
        with np.load(parent / 'initial_present_state.npz', allow_pickle=False) as archive:
            q0 = np.concatenate((
                np.asarray(archive['white_ic'], dtype=np.float64).reshape(-1),
                np.zeros(24, dtype=np.float64),
            ))
        if q0.shape != (N_IC + 24,) or not np.isfinite(q0).all():
            raise ValueError('frozen N256 initializer is not a finite canonical state')

        _, _, _, _, source, mix, observation, geometry = load_inputs()
        linked_radii = linked_point_conditioning_radius(observation)
        if (len(mix['PGC']) != 1414 or len(linked_radii) != 1414
                or mix.get('pre_reconciliation_rows') != 1414
                or len(mix.get('excluded_unresolved_PGCs', ())) != 0):
            raise ValueError('frozen v6 linked conditional cohort contract changed')
        with np.load(BASE / 'r2_sky_closed_split_v6/split.npz', allow_pickle=False) as split:
            train_keys = jnp.asarray(split['train_keys'])
            train_counts = jnp.asarray(split['train_counts'])
            heldout_voxels = split['heldout_flat_voxels']
            train_excluded = split['train_window_excluded_keys']
            heldout_excluded = split['heldout_window_excluded_keys']
        if int(np.asarray(train_counts).sum()) != 47121 or len(train_keys) != 37951:
            raise ValueError('frozen v6 training count graph changed')
        exposure, _ = build_population_exposure_masks(
            128, heldout_voxels, train_excluded, heldout_excluded)
        if not np.asarray(exposure)[np.asarray(train_keys)].all():
            raise ValueError('training count key is outside its frozen exposure')
        source = source_geometry_at_resolution(source, N)
        obs = ResolutionObservationTarget(
            N, source, mix, observation, geometry, train_keys, train_counts, jnp.asarray(exposure),
            force_order=1, fine_order=2, max_support_cells=MAX_SUPPORT_CELLS)
        settings = {key: parent_result['settings'][key]
                    for key in ('cosmology', 'a_start', 'a_stop', 'a_nbody_maxstep')}
        settings.update(n=N, box_cMpc_h=BOX)
        common = json.loads((ROOT / 'config/cf4_r2_common_cosmology_v1.json').read_text())['common_cosmology']
        if settings['cosmology']['h'] != 0.746 or settings['cosmology']['Om'] != common['Omega_m']:
            raise ValueError('N256 dynamics cosmology differs from frozen count/CF4 basis')
        evolve, _, conf, _, particle_mass = make_dynamics(settings)
        particle_masses = jnp.full(N_IC, particle_mass)

        @jax.jit
        def field(white):
            positions, velocities = evolve(white)
            state = particle_grid(positions, velocities, particle_masses, conf)
            return state['rho'], jnp.moveaxis(state['mean_velocity_km_s'], -1, 0)

        source_jax = {key: jnp.asarray(value) for key, value in source.items()}
        obs_jax = {key: jnp.asarray(value) for key, value in observation.items()}

        def evaluate_full(q):
            white = jnp.asarray(q[:N_IC])
            tracer = jnp.asarray(q[N_IC:N_IC + 9])
            population = jnp.asarray(q[N_IC + 9:])
            (rho, velocity), pullback = jax.vjp(field, white)
            packs, support_info = obs.support(rho, velocity, tracer, 2)
            (score, components), grads = obs.derivative(
                rho, velocity, tracer, population, packs, source_jax, obs_jax, 2)
            ic_gradient, = pullback((grads[0], grads[1]))
            value, gradient, detail = assemble_conditional_objective(
                q, score, components, ic_gradient, grads[2], grads[3], support_info)
            return dict(value=value, gradient=gradient, detail=detail, rho=rho, velocity=velocity,
                        tracer=tracer, population=population, packs=packs, pullback=pullback, q=q)

        def evaluate_value(q):
            white = jnp.asarray(q[:N_IC])
            tracer = jnp.asarray(q[N_IC:N_IC + 9])
            population = jnp.asarray(q[N_IC + 9:])
            rho, velocity = field(white)
            packs, support_info = obs.support(rho, velocity, tracer, 2)
            score, components = obs.value(
                rho, velocity, tracer, population, packs, source_jax, obs_jax, 2)
            value, detail = conditional_target_terms(q, score, components, support_info)
            return value, detail

        def compare(label, state, saved_row):
            failures = reproduction_failures(saved_row, state['detail'])
            return dict(label=label, objective=state['value'],
                        total_negative_log_target=state['detail']['total_negative_log_target'],
                        terms={key: state['detail'][key] for key in TERM_KEYS},
                        failures=failures, passed=not failures)

        if fd_only:
            report['initializer'] = dict(
                skipped=True,
                reason='job 414485 already reproduced the initializer inside the declared gate')
        else:
            if not _stage_allowed(started, ADJOINT_STAGE_SECONDS, budget_seconds):
                raise TimeoutError('application budget cannot hold the first replay')
            initial = evaluate_full(q0)
            report['initializer'] = compare('initializer', initial, saved_initial)
            _save(report_path, report, started)
            del initial
        if not _stage_allowed(started, ADJOINT_STAGE_SECONDS, budget_seconds):
            raise TimeoutError('application budget cannot hold the best-state replay')
        best = evaluate_full(q_best)
        report['best_state'] = compare('best', best, saved_best)
        reproduction_passed = report['best_state']['passed'] and (
            fd_only or report['initializer']['passed'])
        report['reproduction_passed'] = reproduction_passed
        _save(report_path, report, started)

        component_split = None
        finite_difference = None
        budget_stopped_early = False
        if reproduction_passed and fd_only:
            report['best_joint_gradient'] = block_gradient_summary(best['gradient'])
            finite_difference = None
            if _stage_allowed(started, VALUE_STAGE_SECONDS, budget_seconds):
                gradient = best['gradient']
                norm = float(np.linalg.norm(gradient))
                if not np.isfinite(norm) or norm == 0.0:
                    raise FloatingPointError('best-state gradient has no finite direction')
                direction = gradient / norm
                best.pop('pullback', None)
                best.pop('packs', None)
                best.pop('rho', None)
                best.pop('velocity', None)
                stepped_value, _ = evaluate_value(q_best + FD_EPS * direction)
                agreement = finite_difference_agreement(
                    best['value'], stepped_value, float(np.dot(gradient, direction)), FD_EPS)
                report['finite_difference'] = agreement
                finite_difference = 'passed' if agreement['passed'] else 'failed'
            report['status'] = finite_difference_status(
                reproduction_passed=True, finite_difference=finite_difference)
            report['longer_warm_start_authorized'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], reproduction_passed=True,
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return
        if reproduction_passed:
            report['best_joint_gradient'] = block_gradient_summary(best['gradient'])
            _save(report_path, report, started)
            if not _stage_allowed(started, ADJOINT_STAGE_SECONDS, budget_seconds):
                budget_stopped_early = True
            else:
                component_gradients = {}
                component_split = 'passed'
                for index, name in COMPONENTS:
                    if not _stage_allowed(started, ADJOINT_STAGE_SECONDS):
                        budget_stopped_early = True
                        component_split = None
                        break
                    compiled = obs.component_derivative.lower(
                        best['rho'], best['velocity'], best['tracer'], best['population'],
                        best['packs'], source_jax, obs_jax, 2, index).compile()
                    analysis = compiled.memory_analysis()
                    current = jax.devices()[0].memory_stats() or {}
                    peak = (current.get('bytes_in_use', 0) + analysis.temp_size_in_bytes
                            + analysis.output_size_in_bytes)
                    limit = int(current.get('bytes_limit', 0) or 0)
                    margin = dict(component=name, estimated_peak_GiB=peak / 1024 ** 3,
                                  limit_GiB=limit / 1024 ** 3 if limit else None,
                                  has_20_percent_margin=bool(limit and 1.2 * peak <= limit))
                    report.setdefault('component_memory', []).append(margin)
                    _save(report_path, report, started)
                    if not margin['has_20_percent_margin']:
                        component_split = 'skipped'
                        del compiled
                        break
                    component_score, gradients = compiled(
                        best['rho'], best['velocity'], best['tracer'], best['population'],
                        best['packs'], source_jax, obs_jax)
                    jax.block_until_ready((component_score, gradients))
                    score_gradient = merge_component_score_gradient(best['pullback'], gradients)
                    if score_gradient.shape != q_best.shape or not np.isfinite(score_gradient).all():
                        raise FloatingPointError(f'{name} component gradient is invalid')
                    component_gradients[name] = score_gradient
                    report.setdefault('components', {})[name] = dict(
                        score=float(component_score), **block_gradient_summary(score_gradient))
                    _save(report_path, report, started)
                    del compiled, gradients, score_gradient
                if component_split == 'passed':
                    reconstructed = (q_best - component_gradients['count_2mpp']
                                     - component_gradients['conditional_FP'])
                    agreement = gradient_agreement(reconstructed, best['gradient'])
                    report['component_gradient_agreement'] = agreement
                    if not agreement['passed']:
                        component_split = 'mismatch'
                    del reconstructed
                component_gradients.clear()
            best.pop('pullback', None)
            best.pop('packs', None)
            best.pop('rho', None)
            best.pop('velocity', None)
            best.pop('tracer', None)
            best.pop('population', None)
            if component_split != 'mismatch' and not budget_stopped_early and _stage_allowed(started, VALUE_STAGE_SECONDS):
                gradient = best['gradient']
                norm = float(np.linalg.norm(gradient))
                if not np.isfinite(norm) or norm == 0.0:
                    raise FloatingPointError('best-state gradient has no finite direction')
                direction = gradient / norm
                stepped_value, _ = evaluate_value(q_best + FD_EPS * direction)
                agreement = finite_difference_agreement(
                    best['value'], stepped_value, float(np.dot(gradient, direction)), FD_EPS)
                report['finite_difference'] = agreement
                finite_difference = 'passed' if agreement['passed'] else 'failed'
            elif finite_difference is None and component_split != 'mismatch':
                budget_stopped_early = True
        report['status'] = diagnostic_status(
            reproduction_passed=reproduction_passed, component_split=component_split,
            finite_difference=finite_difference, budget_stopped_early=budget_stopped_early)
        report['longer_warm_start_authorized'] = False
        _save(report_path, report, started)
        print(json.dumps(dict(status=report['status'], reproduction_passed=reproduction_passed,
                              longer_warm_start_authorized=False), allow_nan=False), flush=True)
    except Exception as error:
        report.update(status='FAILED_CONDITIONAL_OPTIMIZER_DIAGNOSTIC', error=repr(error),
                      longer_warm_start_authorized=False)
        _save(report_path, report, started)
        raise


if __name__ == '__main__':
    main()
