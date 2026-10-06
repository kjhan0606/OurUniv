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
    assemble_conditional_objective, conditional_target_gradient, conditional_target_terms,
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


def nuisance_scale(gradient, floor=1.):
    """Positive coordinate scales. The objective stays in the original parameters."""
    gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)
    if gradient.shape != (24,) or not np.isfinite(gradient).all():
        raise ValueError('nuisance scale requires the finite 24-parameter gradient')
    if not np.isfinite(floor) or floor <= 0.:
        raise ValueError('nuisance scale floor must be positive')
    return 1. / np.maximum(np.abs(gradient), float(floor))


def scaled_nuisance(origin, scale, z):
    origin = np.asarray(origin, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if origin.shape != (24,) or scale.shape != (24,) or z.shape != (24,):
        raise ValueError('scaled nuisance step requires three 24-vectors')
    if np.any(scale <= 0.) or not np.isfinite(scale).all():
        raise ValueError('nuisance scales must be finite and positive')
    return origin + scale * z


def pop9_line_status(initial_objective, best_objective, initial_abs_gradient,
                     best_abs_gradient, fp_initial, fp_best):
    """Tenfold drop in the FP-covariance coordinate, without a worse FP term."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(fp_best) >= float(fp_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_POP9_LINE_REDUCED'
    if improved:
        return 'CONDITIONAL_POP9_LINE_IMPROVED'
    return 'CONDITIONAL_POP9_LINE_NO_IMPROVEMENT'


def coordinate_line_action(improved, sign_flipped, reduced, midpoint_used):
    """Stop on a sign change. A failed step gets one midpoint retry, not another outward step."""
    if reduced or sign_flipped:
        return 'stop'
    if improved:
        return 'continue'
    if midpoint_used:
        return 'stop'
    return 'midpoint'


def tracer0_secant(theta_pos, gradient_pos, theta_neg, gradient_neg):
    """One secant step inside a sign bracket. It does not change the objective."""
    theta_pos, theta_neg = float(theta_pos), float(theta_neg)
    gradient_pos, gradient_neg = float(gradient_pos), float(gradient_neg)
    if not (gradient_pos > 0. and gradient_neg < 0.):
        raise ValueError('secant requires opposite derivative signs')
    if theta_pos == theta_neg:
        raise ValueError('secant endpoints must differ')
    theta = theta_pos - gradient_pos * (theta_neg - theta_pos) / (gradient_neg - gradient_pos)
    low, high = sorted((theta_pos, theta_neg))
    if not low < theta < high:
        raise ValueError('secant left the sign bracket')
    return theta


def tracer0_line_status(initial_objective, best_objective, initial_abs_gradient,
                        best_abs_gradient, count_initial, count_best):
    """Classify a fixed-IC line search along the count-amplitude coordinate."""
    improved = float(best_objective) < float(initial_objective)
    reduced = (float(best_abs_gradient) <= float(initial_abs_gradient) / 10.
               and float(count_best) >= float(count_initial) - 1e-6)
    if improved and reduced:
        return 'CONDITIONAL_TRACER0_LINE_AMPLITUDE_REDUCED'
    if improved:
        return 'CONDITIONAL_TRACER0_LINE_IMPROVED'
    return 'CONDITIONAL_TRACER0_LINE_NO_IMPROVEMENT'


def nuisance_block_status(initial_abs_tracer0, final_abs_tracer0, count_initial, count_final):
    """A 10x smaller amplitude gradient with a count term that does not worsen."""
    reduced = float(final_abs_tracer0) <= float(initial_abs_tracer0) / 10.
    count_not_worse = float(count_final) >= float(count_initial) - 1e-6
    if reduced and count_not_worse:
        return 'CONDITIONAL_NUISANCE_BLOCK_AMPLITUDE_REDUCED'
    return 'CONDITIONAL_NUISANCE_BLOCK_NOT_REDUCED'


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
        'purpose: conditional-target gradient check; IC is not refit in nuisance-block mode\n'
        'DO_NOT_CANCEL\n')
    fd_only = os.environ.get('CF4_R2_FD_ONLY') == '1'
    nuisance_block = os.environ.get('CF4_R2_NUISANCE_BLOCK') == '1'
    tracer0_line = os.environ.get('CF4_R2_TRACER0_LINE') == '1'
    tracer0_secant_mode = os.environ.get('CF4_R2_TRACER0_SECANT') == '1'
    pop9_line = os.environ.get('CF4_R2_POP9_LINE') == '1'
    if sum((fd_only, nuisance_block, tracer0_line, tracer0_secant_mode, pop9_line)) > 1:
        raise RuntimeError('conditional diagnostic modes are separate jobs')
    if tracer0_line or tracer0_secant_mode or pop9_line:
        budget_seconds = 70 * 60
    elif fd_only or nuisance_block:
        budget_seconds = 40 * 60
    else:
        budget_seconds = APP_SECONDS
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
        nuisance_block_only=nuisance_block,
        tracer0_line_only=tracer0_line,
        tracer0_secant_only=tracer0_secant_mode,
        pop9_line_only=pop9_line,
        ic_coordinates_fixed=nuisance_block or tracer0_line or tracer0_secant_mode or pop9_line,
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
        if pop9_line:
            secant_result = json.loads(
                (BASE / 'r2_conditional_tracer0_secant_20261007/result.json').read_text())
            if secant_result.get('status') != 'CONDITIONAL_TRACER0_LINE_AMPLITUDE_REDUCED':
                raise ValueError('population-9 line requires the reduced tracer-0 secant')
            saved = secant_result['evaluations'][1]
            if saved.get('step') != 'secant':
                raise ValueError('population-9 line requires the secant evaluation')
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            origin[0] = float(saved['tracer0'])

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            packs, _ = obs.support(rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]))
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, jnp.asarray(origin[:9]), jnp.asarray(origin[9:]), packs,
                source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                built, support_info = obs.support(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]))
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, jnp.asarray(nuisance[:9]), jnp.asarray(nuisance[9:]),
                    built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('population-9 line produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]),
                    population_coordinate_9=float(nuisance[18]),
                    tracer0_gradient=float(gradient[0]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    terms={key: detail[key] for key in TERM_KEYS})

            baseline = evaluate_at(origin)
            gradient_limit = 1e-4 * max(abs(float(saved['population_coordinate_9_gradient'])), 1.)
            tracer_limit = 1e-4 * max(abs(float(saved['tracer0_gradient'])), 1.)
            objective_ok = baseline is not None and abs(
                baseline['objective'] - float(saved['objective'])) / max(abs(float(saved['objective'])), 1.) <= 1e-8
            gradient_ok = baseline is not None and abs(
                baseline['population_coordinate_9_gradient'] - float(saved['population_coordinate_9_gradient'])) <= gradient_limit
            tracer_ok = baseline is not None and abs(
                baseline['tracer0_gradient'] - float(saved['tracer0_gradient'])) <= tracer_limit
            records = []
            if not (objective_ok and gradient_ok and tracer_ok):
                report['baseline'] = baseline
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                records.append(baseline)
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                accepted_gradient = baseline['population_coordinate_9_gradient']
                step = -0.1 if accepted_gradient > 0. else 0.1
                midpoint_used = False
                initial_abs = abs(accepted_gradient)
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[18] = accepted[18] + step
                    row = evaluate_at(proposal)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    improved = row['objective'] < accepted_objective
                    sign_flipped = row['population_coordinate_9_gradient'] * accepted_gradient < 0.
                    reduced = abs(row['population_coordinate_9_gradient']) <= initial_abs / 10.
                    action = coordinate_line_action(improved, sign_flipped, reduced, midpoint_used)
                    if improved:
                        accepted = proposal
                        accepted_objective = row['objective']
                        accepted_gradient = row['population_coordinate_9_gradient']
                    if action == 'continue':
                        step = -0.1 if accepted_gradient > 0. else 0.1
                        midpoint_used = False
                    elif action == 'midpoint':
                        step = 0.5 * step
                        midpoint_used = True
                    else:
                        break
                best = min(records, key=lambda row: row['objective'])
                report['status'] = pop9_line_status(
                    baseline['objective'], best['objective'], initial_abs,
                    abs(best['population_coordinate_9_gradient']),
                    baseline['terms']['conditional_FP_log_likelihood'],
                    best['terms']['conditional_FP_log_likelihood'])
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return
        if tracer0_secant_mode:
            line = json.loads((BASE / 'r2_conditional_tracer0_line_20261007/result.json').read_text())
            if line.get('status') != 'CONDITIONAL_TRACER0_LINE_IMPROVED':
                raise ValueError('tracer0 secant requires the recorded improved line search')
            positive, negative = line['evaluations'][0], line['evaluations'][1]
            theta_star = tracer0_secant(
                positive['tracer0'], positive['tracer0_gradient'],
                negative['tracer0'], negative['tracer0_gradient'])
            report['secant_tracer0'] = theta_star
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            tracer = jnp.asarray(origin[:9])
            population = jnp.asarray(origin[9:])
            packs, _ = obs.support(rho, velocity, tracer, 2)
            base_shapes = shapes_of(packs)
            compiled = obs.derivative.lower(
                rho, velocity, tracer, population, packs, source_jax, obs_jax, 2).compile()

            def evaluate_at(nuisance):
                tracer_i = jnp.asarray(nuisance[:9])
                population_i = jnp.asarray(nuisance[9:])
                built, support_info = obs.support(rho, velocity, tracer_i, 2)
                if shapes_of(built) != base_shapes:
                    return None
                (score, components), grads = compiled(
                    rho, velocity, tracer_i, population_i, built, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('tracer0 secant produced a nonfinite target')
                return dict(
                    objective=value, tracer0=float(nuisance[0]),
                    tracer0_gradient=float(gradient[0]),
                    population_coordinate_9_gradient=float(gradient[18]),
                    terms={key: detail[key] for key in TERM_KEYS})

            def matched(row, saved_row):
                objective_error = abs(row['objective'] - float(saved_row['objective']))
                gradient_error = abs(row['tracer0_gradient'] - float(saved_row['tracer0_gradient']))
                gradient_limit = 1e-4 * max(abs(float(saved_row['tracer0_gradient'])), 1.)
                relative_objective = objective_error / max(abs(float(saved_row['objective'])), 1.)
                return relative_objective <= 1e-8 and gradient_error <= gradient_limit

            records = []
            verify = origin.copy()
            verify[0] = float(negative['tracer0'])
            verified = evaluate_at(verify)
            if verified is None or not matched(verified, negative):
                report['verification'] = verified
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                records.append(verified)
                proposal = origin.copy()
                proposal[0] = theta_star
                secant_row = evaluate_at(proposal)
                if secant_row is None:
                    report['support_shape_changed'] = True
                    report['status'] = 'CONDITIONAL_TRACER0_LINE_IMPROVED'
                else:
                    secant_row['step'] = 'secant'
                    records.append(secant_row)
                    best = min(records, key=lambda row: row['objective'])
                    report['status'] = tracer0_line_status(
                        float(positive['objective']), best['objective'],
                        abs(float(positive['tracer0_gradient'])), abs(best['tracer0_gradient']),
                        float(positive['terms']['count_log_likelihood']),
                        best['terms']['count_log_likelihood'])
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return
        if tracer0_line:
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            report['fixed_ic_evolution_seconds'] = time.monotonic() - started
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()

            def shapes_of(packs):
                return tuple(tuple((key, tuple(np.shape(value))) for key, value in pack.items())
                             for pack in packs)

            def evaluate_at(nuisance, compiled, base_shapes):
                tic = time.monotonic()
                tracer = jnp.asarray(nuisance[:9])
                population = jnp.asarray(nuisance[9:])
                packs, support_info = obs.support(rho, velocity, tracer, 2)
                shapes = shapes_of(packs)
                if shapes != base_shapes:
                    return None, shapes, time.monotonic() - tic
                (score, components), grads = compiled(
                    rho, velocity, tracer, population, packs, source_jax, obs_jax)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                q = np.concatenate((q_best[:N_IC], nuisance))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('tracer0 line search produced a nonfinite target')
                return dict(
                    objective=value, execute_seconds=time.monotonic() - tic,
                    terms={key: detail[key] for key in TERM_KEYS},
                    tracer0=float(nuisance[0]),
                    population_coordinate_9=float(nuisance[18]),
                    tracer0_gradient=float(gradient[0]),
                    population_coordinate_9_gradient=float(gradient[18]),
                ), shapes, time.monotonic() - tic

            tracer = jnp.asarray(origin[:9])
            population = jnp.asarray(origin[9:])
            support_tic = time.monotonic()
            packs, _ = obs.support(rho, velocity, tracer, 2)
            report['baseline_support_seconds'] = time.monotonic() - support_tic
            base_shapes = shapes_of(packs)
            compile_tic = time.monotonic()
            compiled = obs.derivative.lower(
                rho, velocity, tracer, population, packs, source_jax, obs_jax, 2).compile()
            report['derivative_compile_seconds'] = time.monotonic() - compile_tic
            baseline, _, _ = evaluate_at(origin, compiled, base_shapes)
            if baseline is None:
                raise RuntimeError('baseline support changed before its own derivative')
            failures = reproduction_failures(saved_best, baseline['terms'] | {
                'total_negative_log_target': baseline['objective']})
            report['baseline'] = baseline
            report['baseline_reproduction_failures'] = failures
            records = [baseline]
            if failures:
                report['status'] = 'CONDITIONAL_OPTIMIZER_REPRODUCTION_FAILED'
            else:
                accepted = origin.copy()
                accepted_objective = baseline['objective']
                step = -0.1
                for _ in range(3):
                    if budget_seconds - (time.monotonic() - started) < 60.:
                        report['budget_stop'] = True
                        break
                    proposal = accepted.copy()
                    proposal[0] = accepted[0] + step
                    row, shapes, _ = evaluate_at(proposal, compiled, base_shapes)
                    if row is None:
                        report['support_shape_changed'] = True
                        break
                    row['step'] = step
                    records.append(row)
                    if row['objective'] < accepted_objective:
                        accepted = proposal
                        accepted_objective = row['objective']
                        step = -0.1
                        if abs(row['tracer0_gradient']) <= abs(baseline['tracer0_gradient']) / 10.:
                            break
                    elif step == -0.1:
                        step = -0.05
                    else:
                        break
                best = min(records, key=lambda row: row['objective'])
                report['status'] = tracer0_line_status(
                    baseline['objective'], best['objective'], abs(baseline['tracer0_gradient']),
                    abs(best['tracer0_gradient']), baseline['terms']['count_log_likelihood'],
                    best['terms']['count_log_likelihood'])
            report['evaluations'] = records
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return
        if nuisance_block:
            from scipy.optimize import minimize
            white = jnp.asarray(q_best[:N_IC])
            rho, velocity = field(white)
            jax.block_until_ready((rho, velocity))
            report['fixed_ic_evolution_seconds'] = time.monotonic() - started
            origin = np.asarray(q_best[N_IC:], dtype=np.float64).copy()
            records = []

            def nuisance_value_gradient(nuisance):
                nuisance = np.asarray(nuisance, dtype=np.float64)
                q = np.concatenate((q_best[:N_IC], nuisance))
                tracer = jnp.asarray(nuisance[:9])
                population = jnp.asarray(nuisance[9:])
                tic = time.monotonic()
                packs, support_info = obs.support(rho, velocity, tracer, 2)
                (score, components), grads = obs.derivative(
                    rho, velocity, tracer, population, packs, source_jax, obs_jax, 2)
                jax.block_until_ready((score, components, grads[2], grads[3]))
                value, detail = conditional_target_terms(q, score, components, support_info)
                gradient = conditional_target_gradient(
                    q, np.zeros(N_IC), grads[2], grads[3])[N_IC:]
                if not np.isfinite(value) or not np.isfinite(gradient).all():
                    raise FloatingPointError('nuisance block produced a nonfinite target')
                records.append(dict(
                    objective=value,
                    seconds=time.monotonic() - tic,
                    terms={key: detail[key] for key in TERM_KEYS},
                    tracer0=float(nuisance[0]),
                    population9=float(nuisance[9]),
                    tracer0_gradient=float(gradient[0]),
                    population9_gradient=float(gradient[9]),
                    gradient_rms=float(np.sqrt(np.mean(gradient * gradient))),
                    gradient_inf=float(np.max(np.abs(gradient))),
                ))
                return value, gradient, detail

            value0, gradient0, _ = nuisance_value_gradient(origin)
            report['initial_nuisance_evaluation'] = records[-1]
            if records[-1]['seconds'] > 480.:
                report['status'] = 'CONDITIONAL_NUISANCE_BLOCK_TIMING_STOP'
                report['longer_warm_start_authorized'] = False
                _save(report_path, report, started)
                print(json.dumps(dict(status=report['status'],
                                      longer_warm_start_authorized=False), allow_nan=False), flush=True)
                return
            scale = nuisance_scale(gradient0)
            report['nuisance_scale'] = scale.tolist()
            cache = dict(x=origin.copy(), value=value0, gradient=gradient0.copy())

            def scaled_objective(z):
                if budget_seconds - (time.monotonic() - started) < 60.:
                    raise TimeoutError('nuisance-block budget cannot hold another evaluation')
                nuisance = scaled_nuisance(origin, scale, z)
                if np.array_equal(nuisance, cache['x']):
                    return cache['value'], scale * cache['gradient']
                value, gradient, _ = nuisance_value_gradient(nuisance)
                cache.update(x=nuisance.copy(), value=value, gradient=gradient.copy())
                return value, scale * gradient

            try:
                solver = minimize(
                    scaled_objective, np.zeros(24), method='L-BFGS-B', jac=True,
                    options=dict(maxiter=6, maxfun=10, maxls=5, ftol=1e-12, gtol=1e-8))
                report['optimizer'] = dict(
                    success=bool(solver.success), message=str(solver.message),
                    iterations=int(solver.nit), evaluations=int(solver.nfev))
            except TimeoutError as error:
                report['budget_stop'] = repr(error)
            final = records[-1]
            report['evaluations'] = records
            report['ic_max_abs_change'] = 0.
            report['status'] = nuisance_block_status(
                abs(records[0]['tracer0_gradient']), abs(final['tracer0_gradient']),
                records[0]['terms']['count_log_likelihood'],
                final['terms']['count_log_likelihood'])
            report['longer_warm_start_authorized'] = False
            report['joint_map'] = False
            _save(report_path, report, started)
            print(json.dumps(dict(status=report['status'], evaluations=len(records),
                                  longer_warm_start_authorized=False), allow_nan=False), flush=True)
            return

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
