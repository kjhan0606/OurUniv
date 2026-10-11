"""One bounded N256 conditional MAP diagnostic; never a posterior sample."""
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
from cf4_affine_objective import AffineObjective

from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_resolution_target import (
    ResolutionObservationTarget,
    source_geometry_at_resolution,
)

BASE = Path('/gpfs/kjhan/CF4/z0_density')
ROOT = Path(__file__).resolve().parents[1]
N = 256
BOX = 384.0
N_IC = N ** 3
MAX_EVALUATIONS = 8  # Includes initialization and rejected line-search trials.
MAX_ITERATIONS = 4
MAX_LINE_SEARCH = 2
MAX_SUPPORT_CELLS = 131072  # Workspace ceiling only; support is never truncated.
APP_SECONDS = 6000
KNOWN_DEVICE_PEAK_GIB = 30.909375801682472
KNOWN_DEVICE_LIMIT_GIB = 69.81413269042969
TERM_KEYS = (
    'IC_prior_NLL', 'tracer_nuisance_prior_NLL', 'population_nuisance_prior_NLL',
    'count_log_likelihood', 'conditional_FP_log_likelihood',
)


def conditional_target_terms(q, score, components, support_info, n_ic=N_IC):
    """Prior minus the declared count and conditional-FP scores. No new terms."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    components = np.asarray(components, dtype=np.float64).reshape(-1)
    if q.shape != (n_ic + 24,) or components.shape != (2,):
        raise ValueError('canonical coordinates and two likelihood components required')
    count_score, fp_score = map(float, components)
    prior_ic = 0.5 * float(np.dot(q[:n_ic], q[:n_ic]))
    prior_tracer = 0.5 * float(np.dot(q[n_ic:n_ic + 9], q[n_ic:n_ic + 9]))
    prior_population = 0.5 * float(np.dot(q[n_ic + 9:], q[n_ic + 9:]))
    detail = dict(
        IC_prior_NLL=prior_ic,
        tracer_nuisance_prior_NLL=prior_tracer,
        population_nuisance_prior_NLL=prior_population,
        count_log_likelihood=count_score,
        conditional_FP_log_likelihood=fp_score,
        nuisance_prior_NLL=prior_tracer + prior_population,
        total_negative_log_target=prior_ic + prior_tracer + prior_population - count_score - fp_score,
        support=json.dumps(support_info, sort_keys=True, default=lambda item: np.asarray(item).tolist()),
    )
    return prior_ic + prior_tracer + prior_population - float(score), detail


def conditional_target_gradient(q, ic_score_gradient, tracer_score_gradient,
                                population_score_gradient, n_ic=N_IC):
    """NLL gradient is the standard-normal coordinate minus the score gradient."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    ic_score_gradient = np.asarray(ic_score_gradient, dtype=np.float64).reshape(-1)
    tracer_score_gradient = np.asarray(tracer_score_gradient, dtype=np.float64).reshape(-1)
    population_score_gradient = np.asarray(population_score_gradient, dtype=np.float64).reshape(-1)
    if (q.shape != (n_ic + 24,) or ic_score_gradient.shape != (n_ic,)
            or tracer_score_gradient.shape != (9,) or population_score_gradient.shape != (15,)):
        raise ValueError('score-gradient blocks do not match the canonical coordinates')
    gradient = np.empty_like(q)
    gradient[:n_ic] = q[:n_ic] - ic_score_gradient
    gradient[n_ic:n_ic + 9] = q[n_ic:n_ic + 9] - tracer_score_gradient
    gradient[n_ic + 9:] = q[n_ic + 9:] - population_score_gradient
    return gradient


def assemble_conditional_objective(q, score, components, ic_score_gradient,
                                   tracer_score_gradient, population_score_gradient,
                                   support_info, n_ic=N_IC):
    value, detail = conditional_target_terms(q, score, components, support_info, n_ic)
    gradient = conditional_target_gradient(
        q, ic_score_gradient, tracer_score_gradient, population_score_gradient, n_ic)
    return value, gradient, detail


class EvaluationBudgetStop(RuntimeError):
    pass


class BudgetedObjective:
    """Memoize duplicate requests and cap all actual target evaluations."""

    def __init__(self, evaluate, *, max_evaluations, deadline, on_evaluation=None):
        if max_evaluations < 1:
            raise ValueError('positive exact-target evaluation budget required')
        self.evaluate = evaluate
        self.max_evaluations = int(max_evaluations)
        self.deadline = float(deadline)
        self.on_evaluation = on_evaluation
        self.records = []
        self.best = None
        self._last_x = None
        self._last_result = None

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self._last_x is not None and np.array_equal(x, self._last_x):
            value, gradient, _ = self._last_result
            return value, gradient.copy()
        if len(self.records) >= self.max_evaluations:
            raise EvaluationBudgetStop('exact-target evaluation ceiling reached')
        if time.monotonic() >= self.deadline:
            raise EvaluationBudgetStop('application walltime ceiling reached')

        value, gradient, detail = self.evaluate(x)
        value = float(value)
        gradient = np.asarray(gradient, dtype=np.float64)
        if (not np.isfinite(value) or gradient.shape != x.shape
                or not np.isfinite(gradient).all()):
            raise FloatingPointError('nonfinite conditional target value/gradient')
        row = dict(evaluation=len(self.records) + 1,
                   objective=value,
                   gradient_inf=float(np.max(np.abs(gradient))),
                   gradient_rms=float(np.sqrt(np.mean(gradient * gradient))),
                   optimizer_accepted=False,
                   **detail)
        self.records.append(row)
        self._last_x = x.copy()
        self._last_result = (value, gradient.copy(), row)
        if self.best is None or value < self.best['objective']:
            self.best = dict(x=x.copy(), objective=value, gradient=gradient.copy(),
                             detail=row.copy())
            row['new_best'] = True
        else:
            row['new_best'] = False
        if self.on_evaluation is not None:
            self.on_evaluation(x, row, self.best)
        return value, gradient.copy()


def _save_report(path, report, started):
    report['elapsed_seconds'] = time.monotonic() - started
    report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def _resource_snapshot_count(name):
    value = os.environ.get(name)
    return None if value is None else int(value)


def main(*, start_checkpoint=None, max_evaluations=MAX_EVALUATIONS,
         max_iterations=MAX_ITERATIONS, max_line_search=MAX_LINE_SEARCH,
         maxcor=3, app_seconds=APP_SECONDS, nuisance_scale=1.0,
         support_chunk_cells=None, recorded_trial=None, sampling_driver=None):
    if recorded_trial is not None and (start_checkpoint is not None or max_evaluations != 1):
        raise ValueError('recorded trial is a one-evaluation diagnostic, not an optimizer restart')
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU allocation required')
    expected = os.environ['CF4_EXPECTED_COMMIT']
    actual = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    if actual != expected:
        raise RuntimeError('source commit does not match submitted allocation')
    out = Path(os.environ['CF4_R2_OUT_DIR'])
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report_path = out / 'result.json'
    report = dict(
        status='STARTED', job_id=os.environ['SLURM_JOB_ID'], source_commit=expected,
        N=N, box_cMpc_h=BOX, field_grid_spacing_cMpc_h=1.5,
        observed_count_grid_spacing_cMpc_h=3.0, R2_complete=False,
        posterior_sample=False, posterior_uncertainty=False, heldout_scored=False,
        heldout_count_values_loaded=0, heldout_FP_marks_loaded=0,
        conditional_count_rows=47121, conditional_FP_rows=1414,
        estimand=('conditional v6 training-field MAP diagnostic: linked FP mark factors '
                  'given observed 2M++ point and observed mark presence'),
        selection_limit=('mark-presence ignorability is not established; no full CF4 '
                         'group-incidence/survival calibration'),
        target_rules=('count occurrence once; no extra f_n or group-incidence factor; '
                      'current LCDM and 24 standard-normal nuisance priors once'),
        target_orders=dict(value=2, gradient=2, matched_primal=True),
        support_rule='rebuild shifted-source support for every distinct optimizer candidate; existing 8-sigma truncation remains approximate',
        optimizer=dict(method='L-BFGS-B', max_exact_value_gradient_evaluations=max_evaluations,
                       max_iterations=max_iterations, max_line_search_trials=max_line_search,
                       maxcor=maxcor, application_seconds=app_seconds),
        pmwd_forward_evolution_per_target_evaluation=True,
        separate_ramses_or_standalone_simulation=False,
        physical_velocity_variance_in_likelihood=False,
        physical_velocity_variance_role='same-state readout only; not posterior uncertainty or tracer sigma_los',
        angular_selection='frozen N128 map replicated to N256 source nodes; dx=1.5 is field sampling, not proven observational resolution',
        initialized_nuisances='all 24 current standard-normal coordinates at zero; not a continuation of older fitted nuisance coordinates',
        IC_lineage='fixed N256 initializer; inherited N128 low modes are nonstationary historical coordinates, new high modes are one prior draw',
        LG_roles=dict(MW='ambiguous', M31='ambiguous', M33='unresolved'),
        LG_requirement='MW/M31/M33 observables must later constrain these roles on the same NEW evolved LG field at <=0.3 cMpc/h; native truth IDs calibration/evaluation only',
        Q_GOAL='one conditional actual-data z=0 field diagnostic; does not deliver a calibrated posterior or LG result',
        Q_LEAN='one bounded exact-target optimization; no new data, source census, holdout score, long sampler or separate simulation',
        evaluations=[],
    )
    _save_report(report_path, report, started)
    objective = None

    try:
        gpu_mode = os.environ.get('CF4_GPU_MODE', 'h200')
        if (gpu_mode not in ('h200', 'h100', 'a100')
                or gpu_mode.upper() not in str(jax.devices()[0].device_kind).upper()):
            raise RuntimeError('allocation does not match the requested typed GPU mode')
        stats = jax.devices()[0].memory_stats() or {}
        device_limit = stats.get('bytes_limit', 0)
        h200_free = _resource_snapshot_count('CF4_H200_FREE_TYPED_GPUS')
        report['resource_evidence'] = dict(
            selected_mode=f'{gpu_mode} / gpu:{gpu_mode.upper()}:1',
            h200_was_checked=h200_free is not None,
            h200_idle=None if h200_free is None else h200_free > 0,
            h200_free_typed_gpus_at_submit=h200_free,
            h100_free_typed_gpus_at_submit=_resource_snapshot_count('CF4_H100_FREE_TYPED_GPUS'),
            a100_free_typed_gpus_at_submit=_resource_snapshot_count('CF4_A100_FREE_TYPED_GPUS'),
            measured_prior_exact_GL2_peak_GiB=KNOWN_DEVICE_PEAK_GIB,
            measured_prior_device_limit_GiB=KNOWN_DEVICE_LIMIT_GIB,
            current_device_limit_GiB=float(device_limit / 1024 ** 3) if device_limit else None,
            host_request_GiB=(int(os.environ['SLURM_MEM_PER_NODE']) / 1024
                              if os.environ.get('SLURM_MEM_PER_NODE') else None),
            previous_joint_pilot_host_peak_GiB=15.260330200195312,
            lbfgs_history_estimate_GiB=2 * maxcor * (N_IC + 24) * 8 / 1024 ** 3,
            host_request_has_20_percent_margin=None,
        )
        if device_limit and 1.2 * KNOWN_DEVICE_PEAK_GIB * 1024 ** 3 > device_limit:
            raise MemoryError('measured exact-GL2 device peak lacks 20 percent headroom here')

        parent = BASE / 'r2_n256_dynamics_profile_v1'
        parent_result = json.loads((parent / 'result.json').read_text())
        if parent_result.get('status') != 'N256_DYNAMICS_INITIALIZER_NOT_POSTERIOR':
            raise ValueError('frozen N256 initializer is absent or has an unexpected status')
        with np.load(parent / 'initial_present_state.npz', allow_pickle=False) as f:
            white0 = np.asarray(f['white_ic'], dtype=np.float64).copy()
            initial_rho = np.asarray(f['rho'], dtype=np.float64).copy()
            initial_velocity = np.asarray(f['mean_velocity_km_s'], dtype=np.float64).copy()
        if (white0.shape != (N_IC,) or initial_rho.shape != (N, N, N)
                or initial_velocity.shape != (3, N, N, N)):
            raise ValueError('frozen N256 initializer shape mismatch')
        q0 = np.concatenate((white0, np.zeros(24, dtype=np.float64)))
        del white0
        if q0.shape != (N_IC + 24,):
            raise ValueError('canonical N256 IC plus 24 current nuisance coordinates required')
        replay = None
        replay_gradient = None
        trial_reference = None
        if recorded_trial is not None:
            recorded_trial = Path(recorded_trial)
            provenance = json.loads((recorded_trial.parent / 'result.json').read_text())
            if (recorded_trial != BASE / 'r2_joint_warm_start_20261010/best_parameters.npz'
                    or provenance.get('source_commit') != '4c7ef5133fda4fdf6aa542632c2f9da931694399'
                    or provenance.get('N') != N or provenance.get('box_cMpc_h') != BOX
                    or provenance.get('best_evaluation') != 8
                    or provenance.get('target_orders') != dict(value=2, gradient=2, matched_primal=True)):
                raise ValueError('recorded trial lineage is not the known conditional-v6 evaluation 8')
            trial_reference = provenance['evaluations'][7]
            with np.load(recorded_trial, allow_pickle=False) as saved:
                if bool(saved['optimizer_accepted']) or abs(float(saved['objective']) - trial_reference['objective']) > 1e-6:
                    raise ValueError('recorded trial metadata mismatch')
                q0 = np.concatenate((saved['white_ic'], saved['tracer_white'], saved['population_white']))
            if q0.shape != (N_IC + 24,) or not np.isfinite(q0).all():
                raise ValueError('recorded trial coordinates invalid')
            report.update(recorded_trial=str(recorded_trial), recorded_trial_evaluation=8,
                          recorded_trial_optimizer_accepted=False,
                          initialized_nuisances='saved trial 8 nuisance coordinates; unchanged priors',
                          IC_lineage='unaccepted recorded diagnostic trial; not a posterior draw')
        if start_checkpoint is not None:
            start_checkpoint = Path(start_checkpoint)
            replay = json.loads((start_checkpoint.parent / 'result.json').read_text())
            hmc_replay = False
            if (sampling_driver is not None
                    and replay.get('status') == 'CONDITIONAL_HMC_TRANSPORT_DIAGNOSTIC_NOT_POSTERIOR'
                    and replay.get('source_commit') == 'c4ea6cf149dc4759d1c4cc6226a46502c2e3f1d5'
                    and start_checkpoint == BASE / 'r2_hmc_transport_20261011/hmc_retained_state.npz'
                    and replay.get('checkpoint_replay_passed') is True
                    and replay.get('target_orders') == dict(value=2, gradient=2, matched_primal=True)):
                with np.load(start_checkpoint, allow_pickle=False) as saved:
                    terminal_value = float(saved['objective'])
                    terminal_terms = json.loads(str(saved['current_terms']))
                terminal = replay['sampler_trace'][-1]
                if (abs(terminal_value-terminal['objective']) > 1e-6
                        or terminal_terms != terminal['current_terms']):
                    raise ValueError('retained transport state has mismatched terms')
                hmc_replay = True
                replay = dict(replay, likelihood_changed=False,
                              objective=terminal_value, terms=terminal_terms)
                report['restart_kind'] = 'terminal retained transport state; not best evaluation'
            if (sampling_driver is not None
                    and start_checkpoint == BASE / 'r2_conditional_hmc_step010_20261011/hmc_retained_state.npz'
                    and replay.get('status') == 'CONDITIONAL_HMC_TRANSITION_PILOT_NOT_POSTERIOR'
                    and replay.get('accepted_proposals') == 2
                    and replay.get('source_commit') == 'a7f0a071fac14abe18e8dea1ed44f00ae12dcfdf'
                    and replay.get('target_orders') == dict(value=2, gradient=2, matched_primal=True)
                    and replay.get('checkpoint_replay_passed') is True):
                # Known terminal retained state, not the MAP best-evaluation file.
                with np.load(start_checkpoint, allow_pickle=False) as saved:
                    terminal_value = float(saved['objective'])
                terminal = replay['evaluations'][-1]
                if abs(terminal_value-terminal['objective']) > 1e-6:
                    raise ValueError('terminal retained HMC state has mismatched terms')
                hmc_replay = True
                replay = dict(replay, likelihood_changed=False,
                              objective=terminal_value, terms=terminal)
                report['restart_kind'] = 'terminal retained HMC state, including original RNG'
            if ((not hmc_replay and replay.get('status') != 'CONDITIONAL_FULL_GRADIENT_RECORDED')
                    or replay.get('N') != N or replay.get('box_cMpc_h') != BOX
                    or replay.get('likelihood_changed') is not False):
                raise ValueError('restart requires a recorded gradient of the same conditional target')
            with np.load(start_checkpoint, allow_pickle=False) as saved:
                q0 = np.asarray(saved['q'], dtype=np.float64).copy()
                replay_gradient = np.asarray(saved['gradient'], dtype=np.float64).copy()
            if (q0.shape != (N_IC + 24,) or replay_gradient.shape != q0.shape
                    or not np.isfinite(q0).all() or not np.isfinite(replay_gradient).all()):
                raise ValueError('restart coordinates/gradient are invalid')
            report.update(start_checkpoint=str(start_checkpoint),
                          initialized_nuisances='saved conditional gradient checkpoint; unchanged priors',
                          IC_lineage='saved conditional diagnostic, not a posterior draw',
                          optimizer_coordinates=dict(IC_scale=1.0,
                              nuisance_scale=np.broadcast_to(nuisance_scale, (24,)).tolist(),
                              rule='q=q_start+D*z; fixed D; same target and chain-rule gradient'),
                          longer_warm_start_authorized=True)

        rho128, velocity128, _, _, source, mix, observation, geometry = load_inputs()
        del rho128, velocity128
        from cf4_r2_resolution_target import linked_point_conditioning_radius
        linked_radii = linked_point_conditioning_radius(observation)
        if (len(mix['PGC']) != 1414 or len(linked_radii) != 1414
                or mix.get('pre_reconciliation_rows') != 1414
                or len(mix.get('excluded_unresolved_PGCs', ())) != 0):
            raise ValueError('frozen v6 linked conditional cohort contract changed')
        with np.load(BASE / 'r2_sky_closed_split_v6/split.npz', allow_pickle=False) as f:
            train_keys = jnp.asarray(f['train_keys'])
            train_counts = jnp.asarray(f['train_counts'])
            # These are geometric masks only; no heldout count or mark values are loaded.
            heldout_voxels = f['heldout_flat_voxels']
            train_excluded = f['train_window_excluded_keys']
            heldout_excluded = f['heldout_window_excluded_keys']
        if (int(np.asarray(train_counts).sum()) != 47121 or len(train_keys) != 37951):
            raise ValueError('frozen v6 training count graph changed')
        exposure, _ = build_population_exposure_masks(
            128, heldout_voxels, train_excluded, heldout_excluded)
        if not np.asarray(exposure)[np.asarray(train_keys)].all():
            raise ValueError('training count key is outside its frozen exposure')
        report.update(
            count_training_rows=int(np.asarray(train_counts).sum()),
            count_training_keys=len(train_keys),
            linked_FP_rows=len(mix['PGC']),
            linked_radius_range_cMpc_h=[float(linked_radii.min()), float(linked_radii.max())],
            source_conditioning_rule=mix['source_conditioning_rule'],
            association_ledger_source_commit=mix['association_ledger_source_commit'],
            split_source='r2_sky_closed_split_v6; heldout values not accessed',
        )
        _save_report(report_path, report, started)

        source = source_geometry_at_resolution(source, N)
        obs = ResolutionObservationTarget(
            N, source, mix, observation, geometry, train_keys, train_counts, jnp.asarray(exposure),
            force_order=1, fine_order=2, max_support_cells=MAX_SUPPORT_CELLS,
            support_chunk_cells=support_chunk_cells)
        report['support_chunk_cells']=support_chunk_cells
        settings = {k: parent_result['settings'][k]
                    for k in ('cosmology', 'a_start', 'a_stop', 'a_nbody_maxstep')}
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

        @jax.jit
        def moments(white):
            positions, velocities = evolve(white)
            state = particle_grid(positions, velocities, particle_masses, conf)
            return (state['rho'], jnp.moveaxis(state['mean_velocity_km_s'], -1, 0),
                    jnp.moveaxis(state['variance_km2_s2'], -1, 0), state['valid'])

        source_jax = {k: jnp.asarray(v) for k, v in source.items()}
        obs_jax = {k: jnp.asarray(v) for k, v in observation.items()}
        start_field = None
        def evaluate(q):
            nonlocal start_field
            white = jnp.asarray(q[:N_IC])
            tracer = jnp.asarray(q[N_IC:N_IC + 9])
            population = jnp.asarray(q[N_IC + 9:])
            (rho, velocity), pullback = jax.vjp(field, white)
            if start_field is None:
                start_field = (np.asarray(rho).copy(), np.asarray(velocity).copy())
            packs, support_info = obs.support(rho, velocity, tracer, 2)
            (score, components), grads = obs.derivative(
                rho, velocity, tracer, population, packs, source_jax, obs_jax, 2)
            ic_gradient, = pullback((grads[0], grads[1]))
            return assemble_conditional_objective(
                q, score, components, ic_gradient, grads[2], grads[3], support_info)

        def persist_report(_x, row, best):
            report['evaluations'] = objective.records.copy()
            report['exact_target_evaluations'] = len(objective.records)
            if best is not None:
                report['best_evaluation'] = best['detail']['evaluation']
                report['best_objective'] = best['objective']
                if row['new_best']:
                    np.savez(out / 'best_parameters.npz', white_ic=best['x'][:N_IC],
                             tracer_white=best['x'][N_IC:N_IC + 9],
                             population_white=best['x'][N_IC + 9:],
                             objective=best['objective'], optimizer_accepted=False)
            _save_report(report_path, report, started)

        objective = BudgetedObjective(
            evaluate, max_evaluations=max_evaluations,
            deadline=started + app_seconds, on_evaluation=persist_report)
        solver = None
        budget_stop = None
        optimizer_objective = objective
        optimizer_start = q0
        if trial_reference is not None:
            trial_value, trial_gradient = objective(q0)
            if (abs(trial_value - trial_reference['objective']) > 1e-8 * abs(trial_reference['objective'])
                    or any(abs(objective.records[0][k] - trial_reference[k]) > 1e-3 for k in TERM_KEYS)):
                raise AssertionError('recorded trial objective terms changed')
            report['recorded_trial_terms_replay_passed'] = True
            report['recorded_trial_full_gradient_reference_available'] = False
            del trial_gradient
        if replay is not None:
            replay_value, live_gradient = objective(q0)
            live_terms = objective.records[0]
            if (abs(replay_value - replay['objective']) > 1e-8 * abs(replay['objective'])
                    or any(abs(live_terms[k] - replay['terms'][k]) > 1e-3 for k in TERM_KEYS)
                    or not np.allclose(live_gradient, replay_gradient, rtol=1e-4, atol=1e-4)):
                raise AssertionError('checkpoint replay does not match the frozen target/gradient')
            report['checkpoint_replay_passed'] = True
            del replay_gradient, live_gradient
            scale = np.ones_like(q0)
            scale[N_IC:] = nuisance_scale
            optimizer_objective = AffineObjective(objective, q0, scale, replay_value)
            optimizer_start = np.zeros_like(q0)
            del scale

        if sampling_driver is not None:
            if replay is None or recorded_trial is not None:
                raise ValueError('sampling pilot requires the replayed canonical gradient checkpoint')
            sampling_driver(objective, q0, out, report,
                            lambda: _save_report(report_path, report, started))
            print(json.dumps(report, allow_nan=False), flush=True)
            return

        def accepted_step(z):
            x = (optimizer_objective.physical(z)
                 if isinstance(optimizer_objective, AffineObjective) else np.asarray(z))
            # Callback metadata must not trigger another expensive evaluation.
            if not np.array_equal(x, objective._last_x):
                raise RuntimeError('accepted optimizer state is absent from the evaluation cache')
            row = objective.records[-1]
            row['optimizer_accepted'] = True
            report.setdefault('accepted_evaluations', []).append(row['evaluation'])
            np.savez(out / 'accepted_parameters.npz', white_ic=x[:N_IC],
                     tracer_white=x[N_IC:N_IC + 9], population_white=x[N_IC + 9:],
                     objective=row['objective'], optimizer_accepted=True)
            if np.array_equal(x, objective.best['x']):
                objective.best['detail']['optimizer_accepted'] = True
                np.savez(out / 'best_parameters.npz', white_ic=x[:N_IC],
                         tracer_white=x[N_IC:N_IC + 9], population_white=x[N_IC + 9:],
                         objective=row['objective'], optimizer_accepted=True)
            report['evaluations'] = objective.records.copy()
            _save_report(report_path, report, started)

        try:
            solver = minimize(
                optimizer_objective, optimizer_start, method='L-BFGS-B', jac=True,
                callback=accepted_step,
                options=dict(maxiter=max_iterations, maxfun=max_evaluations,
                             maxls=max_line_search, maxcor=maxcor,
                             ftol=0.0 if replay is not None else 1e-9, gtol=1e-4))
        except EvaluationBudgetStop as stop:
            budget_stop = str(stop)

        if objective.best is None:
            report.update(status='CONDITIONAL_MAP_NO_FINITE_TARGET_EVALUATION',
                          budget_stop=budget_stop)
            _save_report(report_path, report, started)
            return

        best = objective.best
        white_best = jnp.asarray(best['x'][:N_IC])
        rho, velocity, variance, valid = moments(white_best)
        rho, velocity, variance, valid = map(np.asarray, (rho, velocity, variance, valid))
        if not all(np.isfinite(x).all() for x in (rho, velocity, variance)):
            raise FloatingPointError('best-state present-field moments are nonfinite')
        if abs(float(rho.mean()) - 1.0) > 1e-10 or np.min(rho) < 0:
            raise AssertionError('best-state density is not nonnegative and mass-conserving')
        qwhite = best['x'][:N_IC]
        report.update(
            status=('CONDITIONAL_MAP_OPTIMIZER_CONVERGED_NOT_POSTERIOR'
                    if solver is not None and solver.success
                    and float(np.max(np.abs(best['gradient']))) <= 1e-4
                    else 'CONDITIONAL_MAP_ATTEMPT_INCOMPLETE'),
            budget_stop=budget_stop,
            optimizer=None if solver is None else dict(
                success=bool(solver.success), status=int(solver.status),
                message=str(solver.message), iterations=int(solver.nit),
                scipy_function_evaluations=int(solver.nfev),
                best_gradient_inf=float(np.max(np.abs(best['gradient']))),
                convergence_requires_gradient_inf_le_1e_4=True),
            exact_target_evaluations=len(objective.records),
            initial_objective=objective.records[0]['objective'],
            best_objective=best['objective'],
            best_is_optimizer_accepted=bool(best['detail'].get('optimizer_accepted', False)),
            objective_improvement=objective.records[0]['objective'] - best['objective'],
            initial_components={k: objective.records[0][k] for k in (
                'IC_prior_NLL', 'tracer_nuisance_prior_NLL', 'population_nuisance_prior_NLL',
                'count_log_likelihood', 'conditional_FP_log_likelihood')},
            best_components={k: best['detail'][k] for k in (
                'IC_prior_NLL', 'tracer_nuisance_prior_NLL', 'population_nuisance_prior_NLL',
                'count_log_likelihood', 'conditional_FP_log_likelihood')},
            initial_white_mean_square=float(np.mean(q0[:N_IC] ** 2)),
            best_white_mean_square=float(np.mean(qwhite ** 2)),
            best_density_mean=float(rho.mean()), best_density_min=float(rho.min()),
            best_density_max=float(rho.max()),
            density_change_rms_from_initializer=float(np.sqrt(np.mean((rho - initial_rho) ** 2))),
            velocity_change_rms_km_s=np.sqrt(np.mean((velocity - initial_velocity) ** 2, axis=(1, 2, 3))).tolist(),
            density_change_rms_from_run_start=float(np.sqrt(np.mean((rho - start_field[0]) ** 2))),
            velocity_change_rms_from_run_start_km_s=np.sqrt(
                np.mean((velocity - start_field[1]) ** 2, axis=(1, 2, 3))).tolist(),
            physical_mass_weighted_dispersion_km_s=np.sqrt(
                (variance * rho[None, ...]).sum(axis=(1, 2, 3)) / rho.sum()).tolist(),
            physical_dispersion_not_in_likelihood=True,
            PMWD_candidate_evolutions=len(objective.records) + 1,
            canonical_gradient_blocks={name: dict(
                rms=float(np.sqrt(np.mean(best['gradient'][region] ** 2))),
                inf=float(np.max(np.abs(best['gradient'][region]))))
                for name, region in (('IC', slice(0, N_IC)),
                                     ('tracer', slice(N_IC, N_IC + 9)),
                                     ('population', slice(N_IC + 9, None)))},
            IC_update_rms=float(np.sqrt(np.mean((qwhite - q0[:N_IC]) ** 2))),
        )
        np.savez(out / 'conditional_best_field.npz', white_ic=qwhite,
                 tracer_white=best['x'][N_IC:N_IC + 9],
                 population_white=best['x'][N_IC + 9:], rho=rho.astype(np.float32),
                 mean_velocity_km_s=velocity.astype(np.float32),
                 physical_velocity_variance_km2_s2=variance.astype(np.float32),
                 velocity_valid=valid, box_cMpc_h=BOX,
                 native_mesh_origin_fraction=0., R2_complete=False,
                 optimizer_accepted=bool(best['detail'].get('optimizer_accepted', False)),
                 posterior_sample=False)
        _save_report(report_path, report, started)
        print(json.dumps(report, allow_nan=False), flush=True)
    except Exception as error:
        if objective is not None:
            report['evaluations'] = objective.records.copy()
            report['exact_target_evaluations'] = len(objective.records)
            if objective.best is not None:
                report['best_evaluation'] = objective.best['detail']['evaluation']
                report['best_objective'] = objective.best['objective']
        report.update(status='FAILED_CONDITIONAL_MAP_DIAGNOSTIC', error=repr(error))
        _save_report(report_path, report, started)
        raise


if __name__ == '__main__':
    main()
