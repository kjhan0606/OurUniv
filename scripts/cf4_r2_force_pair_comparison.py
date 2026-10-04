"""Four matched terminal-state force trials; not posterior sampling or UQ."""
import json
import math
import os
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_affine_force import AffineCorrectedForce
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_prior_split_hmc import (
    FixedSplitMetric, inverse_laplacian_metric_symbol, restore_numpy_rng, split_hmc_step,
    split_trajectory,
)
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_resolution_target import (source_geometry_at_resolution,
    ResolutionObservationTarget, linked_point_conditioning_radius)


BASE = Path('/gpfs/kjhan/CF4/z0_density')
N = 256
BOX = 384.0
NIC = N**3
STEP = 0.08
STEPS = 8
APPLICATION_BUDGET = 10 * 3600
MIN_CALL_REMAINING = 1800  # measured GL2-gradient calls were <=18.8 min including compilation


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    out = Path(os.environ['CF4_R2_OUT_DIR'])
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    reanchor_only = os.environ.get('CF4_R2_REANCHOR_ONLY') == '1'
    exact_one_step = os.environ.get('CF4_R2_EXACT_ONE_STEP') == '1'
    exact_chain_steps = int(os.environ.get('CF4_R2_EXACT_CHAIN_STEPS', '0'))
    exact_chain_integrations = int(os.environ.get('CF4_R2_CHAIN_INTEGRATION_STEPS', '1'))
    trajectory_length_text = os.environ.get('CF4_R2_TRAJECTORY_LENGTHS')
    trajectory_lengths = (() if trajectory_length_text is None else
        tuple(int(part.strip()) for part in trajectory_length_text.split(',')))
    fundamental_mass = float(os.environ.get('CF4_R2_FUNDAMENTAL_MASS', '6000'))
    if (exact_chain_steps < 0 or sum((reanchor_only, exact_one_step,
            exact_chain_steps > 0, bool(trajectory_lengths))) > 1):
        raise ValueError('select one valid force-pair or exact-chain mode')
    if trajectory_length_text is not None and trajectory_lengths != (1, 2, 4):
        raise ValueError('trajectory diagnostic requires the declared 1,2,4 step sweep')
    if (exact_chain_steps and exact_chain_integrations < 1) or not math.isfinite(fundamental_mass) or fundamental_mass < 1:
        raise ValueError('invalid exact-chain length or positive fundamental mass')
    application_budget = (4*3600 if (reanchor_only or exact_one_step) else
                          3.5*3600 if (exact_chain_steps or trajectory_lengths) else
                          APPLICATION_BUDGET)
    report = dict(
        status='STARTED', job_id=os.environ['SLURM_JOB_ID'], N=N, box_cMpc_h=BOX,
        dx_cMpc_h=BOX/N, force_comparison='GL2 exact force vs frozen GL1 affine force',
        fine_target='N256 GL2 target conditioned on aligned secure linked-point radii; exact fine Hamiltonian within this run',
        historical_force_pair_comparable=False,
        target_decomposition='prior energy - training-count log score - conditional raw-mark log score',
        matched_settings=dict(step_size=STEP,
                             integration_steps=exact_chain_integrations if exact_chain_steps else STEPS,
                             inverse_laplacian_metric_fundamental_mass=fundamental_mass,
                             nuisance_inverse_mass_diagonal=1e-5),
        matched_momenta=True, heldout_scored=False, posterior_claim=False,
        Q_GOAL='local sampler diagnosis for the actual R2 z=0 field posterior; no LG identification claim',
        Q_LEAN='two fixed endpoints; reuse the same forward model, no new cosmological simulation or particle histories',
        MW_M31='ambiguous; M33 unresolved; native truth IDs cannot seed or select generated components',
        evaluations=[], trials=[], application_budget_seconds=application_budget)
    if reanchor_only:
        report['force_comparison'] = 'terminal-state re-anchored GL1 affine vs saved force-pair references'
        report['reanchor_policy'] = 'one GL2 tangent per predetermined terminal state; correction frozen for one trajectory'
        report['prior_force_pair_result'] = str(BASE/'r2_n256_force_pair_v1/result.json')
        report['Q_LEAN'] = 'two predetermined endpoints, one tangent and one trajectory each; reuse prior comparison, no new cosmological simulation or particle histories'
        report['anchor_depends_on_start_state'] = True
        report['transition_probability_valid_for_chain'] = False
    if exact_one_step:
        report['force_comparison'] = 'state-independent exact GL2 one-step force vs saved exact GL2 eight-step reference'
        report['Q_LEAN'] = 'two predetermined endpoints, one exact GL2 one-step trajectory each; reuse prior comparison, no new cosmological simulation or particle histories'
        report['transition_probability_valid_for_chain'] = True
    if exact_chain_steps:
        report.update(
            force_comparison='state-independent exact GL2 HMC chain',
            chain_settings=dict(step_size=STEP, integration_steps=exact_chain_integrations,
                requested_transitions=exact_chain_steps, warmup=0, adaptation=False,
                inverse_laplacian_fundamental_mass_parameter=fundamental_mass,
                fixed_metric='inverse-Laplacian IC symbol plus unchanged nuisance mass'),
            posterior_claim=False, stationarity_claimed=False, heldout_scored=False,
            Q_GOAL='advance the actual N256 z=0 field posterior with target-preserving transitions; not a final map or LG identification',
            Q_LEAN='one fixed metric and one exact GL2 trajectory per transition; no new gravity simulation or proxy ladder',
            MW_M31='remain role-ambiguous; this bundle does not identify either component',
            M33='unresolved; later observables must constrain this same NEW field; native truth identities are evaluation-only',
            transition_probability_valid_for_chain=True)
    if trajectory_lengths:
        report.update(
            force_comparison='same-state exact GL2 trajectory-length diagnostic',
            trajectory_length_diagnostic=dict(
                requested_integration_steps=list(trajectory_lengths),
                step_size=STEP, warmup=0, retained_samples=0,
                common_momentum=True, metric_fundamental_mass=fundamental_mass),
            posterior_claim=False, stationarity_claimed=False, heldout_scored=False,
            Q_GOAL='diagnose target-preserving N256 sampler movement for the actual R2 z=0 field posterior; no density-map or LG-identification claim',
            Q_LEAN='three exact GL2 paths from one state/common momentum; no chain extension, new likelihood, gravity run or heldout access',
            MW_M31='remain role-ambiguous; this bundle does not identify either component',
            M33='unresolved; later observables must constrain this same NEW evolved field; truth identities are evaluation-only',
            transition_probability_valid_for_chain=False)

    def save():
        report['seconds'] = time.monotonic() - started
        tmp = out/'result.json.tmp'
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        tmp.replace(out/'result.json')

    def check_budget():
        remaining = application_budget - (time.monotonic() - started)
        if remaining < MIN_CALL_REMAINING:
            raise TimeoutError('application cap reached before another expensive force evaluation')

    save()
    try:
        parent = BASE/'r2_n256_dynamics_profile_v1'
        parent_report = json.loads((parent/'result.json').read_text())
        if parent_report['status'] != 'N256_DYNAMICS_INITIALIZER_NOT_POSTERIOR':
            raise ValueError('matching N256 dynamics settings unavailable')
        for label in ('a', 'b'):
            chain = BASE/f'r2_n256_chain_{label}_v1'
            chain_report = json.loads((chain/'result.json').read_text())
            if (chain_report.get('status') != 'N256_CHAIN_FINISHED_REQUIRES_DIAGNOSTICS'
                    or chain_report.get('completed_proposals') != 48
                    or chain_report.get('source_commit') != '4af620d10bd27019ce23bb6780d6bcd12c11feb1'):
                raise ValueError(f'{label}: expected terminal state from the completed R2 chain')

        _, _, _, _, source, mix, observation, growth = load_inputs()
        linked_radii = linked_point_conditioning_radius(observation)
        conditioning_rule = mix.get('source_conditioning_rule', '')
        ledger_commit = mix.get('association_ledger_source_commit', '')
        if (len(linked_radii) != len(mix.get('PGC', ()))
                or not conditioning_rule.startswith('secure linked 2M++ point redshift')
                or not ledger_commit):
            raise ValueError('exact-force target lacks aligned linked-point radius/ledger provenance')
        report.update(conditional_FP_rows=len(linked_radii),
            source_conditioning_rule=conditioning_rule,
            source_conditioning_radius_range_cMpc_h=[
                float(linked_radii.min()), float(linked_radii.max())],
            association_ledger_source_commit=ledger_commit)
        save()
        source = source_geometry_at_resolution(source, N)
        with np.load(BASE/'r2_sky_closed_split_v6/split.npz', allow_pickle=False) as f:
            keys, counts = map(jnp.asarray, (f['train_keys'], f['train_counts']))
            exposure, _ = build_population_exposure_masks(
                128, f['heldout_flat_voxels'], f['train_window_excluded_keys'],
                f['heldout_window_excluded_keys'])
        obs = ResolutionObservationTarget(N, source, mix, observation, growth,
            keys, counts, jnp.asarray(exposure), force_order=1, fine_order=2)
        settings = parent_report['settings']
        evolve, _, config, _, particle_mass = make_dynamics(settings)
        mass = jnp.full(NIC, particle_mass)

        @jax.jit
        def field(white):
            pos, vel = evolve(white)
            state = particle_grid(pos, vel, mass, config)
            return state['rho'], jnp.moveaxis(state['mean_velocity_km_s'], -1, 0)

        memory_checked = False
        exact_target_components = {}
        def oracle(q, order, gradient):
            nonlocal memory_checked
            check_budget()
            tic = time.monotonic()
            white, tracer, population = map(jnp.asarray,
                (q[:NIC], q[NIC:NIC+9], q[NIC+9:]))
            if gradient:
                (rho, velocity), pullback = jax.vjp(field, white)
            else:
                rho, velocity = field(white)
            packs, info = obs.support(rho, velocity, tracer, order)
            if gradient:
                if order == 2 and not memory_checked:
                    compiled = obs.derivative.lower(rho, velocity, tracer, population,
                        packs, source, observation, order).compile()
                    memory = compiled.memory_analysis()
                    stats = jax.devices()[0].memory_stats() or {}
                    peak = stats.get('bytes_in_use', 0) + memory.temp_size_in_bytes + memory.output_size_in_bytes
                    report['fine_gradient_device_memory'] = dict(
                        estimated_peak_GiB=peak/1024**3,
                        limit_GiB=stats.get('bytes_limit', 0)/1024**3)
                    del compiled
                    if stats.get('bytes_limit') and 1.2*peak > stats['bytes_limit']:
                        raise MemoryError('GL2 force lacks the required 20% device-memory margin')
                    memory_checked = True
                (score, parts), grads = obs.derivative(rho, velocity, tracer,
                    population, packs, source, observation, order)
                ic, = pullback((grads[0], grads[1]))
                derivative = q - np.r_[np.asarray(ic), np.asarray(grads[2]), np.asarray(grads[3])]
            else:
                (score, parts) = obs.value(rho, velocity, tracer, population,
                    packs, source, observation, order)
                derivative = None
            prior_energy = .5*float(q@q)
            score_parts = np.asarray(parts, dtype=np.float64)
            if (score_parts.shape != (2,) or not np.isfinite(score_parts).all()
                    or not np.isclose(float(score), score_parts.sum(), rtol=0., atol=1e-7)):
                raise FloatingPointError('R2 count/raw target components do not sum to the joint score')
            energy = prior_energy - float(score)
            if not np.isfinite(energy) or (derivative is not None and not np.isfinite(derivative).all()):
                raise FloatingPointError('nonfinite R2 target value or gradient')
            row = dict(order=order, gradient=gradient, seconds=time.monotonic()-tic,
                       target_energy=energy, prior_energy=prior_energy,
                       count_log_score=float(score_parts[0]),
                       raw_mark_log_score=float(score_parts[1]), **info)
            report['evaluations'].append(row)
            if order == 2:
                exact_target_components.clear()
                exact_target_components.update(prior_energy=prior_energy,
                    count_log_score=float(score_parts[0]),
                    raw_mark_log_score=float(score_parts[1]))
            save()
            return energy, derivative

        metric = FixedSplitMetric(
            inverse_laplacian_metric_symbol(N, fundamental_mass=fundamental_mass),
            np.eye(24)*1e-5)
        previous_trials = {}
        if reanchor_only or exact_one_step:
            previous = json.loads((BASE/'r2_n256_force_pair_v1/result.json').read_text())
            if previous.get('status') != 'FOUR_MATCHED_FORCE_TRIALS_COMPLETE_NOT_POSTERIOR':
                raise ValueError('completed terminal force-pair reference required')
            previous_trials = {(r['chain'], r['force']): r for r in previous['trials']}

        def run_trial(chain_label, force_label, q, fine_energy, force, force_value,
                      force_gradient, seed, setup_seconds=0., steps=STEPS):
            check_budget()
            rng = np.random.default_rng(seed)
            tic = time.monotonic()
            p = metric.momentum(rng)
            initial_h = fine_energy + metric.kinetic(p)
            proposal, p_end, path_force_value, _ = split_trajectory(
                force, metric, q, p, STEP, steps,
                initial_evaluation=(force_value, force_gradient))
            proposed_fine_energy, _ = oracle(proposal, 2, False)
            delta_h = proposed_fine_energy + metric.kinetic(p_end) - initial_h
            log_acceptance = min(0., -delta_h)
            probability = math.exp(log_acceptance)
            accepted = bool(math.log(rng.uniform()) < log_acceptance)
            displacement = proposal - q
            ic_rms = float(np.sqrt(np.mean(displacement[:NIC]**2)))
            nuisance_rms = float(np.sqrt(np.mean(displacement[NIC:]**2)))
            spectrum = np.fft.fftn(displacement[:NIC].reshape((N,)*3), norm='ortho')
            fundamental = {str(mode): [float(spectrum[mode].real), float(spectrum[mode].imag)]
                for mode in ((1,0,0), (0,1,0), (0,0,1))}
            row = dict(chain=chain_label, force=force_label, seed=seed,
                integration_steps=steps,
                proposed_fine_energy=proposed_fine_energy, path_force_energy=float(path_force_value),
                energy_error=float(delta_h), acceptance_probability=probability,
                common_uniform_decision=accepted, proposal_ic_white_rms=ic_rms,
                proposal_nuisance_white_rms=nuisance_rms,
                acceptance_weighted_ic_rms=probability*ic_rms,
                acceptance_weighted_nuisance_rms=probability*nuisance_rms,
                acceptance_weighted_ic_rms_per_second=probability*ic_rms/max(time.monotonic()-tic, 1e-9),
                anchor_setup_seconds=setup_seconds,
                acceptance_weighted_ic_rms_per_total_setup_second=probability*ic_rms/
                    max(setup_seconds+time.monotonic()-tic, 1e-9),
                fundamental_mode_displacement=fundamental,
                actual_accepted_displacement_rms=float(np.sqrt(np.mean(displacement**2))) if accepted else 0.,
                elapsed_seconds=time.monotonic()-tic)
            report['trials'].append(row)
            save()

        if trajectory_lengths:
            chain_label = os.environ.get('CF4_R2_CHAIN_LABEL', '').lower()
            if chain_label not in ('a', 'b'):
                raise ValueError('CF4_R2_CHAIN_LABEL must be a or b')
            checkpoint_path = Path(os.environ['CF4_R2_INITIAL_CHECKPOINT'])
            probe_seed = int(os.environ.get('CF4_R2_PROBE_SEED', '2026100501'))
            with np.load(checkpoint_path, allow_pickle=False) as f:
                q = f['canonical'].astype(np.float64)
                saved_energy = float(f['fine_energy'])
                local_completed = int(f['completed_transitions']) if 'completed_transitions' in f else None
            if q.shape != (NIC+24,):
                raise ValueError(f'{chain_label}: invalid canonical trajectory checkpoint')

            setup_started = time.monotonic()
            fine_energy, gradient = oracle(q, 2, True)
            if abs(fine_energy-saved_energy) > 1e-7:
                raise AssertionError(f'{chain_label}: checkpoint differs from the exact GL2 target')
            probe_rng = np.random.default_rng(probe_seed)
            p = metric.momentum(probe_rng)
            q_start = q.copy()
            p_start = p.copy()
            initial_h = fine_energy + metric.kinetic(p)
            initial_components = dict(exact_target_components)
            diagnostic = dict(
                chain_label=chain_label.upper(), initial_checkpoint=str(checkpoint_path),
                checkpoint_local_completed_transitions=local_completed,
                checkpoint_global_transition=(int(os.environ['CF4_R2_CHAIN_GLOBAL_TRANSITION'])
                    if os.environ.get('CF4_R2_CHAIN_GLOBAL_TRANSITION') else None),
                checkpoint_energy_abs_error=abs(fine_energy-saved_energy),
                common_momentum_seed=probe_seed,
                starting_target_energy=fine_energy,
                starting_components=initial_components,
                initial_gradient_seconds=time.monotonic()-setup_started,
                trials=[])
            report['trajectory_length_diagnostic'].update(diagnostic)
            save()

            for steps in trajectory_lengths:
                check_budget()
                tic = time.monotonic()
                q_trial, p_trial, path_energy, _ = split_trajectory(
                    lambda x: oracle(x, 2, True), metric, q_start, p_start,
                    STEP, steps, initial_evaluation=(fine_energy, gradient))
                endpoint_energy = float(path_energy)
                endpoint_components = dict(exact_target_components)
                delta_h = endpoint_energy + metric.kinetic(p_trial) - initial_h
                if not math.isfinite(delta_h):
                    raise FloatingPointError('nonfinite exact-GL2 trajectory Hamiltonian error')
                probability = math.exp(min(0., -delta_h))
                displacement = q_trial-q_start
                ic_displacement = displacement[:NIC]
                row = dict(
                    integration_steps=steps, endpoint_target_energy=endpoint_energy,
                    endpoint_components=endpoint_components,
                    energy_error=float(delta_h), acceptance_probability=probability,
                    canonical_jump_rms=float(np.sqrt(np.mean(displacement**2))),
                    ic_white_jump_rms=float(np.sqrt(np.mean(ic_displacement**2))),
                    nuisance_white_jump_rms=float(np.sqrt(np.mean(displacement[NIC:]**2))),
                    elapsed_seconds=time.monotonic()-tic)
                diagnostic['trials'].append(row)
                # These paths are deterministic probes, not MH transitions. Restore
                # the initial decomposition for the next same-state trajectory.
                exact_target_components.clear()
                exact_target_components.update(initial_components)
                save()

            if not np.array_equal(q, q_start) or not np.array_equal(p, p_start):
                raise AssertionError('trajectory diagnostic mutated its shared initial state')
            report['status'] = 'EXACT_GL2_TRAJECTORY_LENGTH_DIAGNOSTIC_COMPLETE_NOT_SAMPLES'
            report['conclusion_scope'] = (
                'three deterministic same-state/common-momentum exact-target paths; '
                'not MH transitions, retained samples, stationarity, or posterior evidence')
            report['same_field_LG_constraint'] = (
                'MW/M31 remain role-ambiguous and M33 unresolved; later observables must '
                'constrain these same roles on the same NEW evolved field; truth IDs are evaluation-only')
            report['source_head'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
            save()
            return

        if exact_chain_steps:
            chain_label = os.environ.get('CF4_R2_CHAIN_LABEL', '').lower()
            if chain_label not in ('a', 'b'):
                raise ValueError('CF4_R2_CHAIN_LABEL must be a or b')

            def json_number(value):
                value = float(value)
                return value if math.isfinite(value) else None

            def checkpoint(path, q, energy, gradient, completed, rng):
                temporary = path.with_name(path.name + '.tmp')
                with temporary.open('wb') as stream:
                    np.savez(stream, canonical=q, fine_energy=energy,
                        fine_gradient=gradient, completed_transitions=completed,
                        rng_state=np.array(json.dumps(rng.bit_generator.state)))
                temporary.replace(path)

            chain_path = BASE/f'r2_n256_chain_{chain_label}_v1'
            initial_checkpoint = Path(os.environ.get('CF4_R2_INITIAL_CHECKPOINT',
                str(chain_path/'accepted_checkpoint.npz')))
            with np.load(initial_checkpoint, allow_pickle=False) as f:
                q = f['canonical'].astype(np.float64)
                saved_energy = float(f['fine_energy'])
                saved_rng_state = (json.loads(str(f['rng_state'].item()))
                    if 'rng_state' in f else None)
            if q.shape != (NIC+24,):
                raise ValueError(f'{chain_label}: invalid canonical chain checkpoint')

            setup_started = time.monotonic()
            fine_energy, gradient = oracle(q, 2, True)
            if abs(fine_energy-saved_energy) > 1e-7:
                raise AssertionError(f'{chain_label}: checkpoint energy differs from the exact GL2 target')
            default_seed = 2026100200 + (1 if chain_label == 'a' else 2)
            explicit_seed = os.environ.get('CF4_R2_CHAIN_SEED')
            rng_seed = int(explicit_seed) if explicit_seed is not None else (
                None if saved_rng_state is not None else default_seed)
            rng = restore_numpy_rng(rng_seed,
                saved_rng_state if explicit_seed is None else None)
            chain_result = dict(label=chain_label.upper(),
                initial_source=str(initial_checkpoint),
                rng_seed=rng_seed,
                rng_state_restored=saved_rng_state is not None and explicit_seed is None,
                endpoint_target_value_source=(
                    'exact GL2 value returned by the endpoint value-and-gradient call; '
                    'no redundant separately compiled value-only reevaluation'),
                initial_exact_gradient_seconds=time.monotonic()-setup_started,
                initial_fine_energy=fine_energy, completed_transitions=0, trace=[])
            report['chain'] = chain_result
            state_path = out/f'chain_{chain_label}_accepted_checkpoint.npz'
            checkpoint(state_path, q, fine_energy, gradient, 0, rng)
            save()

            for index in range(exact_chain_steps):
                check_budget()
                previous_q = q.copy()
                previous_exact_components = dict(exact_target_components)
                tic = time.monotonic()
                q, fine_energy, gradient, info = split_hmc_step(
                    lambda x: oracle(x, 2, True), metric, q, fine_energy, gradient,
                    rng, step=STEP, steps=exact_chain_integrations)
                if not info['accepted']:
                    exact_target_components.clear()
                    exact_target_components.update(previous_exact_components)
                displacement = q-previous_q
                spectrum = np.fft.fftn(q[:NIC].reshape((N,)*3), norm='ortho')
                row = dict(transition=index+1, integration_steps=exact_chain_integrations,
                    step_size=STEP, accepted=bool(info['accepted']),
                    energy_error=json_number(info['energy_error']),
                    exact_target_energy=json_number(fine_energy),
                    **exact_target_components,
                    acceptance_probability=math.exp(info['log_acceptance'])
                        if math.isfinite(info['log_acceptance']) else 0.,
                    force_evaluations=info['force_evaluations'],
                    elapsed_seconds=time.monotonic()-tic,
                    accepted_ic_white_jump_rms=float(np.sqrt(np.mean(displacement[:NIC]**2))),
                    current_ic_white_mean_square=float(np.mean(q[:NIC]**2)),
                    current_nuisance_white=q[NIC:].tolist(),
                    fundamental_modes={str(mode): [float(spectrum[mode].real),
                        float(spectrum[mode].imag)] for mode in
                        ((1,0,0), (0,1,0), (0,0,1))})
                chain_result['trace'].append(row)
                chain_result['completed_transitions'] = index+1
                checkpoint(state_path, q, fine_energy, gradient, index+1, rng)
                save()

            chain_result['checkpoint'] = str(state_path)
            chain_result['requested_transitions'] = exact_chain_steps
            report['status'] = ('ONE_EXACT_GL2_CHAIN_COMPLETE_NOT_POSTERIOR'
                if chain_result['completed_transitions'] == exact_chain_steps
                else 'ONE_EXACT_GL2_CHAIN_BUDGET_STOP_NOT_POSTERIOR')
            report['conclusion_scope'] = ('one short state-independent exact-target chain from one prior accepted state; '
                'not stationarity, posterior uncertainty, or global mixing evidence')
            report['same_field_LG_constraint'] = ('MW/M31 remain ambiguous and M33 unresolved; later component '
                'observables must constrain this same NEW field; truth IDs are evaluation-only')
            report['source_head'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
            save()
            return

        for index, label in enumerate(('a', 'b')):
            chain = BASE/f'r2_n256_chain_{label}_v1'
            with np.load(chain/'accepted_checkpoint.npz', allow_pickle=False) as f:
                q = f['canonical'].astype(np.float64)
                fine_energy = float(f['fine_energy'])
                cached_proxy_energy = float(f['force_energy'])
                cached_proxy_gradient = f['force_gradient'].astype(np.float64)
            if q.shape != (NIC+24,) or (not exact_one_step and cached_proxy_gradient.shape != q.shape):
                raise ValueError(f'{label}: invalid terminal canonical checkpoint')

            seed = 2026093001 + index
            if exact_one_step:
                setup_started = time.monotonic()
                actual_fine_energy, actual_fine_gradient = oracle(q, 2, True)
                setup_seconds = time.monotonic()-setup_started
                if abs(actual_fine_energy-fine_energy) > 1e-7:
                    raise AssertionError(f'{label}: saved state no longer matches the GL2 target')
                run_trial(label.upper(), 'exact_GL2_one_step', q, fine_energy,
                    lambda x: oracle(x, 2, True), actual_fine_energy,
                    actual_fine_gradient, seed, setup_seconds, steps=1)
                report['trials'][-1]['comparison_reference'] = {
                    key: previous_trials[(label.upper(), 'exact_GL2')][key]
                    for key in ('energy_error', 'acceptance_probability', 'elapsed_seconds',
                                'acceptance_weighted_ic_rms_per_second')}
                save()
                continue

            force_setup_started = time.monotonic()
            with np.load(chain/'fixed_force_anchor.npz', allow_pickle=False) as f:
                anchor, correction = f['anchor'].copy(), f['gradient_correction'].copy()
            proxy = AffineCorrectedForce(lambda x: oracle(x, 1, True), anchor, correction)
            proxy_value, proxy_gradient = proxy(q)
            if (abs(proxy_value-cached_proxy_energy) > 1e-7
                    or not np.allclose(proxy_gradient, cached_proxy_gradient, rtol=0., atol=1e-7)):
                raise AssertionError(f'{label}: saved proxy cache does not match its fixed force')
            actual_fine_energy, actual_fine_gradient = oracle(q, 2, True)
            force_setup_seconds = time.monotonic()-force_setup_started
            if abs(actual_fine_energy-fine_energy) > 1e-7:
                raise AssertionError(f'{label}: saved state no longer matches the GL2 fine target')

            if reanchor_only:
                raw_gl1_gradient = proxy_gradient - correction
                raw_gl1_energy = proxy_value - float(correction@(q-anchor))
                local_correction = actual_fine_gradient - raw_gl1_gradient
                local_base = AffineCorrectedForce(
                    lambda x: oracle(x, 1, True), q, local_correction)
                energy_shift = actual_fine_energy - raw_gl1_energy
                def local_proxy(x):
                    value, gradient = local_base(x)
                    return value+energy_shift, gradient
                local_gradient = raw_gl1_gradient + local_correction
                tangent_gradient_max_error = float(np.max(np.abs(local_gradient-actual_fine_gradient)))
                tangent_value_error = abs(raw_gl1_energy+energy_shift-actual_fine_energy)
                if tangent_gradient_max_error > 1e-7 or tangent_value_error > 1e-7:
                    raise AssertionError(f'{label}: re-anchored affine force fails its tangent check')
                report.setdefault('anchor_setups', []).append(dict(chain=label.upper(),
                    exact_gradient_seconds=report['evaluations'][-1]['seconds'],
                    anchor_state='predetermined terminal accepted checkpoint',
                    correction_norm=float(np.linalg.norm(local_correction)),
                    energy_shift_to_fine_target=energy_shift,
                    tangent_value_abs_error=tangent_value_error,
                    tangent_gradient_max_abs_error=tangent_gradient_max_error,
                    frozen_for_one_trajectory=True))
                run_trial(label.upper(), 'terminal_reanchored_GL1_affine', q, fine_energy,
                    local_proxy, raw_gl1_energy, local_gradient, seed, force_setup_seconds)
                for trial in report['trials'][-1:]:
                    trial['comparison_references'] = {
                        force: {key: previous_trials[(label.upper(), force)][key] for key in
                            ('energy_error', 'acceptance_probability', 'elapsed_seconds',
                             'acceptance_weighted_ic_rms_per_second')}
                        for force in ('frozen_GL1_affine', 'exact_GL2')}
                save()
            else:
                run_trial(label.upper(), 'frozen_GL1_affine', q, fine_energy,
                    proxy, proxy_value, proxy_gradient, seed)
                run_trial(label.upper(), 'exact_GL2', q, fine_energy,
                    lambda x: oracle(x, 2, True), actual_fine_energy, actual_fine_gradient, seed)

        if reanchor_only:
            report['status'] = 'TWO_REANCHORED_AFFINE_TRIALS_COMPLETE_NOT_POSTERIOR'
        elif exact_one_step:
            report['status'] = 'TWO_EXACT_GL2_ONE_STEP_TRIALS_COMPLETE_NOT_POSTERIOR'
        else:
            report['status'] = 'FOUR_MATCHED_FORCE_TRIALS_COMPLETE_NOT_POSTERIOR'
        report['conclusion_scope'] = 'local proposal efficiency at two predetermined terminal states only; no stationarity or global mixing inference'
        report['same_field_LG_constraint'] = 'later MW/M31/M33 role assignment and observables must constrain this same NEW field; M33 remains unresolved; truth IDs are evaluation-only'
        report['source_head'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        save()
    except TimeoutError as error:
        status = ('ONE_EXACT_GL2_CHAIN_BUDGET_STOP_NOT_POSTERIOR'
            if exact_chain_steps else 'FORCE_PAIR_BUDGET_STOP_INCOMPLETE')
        report.update(status=status, stop_reason=str(error))
        save()
    except Exception as error:
        report.update(status='FORCE_PAIR_DIAGNOSTIC_FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
