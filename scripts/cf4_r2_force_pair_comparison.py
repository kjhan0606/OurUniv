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
from cf4_r2_prior_split_hmc import FixedSplitMetric, inverse_laplacian_metric_symbol, split_trajectory
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_resolution_target import source_geometry_at_resolution, ResolutionObservationTarget


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
    application_budget = 4*3600 if reanchor_only else APPLICATION_BUDGET
    report = dict(
        status='STARTED', job_id=os.environ['SLURM_JOB_ID'], N=N, box_cMpc_h=BOX,
        dx_cMpc_h=BOX/N, force_comparison='GL2 exact force vs frozen GL1 affine force',
        fine_target='unchanged N256 GL2 target; same Gaussian prior and exact fine Hamiltonian',
        matched_settings=dict(step_size=STEP, integration_steps=STEPS,
                             inverse_laplacian_metric_fundamental_mass=6000,
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
            energy = .5*float(q@q) - float(score)
            if not np.isfinite(energy) or (derivative is not None and not np.isfinite(derivative).all()):
                raise FloatingPointError('nonfinite R2 target value or gradient')
            row = dict(order=order, gradient=gradient, seconds=time.monotonic()-tic,
                       target_energy=energy, **info)
            report['evaluations'].append(row)
            save()
            return energy, derivative

        metric = FixedSplitMetric(inverse_laplacian_metric_symbol(N), np.eye(24)*1e-5)
        previous_trials = {}
        if reanchor_only:
            previous = json.loads((BASE/'r2_n256_force_pair_v1/result.json').read_text())
            if previous.get('status') != 'FOUR_MATCHED_FORCE_TRIALS_COMPLETE_NOT_POSTERIOR':
                raise ValueError('completed terminal force-pair reference required')
            previous_trials = {(r['chain'], r['force']): r for r in previous['trials']}

        def run_trial(chain_label, force_label, q, fine_energy, force, force_value,
                      force_gradient, seed, setup_seconds=0.):
            check_budget()
            rng = np.random.default_rng(seed)
            tic = time.monotonic()
            p = metric.momentum(rng)
            initial_h = fine_energy + metric.kinetic(p)
            proposal, p_end, path_force_value, _ = split_trajectory(
                force, metric, q, p, STEP, STEPS,
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

        for index, label in enumerate(('a', 'b')):
            chain = BASE/f'r2_n256_chain_{label}_v1'
            with np.load(chain/'accepted_checkpoint.npz', allow_pickle=False) as f:
                q = f['canonical'].astype(np.float64)
                fine_energy = float(f['fine_energy'])
                cached_proxy_energy = float(f['force_energy'])
                cached_proxy_gradient = f['force_gradient'].astype(np.float64)
            with np.load(chain/'fixed_force_anchor.npz', allow_pickle=False) as f:
                anchor, correction = f['anchor'].copy(), f['gradient_correction'].copy()
            if q.shape != (NIC+24,) or cached_proxy_gradient.shape != q.shape:
                raise ValueError(f'{label}: invalid terminal canonical checkpoint')

            force_setup_started = time.monotonic()
            proxy = AffineCorrectedForce(lambda x: oracle(x, 1, True), anchor, correction)
            proxy_value, proxy_gradient = proxy(q)
            if (abs(proxy_value-cached_proxy_energy) > 1e-7
                    or not np.allclose(proxy_gradient, cached_proxy_gradient, rtol=0., atol=1e-7)):
                raise AssertionError(f'{label}: saved proxy cache does not match its fixed force')
            actual_fine_energy, actual_fine_gradient = oracle(q, 2, True)
            force_setup_seconds = time.monotonic()-force_setup_started
            if abs(actual_fine_energy-fine_energy) > 1e-7:
                raise AssertionError(f'{label}: saved state no longer matches the GL2 fine target')

            seed = 2026093001 + index
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

        report['status'] = ('TWO_REANCHORED_AFFINE_TRIALS_COMPLETE_NOT_POSTERIOR' if reanchor_only
                            else 'FOUR_MATCHED_FORCE_TRIALS_COMPLETE_NOT_POSTERIOR')
        report['conclusion_scope'] = 'local proposal efficiency at two predetermined terminal states only; no stationarity or global mixing inference'
        report['same_field_LG_constraint'] = 'later MW/M31/M33 role assignment and observables must constrain this same NEW field; M33 remains unresolved; truth IDs are evaluation-only'
        report['source_head'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
        save()
    except TimeoutError as error:
        report.update(status='FORCE_PAIR_BUDGET_STOP_INCOMPLETE', stop_reason=str(error))
        save()
    except Exception as error:
        report.update(status='FORCE_PAIR_DIAGNOSTIC_FAILED', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
