#!/usr/bin/env python3
"""Split the active N256 R2 score gradient at two saved accepted states."""
import json
import os
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_count_exposure import build_population_exposure_masks
from cf4_r2_raw_field_profile import load_inputs
from cf4_r2_resolution_target import source_geometry_at_resolution, ResolutionObservationTarget


BASE = Path('/gpfs/kjhan/CF4/z0_density')
N = 256
BOX = 384.0
NIC = N**3
NNUISANCE = 24
MODES = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
COMPONENTS = ((0, 'count_2mpp'), (1, 'raw_cf4_fp_marks'))
APPLICATION_BUDGET = 2*3600 + 20*60
MIN_CALL_REMAINING = 20*60


def complex_pair(value):
    return [float(value.real), float(value.imag)]


def mode_summary(field, q_modes):
    spectrum = np.fft.fftn(field[:NIC].reshape((N,)*3), norm='ortho')
    output = []
    for mode, qk in zip(MODES, q_modes):
        score_k = spectrum[mode]
        nll_k = -score_k
        radial = 2*float(np.real(np.conj(qk)*nll_k))
        output.append(dict(mode=list(mode), score_gradient=complex_pair(score_k),
            negative_log_likelihood_gradient=complex_pair(nll_k),
            radial_nloglikelihood_derivative=radial,
            phase_nloglikelihood_derivative=2*float(np.imag(np.conj(qk)*nll_k))))
    return output


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('run as a Slurm GPU job')
    output = Path(os.environ['CF4_R2_OUT_DIR'])
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='STARTED', job_id=os.environ['SLURM_JOB_ID'],
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        classification='SAVED_N256_R2_COUNT_VS_RAW_FP_GRADIENT_ATTRIBUTION_NOT_POSTERIOR',
        grid=N, box_cMpc_h=BOX, cell_cMpc_h=BOX/N, modes=[list(m) for m in MODES],
        components={'count_2mpp': 'frozen graph-closed training voxel counts',
                    'raw_cf4_fp_marks': 'selected raw FP-distance-mark subset only; no TF/SNIa/SBF terms'},
        heldout_scored=False, chain_transitions=0, PMWD_same_state_replays=0,
        Q_GOAL='attribute the partial R2 target force on anomalous fundamental IC-white modes before any model change or new chain',
        Q_LEAN='two already saved accepted states; one deterministic replay per state and two component adjoints; no new simulation or heldout score',
        MW_M31='role-ambiguous; no candidate selected by truth identity',
        M33='unresolved; eventual observables must constrain the same NEW field',
        checkpoints={}, application_budget_seconds=APPLICATION_BUDGET)

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        temporary = output/'result.json.tmp'
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(output/'result.json')

    def check_call_budget():
        if APPLICATION_BUDGET-(time.monotonic()-started) < MIN_CALL_REMAINING:
            raise TimeoutError('less than20 minutes remain for another component adjoint')

    save()
    try:
        parent_dir = BASE/'r2_n256_dynamics_profile_v1'
        parent_report = json.loads((parent_dir/'result.json').read_text())
        if parent_report.get('status') != 'N256_DYNAMICS_INITIALIZER_NOT_POSTERIOR':
            raise ValueError('expected frozen N256 dynamics configuration is unavailable')
        settings = parent_report['settings']

        _, _, _, _, source, mix, observation, geometry = load_inputs()
        source = source_geometry_at_resolution(source, N)
        with np.load(BASE/'r2_sky_closed_split_v6/split.npz', allow_pickle=False) as split:
            keys, counts = map(jnp.asarray, (split['train_keys'], split['train_counts']))
            exposure, _ = build_population_exposure_masks(128, split['heldout_flat_voxels'],
                split['train_window_excluded_keys'], split['heldout_window_excluded_keys'])
        target = ResolutionObservationTarget(N, source, mix, observation, geometry,
            keys, counts, jnp.asarray(exposure), force_order=1, fine_order=2)

        evolve, _, config, _, particle_mass = make_dynamics(settings)
        mass = jnp.full(NIC, particle_mass)

        @jax.jit
        def field(white):
            positions, velocities = evolve(white)
            state = particle_grid(positions, velocities, mass, config)
            return state['rho'], jnp.moveaxis(state['mean_velocity_km_s'], -1, 0)

        for label in ('a', 'b'):
            check_call_budget()
            chain_dir = BASE/f'r2_n256_gl2_metric6000_control_{label}_v1'
            chain_report = json.loads((chain_dir/'result.json').read_text())
            chain = chain_report.get('chain', {})
            if (chain_report.get('status') != 'ONE_EXACT_GL2_CHAIN_COMPLETE_NOT_POSTERIOR'
                    or chain.get('completed_transitions') != 1
                    or chain.get('trace', [{}])[0].get('accepted') is not True):
                raise ValueError(f'{label}: expected accepted exact-GL2 endpoint missing')
            checkpoint_path = chain_dir/f'chain_{label}_accepted_checkpoint.npz'
            with np.load(checkpoint_path, allow_pickle=False) as archive:
                q = np.asarray(archive['canonical'], dtype=np.float64)
                saved_energy = float(archive['fine_energy'])
                saved_gradient = np.asarray(archive['fine_gradient'], dtype=np.float64)
                completed = int(archive['completed_transitions'])
            if (q.shape != (NIC+NNUISANCE,) or saved_gradient.shape != q.shape
                    or completed != 1 or not np.isfinite(q).all()
                    or not np.isfinite(saved_gradient).all()):
                raise ValueError(f'{label}: checkpoint shape/state is invalid')

            qj = jnp.asarray(q)
            tracer, population = qj[NIC:NIC+9], qj[NIC+9:]
            tic = time.monotonic()
            (rho, velocity), field_pullback = jax.vjp(field, qj[:NIC])
            jax.block_until_ready((rho, velocity))
            packs, support_info = target.support(rho, velocity, tracer, 2)
            score, parts = target.value(rho, velocity, tracer, population,
                packs, source, observation, 2)
            score, parts = float(score), np.asarray(jax.device_get(parts), dtype=np.float64)
            energy = .5*float(q@q)-score
            energy_error = energy-saved_energy
            if abs(energy_error) > 1e-7 or not np.allclose(parts.sum(), score, rtol=0., atol=1e-9):
                raise AssertionError(f'{label}: component scores do not reproduce saved exact energy')
            q_modes = np.fft.fftn(q[:NIC].reshape((N,)*3), norm='ortho')
            q_modes = np.asarray([q_modes[m] for m in MODES])
            result = dict(checkpoint=str(checkpoint_path), completed_transitions=completed,
                accepted_endpoint=True, saved_energy=saved_energy, replayed_energy=energy,
                energy_abs_error=abs(energy_error), component_scores={
                    'count_2mpp': float(parts[0]), 'raw_cf4_fp_marks': float(parts[1])},
                support=support_info, state_replay_seconds=time.monotonic()-tic,
                phase_offset_from_observer_degrees=[float(np.degrees(abs(np.angle(z/(-abs(z))))))
                    for z in q_modes], component_gradients={})
            report['PMWD_same_state_replays'] += 1
            report['checkpoints'][label] = result
            save()
            component_score_gradients = {}

            for component_index, name in COMPONENTS:
                check_call_budget()
                tic = time.monotonic()
                compiled = target.component_derivative.lower(rho, velocity, tracer,
                    population, packs, source, observation, 2, component_index).compile()
                analysis = compiled.memory_analysis()
                stats = jax.devices()[0].memory_stats() or {}
                peak = (stats.get('bytes_in_use', 0)+analysis.temp_size_in_bytes+
                        analysis.output_size_in_bytes)
                limit = stats.get('bytes_limit', 0)
                memory = dict(estimated_peak_GiB=peak/1024**3,
                    limit_GiB=limit/1024**3 if limit else None)
                result.setdefault('component_memory', {})[name] = memory
                save()
                if limit and 1.2*peak > limit:
                    raise MemoryError(f'{name}: compiled component adjoint lacks20% device margin')
                component_score, gradients = compiled(rho, velocity, tracer,
                    population, packs, source, observation)
                jax.block_until_ready((component_score, gradients))
                component_score = float(component_score)
                component_score_error = abs(component_score-float(parts[component_index]))
                field_gradient, tracer_gradient, population_gradient = gradients
                ic_gradient, = field_pullback(field_gradient)
                score_gradient = np.concatenate((np.asarray(jax.device_get(ic_gradient)).ravel(),
                    np.asarray(jax.device_get(tracer_gradient)).ravel(),
                    np.asarray(jax.device_get(population_gradient)).ravel()))
                if score_gradient.shape != q.shape or not np.isfinite(score_gradient).all():
                    raise FloatingPointError(f'{label}/{name}: invalid component gradient')
                modes = mode_summary(score_gradient, q_modes)
                negative_log_gradient = -score_gradient
                component_result = dict(score=float(parts[component_index]),
                    independently_returned_score=component_score,
                    score_abs_error=component_score_error,
                    score_gradient_allmode_ic_rms=float(np.sqrt(np.mean(score_gradient[:NIC]**2))),
                    negative_log_gradient_allmode_ic_rms=float(np.sqrt(np.mean(negative_log_gradient[:NIC]**2))),
                    score_gradient_nuisance_l2=float(np.linalg.norm(score_gradient[NIC:])),
                    score_gradient_tracer9=score_gradient[NIC:NIC+9].tolist(),
                    score_gradient_population15=score_gradient[NIC+9:].tolist(),
                    negative_log_gradient_tracer9=(-score_gradient[NIC:NIC+9]).tolist(),
                    negative_log_gradient_population15=(-score_gradient[NIC+9:]).tolist(),
                    modes=modes, adjoint_seconds=time.monotonic()-tic)
                result['component_gradients'][name] = component_result
                component_score_gradients[name] = score_gradient.copy()
                save()
                if component_score_error > 1e-7:
                    raise AssertionError(f'{label}/{name}: component score differs from exact sum output')
                del compiled, gradients, score_gradient, negative_log_gradient

            total_score_gradient = (component_score_gradients['count_2mpp']+
                                    component_score_gradients['raw_cf4_fp_marks'])
            reconstructed_gradient = q-total_score_gradient
            difference = reconstructed_gradient-saved_gradient
            gradient_abs_max = float(np.max(np.abs(difference)))
            gradient_relative_l2 = float(np.linalg.norm(difference)/
                max(float(np.linalg.norm(saved_gradient)), 1.))
            gradient_relative_max = gradient_abs_max/max(float(np.max(np.abs(saved_gradient))), 1.)
            result['gradient_reproduction'] = dict(
                relative_l2=gradient_relative_l2, relative_max_abs=gradient_relative_max,
                absolute_max=gradient_abs_max,
                reconstructed_total_ic_rms=float(np.sqrt(np.mean(reconstructed_gradient[:NIC]**2))),
                saved_total_ic_rms=float(np.sqrt(np.mean(saved_gradient[:NIC]**2))),
                total_score_gradient_ic_rms=float(np.sqrt(np.mean(total_score_gradient[:NIC]**2))),
                component_sum_matches_saved=gradient_relative_l2 <= 1e-6 and gradient_relative_max <= 1e-6)
            result['total_likelihood_gradient_modes'] = mode_summary(total_score_gradient, q_modes)
            total_target_spectrum = np.fft.fftn(reconstructed_gradient[:NIC].reshape((N,)*3), norm='ortho')
            result['total_target_gradient_modes'] = [
                dict(mode=list(mode), coefficient=complex_pair(total_target_spectrum[mode]))
                for mode in MODES]
            result['nuisance_sharing'] = dict(
                count_tracer9_l2=float(np.linalg.norm(component_score_gradients['count_2mpp'][NIC:NIC+9])),
                raw_tracer9_l2=float(np.linalg.norm(component_score_gradients['raw_cf4_fp_marks'][NIC:NIC+9])),
                count_population15_l2=float(np.linalg.norm(component_score_gradients['count_2mpp'][NIC+9:])),
                raw_population15_l2=float(np.linalg.norm(component_score_gradients['raw_cf4_fp_marks'][NIC+9:])),
                interpretation='component gradients conditional on the saved IC/nuisance state')
            save()
            if not result['gradient_reproduction']['component_sum_matches_saved']:
                raise AssertionError(f'{label}: count+raw gradient does not reproduce saved exact gradient')
            del component_score_gradients, total_score_gradient, reconstructed_gradient, difference
            del field_pullback, rho, velocity, packs

        report['status'] = 'COMPONENT_ATTRIBUTION_COMPLETE_NOT_POSTERIOR'
        report['outcome_mapping'] = 'interpret only same-state complex gradients, radial/transverse pieces, nuisance gradients and state dependence; no threshold or automatic model change'
        save()
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    except Exception as error:
        report['status'] = 'COMPONENT_ATTRIBUTION_FAILED_OR_INCOMPLETE_NOT_POSTERIOR'
        report['error'] = f'{type(error).__name__}: {error}'
        save()
        raise


if __name__ == '__main__':
    main()
