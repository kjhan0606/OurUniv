"""One bounded R1 PM/aperture/sampling entry; never actual LG reconstruction."""
import json
import os
from pathlib import Path
import resource
import time
import unittest

import blackjax
import jax
import jax.numpy as jnp
import numpy as np

from cf4_chunked_hmc import make_chunks, checked_record
from cf4_r1_particle_forward import make_dynamics, aperture_moments, particle_grid, periodic_delta

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def ready(tree):
    return jax.tree_util.tree_map(lambda x: x.block_until_ready(), tree)


class BudgetReached(Exception):
    pass


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required; never run on Syntax login')
    cfg = json.loads((ROOT/'config/cf4_r1_particle_entry_v1.json').read_text())
    out = Path(cfg['output_root']) / ('job_' + os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(status='RUNNING', observed_posterior=False, resolved_LG=False,
                  nested_gravity_verified=False, independent_solver_calibrated=False,
                  R1_complete=False, source_commit=os.environ['EXPECTED_COMMIT'], config=cfg)
    write(out/'result.json', report)

    def progress(stage, **kwargs):
        row = dict(stage=stage, elapsed_seconds=time.monotonic()-start, **kwargs)
        write(out/'progress.json', row)
        print(json.dumps(row, allow_nan=False), flush=True)

    def budget():
        if time.monotonic()-start >= cfg['application_seconds_cap']:
            raise BudgetReached()

    try:
        tests = unittest.defaultTestLoader.loadTestsFromNames([
            'test_cf4_r1_particle_forward', 'test_cf4_chunked_hmc', 'test_cf4_bundle_b'])
        result = unittest.TextTestRunner(verbosity=2).run(tests)
        report['tests'] = result.testsRun
        if not result.wasSuccessful():
            raise RuntimeError('focused regressions failed')
        progress('tests_pass', tests=result.testsRun)
        budget()
        n, box = cfg['n'], cfg['box_cMpc_h']
        evolve, initial, conf, cosmo, mp = make_dynamics(cfg)
        mass = jnp.full(n**3, mp)
        centers = jnp.asarray(cfg['aperture_centers_cMpc_h'])
        radius = cfg['aperture_radius_cMpc_h']
        truth = jnp.asarray(np.random.default_rng(cfg['truth_seed']).normal(size=n**3))
        truth_initial = ready(initial(truth))
        states = {}
        reference_position = None
        timing = {}
        for label, mr, tf in [('base', 1, 1), ('force2', 2, 1), ('force2_time2', 2, 2)]:
            budget()
            tick = time.monotonic()
            ef, initf, _, _, other_mass = ((evolve, initial, conf, cosmo, mp) if label == 'base'
                                          else make_dynamics(cfg, mesh_ratio=mr, time_factor=tf))
            ic = ready(initf(truth))
            for actual, expected in zip(ic, truth_initial):
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(other_mass, mp, rtol=1e-14)
            position, velocity = ready(ef(truth))
            if not np.isfinite(np.asarray(position)).all() or not np.isfinite(np.asarray(velocity)).all():
                raise FloatingPointError(f'{label}: nonfinite evolution')
            moments = ready(aperture_moments(position, velocity, mass, centers, radius, box))
            field = ready(particle_grid(position, velocity, mass, conf))  # COMMON readout mesh
            np.testing.assert_allclose(field['rho'].mean(), 1., atol=1e-12)
            np.testing.assert_allclose(field['momentum'].sum(axis=(0, 1, 2)),
                                       (mass[:, None]*velocity).sum(axis=0), rtol=1e-9, atol=10.)
            states[label] = {key: np.asarray(value).tolist() for key, value in moments.items()}
            timing[label] = time.monotonic()-tick
            if label == 'base':
                reference_position = position
                base_velocity = velocity
            else:
                states[label]['particle_displacement_RMS_vs_base_cMpc_h'] = float(jnp.sqrt(jnp.mean(
                    periodic_delta(position, reference_position, box)**2)))
                states[label]['particle_velocity_RMS_vs_base_km_s'] = float(jnp.sqrt(jnp.mean((velocity-base_velocity)**2)))
            np.savez_compressed(out/(label+'.npz'), white=np.asarray(truth),
                position_cMpc_h=np.asarray(position), velocity_km_s=np.asarray(velocity),
                **{key: np.asarray(field[key]) for key in ('rho', 'mean_velocity_km_s', 'variance_km2_s2', 'valid')})
            progress('fidelity_control', arm=label, seconds=timing[label])
        report['fidelity'] = states
        report['fidelity_compile_and_forward_seconds'] = timing
        report['particle_mass_Msun_h'] = mp
        report['readout_dx_cMpc_h'] = box/n
        write(out/'result.json', report)

        mean_mass = (n**3*mp/box**3) * (2*np.pi)**1.5 * radius**3

        @jax.jit
        def observables(white):
            x, v = evolve(white)
            m = aperture_moments(x, v, mass, centers, radius, box)
            return jnp.concatenate((jnp.log(m['mass_Msun_h']/mean_mass)[:, None],
                                    m['offset_cMpc_h'], m['mean_velocity_km_s']), axis=1).ravel()

        sigma = cfg['mock_sigma']
        noise_scale = jnp.tile(jnp.asarray([sigma['log_mass']] + [sigma['offset_cMpc_h']]*3
                                          + [sigma['mean_velocity_km_s']]*3), 3)
        exact_obs = ready(observables(truth))
        mock_data = exact_obs + noise_scale * jnp.asarray(np.random.default_rng(cfg['noise_seed']).normal(size=21))
        write(out/'mock_data.json', dict(values=np.asarray(mock_data).tolist(),
              sigma=np.asarray(noise_scale).tolist(), likelihood='independent Gaussian synthetic aperture observations',
              physical_velocity_dispersion='Predicted from particles, not an observation or covariance fudge here',
              actual_LG_contract='config/cf4_lg_observation_contract_v1.json remains unchanged and NOT used as data'))

        @jax.jit
        def loglike(white):
            return -.5*jnp.sum(((observables(white)-mock_data)/noise_scale)**2)

        @jax.jit
        def logdensity(white):
            return loglike(white) - .5*jnp.vdot(white, white)

        vg = jax.jit(jax.value_and_grad(loglike))
        x0 = jnp.asarray(np.random.default_rng(cfg['initialization_seeds'][0]).normal(size=n**3))
        tick = time.monotonic()
        value, grad = ready(vg(x0))
        report['likelihood_value_gradient_compile_seconds'] = time.monotonic()-tick
        if not np.isfinite(float(value)) or not np.isfinite(np.asarray(grad)).all():
            raise FloatingPointError('nonfinite likelihood derivative')
        # Check LIKELIHOOD gradient alone so the analytic prior cannot hide an adjoint bug.
        rng = np.random.default_rng(2026091331)
        directions = [jnp.asarray(rng.normal(size=n**3)), grad]
        checks = []
        for direction in directions:
            norm = float(jnp.linalg.norm(direction))
            if norm == 0:
                raise RuntimeError('zero direction/likelihood gradient')
            direction = direction/norm
            automatic = float(jnp.vdot(grad, direction))
            rows = []
            for eps in cfg['directional_epsilons']:
                budget()
                finite = float((loglike(x0+eps*direction)-loglike(x0-eps*direction))/(2*eps))
                error = abs(automatic-finite)/max(abs(automatic), abs(finite), 1e-8)
                rows.append(dict(epsilon=eps, finite_difference=finite, relative_error=error))
            checks.append(dict(adjoint=automatic, probes=rows))
        report['likelihood_gradient_checks'] = checks
        write(out/'result.json', report)
        if any(row['relative_error'] > cfg['gradient_relative_tolerance'] for check in checks for row in check['probes']):
            report['status'] = 'STOP_LIKELIHOOD_ADJOINT_CHECK_FAILED'
            return
        repeat_times = []
        for _ in range(3):
            tick = time.monotonic()
            ready(vg(x0))
            repeat_times.append(time.monotonic()-tick)
        report['likelihood_value_gradient_seconds'] = repeat_times
        progress('adjoint_pass', gradient_seconds=repeat_times)

        initialize, warm, sample, final = make_chunks(logdensity, n**3, cfg['sampler'])
        chain_reports, diagnostic_arrays = [], []
        for chain, seed in enumerate(cfg['initialization_seeds']):
            budget()
            chain_dir = out/f'chain_{chain}'
            chain_dir.mkdir()
            init = jnp.asarray(np.random.default_rng(seed).normal(size=n**3))
            state, adaptation = ready(initialize(init))
            initial_ll = float(loglike(init))
            tick = time.monotonic()
            key = jax.random.PRNGKey(cfg['sampler_seed']+chain)
            warm_accept, warm_divergences = [], 0
            for begin in range(0, cfg['warmup_steps'], cfg['chunk_size']):
                budget()
                key, sub = jax.random.split(key)
                (state, adaptation), record = ready(warm(state, adaptation, jax.random.split(sub, cfg['chunk_size'])))
                arrays = checked_record(record, state)
                warm_accept.extend(arrays[2].tolist())
                warm_divergences += int(arrays[3].sum())
                progress('warmup', chain=chain, steps=begin+cfg['chunk_size'],
                         acceptance=float(np.mean(arrays[2])))
            step = final(adaptation)
            warm_seconds = time.monotonic()-tick
            records = []
            tick = time.monotonic()
            try:
                for begin in range(0, cfg['sampling_steps'], cfg['chunk_size']):
                    budget()
                    key, sub = jax.random.split(key)
                    state, record = ready(sample(state, step, jax.random.split(sub, cfg['chunk_size'])))
                    arrays = checked_record(record, state)
                    records.append(arrays)
                    progress('sampling', chain=chain, steps=begin+cfg['chunk_size'],
                             acceptance=float(np.mean(arrays[2])))
            finally:
                # Save even a budget-interrupted sampling segment without calling it complete.
                np.savez_compressed(chain_dir/'last_state.npz', white=np.asarray(state.position),
                                    step_size=np.asarray(step), random_key=np.asarray(key))
                if records:
                    all_records = tuple(np.concatenate([r[i] for r in records]) for i in range(7))
                    white, logp, accept, divergent, energy, raw, used = all_records
                    np.savez_compressed(chain_dir/'samples.npz', white=white, logdensity=logp,
                                        acceptance=accept, divergent=divergent, energy=energy, step_size=used)
            sample_seconds = time.monotonic()-tick
            norm2 = np.sum(white**2, axis=1)
            diagnostics = np.column_stack((logp+.5*norm2, norm2, white[:, :3]))
            diagnostic_arrays.append(diagnostics)
            row = dict(chain=chain, initialization_seed=seed, initialized_from_truth=False,
                       warmup_seconds=warm_seconds, sampling_seconds=sample_seconds,
                       warmup_divergences=warm_divergences, warmup_mean_acceptance=float(np.mean(warm_accept)),
                       sample_divergences=int(divergent.sum()), sample_mean_acceptance=float(accept.mean()),
                       sampling_step_size=float(step), initial_log_likelihood=initial_ll,
                       final_log_likelihood=float(diagnostics[-1, 0]), draws=len(white))
            chain_reports.append(row)
            report['chains'] = chain_reports
            write(out/'result.json', report)
            position, velocity = ready(evolve(state.position))
            field = ready(particle_grid(position, velocity, mass, conf))
            np.savez_compressed(chain_dir/'final_state.npz', position_cMpc_h=np.asarray(position),
                               velocity_km_s=np.asarray(velocity),
                               **{k: np.asarray(field[k]) for k in ('rho', 'mean_velocity_km_s', 'variance_km2_s2', 'valid')})
        diagnostic_arrays = jnp.asarray(np.stack(diagnostic_arrays))
        ess = np.asarray(blackjax.diagnostics.effective_sample_size(diagnostic_arrays))
        rhat = np.asarray(blackjax.diagnostics.potential_scale_reduction(diagnostic_arrays))
        safe = lambda a: [float(x) if np.isfinite(x) else None for x in np.ravel(a)]
        report['short_chain_diagnostics'] = dict(names=['log_likelihood', 'white_norm2', 'white0', 'white1', 'white2'],
            ESS=safe(ess), unsplit_Rhat=safe(rhat),
            limitation='Short engineering chains; not rank-normalized, not coverage, mixing certification or production cost forecast.')
        report['status'] = 'COMPLETE_R1_ENTRY_DRIVER_JUDGMENT_REQUIRED'
    except BudgetReached:
        report['status'] = 'INCOMPLETE_APPLICATION_BUDGET'
    except Exception as exc:
        report['status'] = 'FAILED'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-start
        report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        stats = jax.devices()[0].memory_stats() or {}
        report['device_peak_bytes'] = stats.get('peak_bytes_in_use')
        write(out/'result.json', report)
        progress('finish', status=report['status'])


if __name__ == '__main__':
    main()
