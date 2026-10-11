"""Bounded R1 force/time and equal-expected-work HMC comparison; no LG claim."""
import gc
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
from cf4_datum_bearing_z0_phasec_pilot import (
    _rank_normalize, _split_chains, rank_normalized_rhat)
from cf4_r1_particle_forward import make_dynamics, aperture_moments
from cf4_r1_particle_entry import ROOT, write, ready, BudgetReached


def observation_vector(moments, mean_mass):
    return jnp.concatenate((jnp.log(moments['mass_Msun_h']/mean_mass)[:, None],
                            moments['offset_cMpc_h'], moments['mean_velocity_km_s']), axis=1).ravel()


def diagnostics(values, names):
    """Reuse rank/folded Rhat; library split-chain ESS, not legacy local ESS."""
    ranked, lower, upper, rhats = [], [], [], []
    for index in range(len(names)):
        x = values[:, :, index]
        ranked.append(_split_chains(_rank_normalize(x)))
        q05, q95 = np.quantile(x, [.05, .95])
        lower.append(_split_chains((x <= q05).astype(float)))
        upper.append(_split_chains((x >= q95).astype(float)))
        rhats.append(rank_normalized_rhat(x))
    ess = lambda rows: np.asarray(blackjax.diagnostics.effective_sample_size(
        jnp.asarray(np.stack(rows, axis=-1))))
    bulk, tail = ess(ranked), np.minimum(ess(lower), ess(upper))
    if not np.isfinite(np.r_[rhats, bulk, tail]).all():
        raise FloatingPointError('nonfinite projected chain diagnostics')
    rows = [dict(name=name, rank_normalized_split_Rhat=float(rhats[i]),
                 bulk_ESS=float(bulk[i]), tail_ESS=float(tail[i])) for i, name in enumerate(names)]
    return dict(parameters=rows, max_Rhat=max(rhats), min_bulk_ESS=float(bulk.min()),
                min_tail_ESS=float(tail.min()),
                limitation='Finite predeclared projections, not all-mode convergence or mock coverage.')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    cfg = json.loads((ROOT/'config/cf4_r1_resolution_mixing_v2.json').read_text())
    entry = Path(cfg['entry_directory'])
    old = json.loads((entry/'result.json').read_text())
    model = old['config']
    data = json.loads((entry/'mock_data.json').read_text())
    out = Path(cfg['output_root'])/('job_'+os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(status='RUNNING', config=cfg, model=model,
                  source_commit=os.environ['EXPECTED_COMMIT'], observed_posterior=False,
                  resolved_LG=False, independent_solver_calibrated=False,
                  nested_gravity_verified=False, R1_complete=False)

    def progress(stage, **extra):
        row = dict(stage=stage, elapsed_seconds=time.monotonic()-start, **extra)
        write(out/'progress.json', row)
        write(out/'result.json', report)
        print(json.dumps(row, allow_nan=False), flush=True)

    def budget():
        if time.monotonic()-start >= cfg['application_seconds_cap']:
            raise BudgetReached()

    try:
        tests = unittest.defaultTestLoader.loadTestsFromNames([
            'test_cf4_r1_particle_forward', 'test_cf4_chunked_hmc', 'test_cf4_r1_resolution_mixing'])
        result = unittest.TextTestRunner(verbosity=2).run(tests)
        report['tests'] = result.testsRun
        if not result.wasSuccessful():
            raise RuntimeError('focused regressions failed')
        progress('tests_pass')
        n, box = model['n'], model['box_cMpc_h']
        centers, radius = jnp.asarray(model['aperture_centers_cMpc_h']), model['aperture_radius_cMpc_h']
        observed, sigma = jnp.asarray(data['values']), jnp.asarray(data['sigma'])
        write(out/'mock_data.json', data)  # Exact reused data, NEVER generated with a new solver.
        evolves, initials, particle_mass = {}, {}, None
        report['fidelity'] = {}
        for label, mesh, tf in cfg['fidelity_arms']:
            budget()
            tick = time.monotonic()
            evolve, initial, _, _, mp = make_dynamics(model, mesh_ratio=mesh, time_factor=tf)
            if particle_mass is not None:
                np.testing.assert_allclose(mp, particle_mass, rtol=1e-14)
            particle_mass = mp
            mass = jnp.full(n**3, mp)
            mean_mass = n**3*mp/box**3*(2*np.pi)**1.5*radius**3
            rows = []
            for seed in cfg['fidelity_seeds']:
                budget()
                white = jnp.asarray(np.random.default_rng(seed).normal(size=n**3))
                ic = ready(initial(white))
                if seed in initials:
                    for actual, expected in zip(ic, initials[seed]):
                        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                else:
                    initials[seed] = tuple(np.asarray(x) for x in ic)
                x, v = ready(evolve(white))
                if not all(np.isfinite(np.asarray(a)).all() for a in (x, v)):
                    raise FloatingPointError('nonfinite fidelity evolution')
                m = ready(aperture_moments(x, v, mass, centers, radius, box))
                vector = np.asarray(observation_vector(m, mean_mass))
                rows.append(dict(seed=seed, observables=vector.tolist(),
                                 moments={k: np.asarray(a).tolist() for k, a in m.items()}))
                np.savez_compressed(out/f'{label}_seed{seed}.npz', white=np.asarray(white),
                                    position_cMpc_h=np.asarray(x), velocity_km_s=np.asarray(v))
                progress('fidelity', arm=label, seed=seed)
            report['fidelity'][label] = dict(cases=rows, seconds=time.monotonic()-tick,
                                          mesh_dx_cMpc_h=box/(n*mesh), max_delta_a=model['a_nbody_maxstep']/tf)
            # Retain compact observables; executable caches are released before sampling.
            evolves[label] = np.array([r['observables'] for r in rows])
            progress('fidelity_arm_complete', arm=label)
        contrasts = {}
        for name, lo, hi in [('force_1_to_2', 'mesh1_dt128', 'mesh2_dt128'),
                             ('force_2_to_4', 'mesh2_dt128', 'mesh4_dt128'),
                             ('time_128_to_256', 'mesh4_dt128', 'mesh4_dt256')]:
            delta = (evolves[hi]-evolves[lo])/np.asarray(sigma)
            contrasts[name] = dict(delta_in_fixed_mock_sigma=delta.tolist(),
                RMS_in_sigma=float(np.sqrt(np.mean(delta**2))), max_abs_in_sigma=float(np.max(np.abs(delta))),
                mass_change_fraction=np.expm1((evolves[hi]-evolves[lo])[:, ::7]).tolist())
        report['fidelity_comparison'] = contrasts
        report['force_difference_decreases'] = (
            contrasts['force_2_to_4']['RMS_in_sigma'] < contrasts['force_1_to_2']['RMS_in_sigma'])
        report['particle_mass_Msun_h'] = particle_mass
        report['fidelity_limit'] = 'Same particle sampling, different PM force resolution; finer arm is not truth or independent calibration.'
        del initial, evolve, initials, x, v, m, ic
        jax.clear_caches()
        gc.collect()
        progress('fidelity_complete')
        budget()

        evolve, _, _, _, mp = make_dynamics(model)  # UNCHANGED entry posterior target.
        mass = jnp.full(n**3, mp)

        @jax.jit
        def physical(white):
            x, v = evolve(white)
            m = aperture_moments(x, v, mass, centers, radius, box)
            return jnp.concatenate((observation_vector(m, mean_mass), jnp.sqrt(m['variance_km2_s2']).ravel()))

        @jax.jit
        def logdensity(white):
            return -.5*jnp.sum(((physical(white)[:21]-observed)/sigma)**2)-.5*jnp.vdot(white, white)

        # Data-generating truth is used only to verify the old forward, never initialization.
        truth = jnp.asarray(np.random.default_rng(model['truth_seed']).normal(size=n**3))
        truth_obs = np.asarray(ready(physical(truth)))[:21]
        old_obs = observation_vector({k: jnp.asarray(v) for k, v in old['fidelity']['base'].items()}, mean_mass)
        np.testing.assert_allclose(truth_obs, old_obs, rtol=1e-10, atol=1e-10)
        report['unchanged_baseline_observables_max_abs_error'] = float(np.max(np.abs(truth_obs-np.asarray(old_obs))))
        physical_chunk = jax.jit(lambda w: jax.lax.map(physical, w))
        coordinates = np.indices((n,)*3).reshape(3, -1).T/n
        basis, names = [np.ones(n**3)/np.sqrt(n**3)], ['white_DC']
        for k in cfg['projection_wavevectors']:
            angle = 2*np.pi*(coordinates@np.asarray(k))
            for label, fn in [('cos', np.cos), ('sin', np.sin)]:
                mode = fn(angle)
                basis.append(mode/np.linalg.norm(mode))
                names.append(f'white_k{k}_{label}')
        basis = np.stack(basis, axis=1)
        names = ['log_likelihood', 'white_norm2', 'white0', 'white1', 'white2']+names
        names += [f'aperture{p}_{q}' for p in range(3) for q in
                  ['log_mass', 'offset_x', 'offset_y', 'offset_z', 'velocity_x', 'velocity_y', 'velocity_z']]
        names += [f'aperture{p}_sigma_{q}' for p in range(3) for q in 'xyz']
        report['sampling'] = {}
        for arm in cfg['sampling_arms']:
            budget()
            arm_dir = out/arm['name']
            arm_dir.mkdir()
            settings = {**model['sampler'], **{k: v for k, v in arm.items() if k.startswith('integration_steps')},
                        'maximum_step_size': cfg['sampler_max_step']}
            initialize, warm, sample, final = make_chunks(logdensity, n**3, settings, record_steps=True)
            all_diagnostics, chain_rows = [], []
            arm_start = time.monotonic()
            report['sampling'][arm['name']] = dict(settings=settings, chains=chain_rows, complete=False)
            for chain, seed in enumerate(cfg['initialization_seeds']):
                budget()
                chain_dir = arm_dir/f'chain_{chain}'
                chain_dir.mkdir()
                init = jnp.asarray(np.random.default_rng(seed).normal(size=n**3))
                state, adaptation = ready(initialize(init))
                key = jax.random.PRNGKey(cfg['sampler_seed']+chain)
                warm_div, warm_work = 0, 0
                tick = time.monotonic()
                for begin in range(0, cfg['warmup_steps'], cfg['chunk_size']):
                    budget()
                    key, sub = jax.random.split(key)
                    (state, adaptation), record = ready(warm(state, adaptation, jax.random.split(sub, cfg['chunk_size'])))
                    arrays = checked_record(record, state)
                    warm_div += int(arrays[3].sum())
                    warm_work += int(arrays[7].sum())
                    if (begin+cfg['chunk_size']) % 32 == 0:
                        progress('warmup', arm=arm['name'], chain=chain, steps=begin+cfg['chunk_size'])
                step = ready(final(adaptation))
                warm_seconds = time.monotonic()-tick
                traces, projections, white_saved = [], [], []
                tick = time.monotonic()
                try:
                    for begin in range(0, arm['sampling_steps'], cfg['chunk_size']):
                        budget()
                        key, sub = jax.random.split(key)
                        state, record = ready(sample(state, step, jax.random.split(sub, cfg['chunk_size'])))
                        arrays = checked_record(record, state)
                        white, logp = arrays[:2]
                        fields = np.asarray(ready(physical_chunk(record[0])))
                        norm2 = np.sum(white**2, axis=1)
                        projections.append(np.column_stack((logp+.5*norm2, norm2, white[:, :3], white@basis, fields)))
                        traces.append(np.column_stack(arrays[1:]))
                        white_saved.append(white.astype(np.float32))
                        if (begin+cfg['chunk_size']) % 64 == 0:
                            progress('sampling', arm=arm['name'], chain=chain, steps=begin+cfg['chunk_size'],
                                     acceptance=float(arrays[2].mean()))
                finally:
                    np.savez_compressed(chain_dir/'last_state.npz', white=np.asarray(state.position),
                                        step_size=np.asarray(step), random_key=np.asarray(key))
                    if traces:
                        trace = np.concatenate(traces)
                        projected = np.concatenate(projections)
                        np.savez_compressed(chain_dir/'samples.npz', white_float32=np.concatenate(white_saved),
                            diagnostics=projected, diagnostic_names=np.array(names),
                            logdensity=trace[:, 0], acceptance=trace[:, 1], divergent=trace[:, 2].astype(bool),
                            energy=trace[:, 3], step_size=trace[:, 5], integration_steps=trace[:, 6].astype(int))
                seconds = time.monotonic()-tick
                row = dict(chain=chain, initialization_seed=seed, initialized_from_truth=False,
                    draws=len(trace), warmup_divergences=warm_div, warmup_integration_steps=warm_work,
                    warmup_seconds=warm_seconds, sampling_seconds=seconds, step_size=float(step),
                    sample_divergences=int(trace[:, 2].sum()), mean_acceptance=float(trace[:, 1].mean()),
                    sample_integration_steps=int(trace[:, 6].sum()),
                    mean_trajectory_length=float(np.mean(trace[:, 5]*trace[:, 6])))
                chain_rows.append(row)
                all_diagnostics.append(projected)
                progress('chain_complete', arm=arm['name'], **row)
            d = diagnostics(np.stack(all_diagnostics), names)
            limits = cfg['engineering_mixing_limits']
            d['engineering_projection_pass'] = bool(d['max_Rhat'] <= limits['Rhat_max'] and
                d['min_bulk_ESS'] >= limits['bulk_ESS_min'] and d['min_tail_ESS'] >= limits['tail_ESS_min'] and
                sum(c['sample_divergences'] for c in chain_rows) == 0)
            d['sampling_seconds_per_min_bulk_ESS'] = sum(c['sampling_seconds'] for c in chain_rows)/d['min_bulk_ESS']
            d['integration_steps_per_min_bulk_ESS'] = sum(c['sample_integration_steps'] for c in chain_rows)/d['min_bulk_ESS']
            report['sampling'][arm['name']].update(complete=True, diagnostics=d, total_seconds=time.monotonic()-arm_start)
            progress('sampling_arm_complete', arm=arm['name'], max_Rhat=d['max_Rhat'], min_bulk_ESS=d['min_bulk_ESS'])
        report['status'] = 'COMPLETE_DRIVER_JUDGMENT_REQUIRED'
    except BudgetReached:
        report['status'] = 'INCOMPLETE_APPLICATION_BUDGET'
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-start
        report['R1_cumulative_application_plus_previous_allocation_seconds'] = report['elapsed_seconds']+cfg['R1_previous_GPU_seconds']
        report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        report['device_peak_bytes'] = (jax.devices()[0].memory_stats() or {}).get('peak_bytes_in_use')
        progress('finish', status=report['status'])


if __name__ == '__main__':
    main()
