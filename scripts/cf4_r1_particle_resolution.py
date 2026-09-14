"""R1 particle/force/time control, preserving the original continuous LPT state."""
import gc
import json
import os
from pathlib import Path
import resource
import time
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from cf4_r1_particle_entry import ROOT, write, ready, BudgetReached
from cf4_r1_particle_forward import (
    make_dynamics, make_configuration, aperture_moments, particle_grid, periodic_delta)
from cf4_r1_particle_resolution import refine_lpt_state, make_state_evolution
from cf4_r1_resolution_mixing import observation_vector


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    cfg = json.loads((ROOT/'config/cf4_r1_particle_resolution_v3.json').read_text())
    entry, comparison = Path(cfg['entry_directory']), Path(cfg['comparison_directory'])
    model = json.loads((entry/'result.json').read_text())['config']
    data = json.loads((entry/'mock_data.json').read_text())
    sigma = np.asarray(data['sigma'])
    out = Path(cfg['output_root'])/('job_'+os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(status='RUNNING', source_commit=os.environ['EXPECTED_COMMIT'], config=cfg,
        model=model, observed_posterior=False, resolved_LG=False, full_power_fine_IC=False,
        independent_solver_calibrated=False, nested_gravity_verified=False, R1_complete=False,
        arms={}, endpoint_reproduction=[], planar_analytic_control={})

    def progress(stage, **extra):
        row = dict(stage=stage, elapsed_seconds=time.monotonic()-start, **extra)
        write(out/'result.json', report)
        write(out/'progress.json', row)
        print(json.dumps(row, allow_nan=False), flush=True)

    def budget():
        if time.monotonic()-start >= cfg['application_seconds_cap']:
            raise BudgetReached()

    try:
        tests = unittest.defaultTestLoader.loadTestsFromNames([
            'test_cf4_r1_particle_forward', 'test_cf4_r1_particle_resolution'])
        result = unittest.TextTestRunner(verbosity=2).run(tests)
        report['tests'] = result.testsRun
        if not result.wasSuccessful():
            raise RuntimeError('focused regressions failed')
        progress('tests_pass')
        coarse_n, box = model['n'], model['box_cMpc_h']
        centers, radius = jnp.asarray(model['aperture_centers_cMpc_h']), model['aperture_radius_cMpc_h']
        _, initial, _, _, coarse_mass = make_dynamics(model)
        q = np.indices((coarse_n,)*3).reshape(3, -1).T*(box/coarse_n)
        common_initial = {}
        for seed in cfg['seeds']:
            white = jnp.asarray(np.random.default_rng(seed).normal(size=coarse_n**3))
            x, v = ready(initial(white))
            psi = (np.asarray(x)-q).reshape((coarse_n,)*3+(3,))
            vel = np.asarray(v).reshape(psi.shape)
            common_initial[seed] = (psi, vel)
            np.savez_compressed(out/f'coarse_initial_seed{seed}.npz', displacement_cMpc_h=psi,
                                velocity_km_s=vel, white=np.asarray(white))
        total_mass = coarse_n**3*coarse_mass
        mean_mass = total_mass/box**3*(2*np.pi)**1.5*radius**3
        readout_conf, _ = make_configuration({**model, 'n': cfg['readout_n']})
        report['common_readout_dx_cMpc_h'] = box/cfg['readout_n']
        del initial, x, v
        jax.clear_caches()
        gc.collect()

        for arm in cfg['arms']:
            budget()
            tick = time.monotonic()
            n, mesh = arm['n'], arm['mesh_n']
            evolve, conf, cosmo, mp = make_state_evolution(
                {**model, 'n': n}, mesh_ratio=mesh//n, time_factor=arm['time_factor'])
            np.testing.assert_allclose(mp*n**3, total_mass, rtol=1e-13)
            mass = jnp.full(n**3, mp)
            cases = []
            report['arms'][arm['name']] = dict(particle_n=n, force_n=mesh,
                particle_mass_Msun_h=mp, particle_spacing_cMpc_h=box/n,
                force_dx_cMpc_h=box/mesh, max_delta_a=model['a_nbody_maxstep']/arm['time_factor'], cases=cases)
            for seed in cfg['seeds']:
                budget()
                psi, vel = refine_lpt_state(*common_initial[seed], n)
                ratio = n//coarse_n
                checks = []
                for fine, coarse in zip((psi, vel), common_initial[seed]):
                    np.testing.assert_allclose(fine[::ratio, ::ratio, ::ratio], coarse, rtol=1e-12, atol=1e-12)
                    np.testing.assert_allclose(fine.mean(axis=(0, 1, 2)), coarse.mean(axis=(0, 1, 2)), atol=1e-11)
                    checks.append(float(np.max(np.abs(fine[::ratio, ::ratio, ::ratio]-coarse))))
                t0 = time.monotonic()
                x, v = ready(evolve(jnp.asarray(psi), jnp.asarray(vel)))
                forward_seconds = time.monotonic()-t0
                if not all(np.isfinite(np.asarray(a)).all() for a in (x, v)):
                    raise FloatingPointError('nonfinite forward particle state')
                moments = ready(aperture_moments(x, v, mass, centers, radius, box))
                vector = np.asarray(observation_vector(moments, mean_mass))
                if n == coarse_n:
                    # Same initial state, same PM integrator; catches conversion/scan defects.
                    with np.load(comparison/f'mesh4_dt256_seed{seed}.npz') as previous:
                        dx = np.asarray(periodic_delta(x, jnp.asarray(previous['position_cMpc_h']), box))
                        dv = np.asarray(v)-previous['velocity_km_s']
                    check = dict(seed=seed, max_position_error_cMpc_h=float(np.max(np.abs(dx))),
                                 max_velocity_error_km_s=float(np.max(np.abs(dv))))
                    report['endpoint_reproduction'].append(check)
                    np.testing.assert_allclose(dx, 0., atol=1e-7)
                    np.testing.assert_allclose(dv, 0., atol=1e-4)
                field = ready(particle_grid(x, v, mass, readout_conf))
                np.testing.assert_allclose(field['rho'].mean(), 1., atol=1e-12)
                np.testing.assert_allclose(field['mass'].sum(), total_mass, rtol=1e-12)
                np.testing.assert_allclose(field['momentum'].sum(axis=(0, 1, 2)),
                                          (mass[:, None]*v).sum(axis=0), rtol=1e-8, atol=100.)
                np.savez_compressed(out/f"{arm['name']}_seed{seed}_field.npz",
                    **{k: np.asarray(field[k]).astype(np.float32) for k in
                       ('rho', 'mean_velocity_km_s', 'variance_km2_s2')}, valid=np.asarray(field['valid']))
                if arm['name'] == cfg['arms'][-1]['name']:
                    # Exact FP64 reference IC AND endpoint retained for an independent solver.
                    np.savez_compressed(out/f'reference_particles_seed{seed}.npz',
                        initial_displacement_cMpc_h=psi, initial_velocity_km_s=vel,
                        position_cMpc_h=np.asarray(x), velocity_km_s=np.asarray(v),
                        particle_mass_Msun_h=np.asarray(mp))
                cases.append(dict(seed=seed, observables=vector.tolist(),
                    moments={k: np.asarray(a).tolist() for k, a in moments.items()},
                    common_coarse_node_max_errors=checks, forward_seconds=forward_seconds))
                progress('case_complete', arm=arm['name'], seed=seed, forward_seconds=forward_seconds)
            if arm['name'] in cfg['planar_control']['arms']:
                # Plane-parallel cold flow is exactly ZA before shell crossing.
                # Background growth is shared; the reference does NOT integrate particles.
                from pmwd import growth
                from pmwd.cosmology import E2
                budget()
                t0 = time.monotonic()
                qwave = np.indices((n,)*3).reshape(3, -1).T*(box/n)
                wave_k = cfg['planar_control']['wave_number']*2*np.pi/box
                amplitude = cfg['planar_control']['final_linear_amplitude']
                a0, a1 = model['a_start'], model['a_stop']
                d0, d1 = float(growth(a0, cosmo, conf)), float(growth(a1, cosmo, conf))
                psi_wave = np.zeros_like(qwave)
                psi_wave[:, 0] = -amplitude/(wave_k*d1)*np.sin(wave_k*qwave[:, 0])
                vel_factor = lambda a: float(100*a*jnp.sqrt(E2(a, cosmo))*growth(a, cosmo, conf, deriv=1))
                exact_x, exact_v = qwave+d1*psi_wave, vel_factor(a1)*psi_wave
                wave_x, wave_v = ready(evolve(jnp.asarray(d0*psi_wave), jnp.asarray(vel_factor(a0)*psi_wave)))
                wave_dx = np.asarray(periodic_delta(wave_x, jnp.asarray(exact_x), box))
                wave_dv = np.asarray(wave_v)-exact_v
                measured = aperture_moments(wave_x, wave_v, mass, centers, radius, box)
                expected = aperture_moments(jnp.asarray(exact_x), jnp.asarray(exact_v), mass, centers, radius, box)
                obs_error = np.asarray(observation_vector(measured, mean_mass)-observation_vector(expected, mean_mass))/sigma
                report['planar_analytic_control'][arm['name']] = dict(
                    min_exact_jacobian=1-amplitude, displacement_relative_RMS=float(np.linalg.norm(wave_dx)/np.linalg.norm(d1*psi_wave)),
                    velocity_relative_RMS=float(np.linalg.norm(wave_dv)/np.linalg.norm(exact_v)),
                    max_observable_error_in_fixed_mock_sigma=float(np.max(np.abs(obs_error))),
                    seconds=time.monotonic()-t0,
                    scope='Exact pre-crossing planar gravity/time reference using shared background growth; not a collapsed-halo validation.')
                progress('planar_control', arm=arm['name'])
                del qwave, psi_wave, exact_x, exact_v, wave_x, wave_v, wave_dx, wave_dv, measured, expected
            report['arms'][arm['name']]['seconds'] = time.monotonic()-tick
            del evolve, conf, cosmo, mass, psi, vel, x, v, field, moments
            jax.clear_caches()
            gc.collect()
            progress('arm_complete', arm=arm['name'])

        report['comparisons'] = {}
        for name, lo, hi in cfg['contrasts']:
            left = np.asarray([c['observables'] for c in report['arms'][lo]['cases']])
            right = np.asarray([c['observables'] for c in report['arms'][hi]['cases']])
            delta = right-left
            physical_delta = delta.reshape(len(cfg['seeds']), 3, 7)
            dm = np.expm1(delta[:, ::7])
            report['comparisons'][name] = dict(
                delta_in_fixed_mock_sigma=(delta/sigma).tolist(),
                RMS_in_sigma=float(np.sqrt(np.mean((delta/sigma)**2))),
                max_abs_in_sigma=float(np.max(np.abs(delta/sigma))),
                mass_change_fraction=dm.tolist(), max_abs_mass_change_fraction=float(np.max(np.abs(dm))),
                max_centroid_shift_cMpc_h=float(np.max(np.linalg.norm(physical_delta[:, :, 1:4], axis=-1))),
                max_mean_velocity_change_km_s=float(np.max(np.linalg.norm(physical_delta[:, :, 4:7], axis=-1))))
        report['particle_difference_decreases'] = (
            report['comparisons']['particles_64_to_128']['RMS_in_sigma'] <
            report['comparisons']['particles_32_to_64']['RMS_in_sigma'])
        report['status'] = 'COMPLETE_DRIVER_JUDGMENT_REQUIRED'
    except BudgetReached:
        report['status'] = 'INCOMPLETE_APPLICATION_BUDGET'
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-start
        report['R1_previous_allocation_plus_current_application_seconds'] = (
            cfg['previous_R1_allocation_seconds']+report['elapsed_seconds'])
        report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        report['device_peak_bytes'] = (jax.devices()[0].memory_stats() or {}).get('peak_bytes_in_use')
        progress('finish', status=report['status'])


if __name__ == '__main__':
    main()
