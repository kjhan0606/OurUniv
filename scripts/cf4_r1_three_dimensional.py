"""Fixed 3D same-state CIC/TSC sensitivity; independent solver is still separate."""
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

from cf4_r1_particle_entry import ROOT, ready, write, BudgetReached
from cf4_r1_particle_forward import make_configuration, aperture_moments, particle_grid, periodic_delta
from cf4_r1_tsc_diagnostic import make_evolution


def vector(moments, mean_mass):
    return jnp.concatenate((jnp.log(moments['mass_Msun_h']/mean_mass)[:, None],
                            moments['offset_cMpc_h'], moments['mean_velocity_km_s']), axis=1).ravel()


def bands(density, reference, box):
    """Readout band differences, never observational information boundaries."""
    n = len(density)
    k = 2*np.pi*np.fft.fftfreq(n, d=box/n)
    kr = 2*np.pi*np.fft.rfftfreq(n, d=box/n)
    magnitude = np.sqrt(k[:, None, None]**2+k[None, :, None]**2+kr[None, None, :]**2)
    weights = np.broadcast_to(np.where((np.arange(len(kr)) == 0) | (np.arange(len(kr)) == n//2), 1., 2.), magnitude.shape)
    left, right = np.fft.rfftn(density-1), np.fft.rfftn(reference-1)
    edges = [0., np.pi/2, np.pi, np.pi/.3, np.pi/(box/n)]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (magnitude > lo) & (magnitude <= hi)
        w, a, b = weights[mask], left[mask], right[mask]
        pa, pb = np.sum(w*abs(a)**2), np.sum(w*abs(b)**2)
        rows.append(dict(k_min_h_cMpc=lo, k_max_h_cMpc=hi,
            weighted_mode_count=float(w.sum()), power_ratio=float(pa/pb),
            cross_correlation=float(np.sum(w*(a*b.conj()).real)/np.sqrt(pa*pb)),
            difference_relative_RMS=float(np.sqrt(np.sum(w*abs(a-b)**2)/pb))))
    return rows


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    cfg = json.loads((ROOT/'config/cf4_r1_three_dimensional_v6.json').read_text())
    reference = Path(cfg['reference_directory'])
    entry = Path(cfg['entry_directory'])
    model = json.loads((entry/'result.json').read_text())['config']
    data = json.loads((entry/'mock_data.json').read_text())
    model = {**model, 'n': cfg['particle_n']}
    box, n = model['box_cMpc_h'], model['n']
    centers, radius = jnp.asarray(model['aperture_centers_cMpc_h']), model['aperture_radius_cMpc_h']
    out = Path(cfg['output_root'])/('job_'+os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(status='RUNNING', source_commit=os.environ['EXPECTED_COMMIT'], config=cfg,
        observed_posterior=False, independent_solver_calibrated=False, resolved_LG=False,
        R1_complete=False, production_force_changed=False, arms={})

    def progress(stage, **more):
        write(out/'result.json', report)
        item = dict(stage=stage, elapsed_seconds=time.monotonic()-start, **more)
        write(out/'progress.json', item)
        print(json.dumps(item, allow_nan=False), flush=True)

    def budget():
        if time.monotonic()-start > cfg['application_seconds_cap']:
            raise BudgetReached()

    try:
        tests = unittest.defaultTestLoader.loadTestsFromName('test_cf4_r1_tsc_diagnostic')
        result = unittest.TextTestRunner(verbosity=2).run(tests)
        report['tests'] = result.testsRun
        if not result.wasSuccessful():
            raise RuntimeError('TSC regressions failed')
        progress('tests_pass')
        readout, _ = make_configuration({**model, 'n': cfg['readout_n']})
        q = np.indices((n,)*3).reshape(3, -1).T*box/n
        for arm in cfg['arms']:
            budget()
            evolve, force_at, conf, cosmo = make_evolution(model, assignment=arm['assignment'],
                mesh_ratio=arm['mesh_ratio'], time_factor=arm['time_factor'])
            cases = []
            report['arms'][arm['name']] = cases
            for seed in cfg['seeds']:
                budget()
                tick = time.monotonic()
                with np.load(reference/f'reference_particles_seed{seed}.npz') as stored:
                    psi = jnp.asarray(stored['initial_displacement_cMpc_h'])
                    vi = jnp.asarray(stored['initial_velocity_km_s'])
                    xr = jnp.asarray(stored['position_cMpc_h'])
                    vr = jnp.asarray(stored['velocity_km_s'])
                    mp = float(stored['particle_mass_Msun_h'])
                np.testing.assert_allclose(mp, float(cosmo.ptcl_mass)*1e10, rtol=1e-13)
                mass = jnp.full(n**3, mp)
                x, v = ready(evolve(psi, vi))
                if not all(np.isfinite(np.asarray(a)).all() for a in (x, v)):
                    raise FloatingPointError('nonfinite trajectory')
                dx = np.asarray(periodic_delta(x, xr, box))
                dv = np.asarray(v-vr)
                if arm['assignment'] == 'cic':
                    np.testing.assert_allclose(dx, 0., atol=1e-7)
                    np.testing.assert_allclose(dv, 0., atol=1e-4)
                mean_mass = mp*n**3/box**3*(2*np.pi)**1.5*radius**3
                moments = ready(aperture_moments(x, v, mass, centers, radius, box))
                observable = np.asarray(vector(moments, mean_mass))
                field = ready(particle_grid(x, v, mass, readout))
                rho = np.asarray(field['rho'])
                np.testing.assert_allclose(rho.mean(), 1., atol=1e-12)
                np.testing.assert_allclose(field['momentum'].sum(axis=(0, 1, 2)),
                                          (mass[:, None]*v).sum(axis=0), rtol=1e-8, atol=100.)
                if arm['assignment'] == 'cic':
                    np.savez_compressed(out/f'cic_reference_seed{seed}.npz', rho=rho,
                                        observables=observable)
                with np.load(out/f'cic_reference_seed{seed}.npz') as old:
                    rho_ref, obs_ref = old['rho'], old['observables']
                row = dict(seed=seed, observables=observable.tolist(),
                    relative_probe_mass_difference=np.expm1((observable-obs_ref)[::7]).tolist(),
                    observable_difference_in_fixed_mock_sigma=((observable-obs_ref)/np.asarray(data['sigma'])).tolist(),
                    position_difference_RMS_cMpc_h=float(np.sqrt(np.mean(np.sum(dx**2, axis=1)))),
                    velocity_difference_RMS_km_s=float(np.sqrt(np.mean(np.sum(dv**2, axis=1)))),
                    velocity_difference_relative_RMS=float(np.linalg.norm(dv)/np.linalg.norm(np.asarray(vr))),
                    density_bands=bands(rho, rho_ref, box))
                # Compare operators at IDENTICAL states, not each arm's evolved state.
                for label, position in [('initial', jnp.asarray(q)+psi.reshape((-1, 3))), ('common_final', xr)]:
                    acceleration = np.asarray(ready(force_at(position)))
                    if arm['assignment'] == 'cic':
                        np.save(out/f'cic_force_{label}_seed{seed}.npy', acceleration)
                    old_force = np.load(out/f'cic_force_{label}_seed{seed}.npy')
                    row[label+'_force_difference_relative_RMS'] = float(np.linalg.norm(acceleration-old_force)/np.linalg.norm(old_force))
                if arm['name'] == cfg['arms'][-1]['name']:
                    with np.load(out/f'tsc_dt256_seed{seed}.npz') as coarse_time:
                        time_dx = np.asarray(periodic_delta(x, jnp.asarray(coarse_time['position_cMpc_h']), box))
                        time_dv = np.asarray(v)-coarse_time['velocity_km_s']
                    row['time_halving_position_difference_RMS_cMpc_h'] = float(np.sqrt(np.mean(np.sum(time_dx**2, axis=1))))
                    row['time_halving_velocity_difference_RMS_km_s'] = float(np.sqrt(np.mean(np.sum(time_dv**2, axis=1))))
                    row['time_halving_velocity_difference_relative_RMS'] = float(np.linalg.norm(time_dv)/np.linalg.norm(np.asarray(v)))
                    del time_dx, time_dv
                    np.savez_compressed(out/f'tsc_final_seed{seed}.npz', position_cMpc_h=np.asarray(x),
                                        velocity_km_s=np.asarray(v), particle_mass_Msun_h=mp)
                    np.savez_compressed(out/f'tsc_field_seed{seed}.npz',
                        **{k: np.asarray(field[k]) for k in ('rho', 'mean_velocity_km_s', 'variance_km2_s2')})
                if arm['name'] == 'tsc256_t256':
                    np.savez_compressed(out/f'tsc_dt256_seed{seed}.npz', position_cMpc_h=np.asarray(x),
                                        velocity_km_s=np.asarray(v))
                row['seconds'] = time.monotonic()-tick
                cases.append(row)
                progress('case_complete', arm=arm['name'], seed=seed,
                         velocity_difference_relative_RMS=row['velocity_difference_relative_RMS'])
                del psi, vi, xr, vr, x, v, dx, dv, mass, field, rho, rho_ref, acceleration, old_force
            del evolve, force_at, conf, cosmo
            jax.clear_caches()
            gc.collect()
        report['status'] = 'COMPLETE_SENSITIVITY_NOT_INDEPENDENT_VALIDATION'
    except BudgetReached:
        report['status'] = 'INCOMPLETE_APPLICATION_BUDGET'
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic()-start
        report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        report['device_peak_bytes'] = (jax.devices()[0].memory_stats() or {}).get('peak_bytes_in_use')
        progress('finish', status=report['status'])


if __name__ == '__main__':
    main()
