"""Separate analytic-reference, integrator, lattice-phase and PM-force effects."""
import gc
import argparse
import json
import os
from pathlib import Path
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from cf4_r1_particle_entry import ROOT, ready, write, BudgetReached
from cf4_r1_particle_forward import make_configuration


def growth_quadrature(a, omega_m):
    """Independent flat matter+Lambda growth D and dD/dln(a), D~a early."""
    e = lambda x: np.sqrt(omega_m/x**3+1-omega_m)
    integral = quad(lambda x: 0. if x == 0 else 1/(x*e(x))**3,
                    0., a, epsabs=1e-14, epsrel=1e-11)[0]
    d = 2.5*omega_m*e(a)*integral
    derivative = d*(-1.5*omega_m/(a**3*e(a)**2)+1/(a*a*e(a)**3*integral))
    return np.array([d, derivative])


def gain_error(value, reference):
    norm2 = jnp.sum(reference**2)
    gain = jnp.sum(value*reference)/norm2
    relative = jnp.sqrt(jnp.sum((value-reference)**2)/norm2)
    shape = jnp.sqrt(jnp.sum((value-gain*reference)**2)/norm2)
    return jnp.stack((gain, relative, shape))


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/cf4_r1_planar_diagnosis_v4.json')
    args = parser.parse_args()
    cfg = json.loads((ROOT/args.config).read_text())
    model = json.loads((Path(cfg['entry_directory'])/'result.json').read_text())['config']
    previous = json.loads((Path(cfg['reference_directory'])/'result.json').read_text())
    model = {**model, 'n': cfg['particle_n']}
    out = Path(cfg['output_root'])/('job_'+os.environ['SLURM_JOB_ID'])
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(status='RUNNING', config=cfg, source_commit=os.environ['EXPECTED_COMMIT'], cases={},
                  production_force_changed=False, accuracy_promoted=False, R1_complete=False)

    def progress(stage, **extra):
        row = dict(stage=stage, elapsed_seconds=time.monotonic()-start, **extra)
        write(out/'result.json', report)
        write(out/'progress.json', row)
        print(json.dumps(row, allow_nan=False), flush=True)

    def budget():
        if time.monotonic()-start >= cfg['application_seconds_cap']:
            raise BudgetReached()

    try:
        # Independent quadrature is used BEFORE accepting the planar reference.
        conf, cosmo = make_configuration(model, time_factor=cfg['time_factor'])
        from pmwd import growth, lpt, scatter, gather
        from pmwd.cosmology import E2
        from pmwd.particles import Particles
        from pmwd.gravity import gravity, neg_grad
        from pmwd.nbody import drift, kick
        a0, a1 = model['a_start'], model['a_stop']
        n, box = model['n'], model['box_cMpc_h']
        omega_m = model['cosmology']['Om']
        scales = np.array([a0, .03, .05, .1, .2, .4, .7, a1])
        independent = np.stack([growth_quadrature(a, omega_m) for a in scales])
        tabulated = np.array([[float(growth(a, cosmo, conf, deriv=i)) for i in (0, 1)] for a in scales])
        relative = (tabulated-independent)/independent
        report['growth_reference'] = dict(scale_factors=scales.tolist(), independent=independent.tolist(),
            pmwd=tabulated.tolist(), relative_difference=relative.tolist(), max_abs_relative_difference=float(np.max(np.abs(relative))))
        np.testing.assert_allclose(tabulated, independent, rtol=2e-4, atol=1e-10)
        del conf, cosmo
        progress('growth_reference_checked')

        for case in cfg['cases']:
            budget()
            tick = time.monotonic()
            conf, cosmo = make_configuration(model, mesh_ratio=case['mesh_ratio'], time_factor=cfg['time_factor'])
            grid = Particles.gen_grid(conf, vel=True, acc=True)
            offset = case['mesh_shift']*conf.cell_size
            q = grid.pos(wrap=False)+offset
            wave_k = cfg['wave_number']*2*np.pi/box
            d1 = float(growth(a1, cosmo, conf))
            psi = jnp.zeros_like(q).at[:, 0].set(-cfg['final_linear_amplitude']/(wave_k*d1)*jnp.sin(wave_k*q[:, 0]))
            d = lambda a: growth(a, cosmo, conf)
            canonical_factor = lambda a: a*a*jnp.sqrt(E2(a, cosmo))*growth(a, cosmo, conf, deriv=1)
            initial = grid.replace(disp=grid.disp+offset+d(a0)*psi, vel=canonical_factor(a0)*psi)
            if case['name'] == 'pm128':
                # Independent pure Fourier-mode LPT construction checks sign and km/s conversion.
                modes = jnp.zeros((n, n, n//2+1), dtype=jnp.complex128)
                coeff = box**3*cfg['final_linear_amplitude']/(2*d1)
                modes = modes.at[cfg['wave_number'], 0, 0].set(coeff).at[-cfg['wave_number'], 0, 0].set(coeff)
                lpt_state, _ = ready(lpt(modes, cosmo, conf))
                dx = np.asarray(lpt_state.disp-initial.disp)
                dv = np.asarray((lpt_state.vel-initial.vel)*100/a0)
                report['initial_LPT_reference'] = dict(max_displacement_error_cMpc_h=float(np.max(np.abs(dx))),
                                                     max_velocity_error_km_s=float(np.max(np.abs(dv))))
                np.testing.assert_allclose(dx, 0., atol=1e-11)
                np.testing.assert_allclose(dv, 0., atol=1e-9)
                del lpt_state, modes

            def force(particles):
                if case['force'] == 'pm':
                    return gravity(a0, particles, cosmo, conf)  # PMWD spatial kernel is a-independent.
                if case['force'] == 'tsc':
                    from cf4_r1_tsc_diagnostic import force as tsc_force
                    return tsc_force(particles.pos(wrap=False), conf.mesh_shape[0],
                                     conf.cell_size, omega_m)
                if case['force'] == 'sheets':
                    displacement = particles.pos(wrap=False)-q
                    return 1.5*omega_m*(displacement-displacement.mean(axis=0))
                # Project out transverse particle-lattice modes; keep axial CIC and Poisson/gather.
                density = scatter(particles, conf).mean(axis=(1, 2))-1
                k = 2*jnp.pi*jnp.fft.rfftfreq(conf.mesh_shape[0], d=conf.cell_size)
                src = jnp.fft.rfft(density)*1.5*omega_m
                if case['force'] == 'plane_lowpass':
                    src *= jnp.arange(len(k)) < n//2
                potential = jnp.where(k != 0, -src/jnp.where(k != 0, k*k, 1), 0.)
                force_x = jnp.fft.irfft(neg_grad(k, potential, conf.cell_size), n=conf.mesh_shape[0])
                field = jnp.broadcast_to(force_x[:, None, None], conf.mesh_shape)
                fx = gather(particles, conf, field)
                return jnp.stack((fx, jnp.zeros_like(fx), jnp.zeros_like(fx)), axis=-1)

            reference_force = 1.5*omega_m*d(a0)*psi
            init_force = ready(jax.jit(force)(initial))
            fmetrics = np.asarray(gain_error(init_force, reference_force))
            tiny = initial.replace(disp=grid.disp+offset+.001*d(a0)*psi)
            tiny_metrics = np.asarray(gain_error(ready(jax.jit(force)(tiny)), .001*reference_force))
            row = dict(case=case, initial_force=dict(gain=float(fmetrics[0]), relative_RMS=float(fmetrics[1]),
                residual_after_gain_RMS=float(fmetrics[2])), infinitesimal_force=dict(gain=float(tiny_metrics[0]),
                relative_RMS=float(tiny_metrics[1]), residual_after_gain_RMS=float(tiny_metrics[2])))
            report['cases'][case['name']] = row
            progress('initial_force', case=case['name'], **row['initial_force'])

            def metrics(particles, a):
                disp = particles.pos(wrap=False)-q
                reference_disp = d(a)*psi
                reference_vel = canonical_factor(a)*psi
                return jnp.concatenate((gain_error(disp, reference_disp), gain_error(particles.vel, reference_vel),
                                        gain_error(particles.acc, 1.5*omega_m*reference_disp)))

            @jax.jit
            def integrate(state):
                state = state.replace(acc=force(state))
                time_grid = jnp.asarray(conf.a_nbody)
                def step(particles, times):
                    prev, nxt = times
                    mid = .5*prev+.5*nxt
                    particles = kick(prev, prev, mid, particles, cosmo, conf)
                    particles = drift(mid, prev, nxt, particles, cosmo, conf)
                    particles = particles.replace(acc=force(particles))
                    particles = kick(nxt, mid, nxt, particles, cosmo, conf)
                    return particles, metrics(particles, nxt)
                return jax.lax.scan(step, state, (time_grid[:-1], time_grid[1:]))

            budget()
            final, history = ready(integrate(initial))
            history = np.asarray(history)
            if not np.isfinite(history).all():
                raise FloatingPointError('nonfinite diagnostic trajectory')
            names = [f'{quantity}_{metric}' for quantity in ['displacement', 'velocity', 'force']
                     for metric in ['gain', 'relative_RMS', 'residual_after_gain_RMS']]
            row['final'] = dict(zip(names, map(float, history[-1])))
            if case['name'] in ('pm128', 'pm256'):
                previous_key = 'p128_f128_t256' if case['name'] == 'pm128' else 'p128_f256_t256'
                old = previous['planar_analytic_control'][previous_key]
                for quantity in ['displacement', 'velocity']:
                    np.testing.assert_allclose(row['final'][quantity+'_relative_RMS'], old[quantity+'_relative_RMS'], rtol=1e-7, atol=1e-10)
                row['previous_endpoint_reproduced'] = True
            time_grid = np.asarray(conf.a_nbody)[1:]
            indices = np.unique(np.searchsorted(time_grid, scales[1:]).clip(max=len(time_grid)-1))
            row['growth_history'] = [dict(a=float(time_grid[i]), **dict(zip(names, map(float, history[i])))) for i in indices]
            np.savez_compressed(out/(case['name']+'_history.npz'), scale_factors=time_grid, history=history, names=np.array(names))
            row['seconds'] = time.monotonic()-tick
            progress('case_complete', case=case['name'], **row['final'])
            del integrate, final, initial, grid, q, psi, init_force, tiny, reference_force, conf, cosmo
            jax.clear_caches()
            gc.collect()
        report['status'] = 'COMPLETE_DIAGNOSIS_DRIVER_JUDGMENT_REQUIRED'
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
