"""Fixed-state count/CF4 IC-gradient alignment; not posterior inference."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_coarsened_live_joint import count_and_mark_parts

BASE = Path('/gpfs/kjhan/CF4/z0_density')
OUT = BASE/'r2_live_factor_tension_v2'
STATES = BASE/'r2_hierarchical_field_pilot_v1'
N, BOX = 128, 384.
LABELS = ('initial_chain0', 'chain0_retained32', 'chain1_retained32')
BANDS = ((0., .1), (.1, .3), (.3, float('inf')))


def gradient_alignment(a, b):
    dot = float(np.vdot(a, b))
    a2, b2 = float(np.vdot(a, a)), float(np.vdot(b, b))
    return dict(count_norm=np.sqrt(a2), mark_norm=np.sqrt(b2),
        count_mark_dot=dot, cosine=dot/np.sqrt(a2*b2) if a2 > 0 and b2 > 0 else None)


def band_alignment(a, b, n=N, box=BOX):
    """Real-FFT half-plane accounting; band sums recover real-space dots."""
    a = np.asarray(a).reshape(n, n, n)
    b = np.asarray(b).reshape(n, n, n)
    fa = np.fft.rfftn(a, norm='ortho')
    fb = np.fft.rfftn(b, norm='ortho')
    kx = 2*np.pi*np.fft.fftfreq(n, d=box/n)
    kz = 2*np.pi*np.fft.rfftfreq(n, d=box/n)
    kmag = np.sqrt(kx[:, None, None]**2 + kx[None, :, None]**2 + kz[None, None, :]**2)
    weight = np.ones(len(kz))
    weight[1:-1 if n % 2 == 0 else None] = 2.
    weights = weight[None, None, :]
    results = []
    energy_a = energy_b = cross = 0.
    for low, high in BANDS:
        mask = (kmag >= low) & (kmag < high)
        aa = float(np.sum((fa.real**2+fa.imag**2)*weights*mask))
        bb = float(np.sum((fb.real**2+fb.imag**2)*weights*mask))
        ab = float(np.sum((fa.real*fb.real+fa.imag*fb.imag)*weights*mask))
        energy_a += aa
        energy_b += bb
        cross += ab
        results.append(dict(k_min_h_Mpc=low,
            k_max_h_Mpc=high if np.isfinite(high) else None,
            count_norm=np.sqrt(aa), mark_norm=np.sqrt(bb),
            count_mark_dot=ab, cosine=ab/np.sqrt(aa*bb) if aa > 0 and bb > 0 else None))
    for observed, direct in ((energy_a, np.vdot(a, a)),
                             (energy_b, np.vdot(b, b)), (cross, np.vdot(a, b))):
        if not np.isclose(observed, direct, rtol=2e-10, atol=1e-8):
            raise ArithmeticError('real-FFT band/Parseval accounting failed')
    return results


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU allocation required')
    if OUT.exists():
        raise FileExistsError(OUT)
    # Small exact test before reading the survey arrays.
    toy = np.arange(64., dtype=float).reshape(4, 4, 4)
    json.dumps(band_alignment(toy, np.flip(toy, axis=0), n=4, box=12.), allow_nan=False)
    started = time.monotonic()
    geometry_path = BASE/'r2_hierarchical_field_geometry_v1/geometry_q257.npz'
    count_path = BASE/'r2_inclusive_count_diagnostic_v1/inclusive_counts_3_sparse.npz'
    point_path = BASE/'r2_point_mark_manifest_v1/points.npz'
    selection_path = BASE/'r2_common_selection_128_v1/selection_3.h5'
    with np.load(geometry_path, allow_pickle=False) as f:
        geometry = {k:jnp.asarray(f[k]) for k in f.files
                    if k not in ('method_names', 'group_labels')}
    with np.load(count_path, allow_pickle=False) as f:
        keys = np.asarray(f['parent_keys'], dtype=np.int32)
        counts = np.asarray(f['parent_counts'], dtype=np.int32)
    with np.load(point_path, allow_pickle=False) as f:
        point_keys = f['population'].astype(np.int64)*N**3+f['flat_cell']
    unique, projected = np.unique(point_keys, return_counts=True)
    np.testing.assert_array_equal(unique, keys)
    np.testing.assert_array_equal(projected, counts)
    if len(point_keys) != 57238 or len(keys) != 45776:
        raise ValueError('inclusive count ownership changed')
    with h5py.File(selection_path) as f:
        if f.attrs['status'] != 'INTEGRATION_COMPLETE_NOT_CALIBRATED':
            raise ValueError('selection status changed')
        exposure = np.empty((6, N, N, N), dtype=np.float64)
        for lo in range(0, N, 4):
            exposure[:, lo:lo+4] = f['selection_shells'][:, :, lo:lo+4].sum(axis=1, dtype=np.float64)
    if not np.isfinite(exposure).all() or np.any(exposure.reshape(-1)[keys] <= 0):
        raise ValueError('observed count has no integrated-selection support')
    cfg = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())
    common, published = cfg['common_cosmology'], cfg['published_prior']
    nbar = np.asarray(published['original_mean_count_per_cell_bright_first'])*(
        (BOX/N)/published['original_cell_cMpc_h'])**3
    bias = jnp.asarray(published['linear_regime_bias_bright_first'])
    rate_cfg = json.loads((ROOT/'config/cf4_r2_rate_nuisance_v1.json').read_text())
    shape = jnp.full(6, rate_cfg['rate_shape'])
    frozen = json.loads((ROOT/'config/cf4_datum_bearing_z0_phasec_program_v1.json').read_text())['inference_model']
    fog = jnp.asarray(frozen['FoG_prior_median_km_s'])
    redshift = jnp.asarray(frozen['fixed_redshift_error_km_s'])
    sd = jnp.asarray([.004, 1., 1., 1., 1., 2.])
    base = json.loads((ROOT/'config/cf4_r1_particle_entry_v1.json').read_text())
    settings = {k:base[k] for k in ('cosmology', 'a_start', 'a_stop', 'a_nbody_maxstep')}
    settings.update(n=N, box_cMpc_h=BOX)
    if settings['cosmology']['h'] != common['h'] or settings['cosmology']['Om'] != common['Omega_m']:
        raise ValueError('PM/observation cosmology mismatch')
    size, nh, ng = N**3, len(sd)+2, len(geometry['train_group_index'])
    e, k, c = jnp.asarray(exposure), jnp.asarray(keys), jnp.asarray(counts)
    rate_mean = jnp.asarray(nbar)
    report = dict(classification='R2_FIXED_STATE_COUNT_CF4_FIELD_TENSION',
        status='STARTED', job_id=os.environ['SLURM_JOB_ID'],
        parent_jobs=['406356', '406377'], n=N, box_cMpc_h=BOX,
        observed_2mpp_points=len(point_keys), occupied_population_cells=len(keys),
        state_labels=list(LABELS), bias_scales=[.8, 1., 1.2], fog_scales=[.5, 1., 2.],
        sensitivity_scales_are_not_physical_prior_intervals=True,
        k_bands_h_Mpc=[[a, b if np.isfinite(b) else None] for a, b in BANDS],
        source_paths=[str(x) for x in (geometry_path, count_path, point_path, selection_path)],
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT/'src/cf4_r2_coarsened_live_joint.py',
                ROOT/'src/cf4_r2_native_to_count_cells.py')},
        results=[], calibrated_tracer=False, calibrated_group_inclusion=False,
        posterior_sample=False, information_gain_measurement=False,
        MW_M31_M33_identified=False)
    OUT.mkdir(parents=True)
    def write():
        report['runtime_seconds'] = time.monotonic()-started
        report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (OUT/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write()
    try:
        evolve, _initial, conf, _cosmo, particle_mass = make_dynamics(settings)
        mass = jnp.full((size,), particle_mass)
        def parts(ic, hyper, group, trial_bias, trial_fog):
            pos, vel = evolve(ic)
            field = particle_grid(pos, vel, mass, conf)
            return count_and_mark_parts(field['rho'],
                jnp.moveaxis(field['mean_velocity_km_s'], -1, 0),
                ic, hyper, group, geometry, sd, e, k, c, rate_mean, shape,
                trial_bias, trial_fog, redshift, box=BOX,
                hubble=common['H0_km_s_Mpc'], h=common['h'])
        def target_with_aux(ic, hyper, group, trial_bias, trial_fog):
            factors, unit = parts(ic, hyper, group, trial_bias, trial_fog)
            occupied = jnp.take(unit.reshape(-1), k)
            return factors.sum(), (factors, jnp.min(occupied))
        value_grad = jax.jit(jax.value_and_grad(target_with_aux, argnums=0, has_aux=True))
        mark_grad = jax.jit(jax.grad(lambda ic, h, u, b, f: parts(ic, h, u, b, f)[0][1], argnums=0))
        count_score = jax.jit(lambda ic, h, u, b, f: parts(ic, h, u, b, f)[0][0])
        for label in LABELS:
            if time.monotonic()-started > 950:
                raise TimeoutError('bounded application time exhausted')
            state_path = STATES/f'{label}.npz'
            with np.load(state_path, allow_pickle=False) as f:
                ic = jnp.asarray(f['white_ic'])
                hyper = jnp.asarray(f['white_hyper'])
                group = jnp.asarray(f['white_training_group'])
            if ic.shape != (size,) or hyper.shape != (nh,) or group.shape != (ng,):
                raise ValueError(f'invalid saved-state coordinate shapes: {label}')
            (value, (factors, min_occupied)), full_grad = value_grad(ic, hyper, group, bias, fog)
            full_grad.block_until_ready()
            mg = mark_grad(ic, hyper, group, bias, fog)
            mg.block_until_ready()
            fg = np.asarray(full_grad)
            mg_np = np.asarray(mg)
            cg = fg-mg_np+np.asarray(ic)  # full=count+mark-.5*IC^2
            factor_values = np.asarray(factors)
            if not all(np.isfinite(x).all() for x in (fg, mg_np, cg, factor_values)):
                raise FloatingPointError(f'nonfinite factors/IC gradients: {label}')
            if float(min_occupied) <= 0:
                raise ValueError(f'occupied intensity support failure: {label}')
            scores = {}
            for name, trial_bias, trial_fog in (
                ('bias_x0p8', bias*.8, fog), ('bias_x1p2', bias*1.2, fog),
                ('fog_x0p5', bias, fog*.5), ('fog_x2', bias, fog*2.)):
                score = float(count_score(ic, hyper, group, trial_bias, trial_fog))
                if not np.isfinite(score):
                    raise FloatingPointError(f'nonfinite sensitivity count score: {label}/{name}')
                scores[name] = dict(logfactor=score, difference_from_baseline=score-float(factor_values[0]))
            row = dict(label=label, saved_state_path=str(state_path),
                factor_values=dict(count=float(factor_values[0]), mark=float(factor_values[1]),
                    white_prior=float(factor_values[2]), total=float(value)),
                occupied_min_unit_intensity=float(min_occupied),
                IC_gradient_alignment=gradient_alignment(cg, mg_np),
                IC_gradient_bands=band_alignment(cg, mg_np),
                fixed_state_count_response_sensitivity=scores)
            report['results'].append(row)
            write()
            print(f'{label}: count={row["factor_values"]["count"]:.3f}, '
                f'cosine={row["IC_gradient_alignment"]["cosine"]:.4g}', flush=True)
        report['status'] = 'COMPLETE_FIXED_STATE_MODEL_STRESS_NOT_POSTERIOR'
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write()
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
