"""Short actual-data live-IC/galaxy-response development sampler."""
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import h5py
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_chunked_hmc import make_chunks
from cf4_r1_particle_forward import make_dynamics, particle_grid
from cf4_r2_coarsened_live_joint import count_and_mark_parts

BASE = Path('/gpfs/kjhan/CF4/z0_density')
OUT = BASE/'r2_joint_nuisance_pilot_v2'
N, BOX = 128, 384.
SEEDS = (2026092801, 2026092807)
WARMUP = RETAINED = 64


def checked_development_record(record, state):
    """Keep finite accepted states; count nonfinite *rejected* HMC proposals."""
    arrays = tuple(np.asarray(a) for a in record)
    for index in (0, 1, 2, 5, 6, 7):
        if not np.isfinite(arrays[index]).all():
            raise FloatingPointError(f'nonfinite HMC accepted/step record field {index}')
    if not np.isfinite(np.asarray(state.logdensity_grad)).all():
        raise FloatingPointError('nonfinite accepted HMC state gradient')
    rejected_bad_energy = ~np.isfinite(arrays[4])
    if np.any(rejected_bad_energy & ~arrays[3]):
        raise FloatingPointError('nonfinite proposal energy without divergence rejection')
    return arrays, int(np.count_nonzero(rejected_bad_energy))


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU allocation required')
    if OUT.exists():
        raise FileExistsError(OUT)
    # Exact recovery regression: rejected bad proposal energy is observable,
    # while the accepted state/gradient must still be finite.
    toy_record = (np.zeros((2, 2)), np.zeros(2), np.ones(2),
        np.array([False, True]), np.array([0., np.inf]), np.ones(2),
        np.ones(2), np.ones(2))
    _, toy_rejected = checked_development_record(toy_record,
        SimpleNamespace(logdensity_grad=np.zeros(2)))
    if toy_rejected != 1:
        raise AssertionError('rejected divergent proposal accounting failed')
    started = time.monotonic()
    geometry_path = BASE/'r2_hierarchical_field_geometry_v1/geometry_q257.npz'
    count_path = BASE/'r2_inclusive_count_diagnostic_v1/inclusive_counts_3_sparse.npz'
    point_path = BASE/'r2_point_mark_manifest_v1/points.npz'
    selection_path = BASE/'r2_common_selection_128_v1/selection_3.h5'
    with np.load(geometry_path, allow_pickle=False) as f:
        g = {k:jnp.asarray(f[k]) for k in f.files if k not in ('method_names', 'group_labels')}
    with np.load(count_path, allow_pickle=False) as f:
        keys = np.asarray(f['parent_keys'], dtype=np.int32)
        counts = np.asarray(f['parent_counts'], dtype=np.int32)
    with np.load(point_path, allow_pickle=False) as f:
        point_keys = f['population'].astype(np.int64)*N**3+f['flat_cell']
    unique, projected = np.unique(point_keys, return_counts=True)
    np.testing.assert_array_equal(unique, keys)
    np.testing.assert_array_equal(projected, counts)
    if len(point_keys) != 57238 or len(keys) != 45776:
        raise ValueError('inclusive count source ownership changed')
    with h5py.File(selection_path) as f:
        if f.attrs['status'] != 'INTEGRATION_COMPLETE_NOT_CALIBRATED':
            raise ValueError('integrated selection status changed')
        exposure = np.empty((6, N, N, N), dtype=np.float64)
        for lo in range(0, N, 4):
            exposure[:, lo:lo+4] = f['selection_shells'][:, :, lo:lo+4].sum(axis=1, dtype=np.float64)
    if not np.isfinite(exposure).all() or np.any(exposure.reshape(-1)[keys] <= 0):
        raise ValueError('observed count selection support failure')
    cfg = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())
    common, published = cfg['common_cosmology'], cfg['published_prior']
    nbar = np.asarray(published['original_mean_count_per_cell_bright_first'])*(
        (BOX/N)/published['original_cell_cMpc_h'])**3
    bias0 = jnp.asarray(published['linear_regime_bias_bright_first'])
    rate_cfg = json.loads((ROOT/'config/cf4_r2_rate_nuisance_v1.json').read_text())
    shape = jnp.full(6, rate_cfg['rate_shape'])
    frozen = json.loads((ROOT/'config/cf4_datum_bearing_z0_phasec_program_v1.json').read_text())['inference_model']
    fog0 = jnp.asarray(frozen['FoG_prior_median_km_s'])
    redshift = jnp.asarray(frozen['fixed_redshift_error_km_s'])
    sd = jnp.asarray([.004, 1., 1., 1., 1., 2.])
    base = json.loads((ROOT/'config/cf4_r1_particle_entry_v1.json').read_text())
    settings = {k:base[k] for k in ('cosmology', 'a_start', 'a_stop', 'a_nbody_maxstep')}
    settings.update(n=N, box_cMpc_h=BOX)
    if settings['cosmology']['h'] != common['h'] or settings['cosmology']['Om'] != common['Omega_m']:
        raise ValueError('PM/observation cosmology mismatch')
    size, nh, ng = N**3, len(sd)+2, len(g['train_group_index'])
    dim = size+nh+ng+7
    e, k, c = jnp.asarray(exposure), jnp.asarray(keys), jnp.asarray(counts)
    rate_mean = jnp.asarray(nbar)
    hmc = dict(initial_step_size=.005, maximum_step_size=.05,
        target_acceptance=.8, divergence_threshold=1000.,
        integration_steps=6, integration_steps_range=[4, 8])
    report = dict(classification='R2_JOINT_IC_COUNT_CF4_NUISANCE_DEVELOPMENT_PILOT',
        status='STARTED', job_id=os.environ['SLURM_JOB_ID'], N=N, box_cMpc_h=BOX,
        count_points=len(point_keys), count_occupied_cells=len(keys),
        training_source_groups=ng, source_FP_rows=len(g['row_group']),
        new_bias_prior_log_sd=.5, new_shared_fog_prior_log_sd=.7,
        count_rate_prior_shape=float(shape[0]), calibration_claim=False,
        within_cell_and_group_inclusion_calibrated=False, posterior_converged=False,
        R2_delivery=False, MW_M31_M33_identified=False, seeds=list(SEEDS),
        warmup_per_chain=WARMUP, retained_per_chain=RETAINED,
        chain_map_samples_are_autocorrelated=True, sampler=hmc, chains=[],
        source_paths=[str(p) for p in (geometry_path, count_path, point_path, selection_path)],
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT/'src/cf4_r2_continuous_tracer.py',
                ROOT/'src/cf4_r2_coarsened_live_joint.py',
                ROOT/'src/cf4_r2_native_to_count_cells.py',
                ROOT/'src/cf4_chunked_hmc.py')})
    OUT.mkdir(parents=True)
    traces = {}
    def write():
        report['runtime_seconds'] = time.monotonic()-started
        report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (OUT/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        if traces:
            np.savez_compressed(OUT/'transition_summaries.npz', **traces)
    def deadline():
        if time.monotonic()-started > 6000:
            raise TimeoutError('bounded 100-minute application budget exhausted')
    write()
    try:
        evolve, _initial, conf, _cosmo, particle_mass = make_dynamics(settings)
        mass = jnp.full((size,), particle_mass)
        def split(x):
            return x[:size], x[size:size+nh], x[size+nh:size+nh+ng], x[size+nh+ng:]
        def field_from_ic(ic):
            pos, vel = evolve(ic)
            return particle_grid(pos, vel, mass, conf)
        def factors_and_support(x):
            ic, hyper, group, tracer = split(x)
            field = field_from_ic(ic)
            bias = bias0*jnp.exp(.5*tracer[:6])
            fog = fog0*jnp.exp(.7*tracer[6])
            factors, unit = count_and_mark_parts(field['rho'],
                jnp.moveaxis(field['mean_velocity_km_s'], -1, 0), ic,
                hyper, group, g, sd, e, k, c, rate_mean, shape, bias,
                fog, redshift, box=BOX, hubble=common['H0_km_s_Mpc'],
                h=common['h'])
            factors = factors.at[2].add(-.5*jnp.vdot(tracer, tracer))
            return factors, jnp.min(jnp.take(unit.reshape(-1), k))
        target = lambda x: factors_and_support(x)[0].sum()
        factor_fn = jax.jit(factors_and_support)
        field_fn = jax.jit(field_from_ic)
        def starting_vector(seed, chain):
            ic = np.random.default_rng(seed).standard_normal(size)
            group = np.random.default_rng(2026092820+chain).standard_normal(ng)
            return jnp.asarray(np.r_[ic, np.zeros(nh), group, np.zeros(7)])
        initialize, warm, sample, final = make_chunks(target, dim, hmc, record_steps=True)
        for chain, seed in enumerate(SEEDS):
            deadline()
            vector = starting_vector(seed, chain)
            initial_parts, support = factor_fn(vector)
            if not np.isfinite(np.asarray(initial_parts)).all() or float(support) <= 0:
                raise FloatingPointError(f'initial chain{chain} target/support invalid')
            state, adaptation = initialize(vector)
            key = jax.random.fold_in(jax.random.PRNGKey(2026092828), chain)
            kw, ks = jax.random.split(key)
            keysets = [jax.random.split(kw, WARMUP), jax.random.split(ks, RETAINED)]
            row = dict(chain=chain, initial_seed=seed,
                initial_factors=np.asarray(initial_parts).tolist(),
                initial_min_occupied_unit_intensity=float(support),
                warmup_completed=0, retained_completed=0,
                map_samples=0, rejected_nonfinite_proposal_energies=0)
            report['chains'].append(row)
            buffers = {name:[] for name in ('summary', 'acceptance', 'divergent', 'step', 'leapfrog_steps', 'phase')}
            mean = m2 = None
            for phase in range(2):
                if phase:
                    step = final(adaptation)
                    row['sampling_step'] = float(step)
                for offset in range(0, 64, 4):
                    deadline()
                    keys_chunk = keysets[phase][offset:offset+4]
                    if phase:
                        state, record = sample(state, step, keys_chunk)
                    else:
                        (state, adaptation), record = warm(state, adaptation, keys_chunk)
                    try:
                        a, bad_energy = checked_development_record(record, state)
                    except FloatingPointError:
                        raw = tuple(np.asarray(item) for item in record)
                        row['failed_chunk'] = dict(phase=phase, offset=offset,
                            nonfinite_by_record_field=[int(np.count_nonzero(~np.isfinite(item)))
                                for item in raw], divergent=raw[3].tolist(),
                            accepted_state_gradient_finite=bool(
                                np.isfinite(np.asarray(state.logdensity_grad)).all()))
                        write()
                        raise
                    row['rejected_nonfinite_proposal_energies'] += bad_energy
                    x = a[0]
                    tracer = x[:, size+nh+ng:]
                    summary = np.column_stack((a[1], np.mean(x[:, :size]**2, axis=1),
                        x[:, :4], x[:, size:size+nh],
                        np.mean(x[:, size+nh:size+nh+ng], axis=1), tracer))
                    for name, values in zip(buffers, (summary, a[2], a[3], a[6], a[7],
                        np.full(4, phase, dtype=np.int8))):
                        buffers[name].append(values)
                    for name, values in buffers.items():
                        traces[f'chain{chain}_{name}'] = np.concatenate(values)
                    row['retained_completed' if phase else 'warmup_completed'] = offset+4
                    if phase:
                        for position in x:
                            field = field_fn(jnp.asarray(position[:size]))
                            rho = np.asarray(field['rho'], dtype=np.float64)
                            velocity = np.moveaxis(np.asarray(field['mean_velocity_km_s'],
                                dtype=np.float64), -1, 0)
                            if not np.isfinite(rho).all() or not np.isfinite(velocity).all():
                                raise FloatingPointError('nonfinite retained field')
                            if abs(float(rho.mean())-1.) > 1e-7:
                                raise FloatingPointError('retained mass conservation failure')
                            mapped = np.concatenate((rho[None], velocity), axis=0)
                            row['map_samples'] += 1
                            if mean is None:
                                mean = mapped.copy()
                                m2 = np.zeros_like(mapped)
                            else:
                                delta = mapped-mean
                                mean += delta/row['map_samples']
                                m2 += delta*(mapped-mean)
                    write()
                    print(f'chain{chain} phase{phase} {offset+4}/64 elapsed={time.monotonic()-started:.1f}s',
                          flush=True)
            if row['map_samples'] != RETAINED:
                raise ValueError('retained map-sample count incomplete')
            endpoint_parts, endpoint_support = factor_fn(state.position)
            if not np.isfinite(np.asarray(endpoint_parts)).all() or float(endpoint_support) <= 0:
                raise FloatingPointError(f'endpoint chain{chain} target/support invalid')
            retained = traces[f'chain{chain}_phase'] == 1
            tracer_trace = traces[f'chain{chain}_summary'][retained, -7:]
            row.update(retained_acceptance_mean=float(traces[f'chain{chain}_acceptance'][retained].mean()),
                retained_divergences=int(traces[f'chain{chain}_divergent'][retained].sum()),
                endpoint_factors=np.asarray(endpoint_parts).tolist(),
                endpoint_min_occupied_unit_intensity=float(endpoint_support),
                tracer_white_mean=tracer_trace.mean(axis=0).tolist(),
                tracer_white_sd=tracer_trace.std(axis=0, ddof=1).tolist())
            density_sd = np.sqrt(m2/(row['map_samples']-1))
            np.savez_compressed(OUT/f'chain{chain}_development_field_moments.npz',
                mean=mean.astype(np.float32), sample_sd=density_sd.astype(np.float32),
                channels=np.array(['rho', 'vx_km_s', 'vy_km_s', 'vz_km_s']),
                sample_count=row['map_samples'], box_cMpc_h=BOX, dx_cMpc_h=BOX/N)
            np.savez_compressed(OUT/f'chain{chain}_terminal_white_state.npz',
                white_ic=np.asarray(state.position[:size]),
                white_hyper=np.asarray(state.position[size:size+nh]),
                white_training_group=np.asarray(state.position[size+nh:size+nh+ng]),
                white_tracer=np.asarray(state.position[size+nh+ng:]))
            write()
        report['status'] = 'COMPLETE_SHORT_PARTIAL_MODEL_CHAINS_NOT_CONVERGED'
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write()
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
