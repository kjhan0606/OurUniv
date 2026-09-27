"""Training-count quadrature sensitivity on one archived N128 state.

This is a numerical integration diagnostic only. It does not read FP marks,
fit a field, or claim a CF4-conditioned posterior.
"""

import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/gpfs/kjhan/CF4/z0_density')
OUT = BASE/'r2_count_quadrature_v3'
SPLIT = BASE/'r2_sky_closed_split_v5/split.npz'
SOURCE = BASE/'r2_marked_source_geometry_v1/geometry.npz'
STATE = BASE/'r2_pm128_unconditional_v1/state.npz'
COSMO = ROOT/'config/cf4_r2_common_cosmology_v1.json'
N, BOX = 128, 384.
ORDERS = (3, 9, 15)


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    expected_commit = os.environ.get('CF4_EXPECTED_COMMIT')
    if not expected_commit:
        raise RuntimeError('CF4_EXPECTED_COMMIT must pin the submitted source')
    source_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if source_commit != expected_commit:
        raise RuntimeError(f'source commit mismatch: {source_commit} != {expected_commit}')
    dirty = subprocess.run(
        ['git', 'diff', '--quiet', expected_commit, '--',
         'scripts/cf4_r2_count_quadrature.py',
         'scripts/run_cf4_r2_count_quadrature.sbatch'], cwd=ROOT)
    if dirty.returncode:
        raise RuntimeError('submitted diagnostic sources changed after commit')
    if OUT.exists():
        raise FileExistsError(OUT)
    started = time.monotonic()

    from cf4_r2_marked_tracer_jax import (
        intrinsic_biased_source_masses, intrinsic_lf_bin_fractions,
        predict_source_marked_intensity_los_node,
        sparse_marked_poisson_log_likelihood,
    )
    from cf4_2mpp_joint_likelihood_jax import _gaussian_hermite_rule
    from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells

    with np.load(SPLIT, allow_pickle=False) as f:
        train_keys = f['train_keys'].copy()
        train_counts = f['train_counts'].copy()
        heldout_voxels = f['heldout_flat_voxels'].copy()
    with np.load(STATE, allow_pickle=False) as f:
        rho = jnp.asarray(f['rho'], dtype=jnp.float64)
        velocity = jnp.asarray(f['velocity_km_s'], dtype=jnp.float64)
    with np.load(SOURCE, allow_pickle=False) as f:
        source = {k: jnp.asarray(f[k]) for k in f.files}

    if (train_keys.ndim != 1 or train_counts.shape != train_keys.shape
            or int(train_counts.sum()) != 47542
            or len(heldout_voxels) != N**3//8
            or len(np.unique(train_keys)) != len(train_keys)
            or np.any(train_keys < 0) or np.any(train_keys >= 6*N**3)):
        raise ValueError('frozen v5 training-count geometry changed')
    mask = np.ones(N**3, dtype=bool)
    mask[heldout_voxels] = False
    if np.any(mask[train_keys % N**3] == 0):
        raise ValueError('training count key lies in held-out sky')

    rho_cell, velocity_cell = native_mass_momentum_to_count_cells(rho, velocity, BOX)
    fraction = jnp.sum(intrinsic_lf_bin_fractions()[1:4])
    intrinsic = intrinsic_biased_source_masses(
        rho_cell, jnp.log(fraction), jnp.ones(5),
        reference_interval=(-25., -21.))
    source_velocity = jnp.moveaxis(velocity_cell, 0, -1).reshape(-1, 3)
    cosmology = json.loads(COSMO.read_text())['common_cosmology']
    args = dict(
        observer=jnp.full(3, BOX/2.), box_size_cMpc_h=BOX,
        hubble_km_s_Mpc=cosmology['H0_km_s_Mpc'], little_h=cosmology['h'],
        radius_table_cMpc_h=source['radial_table'],
        modulus_table_h=source['modulus_table'],
        redshift_table=source['redshift_table'], grid_size=N,
        sigma_los_km_s=100., radial_min_cMpc_h=5., radial_max_cMpc_h=180.)

    paths = (SPLIT, SOURCE, STATE, COSMO)
    report = dict(
        classification='R2_TRAIN_COUNT_QUADRATURE_NUMERICAL_DIAGNOSTIC',
        status='STARTED', job_id=os.environ['SLURM_JOB_ID'],
        source_commit=source_commit,
        N=N, box_cMpc_h=BOX, quadrature_orders=list(ORDERS),
        training_count_points=int(train_counts.sum()),
        training_population_voxel_keys=int(len(train_keys)),
        heldout_count_points_read=0, FP_marks_read=0,
        field_fit=False, sampler=False, R2_posterior=False, N256=False,
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in paths},
        interpretation='One archived unconditional N128 state; higher-order '
                        'GH comparison measures numerical count-integral '
                        'sensitivity, not the continuous individual-redshift '
                        'law or a calibrated observation model.',
        MW_M31_M33='Latent roles on any new field remain unresolved; no truth '
                   'identity used here; their observables must constrain that same field.')
    OUT.mkdir(parents=True)

    def save():
        report['elapsed_seconds'] = time.monotonic()-started
        report['host_peak_GiB'] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss/1024**2
        (OUT/'result.json').write_text(
            json.dumps(report, indent=2, allow_nan=False)+'\n')

    save()
    means_by_order = {}
    try:
        key_array = jnp.asarray(train_keys)
        count_array = jnp.asarray(train_counts)
        train_voxel_mask = jnp.asarray(mask)
        for order in ORDERS:
            t0 = time.monotonic()
            nodes, weights = _gaussian_hermite_rule(order)
            one_node = jax.jit(
                predict_source_marked_intensity_los_node,
                static_argnames=('grid_size', 'radial_min_cMpc_h',
                                 'radial_max_cMpc_h'))
            intensity = jnp.zeros((6, N, N, N), dtype=intrinsic.dtype)
            for node, weight in zip(nodes, weights):
                node_intensity = one_node(
                    source['positions'], source_velocity, intrinsic,
                    source['angular'], node, weight, **args)
                node_intensity.block_until_ready()
                intensity = intensity + node_intensity
            intensity.block_until_ready()
            means = np.asarray(intensity.reshape(-1)[key_array])
            if not np.isfinite(means).all() or np.any(means <= 0):
                raise FloatingPointError(f'nonpositive/nonfinite occupied mean at GH{order}')
            loglike = float(sparse_marked_poisson_log_likelihood(
                intensity, key_array, count_array,
                selected_voxel_mask=train_voxel_mask))
            means_by_order[order] = means
            report.setdefault('orders', {})[str(order)] = dict(
                elapsed_seconds=time.monotonic()-t0,
                train_log_likelihood=loglike,
                min_train_key_mean=float(np.min(means)),
                max_train_key_mean=float(np.max(means)))
            save()

        reference = means_by_order[15]
        comparison = {}
        for order in (3, 9):
            relative = np.abs(means_by_order[order]/reference-1.)
            top_index = np.argsort(relative)[-10:][::-1]
            worst_keys = []
            for index in top_index:
                key = int(train_keys[index])
                mean_order = float(means_by_order[order][index])
                mean_reference = float(reference[index])
                worst_keys.append(dict(
                    key=key, population=key//(N**3),
                    voxel=list(np.unravel_index(key % (N**3), (N,)*3)),
                    observed_count=int(train_counts[index]),
                    mean_order=mean_order, mean_GH15=mean_reference,
                    relative_error=float(relative[index]),
                    observed_count_log_term_delta=float(
                        train_counts[index]*np.log(mean_order/mean_reference))))
            comparison[f'GH{order}_vs_GH15'] = dict(
                relative_error_quantiles=dict(zip(
                    ('p50','p90','p95','p99','max'),
                    np.quantile(relative, [0.50,0.90,0.95,0.99,1.0]).tolist())),
                worst_keys=worst_keys,
                poisson_loglike_delta_vs_GH15=(
                    report['orders'][str(order)]['train_log_likelihood']
                    - report['orders']['15']['train_log_likelihood']))
        report.update(comparison=comparison,
                      status='COMPLETED_TRAIN_ONLY_QUADRATURE_SENSITIVITY_NOT_CALIBRATION')
    except Exception as exc:
        report['status'] = 'FAILED'
        report['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise
    save()
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
