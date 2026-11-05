"""One saved-N128 source-marked count support/cost screen, not a fit."""

import json
import os
from pathlib import Path
import resource
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cf4_r2_marked_tracer_jax import predict_source_marked_intensity
from cf4_r2_native_to_count_cells import native_mass_momentum_to_count_cells
from cf4_r2_observed_magnitude_transfer import _schechter_interval_probability

BASE = Path('/gpfs/kjhan/CF4/z0_density')
OUT = BASE/'r2_marked_source_n128_v2'
GEOMETRY = BASE/'r2_marked_source_geometry_v1'
STATE = BASE/'r2_pm128_unconditional_v1'
COUNTS = BASE/'r2_inclusive_count_diagnostic_v1/inclusive_counts_3_sparse.npz'
N, BOX = 128, 384.


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU allocation required')
    if OUT.exists():
        raise FileExistsError(OUT)
    start = time.monotonic()
    cfg = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())
    cosmology = cfg['common_cosmology']
    state_record = json.loads((STATE/'result.json').read_text())
    for pm_key, common_key in (('h', 'h'), ('Om', 'Omega_m'), ('Ob', 'Omega_b'),
                               ('A_s_1e9', 'A_s_1e9'), ('ns', 'ns')):
        if state_record['cosmology'][pm_key] != cosmology[common_key]:
            raise ValueError(f'PM cosmology mismatch: {pm_key}')
    with np.load(STATE/'state.npz', allow_pickle=False) as f:
        rho = np.asarray(f['rho'], dtype=np.float64)
        velocity = np.asarray(f['velocity_km_s'], dtype=np.float64)
    if rho.shape != (N,)*3 or velocity.shape != (3,)+(N,)*3:
        raise ValueError('saved PM state geometry mismatch')
    count_rho, count_velocity = native_mass_momentum_to_count_cells(
        jnp.asarray(rho), jnp.asarray(velocity), BOX)
    count_rho, count_velocity = np.asarray(count_rho), np.asarray(count_velocity)
    if not np.isclose(count_rho.sum(), rho.sum(), rtol=2e-12, atol=1e-8):
        raise ValueError('native-to-count mass readout is not conservative')
    with np.load(COUNTS, allow_pickle=False) as f:
        keys = np.asarray(f['parent_keys'], dtype=np.int64)
        counts = np.asarray(f['parent_counts'], dtype=np.int64)
    if len(keys) != 45776 or counts.sum() != 57238:
        raise ValueError('inclusive observed count contract changed')
    with np.load(GEOMETRY/'geometry.npz', allow_pickle=False) as f:
        positions = np.asarray(f['positions'], dtype=np.float64)
        angular = np.asarray(f['angular'], dtype=np.float64)
        radial_table = np.asarray(f['radial_table'], dtype=np.float64)
        z_table = np.asarray(f['redshift_table'], dtype=np.float64)
        modulus_table = np.asarray(f['modulus_table'], dtype=np.float64)
    if (positions.shape != (N**3, 3) or angular.shape != (2, N**3)
            or not np.isfinite(angular).all() or np.any((angular < 0) | (angular > 1))):
        raise ValueError('fixed geometry or angular map invalid')
    observer = np.full(3, BOX/2.)
    edges = (-np.inf, -25., -23.6666666666667,
             -22.3333333333333, -21., np.inf)
    lf_bin_prob = np.array([_schechter_interval_probability(
        edges[j], edges[j+1], mstar=-23.28, alpha=-.94)
        for j in range(5)])
    if abs(lf_bin_prob.sum()-1.) > 1e-12:
        raise ValueError('intrinsic LF bins do not partition total measure')
    # A unit total intrinsic count per source cell, with b=1, is deliberately
    # a cost/support reference. It is NOT the old six-population fitted nbar.
    response = np.maximum(count_rho.reshape(-1), 0.)
    response /= response.mean()
    intrinsic = lf_bin_prob[:, None]*response[None, :]
    report = dict(classification='R2_SOURCE_MARKED_N128_FORWARD_SUPPORT_ONLY',
                  job_id=os.environ['SLURM_JOB_ID'], status='STARTED',
                  saved_state=str(STATE/'state.npz'), observed_counts=str(COUNTS),
                  observed_points=int(counts.sum()), occupied_population_cells=len(keys),
                  intrinsic_rate='unit total expected count per N128 source cell, not fitted',
                  intrinsic_bias='fixed b=1 on unconditional PM density, not calibrated',
                  intrinsic_LF='Schechter alpha=-0.94 Mstar=-23.28 within-bin',
                  PM_readout='conservative native-node mass/momentum to count-voxel centres',
                  angular_geometry='eight subcell sightlines per source voxel',
                  fixed_geometry=str(GEOMETRY/'geometry.npz'),
                  source_selection='corrected apparent/observed absolute K and 5-180 cMpc/h after RSD',
                  group_selection_or_CF4_overlap_calibrated=False,
                  actual_CF4_conditioned=False, posterior=False, N256=False,
                  MW_M31_M33_role_inference=False)
    OUT.mkdir(parents=True)
    def write():
        report['elapsed_seconds'] = time.monotonic()-start
        report['host_peak_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2
        (OUT/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write()
    try:
        t = time.monotonic()
        intensity = np.asarray(jax.jit(predict_source_marked_intensity,
            static_argnames=('grid_size', 'quadrature_order',
                             'radial_min_cMpc_h', 'radial_max_cMpc_h'))(
            jnp.asarray(positions),
            jnp.asarray(np.moveaxis(count_velocity, 0, -1).reshape(-1, 3)),
            jnp.asarray(intrinsic), jnp.asarray(angular),
            observer=jnp.asarray(observer), box_size_cMpc_h=BOX,
            hubble_km_s_Mpc=cosmology['H0_km_s_Mpc'],
            little_h=cosmology['h'],
            radius_table_cMpc_h=jnp.asarray(radial_table),
            modulus_table_h=jnp.asarray(modulus_table),
            redshift_table=jnp.asarray(z_table), grid_size=N,
            sigma_los_km_s=100., radial_min_cMpc_h=5.,
            radial_max_cMpc_h=180., quadrature_order=3))
        report['compiled_forward_seconds'] = time.monotonic()-t
        occupied = intensity.reshape(-1)[keys]
        if not np.isfinite(intensity).all() or np.any(intensity < -1e-12):
            raise FloatingPointError('nonfinite/negative marked count intensity')
        report['zero_occupied_intensity_cells'] = int(np.count_nonzero(occupied <= 0))
        report['minimum_occupied_intensity'] = float(occupied.min())
        report['predicted_population_totals_arbitrary_unit_rate'] = intensity.sum(
            axis=(1, 2, 3)).tolist()
        report['status'] = ('FORWARD_SUPPORT_PASS_NOT_CALIBRATED'
                            if not report['zero_occupied_intensity_cells']
                            else 'OBSERVED_SUPPORT_FAIL_NOT_CALIBRATED')
    except Exception as exc:
        report['status'] = 'FAILED'
        report['error'] = f'{type(exc).__name__}: {exc}'
        write()
        raise
    write()
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
