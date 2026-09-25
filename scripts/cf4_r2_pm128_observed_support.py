"""Connect native N128 counts/selection to one *unconditional* N128 PM state.

This is an observation-operator support check, not a fitted field or nuisance
calibration. Compare f=0 to the former arbitrary f=.03 without tuning either.
"""

import json
import os
from pathlib import Path
import sys
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cf4_r2_continuous_tracer import predict_continuous_intensity

DATA = Path('/gpfs/kjhan/CF4/z0_density/r2_native_128_observations_v1')
STATE = Path('/gpfs/kjhan/CF4/z0_density/r2_pm128_unconditional_v1')
OUTPUT = Path('/gpfs/kjhan/CF4/z0_density/r2_pm128_observed_support_v1')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with np.load(STATE / 'state.npz', allow_pickle=False) as data:
        rho, velocity = data['rho'].astype(np.float64), data['velocity_km_s'].astype(np.float64)
    with np.load(DATA / 'counts_3_sparse.npz', allow_pickle=False) as data:
        keys, counts = data['all_keys'].copy(), data['all_counts'].copy()
    with np.load('/gpfs/kjhan/CF4/z0_density/actual_data_corrected_v2_run1/corrected_counts.npz',
                 allow_pickle=False) as data:
        yes, no = data['survival_yes'], data['survival_no']
        survival = (yes + 1) / (yes + no + 2)
    with h5py.File(DATA / 'selection_3.h5', 'r') as data:
        shells = data['selection_shells'][:]
    exposure = np.einsum('ps,psijk->pijk', survival, shells, optimize=True)
    n = 128
    pop, cell = np.divmod(keys, n**3)
    x, y, z = np.unravel_index(cell, (n,)*3)
    if np.any(exposure[pop, x, y, z] <= 0) or np.any(counts <= 0):
        raise RuntimeError('observed-support input invalid')
    cfg = json.loads((ROOT / 'config/cf4_datum_bearing_z0_phasec_program_v1.json').read_text())
    bias = jnp.asarray(cfg['external_population_prior']['published_bias'])
    settings = cfg['inference_model']
    args = dict(density=jnp.asarray(rho), velocity=jnp.asarray(velocity),
                exposure=jnp.asarray(exposure), nbar=jnp.ones(6), bias=bias,
                box=384., observer=jnp.asarray([192., 192., 192.]),
                hubble=74.6, little_h=.746,
                sigma_fog=jnp.asarray(settings['FoG_prior_median_km_s']),
                sigma_redshift=jnp.asarray(settings['fixed_redshift_error_km_s']))
    runs = {}
    for fraction in (0., .03):
        tick = time.monotonic()
        intensity = np.asarray(predict_continuous_intensity(
            **args, diffuse_fraction=jnp.full((6,), fraction)))
        selected = intensity[pop, x, y, z]
        runs[str(fraction)] = dict(seconds=time.monotonic()-tick,
                                   occupied_zero_intensity_cells=int(np.count_nonzero(selected <= 0)),
                                   occupied_nonfinite_intensity_cells=int(np.count_nonzero(~np.isfinite(selected))),
                                   minimum_occupied_intensity=float(selected.min()),
                                   mean_occupied_intensity=float(selected.mean()),
                                   total_model_intensity_per_population=intensity.sum(axis=(1, 2, 3)).tolist())
    report = dict(classification='ACTUAL_N128_SELECTION_ON_UNCONDITIONAL_PM_SUPPORT_CHECK',
                  source_commit=os.environ['EXPECTED_COMMIT'],
                  observed_total=int(counts.sum()), occupied_cells=int(len(keys)),
                  survival_calibration_marks=int((yes + no).sum()),
                  survival_beta_mean_range=[float(survival.min()), float(survival.max())],
                  PM_zero_density_cells=int(np.count_nonzero(rho <= 0)),
                  trials=runs, model_rate_per_cell_placeholder=1.0,
                  actual_CF4_conditioned=False, sky_phase_unconstrained=True,
                  diffuse_fraction_calibrated=False, galaxy_bias_calibrated=False,
                  posterior_or_R2_delivery=False)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
