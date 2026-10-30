"""Apply re-based actual 2M++ observation operator to one PM control state.

This is a same-cosmology support/cost check, not a random-phase sky fit.
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
sys.path.insert(0, str(ROOT/'src'))
from cf4_r2_continuous_tracer import predict_continuous_intensity

CATALOGUE = Path('/gpfs/kjhan/CF4/z0_density/r2_common_catalogue_128_v1')
SELECTION = Path('/gpfs/kjhan/CF4/z0_density/r2_common_selection_128_v1')
STATE = Path('/gpfs/kjhan/CF4/z0_density/r2_pm128_unconditional_v1')
OUT = Path('/gpfs/kjhan/CF4/z0_density/r2_common_pm_support_v1')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('submit through Slurm GPU')
    if OUT.exists():
        raise FileExistsError(OUT)
    contract = json.loads((ROOT/'config/cf4_r2_common_cosmology_v1.json').read_text())
    common = contract['common_cosmology']
    state_record = json.loads((STATE/'result.json').read_text())
    selection_record = json.loads((SELECTION/'result.json').read_text())
    catalogue_record = json.loads((CATALOGUE/'result.json').read_text())
    for pm_key, common_key in (('h','h'), ('Om','Omega_m'), ('Ob','Omega_b'),
                               ('A_s_1e9','A_s_1e9'), ('ns','ns')):
        if state_record['cosmology'][pm_key] != common[common_key]:
            raise ValueError(f'PM cosmology mismatch: {pm_key}')
    if selection_record['cosmology']['h'] != common['h'] or selection_record['occupied_zero_selection_keys']:
        raise ValueError('selection mismatch or unsupported observations')
    if catalogue_record['common_cosmology'] != common:
        raise ValueError('catalogue cosmology mismatch')
    with np.load(STATE/'state.npz', allow_pickle=False) as f:
        rho = f['rho']
        velocity = f['velocity_km_s']
    with np.load(CATALOGUE/'counts_3_sparse.npz', allow_pickle=False) as f:
        keys, counts = f['all_keys'], f['all_counts']
    with np.load(CATALOGUE/'survival_marks.npz', allow_pickle=False) as f:
        yes, no = f['yes'], f['no']
    survival = (yes+1)/(yes+no+2)
    with h5py.File(SELECTION/'selection_3.h5', 'r') as f:
        shells = f['selection_shells'][:]
    exposure = np.einsum('ps,psijk->pijk', survival, shells, optimize=True)
    pop, cell = np.divmod(keys, 128**3)
    x,y,z = np.unravel_index(cell, (128,)*3)
    if np.any(exposure[pop,x,y,z] <= 0) or int(counts.sum()) != catalogue_record['used_counts']:
        raise RuntimeError('observed count support or total mismatch')
    prior = contract['published_prior']
    nbar = np.asarray(prior['original_mean_count_per_cell_bright_first'])*(3/prior['original_cell_cMpc_h'])**3
    settings = json.loads((ROOT/'config/cf4_datum_bearing_z0_phasec_program_v1.json').read_text())['inference_model']
    start = time.monotonic()
    intensity = np.asarray(predict_continuous_intensity(
        jnp.asarray(rho), jnp.asarray(velocity), jnp.asarray(exposure),
        jnp.asarray(nbar), jnp.asarray(prior['linear_regime_bias_bright_first']),
        jnp.zeros(6), box=384., observer=jnp.asarray([192.,192.,192.]),
        hubble=common['H0_km_s_Mpc'], little_h=common['h'],
        sigma_fog=jnp.asarray(settings['FoG_prior_median_km_s']),
        sigma_redshift=jnp.asarray(settings['fixed_redshift_error_km_s'])))
    occupied = intensity[pop,x,y,z]
    if not np.isfinite(intensity).all() or np.any(occupied <= 0):
        raise FloatingPointError('nonfinite or unsupported predicted count')
    predicted = intensity.sum(axis=(1,2,3))
    observed = np.bincount(pop, weights=counts, minlength=6)
    report = dict(classification='COMMON_COSMOLOGY_N128_RANDOM_PHASE_SUPPORT_ONLY',
                  cosmology=common, occupied_cells=int(len(keys)),
                  occupied_zero_intensity_cells=int(np.count_nonzero(occupied<=0)),
                  minimum_occupied_intensity=float(occupied.min()),
                  published_nbar_N128=nbar.tolist(),
                  observed_population_counts=observed.tolist(),
                  random_phase_predicted_population_counts=predicted.tolist(),
                  elapsed_seconds=time.monotonic()-start,
                  fixed_diffuse_fraction=0., diffuse_fraction_calibrated=False,
                  published_rate_or_bias_calibrated_to_R2=False,
                  FoG_calibrated=False, survival_uncertainty_marginalized=False,
                  actual_CF4_conditioned=False, joint_IC_gradient=False,
                  posterior_or_LG_identification=False)
    OUT.mkdir(parents=True)
    (OUT/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
