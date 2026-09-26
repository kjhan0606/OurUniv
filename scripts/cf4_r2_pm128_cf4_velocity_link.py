"""Read actual CF4 radial rows and actual 2M++ support from one PM state.

The PM phase is unconditional. Residuals are wiring diagnostics, not a fit.
"""

import json
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cf4_z0_physical_field import read_centred

DATA = Path('/gpfs/kjhan/CF4/z0_density/bundle_c_v1/native_data_v2/CF4_native.npz')
STATE = Path('/gpfs/kjhan/CF4/z0_density/r2_pm128_unconditional_v1/state.npz')
COUNT_SUPPORT = Path('/gpfs/kjhan/CF4/z0_density/r2_pm128_observed_support_v1/result.json')
OUTPUT = Path('/gpfs/kjhan/CF4/z0_density/r2_pm128_cf4_velocity_link_v1')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    count = json.loads(COUNT_SUPPORT.read_text())
    if count['trials']['0.0']['occupied_zero_intensity_cells'] != 0:
        raise RuntimeError('zero-diffuse count support failed')
    with np.load(STATE, allow_pickle=False) as data:
        v = data['velocity_km_s'].astype(np.float64)
    with np.load(DATA, allow_pickle=False) as data:
        pos = data['CF4_pos'].copy()
        direction = data['CF4_rhat'].copy()
        observed = data['radial_observed'].copy()
        variance = data['CF4_variance'].copy()
        holdout = data['CF4_holdout'].copy()
    predicted_vector = jax.jit(lambda x: jnp.stack(
        [read_centred(jnp.asarray(v[k]), x, 384., 0.5) for k in range(3)], axis=-1))(
            jnp.asarray(pos))
    prediction = np.sum(np.asarray(predicted_vector) * direction, axis=1)
    residual = observed - prediction
    if np.any(~np.isfinite(residual)) or np.any(variance <= 0):
        raise RuntimeError('invalid CF4 same-state radial readout')
    report = dict(classification='SAME_UNCONDITIONAL_N128_PM_STATE_CF4_AND_2MPP_WIRING',
                  source_commit=os.environ['EXPECTED_COMMIT'],
                  state=str(STATE), count_support=str(COUNT_SUPPORT),
                  count_occupied_cells=count['occupied_cells'],
                  count_occupied_zero_intensity_cells=count['trials']['0.0']['occupied_zero_intensity_cells'],
                  CF4_rows=int(len(pos)), CF4_train=int(np.count_nonzero(~holdout)),
                  CF4_holdout=int(np.count_nonzero(holdout)),
                  CF4_predicted_radial_rms_km_s=float(np.sqrt(np.mean(prediction**2))),
                  CF4_residual_rms_train_km_s=float(np.sqrt(np.mean(residual[~holdout]**2))),
                  CF4_residual_rms_holdout_km_s=float(np.sqrt(np.mean(residual[holdout]**2))),
                  CF4_standardized_residual_rms_train=float(np.sqrt(np.mean(residual[~holdout]**2/variance[~holdout]))),
                  CF4_standardized_residual_rms_holdout=float(np.sqrt(np.mean(residual[holdout]**2/variance[holdout]))),
                  same_PM_density_velocity_state=True, actual_data_likelihood=False,
                  CF4_phase_conditioned=False, galaxy_bias_calibrated=False,
                  R2_posterior=False)
    OUTPUT.mkdir(parents=True)
    (OUTPUT / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
