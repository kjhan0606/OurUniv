"""Bounded 384-box tracer-operator mechanics/cost check; no posterior fit."""

import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cf4_r2_continuous_tracer import predict_continuous_intensity
from cf4_z0_physical_field import recenter_old_pm_fields

OUTPUT = Path('/gpfs/kjhan/CF4/z0_density/r2_continuous_feasibility_v1')


def input_fields(n):
    if n == 32:
        plan = json.loads((ROOT / 'config/cf4_z6_native_physics_plan_v1.json').read_text())
        path = Path(plan['data']['posterior_root']) / 'posterior_v1_00_mock_00_seed_2026083000_arm_A' / 'posterior_summary.npz'
        with np.load(path, allow_pickle=False) as data:
            rho = 1 + data['truth_coarse_density'].astype(float)
            velocity = data['truth_coarse_velocity'].astype(float)
        rho, velocity = recenter_old_pm_fields(rho, velocity)
        with np.load('/gpfs/kjhan/CF4/z0_density/actual_data_corrected_v2_run1/corrected_counts.npz',
                     allow_pickle=False) as data:
            counts = data['counts_all'].astype(int)
            survival = (data['survival_yes'] + 1) / (data['survival_yes'] + data['survival_no'] + 2)
            exposure = np.einsum('ps,psijk->pijk', survival, data['selection_shells'])
        label = 'ARCHIVED_PM_N32_ACTUAL_SELECTION_SUPPORT_ONLY'
    elif n == 128:
        # Analytical field is ONLY an operator cost/gradient probe, not PM.
        phase = 2 * np.pi * (np.arange(n) + 0.5) / n
        x, y, z = np.meshgrid(phase, phase, phase, indexing='ij')
        rho = np.exp(0.2 * (np.sin(x) + np.cos(y) + np.sin(z)))
        rho /= rho.mean()
        velocity = np.stack([30 * np.cos(x), 20 * np.sin(y), 10 * np.cos(z)])
        exposure = np.ones((6, n, n, n), dtype=np.float64)
        counts = None
        label = 'ANALYTIC_N128_OPERATOR_COST_NOT_PM_OR_OBSERVED_SELECTION'
    else:
        raise ValueError('fixed n32/n128 probe only')
    return rho, velocity, exposure, counts, label


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('numerical probe must run through Slurm')
    if not jax.config.x64_enabled:
        raise RuntimeError('JAX_ENABLE_X64=1 required')
    n = int(sys.argv[1])
    rho, velocity, exposure, counts, label = input_fields(n)
    if np.any(rho <= 0) or np.any(exposure < 0):
        raise ValueError('invalid physical input or selection')
    published = json.loads((ROOT / 'config/cf4_datum_bearing_z0_phasec_program_v1.json').read_text())
    bias = jnp.asarray(published['external_population_prior']['published_bias'])
    settings = published['inference_model']
    fraction = jnp.full((6,), 0.03)
    # These values are explicit mechanics placeholders, NOT calibrated rates.
    nbar = jnp.full((6,), 1.0)
    args = dict(velocity=jnp.asarray(velocity), exposure=jnp.asarray(exposure),
                nbar=nbar, bias=bias, diffuse_fraction=fraction,
                box=384., observer=jnp.asarray([192., 192., 192.]),
                hubble=74.6, little_h=.746,
                sigma_fog=jnp.asarray(settings['FoG_prior_median_km_s']),
                sigma_redshift=jnp.asarray(settings['fixed_redshift_error_km_s']))
    base = jnp.asarray(rho)
    forward = jax.jit(lambda scale: predict_continuous_intensity(base**scale, **args))
    t0 = time.monotonic()
    intensity = forward(1.0)
    intensity.block_until_ready()
    compile_forward_s = time.monotonic() - t0
    t0 = time.monotonic()
    forward(1.01).block_until_ready()
    warm_forward_s = time.monotonic() - t0
    # A uniform sum is conserved by deposition and has a trivial derivative.
    # Use a fixed nonuniform probe to time a genuinely state-sensitive adjoint.
    weight_axis = jnp.sin(2 * jnp.pi * (jnp.arange(n) + 0.5) / n)
    weight = 1.0 + 0.25 * weight_axis[:, None, None]
    score = jax.jit(jax.value_and_grad(lambda scale: jnp.sum(forward(scale) * weight)))
    t0 = time.monotonic()
    value, gradient = score(1.0)
    gradient.block_until_ready()
    compile_gradient_s = time.monotonic() - t0
    t0 = time.monotonic()
    value2, gradient2 = score(1.01)
    gradient2.block_until_ready()
    warm_gradient_s = time.monotonic() - t0
    predicted = np.asarray(intensity)
    observed_unsupported = None if counts is None else int(np.sum((counts > 0) & (predicted <= 0)))
    result = dict(classification=label, grid=n, box_cMpc_h=384., cell_cMpc_h=384./n,
                  jax_backend=jax.default_backend(),
                  source_cell_count=n**3, populations=6,
                  diffuse_unresolved_fraction_uncalibrated=0.03,
                  rate_per_cell_uncalibrated=1.0,
                  minimum_positive_exposure_intensity=float(predicted[np.asarray(exposure) > 0].min()),
                  occupied_zero_intensity_cells=observed_unsupported,
                  compile_forward_s=compile_forward_s, warm_forward_s=warm_forward_s,
                  compile_gradient_s=compile_gradient_s, warm_gradient_s=warm_gradient_s,
                  score_at_1=float(value), derivative_at_1=float(gradient),
                  score_at_1p01=float(value2), derivative_at_1p01=float(gradient2),
                  actual_CF4_or_2Mpp_likelihood=False, calibrated_tracer=False,
                  native_N128_PM_state=False, R2_posterior=False)
    if not np.isfinite([result['score_at_1'], result['derivative_at_1']]).all() or observed_unsupported:
        raise RuntimeError('operator support or derivative failure')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / f"job_{os.environ['SLURM_JOB_ID']}_n{n}.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
