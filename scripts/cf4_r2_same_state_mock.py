"""Small R2 wiring control, not an actual-data likelihood or posterior."""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cf4_2mpp_joint_likelihood_jax import (
    observer_centred_spherical_rsd_jax,
    poisson_log_likelihood_jax,
    predict_selected_intensity_jax,
)


SOURCE = Path('/gpfs/kjhan/CF4/z0_density/r1_particle_entry_v1/job_354568/base.npz')
OUTPUT = Path('/gpfs/kjhan/CF4/z0_density/r2_same_state_mock_v1')


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Submit this numerical control through Slurm')
    if not jax.config.x64_enabled:
        raise RuntimeError('JAX_ENABLE_X64=1 is required')
    with np.load(SOURCE) as state:
        # Fixed, observation-independent subset of one saved evolved PM state.
        index = np.linspace(0, len(state['position_cMpc_h']) - 1, 64, dtype=int)
        position = jnp.asarray(state['position_cMpc_h'][index], dtype=jnp.float64)
        velocity = jnp.asarray(state['velocity_km_s'][index], dtype=jnp.float64)
    box, ngrid, h, hubble = 12.0, 16, 0.746, 74.6
    observer = jnp.asarray([6.0, 6.0, 6.0])
    population_mass = np.zeros((6, len(index)))
    population_mass[np.arange(len(index)) % 6, np.arange(len(index))] = 1.0
    population_mass = jnp.asarray(population_mass)
    exposure = jnp.ones((6, ngrid, ngrid, ngrid), dtype=jnp.float64)
    sigma_fog = jnp.full(6, 100.0)
    sigma_redshift = jnp.full(6, 30.0)
    relative = (position - observer + box / 2) % box - box / 2
    direction = relative / jnp.linalg.norm(relative, axis=1)[:, None]
    rng = np.random.default_rng(20260925)
    observed_radial = jnp.sum(velocity * direction, axis=1) + jnp.asarray(rng.normal(0, 30, len(index)))

    def intensity_at(scale):
        return predict_selected_intensity_jax(
            position, velocity * scale, population_mass, exposure,
            observer=observer, box_size_cMpc_h=box,
            hubble_km_s_Mpc=hubble, little_h=h, scale_factor=1.0,
            sigma_fog_km_s=sigma_fog, sigma_redshift_km_s=sigma_redshift,
            quadrature_order=3,
        )

    true_intensity = intensity_at(1.0)
    counts = jnp.asarray(rng.poisson(np.asarray(true_intensity)), dtype=jnp.float64)

    def factors(scale):
        predicted_radial = jnp.sum(velocity * scale * direction, axis=1)
        radial = -0.5 * jnp.sum(((observed_radial - predicted_radial) / 30.0) ** 2)
        count = poisson_log_likelihood_jax(counts, intensity_at(scale))
        return radial, count

    at_true = factors(1.0)
    at_shifted = factors(0.9)
    derivatives = jax.jacfwd(factors)(1.0)
    # A separate radial readout from the same state must use the same LOS
    # convention as the count operator's coherent RSD.
    _, displacement, _ = observer_centred_spherical_rsd_jax(
        position, velocity, observer, box, hubble, little_h=h, scale_factor=1.0,
    )
    np.testing.assert_allclose(np.asarray(displacement),
                               h * np.asarray(jnp.sum(velocity * direction, axis=1)) / hubble,
                               atol=1e-12)
    result = {
        'classification': 'SYNTHETIC_SAME_STATE_WIRING_ONLY',
        'source': str(SOURCE), 'source_particles': int(len(index)),
        'box_cMpc_h': box, 'count_cell_cMpc_h': box / ngrid,
        'log_factor_at_velocity_scale_1': [float(x) for x in at_true],
        'log_factor_at_velocity_scale_0p9': [float(x) for x in at_shifted],
        'factor_derivative_at_1': [float(x) for x in derivatives],
        'count_total': float(counts.sum()),
        'intensity_total': float(true_intensity.sum()),
        'actual_CF4_or_2Mpp_used': False,
        'physical_galaxy_tracer_model': False,
        'posterior_or_R2_completion': False,
    }
    if not np.isfinite(np.asarray(list(result['factor_derivative_at_1']))).all():
        raise RuntimeError('nonfinite state sensitivity')
    if not all(abs(float(a) - float(b)) > 1e-8 for a, b in zip(at_true, at_shifted)):
        raise RuntimeError('both observation factors must respond to the same state')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / f"job_{os.environ['SLURM_JOB_ID']}.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
