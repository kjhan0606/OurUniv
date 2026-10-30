"""One genuine 384/128 PMWD state for R2 response calibration development.

Unconditional phase: it is NOT a CF4-reconstructed state or an inference draw.
No new RAMSES snapshot or galaxy catalogue is produced.
"""

import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cf4_r1_particle_forward import make_dynamics, particle_grid

OUTPUT = Path('/gpfs/kjhan/CF4/z0_density/r2_pm128_unconditional_v1')


def main():
    if not os.environ.get('SLURM_JOB_ID') or jax.default_backend() != 'gpu':
        raise RuntimeError('Slurm GPU required')
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    base = json.loads((ROOT / 'config/cf4_r1_particle_entry_v1.json').read_text())
    settings = {key: base[key] for key in ('cosmology', 'a_start', 'a_stop', 'a_nbody_maxstep')}
    settings.update(n=128, box_cMpc_h=384.0)
    seed = 2026092501
    start = time.monotonic()
    evolve, _initial, conf, _cosmo, particle_mass = make_dynamics(settings)
    white = jnp.asarray(np.random.default_rng(seed).standard_normal(128**3))
    position, velocity = evolve(white)
    position.block_until_ready()
    forward_s = time.monotonic() - start
    mass = jnp.full((128**3,), particle_mass)
    field = particle_grid(position, velocity, mass, conf)
    rho = np.asarray(field['rho'])
    v = np.asarray(jnp.moveaxis(field['mean_velocity_km_s'], -1, 0))
    if not np.isfinite(rho).all() or not np.isfinite(v).all():
        raise FloatingPointError('invalid PM field')
    if abs(rho.mean() - 1.0) > 1e-8:
        raise RuntimeError('PM mass conservation failure')
    OUTPUT.mkdir(parents=True)
    np.savez_compressed(OUTPUT / 'state.npz', rho=rho.astype(np.float32),
                        velocity_km_s=v.astype(np.float32))
    report = dict(classification='UNCONDITIONAL_384_N128_PM_STATE',
                  source_commit=os.environ['EXPECTED_COMMIT'], seed=seed,
                  grid=128, box_cMpc_h=384.0, dx_cMpc_h=3.0,
                  particle_count=128**3, particle_mass_Msun_h=particle_mass,
                  cosmology=settings['cosmology'], PM_forward_seconds=forward_s,
                  total_seconds=time.monotonic()-start,
                  rho_min=float(rho.min()), rho_max=float(rho.max()),
                  zero_density_cells=int(np.count_nonzero(rho <= 0)),
                  velocity_RMS_km_s=float(np.sqrt(np.mean(v*v))),
                  actual_CF4_conditioned=False, matched_high_fidelity_pair=False,
                  tracer_calibration=False, R2_posterior=False)
    (OUTPUT / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
