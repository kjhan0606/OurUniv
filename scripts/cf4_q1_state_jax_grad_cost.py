"""Gradient-inclusive cost benchmark for the Q1 state candidate."""
import json, os, resource, time
import numpy as np
import jax
import jax.numpy as jnp
from cf4_q1_state_dependent_jax import predict_selected_intensity_state_jax

jax.config.update("jax_enable_x64", True)
if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Slurm required")
rng = np.random.default_rng(20260919)
box = float(os.environ.get("Q1_BENCHMARK_BOX", "384.0"))
sources = int(os.environ.get("Q1_BENCHMARK_SOURCES", "32"))
grid = int(os.environ.get("Q1_JAX_GRAD_GRID", "128"))
positions = jnp.asarray(rng.uniform(0.0, box, (sources, 3)))
velocities = jnp.asarray(rng.normal(0.0, 100.0, (sources, 3)))
masses = jnp.asarray(np.abs(rng.normal(1.0, 0.2, (6, sources))))
exposure = jnp.full((6, grid, grid, grid), 0.8, dtype=jnp.float64)
kw = dict(observer=jnp.full(3, box / 2.0), box_size_cMpc_h=box,
          hubble_km_s_Mpc=74.6, little_h=.746, scale_factor=1.0,
          sigma_fog_km_s=jnp.full(6, 100.0), sigma_redshift_km_s=jnp.full(6, 30.0),
          quadrature_order=64)

def objective(pos, vel):
    field = predict_selected_intensity_state_jax(pos, vel, masses, exposure, **kw)
    return jnp.sum(field * exposure ** 2)

start = time.perf_counter()
value, (grad_pos, grad_vel) = jax.value_and_grad(objective, argnums=(0, 1))(positions, velocities)
jax.block_until_ready(value)
jax.block_until_ready(grad_pos)
jax.block_until_ready(grad_vel)
elapsed = time.perf_counter() - start
row = dict(grid=grid, sources=sources, populations=6, seconds=elapsed,
           peak_host_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
           value=float(value), grad_pos_norm=float(jnp.linalg.norm(grad_pos)),
           grad_vel_norm=float(jnp.linalg.norm(grad_vel)), finite=bool(np.isfinite(np.asarray(grad_pos)).all() and np.isfinite(np.asarray(grad_vel)).all()))
print(json.dumps(row), flush=True)
out = '/gpfs/kjhan/CF4/q1_state_jax_grad_cost/job_' + os.environ['SLURM_JOB_ID']
os.makedirs(out, exist_ok=False)
json.dump(dict(status='COMPLETE_JAX_GRADIENT_COST', source_commit=os.environ['EXPECTED_COMMIT'], box_cMpc_h=box, row=row), open(out + '/result.json', 'w'), indent=2)
