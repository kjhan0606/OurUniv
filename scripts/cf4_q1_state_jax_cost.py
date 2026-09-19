"""Bounded Slurm benchmark for the differentiable Q1 state candidate."""
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
grids = tuple(int(x) for x in os.environ.get("Q1_JAX_BENCH_GRIDS", "128,256").split(","))
positions = rng.uniform(0.0, box, (sources, 3))
velocities = rng.normal(0.0, 100.0, (sources, 3))
masses = np.abs(rng.normal(1.0, 0.2, (6, sources)))
kw = dict(observer=np.full(3, box / 2.0), box_size_cMpc_h=box,
          hubble_km_s_Mpc=74.6, little_h=.746, scale_factor=1.0,
          sigma_fog_km_s=np.full(6, 100.0), sigma_redshift_km_s=np.full(6, 30.0),
          quadrature_order=64)
out = '/gpfs/kjhan/CF4/q1_state_jax_cost/job_' + os.environ['SLURM_JOB_ID']
os.makedirs(out, exist_ok=False)
rows = []
for n in grids:
    exposure = jnp.full((6, n, n, n), 0.8, dtype=jnp.float64)
    start = time.perf_counter()
    field = predict_selected_intensity_state_jax(jnp.asarray(positions), jnp.asarray(velocities),
        jnp.asarray(masses), exposure, **kw)
    field.block_until_ready()
    elapsed = time.perf_counter() - start
    row = dict(grid=n, sources=sources, populations=6, seconds=elapsed,
               peak_host_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               total_mass=float(field.sum()), finite=bool(np.isfinite(np.asarray(field)).all()))
    rows.append(row)
    print(json.dumps(row), flush=True)
json.dump(dict(status='COMPLETE_JAX_STATE_DEVELOPMENT_BENCHMARK', source_commit=os.environ['EXPECTED_COMMIT'],
               box_cMpc_h=box, rows=rows,
               limitations='Candidate fixed Gauss-Legendre operator; no production authorization.'),
          open(out + '/result.json', 'w'), indent=2)
