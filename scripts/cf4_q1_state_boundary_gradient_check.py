"""Boundary/wrap value and state-gradient check for the Q1 JAX candidate."""
import json
import os
import numpy as np
import jax
import jax.numpy as jnp

from cf4_q1_state_dependent_jax import predict_selected_intensity_state_jax
from cf4_q1_cell_integrated_convolution import predict_selected_intensity_cell_integrated

jax.config.update("jax_enable_x64", True)
box = 6.0
observer = np.array([3.0, 3.0, 3.0])
velocities = np.array([[31.0, -12.0, 8.0]], dtype=np.float64)
masses = np.full((6, 1), 0.7, dtype=np.float64)
exposure = np.full((6, 8, 8, 8), 0.8, dtype=np.float64)
exposure[:, 3, 2, 5] = 0.2
exposure[:, 5, 6, 1] = 1.3
kw = dict(observer=observer, box_size_cMpc_h=box, hubble_km_s_Mpc=74.6,
          little_h=0.746, scale_factor=1.0,
          sigma_fog_km_s=np.full(6, 24.0), sigma_redshift_km_s=np.full(6, 11.0))

cases = {
    "cell_boundary": np.array([[0.5 * (box / 8.0) + 0.017, 2.11, 4.23]]),
    "periodic_seam": np.array([[box - 0.006, 2.11, 4.23]]),
    "interior": np.array([[1.37, 2.11, 4.23]]),
}
rows = []
order = int(os.environ.get("Q1_GRAD_ORDER", "64"))
for name, positions in cases.items():
    oracle = predict_selected_intensity_cell_integrated(positions, velocities, masses, exposure, **kw)
    candidate = np.asarray(predict_selected_intensity_state_jax(
        jnp.asarray(positions), jnp.asarray(velocities), jnp.asarray(masses),
        jnp.asarray(exposure), quadrature_order=order, **kw))
    rel = float(np.sum(np.abs(candidate - oracle)) / np.sum(np.abs(oracle)))
    rows.append({"case": name, "relative_l1": rel, "finite": bool(np.isfinite(candidate).all())})

def oracle_scalar(position, velocity):
    field = predict_selected_intensity_cell_integrated(position, velocity, masses, exposure, **kw)
    return float(np.sum(field * exposure ** 2))

def candidate_scalar(position, velocity):
    field = predict_selected_intensity_state_jax(position, velocity, jnp.asarray(masses),
        jnp.asarray(exposure), quadrature_order=order, **kw)
    return jnp.sum(field * jnp.asarray(exposure) ** 2)

position = cases["interior"]
step = 1.0e-5
oracle_px = (oracle_scalar(position + [[step, 0, 0]], velocities) - oracle_scalar(position - [[step, 0, 0]], velocities)) / (2 * step)
oracle_vx = (oracle_scalar(position, velocities + [[step, 0, 0]]) - oracle_scalar(position, velocities - [[step, 0, 0]])) / (2 * step)
_, grad = jax.value_and_grad(lambda p, v: candidate_scalar(p, v), argnums=(0, 1))(jnp.asarray(position), jnp.asarray(velocities))
rows.append({"case": "interior_position_gradient", "jax": float(grad[0][0, 0]), "oracle_fd": float(oracle_px), "abs_error": abs(float(grad[0][0, 0]) - oracle_px)})
rows.append({"case": "interior_velocity_gradient", "jax": float(grad[1][0, 0]), "oracle_fd": float(oracle_vx), "abs_error": abs(float(grad[1][0, 0]) - oracle_vx)})
print(json.dumps({"status": "DIAGNOSTIC_ONLY", "quadrature_order": order, "rows": rows}, indent=2), flush=True)
