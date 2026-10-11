"""Slurm-only value/gradient smoke for the state-dependent JAX candidate."""
import numpy as np
import jax
import jax.numpy as jnp

from cf4_q1_state_dependent_jax import predict_selected_intensity_state_jax
from cf4_q1_cell_integrated_convolution import predict_selected_intensity_cell_integrated

jax.config.update("jax_enable_x64", True)
box = 6.0
positions = np.array([[1.37, 2.11, 4.23]], dtype=np.float64)
velocities = np.array([[31.0, -12.0, 8.0]], dtype=np.float64)
masses = np.full((6, 1), 0.7, dtype=np.float64)
exposure = np.full((6, 8, 8, 8), 0.8, dtype=np.float64)
exposure[:, 3, 2, 5] = 0.2
exposure[:, 5, 6, 1] = 1.3
kw = dict(observer=np.array([3.0, 3.0, 3.0]), box_size_cMpc_h=box,
          hubble_km_s_Mpc=74.6, little_h=0.746, scale_factor=1.0,
          sigma_fog_km_s=np.full(6, 24.0), sigma_redshift_km_s=np.full(6, 11.0))
oracle = predict_selected_intensity_cell_integrated(positions, velocities, masses, exposure, **kw)
candidate = predict_selected_intensity_state_jax(jnp.asarray(positions), jnp.asarray(velocities),
    jnp.asarray(masses), jnp.asarray(exposure), quadrature_order=64, **kw)
candidate_np = np.asarray(candidate)
relative_l1 = np.sum(np.abs(candidate_np - oracle)) / np.sum(np.abs(oracle))
assert np.isfinite(candidate_np).all()
assert relative_l1 < 5.0e-3, f"candidate/oracle relative L1={relative_l1}"

def scalar(pos):
    field = predict_selected_intensity_state_jax(pos, jnp.asarray(velocities), jnp.asarray(masses),
        jnp.asarray(exposure), quadrature_order=64, **kw)
    # Non-uniform weights prevent the conservation identity from masking the
    # position derivative check.
    return jnp.sum(field * jnp.asarray(exposure) ** 2)

grad = jax.grad(scalar)(jnp.asarray(positions))
step = 1.0e-5
plus = float(scalar(jnp.asarray(positions + np.array([[step, 0.0, 0.0]]))))
minus = float(scalar(jnp.asarray(positions - np.array([[step, 0.0, 0.0]]))))
finite_difference = (plus - minus) / (2.0 * step)
assert np.isfinite(np.asarray(grad)).all()
assert abs(float(grad[0, 0]) - finite_difference) < 1.0e-4
print({"status": "STATE_JAX_CANDIDATE_PASS", "relative_l1": float(relative_l1),
       "gradient_x": float(grad[0, 0]), "finite_difference_x": float(finite_difference)}, flush=True)
