"""Determine whether higher fixed quadrature order removes Q1 gradient bias."""
import json
import numpy as np
import jax
import jax.numpy as jnp
from cf4_q1_state_dependent_jax import predict_selected_intensity_state_jax
from cf4_q1_cell_integrated_convolution import predict_selected_intensity_cell_integrated

jax.config.update("jax_enable_x64", True)
box = 6.; observer = np.array([3., 3., 3.])
vel = np.array([[31., -12., 8.]], dtype=np.float64)
masses = np.full((6, 1), .7); exposure = np.full((6, 8, 8, 8), .8)
exposure[:, 3, 2, 5] = .2; exposure[:, 5, 6, 1] = 1.3
kw = dict(observer=observer, box_size_cMpc_h=box, hubble_km_s_Mpc=74.6,
          little_h=.746, scale_factor=1., sigma_fog_km_s=np.full(6, 24.),
          sigma_redshift_km_s=np.full(6, 11.))
positions = np.array([[1.37, 2.11, 4.23]])
oracle = predict_selected_intensity_cell_integrated(positions, vel, masses, exposure, **kw)
oracle_scalar = float(np.sum(oracle * exposure ** 2))
rows = []
for order in (64, 128, 256, 512):
    def scalar(p):
        f = predict_selected_intensity_state_jax(p, jnp.asarray(vel), jnp.asarray(masses),
            jnp.asarray(exposure), quadrature_order=order, **kw)
        return jnp.sum(f * jnp.asarray(exposure) ** 2)
    candidate = np.asarray(predict_selected_intensity_state_jax(jnp.asarray(positions), jnp.asarray(vel),
        jnp.asarray(masses), jnp.asarray(exposure), quadrature_order=order, **kw))
    grad = float(jax.grad(scalar)(jnp.asarray(positions))[0, 0])
    h = 1e-5
    plus = predict_selected_intensity_cell_integrated(positions + [[h, 0, 0]], vel, masses, exposure, **kw)
    minus = predict_selected_intensity_cell_integrated(positions - [[h, 0, 0]], vel, masses, exposure, **kw)
    fd = float((np.sum(plus * exposure ** 2) - np.sum(minus * exposure ** 2)) / (2*h))
    rows.append(dict(order=order, value_relative_l1=float(np.sum(np.abs(candidate-oracle))/np.sum(np.abs(oracle))),
                     gradient=float(grad), oracle_fd=fd, gradient_abs_error=abs(grad-fd),
                     oracle_scalar=oracle_scalar))
print(json.dumps({"status": "QUADRATURE_ORDER_GATE", "rows": rows}, indent=2), flush=True)
