"""Step-size sweep and JVP/VJP adjoint check for the Q1 state candidate."""
import json
import numpy as np
import jax
import jax.numpy as jnp
from cf4_q1_state_dependent_jax import predict_selected_intensity_state_jax
from cf4_q1_cell_integrated_convolution import predict_selected_intensity_cell_integrated

jax.config.update("jax_enable_x64", True)
box = 6.0; observer = np.array([3., 3., 3.])
vel = np.array([[31., -12., 8.]], dtype=np.float64)
masses = np.full((6, 1), .7); exposure = np.full((6, 8, 8, 8), .8)
exposure[:, 3, 2, 5] = .2; exposure[:, 5, 6, 1] = 1.3
kw = dict(observer=observer, box_size_cMpc_h=box, hubble_km_s_Mpc=74.6,
          little_h=.746, scale_factor=1., sigma_fog_km_s=np.full(6, 24.),
          sigma_redshift_km_s=np.full(6, 11.))
cases = {"interior": np.array([[1.37, 2.11, 4.23]]),
         "boundary": np.array([[.5 * (box / 8.) + .017, 2.11, 4.23]]),
         "seam": np.array([[box - .006, 2.11, 4.23]])}

def oracle_scalar(p):
    f = predict_selected_intensity_cell_integrated(p, vel, masses, exposure, **kw)
    return float(np.sum(f * exposure ** 2))
def candidate_scalar(p):
    f = predict_selected_intensity_state_jax(p, jnp.asarray(vel), jnp.asarray(masses), jnp.asarray(exposure), quadrature_order=64, **kw)
    return jnp.sum(f * jnp.asarray(exposure) ** 2)

rows = []
for name, p in cases.items():
    grad = float(jax.grad(candidate_scalar)(jnp.asarray(p))[0, 0])
    sweep = []
    for h in (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
        fd = (oracle_scalar(p + [[h, 0, 0]]) - oracle_scalar(p - [[h, 0, 0]])) / (2*h)
        sweep.append({"h": h, "oracle_fd": fd, "abs_error": abs(grad-fd)})
    rows.append({"case": name, "jax_grad": grad, "sweep": sweep})

p = jnp.asarray(cases["interior"]); v = jnp.asarray(vel)
def f(pos, velocity): return candidate_scalar(pos) + 0.0 * jnp.sum(velocity)
tangent_p = jnp.asarray([[.3, -.2, .1]]); tangent_v = jnp.asarray([[.4, -.1, .2]])
value, jvp_value = jax.jvp(lambda pp, vv: f(pp, vv), (p, v), (tangent_p, tangent_v))
value2, pullback = jax.vjp(lambda pp, vv: f(pp, vv), p, v)
cotangent = jnp.asarray(1.0); vjp_p, vjp_v = pullback(cotangent)
lhs = float(jvp_value); rhs = float(jnp.sum(vjp_p * tangent_p) + jnp.sum(vjp_v * tangent_v))
print(json.dumps({"status": "GRADIENT_SWEEP", "rows": rows,
                  "jvp": lhs, "vjp_dot": rhs, "adjoint_abs_error": abs(lhs-rhs)}, indent=2), flush=True)
