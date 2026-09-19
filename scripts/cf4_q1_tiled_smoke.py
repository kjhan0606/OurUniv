import numpy as np
import jax
import jax.numpy as jnp
from cf4_q1_state_dependent_jax import predict_selected_intensity_state_jax, predict_selected_intensity_state_jax_tiled

jax.config.update('jax_enable_x64', True)
box=6.; positions=jnp.array([[1.37,2.11,4.23],[5.8,.4,2.7]],dtype=jnp.float64)
vel=jnp.array([[31.,-12.,8.],[-15.,4.,9.]],dtype=jnp.float64)
masses=jnp.full((6,2),.7); exposure=jnp.full((6,8,8,8),.8)
kw=dict(observer=jnp.array([3.,3.,3.]),box_size_cMpc_h=box,hubble_km_s_Mpc=74.6,little_h=.746,scale_factor=1.,sigma_fog_km_s=jnp.full(6,24.),sigma_redshift_km_s=jnp.full(6,11.),quadrature_order=32)
dense=predict_selected_intensity_state_jax(positions,vel,masses,exposure,**kw)
tiled=predict_selected_intensity_state_jax_tiled(positions,vel,masses,exposure,tile_depth=2,**kw)
np.testing.assert_allclose(np.asarray(tiled),np.asarray(dense),rtol=0.,atol=1e-12)
def scalar(p):
    return jnp.sum(predict_selected_intensity_state_jax_tiled(p,vel,masses,exposure,tile_depth=2,**kw)**2)
grad=jax.grad(scalar)(positions)
assert np.isfinite(np.asarray(grad)).all()
print({'status':'TILED_Q1_PASS','max_abs':float(np.max(np.abs(np.asarray(tiled-dense)))),'grad_norm':float(jnp.linalg.norm(grad))},flush=True)
