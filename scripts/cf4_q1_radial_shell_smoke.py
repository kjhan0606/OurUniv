import numpy as np
import jax
import jax.numpy as jnp
from cf4_q1_radial_shell_fourier_jax import predict_selected_intensity_radial_shell_fourier_jax

jax.config.update('jax_enable_x64', True)
n=16; box=6.; positions=jnp.array([[1.3,2.1,4.2],[5.7,.4,2.7]],dtype=jnp.float64); vel=jnp.array([[20.,-5.,8.],[-12.,4.,9.]])
masses=jnp.full((6,2),.7); exposure=jnp.full((6,n,n,n),.8); ids=jnp.array([0,1]); rhat=jnp.array([[1.,0.,0.],[0.,1.,0.]]); sigma=jnp.array([.2,.25])
kw=dict(observer=jnp.array([3.,3.,3.]),box_size_cMpc_h=box,hubble_km_s_Mpc=74.6,little_h=.746,scale_factor=1.)
field=predict_selected_intensity_radial_shell_fourier_jax(positions,vel,masses,exposure,ids,rhat,sigma,**kw)
def scalar(p): return jnp.sum(predict_selected_intensity_radial_shell_fourier_jax(p,vel,masses,exposure,ids,rhat,sigma,**kw)**2)
grad=jax.grad(scalar)(positions)
assert np.isfinite(np.asarray(field)).all() and np.isfinite(np.asarray(grad)).all()
print({'status':'RADIAL_SHELL_FOURIER_PASS','mass':float(field.sum()),'grad_norm':float(jnp.linalg.norm(grad))},flush=True)
