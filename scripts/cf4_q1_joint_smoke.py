"""Dependency-light JAX smoke checks for the Q1 joint kernel."""
import numpy as np
import jax
import jax.numpy as jnp
from cf4_2mpp_joint_likelihood_jax import observer_centred_spherical_rsd_jax, poisson_log_likelihood_jax

jax.config.update('jax_enable_x64', True)
observer=jnp.array([3.,3.,3.]); positions=jnp.array([[3.,3.,3.],[1.,2.,4.]],dtype=jnp.float64)
vel=jnp.array([[1.,2.,3.],[4.,5.,6.]],dtype=jnp.float64)
shift, disp, rhat=observer_centred_spherical_rsd_jax(positions,vel,observer,6.,100.,little_h=.746,scale_factor=1.)
assert np.isfinite(np.asarray(shift)).all() and np.isfinite(np.asarray(disp)).all()
counts=jnp.zeros((2,),dtype=jnp.float64); intensity=jnp.array([0.,1.],dtype=jnp.float64)
value, grad=jax.value_and_grad(lambda x: poisson_log_likelihood_jax(counts,x))(intensity)
assert np.isfinite(float(value)) and np.isfinite(np.asarray(grad)).all()
assert abs(float(grad[0])+1.) < 1e-12
print('Q1_JOINT_SMOKE_PASS', flush=True)
