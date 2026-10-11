"""Small direct Slurm smoke for the Q1 shifted-position LOS convention."""
import numpy as np
import jax
import jax.numpy as jnp

from cf4_2mpp_joint_likelihood_jax import observer_centred_spherical_rsd_jax

jax.config.update("jax_enable_x64", True)
observer = np.array([3.0, 3.0, 3.0])
positions = np.array([[5.9, 3.0, 3.0], [0.1, 3.0, 3.0]], dtype=np.float64)
velocities = np.array([[120.0, 0.0, 0.0], [-120.0, 0.0, 0.0]], dtype=np.float64)
shifted, displacement, rhat = observer_centred_spherical_rsd_jax(
    jnp.asarray(positions), jnp.asarray(velocities), jnp.asarray(observer),
    6.0, 100.0, little_h=0.746, scale_factor=1.0,
)
shifted_np = np.asarray(shifted)
relative = (shifted_np - observer + 3.0) % 6.0 - 3.0
expected = relative / np.linalg.norm(relative, axis=1)[:, None]
np.testing.assert_allclose(np.asarray(rhat), expected, rtol=1e-12, atol=1e-12)
assert np.isfinite(np.asarray(displacement)).all()
print("Q1_LOS_CONVENTION_PASS", flush=True)
