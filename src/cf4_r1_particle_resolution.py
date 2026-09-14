"""Common continuous IC sampled by more particles; NOT a full-power zoom IC."""
import jax
import jax.numpy as jnp
import numpy as np

from cf4_r1_particle_forward import make_configuration


def refine_lpt_state(displacement, velocity, fine_n):
    """Fourier-resample the existing periodic LPT displacement AND velocity.

    This holds the interpolated initial field fixed, including its original
    LPT discretization errors. No new random high-k modes or power matching.
    SciPy resample splits even-grid Nyquist bins, preserving coarse-node
    values. Coordinate positions themselves must NEVER be FFT-interpolated.
    """
    from scipy.signal import resample
    displacement, velocity = np.asarray(displacement), np.asarray(velocity)
    n = displacement.shape[0]
    if (displacement.shape != (n, n, n, 3) or velocity.shape != displacement.shape or
            not isinstance(fine_n, int) or fine_n < n or fine_n % n):
        raise ValueError('require cubic vector fields and integer grid refinement')
    refined = []
    for field in (displacement, velocity):
        value = field.astype(np.float64, copy=True)
        if fine_n != n:
            for axis in range(3):
                value = resample(value, fine_n, axis=axis)
        refined.append(value)
    return tuple(refined)


def make_state_evolution(settings, *, mesh_ratio=1, time_factor=1):
    """PMWD forward on supplied growing-mode ICs, using its SAME integrator.

    lax.scan avoids unrolling hundreds of steps into an enormous executable.
    This numerical forward is not an independent gravity solver or a validated
    fine-particle adjoint. Its final output is compared to existing PMWD runs.
    Input displacement: cMpc/h; input/output peculiar velocity: km/s.
    """
    conf, cosmo = make_configuration(settings, mesh_ratio=mesh_ratio, time_factor=time_factor)
    from pmwd.particles import Particles
    from pmwd.nbody import nbody_init, nbody_step
    grid = Particles.gen_grid(conf, vel=True, acc=True)
    scale_factors = jnp.asarray(conf.a_nbody)

    @jax.jit
    def evolve(displacement, velocity):
        particles = grid.replace(disp=grid.disp+displacement.reshape((-1, 3)),
                                 vel=velocity.reshape((-1, 3))*settings['a_start']/100)
        particles, obs = nbody_init(scale_factors[0], particles, None, cosmo, conf)

        def step(carry, scales):
            state, observation = carry
            state, observation = nbody_step(scales[0], scales[1], state, observation, cosmo, conf)
            return (state, observation), None

        (particles, _), _ = jax.lax.scan(step, (particles, obs),
                                        (scale_factors[:-1], scale_factors[1:]))
        return particles.pos(), particles.vel*100/settings['a_stop']

    return evolve, conf, cosmo, float(cosmo.ptcl_mass)*1e10
