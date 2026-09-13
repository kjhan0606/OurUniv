"""R1 particle-backed moments and PM mechanics; no assumed LG identities."""
import jax
import jax.numpy as jnp

from cf4_z0_pm_bridge import enable_pmwd_type_description_compatibility


def periodic_delta(position, center, box):
    return (position - center + box / 2) % box - box / 2


def aperture_moments(position, velocity, mass, centers, radius, box):
    """Unnormalised Gaussian aperture mass, centroid offset, u and physical var.

    Positions/radius: cMpc/h. Mass: Msun/h. Velocity: peculiar km/s.
    Apertures can overlap and are NOT disjoint member masses, M200c or bound
    subhalos. Centers are supplied probes/observational positions, not truth IDs.
    No halo identity or existence flag is inferred from a positive aperture.
    """
    delta = periodic_delta(position[None], centers[:, None], box)
    weights = mass[None] * jnp.exp(-0.5 * jnp.sum((delta / radius) ** 2, axis=-1))
    total = jnp.sum(weights, axis=1)
    denom = jnp.maximum(total, jnp.finfo(total.dtype).tiny)
    normalized = weights / denom[:, None]
    mean = normalized @ velocity
    # Centered form avoids subtracting two large bulk-velocity moments.
    variance = jnp.sum(normalized[..., None] * (velocity[None] - mean[:, None]) ** 2, axis=1)
    return dict(mass_Msun_h=total,
                offset_cMpc_h=jnp.sum(normalized[..., None] * delta, axis=1),
                mean_velocity_km_s=mean, variance_km2_s2=variance,
                effective_particles=1 / jnp.maximum(jnp.sum(normalized**2, axis=1), 1e-300))


def make_dynamics(settings, *, mesh_ratio=1, time_factor=1):
    """Same LPT phases/particle mass, optionally finer force mesh/time steps."""
    enable_pmwd_type_description_compatibility()
    from pmwd import Configuration, SimpleLCDM, boltzmann, linear_modes, lpt, nbody
    n, box = settings['n'], settings['box_cMpc_h']
    c = settings['cosmology']
    conf = Configuration(ptcl_spacing=box/n, ptcl_grid_shape=(n,)*3,
                         mesh_shape=mesh_ratio, cosmo_dtype=jnp.float64,
                         float_dtype=jnp.float64, a_start=settings['a_start'],
                         a_stop=settings['a_stop'], lpt_order=2,
                         a_nbody_maxstep=settings['a_nbody_maxstep']/time_factor)
    cosmo = boltzmann(SimpleLCDM(conf, Omega_m=c['Om'], Omega_b=c['Ob'],
                                h=c['h'], A_s_1e9=c['A_s_1e9'], n_s=c['ns']), conf)

    def initial_particles(white):
        modes = linear_modes(white.reshape((n,)*3), cosmo, conf)
        particles, obs = lpt(modes, cosmo, conf)
        return particles.replace(acc=jnp.zeros_like(particles.disp)), obs

    @jax.jit
    def initial(white):
        particles, _ = initial_particles(white)
        return particles.pos(wrap=False), particles.vel * 100 / settings['a_start']

    @jax.jit
    def evolve(white):
        particles, obs = initial_particles(white)
        particles, _ = nbody(particles, obs, cosmo, conf)
        return particles.pos(), particles.vel * 100 / settings['a_stop']

    # PMWD's default M unit is1e10 Msun/h; rho_crit can be taken from cosmology.
    mass = float(cosmo.ptcl_mass) * 1e10
    return evolve, initial, conf, cosmo, mass


def particle_grid(position, velocity, mass, conf):
    """CIC integrals with EXPLICIT weights, consistent for any mesh/particle ratio."""
    from pmwd import scatter
    from pmwd.particles import Particles
    particles = Particles.from_pos(conf, position)
    m = scatter(particles, conf, val=mass)
    p = scatter(particles, conf, val=mass[:, None] * velocity)
    q = scatter(particles, conf, val=mass[:, None] * velocity**2)
    valid = m > 0
    denom = jnp.maximum(m, jnp.finfo(m.dtype).tiny)
    mean = p / denom[..., None]
    variance = jnp.maximum(q / denom[..., None] - mean**2, 0)
    return dict(rho=m / (jnp.sum(mass)/conf.mesh_size),
                mean_velocity_km_s=mean, variance_km2_s2=variance,
                mass=m, momentum=p, second_moment=q, valid=valid)


def weighted_periodic_force(position, mass, conf, omega_m, *, split_radius=None, part='all'):
    """Mass-weighted periodic reference, NOT a local/nested zoom solver.

    On one common mesh, long+short force filters sum to1. This tests force
    bookkeeping only, not independent-IC covariance or fine boundary padding.
    Total mass is fixed physical background in this reference calculation.
    """
    from pmwd import scatter, gather
    from pmwd.particles import Particles
    from pmwd.pm_util import fftfreq, fftfwd, fftinv
    from pmwd.gravity import laplace, neg_grad
    if part not in ('all', 'long', 'short'):
        raise ValueError('unknown force filter')
    if part != 'all' and (split_radius is None or split_radius <= 0):
        raise ValueError('positive force split radius required')
    particles = Particles.from_pos(conf, position)
    density = scatter(particles, conf, val=mass * conf.mesh_size / jnp.sum(mass))
    kvec = fftfreq(conf.mesh_shape, conf.cell_size, dtype=conf.float_dtype)
    source = fftfwd((density - 1) * (1.5 * omega_m))
    if part != 'all':
        long_filter = jnp.exp(-sum(k*k for k in kvec) * split_radius**2)
        source *= long_filter if part == 'long' else 1 - long_filter
    potential = laplace(kvec, source)
    return jnp.stack([gather(particles, conf, fftinv(neg_grad(k, potential, conf.cell_size),
                       shape=conf.mesh_shape)) for k in kvec], axis=-1)
