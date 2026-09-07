"""Differentiable version of the existing PMWD truth readout; same conventions."""
import jax
import jax.numpy as jnp


def block_sum(field):
    n = field.shape[0]//2
    return field.reshape((n,2,n,2,n,2)+field.shape[3:]).sum(axis=(1,3,5))


def make_forward(program):
    from pmwd import Configuration, SimpleLCDM, boltzmann, linear_modes, lpt, nbody, scatter
    c = program["cosmology"]
    n = program["grid"]["truth_N"]
    if n != 64 or program["grid"]["truth_cell_size_cMpc_h"] != 6:
        raise ValueError("only approved N64/384 development model")
    conf = Configuration(ptcl_spacing=6., ptcl_grid_shape=(n,)*3, mesh_shape=1,
                         cosmo_dtype=jnp.float64, float_dtype=jnp.float64)
    cosmo = boltzmann(SimpleLCDM(conf, Omega_m=c["Om"], Omega_b=c["Ob"],
        h=c["h"], A_s_1e9=c["A_s_1e9"], n_s=c["ns"]), conf)

    @jax.jit
    def forward(white):
        modes = linear_modes(white.reshape((n,)*3), cosmo, conf)
        particles, observables = lpt(modes, cosmo, conf)
        # LPT leaves acc=None. PMWD's custom reverse rule returns an array
        # cotangent for acc, so its input/output tree would otherwise differ.
        # Initial force overwrites this constant; it adds no physical force or
        # gradient path. Check unchanged forward values and directional derivative.
        particles = particles.replace(acc=jnp.zeros_like(particles.disp))
        particles, observables = nbody(particles, observables, cosmo, conf)
        mass = scatter(particles, conf)
        momentum = scatter(particles, conf, val=particles.vel*100.)
        coarse_mass, coarse_momentum = block_sum(mass), block_sum(momentum)
        velocity = jnp.where(coarse_mass[...,None]>1e-10,
                            coarse_momentum/jnp.maximum(coarse_mass[...,None],1e-10), 0.)
        # Native block centres = (i+.25)*12; no interpolation/recentering.
        conservation = jnp.concatenate((jnp.array([coarse_mass.sum()-mass.sum()]),
            (coarse_momentum.sum(axis=(0,1,2))-momentum.sum(axis=(0,1,2))).ravel()))
        return coarse_mass/8, jnp.moveaxis(velocity,-1,0), conservation

    @jax.jit
    def linear_field(white):
        return linear_modes(white.reshape((n,)*3), cosmo, conf, real=True)
    return forward, linear_field
