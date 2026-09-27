"""Conservative native PM node -> observed count-voxel centre readout."""
import jax.numpy as jnp
from cf4_r2_continuous_tracer import cell_centres
from cf4_z0_physical_field import read_centred


def native_mass_momentum_to_count_cells(rho_node, mean_velocity_node, box):
    """CIC interpolate native node mass/momentum to cell centres.

    PMWD particle_grid scatters onto nodes at i*dx. Galaxy count voxels are
    centred at (i+.5)*dx. Interpolating momentum before dividing avoids an
    unphysical velocity from the assigned zero in empty native nodes.
    Periodic CIC preserves the sums of mass and each momentum component.
    """
    n = rho_node.shape[0]
    if rho_node.shape != (n,n,n) or mean_velocity_node.shape != (3,n,n,n):
        raise ValueError('native density/velocity geometry mismatch')
    positions = cell_centres(n,box,.5)
    mass = read_centred(rho_node,positions,box,0.).reshape(n,n,n)
    momentum = jnp.stack([read_centred(rho_node*mean_velocity_node[k],positions,box,0.)
                          .reshape(n,n,n) for k in range(3)])
    safe_mass = jnp.where(mass>0,mass,1.)
    return mass,jnp.where(mass[None]>0,momentum/safe_mass[None],0.)
