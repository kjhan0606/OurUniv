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


def native_moments_to_count_cells(rho_node,mean_velocity_node,variance_node,box):
    """Read mass, momentum AND raw second moment before taking ratios.

    Interpolating variance alone omits the between-node velocity dispersion.
    Output has three diagonal variances, not an inferred mean-field error.
    Nonnegative input variances are the caller's physical state contract;
    convex interpolation guarantees nonnegative output up to roundoff.
    """
    if variance_node.shape!=mean_velocity_node.shape:
        raise ValueError('three native physical variances required')
    mass,mean=native_mass_momentum_to_count_cells(rho_node,mean_velocity_node,box)
    n=rho_node.shape[0];positions=cell_centres(n,box,.5)
    second=jnp.stack([read_centred(rho_node*(variance_node[k]+mean_velocity_node[k]**2),
        positions,box,0.).reshape(n,n,n) for k in range(3)])
    safe_mass=jnp.where(mass>0,mass,1.)
    variance=jnp.where(mass[None]>0,second/safe_mass[None]-mean**2,0.)
    return mass,mean,jnp.maximum(variance,0.)
