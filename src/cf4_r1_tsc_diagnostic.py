"""Isolated periodic TSC force trial. NOT the production PMWD/HMC backend.

The same quadratic weights are used in deposit and gather. JAX differentiates
these operations directly; PMWD's CIC custom backward rule is not reused.
"""
from itertools import product

import jax
import jax.numpy as jnp


def stencil(position, mesh_n, cell_size):
    """27 periodic neighbors; nearest-node quadratic B-spline weights."""
    coordinate = position / cell_size
    nearest = jnp.floor(coordinate + .5)
    delta = coordinate - nearest
    weights = jnp.stack((.5*(.5-delta)**2, .75-delta**2,
                         .5*(.5+delta)**2), axis=-1)
    offsets = jnp.asarray(list(product((-1, 0, 1), repeat=3)), dtype=jnp.int32)
    index = (nearest.astype(jnp.int32)[:, None, :] + offsets[None]) % mesh_n
    weight = (weights[:, 0, offsets[:, 0]+1]
              * weights[:, 1, offsets[:, 1]+1]
              * weights[:, 2, offsets[:, 2]+1])
    flat = (index[..., 0]*mesh_n + index[..., 1])*mesh_n + index[..., 2]
    return flat, weight


def chunks(position, chunk_size):
    size = min(chunk_size, len(position))
    padding = (-len(position)) % size
    padded = jnp.pad(position, ((0, padding), (0, 0)))
    return padded.reshape((-1, size, 3)), padding


def deposit(position, value, mesh_n, cell_size, chunk_size=32768):
    """Scalar weighted deposit, conserving sum(value), including final padding."""
    blocks, padding = chunks(position, chunk_size)
    value = jnp.broadcast_to(jnp.asarray(value, dtype=position.dtype), (len(position),))
    values = jnp.pad(value, (0, padding)).reshape(blocks.shape[:2])

    def add(mesh, block):
        x, v = block
        index, weight = stencil(x, mesh_n, cell_size)
        return mesh.at[index].add(weight*v[:, None]), None

    mesh, _ = jax.lax.scan(add, jnp.zeros(mesh_n**3, dtype=position.dtype), (blocks, values))
    return mesh.reshape((mesh_n,)*3)


def interpolate(position, field, cell_size, chunk_size=32768):
    """Gather scalar or multichannel mesh with exactly the deposit weights."""
    mesh_n = field.shape[0]
    channels = field.shape[3:]
    flat = field.reshape((mesh_n**3,)+channels)
    blocks, _ = chunks(position, chunk_size)

    def read(_, x):
        index, weight = stencil(x, mesh_n, cell_size)
        weight = weight.reshape(weight.shape + (1,)*len(channels))
        return None, jnp.sum(flat[index]*weight, axis=1)

    _, result = jax.lax.scan(read, None, blocks)
    return result.reshape((-1,)+channels)[:len(position)]


def force(position, mesh_n, cell_size, omega_m, chunk_size=32768):
    """Equal-mass PMWD-unit acceleration, unchanged spectral Poisson/gradient."""
    from pmwd.gravity import laplace, neg_grad
    from pmwd.pm_util import fftfreq, fftfwd, fftinv
    density = deposit(position, mesh_n**3/len(position), mesh_n, cell_size, chunk_size)-1
    kvec = fftfreq((mesh_n,)*3, cell_size, dtype=position.dtype)
    potential = laplace(kvec, fftfwd(density*(1.5*omega_m)))
    field = jnp.stack([fftinv(neg_grad(k, potential, cell_size), shape=(mesh_n,)*3)
                       for k in kvec], axis=-1)
    return interpolate(position, field, cell_size, chunk_size)
