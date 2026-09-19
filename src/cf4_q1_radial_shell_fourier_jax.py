"""Bounded radial-shell/Fourier Q1 candidate.

Sources are grouped into fixed radial shells. Each shell deposits a periodic
TSC field, then applies an anisotropic Gaussian LOS transfer function in
Fourier space. This is an approximation to the source-wise oracle; shell
refinement is an explicit accuracy parameter and no production claim follows
from this module alone.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from cf4_2mpp_joint_likelihood_jax import POPULATIONS, VELOCITY_CONVENTION, _require_jax, observer_centred_spherical_rsd_jax, tsc_deposit_jax


def _nearest_delta_jax(position, mass, n, box):
    cell = jnp.floor((jnp.mod(position, box) / (box / n)) + 0.5).astype(jnp.int32) % n
    field = jnp.zeros((n, n, n), dtype=jnp.asarray(mass).dtype)
    return field.at[cell[0], cell[1], cell[2]].add(mass)


def predict_shell_kernel_convolution_jax(
    source_positions, population_masses, selection_exposure, shell_ids,
    shell_kernel_fft, *, box_size_cMpc_h,
):
    """Apply precomputed oracle shell kernels by FFT translation/contraction.

    ``shell_kernel_fft`` has shape (S,N,N,N) and is produced by the sealed
    NumPy oracle. This path is intentionally limited to a fixed shell kernel;
    arbitrary sub-cell source phases require a separate calibrated basis.
    """
    _require_jax()
    positions = jnp.asarray(source_positions, dtype=jnp.float64)
    masses = jnp.asarray(population_masses, dtype=jnp.float64)
    exposure = jnp.asarray(selection_exposure, dtype=jnp.float64)
    ids = jnp.asarray(shell_ids, dtype=jnp.int32)
    kernels = jnp.asarray(shell_kernel_fft)
    n = int(exposure.shape[1]); shells = int(kernels.shape[0])
    rows = []
    for p in range(POPULATIONS):
        field = jnp.zeros((n, n, n), dtype=jnp.float64)
        for shell in range(shells):
            delta = jnp.zeros((n, n, n), dtype=jnp.float64)
            for source in range(positions.shape[0]):
                delta = delta + _nearest_delta_jax(positions[source], jnp.where(ids[source] == shell, masses[p, source], 0.0), n, box_size_cMpc_h)
            field = field + jnp.real(jnp.fft.ifftn(jnp.fft.fftn(delta) * kernels[shell]))
        rows.append(exposure[p] * field)
    return jnp.stack(rows)


def _wavevectors(n: int, box: float):
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=box / n)
    return jnp.meshgrid(jnp.asarray(k), jnp.asarray(k), jnp.asarray(k), indexing="ij")


def predict_selected_intensity_radial_shell_fourier_jax(
    source_positions, source_velocities_km_s, population_masses, selection_exposure,
    shell_ids, shell_rhat, shell_sigma_cMpc_h, *, observer, box_size_cMpc_h,
    hubble_km_s_Mpc, little_h, scale_factor, velocity_convention=VELOCITY_CONVENTION,
):
    """Return six selected fields using a fixed-shell Fourier response."""
    _require_jax()
    positions = jnp.asarray(source_positions, dtype=jnp.float64)
    velocities = jnp.asarray(source_velocities_km_s, dtype=jnp.float64)
    masses = jnp.asarray(population_masses, dtype=jnp.float64)
    exposure = jnp.asarray(selection_exposure, dtype=jnp.float64)
    ids = jnp.asarray(shell_ids, dtype=jnp.int32)
    rhat = jnp.asarray(shell_rhat, dtype=jnp.float64)
    sigma = jnp.asarray(shell_sigma_cMpc_h, dtype=jnp.float64)
    n = int(exposure.shape[1]); shells = int(rhat.shape[0])
    if exposure.shape[0] != POPULATIONS or exposure.ndim != 4:
        raise ValueError("selection_exposure must have shape (6,N,N,N)")
    if masses.shape != (POPULATIONS, positions.shape[0]) or ids.shape != (positions.shape[0],):
        raise ValueError("source and shell shapes are inconsistent")
    shifted, _disp, _rhat = observer_centred_spherical_rsd_jax(
        positions, velocities, observer, box_size_cMpc_h, hubble_km_s_Mpc,
        little_h=little_h, scale_factor=scale_factor, velocity_convention=velocity_convention)
    kx, ky, kz = _wavevectors(n, box_size_cMpc_h)
    rows = []
    for p in range(POPULATIONS):
        field = jnp.zeros((n, n, n), dtype=jnp.float64)
        for shell in range(shells):
            shell_mass = jnp.where(ids == shell, masses[p], 0.0)
            deposited = tsc_deposit_jax(shifted, shell_mass, n, box_size_cMpc_h)
            direction = rhat[shell]
            kdot = kx * direction[0] + ky * direction[1] + kz * direction[2]
            transfer = jnp.exp(-0.5 * (sigma[shell] * kdot) ** 2)
            filtered = jnp.real(jnp.fft.ifftn(jnp.fft.fftn(deposited) * transfer))
            field = field + filtered
        rows.append(exposure[p] * field)
    return jnp.stack(rows)
