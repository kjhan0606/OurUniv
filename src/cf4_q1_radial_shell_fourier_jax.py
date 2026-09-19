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
    # Cell centres are at (i+1/2)h in the project TSC convention.
    cell = jnp.floor(jnp.mod(position, box) / (box / n)).astype(jnp.int32) % n
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


def predict_phase_basis_kernel_jax(source_positions, population_masses, selection_exposure,
                                   phase_kernel_fft, shell_ids=None, *, box_size_cMpc_h):
    """Apply a trilinearly interpolated sub-cell phase kernel basis.

    ``phase_kernel_fft`` has shape (P,P,P,N,N,N), with kernels sampled at
    phase coordinates in [0,1)^3 for one shell. This is a development route;
    shell and population-specific bases can be added after the phase gate.
    """
    _require_jax()
    positions=jnp.asarray(source_positions,dtype=jnp.float64); masses=jnp.asarray(population_masses,dtype=jnp.float64)
    ids=jnp.zeros((positions.shape[0],),dtype=jnp.int32) if shell_ids is None else jnp.asarray(shell_ids,dtype=jnp.int32)
    exposure=jnp.asarray(selection_exposure,dtype=jnp.float64); basis=jnp.asarray(phase_kernel_fft)
    n=int(exposure.shape[1]); pcount=int(basis.shape[-6] if basis.ndim == 7 else basis.shape[0]); h=box_size_cMpc_h/n; field_rows=[]
    for pop in range(POPULATIONS):
        field=jnp.zeros((n,n,n),dtype=jnp.float64)
        for source in range(positions.shape[0]):
            cell=jnp.floor(jnp.mod(positions[source],box_size_cMpc_h)/h).astype(jnp.int32)%n
            phase=jnp.mod(positions[source]/h,1.0)*pcount
            low=jnp.floor(phase).astype(jnp.int32)%pcount; frac=phase-jnp.floor(phase)
            interp=jnp.zeros((n,n,n),dtype=jnp.complex128)
            for ix in (0,1):
                for iy in (0,1):
                    for iz in (0,1):
                        w=(frac[0] if ix else 1-frac[0])*(frac[1] if iy else 1-frac[1])*(frac[2] if iz else 1-frac[2])
                        selected = basis if basis.ndim == 6 else basis[ids[source]]
                        interp=interp+w*selected[(low[0]+ix)%pcount,(low[1]+iy)%pcount,(low[2]+iz)%pcount]
            delta=jnp.zeros((n,n,n),dtype=jnp.float64).at[cell[0],cell[1],cell[2]].add(masses[pop,source])
            field=field+jnp.real(jnp.fft.ifftn(jnp.fft.fftn(delta)*interp))
        field_rows.append(exposure[pop]*field)
    return jnp.stack(field_rows)


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
