"""Development-only positive z=0 field model, not nonlinear gravitational dynamics.

The LCDM Gaussian latent is NOT exported as physical delta.  Coarse log-density
uses a fixed cell-average window; exponentiation then defines a positive,
unit-mean, piecewise-constant density.  exp(average(g)) is not average(exp(g)).
Velocity is a potential-flow approximation from that latent, not a PM solution.
All reported fields and observation reads use cell centres (i+1/2)*dx.
"""
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from cf4_datum_bearing_z0_phasec_pilot import _jax_tsc_deposit_one, selection_bases


def unit_mean_density(log_density):
    return jnp.exp(log_density - logsumexp(log_density) + jnp.log(log_density.size))


def centres(n, box):
    axis = (np.arange(n) + 0.5) * box / n
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)


def read_centred(grid, positions, box):
    """Periodic CIC read, including exactly at centres and across the seam."""
    n = grid.shape[0]
    cell = jnp.asarray(positions) / (box / n) - 0.5
    low = jnp.floor(cell).astype(jnp.int32)
    frac = cell - low
    out = jnp.zeros(cell.shape[:-1], dtype=grid.dtype)
    for corner in product((0, 1), repeat=3):
        weight = jnp.prod(jnp.where(jnp.array(corner), frac, 1 - frac), axis=-1)
        idx = (low + jnp.array(corner)) % n
        out += weight * grid[idx[..., 0], idx[..., 1], idx[..., 2]]
    return out


def recenter_old_pm_fields(rho, velocity):
    """Positive conservative remap of legacy block-node fields to cell centres.

    N64 nodes at 0,6,... were averaged in pairs: coarse origin=3, not 6.
    Target centres are 1/4 coarse cell ahead.  Shift mass AND momentum; this
    interpolation adds known smoothing and does not recreate fresh PM truth.
    """
    if rho.ndim != 3 or velocity.shape != (3,) + rho.shape:
        raise ValueError("expected scalar density and component-first velocity")
    if not np.isfinite(rho).all() or not np.isfinite(velocity).all() or np.any(rho <= 0):
        raise ValueError("invalid physical PM fields")

    def shift(field):
        for axis in range(3):
            field = 0.75 * field + 0.25 * np.roll(field, -1, axis=axis)
        return field

    shifted_rho = shift(rho)
    shifted_momentum = np.stack([shift(rho * v) for v in velocity])
    return shifted_rho, shifted_momentum / shifted_rho


class PhysicalFieldModel:
    """White field + 24 standard-normal nuisance parameters, same old priors."""

    def __init__(self, transfer, growth, box, response, design, nbar, bias, settings):
        self.n = n = transfer.shape[0]
        self.box, self.size = box, n**3 + 24
        if transfer.shape != (n,) * 3 or response.shape != (6, n, n, n):
            raise ValueError("invalid transfer/response shape")
        if not np.isfinite(response).all() or np.any(response < 0):
            raise ValueError("invalid response")
        self.response = jnp.asarray(response)
        self.settings = settings
        self.nbar, self.bias = jnp.asarray(nbar), jnp.asarray(bias)
        self.transfer = jnp.asarray(transfer)
        freq = 2 * np.pi * np.fft.fftfreq(n, d=box / n)
        k = np.array(np.meshgrid(freq, freq, freq, indexing="ij"))
        self.window = jnp.asarray(np.prod(np.sinc(k * (box / n) / (2 * np.pi)), axis=0))
        k2 = np.sum(k**2, axis=0)
        # An odd derivative must be zero on its own even-grid Nyquist plane.
        derivative_k = k.copy()
        if n % 2 == 0:
            for axis in range(3):
                sl = [slice(None)] * 3
                sl[axis] = n // 2
                derivative_k[(axis, *sl)] = 0.0
        self.vkernel = jnp.asarray(1j * 100 * growth * derivative_k / np.where(k2 > 0, k2, 1))
        self.coords = jnp.asarray(centres(n, box))
        relative = self.coords - box / 2
        radius = jnp.linalg.norm(relative, axis=-1, keepdims=True)
        self.radial = relative / jnp.where(radius > 0, radius, 1)
        self.positions = jnp.asarray(design["pos"])
        self.rhat = jnp.asarray(design["rhat"])
        self.B = jnp.asarray(design["B"])
        self.qstd = jnp.asarray(design["q_std"])
        self.variance = jnp.asarray(design["variance"])
        self.train = jnp.asarray(~design["holdout"])
        self.selection_basis = jnp.asarray(np.stack(selection_bases(n, box)))

    def fields(self, vector):
        modes = jnp.fft.fftn(vector[:self.n**3].reshape((self.n,) * 3), norm="ortho")
        modes = modes * self.transfer * self.window
        latent = jnp.fft.ifftn(modes, norm="ortho").real
        rho = unit_mean_density(latent)
        velocity = jnp.fft.ifftn(self.vkernel * modes, axes=(1, 2, 3), norm="ortho").real
        return latent, rho, velocity

    def observe(self, rho, velocity, nuisance):
        """Positive mass -> biased tracer mass -> RSD/FoG -> selection.

        Normalization uses model expectation, NEVER a realized observed total.
        FoG dispersion is a nuisance distinct from mean velocity and its error.
        """
        s = self.settings
        alpha = s["alpha_log_sigma"] * nuisance[:6]
        bias = self.bias * jnp.exp(s["bias_log_sigma"] * nuisance[6:12])
        fog = jnp.asarray(s["FoG_prior_median_km_s"]) * jnp.exp(s["FoG_log_sigma"] * nuisance[12:18])
        sigma = jnp.sqrt(fog**2 + jnp.asarray(s["fixed_redshift_error_km_s"])**2) / 100
        selection = jnp.exp(s["selection_basis_log_amplitude"] * jnp.einsum("p,pijk->ijk", nuisance[18:20], self.selection_basis))
        displacement = jnp.einsum("aijk,ijka->ijk", velocity, self.radial) / 100

        def population(p):
            mass = self.nbar[p] * jnp.exp(alpha[p]) * unit_mean_density(bias[p] * jnp.log(rho))
            pushed = jnp.zeros_like(rho)
            for node, weight in zip(s["Gaussian_radial_quadrature_offsets_sigma"], s["Gaussian_radial_quadrature_weights"], strict=True):
                positions = (self.coords + (displacement + node * sigma[p])[..., None] * self.radial) % self.box
                pushed += weight * _jax_tsc_deposit_one(mass, positions, self.n, self.box / self.n)
            return self.response[p] * selection * pushed

        intensity = jax.lax.map(population, jnp.arange(6))
        sampled = jnp.stack([read_centred(velocity[a], self.positions, self.box) for a in range(3)], axis=-1)
        radial_signal = jnp.sum(sampled * self.rhat, axis=-1) + self.B @ (self.qstd * nuisance[20:24])
        return intensity, radial_signal

    def forward(self, vector):
        _, rho, velocity = self.fields(vector)
        return self.observe(rho, velocity, vector[self.n**3:])

    def nlp(self, vector, counts, radial_data):
        intensity, radial_model = self.forward(vector)
        lam = 0.8 * intensity
        support = self.response > 0
        safe = jnp.where(support, lam, 1.0)
        count_nll = jnp.sum(jnp.where(support, lam - counts * jnp.log(safe), 0.0))
        velocity_nll = 0.5 * jnp.sum(jnp.where(self.train, (radial_model - radial_data)**2 / self.variance, 0))
        return 0.5 * jnp.sum(vector**2) + count_nll + velocity_nll
