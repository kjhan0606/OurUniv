"""PM-calibrated *z=0* log-density/velocity Gaussian covariance prototype.

This neither changes the LCDM IC spectrum nor supplies an IC. Calibration
uses disjoint development simulations at the exact native mass/momentum
averaging convention. Gaussianity, isotropy and fixed covariance remain
approximations; finite training-ensemble uncertainty is not marginalized.
"""
import jax.numpy as jnp
import numpy as np

from cf4_z0_physical_field import PhysicalFieldModel, unit_mean_density


def wave_geometry(n, box):
    frequency = 2 * np.pi * np.fft.fftfreq(n, d=box / n)
    raw = np.array(np.meshgrid(frequency, frequency, frequency, indexing="ij"))
    odd = raw.copy()
    if n % 2 == 0:
        for a in range(3):
            index = [slice(None)] * 3; index[a] = n // 2
            odd[(a, *index)] = 0
    norm = np.sqrt(np.sum(odd**2, axis=0))
    direction = odd / np.where(norm > 0, norm, 1)
    return np.sqrt(np.sum(raw**2, axis=0)), direction, norm > 0


def fit_covariance(fields, box, bins=10):
    """Fit isotropic Fourier covariance using training fields only (ortho FFT).

    u=-i*khat.v is the longitudinal scalar; E[v|g]=i*khat*C_ug/P_g*g.
    Residual velocity has nonnegative longitudinal and transverse spectra.
    """
    n = fields[0][0].shape[0]
    kmag, direction, regular = wave_geometry(n, box)
    edges = np.geomspace(2*np.pi/box * .99, kmag.max() * (1 + 1e-8), bins + 1)
    labels = np.clip(np.searchsorted(edges, kmag, side="right") - 1, 0, bins-1)
    gg = np.zeros_like(kmag); uu = gg.copy(); ug = gg.copy(); vv = gg.copy()
    rows = []
    for rho, velocity in fields:
        if rho.shape != (n,)*3 or velocity.shape != (3,n,n,n) or np.any(rho <= 0) or not np.isfinite(rho).all() or not np.isfinite(velocity).all():
            raise ValueError("invalid native physical training fields")
        g = np.log(rho); g -= g.mean()
        gk = np.fft.fftn(g, norm="ortho")
        vk = np.fft.fftn(velocity, axes=(1,2,3), norm="ortho")
        uk = -1j * np.sum(direction * vk, axis=0)
        gg += abs(gk)**2; uu += abs(uk)**2
        ug += (uk * gk.conj()).real
        vv += np.sum(abs(vk)**2, axis=0)
        rows.append({"density_SD": float(rho.std()), "log_density_SD": float(g.std()),
                     "velocity_RMS": float(np.sqrt(np.mean(velocity**2)))})
    pg = np.zeros_like(gg); coupling = pg.copy(); pl = pg.copy(); pt = pg.copy()
    shell_rows = []
    for b in range(bins):
        mask = (labels == b) & regular
        if not mask.any():
            continue
        divisor = len(fields)
        gvar, uvar, cross = (float(a[mask].mean()) / divisor for a in (gg, uu, ug))
        total_v = float(vv[mask].mean()) / divisor
        residual = uvar - cross**2 / gvar
        transverse = (total_v - uvar) / 2
        tolerance = 1e-10 * max(total_v, 1)
        if gvar <= 0 or residual < -tolerance or transverse < -tolerance:
            raise ValueError("training covariance is not positive semidefinite")
        # Only machine-roundoff negatives, never negative physical draws, are removed.
        pg[labels == b] = gvar
        coupling[labels == b] = cross / gvar
        pl[labels == b] = max(residual, 0)
        pt[labels == b] = max(transverse, 0)
        shell_rows.append({"bin": b, "k_h_Mpc": float(kmag[mask].mean()), "grid_modes": int(mask.sum()),
                           "P_logrho": gvar, "u_given_g": cross / gvar,
                           "P_velocity_longitudinal_residual": max(residual, 0),
                           "P_velocity_transverse_each": max(transverse, 0)})
    # Self-conjugate corners have no odd derivative. Model their velocity as
    # isotropic noise from their measured training power (also includes DC bulk).
    special = ~regular
    pt[special] = vv[special] / (3 * len(fields))
    pl[special] = pt[special]
    coupling[special] = 0
    pg[0,0,0] = 0
    if np.any(pg[kmag > 0] <= 0):
        raise ValueError("unpopulated density shell")
    arrays = dict(log_density_amplitude=np.sqrt(pg), coupling=coupling,
                  velocity_longitudinal_amplitude=np.sqrt(pl), velocity_transverse_amplitude=np.sqrt(pt))
    return arrays, {"training_field_statistics": rows, "shells": shell_rows,
                    "native_origin_fraction": .25, "FFT": "ortho", "new_IC_power_rescaling": False}


class PMCalibratedFieldModel(PhysicalFieldModel):
    def __init__(self, *args, covariance, **kwargs):
        super().__init__(*args, **kwargs)
        nuisance_size = self.size - self.field_size
        self.field_size = 4 * self.n**3
        self.size = self.field_size + nuisance_size
        _, direction, _ = wave_geometry(self.n, self.box)
        self.direction = jnp.asarray(direction)
        self.covariance = {name: jnp.asarray(value) for name, value in covariance.items()}
        if any(a.shape != (self.n,)*3 or not np.isfinite(a).all() for a in covariance.values()):
            raise ValueError("invalid calibrated covariance arrays")

    def fields(self, vector):
        c = self.covariance
        white = vector[:self.field_size].reshape((4,self.n,self.n,self.n))
        modes = jnp.fft.fftn(white, axes=(1,2,3), norm="ortho")
        gk = modes[0] * c["log_density_amplitude"]
        g = jnp.fft.ifftn(gk, norm="ortho").real
        longitudinal = self.direction * jnp.sum(self.direction * modes[1:], axis=0)
        residual = c["velocity_longitudinal_amplitude"] * longitudinal + c["velocity_transverse_amplitude"] * (modes[1:] - longitudinal)
        vk = 1j * self.direction * c["coupling"] * gk + residual
        velocity = jnp.fft.ifftn(vk, axes=(1,2,3), norm="ortho").real
        return g, unit_mean_density(g), velocity
