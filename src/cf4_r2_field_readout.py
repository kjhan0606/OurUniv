"""Fixed native-field diagnostics; neither observational resolution nor UQ.

Consume an already computed field. No dynamics, truth labels or observations.
Velocity summaries are density-weighted, including empty cells with zero weight.
"""
import numpy as np


def present_field_readout(rho, velocity):
    rho = np.asarray(rho)
    velocity = np.asarray(velocity)
    if (rho.ndim != 3 or len(set(rho.shape)) != 1
            or velocity.shape != (3,)+rho.shape or rho.size == 0):
        raise ValueError('aligned cubic density and component-first velocity required')
    if (not np.isfinite(rho).all() or not np.isfinite(velocity).all()
            or np.any(rho < 0)):
        raise ValueError('finite fields and nonnegative density required')
    mass = float(rho.sum(dtype=np.float64))
    if mass <= 0:
        raise ValueError('positive total density required')
    bulk = np.array([np.sum(rho*v, dtype=np.float64)/mass for v in velocity])
    rms = [float(np.sqrt(np.sum(rho*(v-b)**2, dtype=np.float64)/mass))
           for v,b in zip(velocity,bulk)]
    return dict(density_mean=float(rho.mean(dtype=np.float64)),
                density_mean_square=float(np.mean(rho*rho, dtype=np.float64)),
                occupied_fraction=float(np.count_nonzero(rho)/rho.size),
                mass_weighted_bulk_velocity_km_s=bulk.tolist(),
                mass_weighted_spatial_velocity_rms_km_s=rms,
                posterior_uncertainty=False,
                velocity_rms_role='spatial variation of cell means, not within-cell dispersion or UQ')
