"""Fixed native-field diagnostics; neither observational resolution nor UQ.

Consume an already computed field. No dynamics, truth labels or observations.
Velocity summaries are density-weighted, including empty cells with zero weight.
"""
import numpy as np


def present_field_readout(rho, velocity, *, box=None):
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
    result = dict(density_mean=float(rho.mean(dtype=np.float64)),
                density_mean_square=float(np.mean(rho*rho, dtype=np.float64)),
                occupied_fraction=float(np.count_nonzero(rho)/rho.size),
                mass_weighted_bulk_velocity_km_s=bulk.tolist(),
                mass_weighted_spatial_velocity_rms_km_s=rms,
                posterior_uncertainty=False,
                velocity_rms_role='spatial variation of cell means, not within-cell dispersion or UQ')
    if box is not None:
        if not np.isfinite(box) or box <= 200:
            raise ValueError('box must exceed twice the largest diagnostic radius')
        # Input is native PM nodes, not the count-cell readout.
        n = rho.shape[0]
        axis = np.arange(n)*box/n-box/2
        radius2 = axis[:,None,None]**2+axis[None,:,None]**2+axis[None,None,:]**2
        spheres = {}
        for radius in (50,100):
            mask = radius2 <= radius**2
            weights = rho[mask]
            total = float(weights.sum(dtype=np.float64))
            spheres[str(radius)] = dict(
                density_mean=float(weights.mean()) if weights.size else None,
                mass_weighted_velocity_km_s=[float(np.sum(weights*v[mask])/total)
                    for v in velocity] if total > 0 else None)
        result['observer_spheres_cMpc_h'] = spheres
        result['sphere_geometry'] = 'native nodes i*box/n; observer box/2'
    return result
