"""Conservative particle moments, not a Gaussian z=0 prior or a halo painter."""
import numpy as np


def periodic_delta(position, center, box):
    return (np.asarray(position) - center + box / 2) % box - box / 2


def particle_moments(position, velocity, mass, *, lower, dx, n):
    """NGP cell integrals: mass, momentum, diagonal second moments and counts.

    Positions use cMpc/h, peculiar velocities km/s, mass physical Msun.
    Empty cells have undefined mean/dispersion, not a measured zero velocity.
    """
    position, velocity, mass = map(np.asarray, (position, velocity, mass))
    if position.shape != velocity.shape or position.shape != (len(mass), 3):
        raise ValueError('particle arrays must have matching N,3 / N shapes')
    if not all(np.isfinite(x).all() for x in (position, velocity, mass)) or np.any(mass <= 0):
        raise ValueError('finite positions/velocities and positive masses required')
    cell = np.floor((position - np.asarray(lower)) / dx).astype(np.int64)
    inside = np.all((cell >= 0) & (cell < n), axis=1)
    key = np.ravel_multi_index(cell[inside].T, (n,) * 3)
    m, v = mass[inside], velocity[inside]
    def deposit(w):
        return np.bincount(key, weights=w, minlength=n**3).reshape((n,) * 3)
    return dict(mass=deposit(m), momentum=np.array([deposit(m * v[:, i]) for i in range(3)]),
                second_moment=np.array([deposit(m * v[:, i]**2) for i in range(3)]),
                count=deposit(None).astype(np.int64), inside=inside)


def aggregate(integrals, ratio):
    n = integrals['mass'].shape[0]
    if n % ratio:
        raise ValueError('integer coarse/fine ratio required')
    c = n // ratio
    result = {}
    for name in ('mass', 'momentum', 'second_moment', 'count'):
        value = integrals[name]
        shape = value.shape[:-3] + (c, ratio, c, ratio, c, ratio)
        result[name] = value.reshape(shape).sum(axis=(-5, -3, -1))
    return result


def derived(integrals, dx):
    m = integrals['mass']
    valid = m > 0
    mean = np.full_like(integrals['momentum'], np.nan)
    second = np.full_like(mean, np.nan)
    np.divide(integrals['momentum'], m[None], out=mean, where=valid[None])
    np.divide(integrals['second_moment'], m[None], out=second, where=valid[None])
    variance = second - mean**2
    if np.any(variance[:, valid] < -1e-7 * np.maximum(1, second[:, valid])):
        raise ValueError('negative velocity variance beyond roundoff')
    return dict(density_Msun_per_cMpc_h3=m / dx**3, mean_velocity_km_s=mean,
                sigma_velocity_km_s=np.sqrt(np.maximum(variance, 0)),
                valid=valid)


def resolved_catalog(position_ckpc_h, velocity_km_s, bound_mass_1e10_Msun_h,
                     group_id, group_first_sub, group_m200, ids, *, h, a, box_ckpc_h,
                     rotation, frame):
    """Read explicitly assigned SUBFIND identities; never search for a best LG.

    Only FoF primaries receive a host M200c field. Satellite bound mass is
    deliberately not relabelled M200c. Assignment is supplied by the caller.
    """
    if set(ids) != {'MW', 'M31', 'M33'} or len(set(ids.values())) != 3:
        raise ValueError('three distinct explicitly assigned subhalo IDs required')
    rotation = np.asarray(rotation)
    if rotation.shape != (3, 3) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError('a proper frame rotation is required')
    if h <= 0 or abs(a - 1) > 1e-8:
        raise ValueError('positive h and a=1 required for this z0 operator')
    origin = position_ckpc_h[ids['MW']]
    result = dict(resolved_halos=True, frame=frame, operator='native_SUBFIND_catalogue_not_grid_peaks',
                  limits='Potential-minimum position and all-member COM velocity; stellar/LMC offsets not calibrated.')
    for identity, index in ids.items():
        g = int(group_id[index])
        result[identity] = dict(subhalo_id=int(index), group_id=g,
            position_kpc=rotation @ periodic_delta(position_ckpc_h[index], origin, box_ckpc_h) * a / h,
            peculiar_velocity_km_s=rotation @ velocity_km_s[index],
            bound_mass_Msun=float(bound_mass_1e10_Msun_h[index] * 1e10 / h),
            host_M200c_Msun=float(group_m200[g] * 1e10 / h) if index == group_first_sub[g] else None)
    return result
