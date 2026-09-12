"""Finite, whole-patch z0 prior. No phase randomization or halo/field decoupling.

Kernel conditioning is on declared coarse summaries, NOT exact coarse cells.
It is a development support model, not a calibrated cosmological posterior.
"""
import numpy as np
from scipy.special import logsumexp


def read_component(source_path, index):
    """Read an unchanged whole patch with its native catalogue/geometry link.

    Never rescale the mass or shift halo positions to match a requested parent.
    The caller must infer observer identity/orientation and likelihood separately.
    """
    import h5py
    from cf4_resolved_moments import derived
    with h5py.File(source_path, 'r') as f:
        if f.attrs['status'] != 'NATIVE_TOTAL_MATTER_NOT_OBSERVED_LOCAL_UNIVERSE':
            raise ValueError('a complete native total-matter source is required')
        origins = f['patch_origins_coarse'][:]
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(origins):
            raise ValueError('invalid whole-patch component index')
        origin = origins[index] * 8
        x, y, z = origin
        values = f['fine'][:, x:x + 128, y:y + 128, z:z + 128]
        dx = float(f['fine'].attrs['dx_cMpc_h'])
        metadata = dict(index=int(index), lower_cMpc_h=(origin * dx).tolist(),
            dx_cMpc_h=dx, source_catalogue=f.attrs['source_catalogue'],
            cosmology_h=float(f.attrs['h']), native_unmodified=True,
            limits='Native TNG axes. Catalogue membership/observer orientation must be read from the linked source; not an assigned MW/M31/M33 system.')
    integrals = dict(mass=values[0], momentum=values[1:4], second_moment=values[4:7])
    return dict(metadata=metadata, integrals=integrals, fields=derived(integrals, dx))


def conditional_weights(features, condition, bandwidth, *, log_data=None):
    """p(j|c) proportional to exp(-||[C_j-c]/bandwidth||²/2).

    Bandwidth is a PRIOR hyperparameter, never an observational error bar.
    log_data is one new likelihood factor; old CF4 posterior factors must not
    be supplied again. A draw is the entire unchanged particle/halo patch j.
    """
    features, condition, bandwidth = map(lambda x: np.asarray(x, float), (features, condition, bandwidth))
    if features.ndim != 2 or condition.shape != features.shape[1:] or bandwidth.shape != condition.shape:
        raise ValueError('component/feature and condition/bandwidth shapes disagree')
    if not all(np.isfinite(x).all() for x in (features, condition, bandwidth)) or np.any(bandwidth <= 0):
        raise ValueError('finite summaries and explicit positive prior bandwidth required')
    if len(features) < 2:
        raise ValueError('a single FoF fixture is not an empirical field prior')
    log_prior = -.5 * np.sum(((features - condition) / bandwidth)**2, axis=1)
    log_prior -= logsumexp(log_prior)
    likelihood = np.zeros(len(features)) if log_data is None else np.asarray(log_data, float)
    if likelihood.shape != log_prior.shape or np.any(np.isnan(likelihood)) or np.any(likelihood == np.inf):
        raise ValueError('one finite or minus-infinite new log likelihood per component required')
    logz = logsumexp(log_prior + likelihood)
    if not np.isfinite(logz):
        return dict(status='NO_SUPPORT', weights=np.zeros(len(features)), prior_weights=np.exp(log_prior), ess=0., log_evidence=-np.inf)
    weights = np.exp(log_prior + likelihood - logz)
    ess = float(1 / np.dot(weights, weights))
    return dict(status='LIMITED_FINITE_SUPPORT' if ess < 4 or weights.max() > .5 else 'FINITE_SUPPORT_NOT_CALIBRATION',
        weights=weights, prior_weights=np.exp(log_prior), ess=ess, log_evidence=float(logz))


def mixture_moments(masses, momenta, seconds, weights):
    """Separate uncertainty in mean velocity from physical within-cell sigma_v.

    Returns volume-field posterior moments. Does NOT average physical sigma
    and mean-field uncertainty into a single falsely interpreted quantity.
    Empty component cells must be handled explicitly by a different model.
    """
    m, p, s, w = map(np.asarray, (masses, momenta, seconds, weights))
    if p.shape != s.shape or p.shape != (len(m), 3) + m.shape[1:] or w.shape != (len(m),):
        raise ValueError('component-first mass and component/vector-first momentum required')
    if not all(np.isfinite(x).all() for x in (m, p, s, w)) or np.any(m <= 0) or np.any(w < 0) or not np.isclose(w.sum(), 1):
        raise ValueError('positive component masses and normalized finite weights required')
    v = p / m[:, None]
    raw_second = s / m[:, None]
    variance = raw_second - v**2
    if np.any(variance < -1e-7 * np.maximum(1, raw_second)):
        raise ValueError('unphysical component second moment')
    average = lambda value: np.tensordot(w, value, axes=(0, 0))
    mean_v = average(v)
    return dict(mean_mass=average(m), mean_velocity=mean_v,
        variance_mass=average((m - average(m))**2),
        variance_mean_velocity=average((v - mean_v)**2),
        mean_physical_velocity_variance=average(np.maximum(variance, 0)))
